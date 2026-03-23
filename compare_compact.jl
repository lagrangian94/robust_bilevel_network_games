"""
compare_compact.jl

Original vs Compact (dictionary-indexed) LDR 비교 스크립트.

## 목적
Dictionary-indexed compact ISP가 원본과 동일한 결과를 내는지 검증하고,
성능 차이를 측정한다.

## 실험 조건
- 네트워크: 3×3 grid (9 nodes, 18 arcs)
- 시나리오 수: S = 1, 2, 5
- 알고리즘: TR Nested Benders (inner TR only, outer TR off)
- Conic solver: Mosek (SDP + SOC)
- MIP solver: Gurobi (outer master)

## 함수 교체 방식
Julia에서 function은 const binding이므로 @eval로 반복 교체하면 에러 발생.
→ 원본을 전부 실행한 후, 한 번만 swap하여 compact를 전부 실행하는 구조.

## 검증 기준
- 목적함수 값 차이 ≈ 0 (수치 오차 범위 내)
- Inner iteration 패턴 동일
- JIT warm-up 후 공정한 시간 비교
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using Hypatia
using LinearAlgebra
using Infiltrator
using Revise

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("nested_benders_trust_region.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

# Load compact ISP builder + optimize functions
include("compact_ldr_utils.jl")
include("build_isp_compact.jl")

# ===== Parameters =====
γ_ratio = 0.10  # Interdiction budget as fraction of interdictable arcs: γ = ceil(γ_ratio * |A_I|)
                 # Sensitivity: γ_ratio ∈ {0.03, 0.05, 0.10}
ρ = 0.2  # Recovery power ratio: w = ρ·γ·c̄, follower's max recovery = ρ × expected interdiction damage
         # Sensitivity: ρ ∈ {0.05, 0.1, 0.2, 0.3}
v = 1.0
seed = 42
epsilon = 0.5
ϕU = 1/epsilon # valid upper bound?
λU = ϕU  ## 10.0 -> ϕU로 변경. λ ≤ ϕU: LDR P-bound 조건

# ===== JIT Warm-up (원본만) =====
# Julia JIT 특성상 첫 실행에서 컴파일 시간 포함.
# 원본 warm-up 1회로 JIT 컴파일 완료. Compact는 별도 함수이므로
# 첫 compact 실행(S=1)이 warm-up 역할을 겸한다.
println("="^80)
println("JIT WARM-UP (original)")
println("="^80)

S = 1
network = generate_grid_network(5, 5, seed=seed)
warm_num_arcs = length(network.arcs) - 1
warm_interd_count = sum(network.interdictable_arcs[1:warm_num_arcs])
γ = ceil(Int, γ_ratio * warm_interd_count)
warm_cap, _ = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
warm_interd = findall(network.interdictable_arcs[1:warm_num_arcs])
w = round(ρ * γ * sum(warm_cap[warm_interd, :]) / (length(warm_interd) * S), digits=4)
R, r_dict, xi_bar = build_robust_counterpart_matrices(warm_cap[1:end-1, :], epsilon)
warm_uset = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

wm, wv = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer)
tr_nested_benders_optimize!(wm, wv, network, ϕU, λU, γ, w, warm_uset;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    outer_tr=false, inner_tr=true)

println("Warm-up complete.\n")

# ===== Print compact LDR stats =====
print_compact_ldr_stats(network)

# ===== Phase 1: 원본 (Original) 전체 실행 =====
network = generate_grid_network(5, 5, seed=seed)
print_network_summary(network)
num_arcs = length(network.arcs) - 1
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, γ_ratio * num_interdictable)
println("  Interdiction budget: γ = ceil($γ_ratio × $num_interdictable) = $γ")

results_orig = Dict{Int, Any}()

for test_S in [1, 2]
    println("\n" * "="^80)
    println("  ORIGINAL — S = $test_S")
    println("="^80)

    global S = test_S
    capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), test_S, seed=seed)
    num_arcs_c = length(network.arcs) - 1
    interd_c = findall(network.interdictable_arcs[1:num_arcs_c])
    c_bar_c = sum(capacities[interd_c, :]) / (length(interd_c) * test_S)
    global w = round(ρ * γ * c_bar_c, digits=4)
    println("  Recovery budget: w = ρ·γ·c̄ = $ρ × $γ × $(round(c_bar_c, digits=2)) = $(round(w, digits=4))")
    capacity_scenarios_regular = capacities[1:end-1, :]
    global R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
    uset = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

    GC.gc()
    model_o, vars_o = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer)
    t_start = time()
    result_o = tr_nested_benders_optimize!(model_o, vars_o, network, ϕU, λU, γ, w, uset;
        mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
        outer_tr=false, inner_tr=true)
    t_orig = time() - t_start

    obj_orig = haskey(result_o, :past_upper_bound) ? minimum(result_o[:past_upper_bound]) : NaN
    inner_iters = haskey(result_o, :inner_iter) ? result_o[:inner_iter] : []

    println("  Original: time=$(round(t_orig, digits=2))s, obj=$(round(obj_orig, digits=6))")
    println("  Inner iters: $inner_iters")

    results_orig[test_S] = Dict(:time => t_orig, :obj => obj_orig, :inner_iters => inner_iters)
end

# ===== Phase 2: Compact 함수로 메서드 재정의 =====
# Julia에서 function은 const binding → @eval로 재할당 불가.
# 대신 기존 함수의 **메서드를 재정의**하여 compact 버전으로 dispatch한다.
# 이렇게 하면 const binding은 그대로 유지하면서 내부 구현만 교체된다.
println("\n" * "="^80)
println("REDEFINING methods to COMPACT versions")
println("="^80)

@eval Main function initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set; conic_optimizer=nothing, λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing, α_sol=nothing)
    return $initialize_isp_compact(network, S, ϕU, λU, γ, w, v, uncertainty_set; conic_optimizer=conic_optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
end

@eval Main function isp_leader_optimize!(isp_leader_model::Model, isp_leader_vars::Dict; isp_data=nothing, uncertainty_set::Dict=nothing, λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing, α_sol=nothing)
    return $isp_leader_optimize_compact!(isp_leader_model, isp_leader_vars; isp_data=isp_data, uncertainty_set=uncertainty_set, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
end

@eval Main function isp_follower_optimize!(isp_follower_model::Model, isp_follower_vars::Dict; isp_data=nothing, uncertainty_set::Dict=nothing, λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing, α_sol=nothing)
    return $isp_follower_optimize_compact!(isp_follower_model, isp_follower_vars; isp_data=isp_data, uncertainty_set=uncertainty_set, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
end

@eval Main function evaluate_master_opt_cut(isp_leader_instances::Dict, isp_follower_instances::Dict, isp_data::Dict, cut_info::Dict, iter::Int)
    return $evaluate_master_opt_cut_compact(isp_leader_instances, isp_follower_instances, isp_data, cut_info, iter)
end

# ===== Phase 3: Compact 전체 실행 =====
results_compact = Dict{Int, Any}()

for test_S in [1, 2]
    println("\n" * "="^80)
    println("  COMPACT (dict-indexed) — S = $test_S")
    println("="^80)

    global S = test_S
    capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), test_S, seed=seed)
    num_arcs_c = length(network.arcs) - 1
    interd_c = findall(network.interdictable_arcs[1:num_arcs_c])
    c_bar_c = sum(capacities[interd_c, :]) / (length(interd_c) * test_S)
    global w = round(ρ * γ * c_bar_c, digits=4)
    capacity_scenarios_regular = capacities[1:end-1, :]
    global R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
    uset = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

    GC.gc()
    model_c, vars_c = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer)
    t_start = time()
    result_c = tr_nested_benders_optimize!(model_c, vars_c, network, ϕU, λU, γ, w, uset;
        mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
        outer_tr=false, inner_tr=true)
    t_compact = time() - t_start

    obj_compact = haskey(result_c, :past_upper_bound) ? minimum(result_c[:past_upper_bound]) : NaN
    inner_iters = haskey(result_c, :inner_iter) ? result_c[:inner_iter] : []

    println("  Compact: time=$(round(t_compact, digits=2))s, obj=$(round(obj_compact, digits=6))")
    println("  Inner iters: $inner_iters")

    results_compact[test_S] = Dict(:time => t_compact, :obj => obj_compact, :inner_iters => inner_iters)
end

# ===== Final Summary =====
println("\n" * "="^80)
println("FINAL SUMMARY: Original vs Compact (dict-indexed) — 5x5 grid, inner TR only")
println("="^80)
println()
println("  ┌──────┬──────────────┬──────────────┬──────────┬──────────┬────────────────────┐")
println("  │  S   │  Orig Time   │ Compact Time │ Speedup  │ Obj Diff │ Iters Match?       │")
println("  ├──────┼──────────────┼──────────────┼──────────┼──────────┼────────────────────┤")
for test_S in [1, 2]
    t_o = round(results_orig[test_S][:time], digits=2)
    t_c = round(results_compact[test_S][:time], digits=2)
    sp = round(t_o / t_c, digits=3)
    od = round(abs(results_orig[test_S][:obj] - results_compact[test_S][:obj]), digits=8)
    iters_match = results_orig[test_S][:inner_iters] == results_compact[test_S][:inner_iters] ? "YES" : "NO"
    println("  │ $(lpad(test_S, 4)) │ $(lpad(t_o, 9))s   │ $(lpad(t_c, 9))s   │ $(lpad(sp, 7))x │ $(lpad(od, 8)) │ $(rpad(iters_match, 18)) │")
end
println("  └──────┴──────────────┴──────────────┴──────────┴──────────┴────────────────────┘")
println()

# 각 S별 상세 비교
for test_S in [1, 2]
    obj_o = results_orig[test_S][:obj]
    obj_c = results_compact[test_S][:obj]
    iters_o = results_orig[test_S][:inner_iters]
    iters_c = results_compact[test_S][:inner_iters]
    println("S=$test_S: orig_obj=$(round(obj_o, digits=8)), compact_obj=$(round(obj_c, digits=8))")
    println("      orig_iters=$iters_o")
    println("      compact_iters=$iters_c")
end

println()
println("NOTE: Dictionary-indexed approach — non-adjacent variables never created.")
println("      Actual JuMP variable count reduced (not just solver presolve).")
println("      S=1 compact run includes JIT compilation overhead for compact functions.")
println("="^80)
