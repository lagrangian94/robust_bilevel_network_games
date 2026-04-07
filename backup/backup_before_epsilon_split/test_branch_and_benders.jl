"""
Branch-and-Benders (single tree) 테스트 스크립트.
compare_benders.jl과 동일한 인풋으로 TR Nested Benders vs Branch-and-Benders 비교.

병렬 실행하려면:
  julia -t 8 test_branch_and_benders.jl

스레드 설정:
  Mosek 내부 스레드 기본값 = CPU 논리코어 수 ÷ Julia 스레드 수 (최소 1)
  예) 24코어, -t 8 → Mosek 3스레드, 총 24스레드
  예) 24코어, -t 16 → Mosek 1스레드, 총 16스레드

  환경변수 MOSEK_NUM_THREADS로 override 가능:
  CMD:  set MOSEK_NUM_THREADS=1 && julia -t 8 test_branch_and_benders.jl
  PS :  \$env:MOSEK_NUM_THREADS="1"; julia -t 8 test_branch_and_benders.jl
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using LinearAlgebra
using Infiltrator
using Serialization
using JLD2
using Revise

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("parallel_utils.jl")
includet("strict_benders.jl")
includet("nested_benders_trust_region.jl")
includet("branch_and_benders.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary
using .NetworkGenerator: generate_sioux_falls_network, generate_nobel_us_network, generate_abilene_network, generate_polska_network, print_realworld_network_summary

# 스레드 체크 — -t 1이면 병렬 solve 효과 없음
if Threads.nthreads() == 1
    @warn "Julia 스레드가 1개입니다! 병렬 solve 효과 없음.\n" *
          "  julia -t 8 test_branch_and_benders.jl 로 실행하세요."
end

# 메모리 모니터 — 별도 CMD 창에서 Available MB + CPU% 표시 (5초 간격, 창 닫기로 종료)
if Sys.iswindows()
    monitor_bat = joinpath(@__DIR__, "monitor_memory.bat")
    if isfile(monitor_bat)
        run(`cmd /c start "" $monitor_bat`; wait=false)
    end
end

# ===== Network Instance Configs =====
network_configs = Dict(
    :grid_3x3 => Dict(:type => :grid, :m => 3, :n => 3),
    :grid_5x5 => Dict(:type => :grid, :m => 5, :n => 5),
    :sioux_falls => Dict(:type => :real_world, :generator => generate_sioux_falls_network),
    :nobel_us => Dict(:type => :real_world, :generator => generate_nobel_us_network),
    :abilene => Dict(:type => :real_world, :generator => generate_abilene_network),
    :polska => Dict(:type => :real_world, :generator => generate_polska_network),
)

function setup_instance(config_key::Symbol;
    S=10, γ_ratio=0.10, ρ=0.2, v=1.0, seed=42, epsilon=0.5)

    config = network_configs[config_key]
    if config[:type] == :grid
        network = generate_grid_network(config[:m], config[:n], seed=seed)
        print_network_summary(network)
    elseif config[:type] == :real_world
        network = config[:generator]()
        print_realworld_network_summary(network)
    end

    num_arcs = length(network.arcs) - 1
    ϕU = 1/epsilon
    λU = ϕU
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = ceil(Int, γ_ratio * num_interdictable)
    println("  Interdiction budget: γ = ceil($γ_ratio × $num_interdictable) = $γ")

    capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(ρ * γ * c_bar, digits=4)
    println("  Recovery budget: w = ρ·γ·c̄ = $ρ × $γ × $(round(c_bar, digits=2)) = $(round(w, digits=4))")

    capacity_scenarios_regular = capacities[1:end-1, :]
    R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
    uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

    source_arc_idx = findall(a -> a[1] == "s", network.arcs[1:num_arcs])
    max_flow_ub = maximum(sum(capacities[source_arc_idx, s] for s in 1:S) / S)
    max_cap = maximum(capacity_scenarios_regular)
    πU = ϕU
    yU = min(max_cap, ϕU)
    ytsU = min(max_flow_ub, ϕU)
    println("  LDR bounds: ϕU=$ϕU, πU=$πU, yU=$yU, ytsU=$ytsU")

    params = Dict(
        :S => S, :γ => γ, :ϕU => ϕU, :λU => λU, :w => w, :v => v,
        :πU => πU, :yU => yU, :ytsU => ytsU, :epsilon => epsilon,
    )
    return network, uncertainty_set, params
end

# ===== Instance 선택 =====
println("="^80)
println("네트워크 인스턴스 선택")
println("="^80)
println("  1. Grid network")
println("  2. Sioux Falls")
println("  3. Nobel US")
println("  4. Abilene")
println("  5. Polska")
print("선택 (1-5): ")
net_choice = parse(Int, readline())

if net_choice == 1
    print("Grid rows (m): "); m = parse(Int, readline())
    print("Grid cols (n): "); n = parse(Int, readline())
    network_configs[Symbol("grid_$(m)x$(n)")] = Dict(:type => :grid, :m => m, :n => n)
    instance_key = Symbol("grid_$(m)x$(n)")
elseif net_choice == 2
    instance_key = :sioux_falls
elseif net_choice == 3
    instance_key = :nobel_us
elseif net_choice == 4
    instance_key = :abilene
elseif net_choice == 5
    instance_key = :polska
else
    error("잘못된 선택: $net_choice")
end

print("시나리오 수 S: "); S = parse(Int, readline())
print("Cut strengthening (none/mw/sherali): "); strengthen_cuts = Symbol(readline())
print("LDR adjacency mode (both/head/tail/self) [both]: "); ldr_mode_str = strip(readline())
ldr_mode = isempty(ldr_mode_str) ? :both : Symbol(ldr_mode_str)

println("\n" * "="^80)
println("INSTANCE: $instance_key (S=$S, cuts=$strengthen_cuts, ldr_mode=$ldr_mode)")
println("="^80)

network, uncertainty_set, params = setup_instance(instance_key; S=S)
γ, ϕU, λU, w, v = params[:γ], params[:ϕU], params[:λU], params[:w], params[:v]
πU, yU, ytsU = params[:πU], params[:yU], params[:ytsU]

results = Dict{String, Any}()

# ===== 1. TR Nested Benders (baseline) =====
println("\n" * "="^80)
println("1. TR NESTED BENDERS — DUAL (outer=true, inner=true)")
println("="^80)

GC.gc()
model1, vars1 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, S=S)
t1_start = time()
result1 = tr_nested_benders_optimize!(model1, vars1, network, ϕU, λU, γ, w, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    outer_tr=true, inner_tr=true,
    πU=πU, yU=yU, ytsU=ytsU, strengthen_cuts=strengthen_cuts,
    parallel=true, mini_benders=true, max_mini_benders_iter=3, ldr_mode=ldr_mode)
t1_end = time()
results["tr_nested"] = t1_end - t1_start
println("\n>> TR Nested Benders time: $(round(results["tr_nested"], digits=2)) seconds")

# Cut pool 저장
# result1 전체 저장
result1_file = "result1_$(instance_key)_S$(S).jld2"
@save result1_file result1
println("  → result1 saved to $result1_file")

nb_cut_pool = get(result1, :cut_pool, nothing)
if nb_cut_pool !== nothing
    serialize("nb_cut_pool_$(instance_key)_S$(S).jls", nb_cut_pool)
    println("  → NB cut_pool saved ($(length(nb_cut_pool)) cuts)")
end
@infiltrate
# # ===== 2. Branch-and-Benders =====
# println("\n" * "="^80)
# println("2. BRANCH-AND-BENDERS (single tree)")
# println("="^80)

# GC.gc()
# model2, vars2 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, S=S)
# t2_start = time()
# result2 = branch_and_benders_optimize!(model2, vars2, network, ϕU, λU, γ, w, v, uncertainty_set;
#     mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
#     inner_tr=true,
#     strengthen_cuts=strengthen_cuts,
#     use_alpha_history=true, max_alpha_history=3,
#     lp_warmup_iters=1,
#     mipnode_freq=5,
#     πU=πU, yU=yU, ytsU=ytsU,
#     parallel=true)
# t2_end = time()
# results["branch_and_benders"] = t2_end - t2_start
# println("\n>> Branch-and-Benders time: $(round(results["branch_and_benders"], digits=2)) seconds")

# # Cut pool 저장
# bb_cut_pool = get(result2, :cut_pool, nothing)
# if bb_cut_pool !== nothing
#     serialize("bb_cut_pool_$(instance_key)_S$(S).jls", bb_cut_pool)
#     println("  → B&B cut_pool saved ($(length(bb_cut_pool)) cuts)")
# end

# ===== Summary =====
println("\n" * "="^80)
println("COMPARISON SUMMARY")
println("="^80)

# TR Nested Benders bounds
if haskey(result1, :past_local_lower_bound)
    nb_lb = minimum(result1[:past_local_lower_bound])
else
    nb_lb = result1[:past_lower_bound][end]
end
nb_ub = minimum(result1[:past_upper_bound])
nb_gap = abs(nb_ub - nb_lb) / max(abs(nb_ub), 1e-10)
nb_iters = length(result1[:past_upper_bound])

# Branch-and-Benders bounds
bb_status = result2[:termination_status]
if bb_status == MOI.OPTIMAL || bb_status == MOI.ALMOST_OPTIMAL
    bb_obj = result2[:obj_val]
    bb_lb = bb_obj  # optimal → LB = UB
    bb_ub = bb_obj
    bb_gap = 0.0
else
    bb_lb = has_values(model2) ? value(vars2[:t_0]) / S : NaN
    bb_ub = result2[:upper_bound]
    bb_gap = abs(bb_ub - bb_lb) / max(abs(bb_ub), 1e-10)
end
bb_cuts = result2[:lazy_cut_count]
bb_cbs = result2[:callback_count]
bb_ucuts = result2[:usercut_count]
bb_mipnodes = result2[:mipnode_calls]

header = "  " * rpad("Algorithm", 35) * rpad("Time(s)", 10) * rpad("LB", 12) * rpad("UB", 12) * rpad("Gap(%)", 10) * "Info"
println(header)
println("  " * "-"^85)
println("  " * rpad("1. TR Nested Benders (T,T)", 35) *
    rpad(round(results["tr_nested"], digits=2), 10) *
    rpad(round(nb_lb, digits=6), 12) * rpad(round(nb_ub, digits=6), 12) *
    rpad(round(nb_gap*100, digits=2), 10) * "$(nb_iters) iters")
println("  " * rpad("2. Branch-and-Benders", 35) *
    rpad(round(results["branch_and_benders"], digits=2), 10) *
    rpad(round(bb_lb, digits=6), 12) * rpad(round(bb_ub, digits=6), 12) *
    rpad(round(bb_gap*100, digits=2), 10) * "$(bb_cuts) lazy, $(bb_ucuts) user cuts, $(bb_cbs) cb, $(bb_mipnodes) mipnode")
println("  " * "-"^85)

# UB 일치 확인
if !isnan(nb_ub) && !isnan(bb_ub)
    ub_gap = abs(nb_ub - bb_ub)
    if ub_gap < 1e-3
        println("  ✓ Upper bounds match (gap = $(round(ub_gap, sigdigits=3)))")
    else
        println("  ✗ Upper bound mismatch! (gap = $(round(ub_gap, sigdigits=3)))")
    end
end
println("="^80)

# ===== Solution Verification =====
includet("build_dualized_outer_subprob.jl")

println("\n" * "="^80)
println("SOLUTION VERIFICATION (fix 1st-stage → solve dualized outer subproblem)")
println("="^80)

methods_to_verify = []

if @isdefined(result1)
    push!(methods_to_verify, ("1. TR Nested Benders", get(result1, :opt_sol, nothing), vars1, nb_ub))
end

if @isdefined(result2) && haskey(result2, :opt_sol)
    push!(methods_to_verify, ("2. Branch-and-Benders", result2[:opt_sol], vars2, bb_ub))
end

verify_header = "  " * rpad("Method", 35) * rpad("Reported", 12) * rpad("OSP obj", 12) * "Gap"
println(verify_header)
println("  " * "-"^65)

for (name, opt_sol, vrs, reported_obj) in methods_to_verify
    if opt_sol !== nothing
        λ_val = opt_sol[:λ] isa Number ? opt_sol[:λ] : value(opt_sol[:λ])
        x_val = opt_sol[:x] isa AbstractVector{<:Number} ? round.(opt_sol[:x]) : round.(value.(opt_sol[:x]))
        h_val = opt_sol[:h] isa AbstractVector{<:Number} ? opt_sol[:h] : value.(opt_sol[:h])
        ψ0_val = opt_sol[:ψ0] isa AbstractVector{<:Number} ? opt_sol[:ψ0] : value.(opt_sol[:ψ0])
    else
        λ_val = value(vrs[:λ])
        x_val = round.(value.(vrs[:x]))
        h_val = value.(vrs[:h])
        ψ0_val = value.(vrs[:ψ0])
    end

    GC.gc()
    osp_model, osp_vars, osp_data = build_dualized_outer_subproblem(
        network, S, ϕU, λU, γ, w, v, uncertainty_set,
        Mosek.Optimizer, λ_val, x_val, h_val, ψ0_val;
        πU=πU, yU=yU, ytsU=ytsU)

    optimize!(osp_model)
    osp_status = termination_status(osp_model)

    if osp_status == MOI.OPTIMAL || osp_status == MOI.ALMOST_OPTIMAL
        osp_obj = objective_value(osp_model)
        gap = abs(osp_obj - reported_obj)
        println("  " * rpad(name, 35) * rpad(round(reported_obj, digits=6), 12) *
            rpad(round(osp_obj, digits=6), 12) * "$(round(gap, sigdigits=3))")
    else
        println("  " * rpad(name, 35) * rpad(round(reported_obj, digits=6), 12) *
            rpad("$osp_status", 12) * "-")
    end
end
println("  " * "-"^65)
println("  (Gap ≈ 0 → solution is consistent with dualized outer subproblem)")
println("="^80)

@infiltrate
