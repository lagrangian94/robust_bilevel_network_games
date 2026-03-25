"""
Compare Benders decomposition algorithms:
0. Full Model (Pajarito: MIP + Conic outer approximation)
1. Strict Benders
2. TR Nested Benders — Dual (outer_tr=true, inner_tr=true)
3. TR Nested Benders — Hybrid (primal ISP inner + dual ISP outer, outer_tr=true, inner_tr=true)

병렬 실행하려면:
julia -t 4    # 4 threads
julia -t auto # CPU 코어 수만큼 자동

또는 환경변수:
set JULIA_NUM_THREADS=4
julia
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using Hypatia
using LinearAlgebra
using Infiltrator
using Plots
using Serialization
using Revise
using Pajarito

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("build_full_model.jl")
includet("parallel_utils.jl")
includet("strict_benders.jl")
includet("nested_benders_trust_region.jl")
includet("plot_benders.jl")
includet("ccg_benders.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary
using .NetworkGenerator: generate_sioux_falls_network, generate_nobel_us_network, generate_abilene_network, generate_polska_network, print_realworld_network_summary

# ===== Network Instance Configs =====
network_configs = Dict(
    :grid_3x3 => Dict(:type => :grid, :m => 3, :n => 3),
    :grid_5x5 => Dict(:type => :grid, :m => 5, :n => 5),
    :sioux_falls => Dict(:type => :real_world, :generator => generate_sioux_falls_network),
    :nobel_us => Dict(:type => :real_world, :generator => generate_nobel_us_network),
    :abilene => Dict(:type => :real_world, :generator => generate_abilene_network),
    :polska => Dict(:type => :real_world, :generator => generate_polska_network),
)

"""
    setup_instance(config_key; S, γ_ratio, ρ, v, seed, epsilon)

네트워크 인스턴스와 파라미터를 한 번에 생성. grid/real_world 공통 인터페이스.
Returns: (network, uncertainty_set, params::Dict)
"""
function setup_instance(config_key::Symbol;
    S=10, γ_ratio=0.10, ρ=0.2, v=1.0, seed=42, epsilon=0.5)

    config = network_configs[config_key]

    # --- Network 생성 ---
    if config[:type] == :grid
        network = generate_grid_network(config[:m], config[:n], seed=seed)
        print_network_summary(network)
    elseif config[:type] == :real_world
        network = config[:generator]()
        print_realworld_network_summary(network)
    end

    num_arcs = length(network.arcs) - 1

    # --- Parameters ---
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

    # --- Uncertainty set ---
    capacity_scenarios_regular = capacities[1:end-1, :]
    R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
    uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

    # --- LDR coefficient bounds ---
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
        :γ_ratio => γ_ratio, :ρ => ρ, :seed => seed,
    )

    return network, uncertainty_set, params
end

# ===== Interactive Instance 선택 =====
println("="^80)
println("네트워크 인스턴스 선택")
println("="^80)
println("  1. Grid network")
println("  2. Sioux Falls (24 nodes, 76 arcs)")
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

println("\n" * "="^80)
println("INSTANCE: $instance_key (S=$S, cuts=$strengthen_cuts)")
println("="^80)

network, uncertainty_set, params = setup_instance(instance_key; S=S)
γ, ϕU, λU, w, v = params[:γ], params[:ϕU], params[:λU], params[:w], params[:v]
πU, yU, ytsU = params[:πU], params[:yU], params[:ytsU]

results = Dict{String, Any}()


# # ===== 0. Full Model (Pajarito) =====
# println("\n" * "="^80)
# println("0. FULL MODEL (Pajarito: MIP + Conic)")
# println("="^80)

# """
# S=10, 5x5 grid networks에서 이미 한시간넘어도 수렴안함.
#      7    14    0.84251    3   20          -    0.09367      -   0.0  331s
#     27    38    3.73988    5   16   11.28263    1.61417  85.7%   0.0  640s
#     39    44    4.15700    6   16   11.28263    2.27480  79.8%   0.0  855s
#    352    39    8.82591   10    7    9.24680    5.80339  37.2%   0.0 3614s
# """

# GC.gc()
# model0, vars0 = build_full_2DRNDP_model(network, S, ϕU, λU, γ, w, v, uncertainty_set,
#     mip_solver=Gurobi.Optimizer, conic_solver=Mosek.Optimizer)
# add_sparsity_constraints!(model0, vars0, network, S)
# t0_start = time()
# optimize!(model0)
# t0_end = time()
# results["full_model"] = t0_end - t0_start

# t0_status = termination_status(model0)
# if t0_status == MOI.OPTIMAL || t0_status == MOI.ALMOST_OPTIMAL
#     obj0 = objective_value(model0)
#     println("  Optimal objective: $(round(obj0, digits=6))")
# else
#     obj0 = NaN
#     println("  Termination status: $t0_status (no optimal solution)")
# end
# println("\n>> Full Model time: $(results["full_model"]) seconds")
# # ===== 1. Strict Benders =====
# println("\n" * "="^80)
# println("1. STRICT BENDERS DECOMPOSITION (multi-cut)")
# println("="^80)

# GC.gc()
# model1, vars1 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, S=S)
# t1_start = time()
# result1 = strict_benders_optimize!(model1, vars1, network, ϕU, λU, γ, w, uncertainty_set; optimizer=Gurobi.Optimizer, πU=πU, yU=yU, ytsU=ytsU, strengthen_cuts=strengthen_cuts)
# t1_end = time()
# results["strict_benders"] = t1_end - t1_start
# println("\n>> Strict Benders time: $(results["strict_benders"]) seconds")
# @infiltrate
# ===== 2. TR Nested Benders — Dual (T,T) =====
println("\n" * "="^80)
println("2. TR NESTED BENDERS — DUAL (outer=true, inner=true)")
println("="^80)
"""
2700초 걸렸음
"""
GC.gc()
model2, vars2 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, S=S)
t2_start = time()
result2 = tr_nested_benders_optimize!(model2, vars2, network, ϕU, λU, γ, w, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
     outer_tr=true, inner_tr=true,
    πU=πU, yU=yU, ytsU=ytsU, strengthen_cuts=strengthen_cuts, parallel=true, mini_benders=true, max_mini_benders_iter=3)
t2_end = time()
results["tr_dual"] = t2_end - t2_start
println("\n>> Dual TR Both time: $(results["tr_dual"]) seconds")
@infiltrate
# # TODO:: solve only subset of scenarios (partial solve; 첫번째에선 다 풀어서 하한 다 찾아놓음) (upper bound eval. = iter N번마다 한번씩 full evaluate)
# ===== 2.5. Scenario-Decomposed Benders =====
# println("\n" * "="^80)
# println("2.5. SCENARIO-DECOMPOSED BENDERS (OMP → S × OSP(s=1))")
# println("="^80)
# """
# # >> Scenario-Decomposed Benders time: 1623.7960000038147 seconds
# # """

# GC.gc()
# model_sd, vars_sd = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, S=S)
# t_sd_start = time()
# result_sd = scenario_benders_optimize!(model_sd, vars_sd, network, ϕU, λU, γ, w, v, uncertainty_set;
#     conic_optimizer=Mosek.Optimizer, mip_optimizer=Gurobi.Optimizer,
#     πU=πU, yU=yU, ytsU=ytsU, parallel=true, strengthen_cuts=strengthen_cuts, outer_tr=true,inner_tr=true)
# t_sd_end = time()
# results["scenario_decomposed"] = t_sd_end - t_sd_start
# println("\n>> Scenario-Decomposed Benders time: $(results["scenario_decomposed"]) seconds")
# @infiltrate
# ===== 3. C&CG Benders =====
# println("\n" * "="^80)
# println("3. C&CG BENDERS (vertex enumeration + per-scenario Benders)")
# println("="^80)

# GC.gc()
# t3_start = time()
# result3 = ccg_benders_optimize!(network, ϕU, λU, γ, w, v, uncertainty_set;
#     mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
#     πU=πU, yU=yU, ytsU=ytsU, inner_tr=true, strengthen_cuts=strengthen_cuts, warm_start_cuts=true)
# t3_end = time()
# results["ccg_benders"] = t3_end - t3_start
# println("\n>> C&CG Benders time: $(results["ccg_benders"]) seconds")
# @infiltrate
# # ===== Summary =====
# println("\n" * "="^80)
# println("COMPARISON SUMMARY")
# println("="^80)
# println("  Parameters:")
# println("    Network:  |A|=$num_arcs, |A_I|=$num_interdictable")
# println("    S=$S, ε=$epsilon, ϕU=$ϕU, λU=$λU, v=$v, πU=$πU, yU=$yU, ytsU=$ytsU")
# println("    γ=$γ (ratio=$γ_ratio), w=$(round(w, digits=4)) (ρ=$ρ)")
# println()

# # --- 0. Full Model ---
# # Pajarito B&B → 단일 최적해, bound 개념 없음
# obj0_str = (t0_status == MOI.OPTIMAL || t0_status == MOI.ALMOST_OPTIMAL) ?
#     "$(round(obj0, digits=6))" : "$t0_status"

# --- 1. Strict Benders ---
sb_lb = result1[:past_obj][end]
sb_ub = minimum(result1[:past_upper_bound])
sb_gap = abs(sb_ub - sb_lb) / max(abs(sb_ub), 1e-10)
sb_iters = length(result1[:past_obj])

# --- 2. TR Nested Benders — Dual ---
# outer_tr=true → past_local_lower_bound 존재, past_upper_bound = min(subprob over all stages)
if haskey(result2, :past_local_lower_bound)
    nd_lb = minimum(result2[:past_local_lower_bound])
else
    nd_lb = result2[:past_lower_bound][end]
end
nd_ub = minimum(result2[:past_upper_bound])
nd_gap = abs(nd_ub - nd_lb) / max(abs(nd_ub), 1e-10)
nd_iters = length(result2[:past_upper_bound])

# --- 2.5. s
sd_lb = result_sd[:past_obj][end]
sd_ub = minimum(result_sd[:past_upper_bound])
sd_gap = abs(sd_ub - sd_lb) / max(abs(sd_ub), 1e-10)
sd_iters = length(result_sd[:past_obj])

# # --- 3. TR Nested Benders — Hybrid ---
# if haskey(result3, :past_local_lower_bound)
#     nh_lb = minimum(result3[:past_local_lower_bound])
# else
#     nh_lb = result3[:past_lower_bound][end]
# end
# nh_ub = minimum(result3[:past_upper_bound])
# nh_gap = abs(nh_ub - nh_lb) / max(abs(nh_ub), 1e-10)
# nh_iters = length(result3[:past_upper_bound])

# --- 3. C&CG Benders ---
ccg_lb = result3[:lower_bound]
ccg_ub = result3[:obj_val]
ccg_gap = abs(ccg_ub - ccg_lb) / max(abs(ccg_ub), 1e-10)
ccg_iters = length(result3[:history][:ccg_iter])
ccg_J = length(result3[:active_vertices])

# --- Summary Table ---
header = "  " * rpad("Algorithm", 30) * rpad("Time(s)", 10) * rpad("LB", 12) * rpad("UB", 12) * rpad("Gap(%)", 10) * "Iters"
println(header)
println("  " * "-"^76)
println("  " * rpad("0. Full Model (Pajarito)", 30) * rpad(round(results["full_model"], digits=2), 10) *
    rpad(obj0_str, 12) * rpad(obj0_str, 12) * rpad("0.0", 10) * "-")
println("  " * rpad("1. Strict Benders", 30) * rpad(round(results["strict_benders"], digits=2), 10) *
    rpad(round(sb_lb, digits=6), 12) * rpad(round(sb_ub, digits=6), 12) *
    rpad(round(sb_gap*100, digits=2), 10) * "$sb_iters")
println("  " * rpad("2. TR Dual (T,T)", 30) * rpad(round(results["tr_dual"], digits=2), 10) *
    rpad(round(nd_lb, digits=6), 12) * rpad(round(nd_ub, digits=6), 12) *
    rpad(round(nd_gap*100, digits=2), 10) * "$nd_iters")
println("  " * rpad("2.5. Scenario-Decomposed", 30) * rpad(round(results["scenario_decomposed"], digits=2), 10) *
    rpad(round(sd_lb, digits=6), 12) * rpad(round(sd_ub, digits=6), 12) *
    rpad(round(sd_gap*100, digits=2), 10) * "$sd_iters")
# println("  " * rpad("3. TR Hybrid (T,T)", 30) * rpad(round(results["tr_hybrid"], digits=2), 10) *
#     rpad(round(nh_lb, digits=6), 12) * rpad(round(nh_ub, digits=6), 12) *
#     rpad(round(nh_gap*100, digits=2), 10) * "$nh_iters")
println("  " * rpad("3. C&CG Benders", 30) * rpad(round(results["ccg_benders"], digits=2), 10) *
    rpad(round(ccg_lb, digits=6), 12) * rpad(round(ccg_ub, digits=6), 12) *
    rpad(round(ccg_gap*100, digits=2), 10) * "$ccg_iters (|J|=$ccg_J)")
println("  " * "-"^76)

# UB 일치 확인 (best feasible solution 비교)
all_ubs = filter(!isnan, [sb_ub, sd_ub, nd_ub, ccg_ub])
if length(all_ubs) >= 2
    max_ub_gap = maximum(abs(a - b) for a in all_ubs for b in all_ubs)
    if max_ub_gap < 1e-3
        println("  ✓ All upper bounds match (max gap = $(round(max_ub_gap, sigdigits=3)))")
    else
        println("  ✗ Upper bound mismatch! (max gap = $(round(max_ub_gap, sigdigits=3)))")
    end
end
println("="^80)

# ===== Solution Verification via Dualized Outer Subproblem =====
includet("build_dualized_outer_subprob.jl")

println("\n" * "="^80)
println("SOLUTION VERIFICATION (fix 1st-stage → solve dualized outer subproblem)")
println("="^80)

# 각 method의 (name, opt_sol, vars, reported_obj) 수집
# opt_sol: Dict(:λ=>val, :x=>val, :h=>val, :ψ0=>val) 또는 nothing (JuMP vars에서 추출)
methods_to_verify = []

# 0. Full Model
if @isdefined(vars0) && (t0_status == MOI.OPTIMAL || t0_status == MOI.ALMOST_OPTIMAL)
    push!(methods_to_verify, ("0. Full Model", nothing, vars0, obj0))
end

# 1. Strict Benders
if @isdefined(result1)
    push!(methods_to_verify, ("1. Strict Benders", get(result1, :opt_sol, nothing), vars1, sb_ub))
end

# 2. TR Nested Benders — Dual
if @isdefined(result2)
    push!(methods_to_verify, ("2. TR Dual (T,T)", get(result2, :opt_sol, nothing), vars2, nd_ub))
end

# 2.5. Scenario-Decomposed Benders
if @isdefined(result_sd)
    push!(methods_to_verify, ("2.5. Scenario-Decomposed", get(result_sd, :opt_sol, nothing), vars_sd, sd_ub))
end

# 3. C&CG Benders
if @isdefined(result3) && haskey(result3, :opt_sol)
    push!(methods_to_verify, ("3. C&CG Benders", result3[:opt_sol], nothing, ccg_ub))
end

verify_header = "  " * rpad("Method", 30) * rpad("Reported", 12) * rpad("OSP obj", 12) * "Gap"
println(verify_header)
println("  " * "-"^60)

for (name, opt_sol, vrs, reported_obj) in methods_to_verify
    # 1st-stage solution 추출: opt_sol이 있으면 거기서, 없으면 JuMP 변수에서
    if opt_sol !== nothing
        λ_val = opt_sol[:λ] isa Number ? opt_sol[:λ] : value(opt_sol[:λ])
        x_val = opt_sol[:x] isa AbstractVector{<:Number} ? round.(opt_sol[:x]) : round.(value.(opt_sol[:x]))
        h_val = opt_sol[:h] isa AbstractVector{<:Number} ? opt_sol[:h] : value.(opt_sol[:h])
        ψ0_val = opt_sol[:ψ0] isa AbstractVector{<:Number} ? opt_sol[:ψ0] : value.(opt_sol[:ψ0])
        println("  [using result[:opt_sol]]")
    else
        λ_val = value(vrs[:λ])
        x_val = round.(value.(vrs[:x]))
        h_val = value.(vrs[:h])
        ψ0_val = value.(vrs[:ψ0])
    end
    GC.gc()
    # Dualized outer subproblem 구축 및 풀기
    osp_model, osp_vars, osp_data = build_dualized_outer_subproblem(
        network, S, ϕU, λU, γ, w, v, uncertainty_set,
        Mosek.Optimizer, λ_val, x_val, h_val, ψ0_val;
        πU=πU, yU=yU, ytsU=ytsU)

    optimize!(osp_model)
    osp_status = termination_status(osp_model)

    if osp_status == MOI.OPTIMAL || osp_status == MOI.ALMOST_OPTIMAL
        osp_obj = objective_value(osp_model)
        gap = abs(osp_obj - reported_obj)
        gap_str = round(gap, sigdigits=3)
        println("  " * rpad(name, 30) * rpad(round(reported_obj, digits=6), 12) *
            rpad(round(osp_obj, digits=6), 12) * "$gap_str")
    else
        println("  " * rpad(name, 30) * rpad(round(reported_obj, digits=6), 12) *
            rpad("$osp_status", 12) * "-")
    end
end
println("  " * "-"^60)
println("  (Gap ≈ 0 → solution is consistent with dualized outer subproblem)")
println("="^80)
