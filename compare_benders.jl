"""
Compare Benders decomposition algorithms:
0. Full Model (Pajarito: MIP + Conic outer approximation)
1. Strict Benders
2. TR Nested Benders — Dual (outer_tr=true, inner_tr=true)
3. TR Nested Benders — Hybrid (primal ISP inner + dual ISP outer, outer_tr=true, inner_tr=true)
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
includet("strict_benders.jl")
includet("nested_benders_trust_region.jl")
includet("plot_benders.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

# ===== Common Parameters =====
S = 10
γ_ratio = 0.10
ρ = 0.2
v = 1.0
seed = 42
epsilon = 0.5
ϕU = 1/epsilon ## 10.0 -> 1/epsilon으로 변경.
λU = ϕU ## 10.0 -> ϕU로 변경.


# ===== Generate Network & Uncertainty Set =====
println("="^80)
println("GENERATING NETWORK AND UNCERTAINTY SET")
println("="^80)

network = generate_grid_network(5, 5, seed=seed)
print_network_summary(network)

num_arcs = length(network.arcs) - 1
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

# --- Tight LDR coefficient bounds (Idea 3: analytic bounds) ---
# ϕU: flow dual Φ̂/Φ̃ LDR coeff bound (also McCormick) — stays as is
# πU: node price Π̂/Π̃ — bounded by 1 (max-flow value in standard form, d0=[0,...,1])
# yU: follower flow Ỹ — bounded by max arc capacity (per-arc flow ≤ capacity)
# ytsU: dummy arc Ỹ_ts — bounded by max-flow value (≤ sum of source-out capacities)
#
# 물리적 해석: node price ∈ [0,1], flow ≤ capacity, total flow ≤ min-cut
# LDR coeff는 value 범위에서 결정됨 (ζ는 dimensionless, ||ζ||≤ε)
source_arc_idx = findall(a -> a[1] == "s", network.arcs[1:num_arcs])
max_flow_ub = maximum(sum(capacities[source_arc_idx, s] for s in 1:S) / S)  # avg max-flow upper bound
max_cap = maximum(capacity_scenarios_regular)

πU = ϕU                          # node price: 이론적으로 동일 (LDR coeff ≤ 1/ε)
yU = min(max_cap, ϕU)            # analytic > ϕU이면 ϕU 유지
ytsU = min(max_flow_ub, ϕU)      # analytic > ϕU이면 ϕU 유지
println("  LDR bounds: ϕU=$ϕU, πU=$πU, yU=$yU, ytsU=$ytsU")
strengthen_cuts = :mw # :none, :mw, :sherali
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
# model1, vars1 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut_lf=true)
# t1_start = time()
# result1 = strict_benders_optimize!(model1, vars1, network, ϕU, λU, γ, w, uncertainty_set; optimizer=Gurobi.Optimizer, multi_cut_lf=true, πU=πU, yU=yU, ytsU=ytsU, strengthen_cuts=strengthen_cuts)
# t1_end = time()
# results["strict_benders"] = t1_end - t1_start
# println("\n>> Strict Benders time: $(results["strict_benders"]) seconds")


# ===== 2. TR Nested Benders — Dual (T,T) =====
println("\n" * "="^80)
println("2. TR NESTED BENDERS — DUAL (outer=true, inner=true)")
println("="^80)

GC.gc()
model2, vars2 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut_lf=true, multi_cut_scenario=true, S=S)
t2_start = time()
result2 = tr_nested_benders_optimize!(model2, vars2, network, ϕU, λU, γ, w, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    multi_cut_lf=true, multi_cut_scenario=true, outer_tr=false, inner_tr=true,
    πU=πU, yU=yU, ytsU=ytsU, strengthen_cuts=strengthen_cuts)
t2_end = time()
results["tr_dual"] = t2_end - t2_start
println("\n>> Dual TR Both time: $(results["tr_dual"]) seconds")

@infiltrate

# ===== 3. TR Nested Benders — Hybrid (T,T) =====
println("\n" * "="^80)
println("3. TR NESTED BENDERS — HYBRID (primal ISP inner + dual ISP outer)")
println("="^80)

GC.gc()
model3, vars3 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut_lf=true)
t3_start = time()
result3 = tr_nested_benders_optimize_hybrid!(model3, vars3, network,
    ϕU, λU, γ, w, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    multi_cut_lf=true, outer_tr=true, inner_tr=true,
    πU=πU, yU=yU, ytsU=ytsU, strengthen_cuts=strengthen_cuts)
t3_end = time()
results["tr_hybrid"] = t3_end - t3_start
println("\n>> Hybrid time: $(results["tr_hybrid"]) seconds")

# ===== Summary =====
println("\n" * "="^80)
println("COMPARISON SUMMARY")
println("="^80)
println("  Parameters:")
println("    Network:  |A|=$num_arcs, |A_I|=$num_interdictable")
println("    S=$S, ε=$epsilon, ϕU=$ϕU, λU=$λU, v=$v, πU=$πU, yU=$yU, ytsU=$ytsU")
println("    γ=$γ (ratio=$γ_ratio), w=$(round(w, digits=4)) (ρ=$ρ)")
println()

# --- 0. Full Model ---
# Pajarito B&B → 단일 최적해, bound 개념 없음
obj0_str = (t0_status == MOI.OPTIMAL || t0_status == MOI.ALMOST_OPTIMAL) ?
    "$(round(obj0, digits=6))" : "$t0_status"

# --- 1. Strict Benders ---
# past_obj = OMP objective (lower bound), past_upper_bound = min(subprob_obj) over iters
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

# --- 3. TR Nested Benders — Hybrid ---
if haskey(result3, :past_local_lower_bound)
    nh_lb = minimum(result3[:past_local_lower_bound])
else
    nh_lb = result3[:past_lower_bound][end]
end
nh_ub = minimum(result3[:past_upper_bound])
nh_gap = abs(nh_ub - nh_lb) / max(abs(nh_ub), 1e-10)
nh_iters = length(result3[:past_upper_bound])

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
println("  " * rpad("3. TR Hybrid (T,T)", 30) * rpad(round(results["tr_hybrid"], digits=2), 10) *
    rpad(round(nh_lb, digits=6), 12) * rpad(round(nh_ub, digits=6), 12) *
    rpad(round(nh_gap*100, digits=2), 10) * "$nh_iters")
println("  " * "-"^76)

# UB 일치 확인 (best feasible solution 비교)
all_ubs = filter(!isnan, [obj0, sb_ub, nd_ub, nh_ub])
if length(all_ubs) >= 2
    max_ub_gap = maximum(abs(a - b) for a in all_ubs for b in all_ubs)
    if max_ub_gap < 1e-3
        println("  ✓ All upper bounds match (max gap = $(round(max_ub_gap, sigdigits=3)))")
    else
        println("  ✗ Upper bound mismatch! (max gap = $(round(max_ub_gap, sigdigits=3)))")
    end
end
println("="^80)
