"""
Compare Benders decomposition algorithms:
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

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("strict_benders.jl")
includet("nested_benders_trust_region.jl")
includet("plot_benders.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

# ===== Common Parameters =====
S = 1
λU = 10.0
γ_ratio = 0.10
ρ = 0.2
v = 1.0
seed = 42
epsilon = 0.5
ϕU = 1/epsilon
λU = ϕU ## 왜 λU <= ϕU 해야 에러가 안나는지?

# ===== Generate Network & Uncertainty Set =====
println("="^80)
println("GENERATING NETWORK AND UNCERTAINTY SET")
println("="^80)

network = generate_grid_network(4, 4, seed=seed)
print_network_summary(network)

num_arcs = length(network.arcs) - 1
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, γ_ratio * num_interdictable)
println("  Interdiction budget: γ = ceil($γ_ratio × $num_interdictable) = $γ")

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)

interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = ρ * γ * c_bar
println("  Recovery budget: w = ρ·γ·c̄ = $ρ × $γ × $(round(c_bar, digits=2)) = $(round(w, digits=4))")

capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

results = Dict{String, Any}()

γ=2.0
w=1.0

# ===== 1. Strict Benders =====
println("\n" * "="^80)
println("1. STRICT BENDERS DECOMPOSITION")
println("="^80)

GC.gc()
model1, vars1 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=false)
t1_start = time()
result1 = strict_benders_optimize!(model1, vars1, network, ϕU, λU, γ, w, uncertainty_set; optimizer=Gurobi.Optimizer)
t1_end = time()
results["strict_benders"] = t1_end - t1_start
println("\n>> Strict Benders time: $(results["strict_benders"]) seconds")

# ===== 2. TR Nested Benders — Dual (T,T) =====
println("\n" * "="^80)
println("2. TR NESTED BENDERS — DUAL (outer=true, inner=true)")
println("="^80)

GC.gc()
model2, vars2 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
t2_start = time()
result2 = tr_nested_benders_optimize!(model2, vars2, network, ϕU, λU, γ, w, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    multi_cut=true, outer_tr=true, inner_tr=true)
t2_end = time()
results["tr_dual"] = t2_end - t2_start
println("\n>> Dual TR Both time: $(results["tr_dual"]) seconds")

# ===== 3. TR Nested Benders — Hybrid (T,T) =====
println("\n" * "="^80)
println("3. TR NESTED BENDERS — HYBRID (primal ISP inner + dual ISP outer)")
println("="^80)

GC.gc()
model3, vars3 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
t3_start = time()
result3 = tr_nested_benders_optimize_hybrid!(model3, vars3, network,
    ϕU, λU, γ, w, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    multi_cut=true, outer_tr=true, inner_tr=true)
t3_end = time()
results["tr_hybrid"] = t3_end - t3_start
println("\n>> Hybrid time: $(results["tr_hybrid"]) seconds")

# ===== Summary =====
println("\n" * "="^80)
println("COMPARISON SUMMARY")
println("="^80)
println("  Parameters:")
println("    Network:  3×3 grid, |A|=$num_arcs, |A_I|=$num_interdictable")
println("    S=$S, ε=$epsilon, ϕU=$ϕU, λU=$λU, v=$v")
println("    γ=$γ (ratio=$γ_ratio), w=$(round(w, digits=4)) (ρ=$ρ)")
println()

function extract_obj(r)
    if haskey(r, :past_local_lower_bound)
        return minimum(r[:past_local_lower_bound])
    elseif haskey(r, :past_lower_bound)
        return r[:past_lower_bound][end]
    elseif haskey(r, :past_subprob_obj)
        return r[:past_subprob_obj][end]
    else
        return NaN
    end
end

obj1 = extract_obj(result1)
obj2 = extract_obj(result2)
obj3 = extract_obj(result3)

println("  " * rpad("Algorithm", 30) * rpad("Time (sec)", 14) * "Obj. value")
println("  " * "-"^56)
println("  " * rpad("1. Strict Benders", 30) * rpad(round(results["strict_benders"], digits=2), 14) * "$(round(obj1, digits=6))")
println("  " * rpad("2. TR Dual (T,T)", 30) * rpad(round(results["tr_dual"], digits=2), 14) * "$(round(obj2, digits=6))")
println("  " * rpad("3. TR Hybrid (T,T)", 30) * rpad(round(results["tr_hybrid"], digits=2), 14) * "$(round(obj3, digits=6))")
println("  " * "-"^56)

# 목적함수 일치 확인
all_objs = filter(!isnan, [obj1, obj2, obj3])
if length(all_objs) >= 2
    max_obj_gap = maximum(abs(a - b) for a in all_objs for b in all_objs)
    if max_obj_gap < 1e-3
        println("  ✓ All objectives match (max gap = $(round(max_obj_gap, sigdigits=3)))")
    else
        println("  ✗ Objective mismatch! (max gap = $(round(max_obj_gap, sigdigits=3)))")
    end
end
println("="^80)
