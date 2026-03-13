"""
test_intercept_fix.jl — Compare original (inner_tr=false) vs patched (inner_tr=true, intercept 역산)
on 4×4 grid with network-dependent γ, w.
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using LinearAlgebra
using Infiltrator
using Revise

includet("../network_generator.jl")
includet("../build_uncertainty_set.jl")
includet("../strict_benders.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

# ===== Parameters =====
S = 1
λU = 10.0
γ_ratio = 0.10
ρ_param = 0.2
v = 1.0
seed = 42
epsilon = 0.5
ϕU = 1/epsilon

# ===== Generate 4×4 Network =====
network = generate_grid_network(4, 4, seed=seed)
num_arcs = length(network.arcs) - 1
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, γ_ratio * num_interdictable)
capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = ρ_param * γ * c_bar
println("γ=$γ, w=$(round(w, digits=4)), w/S=$(round(w/S, digits=4))")

capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

# =====================================================================
# TEST 1: Original solver, inner_tr=false (baseline)
# =====================================================================
println("\n" * "="^70)
println("TEST 1: Original solver, inner_tr=FALSE (baseline)")
println("="^70)

includet("../nested_benders_trust_region.jl")

omp1, vars1 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
result1 = tr_nested_benders_optimize!(omp1, vars1, network, ϕU, λU, γ, w, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    multi_cut=true, outer_tr=true, inner_tr=false)

println("  Time: $(round(result1[:solution_time], digits=2))s")
println("  Outer iters: $(length(result1[:inner_iter]))")
println("  Total inner iters: $(sum(result1[:inner_iter]))")
if haskey(result1, :past_local_lower_bound)
    obj1 = minimum(result1[:past_local_lower_bound])
    println("  Objective: $obj1")
end
if haskey(result1, :opt_sol)
    println("  x* = $(result1[:opt_sol][:x])")
    println("  λ* = $(result1[:opt_sol][:λ])")
end

# =====================================================================
# TEST 2: Patched solver (intercept 역산), inner_tr=true
# =====================================================================
println("\n" * "="^70)
println("TEST 2: Patched solver (intercept fix), inner_tr=TRUE")
println("="^70)

includet("nested_benders_trust_region_debug.jl")

omp2, vars2 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
result2 = tr_nested_benders_optimize!(omp2, vars2, network, ϕU, λU, γ, w, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    multi_cut=true, outer_tr=true, inner_tr=true)

println("  Time: $(round(result2[:solution_time], digits=2))s")
println("  Outer iters: $(length(result2[:inner_iter]))")
println("  Total inner iters: $(sum(result2[:inner_iter]))")
if haskey(result2, :past_local_lower_bound)
    obj2 = minimum(result2[:past_local_lower_bound])
    println("  Objective: $obj2")
end
if haskey(result2, :opt_sol)
    println("  x* = $(result2[:opt_sol][:x])")
    println("  λ* = $(result2[:opt_sol][:λ])")
end

# =====================================================================
# COMPARISON
# =====================================================================
println("\n" * "="^70)
println("COMPARISON")
println("="^70)

if haskey(result1, :past_local_lower_bound) && haskey(result2, :past_local_lower_bound)
    obj1 = minimum(result1[:past_local_lower_bound])
    obj2 = minimum(result2[:past_local_lower_bound])
    gap = abs(obj1 - obj2)
    println("  Baseline obj (inner_tr=false): $obj1")
    println("  Patched obj  (inner_tr=true):  $obj2")
    println("  Gap: $gap  $(gap < 1e-3 ? "✓ OK" : "⚠ MISMATCH")")
end

if haskey(result1, :opt_sol) && haskey(result2, :opt_sol)
    x_gap = maximum(abs.(result1[:opt_sol][:x] - result2[:opt_sol][:x]))
    println("  x* max diff: $x_gap  $(x_gap < 1e-3 ? "✓" : "⚠ different x")")
end

println("\n  Baseline: outer=$(length(result1[:inner_iter])), inner=$(sum(result1[:inner_iter])), time=$(round(result1[:solution_time], digits=2))s")
println("  Patched:  outer=$(length(result2[:inner_iter])), inner=$(sum(result2[:inner_iter])), time=$(round(result2[:solution_time], digits=2))s")
