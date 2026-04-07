"""
test_inner_tr_duality.jl — Reproduce ISP follower strong duality failure
when inner_tr=true with network-dependent γ, w on 4×4 grid.
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
includet("../nested_benders_trust_region.jl")

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

# ===== Run tr_nested_benders_optimize! with inner_tr=TRUE =====
println("\n" * "="^60)
println("Running tr_nested_benders_optimize! with inner_tr=TRUE")
println("="^60)

omp_model, omp_vars = build_omp(network, ϕU, λU, γ, w;
    optimizer=Gurobi.Optimizer, multi_cut=true)

result = tr_nested_benders_optimize!(omp_model, omp_vars, network,
    ϕU, λU, γ, w, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    multi_cut=true, outer_tr=true, inner_tr=true)

println("\nResult:")
println("  Solution time: $(result[:solution_time]) s")
if haskey(result, :opt_sol)
    println("  x* = $(result[:opt_sol][:x])")
    println("  λ* = $(result[:opt_sol][:λ])")
end
