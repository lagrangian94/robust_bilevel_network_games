"""
Test CCG Benders vs Strict Benders on small instances.

Usage (in Julia REPL):
    include("test_ccg_benders.jl")
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
includet("build_full_model.jl")
includet("parallel_utils.jl")
includet("strict_benders.jl")
includet("nested_benders_trust_region.jl")
includet("ccg_benders.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

# ===== Instance setup =====
m, n = 3, 3
S = 1
seed = 42
epsilon = 0.5
γ_ratio = 0.10
ρ = 0.2
v = 1.0

network = generate_grid_network(m, n, seed=seed)
print_network_summary(network)
num_arcs = length(network.arcs) - 1

ϕU = 1 / epsilon
λU = ϕU
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, γ_ratio * num_interdictable)
println("  Interdiction budget: γ = $γ")

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = round(ρ * γ * c_bar, digits=4)
println("  Recovery budget: w = $w")

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

# ===== 1. Strict Benders (baseline) =====
println("\n" * "=" ^ 60)
println("1. STRICT BENDERS (baseline)")
println("=" ^ 60)

GC.gc()
model1, vars1 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut_lf=true, S=S)
t1_start = time()
result1 = strict_benders_optimize!(model1, vars1, network, ϕU, λU, γ, w, uncertainty_set;
    optimizer=Gurobi.Optimizer, πU=πU, yU=yU, ytsU=ytsU)
t1_end = time()
obj1 = result1[:past_upper_bound][end]
println("\n>> Strict Benders: obj=$(round(obj1, digits=6)), time=$(round(t1_end - t1_start, digits=1))s")

# ===== 2. CCG Benders =====
println("\n" * "=" ^ 60)
println("2. CCG BENDERS")
println("=" ^ 60)

GC.gc()
t2_start = time()
result2 = ccg_benders_optimize!(network, ϕU, λU, γ, w, v, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    πU=πU, yU=yU, ytsU=ytsU,
    ε_benders=1e-4, ε_pricing=1e-4,
    inner_tr=true, tol=1e-4)
t2_end = time()
obj2 = result2[:obj_val]
println("\n>> CCG Benders: obj=$(round(obj2, digits=6)), time=$(round(t2_end - t2_start, digits=1))s")
println("   |J| = $(length(result2[:active_vertices])), J = $(result2[:active_vertices])")

# ===== Comparison =====
println("\n" * "=" ^ 60)
println("COMPARISON")
println("=" ^ 60)
gap = abs(obj1 - obj2) / max(abs(obj1), 1e-10)
println("  Strict Benders obj: $(round(obj1, digits=6))")
println("  CCG Benders obj:    $(round(obj2, digits=6))")
println("  Gap:                $(round(gap * 100, digits=4))%")
println("  Match: $(gap < 1e-3 ? "YES" : "NO")")
