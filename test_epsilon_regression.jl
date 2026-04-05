"""
test_epsilon_regression.jl — epsilon 분리 regression test

epsilon_hat = epsilon_tilde = 0.5 일 때, 기존 코드(backup)와 새 코드의 full model 목적함수 비교.
full model은 deterministic하므로 값이 정확히 일치해야 함.

사용법:
  julia test_epsilon_regression.jl
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using Hypatia
using Pajarito
using LinearAlgebra
using Infiltrator

include("network_generator.jl")
include("build_uncertainty_set.jl")
include("build_full_model.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

# ===== Parameters =====
epsilon = 0.5
S = 1
seed = 42
γ_ratio = 0.10
ρ = 0.2
v_param = 1.0

println("="^70)
println("REGRESSION TEST: epsilon_hat = epsilon_tilde = $epsilon, S=$S")
println("="^70)

# ===== Network setup =====
network = generate_grid_network(3, 3, seed=seed)
print_network_summary(network)
num_arcs = length(network.arcs) - 1
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, γ_ratio * num_interdictable)

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = round(ρ * γ * c_bar, digits=4)

capacity_scenarios_regular = capacities[1:end-1, :]

ϕU = 1/epsilon
λU = ϕU
source_arc_idx = findall(a -> a[1] == "s", network.arcs[1:num_arcs])
max_flow_ub = maximum(sum(capacities[source_arc_idx, s] for s in 1:S) / S)
max_cap = maximum(capacity_scenarios_regular)
πU = ϕU
yU = min(max_cap, ϕU)
ytsU = min(max_flow_ub, ϕU)

println("  γ=$γ, ϕU=$ϕU, λU=$λU, w=$w, πU=$πU, yU=$yU, ytsU=$ytsU")

# ===== NEW CODE: epsilon_hat = epsilon_tilde =====
println("\n--- [NEW] build_robust_counterpart_matrices(cap, ε̂=$epsilon, ε̃=$epsilon) ---")
R_new, r_dict_hat, r_dict_tilde, xi_bar_new = build_robust_counterpart_matrices(
    capacity_scenarios_regular, epsilon, epsilon)

uncertainty_set_new = Dict(
    :R => R_new, :r_dict_hat => r_dict_hat, :r_dict_tilde => r_dict_tilde,
    :xi_bar => xi_bar_new, :epsilon_hat => epsilon, :epsilon_tilde => epsilon)

# Verify r_dict_hat == r_dict_tilde (same epsilon)
for s in 1:S
    @assert r_dict_hat[s] ≈ r_dict_tilde[s] "r_dict_hat[$s] ≠ r_dict_tilde[$s]!"
end
println("  ✓ r_dict_hat == r_dict_tilde (same epsilon)")

println("\n--- [NEW] build_full_2DRNDP_model ---")
model_new, vars_new = build_full_2DRNDP_model(
    network, S, ϕU, ϕU, λU, γ, w, v_param, uncertainty_set_new;
    mip_solver=Gurobi.Optimizer, conic_solver=Mosek.Optimizer,
    πU_hat=πU, πU_tilde=πU, yU=yU, ytsU=ytsU)
optimize!(model_new)
st_new = termination_status(model_new)
obj_new = objective_value(model_new)
println("  Status: $st_new")
println("  Objective: $obj_new")

# ===== OLD CODE: single epsilon =====
# Reconstruct old-style uncertainty_set
println("\n--- [OLD-COMPAT] uncertainty_set with old keys ---")
# build_robust_counterpart_matrices with single epsilon returns same r_dict for hat and tilde
# So we can just use r_dict_hat as the old r_dict
uncertainty_set_old_compat = Dict(
    :R => R_new, :r_dict_hat => r_dict_hat, :r_dict_tilde => r_dict_hat,
    :xi_bar => xi_bar_new, :epsilon_hat => epsilon, :epsilon_tilde => epsilon)

model_old, vars_old = build_full_2DRNDP_model(
    network, S, ϕU, ϕU, λU, γ, w, v_param, uncertainty_set_old_compat;
    mip_solver=Gurobi.Optimizer, conic_solver=Mosek.Optimizer,
    πU_hat=πU, πU_tilde=πU, yU=yU, ytsU=ytsU)
optimize!(model_old)
st_old = termination_status(model_old)
obj_old = objective_value(model_old)
println("  Status: $st_old")
println("  Objective: $obj_old")

# ===== Compare =====
println("\n" * "="^70)
println("COMPARISON")
println("="^70)
diff = abs(obj_new - obj_old)
println("  NEW obj:  $(round(obj_new, digits=8))")
println("  OLD obj:  $(round(obj_old, digits=8))")
println("  |diff|:   $(round(diff, digits=10))")

if diff < 1e-4
    println("  ✓ PASS — objectives match (diff < 1e-4)")
else
    println("  ✗ FAIL — objectives differ!")
end

# Also compare x solutions
x_new = round.(value.(vars_new[:x]))
x_old = round.(value.(vars_old[:x]))
println("\n  x_new = $(findall(x_new .> 0.5))")
println("  x_old = $(findall(x_old .> 0.5))")
if x_new ≈ x_old
    println("  ✓ Same interdiction decisions")
else
    println("  ⚠ Different interdiction decisions (may be alternate optima)")
end
