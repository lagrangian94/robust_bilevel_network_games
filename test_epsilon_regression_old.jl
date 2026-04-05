"""
test_epsilon_regression_old.jl — OLD vs NEW full model 비교

OLD: backup_before_epsilon_split/build_full_model_old_renamed.jl (git HEAD)
  - 단일 ϕU, uncertainty_set[:r_dict], [:epsilon]
NEW: build_full_model.jl (epsilon 분리 후)
  - ϕU_hat=ϕU_tilde=ϕU, uncertainty_set[:r_dict_hat]=[:r_dict_tilde], [:epsilon_hat]=[:epsilon_tilde]

사용법: julia test_epsilon_regression_old.jl
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using Hypatia
using Pajarito
using LinearAlgebra
using SparseArrays
using Infiltrator

include("network_generator.jl")
include("build_uncertainty_set.jl")  # NEW code (returns 4 values with default)

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model

# OLD function (renamed to avoid conflict)
include("backup_before_epsilon_split/build_full_model_old_renamed.jl")
# NEW function
include("build_full_model.jl")

# ===== Parameters =====
epsilon = 0.5
S = 1
seed = 42
γ_ratio = 0.10
ρ = 0.2
v_param = 1.0

network = generate_grid_network(3, 3, seed=seed)
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

println("="^70)
println("REGRESSION TEST: OLD vs NEW (3x3 grid, S=$S, ε=$epsilon)")
println("  γ=$γ, ϕU=$ϕU, λU=$λU, w=$w, πU=$πU, yU=$yU, ytsU=$ytsU")
println("="^70)

# ===== Build uncertainty sets =====
# NEW: returns 4 values
R, r_dict_hat, r_dict_tilde, xi_bar = build_robust_counterpart_matrices(
    capacity_scenarios_regular, epsilon, epsilon)

# Verify hat == tilde
for s in 1:S
    @assert r_dict_hat[s] ≈ r_dict_tilde[s] "r_dict mismatch!"
end
println("  ✓ r_dict_hat == r_dict_tilde")

# OLD format
uncertainty_set_old = Dict(:R => R, :r_dict => r_dict_hat, :xi_bar => xi_bar, :epsilon => epsilon)
# NEW format
uncertainty_set_new = Dict(:R => R, :r_dict_hat => r_dict_hat, :r_dict_tilde => r_dict_tilde,
    :xi_bar => xi_bar, :epsilon_hat => epsilon, :epsilon_tilde => epsilon)

# ===== 1. OLD code =====
println("\n--- [OLD] build_full_2DRNDP_model_OLD ---")
model_old, vars_old = build_full_2DRNDP_model_OLD(
    network, S, ϕU, λU, γ, w, v_param, uncertainty_set_old;
    mip_solver=Gurobi.Optimizer, conic_solver=Mosek.Optimizer,
    πU=πU, yU=yU, ytsU=ytsU)
optimize!(model_old)
st_old = termination_status(model_old)
obj_old = objective_value(model_old)
x_old = round.(value.(vars_old[:x]))
println("  Status: $st_old")
println("  Obj:    $obj_old")
println("  x:      $(findall(x_old .> 0.5))")

# ===== 2. NEW code =====
println("\n--- [NEW] build_full_2DRNDP_model ---")
model_new, vars_new = build_full_2DRNDP_model(
    network, S, ϕU, ϕU, λU, γ, w, v_param, uncertainty_set_new;
    mip_solver=Gurobi.Optimizer, conic_solver=Mosek.Optimizer,
    πU_hat=πU, πU_tilde=πU, yU=yU, ytsU=ytsU)
optimize!(model_new)
st_new = termination_status(model_new)
obj_new = objective_value(model_new)
x_new = round.(value.(vars_new[:x]))
println("  Status: $st_new")
println("  Obj:    $obj_new")
println("  x:      $(findall(x_new .> 0.5))")

# ===== 3. Compare =====
println("\n" * "="^70)
println("RESULT")
println("="^70)
diff = abs(obj_new - obj_old)
println("  OLD obj: $(round(obj_old, digits=8))")
println("  NEW obj: $(round(obj_new, digits=8))")
println("  |diff|:  $(round(diff, digits=10))")

if diff < 1e-4
    println("\n  ✓ PASS — objectives match (diff < 1e-4)")
else
    println("\n  ✗ FAIL — objectives DIFFER by $diff")
end

if x_new ≈ x_old
    println("  ✓ Same interdiction decisions")
else
    println("  ⚠ Different interdiction (alternate optima possible)")
end
