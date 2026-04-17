"""
test_vi_infeas.jl — obj_F ≥ 0 VI feasibility 진단.
"""

using Revise
using JuMP
using Gurobi
using Printf
using LinearAlgebra
using Random
using Infiltrator

include("../network_generator.jl")
using .NetworkGenerator

include("true_dro_data.jl")
include("true_dro_build_omp.jl")
include("true_dro_build_subproblem.jl")

# Abilene γ=2
network = generate_abilene_network()
num_arcs = length(network.arcs) - 1
S = 10
γ = 2
ρ = 0.2
seed = 42
ε = 0.1

capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
    interdictable_arcs=network.interdictable_arcs, seed=seed)

interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = round(ρ * γ * c_bar; digits=4)
λU = 2.0
q_hat = fill(1.0 / S, S)

td = make_true_dro_data(network, capacities, q_hat, ε, ε; w=w, lambda_U=λU, gamma=γ)
K = td.num_arcs

println("=" ^ 70)
println("Abilene γ=2, S=10, with VI")
println("=" ^ 70)

# Test 1: x=0 build + solve
println("\n--- Test 1: Build at x=0, solve directly (no update) ---")
model1, vars1 = build_true_dro_subproblem(td, zeros(K);
    optimizer=Gurobi.Optimizer, silent=true, add_objF_vi=true)
optimize!(model1)
println("  status = $(termination_status(model1))")
if termination_status(model1) == MOI.OPTIMAL
    @printf("  Z₀ = %.6f\n", objective_value(model1))
end

# Test 2: build at x=0, update to random x, solve
rng = MersenneTwister(42)
x1 = zeros(K); x1[interdictable_idx[1:γ]] .= 1.0
println("\n--- Test 2: Build at x=0, update to x=$(round.(Int, x1)), solve ---")
update_true_dro_subproblem_objective!(model1, vars1, td, x1)
optimize!(model1)
println("  status = $(termination_status(model1))")
if termination_status(model1) == MOI.OPTIMAL
    @printf("  Z₀ = %.6f\n", objective_value(model1))
end

# Test 3: build at x1 directly
println("\n--- Test 3: Build at x=$(round.(Int, x1)) directly ---")
model2, vars2 = build_true_dro_subproblem(td, x1;
    optimizer=Gurobi.Optimizer, silent=true, add_objF_vi=true)
optimize!(model2)
println("  status = $(termination_status(model2))")
if termination_status(model2) == MOI.OPTIMAL
    @printf("  Z₀ = %.6f\n", objective_value(model2))
end

# Test 4: 여러 random x
println("\n--- Test 4: build at x=0, sequential updates ---")
for i in 1:5
    selected = shuffle(rng, interdictable_idx)[1:γ]
    x = zeros(K); x[selected] .= 1.0
    update_true_dro_subproblem_objective!(model1, vars1, td, x)
    optimize!(model1)
    st = termination_status(model1)
    if st == MOI.OPTIMAL
        @printf("  i=%d  Z₀=%.4f  [%s]\n", i, objective_value(model1), st)
    else
        @printf("  i=%d  [%s]  ← 문제!\n", i, st)
    end
end

println("\n--- Test 5: Compare without VI ---")
model3, vars3 = build_true_dro_subproblem(td, zeros(K);
    optimizer=Gurobi.Optimizer, silent=true, add_objF_vi=false)
optimize!(model3)
@printf("  at x=0: status=%s, Z₀=%.6f\n", termination_status(model3),
    termination_status(model3) == MOI.OPTIMAL ? objective_value(model3) : NaN)

# ===== Test 6: Inexact mode (OptimalityTarget=1) + NonConvex=2 =====
println("\n--- Test 6: inexact (OptimalityTarget=1) + NonConvex=2 with VI ---")
model4, vars4 = build_true_dro_subproblem(td, zeros(K);
    optimizer=Gurobi.Optimizer, silent=true, add_objF_vi=true)
set_optimizer_attribute(model4, "NonConvex", 2)
set_optimizer_attribute(model4, "OptimalityTarget", 1)
set_time_limit_sec(model4, 30.0)

for i in 1:5
    selected = shuffle(MersenneTwister(100 + i), interdictable_idx)[1:γ]
    x = zeros(K); x[selected] .= 1.0
    update_true_dro_subproblem_objective!(model4, vars4, td, x)
    optimize!(model4)
    st = termination_status(model4)
    if st in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.TIME_LIMIT) && has_values(model4)
        @printf("  i=%d  Z₀=%.4f  [%s]\n", i, objective_value(model4), st)
    else
        @printf("  i=%d  [%s]  ← 문제!\n", i, st)
    end
end

# ===== Test 7: OptimalityTarget=1 + NonConvex=2 WITHOUT VI =====
println("\n--- Test 7: inexact without VI (baseline reference) ---")
model5, vars5 = build_true_dro_subproblem(td, zeros(K);
    optimizer=Gurobi.Optimizer, silent=true, add_objF_vi=false)
set_optimizer_attribute(model5, "NonConvex", 2)
set_optimizer_attribute(model5, "OptimalityTarget", 1)
set_time_limit_sec(model5, 30.0)

for i in 1:5
    selected = shuffle(MersenneTwister(100 + i), interdictable_idx)[1:γ]
    x = zeros(K); x[selected] .= 1.0
    update_true_dro_subproblem_objective!(model5, vars5, td, x)
    optimize!(model5)
    st = termination_status(model5)
    if st in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.TIME_LIMIT) && has_values(model5)
        @printf("  i=%d  Z₀=%.4f  [%s]\n", i, objective_value(model5), st)
    else
        @printf("  i=%d  [%s]  ← 문제!\n", i, st)
    end
end
