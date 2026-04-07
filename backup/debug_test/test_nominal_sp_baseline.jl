"""
Nominal 2-SP baseline 확인.
Grid 3×3 S=2 + Polska S=2 에서 objective를 Full Model / Benders와 비교.

julia debug_test/test_nominal_sp_baseline.jl
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using Hypatia
using Pajarito
using LinearAlgebra
using Infiltrator
using Printf

const PROJECT_ROOT = joinpath(@__DIR__, "..")
include(joinpath(PROJECT_ROOT, "network_generator.jl"))
include(joinpath(PROJECT_ROOT, "build_uncertainty_set.jl"))
include(joinpath(PROJECT_ROOT, "build_nominal_sp.jl"))

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary
using .NetworkGenerator: generate_polska_network, print_realworld_network_summary

function run_nominal_sp(network, network_name; S=2, epsilon=0.5, seed=42)
    num_arcs = length(network.arcs) - 1

    ϕU = 1/epsilon
    λU = ϕU
    γ = ceil(Int, 0.10 * sum(network.interdictable_arcs[1:num_arcs]))

    capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(0.2 * γ * c_bar, digits=4)
    v = 1.0

    capacity_scenarios_regular = capacities[1:end-1, :]
    R, r_dict_hat, r_dict_tilde, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon, epsilon)
    uncertainty_set = Dict(
        :R => R, :r_dict_hat => r_dict_hat, :r_dict_tilde => r_dict_tilde,
        :xi_bar => xi_bar, :epsilon_hat => epsilon, :epsilon_tilde => epsilon)

    println("\n  ε=$epsilon, ϕU=$ϕU, γ=$γ, w=$w, v=$v, S=$S")

    model_sp, vars_sp = build_full_2SP_model(network, S, ϕU, λU, γ, w, v, uncertainty_set;
        optimizer=Gurobi.Optimizer)
    optimize!(model_sp)

    st = termination_status(model_sp)
    if st == MOI.OPTIMAL || st == MOI.ALMOST_OPTIMAL
        obj = objective_value(model_sp)
        println("  Status: $st")
        println("  Objective: $(round(obj, digits=6))")
        println("  x*: $(round.(value.(vars_sp[:x])))")
        println("  λ*: $(round(value(vars_sp[:λ]), digits=6))")
        return obj
    else
        println("  Status: $st (failed)")
        return NaN
    end
end

# ===== 1. Grid 3×3, S=2 =====
println("="^70)
println("1. GRID 3×3, S=2")
println("="^70)
net_grid = generate_grid_network(3, 3, seed=42)
print_network_summary(net_grid)
obj_grid = run_nominal_sp(net_grid, "grid_3x3"; S=2)

# ===== 2. Polska, S=2 =====
println("\n" * "="^70)
println("2. POLSKA, S=2")
println("="^70)
net_polska = generate_polska_network()
print_realworld_network_summary(net_polska)
obj_polska = run_nominal_sp(net_polska, "polska"; S=2)

# ===== Summary =====
println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println("┌──────────────┬──────────────┬──────────────┬──────────────┐")
println("│ Network      │  Nominal SP  │  Full Model  │   Benders    │")
println("├──────────────┼──────────────┼──────────────┼──────────────┤")
@printf("│ Grid 3×3     │ %11.4f  │    10.1820   │     5.3227   │\n", isnan(obj_grid) ? -999.0 : obj_grid)
@printf("│ Polska       │ %11.4f  │    -0.983*   │     1.017*   │\n", isnan(obj_polska) ? -999.0 : obj_polska)
println("└──────────────┴──────────────┴──────────────┴──────────────┘")
println("  * Polska 값은 full_model_ytsU_issue.md 기준 (ytsU=2.0)")
