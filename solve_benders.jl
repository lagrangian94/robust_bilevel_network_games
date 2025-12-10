using JuMP
using Gurobi
using HiGHS
using LinearAlgebra
using Infiltrator
# Load modules
using Revise
includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("strict_benders.jl")
includet("nested_benders.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_factor_model, generate_capacity_scenarios_uniform_model, print_network_summary

println("="^80)
println("TESTING STRICT BENDERS MODEL CONSTRUCTION")
println("="^80)

# Model parameters
S = 2  # Number of scenarios
ϕU = 10.0  # Upper bound on interdiction effectiveness
λU = 10.0  # Upper bound on λ
γ = 2.0  # Interdiction budget
w = 1.0  # Budget weight
v = 1.0  # Interdiction effectiveness parameter (NOT the decision variable ν!)


# Generate a small test network
println("\n[1] Generating 3×3 grid network...")
network = generate_grid_network(3, 3, seed=42)
print_network_summary(network)
# ===== Use Factor Model =====
# capacities, F = generate_capacity_scenarios(length(network.arcs), network.interdictable_arcs, S, seed=120)
# ===== Use Uniform Model =====
capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=42)

# Build uncertainty set
# ===== BUILD ROBUST COUNTERPART MATRICES R AND r =====
println("\n" * "="^80)
println("BUILD ROBUST COUNTERPART MATRICES (R, r)")
println("="^80)

# Remove dummy arc from capacity scenarios (|A| = regular arcs only)
capacity_scenarios_regular = capacities[1:end-1, :]  # Remove last row (dummy arc)
epsilon = 0.5  # Robustness parameter

println("\n[3] Building R and r matrices...")
println("Number of regular arcs |A|: $(size(capacity_scenarios_regular, 1))")
println("Number of scenarios S: $S")
println("Robustness parameter ε: $epsilon")

R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)


println("\n[2] Building model...")
println("  Parameters:")
println("    S (scenarios) = $S")
println("    ϕU (interdiction bound) = $ϕU")
println("    γ (interdiction budget) = $γ")
println("    w (budget weight) = $w")
println("    v (interdiction effectiveness param) = $v")
println("  Note: v is a parameter in COP matrix [Φ - v*W]")
println("        ν (nu) is a decision variable in objective t + w*ν")
# Build model (without optimizer for initial testing)
model, vars = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer)

nested_benders = true

if nested_benders
    result = nested_benders_optimize!(model, vars, network, ϕU, λU, γ, w, uncertainty_set; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer)
else
    result = strict_benders_optimize!(model, vars, network, ϕU, λU, γ, w, uncertainty_set; optimizer=Gurobi.Optimizer)
end




