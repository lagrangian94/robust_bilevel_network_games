using JuMP
using Gurobi
using HiGHS
using LinearAlgebra
using Infiltrator
# Load modules
using Revise
includet("network_generator.jl")
includet("build_uncertainty_set.jl")
include("build_full_model.jl")


using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios, print_network_summary
# using .BuildFullModel

println("="^80)
println("TESTING FULL 2DRNDP MODEL CONSTRUCTION")
println("="^80)

# Model parameters
S = 3  # Number of scenarios
ϕU = 10.0  # Upper bound on interdiction effectiveness
w = 1.0  # Budget weight
v = 1.0  # Interdiction effectiveness parameter (NOT the decision variable ν!)


# Generate a small test network
println("\n[1] Generating 3×3 grid network...")
network = generate_grid_network(3, 3, seed=42)
print_network_summary(network)
capacities, F, μ = generate_capacity_scenarios(length(network.arcs), S, seed=120)

# Build uncertainty set
# ===== BUILD ROBUST COUNTERPART MATRICES R AND r =====
println("\n" * "="^80)
println("BUILD ROBUST COUNTERPART MATRICES (R, r)")
println("="^80)

# Remove dummy arc from capacity scenarios (|A| = regular arcs only)
capacity_scenarios_regular = capacities[1:end-1, :]  # Remove last row (dummy arc)
epsilon = 0.005  # Robustness parameter

println("\n[3] Building R and r matrices...")
println("Number of regular arcs |A|: $(size(capacity_scenarios_regular, 1))")
println("Number of scenarios S: $S")
println("Robustness parameter ε: $epsilon")

R, r_dict = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)


println("\n[2] Building model...")
println("  Parameters:")
println("    S (scenarios) = $S")
println("    ϕU (interdiction bound) = $ϕU")
println("    w (budget weight) = $w")
println("    v (interdiction effectiveness param) = $v")
println("  Note: v is a parameter in COP matrix [Φ - v*W]")
println("        ν (nu) is a decision variable in objective t + w*ν")
# Build model (without optimizer for initial testing)
model, vars = build_full_2DRNDP_model(network, S, ϕU, w, v, R, r_dict)

println("\n[3] Model structure verification:")
println("  Scalar variables:")
println("    t: $(vars[:t])")
println("    nu: $(vars[:nu])")
println("    λ: $(vars[:λ])")

println("\n  Vector variables:")
println("    x: $(length(vars[:x])) binary variables")
println("    h: $(length(vars[:h])) continuous variables")
println("    ψ0: $(length(vars[:ψ0])) auxiliary variables")

println("\n  Scenario-indexed scalars:")
println("    ηhat: $(length(vars[:ηhat])) variables (S scenarios)")
println("    ηtilde: $(length(vars[:ηtilde])) variables (S scenarios)")

println("\n  Matrix variables per scenario:")
num_arcs = length(network.arcs)-1
num_nodes = length(network.nodes)
println("    Φhat: $(S) × $(num_arcs) × $(num_arcs) = $(S * num_arcs * num_arcs) variables")
println("    Ψhat: $(S) × $(num_arcs) × $(num_arcs) = $(S * num_arcs * num_arcs) variables")
println("    Φtilde: $(S) × $(num_arcs) × $(num_arcs) = $(S * num_arcs * num_arcs) variables")
println("    Ψtilde: $(S) × $(num_arcs) × $(num_arcs) = $(S * num_arcs * num_arcs) variables")
println("    Πhat: $(S) × $(num_nodes-1) × $(num_arcs) = $(S * (num_nodes-1) * num_arcs) variables")
println("    Πtilde: $(S) × $(num_nodes-1) × $(num_arcs) = $(S * (num_nodes-1) * num_arcs) variables")
println("    Y: $(S) × $(num_arcs) × $(num_arcs) = $(S * num_arcs * num_arcs) variables")
println("    Yts: $(S) × $(num_arcs) = $(S * num_arcs) variables")

println("\n[4] Checking specific constraints:")

# Check constraint 14b (resource budget)
budget_con = constraint_by_name(model, "resource_budget")
println("  (14b) Resource budget constraint: ", budget_con !== nothing ? "✓" : "✗")

# Check constraint 14c (total cost)
cost_con = constraint_by_name(model, "total_cost")
println("  (14c) Total cost constraint: ", cost_con !== nothing ? "✓" : "✗")


# Check budget constraint for dual variables (14l)
println("  (14l) Dual budget constraints: $(num_arcs) constraints (one per arc)")

# Check linearization constraints (14q)
println("  (14q) Linearization constraints: $(4 * num_arcs) constraints")

println("\n[5] Model statistics:")
println("  Total variables: $(num_variables(model))")
println("  Binary variables: $(sum(is_binary(vars[:x][i]) for i in 1:num_arcs))")
println("  Total constraints: $(sum(num_constraints(model, F, S) for (F, S) in list_of_constraint_types(model)))")

println("\n" * "="^80)
println("MODEL CONSTRUCTION TEST COMPLETE")
println("="^80)

println("\n⚠️  IMPORTANT NOTES:")
println("  1. COP constraints (14f, 14i) are NOT implemented yet")
println("  2. Dual constraints (14m-14p) are INCOMPLETE")
println("  3. Problem-specific data (d0, R matrix) need to be added")
println("  4. Model is NOT ready to solve - this is a structure test only")


# Set Gurobi
println("[4] Setting Gurobi solver...")
set_optimizer(model, Gurobi.Optimizer)
set_optimizer_attribute(model, "TimeLimit", 3600)
set_optimizer_attribute(model, "MIPGap", 1e-4)
@constraint(model, vars[:t]>=0) #COP 제약 넣고 이거 빼야함.
@infiltrate
optimize!(model)