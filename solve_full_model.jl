using JuMP
using Gurobi
using HiGHS
using LinearAlgebra
using Infiltrator
# Load modules
using Revise
includet("network_generator.jl")
includet("sdp_build_uncertainty_set.jl")
include("sdp_build_full_model.jl")


using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios, print_network_summary

println("="^80)
println("TESTING FULL 2DRNDP MODEL CONSTRUCTION")
println("="^80)

# Model parameters
S = 3  # Number of scenarios
ϕU = 100.0  # Upper bound on interdiction effectiveness
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
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :epsilon => epsilon)

println("\n[2] Building model...")
println("  Parameters:")
println("    S (scenarios) = $S")
println("    ϕU (interdiction bound) = $ϕU")
println("    w (budget weight) = $w")
println("    v (interdiction effectiveness param) = $v")
println("  Note: v is a parameter in COP matrix [Φ - v*W]")
println("        ν (nu) is a decision variable in objective t + w*ν")
# Build model (without optimizer for initial testing)
model, vars = build_full_2DRNDP_model(network, S, ϕU, w, v,uncertainty_set)

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
# set_optimizer_attribute(model, "TimeLimit", 3600)
# set_optimizer_attribute(model, "MIPGap", 1e-4)
@constraint(model, vars[:t]>=0) #COP 제약 넣고 이거 빼야함.
optimize!(model)
# Save objective value and optimal solution after solving

# Check if the model has a feasible solution
t_status = termination_status(model)
p_status = primal_status(model)

if t_status == MOI.OPTIMAL || t_status == MOI.FEASIBLE_POINT
    obj_value = objective_value(model)
    println("\nOptimal objective value: ", obj_value)

    # Save solution for selected variables
    sol = Dict()
    sol[:objective_value] = obj_value

    # Extract primal variables (add more if needed)
    sol[:x] = value.(vars[:x])
    sol[:h] = value.(vars[:h])
    sol[:nu] = value(vars[:nu])
    sol[:λ] = value(vars[:λ])
    sol[:t] = value(vars[:t])

    # Save scenario values (examples)
    sol[:ηhat] = value.(vars[:ηhat])
    sol[:ηtilde] = value.(vars[:ηtilde])

    # Optionally save any other variables you want (e.g., dual variables)
    # ...

    # Save to .jld2 file (assumes JLD2 is loaded elsewhere if not, replace with Serialization or plain text)
    try
        @info "Saving solution to 'full_model_solution.jld2'..."
        using JLD2
        JLD2.save("full_model_solution.jld2", "sol", sol)
    catch err
        @warn "Could not save using JLD2, writing to JSON instead."
        try
            using JSON
            open("full_model_solution.json", "w") do io
                JSON.print(io, sol)
            end
        catch err2
            @error "Could not save solution to file: $err2"
        end
    end

else
    println("\nNo feasible/optimal solution found. Status: $t_status")
end

