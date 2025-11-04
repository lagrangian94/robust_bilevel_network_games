using JuMP
using HiGHS  # Fast open-source LP solver
using Random
using Distributions
using LinearAlgebra
using Statistics

# Use Revise for automatic code reloading (if available)
# If Revise is not installed, run: using Pkg; Pkg.add("Revise")
# Note: Revise automatically tracks files included with 'include()' when loaded
try
    using Revise
    # Revise will automatically track this file when it's included
    includet("network_generator.jl")
catch
    # If Revise is not available, fall back to regular include
    @warn "Revise.jl not available. Code changes will not be automatically reloaded."
    @warn "Install Revise with: using Pkg; Pkg.add(\"Revise\")"
    include("network_generator.jl")
end

using .NetworkGenerator

println("="^80)
println("MAXIMUM FLOW PROBLEM - Debugging Node-Incidence Matrix")
println("="^80)

# Generate a small test network
println("\n[1] Generating 3×3 grid network...")
network = generate_grid_network(3, 3, seed=120)
print_network_summary(network)

# Generate capacity for one scenario
println("\n[2] Generating capacity for a single scenario...")
num_arcs = length(network.arcs)
capacities, F, μ = generate_capacity_scenarios(num_arcs, 1, seed=120)
C = vec(capacities[:, 1])  # Single scenario capacity vector

# Display capacities
println("\nCapacity for each arc:")
for i in 1:length(network.arcs)
    arc = network.arcs[i]
    cap = C[i]
    println("  Arc $i: $arc -> Capacity = $(round(cap, digits=2))")
end

# Identify dummy arc index
dummy_arc_idx = findfirst(arc -> arc == ("t", "s"), network.arcs)
println("\nDummy arc (t,s) is at index: $dummy_arc_idx")

# Set up the maximum flow problem following Lei & Song formulation
println("\n[3] Setting up Maximum Flow Problem...")
println("    Formulation from Lei & Song (2018):")
println("    max  y_ts  (dummy arc flow)")
println("    s.t. N*y = 0  (flow conservation)")
println("         y ≤ C  (capacity constraints)")
println("         y ≥ 0  (non-negativity)")

model = Model(HiGHS.Optimizer)
set_silent(model)

# Decision variables: flow on each arc
@variable(model, y[1:num_arcs] >= 0)

# Objective: maximize dummy arc flow y_ts
@objective(model, Max, y[dummy_arc_idx])

# Flow conservation constraints: N*y = 0
# N is (|V|-1) × |A| matrix (source row removed)
@constraint(model, flow_conservation, network.N * y .== 0)

# Capacity constraint: 오직 regular arcs만
regular_arcs = [i for i in 1:num_arcs if i != dummy_arc_idx]
# @constraint(model, capacity_bounds[i=regular_arcs], y[i] <= C[i])
@constraint(model, capacity_bounds, y .<= C)
# println("capacity_bounds: ", capacity_bounds)
# Solve the problem
println("\n[4] Solving the maximum flow problem...")
optimize!(model)

# Check solution status
status = termination_status(model)
println("\nSolution Status: $status")

if status == MOI.OPTIMAL
    println("✓ Optimal solution found!")
    
    # Get optimal values
    y_opt = value.(y)
    max_flow = objective_value(model)
    
    println("\n[5] Results:")
    println("    Maximum Flow Value: $(round(max_flow, digits=4))")
    
    # Display non-zero flows
    println("\n    Non-zero flows:")
    for i in 1:num_arcs
        if abs(y_opt[i]) > 1e-6
            arc = network.arcs[i]
            is_dummy = (i == dummy_arc_idx) ? " (DUMMY ARC)" : ""
            println("      Arc $i: $arc -> Flow = $(round(y_opt[i], digits=4))$is_dummy")
        end
    end
    
    # Verify flow conservation at each node
    println("\n[6] Verification - Flow Conservation Check:")
    println("    Checking N*y = 0 (should be all zeros)...")
    residual = network.N * y_opt
    max_residual = maximum(abs.(residual))
    println("    Maximum absolute residual: $(round(max_residual, digits=10))")
    
    if max_residual < 1e-6
        println("    ✓ Flow conservation satisfied!")
    else
        println("    ✗ Flow conservation VIOLATED!")
        println("    Residuals:")
        for i in 1:length(residual)
            if abs(residual[i]) > 1e-6
                # Map back to node (note: source is removed, so index i corresponds to node i+1)
                node = network.nodes[i+1]  # +1 because source is at index 1
                println("      Node $node: residual = $(residual[i])")
            end
        end
    end
    
    # Verify capacity constraints
    println("\n[7] Verification - Capacity Constraint Check:")
    println("    Checking y ≤ C...")
    violations = findall(y_opt .> C .+ 1e-6)
    if isempty(violations)
        println("    ✓ All capacity constraints satisfied!")
    else
        println("    ✗ Capacity constraints violated at arcs: $violations")
        for i in violations
            println("      Arc $i: flow = $(y_opt[i]), capacity = $(C[i])")
        end
    end
    
    # Additional verification: Check flow balance manually at each node
    println("\n[8] Manual Flow Balance Check:")
    println("    Computing incoming - outgoing flow at each node...")
    
    # Create a dictionary to store net flow at each node
    node_balance = Dict(node => 0.0 for node in network.nodes)
    
    for (i, (from_node, to_node)) in enumerate(network.arcs)
        flow = y_opt[i]
        node_balance[from_node] -= flow  # Outgoing flow
        node_balance[to_node] += flow    # Incoming flow
    end
    
    println("\n    Net flow balance at each node (should be 0 for all):")
    for node in network.nodes
        balance = node_balance[node]
        status_str = abs(balance) < 1e-6 ? "✓" : "✗"
        println("      $status_str Node $node: $(round(balance, digits=10))")
    end
    
else
    println("✗ Optimization failed or infeasible!")
    println("   Status: $status")
end

println("\n" * "="^80)
println("Testing completed!")
println("="^80)