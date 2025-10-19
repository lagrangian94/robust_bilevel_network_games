using Random
using Distributions
using LinearAlgebra

"""
    GridNetworkData

Stores the complete data structure for a grid network instance.

# Fields
- `m::Int`: number of rows in the grid
- `n::Int`: number of columns in the grid  
- `nodes::Vector{String}`: list of all node names including source 's' and sink 't'
- `arcs::Vector{Tuple{String,String}}`: list of all arcs as (from, to) tuples
- `N::Matrix{Float64}`: node-incidence matrix (|V|-1 × |A|+1), first row (source) removed
- `interdictable_arcs::Vector{Bool}`: indicates which arcs can be interdicted
- `arc_directions::Vector{Int}`: stores random directions (+1 or -1) for within-column arcs
"""
struct GridNetworkData
    m::Int  # number of rows
    n::Int  # number of columns
    nodes::Vector{String}
    arcs::Vector{Tuple{String,String}}
    N::Matrix{Float64}
    interdictable_arcs::Vector{Bool}
    arc_directions::Vector{Int}
end

"""
    generate_grid_network(m::Int, n::Int; seed::Union{Int,Nothing}=nothing)

Generate a grid network with m rows and n columns following Sadana & Delage (2022).

# Network Structure
- Source node 's' on the left
- Sink node 't' on the right
- Internal nodes arranged in m×n grid
- Within same column: arcs can point upward or downward with equal probability
- Between columns: arcs always point toward the sink (rightward)
- Arcs in first/last columns and arcs from source/to sink: NOT interdictable, infinite capacity

# Arguments
- `m::Int`: number of rows
- `n::Int`: number of columns
- `seed::Union{Int,Nothing}`: random seed for reproducibility (optional)

# Returns
- `GridNetworkData`: complete network data structure
"""
function generate_grid_network(m::Int, n::Int; seed::Union{Int,Nothing}=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    # Initialize nodes
    nodes = Vector{String}()
    push!(nodes, "s")  # source
    
    # Internal nodes: node_{row}_{col}
    for col in 1:n
        for row in 1:m
            push!(nodes, "node_$(row)_$(col)")
        end
    end
    
    push!(nodes, "t")  # sink
    
    num_nodes = length(nodes)
    
    # Generate arcs
    arcs = Vector{Tuple{String,String}}()
    interdictable = Vector{Bool}()
    arc_directions = Vector{Int}()  # Store random directions for within-column arcs
    
    # Helper function to get node index
    function get_node_idx(row::Int, col::Int)
        return 1 + (col-1)*m + row  # +1 because source is at index 1
    end
    
    # 1. Arcs from source to first column (NOT interdictable)
    for row in 1:m
        push!(arcs, ("s", "node_$(row)_1"))
        push!(interdictable, false)
    end
    
    # 2. Arcs within each column (upward or downward, INTERDICTABLE except first and last columns)
    for col in 1:n
        for row in 1:m-1
            # Randomly choose direction: +1 (downward) or -1 (upward)
            direction = rand([-1, 1])
            push!(arc_directions, direction)
            
            from_node = "node_$(row)_$(col)"
            to_node = "node_$(row + direction)_$(col)"
            
            # Ensure valid row index
            if 1 <= row + direction <= m
                push!(arcs, (from_node, to_node))
                # First and last columns are NOT interdictable
                push!(interdictable, !(col == 1 || col == n))
            end
        end
    end
    
    # 3. Arcs between columns (always toward sink, INTERDICTABLE except last column)
    for col in 1:n-1
        for row in 1:m
            from_node = "node_$(row)_$(col)"
            to_node = "node_$(row)_$(col+1)"
            push!(arcs, (from_node, to_node))
            # Last column arcs are NOT interdictable
            push!(interdictable, col < n-1)
        end
    end
    
    # 4. Arcs from last column to sink (NOT interdictable)
    for row in 1:m
        push!(arcs, ("node_$(row)_$(n)", "t"))
        push!(interdictable, false)
    end
    
    # 5. Add dummy arc from sink to source (for flow conservation)
    push!(arcs, ("t", "s"))
    push!(interdictable, false)
    
    num_arcs = length(arcs)
    
    # Build node-incidence matrix N
    # N has dimensions (|V|-1) × (|A|+1) where we remove the source node row
    # Columns: regular arcs + dummy arc
    N = zeros(Float64, num_nodes - 1, num_arcs)
    
    node_to_idx = Dict(node => idx for (idx, node) in enumerate(nodes))
    
    for (arc_idx, (from_node, to_node)) in enumerate(arcs)
        from_idx = node_to_idx[from_node]
        to_idx = node_to_idx[to_node]
        
        # Skip source node (row 1) - we remove it from the matrix
        if from_idx > 1  # not source
            N[from_idx - 1, arc_idx] = 1.0  # flow leaving
        end
        
        if to_idx > 1  # not source
            N[to_idx - 1, arc_idx] = -1.0  # flow entering
        end
    end
    
    return GridNetworkData(m, n, nodes, arcs, N, interdictable, arc_directions)
end

"""
    generate_capacity_scenarios(num_arcs::Int, num_scenarios::Int; seed::Union{Int,Nothing}=nothing)

Generate capacity scenarios using the factor model from Sadana & Delage (2022).

The factor model is: c = F * ξ, where
- F ∈ R^{|E| × 2}_+ is a randomly generated factor loading matrix
- ξ_i ~ Exponential(μ_i) for i=1,2 are independent factor realizations
- μ ∈ R^2_+ is randomly generated mean vector

# Arguments
- `num_arcs::Int`: number of arcs in the network (|E|)
- `num_scenarios::Int`: number of scenarios to generate (|K|)
- `seed::Union{Int,Nothing}`: random seed for reproducibility (optional)

# Returns
- `capacity_scenarios::Matrix{Float64}`: matrix of size |E| × |K| where each column is a scenario
- `F::Matrix{Float64}`: factor loading matrix (for reference)
- `μ::Vector{Float64}`: mean vector (for reference)
"""
function generate_capacity_scenarios(num_arcs::Int, num_scenarios::Int; 
                                     seed::Union{Int,Nothing}=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    # Randomly generate factor loading matrix F ∈ R^{|E| × 2}_+
    # Using uniform distribution on [0, 10] as in typical OR literature
    F = 10.0 * rand(num_arcs, 2)
    
    # Randomly generate mean vector μ ∈ R^2_+
    μ = 5.0 * rand(2)
    
    # Generate scenarios: c^k = F * ξ^k for k = 1, ..., |K|
    capacity_scenarios = zeros(Float64, num_arcs, num_scenarios)
    
    for k in 1:num_scenarios
        # Generate independent ξ_i ~ Exponential(μ_i) for i=1,2
        ξ = [rand(Exponential(μ[i])) for i in 1:2]
        
        # Compute capacity vector c^k = F * ξ
        capacity_scenarios[:, k] = F * ξ
    end
    
    return capacity_scenarios, F, μ
end

"""
    print_network_summary(network::GridNetworkData)

Print a summary of the network structure.
"""
function print_network_summary(network::GridNetworkData)
    println("=" ^ 60)
    println("Grid Network Summary")
    println("=" ^ 60)
    println("Grid size: $(network.m) rows × $(network.n) columns")
    println("Number of nodes: $(length(network.nodes))")
    println("Number of arcs: $(length(network.arcs))")
    println("Number of interdictable arcs: $(sum(network.interdictable_arcs))")
    println("Node-incidence matrix dimensions: $(size(network.N))")
    println("\nNodes: $(network.nodes)")
    println("\nFirst 10 arcs:")
    for i in 1:min(10, length(network.arcs))
        interdictable_str = network.interdictable_arcs[i] ? "✓" : "✗"
        println("  Arc $i: $(network.arcs[i]) [Interdictable: $interdictable_str]")
    end
    println("=" ^ 60)
end

# Example usage
println("Generating 5×5 grid network...")
network = generate_grid_network(5, 5, seed=42)
print_network_summary(network)

println("\nGenerating capacity scenarios...")
num_arcs = length(network.arcs)
num_scenarios = 100
capacities, F, μ = generate_capacity_scenarios(num_arcs, num_scenarios, seed=42)

println("\nCapacity Scenarios Summary:")
println("  Number of scenarios: $num_scenarios")
println("  Factor loading matrix F dimensions: $(size(F))")
println("  Mean vector μ: $μ")
println("  Capacity range: [$(minimum(capacities)), $(maximum(capacities))]")
println("  Average capacity: $(mean(capacities))")
println("\nFirst scenario capacities (first 10 arcs):")
println(capacities[1:min(10, num_arcs), 1])