# ==============================================================================
# Network Size Comparison (dummy arc 제외)
# ------------------------------------------------------------------------------
#  Network          |  Nodes  |  Arcs   |  Note
# ------------------------------------------------------------------------------
#  Grid 3×3         |    11   |    17   |  seed=42
#  Grid 4×4         |    18   |    31   |  seed=42
#  Grid 5×5         |    27   |    47   |  seed=42
#  Sioux-Falls      |    24   |    76   |  LeBlanc et al. (1975)
#  NOBEL-US         |    14   |    38   |  SNDlib, Orlowski et al. (2010)
#  ABILENE          |    12   |    30   |  SNDlib, Internet2 backbone
#  POLSKA           |    12   |    36   |  SNDlib, Polish telecom
# ==============================================================================

module NetworkGenerator

using Random
using Distributions
using LinearAlgebra
using Statistics
using Revise
export GridNetworkData, generate_grid_network, generate_capacity_scenarios_factor_model, generate_capacity_scenarios_uniform_model, print_network_summary
export RealWorldNetworkData, generate_sioux_falls_network, generate_nobel_us_network, generate_abilene_network, generate_polska_network, print_realworld_network_summary
using Infiltrator
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
- `arc_adjacency::Matrix{Bool}`: (|A| × |A|) arc i and arc j are adjacent if they share a node
- `node_arc_incidence::Matrix{Bool}`: ((|V|-1) × |A|) node i is incident to arc j
"""
struct GridNetworkData
    m::Int  # number of rows
    n::Int  # number of columns
    nodes::Vector{String}
    arcs::Vector{Tuple{String,String}}
    N::Matrix{Float64}
    interdictable_arcs::Vector{Bool}
    arc_directions::Vector{Int}
    arc_adjacency::Matrix{Bool}
    node_arc_incidence::Matrix{Bool}
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


function generate_arc_adjacency(arcs::Vector{Tuple{String,String}}, num_arcs::Int)
    """
    # Generate arc adjacency matrix where arc_adjacency[i,j] = true if arcs i and j share a common node.

    # Arguments
    - `arcs::Vector{Tuple{String,String}}`: list of all arcs (including dummy arc)
    - `num_arcs::Int`: number of regular arcs (excluding dummy arc)

    # Returns
    - `arc_adjacency::Matrix{Bool}`: num_arcs × num_arcs adjacency matrix
    """
    arc_adjacency = falses(num_arcs, num_arcs)
    
    for i in 1:num_arcs, j in 1:num_arcs
        arc_i = arcs[i]
        arc_j = arcs[j]
        
        # Arcs are adjacent if they share any common node
        if (arc_i[1] == arc_j[1] || arc_i[1] == arc_j[2] ||
            arc_i[2] == arc_j[1] || arc_i[2] == arc_j[2])
            arc_adjacency[i,j] = true
        end
    end
    
    return arc_adjacency
end

function generate_node_arc_incidence(nodes::Vector{String}, 
                                     arcs::Vector{Tuple{String,String}},
                                     num_nodes::Int, num_arcs::Int)
    """
    Generate node-arc incidence matrix where node_arc_incidence[i,j] = true if node i is 
    incident to arc j (i.e., arc j has node i as either tail or head).

    # Arguments
    - `nodes::Vector{String}`: list of all nodes (including source)
    - `arcs::Vector{Tuple{String,String}}`: list of all arcs (including dummy arc)
    - `num_nodes::Int`: number of nodes excluding source
    - `num_arcs::Int`: number of regular arcs (excluding dummy arc)

    # Returns
    - `node_arc_incidence::Matrix{Bool}`: num_nodes × num_arcs incidence matrix
    """
    node_arc_incidence = falses(num_nodes, num_arcs)
    
    for j in 1:num_arcs
        arc = arcs[j]
        
        for i in 1:num_nodes
            node_name = nodes[i+1]  # Skip source node (index 1)
            if arc[1] == node_name || arc[2] == node_name
                node_arc_incidence[i,j] = true
            end
        end
    end
    
    return node_arc_incidence
end


function generate_grid_network(m::Int, n::Int; seed::Union{Int,Nothing}=nothing)
    """
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
    num_regular_arcs = num_arcs - 1  # Exclude dummy arc
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
    # Generate adjacency structures
    arc_adjacency = generate_arc_adjacency(arcs, num_regular_arcs)
    node_arc_incidence = generate_node_arc_incidence(nodes, arcs, 
                                                      num_nodes - 1, num_regular_arcs)
    return GridNetworkData(m, n, nodes, arcs, N, interdictable, arc_directions, arc_adjacency, node_arc_incidence)
end
"""
    generate_capacity_scenarios(num_arcs::Int, num_scenarios::Int; seed::Union{Int,Nothing}=nothing)

Generate capacity scenarios using the factor model from Sadana & Delage (2022).

*** CORRECTED VERSION ***
The factor loading matrix F is generated ONLY for regular arcs (excluding dummy arc).

The factor model is: c = F * ξ, where
- F ∈ R^{|A| × 2}_+ is a randomly generated factor loading matrix (REGULAR ARCS ONLY)
- ξ_i ~ Exponential(μ_i) for i=1,2 are independent factor realizations
- μ ∈ R^2_+ is randomly generated mean vector

# Arguments
- `num_arcs::Int`: number of arcs in the network (|E|) INCLUDING dummy arc
- `num_scenarios::Int`: number of scenarios to generate (|K|)
- `seed::Union{Int,Nothing}`: random seed for reproducibility (optional)

# Returns
- `capacity_scenarios::Matrix{Float64}`: matrix of size |E| × |K| where each column is a scenario
- `F::Matrix{Float64}`: factor loading matrix (|A| × 2, for regular arcs only)
- `μ::Vector{Float64}`: mean vector (for reference)

# Fix Summary
- OLD (WRONG): F = rand(num_arcs, k)  # included dummy arc → dimension (|E|, 2)
- NEW (CORRECT): F = rand(num_arcs - 1, k)  # excludes dummy arc → dimension (|A|, 2)

This matches Sadana's MATLAB code: F = rand(E-size_set_non_rem, k)
"""
function generate_capacity_scenarios_factor_model(num_arcs::Int, num_scenarios::Int; 
                                     seed::Union{Int,Nothing}=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    

    
    # ============================================================
    # CRITICAL FIX: F matrix는 regular arcs (더미 arc 제외)에 대해서만 생성
    # ============================================================
    # Number of REGULAR arcs (excluding dummy arc)
    num_regular_arcs = num_arcs - 1
    
    # Randomly generate factor loading matrix F from Uniform(0,1)
    # F ∈ R^{|A| × 2}_+ where |A| = num_regular_arcs
    # F = rand(num_regular_arcs, k)
    # Number of factors
    k = 1
    F = rand(1:10, num_regular_arcs, k)
    
    # Randomly generate mean vector μ from Uniform(0,1)
    μ = rand(k)
    
    # Generate scenarios: c^k = F * ξ^k for k = 1, ..., |K|
    capacity_scenarios = zeros(Float64, num_arcs, num_scenarios)
    
    for scenario in 1:num_scenarios
        # Generate independent ξ_i ~ Exponential(μ_i) for i=1,2
        ξ = [rand(Exponential(μ[i])) for i in 1:k]
        # ξ = ones(k)
        
        # Apply factor model ONLY to regular arcs
        scenarios = F * ξ
        # scenarios = scenarios ./mean(scenarios, dims=1)
        capacity_scenarios[1:num_regular_arcs, scenario] = scenarios

        # 만약 scenarios가 행렬이고, 각 column별 평균을 구하려면:
        # col_means = mean(scenarios, dims=1)  # 결과는 1행 N열 (여러 시나리오의 열 평균)
        
        # Set dummy arc (last arc) capacity to sum of all regular arc capacities
        # This is large enough but numerically stable
        regular_arcs_capacity_sum = sum(capacity_scenarios[1:num_regular_arcs, scenario])
        capacity_scenarios[end, scenario] = regular_arcs_capacity_sum
    end
    
    return capacity_scenarios, F
end

function generate_capacity_scenarios_uniform_model(num_arcs::Int, num_scenarios::Int; 
    seed::Union{Int,Nothing}=nothing)
    if !isnothing(seed)
    Random.seed!(seed)
    end



    # ============================================================
    # CRITICAL FIX: F matrix는 regular arcs (더미 arc 제외)에 대해서만 생성
    # ============================================================
    # Number of REGULAR arcs (excluding dummy arc)
    num_regular_arcs = num_arcs - 1

    # Randomly generate factor loading matrix F from Uniform(0,1)
    # F ∈ R^{|A| × 2}_+ where |A| = num_regular_arcs
    # F = rand(num_regular_arcs, k)
    
    F = rand(1:10, num_regular_arcs, num_scenarios)

    # Generate scenarios: c^k = F * ξ^k for k = 1, ..., |K|
    capacity_scenarios = zeros(Float64, num_arcs, num_scenarios)

    for scenario in 1:num_scenarios
        capacity_scenarios[1:num_regular_arcs, scenario] = F[:, scenario]
        regular_arcs_capacity_sum = sum(capacity_scenarios[1:num_regular_arcs, scenario])
        capacity_scenarios[end, scenario] = regular_arcs_capacity_sum
    end

    return capacity_scenarios, F
end
"""
    print_network_summary(network::GridNetworkData)

Print a summary of the network structure.
"""
function print_network_summary(network::GridNetworkData)
    num_regular_arcs = length(network.arcs) - 1
    num_nodes_excl_source = length(network.nodes) - 1
    
    println("=" ^ 60)
    println("Grid Network Summary")
    println("=" ^ 60)
    println("Grid size: $(network.m) rows × $(network.n) columns")
    println("Number of nodes: $(length(network.nodes))")
    println("Number of arcs (incl. dummy): $(length(network.arcs))")
    println("Number of regular arcs: $num_regular_arcs")
    println("Number of interdictable arcs: $(sum(network.interdictable_arcs))")
    println("Node-incidence matrix dimensions: $(size(network.N))")
    
    # Sparsity statistics
    num_arc_pairs = num_regular_arcs * num_regular_arcs
    num_adjacent_pairs = sum(network.arc_adjacency)
    sparsity_arc = 100 * (1 - num_adjacent_pairs / num_arc_pairs)
    
    num_node_arc_pairs = num_nodes_excl_source * num_regular_arcs
    num_incident_pairs = sum(network.node_arc_incidence)
    sparsity_node = 100 * (1 - num_incident_pairs / num_node_arc_pairs)
    
    println("\nSparsity Information:")
    println("  Arc-arc adjacency: $num_adjacent_pairs / $num_arc_pairs pairs " *
            "($(round(sparsity_arc, digits=1))% sparse)")
    println("  Node-arc incidence: $num_incident_pairs / $num_node_arc_pairs pairs " *
            "($(round(sparsity_node, digits=1))% sparse)")
    
    println("\nNodes: $(network.nodes)")
    println("\nFirst 10 arcs:")
    for i in 1:min(10, length(network.arcs))
        interdictable_str = network.interdictable_arcs[i] ? "✓" : "✗"
        println("  Arc $i: $(network.arcs[i]) [Interdictable: $interdictable_str]")
    end
    println("=" ^ 60)
end

# ==============================================================================
# Real-World Network Generators
# ==============================================================================

"""
    RealWorldNetworkData

Stores the complete data structure for a real-world network instance.
Source/sink are remapped to "s"/"t" for solver compatibility with GridNetworkData.

# Fields
- `name::String`: network name
- `original_node_names::Dict{String,String}`: mapping from "s"/"t" back to original names
- `nodes::Vector{String}`: list of all node names (source="s", sink="t")
- `arcs::Vector{Tuple{String,String}}`: list of all arcs as (from, to) tuples
- `N::Matrix{Float64}`: node-incidence matrix (|V|-1 × |A|), source row removed
- `interdictable_arcs::Vector{Bool}`: all regular arcs are interdictable
- `arc_adjacency::Matrix{Bool}`: (|A| × |A|) arc adjacency matrix
- `node_arc_incidence::Matrix{Bool}`: ((|V|-1) × |A|) node-arc incidence matrix
"""
struct RealWorldNetworkData
    name::String
    original_node_names::Dict{String,String}
    nodes::Vector{String}
    arcs::Vector{Tuple{String,String}}
    N::Matrix{Float64}
    interdictable_arcs::Vector{Bool}
    arc_adjacency::Matrix{Bool}
    node_arc_incidence::Matrix{Bool}
end

"""
    _build_realworld_network(name, nodes, arcs, source, sink)

Internal helper to build RealWorldNetworkData from node/arc lists.
Remaps source→"s", sink→"t" for solver compatibility (dummy arc becomes ("t","s")).
All regular arcs are marked as interdictable.
"""
function _build_realworld_network(name::String, nodes::Vector{String},
                                   arcs::Vector{Tuple{String,String}},
                                   source::String, sink::String)
    # Store original names for display
    original_node_names = Dict("s" => source, "t" => sink)

    # Remap source→"s", sink→"t" in nodes
    nodes = [n == source ? "s" : (n == sink ? "t" : n) for n in nodes]

    # Remap source→"s", sink→"t" in arcs
    function remap(n)
        n == source ? "s" : (n == sink ? "t" : n)
    end
    arcs = [(remap(a), remap(b)) for (a, b) in arcs]

    # Add dummy arc (t, s) for flow conservation
    push!(arcs, ("t", "s"))

    # Sort arcs
    sort!(arcs, by = arc -> (arc[1], arc[2]))

    num_nodes = length(nodes)
    num_arcs = length(arcs)
    num_regular_arcs = num_arcs - 1

    # All regular arcs interdictable, dummy arc not
    interdictable = vcat(fill(true, num_regular_arcs), [false])

    # Build node-incidence matrix (source="s" is first node after remap)
    # Ensure "s" is the first node
    s_idx = findfirst(==("s"), nodes)
    if s_idx != 1
        deleteat!(nodes, s_idx)
        pushfirst!(nodes, "s")
    end

    N_full = zeros(Float64, num_nodes, num_arcs)
    node_to_idx = Dict(node => idx for (idx, node) in enumerate(nodes))

    for (arc_idx, (from_node, to_node)) in enumerate(arcs)
        from_idx = node_to_idx[from_node]
        to_idx = node_to_idx[to_node]
        N_full[from_idx, arc_idx] = 1.0
        N_full[to_idx, arc_idx] = -1.0
    end

    # Remove source row (index 1, since "s" is first)
    N = N_full[2:end, :]

    # Generate adjacency structures
    arc_adjacency = generate_arc_adjacency(arcs, num_regular_arcs)
    node_arc_incidence_mat = falses(num_nodes - 1, num_regular_arcs)
    for j in 1:num_regular_arcs
        arc = arcs[j]
        for i in 1:(num_nodes - 1)
            node_name = nodes[i + 1]  # skip source
            if arc[1] == node_name || arc[2] == node_name
                node_arc_incidence_mat[i, j] = true
            end
        end
    end

    return RealWorldNetworkData(name, original_node_names, nodes, arcs, N,
                                 interdictable, arc_adjacency, node_arc_incidence_mat)
end

"""
    generate_sioux_falls_network()

Generate the Sioux-Falls road network (LeBlanc et al., 1975).
24 nodes, 76 directed arcs. Source: node 1, Sink: node 24.
"""
function generate_sioux_falls_network()
    nodes = ["$i" for i in 1:24]
    source = "1"
    sink = "24"

    arcs = [
        ("1","2"), ("1","3"),
        ("2","1"), ("2","6"),
        ("3","1"), ("3","4"), ("3","12"),
        ("4","3"), ("4","5"), ("4","11"),
        ("5","4"), ("5","6"), ("5","9"),
        ("6","2"), ("6","5"), ("6","8"),
        ("7","8"), ("7","18"),
        ("8","6"), ("8","7"), ("8","9"), ("8","16"),
        ("9","5"), ("9","8"), ("9","10"),
        ("10","9"), ("10","11"), ("10","15"), ("10","16"), ("10","17"),
        ("11","4"), ("11","10"), ("11","12"), ("11","14"),
        ("12","3"), ("12","11"), ("12","13"),
        ("13","12"), ("13","24"),
        ("14","11"), ("14","15"), ("14","23"),
        ("15","10"), ("15","14"), ("15","19"), ("15","22"),
        ("16","8"), ("16","10"), ("16","17"), ("16","18"),
        ("17","10"), ("17","16"), ("17","19"),
        ("18","7"), ("18","16"), ("18","20"),
        ("19","15"), ("19","17"), ("19","20"),
        ("20","18"), ("20","19"), ("20","21"), ("20","22"),
        ("21","20"), ("21","22"), ("21","24"),
        ("22","15"), ("22","20"), ("22","21"), ("22","23"),
        ("23","14"), ("23","22"), ("23","24"),
        ("24","13"), ("24","21"), ("24","23")
    ]

    return _build_realworld_network("Sioux-Falls", nodes, arcs, source, sink)
end

"""
    generate_nobel_us_network()

Generate the NOBEL-US network from SNDlib (Orlowski et al., 2010).
14 nodes (major US cities), 42 directed arcs. Source: Seattle, Sink: Princeton.
"""
function generate_nobel_us_network()
    nodes = [
        "Seattle", "Palo_Alto", "San_Diego", "Salt_Lake_City", "Boulder",
        "Lincoln", "Urbana_Champaign", "Ann_Arbor", "Ithaca",
        "Princeton", "Atlanta", "Houston", "Washington", "New_York"
    ]
    source = "Seattle"
    sink = "Princeton"

    arcs = [
        ("Seattle","Palo_Alto"), ("Seattle","Salt_Lake_City"),
        ("Palo_Alto","Seattle"), ("Palo_Alto","San_Diego"), ("Palo_Alto","Salt_Lake_City"),
        ("San_Diego","Palo_Alto"), ("San_Diego","Houston"),
        ("Salt_Lake_City","Seattle"), ("Salt_Lake_City","Palo_Alto"), ("Salt_Lake_City","Boulder"),
        ("Boulder","Salt_Lake_City"), ("Boulder","Lincoln"),
        ("Lincoln","Boulder"), ("Lincoln","Urbana_Champaign"),
        ("Urbana_Champaign","Lincoln"), ("Urbana_Champaign","Ann_Arbor"), ("Urbana_Champaign","Atlanta"),
        ("Ann_Arbor","Urbana_Champaign"), ("Ann_Arbor","Ithaca"), ("Ann_Arbor","Princeton"),
        ("Ithaca","Ann_Arbor"), ("Ithaca","Princeton"), ("Ithaca","New_York"),
        ("Princeton","Ann_Arbor"), ("Princeton","Ithaca"), ("Princeton","Washington"), ("Princeton","New_York"),
        ("Atlanta","Urbana_Champaign"), ("Atlanta","Houston"), ("Atlanta","Washington"),
        ("Houston","San_Diego"), ("Houston","Atlanta"),
        ("Washington","Princeton"), ("Washington","Atlanta"), ("Washington","New_York"),
        ("New_York","Ithaca"), ("New_York","Princeton"), ("New_York","Washington")
    ]

    return _build_realworld_network("NOBEL-US", nodes, arcs, source, sink)
end

"""
    generate_abilene_network()

Generate the ABILENE network from SNDlib (Orlowski et al., 2010).
12 nodes, 30 directed arcs (15 bidirectional links). Source: STTLng, Sink: NYCMng.
"""
function generate_abilene_network()
    nodes = [
        "ATLAng", "ATLAM5", "CHINng", "DNVRng", "HSTNng", "IPLSng",
        "KSCYng", "LOSAng", "NYCMng", "SNVAng", "STTLng", "WASHng"
    ]
    source = "STTLng"
    sink = "NYCMng"

    arcs = [
        ("ATLAng","ATLAM5"), ("ATLAM5","ATLAng"),
        ("ATLAng","HSTNng"), ("HSTNng","ATLAng"),
        ("ATLAng","IPLSng"), ("IPLSng","ATLAng"),
        ("ATLAng","WASHng"), ("WASHng","ATLAng"),
        ("CHINng","IPLSng"), ("IPLSng","CHINng"),
        ("CHINng","NYCMng"), ("NYCMng","CHINng"),
        ("DNVRng","KSCYng"), ("KSCYng","DNVRng"),
        ("DNVRng","SNVAng"), ("SNVAng","DNVRng"),
        ("DNVRng","STTLng"), ("STTLng","DNVRng"),
        ("HSTNng","KSCYng"), ("KSCYng","HSTNng"),
        ("HSTNng","LOSAng"), ("LOSAng","HSTNng"),
        ("IPLSng","KSCYng"), ("KSCYng","IPLSng"),
        ("LOSAng","SNVAng"), ("SNVAng","LOSAng"),
        ("NYCMng","WASHng"), ("WASHng","NYCMng"),
        ("SNVAng","STTLng"), ("STTLng","SNVAng")
    ]

    return _build_realworld_network("ABILENE", nodes, arcs, source, sink)
end

"""
    generate_polska_network()

Generate the POLSKA network from SNDlib (Orlowski et al., 2010).
12 nodes (Polish cities), 36 directed arcs (18 bidirectional links). Source: Szczecin, Sink: Rzeszow.
"""
function generate_polska_network()
    nodes = [
        "Bialystok", "Bydgoszcz", "Gdansk", "Katowice", "Kolobrzeg",
        "Krakow", "Lodz", "Poznan", "Rzeszow", "Szczecin", "Warsaw", "Wroclaw"
    ]
    source = "Szczecin"
    sink = "Rzeszow"

    arcs = [
        ("Gdansk","Warsaw"), ("Warsaw","Gdansk"),
        ("Gdansk","Kolobrzeg"), ("Kolobrzeg","Gdansk"),
        ("Bydgoszcz","Kolobrzeg"), ("Kolobrzeg","Bydgoszcz"),
        ("Bydgoszcz","Poznan"), ("Poznan","Bydgoszcz"),
        ("Bydgoszcz","Warsaw"), ("Warsaw","Bydgoszcz"),
        ("Kolobrzeg","Szczecin"), ("Szczecin","Kolobrzeg"),
        ("Katowice","Krakow"), ("Krakow","Katowice"),
        ("Katowice","Lodz"), ("Lodz","Katowice"),
        ("Katowice","Wroclaw"), ("Wroclaw","Katowice"),
        ("Krakow","Rzeszow"), ("Rzeszow","Krakow"),
        ("Krakow","Warsaw"), ("Warsaw","Krakow"),
        ("Bialystok","Rzeszow"), ("Rzeszow","Bialystok"),
        ("Bialystok","Warsaw"), ("Warsaw","Bialystok"),
        ("Lodz","Warsaw"), ("Warsaw","Lodz"),
        ("Lodz","Wroclaw"), ("Wroclaw","Lodz"),
        ("Poznan","Szczecin"), ("Szczecin","Poznan"),
        ("Poznan","Wroclaw"), ("Wroclaw","Poznan"),
        ("Gdansk","Bialystok"), ("Bialystok","Gdansk")
    ]

    return _build_realworld_network("POLSKA", nodes, arcs, source, sink)
end

"""
    print_realworld_network_summary(network::RealWorldNetworkData)

Print a summary of the real-world network structure.
"""
function print_realworld_network_summary(network::RealWorldNetworkData)
    num_regular_arcs = length(network.arcs) - 1
    num_nodes_excl_source = length(network.nodes) - 1

    println("=" ^ 60)
    println("$(network.name) Network Summary")
    println("=" ^ 60)
    println("Number of nodes: $(length(network.nodes))")
    println("Number of arcs (incl. dummy): $(length(network.arcs))")
    println("Number of regular arcs: $num_regular_arcs")
    println("Number of interdictable arcs: $(sum(network.interdictable_arcs))")
    println("Source node: s ($(network.original_node_names["s"]))")
    println("Sink node: t ($(network.original_node_names["t"]))")
    println("Node-incidence matrix N dimensions: $(size(network.N))")

    # Sparsity statistics
    num_arc_pairs = num_regular_arcs * num_regular_arcs
    num_adjacent_pairs = sum(network.arc_adjacency)
    sparsity_arc = 100 * (1 - num_adjacent_pairs / num_arc_pairs)

    num_node_arc_pairs = num_nodes_excl_source * num_regular_arcs
    num_incident_pairs = sum(network.node_arc_incidence)
    sparsity_node = 100 * (1 - num_incident_pairs / num_node_arc_pairs)

    println("\nSparsity Information:")
    println("  Arc-arc adjacency: $num_adjacent_pairs / $num_arc_pairs pairs " *
            "($(round(sparsity_arc, digits=1))% sparse)")
    println("  Node-arc incidence: $num_incident_pairs / $num_node_arc_pairs pairs " *
            "($(round(sparsity_node, digits=1))% sparse)")

    println("\nFirst 15 arcs:")
    for i in 1:min(15, length(network.arcs))
        interdictable_str = network.interdictable_arcs[i] ? "✓" : "✗"
        println("  Arc $i: $(network.arcs[i]) [Interdictable: $interdictable_str]")
    end
    println("=" ^ 60)
end

end # module NetworkGenerator