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
export GridNetworkData, generate_grid_network, generate_capacity_scenarios_factor_model, generate_capacity_scenarios_factor_clustered, generate_capacity_scenarios_factor_sparse, generate_capacity_scenarios_uniform_model, generate_capacity_scenarios_contaminated, generate_capacity_scenarios_trunc_normal, print_network_summary
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
    generate_capacity_scenarios_factor_model(num_arcs, num_scenarios; interdictable_arcs, seed)

Generate capacity scenarios using the factor model from Sadana & Delage (2022).

Factor model: c = F * ξ (interdictable arcs only)
- F ∈ R₊^{num_interdictable × k}, entries ~ Uniform{1,...,10}
- ξᵢ ~ Exponential(μᵢ), μ ~ Uniform(0,1), k=2 factors
- Non-interdictable arcs: max(interdictable capacities) — 스케일 통일, big-M 폭발 방지
- Dummy arc: sum of all regular arc capacities

# Arguments
- `num_arcs`: total arcs INCLUDING dummy arc
- `num_scenarios`: number of scenarios |K|
- `interdictable_arcs`: Bool vector (length ≥ num_arcs-1). If nothing, all regular arcs treated as interdictable (하위호환)
- `seed`: random seed (optional)

# Returns
- `capacity_scenarios`: |E| × |K| matrix
- `F`: factor loading matrix (num_interdictable × k)
"""
function generate_capacity_scenarios_factor_model(num_arcs::Int, num_scenarios::Int;
                                     interdictable_arcs::Union{Vector{Bool},Nothing}=nothing,
                                     seed::Union{Int,Nothing}=nothing,
                                     num_factors::Int=2)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    num_regular_arcs = num_arcs - 1

    # Interdictable arc indices (regular arcs only, excluding dummy)
    if isnothing(interdictable_arcs)
        # 기본: 모든 regular arcs가 interdictable (하위호환)
        intd_idx = collect(1:num_regular_arcs)
    else
        intd_idx = findall(interdictable_arcs[1:num_regular_arcs])
    end
    non_intd_idx = setdiff(1:num_regular_arcs, intd_idx)
    num_intd = length(intd_idx)

    # Factor model: c = F * ξ, k factors (Sadana & Delage 2022)
    k = num_factors
    F = rand(0:10, num_intd, k)
    μ = rand(k)

    capacity_scenarios = zeros(Float64, num_arcs, num_scenarios)

    # 1) Interdictable arcs: factor model
    for scenario in 1:num_scenarios
        ξ = [rand(Exponential(μ[i])) for i in 1:k]
        capacity_scenarios[intd_idx, scenario] = (F * ξ) ./ k
    end

    # 2) Non-interdictable arcs: max of interdictable capacities (스케일 통일)
    if !isempty(non_intd_idx) && !isempty(intd_idx)
        max_intd_cap = maximum(capacity_scenarios[intd_idx, :])
        capacity_scenarios[non_intd_idx, :] .= max_intd_cap
    end

    # 3) Dummy arc: sum of all regular arc capacities
    for scenario in 1:num_scenarios
        capacity_scenarios[end, scenario] = sum(capacity_scenarios[1:num_regular_arcs, scenario])
    end

    return capacity_scenarios, F
end

"""
    generate_capacity_scenarios_factor_clustered(num_arcs, num_scenarios; ...)

Factor model로 N_pool개 시나리오를 대량 생성한 뒤 k-means clustering으로
num_scenarios개 representative scenario (centroid)를 추출.

Factor model 구조 유지 + 시나리오 diversity 극대화.
"""
function generate_capacity_scenarios_factor_clustered(num_arcs::Int, num_scenarios::Int;
        interdictable_arcs::Union{Vector{Bool},Nothing}=nothing,
        seed::Union{Int,Nothing}=nothing,
        num_factors::Int=5,
        N_pool::Int=2000,
        max_kmeans_iter::Int=100)

    if !isnothing(seed)
        Random.seed!(seed)
    end

    num_regular_arcs = num_arcs - 1

    if isnothing(interdictable_arcs)
        intd_idx = collect(1:num_regular_arcs)
    else
        intd_idx = findall(interdictable_arcs[1:num_regular_arcs])
    end
    non_intd_idx = setdiff(1:num_regular_arcs, intd_idx)
    num_intd = length(intd_idx)

    k = num_factors
    F = rand(0:10, num_intd, k)
    μ = rand(k)

    # 1) 대량 생성: N_pool개 시나리오
    pool = zeros(Float64, num_intd, N_pool)
    for j in 1:N_pool
        ξ = [rand(Exponential(μ[i])) for i in 1:k]
        pool[:, j] = (F * ξ) ./ k
    end

    # 2) k-means clustering → num_scenarios개 centroid
    S = num_scenarios
    # 초기 centroid: 균등 간격으로 선택
    idx_init = round.(Int, range(1, N_pool; length=S))
    centroids = pool[:, idx_init]  # num_intd × S
    assignments = zeros(Int, N_pool)

    for iter in 1:max_kmeans_iter
        # Assign
        changed = false
        for j in 1:N_pool
            best_c = 1
            best_d = Inf
            for c in 1:S
                d = sum((pool[a, j] - centroids[a, c])^2 for a in 1:num_intd)
                if d < best_d
                    best_d = d
                    best_c = c
                end
            end
            if assignments[j] != best_c
                assignments[j] = best_c
                changed = true
            end
        end

        # Update centroids
        for c in 1:S
            members = findall(==(c), assignments)
            if !isempty(members)
                for a in 1:num_intd
                    centroids[a, c] = sum(pool[a, j] for j in members) / length(members)
                end
            end
        end

        !changed && break
    end

    # 3) Centroid를 capacity scenario로 변환
    capacity_scenarios = zeros(Float64, num_arcs, S)
    capacity_scenarios[intd_idx, :] = centroids

    # Non-interdictable arcs
    if !isempty(non_intd_idx) && !isempty(intd_idx)
        max_intd_cap = maximum(capacity_scenarios[intd_idx, :])
        capacity_scenarios[non_intd_idx, :] .= max_intd_cap
    end

    # Dummy arc
    for s in 1:S
        capacity_scenarios[end, s] = sum(capacity_scenarios[1:num_regular_arcs, s])
    end

    return capacity_scenarios, F
end

"""
    generate_capacity_scenarios_factor_sparse(num_arcs, num_scenarios; ...)

Additive factor model: c_e^s = max(ε, c_bar + (1/k) Σ_j F_{ej} ξ_j^s)
F_{ej} ~ Uniform(-a, a), ξ_j^s ~ Exp(1). Baseline c_bar에 factor perturbation 추가.
음수 F → capacity 감소 방향, 양수 F → 증가 방향 → arc별 ranking reversal 유도.
"""
function generate_capacity_scenarios_factor_sparse(num_arcs::Int, num_scenarios::Int;
                                     interdictable_arcs::Union{Vector{Bool},Nothing}=nothing,
                                     seed::Union{Int,Nothing}=nothing,
                                     num_factors::Int=5,
                                     c_bar::Float64=10.0,
                                     a::Float64=4.0,
                                     clip_eps::Float64=0.1)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    num_regular_arcs = num_arcs - 1

    if isnothing(interdictable_arcs)
        intd_idx = collect(1:num_regular_arcs)
    else
        intd_idx = findall(interdictable_arcs[1:num_regular_arcs])
    end
    non_intd_idx = setdiff(1:num_regular_arcs, intd_idx)
    num_intd = length(intd_idx)

    k = num_factors
    # F_{ej} ~ Uniform(-a, a)
    F = a * (2.0 * rand(num_intd, k) .- 1.0)
    # ξ_j ~ Exp(1)

    capacity_scenarios = zeros(Float64, num_arcs, num_scenarios)

    for scenario in 1:num_scenarios
        ξ = [rand(Exponential(1.0)) for _ in 1:k]
        for (idx, e) in enumerate(intd_idx)
            val = c_bar + sum(F[idx, j] * ξ[j] for j in 1:k) / k
            capacity_scenarios[e, scenario] = max(clip_eps, val)
        end
    end

    if !isempty(non_intd_idx) && !isempty(intd_idx)
        max_intd_cap = maximum(capacity_scenarios[intd_idx, :])
        capacity_scenarios[non_intd_idx, :] .= max_intd_cap
    end

    for scenario in 1:num_scenarios
        capacity_scenarios[end, scenario] = sum(capacity_scenarios[1:num_regular_arcs, scenario])
    end

    return capacity_scenarios, F
end

function generate_capacity_scenarios_uniform_model(num_arcs::Int, num_scenarios::Int;
    interdictable_arcs::Union{Vector{Bool},Nothing}=nothing,
    seed::Union{Int,Nothing}=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    num_regular_arcs = num_arcs - 1

    # Interdictable arc indices (regular arcs only, excluding dummy)
    if isnothing(interdictable_arcs)
        # 기본: 모든 regular arcs (하위호환)
        intd_idx = collect(1:num_regular_arcs)
    else
        intd_idx = findall(interdictable_arcs[1:num_regular_arcs])
    end
    non_intd_idx = setdiff(1:num_regular_arcs, intd_idx)

    # Interdictable arcs: Uniform(1,10) i.i.d.
    F = rand(1:10, length(intd_idx), num_scenarios)

    capacity_scenarios = zeros(Float64, num_arcs, num_scenarios)

    # 1) Interdictable arcs: random
    for scenario in 1:num_scenarios
        capacity_scenarios[intd_idx, scenario] = F[:, scenario]
    end

    # 2) Non-interdictable arcs: max of interdictable capacities (스케일 통일)
    if !isempty(non_intd_idx) && !isempty(intd_idx)
        max_intd_cap = maximum(capacity_scenarios[intd_idx, :])
        capacity_scenarios[non_intd_idx, :] .= max_intd_cap
    end

    for scenario in 1:num_scenarios

        # Dummy arc: sum of all regular arc capacities
        capacity_scenarios[end, scenario] = sum(capacity_scenarios[1:num_regular_arcs, scenario])
    end

    return capacity_scenarios, F
end

"""
    generate_capacity_scenarios_contaminated(num_arcs, num_scenarios, δ; seed=nothing)

Contaminated DGP: (1-δ)·Uniform{1,...,10} + δ·Uniform{0,1,2}.
각 (arc, scenario)마다 확률 δ로 low-capacity component에서 추출.
Dummy arc (마지막) = regular arcs 합산.
"""
function generate_capacity_scenarios_contaminated(num_arcs::Int, num_scenarios::Int, δ::Float64;
    seed::Union{Int,Nothing}=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    num_regular_arcs = num_arcs - 1
    F = zeros(Int, num_regular_arcs, num_scenarios)

    for j in 1:num_scenarios
        for i in 1:num_regular_arcs
            if rand() < δ
                F[i, j] = rand(0:2)    # contamination component
            else
                F[i, j] = rand(1:10)   # nominal component
            end
        end
    end

    capacity_scenarios = zeros(Float64, num_arcs, num_scenarios)
    for scenario in 1:num_scenarios
        capacity_scenarios[1:num_regular_arcs, scenario] = F[:, scenario]
        capacity_scenarios[end, scenario] = sum(capacity_scenarios[1:num_regular_arcs, scenario])
    end

    return capacity_scenarios, F
end

"""
    generate_capacity_scenarios_trunc_normal(num_arcs, num_scenarios, μ, σ; lb=0.0, ub=10.0, seed=nothing)

Truncated Normal(μ, σ) on [lb, ub]. Rejection sampling.
Dummy arc (마지막) = regular arcs 합산.
Returns Float64 matrix (정수 아님).
"""
function generate_capacity_scenarios_trunc_normal(num_arcs::Int, num_scenarios::Int,
    μ::Float64, σ::Float64; lb::Float64=0.0, ub::Float64=10.0,
    seed::Union{Int,Nothing}=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    num_regular_arcs = num_arcs - 1
    F = zeros(Float64, num_regular_arcs, num_scenarios)

    for j in 1:num_scenarios
        for i in 1:num_regular_arcs
            while true
                x = μ + σ * randn()
                if lb <= x <= ub
                    F[i, j] = x
                    break
                end
            end
        end
    end

    capacity_scenarios = zeros(Float64, num_arcs, num_scenarios)
    for scenario in 1:num_scenarios
        capacity_scenarios[1:num_regular_arcs, scenario] = F[:, scenario]
        capacity_scenarios[end, scenario] = sum(capacity_scenarios[1:num_regular_arcs, scenario])
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

    # Source/sink 연결 arcs: non-interdictable (Cormican et al. 1998, Sadana & Delage 2022)
    # Dummy arc: non-interdictable
    interdictable = Vector{Bool}(undef, num_arcs)
    for i in 1:num_regular_arcs
        from, to = arcs[i]
        interdictable[i] = !(from == "s" || to == "s" || from == "t" || to == "t")
    end
    interdictable[end] = false  # dummy arc

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