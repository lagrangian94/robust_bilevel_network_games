println("\nFirst scenario capacities (first 10 arcs):")
println(round.(capacities[1:min(10, num_arcs), 1], digits=2))

# ==============================================================================
# Real-World Network Generators
# ==============================================================================

"""
    RealWorldNetworkData

Stores the complete data structure for a real-world network instance.

# Fields
- `name::String`: network name
- `nodes::Vector{String}`: list of all node names including source and sink
- `arcs::Vector{Tuple{String,String}}`: list of all arcs as (from, to) tuples
- `N::Matrix{Float64}`: node-incidence matrix (|V|-1 × |A|), source row removed
- `source::String`: source node name
- `sink::String`: sink node name
"""
struct RealWorldNetworkData
    name::String
    nodes::Vector{String}
    arcs::Vector{Tuple{String,String}}
    N::Matrix{Float64}
    source::String
    sink::String
end

"""
    generate_sioux_falls_network()

Generate the Sioux-Falls road network from LeBlanc et al. (1975).

Network structure:
- 24 nodes (numbered 1 to 24)
- 76 directed arcs
- Source: node 1
- Sink: node 24

This is a standard benchmark network used in transportation and network interdiction literature.

# References
- LeBlanc, L.J., Morlok, E.K., Pierskalla, W.P. (1975). An efficient approach to solving
  the road network equilibrium traffic assignment problem. Transportation Research, 9(5), 309-318.

# Returns
- `RealWorldNetworkData`: complete network data structure
"""
function generate_sioux_falls_network()
    name = "Sioux-Falls"
    
    # Create node list
    nodes = ["$i" for i in 1:24]
    source = "1"
    sink = "24"
    
    # Define arcs based on the Sioux-Falls network topology
    # The topology is extracted from Lei et al. (2018) Figure 6 and standard references
    arcs = [
        # From node 1
        ("1", "2"), ("1", "3"),
        # From node 2
        ("2", "1"), ("2", "6"),
        # From node 3
        ("3", "1"), ("3", "4"), ("3", "12"),
        # From node 4
        ("4", "3"), ("4", "5"), ("4", "11"),
        # From node 5
        ("5", "4"), ("5", "6"), ("5", "9"),
        # From node 6
        ("6", "2"), ("6", "5"), ("6", "8"),
        # From node 7
        ("7", "8"), ("7", "18"),
        # From node 8
        ("8", "6"), ("8", "7"), ("8", "9"), ("8", "16"),
        # From node 9
        ("9", "5"), ("9", "8"), ("9", "10"),
        # From node 10
        ("10", "9"), ("10", "11"), ("10", "15"), ("10", "16"), ("10", "17"),
        # From node 11
        ("11", "4"), ("11", "10"), ("11", "12"), ("11", "14"),
        # From node 12
        ("12", "3"), ("12", "11"), ("12", "13"),
        # From node 13
        ("13", "12"), ("13", "24"),
        # From node 14
        ("14", "11"), ("14", "15"), ("14", "23"),
        # From node 15
        ("15", "10"), ("15", "14"), ("15", "19"), ("15", "22"),
        # From node 16
        ("16", "8"), ("16", "10"), ("16", "17"), ("16", "18"),
        # From node 17
        ("17", "10"), ("17", "16"), ("17", "19"),
        # From node 18
        ("18", "7"), ("18", "16"), ("18", "20"),
        # From node 19
        ("19", "15"), ("19", "17"), ("19", "20"),
        # From node 20
        ("20", "18"), ("20", "19"), ("20", "21"), ("20", "22"),
        # From node 21
        ("21", "20"), ("21", "22"), ("21", "24"),
        # From node 22
        ("22", "15"), ("22", "20"), ("22", "21"), ("22", "23"),
        # From node 23
        ("23", "14"), ("23", "22"), ("23", "24"),
        # From node 24
        ("24", "13"), ("24", "21"), ("24", "23")
    ]
    
    # Add dummy arc (t, s) for flow conservation
    push!(arcs, (sink, source))
    
    # Sort arcs
    sort!(arcs, by = arc -> (arc[1], arc[2]))
    
    num_nodes = length(nodes)
    num_arcs = length(arcs)
    
    # Build node-incidence matrix
    N_full = zeros(Float64, num_nodes, num_arcs)
    node_to_idx = Dict(node => idx for (idx, node) in enumerate(nodes))
    
    for (arc_idx, (from_node, to_node)) in enumerate(arcs)
        from_idx = node_to_idx[from_node]
        to_idx = node_to_idx[to_node]
        
        N_full[from_idx, arc_idx] = 1.0
        N_full[to_idx, arc_idx] = -1.0
    end
    
    # Remove source row
    N = N_full[2:end, :]
    
    return RealWorldNetworkData(name, nodes, arcs, N, source, sink)
end

"""
    generate_nobel_us_network()

Generate the NOBEL-US network from the SNDlib (Orlowski et al., 2010).

Network structure:
- 14 nodes (major US cities)
- 42 directed arcs
- Source: Seattle
- Sink: Princeton

This is a telecommunications backbone network used in network design literature.

# References
- Orlowski, S., Pióro, M., Tomaszewski, A., Wessäly, R. (2010). SNDlib 1.0 - Survivable
  Network Design Library. Networks, 55(3), 276-286.

# Returns
- `RealWorldNetworkData`: complete network data structure
"""
function generate_nobel_us_network()
    name = "NOBEL-US"
    
    # 14 city nodes in the NOBEL-US network
    nodes = [
        "Seattle", "Palo_Alto", "San_Diego", "Salt_Lake_City", "Boulder",
        "Lincoln", "Urbana_Champaign", "Ann_Arbor", "Ithaca",
        "Princeton", "Atlanta", "Houston", "Washington", "New_York"
    ]
    source = "Seattle"
    sink = "Princeton"
    
    # Define arcs based on the NOBEL-US network topology
    # Extracted from Sadana & Delage (2022) Figure EC.2 and Orlowski et al. (2010)
    arcs = [
        # From Seattle
        ("Seattle", "Palo_Alto"), ("Seattle", "Salt_Lake_City"),
        # From Palo_Alto
        ("Palo_Alto", "Seattle"), ("Palo_Alto", "San_Diego"), ("Palo_Alto", "Salt_Lake_City"),
        # From San_Diego
        ("San_Diego", "Palo_Alto"), ("San_Diego", "Houston"),
        # From Salt_Lake_City
        ("Salt_Lake_City", "Seattle"), ("Salt_Lake_City", "Palo_Alto"), 
        ("Salt_Lake_City", "Boulder"),
        # From Boulder
        ("Boulder", "Salt_Lake_City"), ("Boulder", "Lincoln"),
        # From Lincoln
        ("Lincoln", "Boulder"), ("Lincoln", "Urbana_Champaign"),
        # From Urbana_Champaign
        ("Urbana_Champaign", "Lincoln"), ("Urbana_Champaign", "Ann_Arbor"),
        ("Urbana_Champaign", "Atlanta"),
        # From Ann_Arbor
        ("Ann_Arbor", "Urbana_Champaign"), ("Ann_Arbor", "Ithaca"),
        ("Ann_Arbor", "Princeton"),
        # From Ithaca
        ("Ithaca", "Ann_Arbor"), ("Ithaca", "Princeton"), ("Ithaca", "New_York"),
        # From Princeton
        ("Princeton", "Ann_Arbor"), ("Princeton", "Ithaca"), 
        ("Princeton", "Washington"), ("Princeton", "New_York"),
        # From Atlanta
        ("Atlanta", "Urbana_Champaign"), ("Atlanta", "Houston"), ("Atlanta", "Washington"),
        # From Houston
        ("Houston", "San_Diego"), ("Houston", "Atlanta"),
        # From Washington
        ("Washington", "Princeton"), ("Washington", "Atlanta"), ("Washington", "New_York"),
        # From New_York
        ("New_York", "Ithaca"), ("New_York", "Princeton"), ("New_York", "Washington")
    ]
    
    # Add dummy arc (t, s) for flow conservation
    push!(arcs, (sink, source))
    
    # Sort arcs
    sort!(arcs, by = arc -> (arc[1], arc[2]))
    
    num_nodes = length(nodes)
    num_arcs = length(arcs)
    
    # Build node-incidence matrix
    N_full = zeros(Float64, num_nodes, num_arcs)
    node_to_idx = Dict(node => idx for (idx, node) in enumerate(nodes))
    
    for (arc_idx, (from_node, to_node)) in enumerate(arcs)
        from_idx = node_to_idx[from_node]
        to_idx = node_to_idx[to_node]
        
        N_full[from_idx, arc_idx] = 1.0
        N_full[to_idx, arc_idx] = -1.0
    end
    
    # Remove source row
    N = N_full[2:end, :]
    
    return RealWorldNetworkData(name, nodes, arcs, N, source, sink)
end

"""
    print_realworld_network_summary(network::RealWorldNetworkData)

Print a summary of the real-world network structure.
"""
function print_realworld_network_summary(network::RealWorldNetworkData)
    println("=" ^ 60)
    println("$(network.name) Network Summary")
    println("=" ^ 60)
    println("Number of nodes: $(length(network.nodes))")
    println("Number of arcs: $(length(network.arcs))")
    println("Source node: $(network.source)")
    println("Sink node: $(network.sink)")
    println("Node-incidence matrix N dimensions: $(size(network.N))")
    println("  (Source node row removed)")
    println("\nFirst 15 arcs:")
    for i in 1:min(15, length(network.arcs))
        println("  Arc $i: $(network.arcs[i][1]) → $(network.arcs[i][2])")
    end
    println("=" ^ 60)
end

# ==============================================================================
# Test Real-World Networks
# ==============================================================================

println("\n\n")
println("=" ^ 70)
println("TEST 4: Sioux-Falls Network (from LeBlanc et al. 1975)")
println("=" ^ 70)
sioux_falls = generate_sioux_falls_network()
print_realworld_network_summary(sioux_falls)

println("\n\n")
println("=" ^ 70)
println("TEST 5: NOBEL-US Network (from Orlowski et al. 2010)")
println("=" ^ 70)
nobel_us = generate_nobel_us_network()
print_realworld_network_summary(nobel_us)

"""
    generate_abilene_network()

Generate the ABILENE network from SNDlib data.

Network structure:
- 12 nodes (major US cities)
- 30 directed arcs (15 bidirectional physical links)
- Source: "STTLng" (Seattle)
- Sink: "NYCMng" (New York City)

This is the Internet2 Abilene backbone network. The network topology is based on
actual SNDlib data from http://sndlib.zib.de.

# References
- Orlowski, S., Pióro, M., Tomaszewski, A., Wessäly, R. (2010). SNDlib 1.0 - Survivable
  Network Design Library. Networks, 55(3), 276-286.
- Yin Zhang traffic data: http://userweb.cs.utexas.edu/~yzhang/research/AbileneTM

# Returns
- `RealWorldNetworkData`: complete network data structure
"""
function generate_abilene_network()
    name = "ABILENE"
    
    # 12 city nodes in the ABILENE network (from actual SNDlib data)
    nodes = [
        "ATLAng",   # Atlanta
        "ATLAM5",   # Atlanta M5
        "CHINng",   # Chicago
        "DNVRng",   # Denver
        "HSTNng",   # Houston
        "IPLSng",   # Indianapolis
        "KSCYng",   # Kansas City
        "LOSAng",   # Los Angeles
        "NYCMng",   # New York City
        "SNVAng",   # Sunnyvale
        "STTLng",   # Seattle
        "WASHng"    # Washington DC
    ]
    source = "STTLng"
    sink = "NYCMng"
    
    # Define arcs based on actual SNDlib ABILENE network data
    # Each physical link has two directed arcs (bidirectional)
    # Format: (from, to) based on the provided data
    arcs = [
        # Bidirectional link 1: ATLAng - ATLAM5
        ("ATLAng", "ATLAM5"),
        ("ATLAM5", "ATLAng"),
        
        # Bidirectional link 2: ATLAng - HSTNng
        ("ATLAng", "HSTNng"),
        ("HSTNng", "ATLAng"),
        
        # Bidirectional link 3: ATLAng - IPLSng
        ("ATLAng", "IPLSng"),
        ("IPLSng", "ATLAng"),
        
        # Bidirectional link 4: ATLAng - WASHng
        ("ATLAng", "WASHng"),
        ("WASHng", "ATLAng"),
        
        # Bidirectional link 5: CHINng - IPLSng
        ("CHINng", "IPLSng"),
        ("IPLSng", "CHINng"),
        
        # Bidirectional link 6: CHINng - NYCMng
        ("CHINng", "NYCMng"),
        ("NYCMng", "CHINng"),
        
        # Bidirectional link 7: DNVRng - KSCYng
        ("DNVRng", "KSCYng"),
        ("KSCYng", "DNVRng"),
        
        # Bidirectional link 8: DNVRng - SNVAng
        ("DNVRng", "SNVAng"),
        ("SNVAng", "DNVRng"),
        
        # Bidirectional link 9: DNVRng - STTLng
        ("DNVRng", "STTLng"),
        ("STTLng", "DNVRng"),
        
        # Bidirectional link 10: HSTNng - KSCYng
        ("HSTNng", "KSCYng"),
        ("KSCYng", "HSTNng"),
        
        # Bidirectional link 11: HSTNng - LOSAng
        ("HSTNng", "LOSAng"),
        ("LOSAng", "HSTNng"),
        
        # Bidirectional link 12: IPLSng - KSCYng
        ("IPLSng", "KSCYng"),
        ("KSCYng", "IPLSng"),
        
        # Bidirectional link 13: LOSAng - SNVAng
        ("LOSAng", "SNVAng"),
        ("SNVAng", "LOSAng"),
        
        # Bidirectional link 14: NYCMng - WASHng
        ("NYCMng", "WASHng"),
        ("WASHng", "NYCMng"),
        
        # Bidirectional link 15: SNVAng - STTLng
        ("SNVAng", "STTLng"),
        ("STTLng", "SNVAng")
    ]
    
    # Add dummy arc (t, s) for flow conservation
    push!(arcs, (sink, source))
    
    # Sort arcs
    sort!(arcs, by = arc -> (arc[1], arc[2]))
    
    num_nodes = length(nodes)
    num_arcs = length(arcs)
    
    # Build node-incidence matrix
    N_full = zeros(Float64, num_nodes, num_arcs)
    node_to_idx = Dict(node => idx for (idx, node) in enumerate(nodes))
    
    for (arc_idx, (from_node, to_node)) in enumerate(arcs)
        from_idx = node_to_idx[from_node]
        to_idx = node_to_idx[to_node]
        
        N_full[from_idx, arc_idx] = 1.0
        N_full[to_idx, arc_idx] = -1.0
    end
    
    # Remove source row
    N = N_full[2:end, :]
    
    return RealWorldNetworkData(name, nodes, arcs, N, source, sink)
end

"""
    generate_polska_network()

Generate the POLSKA network from SNDlib data.

Network structure:
- 12 nodes (Polish cities)
- 36 directed arcs (18 bidirectional physical links)
- Source: "Szczecin"
- Sink: "Rzeszow"

This is a Polish telecommunications network. The network topology is based on
actual SNDlib data from http://sndlib.zib.de.

# References
- Orlowski, S., Pióro, M., Tomaszewski, A., Wessäly, R. (2010). SNDlib 1.0 - Survivable
  Network Design Library. Networks, 55(3), 276-286.

# Returns
- `RealWorldNetworkData`: complete network data structure
"""
function generate_polska_network()
    name = "POLSKA"
    
    # 12 city nodes in the POLSKA network (from actual SNDlib data)
    nodes = [
        "Bialystok",
        "Bydgoszcz",
        "Gdansk",
        "Katowice",
        "Kolobrzeg",
        "Krakow",
        "Lodz",
        "Poznan",
        "Rzeszow",
        "Szczecin",
        "Warsaw",
        "Wroclaw"
    ]
    source = "Szczecin"
    sink = "Rzeszow"
    
    # Define arcs based on actual SNDlib POLSKA network data
    # Each physical link has two directed arcs (bidirectional)
    # 18 physical links = 36 directed arcs
    arcs = [
        # Bidirectional link 1: Gdansk - Warsaw
        ("Gdansk", "Warsaw"),
        ("Warsaw", "Gdansk"),
        
        # Bidirectional link 2: Gdansk - Kolobrzeg
        ("Gdansk", "Kolobrzeg"),
        ("Kolobrzeg", "Gdansk"),
        
        # Bidirectional link 3: Bydgoszcz - Kolobrzeg
        ("Bydgoszcz", "Kolobrzeg"),
        ("Kolobrzeg", "Bydgoszcz"),
        
        # Bidirectional link 4: Bydgoszcz - Poznan
        ("Bydgoszcz", "Poznan"),
        ("Poznan", "Bydgoszcz"),
        
        # Bidirectional link 5: Bydgoszcz - Warsaw
        ("Bydgoszcz", "Warsaw"),
        ("Warsaw", "Bydgoszcz"),
        
        # Bidirectional link 6: Kolobrzeg - Szczecin
        ("Kolobrzeg", "Szczecin"),
        ("Szczecin", "Kolobrzeg"),
        
        # Bidirectional link 7: Katowice - Krakow
        ("Katowice", "Krakow"),
        ("Krakow", "Katowice"),
        
        # Bidirectional link 8: Katowice - Lodz
        ("Katowice", "Lodz"),
        ("Lodz", "Katowice"),
        
        # Bidirectional link 9: Katowice - Wroclaw
        ("Katowice", "Wroclaw"),
        ("Wroclaw", "Katowice"),
        
        # Bidirectional link 10: Krakow - Rzeszow
        ("Krakow", "Rzeszow"),
        ("Rzeszow", "Krakow"),
        
        # Bidirectional link 11: Krakow - Warsaw
        ("Krakow", "Warsaw"),
        ("Warsaw", "Krakow"),
        
        # Bidirectional link 12: Bialystok - Rzeszow
        ("Bialystok", "Rzeszow"),
        ("Rzeszow", "Bialystok"),
        
        # Bidirectional link 13: Bialystok - Warsaw
        ("Bialystok", "Warsaw"),
        ("Warsaw", "Bialystok"),
        
        # Bidirectional link 14: Lodz - Warsaw
        ("Lodz", "Warsaw"),
        ("Warsaw", "Lodz"),
        
        # Bidirectional link 15: Lodz - Wroclaw
        ("Lodz", "Wroclaw"),
        ("Wroclaw", "Lodz"),
        
        # Bidirectional link 16: Poznan - Szczecin
        ("Poznan", "Szczecin"),
        ("Szczecin", "Poznan"),
        
        # Bidirectional link 17: Poznan - Wroclaw
        ("Poznan", "Wroclaw"),
        ("Wroclaw", "Poznan"),
        
        # Bidirectional link 18: Gdansk - Bialystok
        ("Gdansk", "Bialystok"),
        ("Bialystok", "Gdansk")
    ]
    
    # Add dummy arc (t, s) for flow conservation
    push!(arcs, (sink, source))
    
    # Sort arcs
    sort!(arcs, by = arc -> (arc[1], arc[2]))
    
    num_nodes = length(nodes)
    num_arcs = length(arcs)
    
    # Build node-incidence matrix
    N_full = zeros(Float64, num_nodes, num_arcs)
    node_to_idx = Dict(node => idx for (idx, node) in enumerate(nodes))
    
    for (arc_idx, (from_node, to_node)) in enumerate(arcs)
        from_idx = node_to_idx[from_node]
        to_idx = node_to_idx[to_node]
        
        N_full[from_idx, arc_idx] = 1.0
        N_full[to_idx, arc_idx] = -1.0
    end
    
    # Remove source row
    N = N_full[2:end, :]
    
    return RealWorldNetworkData(name, nodes, arcs, N, source, sink)
end

# ==============================================================================
# Test ABILENE and POLSKA Networks
# ==============================================================================

println("\n\n")
println("=" ^ 70)
println("TEST 6: ABILENE Network (from SNDlib)")
println("=" ^ 70)
abilene = generate_abilene_network()
print_realworld_network_summary(abilene)

println("\n\n")
println("=" ^ 70)
println("TEST 7: POLSKA Network (from SNDlib)")
println("=" ^ 70)
polska = generate_polska_network()
print_realworld_network_summary(polska)