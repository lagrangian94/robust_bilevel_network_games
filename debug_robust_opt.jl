using JuMP
using LinearAlgebra
using SparseArrays
using Infiltrator
using Pajarito
using Gurobi
using Mosek, MosekTools


# Load network generator
include("network_generator.jl")
includet("sdp_build_uncertainty_set.jl")

using .NetworkGenerator


function build_robust_model(network, S, ϕU, w, v, uncertainty_set; optimizer=nothing)
    
    # Extract network dimensions
    num_nodes = length(network.nodes)
    num_arcs = length(network.arcs)-1 #dummy arc 제외
    num_interdictable = sum(network.interdictable_arcs)
    
    # Node-arc incidence matrix (excluding source row)
    N = network.N
    R, r_dict, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:epsilon]
    # Dummy arc index (t,s)
    dummy_arc_idx = findfirst(arc -> arc == ("t", "s"), network.arcs)
    
    # Create model
    # model = Model(
    #     optimizer_with_attributes(
    #         Pajarito.Optimizer,
    #         "oa_solver" => optimizer_with_attributes(
    #             Gurobi.Optimizer,
    #             MOI.Silent() => false,
    #         ),
    #         "conic_solver" =>
    #             optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => false),
    #     )
    # )
    model = Model(Mosek.Optimizer)
    if !isnothing(optimizer)
        set_optimizer(model, optimizer)
    end
    
    println("Building 2DRNDP model...")
    println("  Nodes: $num_nodes, Arcs: $num_arcs, Scenarios: $S")
    println("  Interdictable arcs: $num_interdictable")
    println("  Dummy arc index: $dummy_arc_idx")
    println("  Parameters: ϕU = $ϕU, w = $w, v = $v")
    
    # =========================================================================
    # DECISION VARIABLES
    # =========================================================================
    
    # --- Scalar variables ---
    @variable(model, t1[s=1:S]>=0)  # Objective epigraph variable
    # @variable(model, nu>=-500)  # Budget for recourse decisions
    # @variable(model, λ >= 0)  # Budget allocation parameter
    @objective(model, Max, sum(t1))

    # --- Vector variables ---
    # x: interdiction decisions (binary for interdictable arcs, 0 for others)
    @variable(model, x[1:num_arcs]>=0)#, Bin)
    @constraint(model, [i=1:num_arcs], x[i] <= 0)
    # h: initial resource allocation
    @variable(model, h[1:num_arcs] >= 0)
    @constraint(model, sum(h) == 0.0)
    
    # Y: |A| × |A| matrix (follower's additional LDR coefficient)
    @variable(model, Ytilde[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound= -ϕU, upper_bound = ϕU)  
    
    # Yts: 1 x (|A|+1) matrix (coefficient for dummy arc t->s)
    @variable(model, Yts_tilde[s=1:S, 1, 1:num_arcs+1], lower_bound= -ϕU, upper_bound = ϕU)
    Ytilde_L, Yts_tilde_L =  Ytilde[:,:,1:num_arcs], Yts_tilde[:,:,1:num_arcs]
    Ytilde_0, Yts_tilde_0 =  Ytilde[:,:,num_arcs+1], Yts_tilde[:,1,num_arcs+1]
    dim_Λtilde1_rows = (num_nodes - 1) + num_arcs + num_arcs
    @variable(model, Λtilde1[s=1:S, 1:dim_Λtilde1_rows, 1:size(R, 1)])
    @constraint(model, [s=1:S, i=1:dim_Λtilde1_rows], Λtilde1[s, i, :] in SecondOrderCone())
    @variable(model, mu[s=1:S, 1:size(R, 1)])
    @constraint(model, [s=1:S], mu[s, :] in SecondOrderCone())
    I_0 = [Matrix{Float64}(I, num_arcs, num_arcs) zeros(num_arcs)]
    # Split N matrix: N = [N_y | N_ts]
    N_y = N[:, 1:num_arcs]  # Regular arcs: (|V|-1) × |A|
    N_ts = N[:, end]         # Dummy arc: (|V|-1) × 1
    for s in 1:S
        r_bar = r_dict[s]  # Get r̄ for this scenario
        # =====================================================================
        # Follower's Lambda_tilde1 constraint 1: Λ˜s_1 * R = [Q˜s; Π˜s; Φ˜s]
        # =====================================================================
        # Block 2: -N_y * Y˜s - N_ts * Y˜s_ts
        block2 = -N_y * Ytilde_L[s, :, :] - N_ts * Yts_tilde_L[s, :,:]
        # Block 3: -Y˜s
        block3 = -Ytilde_L[s, :, :]
        # Block 6: Y˜s
        block6 = Ytilde_L[s, :, :]
        lhs_mat = vcat(block2, block3, block6)
        # RHS: [0; 0; diag(λ-v*ψ0); 0; 0; 0]
        # Block 2: zeros(num_nodes-1, num_arcs)
        block2_rhs = zeros(num_nodes-1, num_arcs)
        # Block 3: diag(λ-v*ψ0) - 대각선 행렬 (JuMP 표현식 사용)
        block3_rhs = [AffExpr(0.0) for i in 1:num_arcs, j in 1:num_arcs]
        for j in 1:num_arcs
            block3_rhs[j, j] = 1 - v*x[j]
        end
        block6_rhs = zeros(num_arcs, num_arcs)
        rhs_mat = vcat(block2_rhs, block3_rhs, block6_rhs)
        @constraint(model, Λtilde1[s, :, :] * R -lhs_mat .== rhs_mat)
        # Λ˜s_1 * r̄ ≥ [λ*d0; 0; h; 0; 0; 0]
        lhs_vec_2 = -N_y * Ytilde_0[s,:] - N_ts * Yts_tilde_0[s]
        lhs_vec_3 = -Ytilde_0[s,:]
        lhs_vec_6 = Ytilde_0[s,:]
        lhs_vec = vcat(lhs_vec_2, lhs_vec_3, lhs_vec_6)
        @constraint(model, Λtilde1[s, :, :] * r_bar .+ lhs_vec .>= 0.0)
        @constraint(model, mu[s,:]'*R .== Yts_tilde_L[s,:,:])
        @constraint(model, mu[s,:]'*r_bar .- t1[s] .+ Yts_tilde_0[s] .>= 0.0)
    end

     # =========================================================================
    # Return model and variables
    # =========================================================================
    vars = Dict(
        # Scalar
        :t1 => t1,
        # Vector
        :x => x,
        :h => h,
        # Matrices
        :Ytilde => Ytilde,
        :Yts_tilde => Yts_tilde,
        # Dual variables
        :Λtilde1 => Λtilde1,
        :mu => mu
    )
    
    println("\nModel construction summary:")
    println("  - Variables: $(num_variables(model))")
    println("  - Constraints: $(num_constraints(model, AffExpr, MOI.LessThan{Float64}) + 
                                 num_constraints(model, AffExpr, MOI.EqualTo{Float64}))")
    println("  - Binary variables: $(sum(is_binary(x[i]) for i in 1:num_arcs))")
    println("="^80)
    optimize!(model)
    @infiltrate
    return model, vars
end

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
build_robust_model(network, S, ϕU, w, v, uncertainty_set)
