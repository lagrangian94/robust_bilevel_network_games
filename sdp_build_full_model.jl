using JuMP
using LinearAlgebra
using SparseArrays
using Infiltrator
using Pajarito
using Gurobi
using Mosek, MosekTools
"""
@infiltrate 지점에서 멈추고 변수 확인 가능
@locals 입력하면 모든 변수 확인
@continue 또는 @exit로 계속 진행
"""
# Load network generator
include("network_generator.jl")
using .NetworkGenerator
"""
Build the full 2DRNDP model (14) without COP constraints (14f, 14i)

Arguments:
- network: Network structure from NetworkGenerator
- S: Number of scenarios
- ϕU: Upper bound on interdiction effectiveness
- w: Budget weight parameter
- v: Interdiction effectiveness parameter (used in COP matrix structure)
- optimizer: JuMP optimizer (e.g., Gurobi.Optimizer)

Returns:
- model: JuMP model
- vars: Dictionary containing all decision variables

Note: 
- ν (nu) is a DECISION VARIABLE (appears in objective and constraint 14l)
- v is a PARAMETER (appears in COP matrix Φ - vW)
"""
function build_full_2DRNDP_model(network, S, ϕU, w, v, uncertainty_set; optimizer=nothing)
    
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
    @variable(model, t)  # Objective epigraph variable
    @variable(model, nu)  # Budget for recourse decisions
    @variable(model, λ >= 0)  # Budget allocation parameter
    
    # --- Vector variables ---
    # x: interdiction decisions (binary for interdictable arcs, 0 for others)
    @variable(model, x[1:num_arcs]>=0)#, Bin)
    # h: initial resource allocation
    @variable(model, h[1:num_arcs] >= 0)
    
    # ψ0: auxiliary variable for linearization (14q)
    @variable(model, ψ0[1:num_arcs] >= 0)
    
    # --- Scenario-indexed variables (scalar per scenario) ---
    @variable(model, ηhat[1:S])   # Leader's scenario cost
    @variable(model, ηtilde[1:S]) # Follower's scenario cost
    
    # --- Scenario and arc-indexed variables ---
    @variable(model, μhat[1:S, 1:num_arcs])   # Leader's dual variables
    @variable(model, μtilde[1:S, 1:num_arcs]) # Follower's dual variables
    
    # --- Matrix variables (scenario-indexed) ---
    # LDR coefficient matrices - all are |A| × |A| matrices
    @variable(model, Φhat[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound= -ϕU, upper_bound = ϕU)    # Leader's flow coefficient
    @variable(model, Ψhat[s=1:S, 1:num_arcs, 1:num_arcs+1])    # Leader's W matrix
    @variable(model, Φtilde[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound= -ϕU, upper_bound = ϕU)  # Follower's flow coefficient
    @variable(model, Ψtilde[s=1:S, 1:num_arcs, 1:num_arcs+1])  # Follower's W matrix
    
    # Additional LDR coefficients
    # Π: (|V|-1) × |A| matrices (node prices, excluding source)
    @variable(model, Πhat[s=1:S, 1:num_nodes-1, 1:num_arcs+1], lower_bound= -ϕU, upper_bound = ϕU)  # Leader's price coefficient
    @variable(model, Πtilde[s=1:S, 1:num_nodes-1, 1:num_arcs+1], lower_bound= -ϕU, upper_bound = ϕU) # Follower's price coefficient
    
    # Y: |A| × |A| matrix (follower's additional LDR coefficient)
    @variable(model, Ytilde[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound= -ϕU, upper_bound = ϕU)  
    
    # Yts: 1 x (|A|+1) matrix (coefficient for dummy arc t->s)
    @variable(model, Yts_tilde[s=1:S, 1, 1:num_arcs+1], lower_bound= -ϕU, upper_bound = ϕU)  
    

    # 변수 따로 정리
    Φhat_L, Ψhat_L, Φtilde_L, Ψtilde_L = Φhat[:,:,1:num_arcs], Ψhat[:,:,1:num_arcs], Φtilde[:,:,1:num_arcs], Ψtilde[:,:,1:num_arcs]
    Πhat_L, Πtilde_L = Πhat[:,:,1:num_arcs], Πtilde[:,:,1:num_arcs]
    Ytilde_L, Yts_tilde_L =  Ytilde[:,:,1:num_arcs], Yts_tilde[:,:,1:num_arcs]
    Φhat_0, Ψhat_0, Φtilde_0, Ψtilde_0 = Φhat[:,:,num_arcs+1], Ψhat[:,:,num_arcs+1], Φtilde[:,:,num_arcs+1], Ψtilde[:,:,num_arcs+1]
    Πhat_0, Πtilde_0 = Πhat[:,:,num_arcs+1], Πtilde[:,:,num_arcs+1]
    Ytilde_0, Yts_tilde_0 =  Ytilde[:,:,num_arcs+1], Yts_tilde[:,1,num_arcs+1]
    # --- Dual variables for inner problems ---
    # These arise from dualizing the inner problems
    # Dimensions are based on the dual formulation structure
    
    # For Λhat1: corresponds to constraint (14m)
    # First block: (|V|-1) rows for N^T Π
    # Second block: |A| rows for I_0^T Φ  
    # Third block: |A| rows for Φ itself
    # Total: (|V|-1 + |A| + |A|) × |A|
    dim_Λhat1_rows = num_arcs+1 +(num_nodes - 1) + num_arcs #여기 차원 확인.
    @variable(model, Λhat1[s=1:S, 1:dim_Λhat1_rows, 1:size(R, 1)])
    # For Λhat2: corresponds to constraint (14n)
    # Dimension: |A| × |A|
    @variable(model, Λhat2[s=1:S, 1:num_arcs, 1:size(R, 1)])
    
    # For Λtilde1: corresponds to constraint (14o)  
    # Has additional blocks for Y terms
    # Structure: similar to Λhat1 but with more rows
    dim_Λtilde1_rows = num_arcs+1 + (num_nodes - 1) + num_arcs + num_nodes-1 + num_arcs + num_arcs
    @variable(model, Λtilde1[s=1:S, 1:dim_Λtilde1_rows, 1:size(R, 1)])
    
    # For Λtilde2: corresponds to constraint (14p)
    @variable(model, Λtilde2[s=1:S, 1:num_arcs, 1:size(R, 1)])

    # Second order cone constraints
    @constraint(model, [s=1:S, i=1:dim_Λhat1_rows], Λhat1[s, i, :] in SecondOrderCone())
    @constraint(model, [s=1:S, i=1:num_arcs], Λhat2[s, i, :] in SecondOrderCone())
    @constraint(model, [s=1:S, i=1:dim_Λtilde1_rows], Λtilde1[s, i, :] in SecondOrderCone())
    @constraint(model, [s=1:S, i=1:num_arcs], Λtilde2[s, i, :] in SecondOrderCone())
    
    println("  ✓ Decision variables created")
    
    # =========================================================================
    # CONSTRAINTS
    # =========================================================================
    
    # --- (14a) Objective function ---
    @objective(model, Min, t + w * nu)
    
    # --- (14b) Initial resource and domain constraints ---
    @constraint(model, resource_budget, sum(h) <= λ * w)
    
    # x must be binary, and only interdictable arcs can be selected
    for i in 1:num_arcs
        if !network.interdictable_arcs[i]
            @constraint(model, x[i] == 0)
            println("Arc $i is not interdictable")
        end
    end
    @constraint(model, x[9] == 1)
    # --- (14c) Scenario cost constraint ---
    @constraint(model, total_cost, sum(ηhat[s] + ηtilde[s] for s in 1:S) <= S * t)
    
    println("  ✓ Constraints (14a-14c) added")
    # =========================================================================
    # COP CONSTRAINTS (14d-14i) - NOT IMPLEMENTED
    # =========================================================================
    @variable(model, ϑhat[1:S]>=0)   # Leader's auxiliary variable for SDP (14d, 14e)
    @variable(model, ϑtilde[1:S]>=0) # Follower's auxiliary variable for SDP (14g, 14h)
    # --- COP matrices for constraints (14e, 14h) ---
    # M̂s: Leader's SDP matrix (|A|+1) × (|A|+1)
    # M̃s: Follower's SDP matrix (|A|+1) × (|A|+1)
    @variable(model, Mhat[1:S, 1:num_arcs+1, 1:num_arcs+1])
    @variable(model, Mtilde[1:S, 1:num_arcs+1, 1:num_arcs+1])
    # --- (14d) Leader's scenario bound: ϑˆs <= ηˆs ---
    # --- (14g) Follower's scenario bound: ϑ˜s <= η˜s ---
    for s in 1:S
        xi_bar = r_dict[s][2:num_arcs+1]
        # --- (18d) Leader's SDP constraint ---
        # Matrix structure: 
        for i in 1:(num_arcs+1), j in 1:(num_arcs+1)
            if i <= num_arcs && j <= num_arcs
                if i == j
                    @constraint(model, Mhat[s,i,j] == ϑhat[s]-(Φhat_L[s,i,j] - v*Ψhat_L[s,i,j]))
                else
                    @constraint(model, Mhat[s,i,j] == -(Φhat_L[s,i,j] - v*Ψhat_L[s,i,j]))
                end
            elseif i == num_arcs+1 && j <= num_arcs
                @constraint(model, Mhat[s,i,j] == -(1/2)*(Φhat_0[s,j]-v*Ψhat_0[s,j])-ϑhat[s]*xi_bar[j])
            elseif i <= num_arcs && j == num_arcs+1
                @constraint(model, Mhat[s,i,j] == -(1/2)*(Φhat_0[s,i]-v*Ψhat_0[s,i])-ϑhat[s]*xi_bar[i])
            elseif i == num_arcs+1 && j == num_arcs+1
                @constraint(model, Mhat[s,i,j] == ηhat[s] - ϑhat[s]*(epsilon^2 - sum(xi_bar.^2)))
            else
                @constraint(model, Mhat[s,i,j] == 0)
            end
        end
        # --- (18e) Follower's SDP constraint ---
        for i in 1:(num_arcs+1), j in 1:(num_arcs+1)
            if i <= num_arcs && j <= num_arcs
                if i == j
                    @constraint(model, Mtilde[s,i,j] == ϑtilde[s]-(Φtilde_L[s,i,j] - v*Ψtilde_L[s,i,j]))
                else
                    @constraint(model, Mtilde[s,i,j] == -(Φtilde_L[s,i,j] - v*Ψtilde_L[s,i,j]))
                end
            elseif i == num_arcs+1 && j <= num_arcs
                @constraint(model, Mtilde[s,i,j] == -(1/2)*(Φtilde_0[s,j]-v*Ψtilde_0[s,j]-Yts_tilde_L[s,1,j])-ϑtilde[s]*xi_bar[j])
            elseif i <= num_arcs && j == num_arcs+1
                @constraint(model, Mtilde[s,i,j] == -(1/2)*(Φtilde_0[s,i]-v*Ψtilde_0[s,i]-Yts_tilde_L[s,1,i])-ϑtilde[s]*xi_bar[i])
            elseif i == num_arcs+1 && j == num_arcs+1
                @constraint(model, Mtilde[s,i,j] == ηtilde[s] + Yts_tilde_0[s] - ϑtilde[s]*(epsilon^2 - sum(xi_bar.^2)))
            else
                @constraint(model, Mtilde[s,i,j] == 0)
            end
        end
    end
    @constraint(model, [s=1:S], Mhat[s,:,:] in PSDCone())
    @constraint(model, [s=1:S], Mtilde[s,:,:] in PSDCone())
    # where:
    #   - ηˆs: scalar (scenario cost upper bound)
    #   - Φˆs: |A|×|A| matrix (flow LDR coefficient)
    #   - Ψˆs (=Wˆs): |A|×|A| matrix (interdiction LDR coefficient)
    #   - v: PARAMETER (interdiction effectiveness, NOT decision variable ν)
    #   - Augmented to (|A|+1)×(|A|+1) by adding row/column of zeros except (|A|+1,|A|+1) = ηˆs
    # 
    
    
    # --- Big-M constraints for Ψhat and Ψtilde ---
    # These enforce the relationships between W (Ψ), Φ, and x
    # Using diagonal structure: Ψ can only be non-zero on diagonal
    
    println("  Adding Big-M constraints (14j, 14k)...")
    
    for s in 1:S
        for i in 1:num_arcs
            for j in 1:num_arcs+1
                # Leader constraints (14j)
                @constraint(model, Ψhat[s,i,j] <= ϕU * x[i])
                @constraint(model, Ψhat[s,i,j] - Φhat[s,i,j] <= 0)
                @constraint(model, Φhat[s,i,j] - Ψhat[s,i,j] <= ϕU * (1 - x[i]))
                
                # Follower constraints (14k)
                @constraint(model, Ψtilde[s,i,j] <= ϕU * x[i])
                @constraint(model, Ψtilde[s,i,j] - Φtilde[s,i,j] <= 0)
                @constraint(model, Φtilde[s,i,j] - Ψtilde[s,i,j] <= ϕU * (1 - x[i]))
            end
        end
    end
    
    println("  ✓ Big-M constraints (14j, 14k) added")
    
    # --- (14l) Budget constraint for dual variables ---
    for k in 1:num_arcs
        @constraint(model, sum(μtilde[s,k] + μhat[s,k] for s in 1:S) <= S * nu)
    end
    
    println("  ✓ Budget constraint (14l) added")
    
    # --- (14m) Leader's dual feasibility constraints ---
    # Λˆs_1 R = [Qˆs(Πˆs, Φˆs); Πˆs; Φˆs]
    # where Qˆs = N^T Πˆs + I_0^T Φˆs
    # R is the constraint matrix from the inner problem
    
    # We need to define:
    # - d0: demand vector (assuming it's related to source-sink flow) -> demand vector 아님. num_arcs+1 차원 벡터고 마지막 num_arcs+1 index만 1인 standard basis vector
    # - r̄: right-hand side vector for the dual constraints -> r_dict[s] 하면 됨
    
    println("  Adding dual constraints (14m-14p)...")
    # --- Define auxiliary structures ---
    # d0 = [0; 0; ...; 0; 1] ∈ ℝ^(|A|+1) - standard basis vector
    d0 = zeros(num_arcs + 1)
    d0[end] = 1.0
    # I_0 = [I | 0] ∈ ℝ^(|A| × (|A|+1)) - identity with zero column
    I_0 = [Matrix{Float64}(I, num_arcs, num_arcs) zeros(num_arcs)]
    # Split N matrix: N = [N_y | N_ts]
    N_y = N[:, 1:num_arcs]  # Regular arcs: (|V|-1) × |A|
    N_ts = N[:, end]         # Dummy arc: (|V|-1) × 1
    # --- Add constraints for each scenario ---
    for s in 1:S
        r_bar = r_dict[s]  # Get r̄ for this scenario
        # =====================================================================
        # Leader's Lambda_hat1 constraint 1: Λˆs_1 * R = [Qˆs; Πˆs; Φˆs]
        # =====================================================================
        Q_hat = N' * Πhat_L[s, :, :] + I_0' * Φhat_L[s, :, :]
        rhs_mat = vcat(Q_hat, Πhat_L[s, :, :], Φhat_L[s, :, :])
        @constraint(model, Λhat1[s, :, :] * R .== rhs_mat)
        # =====================================================================
        # Leader's Lambda_hat1 constraint 2
        lhs_mat = vcat(N' * Πhat_0[s, :] + I_0' * Φhat_0[s, :], Πhat_0[s, :], Φhat_0[s, :])
        # Λˆs_1 * r̄ ≥ [d0; 0; 0]
        rhs_rbar = vcat(d0, zeros(num_nodes-1), zeros(num_arcs))
        @constraint(model, Λhat1[s, :, :] * r_bar .+ lhs_mat .>= rhs_rbar)
        # =====================================================================
        # Leader's Lambda_hat2 constraint: Λˆs_2 * R = -Φˆs
        # =====================================================================
        @constraint(model, Λhat2[s, :, :] * R .== -Φhat_L[s, :, :])
        # Λˆs_2 * r̄ ≥ -μˆs + phi_hat_0
        @constraint(model, Λhat2[s, :, :] * r_bar .- Φhat_0[s, :] .+ μhat[s, :] .>= 0.0)
        # =====================================================================
        # Follower's Lambda_tilde1 constraint 1: Λ˜s_1 * R = [Q˜s; Π˜s; Φ˜s]
        # =====================================================================
        # Block 1: Q˜s
        Q_tilde_col = N' * Πtilde_L[s, :, :] + I_0' * Φtilde_L[s, :, :]
        # Block 2: -N_y * Y˜s - N_ts * Y˜s_ts
        block2 = -N_y * Ytilde_L[s, :, :] - N_ts * Yts_tilde_L[s, :,:]
        # Block 3: -Y˜s
        block3 = -Ytilde_L[s, :, :]
        # Block 4: Π˜s
        block4 = Πtilde_L[s, :, :]
        # Block 5: Φ˜s
        block5 = Φtilde_L[s, :, :]
        # Block 6: Y˜s
        block6 = Ytilde_L[s, :, :]
        lhs_mat = vcat(Q_tilde_col, block2, block3, block4, block5, block6)
        # RHS: [0; 0; diag(λ-v*ψ0); 0; 0; 0]
        # Block 1: zeros(num_arcs+1, num_arcs)
        block1_rhs = zeros(num_arcs+1, num_arcs)
        # Block 2: zeros(num_nodes-1, num_arcs)
        block2_rhs = zeros(num_nodes-1, num_arcs)
        # Block 3: diag(λ-v*ψ0) - 대각선 행렬 (JuMP 표현식 사용)
        block3_rhs = [AffExpr(0.0) for i in 1:num_arcs, j in 1:num_arcs]
        for j in 1:num_arcs
            block3_rhs[j, j] = λ - v*ψ0[j]
        end
        # Block 4: zeros(num_nodes-1, num_arcs)
        block4_rhs = zeros(num_nodes-1, num_arcs)
        # Block 5: zeros(num_arcs, num_arcs)
        block5_rhs = zeros(num_arcs, num_arcs)
        # Block 6: zeros(num_arcs, num_arcs)
        block6_rhs = zeros(num_arcs, num_arcs)
        rhs_mat = vcat(block1_rhs, block2_rhs, block3_rhs, block4_rhs, block5_rhs, block6_rhs)
        @constraint(model, Λtilde1[s, :, :] * R .- lhs_mat .== rhs_mat)
        # Λ˜s_1 * r̄ ≥ [λ*d0; 0; h; 0; 0; 0]
        lhs_vec_1 =  N' * Πtilde_0[s, :] + I_0' * Φtilde_0[s, :]
        lhs_vec_2 = -N_y * Ytilde_0[s,:] - N_ts * Yts_tilde_0[s]
        lhs_vec_3 = -Ytilde_0[s,:]
        lhs_vec_4 = Πtilde_0[s,:]
        lhs_vec_5 = Φtilde_0[s,:]
        lhs_vec_6 = Ytilde_0[s,:]
        lhs_vec = vcat(lhs_vec_1, lhs_vec_2, lhs_vec_3, lhs_vec_4, lhs_vec_5, lhs_vec_6)
        rhs_rbar_tilde = vcat(
            λ .* d0,
            zeros(num_nodes-1),
            h,
            zeros(num_nodes-1),
            zeros(num_arcs),
            zeros(num_arcs)
        )
        @constraint(model, Λtilde1[s, :, :] * r_bar .+ lhs_vec .>= rhs_rbar_tilde)
        # 
        # =====================================================================
        # (14p) Follower's capacity dual: Λ˜s_2 * R = -Φ˜s
        # =====================================================================
        @constraint(model, Λtilde2[s, :, :] * R .+ Φtilde_L[s, :, :] .== 0.0)
        # Λ˜s_2 * r̄ ≥ -μ˜s
        @constraint(model, Λtilde2[s, :, :] * r_bar .- Φtilde_0[s, :] .+ μtilde[s, :] .>= 0.0)
    end
    println("  ✓ Dual constraints (14m-14p) added for all scenarios")
        # 
    # --- (14q) Linearization constraints for ψ0 ---
    # These linearize the product λ * x
    λU = 100.0  # Upper bound on λ (should be set based on problem)
    for k in 1:num_arcs
        @constraint(model, ψ0[k] <= λU * x[k])
        @constraint(model, ψ0[k] <= λ)
        @constraint(model, ψ0[k] >= λ - λU * (1 - x[k]))
        @constraint(model, ψ0[k] >= 0)
    end
    
    println("  ✓ Linearization constraints (14q) added")
    
    # =========================================================================
    # Return model and variables
    # =========================================================================
    vars = Dict(
        # Scalar
        :t => t,
        :nu => nu,
        :λ => λ,
        # Vector
        :x => x,
        :h => h,
        :ψ0 => ψ0,
        # Scenario scalar
        :ηhat => ηhat,
        :ηtilde => ηtilde,
        # Scenario × Arc
        :μhat => μhat,
        :μtilde => μtilde,
        # Matrices
        :Φhat => Φhat,
        :Ψhat => Ψhat,
        :Φtilde => Φtilde,
        :Ψtilde => Ψtilde,
        :Πhat => Πhat,
        :Πtilde => Πtilde,
        :Ytilde => Ytilde,
        :Yts_tilde => Yts_tilde,
        # Dual variables
        :Λhat1 => Λhat1,
        :Λhat2 => Λhat2,
        :Λtilde1 => Λtilde1,
        :Λtilde2 => Λtilde2
    )
    
    println("\nModel construction summary:")
    println("  - Variables: $(num_variables(model))")
    println("  - Constraints: $(num_constraints(model, AffExpr, MOI.LessThan{Float64}) + 
                                 num_constraints(model, AffExpr, MOI.EqualTo{Float64}))")
    println("  - Binary variables: $(sum(is_binary(x[i]) for i in 1:num_arcs))")
    println("="^80)
    
    return model, vars
end


"""
Add network-specific data (d0, capacities, etc.) to the model
"""
function add_problem_data!(model, vars, network, d0, capacities)
    println("Adding problem-specific data...")
    
    # d0: demand vector (source to sink flow requirement)
    # capacities: arc capacity data for each scenario
    
    # This function would add the remaining constraints (14m-14p)
    # using the actual problem data
    
    println("  ✓ Problem data added")
end


"""
Test function to verify model construction
"""
function test_model_construction()
    println("="^80)
    println("Testing model construction")
    println("="^80)
    

    
    # Generate a small test network
    network = generate_grid_network(3, 3, seed=42)
    print_network_summary(network)
    
    # Model parameters
    S = 3  # Number of scenarios
    ϕU = 10.0  # Upper bound on interdiction effectiveness
    w = 1.0  # Budget weight
    v = 0.5  # Interdiction effectiveness parameter (used in COP matrix Φ - v*W)
    
    # Build model (without optimizer for testing)
    model, vars = build_full_2DRNDP_model(network, S, ϕU, w, v)
    
    println("\n✓ Model construction test PASSED")
    
    return model, vars
end