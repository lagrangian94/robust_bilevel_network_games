using JuMP
using LinearAlgebra
using SparseArrays
using Infiltrator
using Pajarito
using Gurobi
using Mosek, MosekTools
using Hypatia, HiGHS
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
- λU: Upper bound on λ
- w: Budget weight parameter
- v: Interdiction effectiveness parameter (used in COP matrix structure)
- uncertainty_set: Dictionary containing uncertainty set
- optimizer: JuMP optimizer (e.g., Gurobi.Optimizer)

Returns:
- model: JuMP model
- vars: Dictionary containing all decision variables

Note: 
- ν (nu) is a DECISION VARIABLE (appears in objective and constraint 14l)
- v is a PARAMETER (appears in COP matrix Φ - vW)
"""
function build_full_2DRNDP_model(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_solver=nothing, conic_solver=nothing,
    # Optional: if provided, these are treated as fixed parameters
    x_fixed=nothing, λ_fixed=nothing, h_fixed=nothing, ψ0_fixed=nothing)

    # Extract network dimensions
    num_nodes = length(network.nodes)
    num_arcs = length(network.arcs)-1 #dummy arc 제외
    num_interdictable = sum(network.interdictable_arcs)

    # Node-arc incidence matrix (excluding source row)
    N = network.N
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    # Dummy arc index (t,s)
    dummy_arc_idx = findfirst(arc -> arc == ("t", "s"), network.arcs)

    # Create model
    if !isnothing(mip_solver)
        model = Model(
            optimizer_with_attributes(
                Pajarito.Optimizer,
                "oa_solver" => optimizer_with_attributes(
                    mip_solver,
                    MOI.Silent() => false,
                ),
                "conic_solver" =>
                    optimizer_with_attributes(conic_solver, MOI.Silent() => false),
            )
        )
    else
        model = Model(conic_solver)
    end
    if conic_solver == MosekTools.Optimizer
        set_optimizer_attribute(model, "MSK_IPAR_LOG", 10)  # log 활성화
        set_optimizer_attribute(model, "MSK_IPAR_LOG_PRESOLVE", 1)
        # set_optimizer_attribute(model, "MSK_IPAR_LOG_OPTIMIZER", 1)
        set_optimizer_attribute(model, "MSK_IPAR_LOG_INFEAS_ANA", 1)  # infeasibility analysis
        set_optimizer_attribute(model, "MSK_IPAR_INFEAS_REPORT_AUTO", 1)

    end

    println("Building 2DRNDP model...")
    println("  Nodes: $num_nodes, Arcs: $num_arcs, Scenarios: $S")
    println("  Interdictable arcs: $num_interdictable")
    println("  Dummy arc index: $dummy_arc_idx")
    println("  Parameters: ϕU = $ϕU, γ = $γ, w = $w, v = $v")

    # =========================================================================
    # DECISION VARIABLES
    # =========================================================================

    # --- Scalar variables ---
    # @variable(model, t)  # Objective epigraph variable
    @variable(model, nu>= 0)  # Budget for recourse decisions
    if isnothing(λ_fixed)
        @variable(model, λ >= 0)  # Budget allocation parameter
    else
        λ=λ_fixed
    end
    # --- Vector variables ---
    # x: interdiction decisions (binary for interdictable arcs, 0 for others)
    if isnothing(x_fixed)
        @variable(model, x[1:num_arcs], Bin)
    else
        x=x_fixed
    end
    # h: initial resource allocation
    if isnothing(h_fixed)
        @variable(model, h[1:num_arcs] >= 0)
    else
        h=h_fixed
    end

    # ψ0: auxiliary variable for linearization (14q)
    if isnothing(ψ0_fixed)
        @variable(model, ψ0[1:num_arcs] >= 0)
    else
        ψ0=ψ0_fixed
    end

    # --- Scenario-indexed variables (scalar per scenario) ---
    @variable(model, ηhat[1:S]>=0)   # Leader's scenario cost
    @variable(model, ηtilde[1:S]) #IMPORTANT: lower bound is NOT 0!! # Follower's scenario cost

    # --- Scenario and arc-indexed variables ---
    @variable(model, μhat[1:S, 1:num_arcs]>=0)   # Leader's dual variables
    @variable(model, μtilde[1:S, 1:num_arcs]>=0) # Follower's dual variables

    # --- Matrix variables (scenario-indexed) ---
    # LDR coefficient matrices - all are |A| × |A| matrices
    @variable(model, Φhat[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound= -ϕU, upper_bound = ϕU)    # Leader's flow coefficient
    @variable(model, Ψhat[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound= 0.0)    # Leader's W matrix
    @variable(model, Φtilde[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound= -ϕU, upper_bound = ϕU)  # Follower's flow coefficient
    @variable(model, Ψtilde[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound= 0.0)  # Follower's W matrix

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
    @variable(model, Λhat1[s=1:S, 1:dim_Λhat1_rows, 1:size(R[1], 1)])
    # For Λhat2: corresponds to constraint (14n)
    # Dimension: |A| × |A|
    @variable(model, Λhat2[s=1:S, 1:num_arcs, 1:size(R[1], 1)])

    # For Λtilde1: corresponds to constraint (14o)  
    # Has additional blocks for Y terms
    # Structure: similar to Λhat1 but with more rows
    dim_Λtilde1_rows = num_arcs+1 + (num_nodes - 1) + num_arcs + num_nodes-1 + num_arcs + num_arcs
    @variable(model, Λtilde1[s=1:S, 1:dim_Λtilde1_rows, 1:size(R[1], 1)])

    # For Λtilde2: corresponds to constraint (14p)
    @variable(model, Λtilde2[s=1:S, 1:num_arcs, 1:size(R[1], 1)])

    # Second order cone constraints
    @constraint(model, [s=1:S, i=1:dim_Λhat1_rows], Λhat1[s, i, :] in SecondOrderCone()) ## TODO:: 주석풀기
    @constraint(model, [s=1:S, i=1:num_arcs], Λhat2[s, i, :] in SecondOrderCone()) ## TODO:: 주석풀기
    # @constraint(model, [s=1:S, i=1:dim_Λtilde1_rows], Λtilde1[s, i, :] in SecondOrderCone()) ## TODO:: 주석풀기 ## 이게 numerical 문제??
    @constraint(model, [s=1:S, i=1:num_arcs], Λtilde2[s, i, :] in SecondOrderCone()) ## TODO:: 주석풀기

    # @constraint(model, [s=1:S, i=1:num_arcs], Λtilde1[s, i, 1] <= model[:nu])
    # @constraint(model, [s=1:S, i=1:num_arcs], Λtilde2[s, i, 1] <= model[:nu])
    println("  ✓ Decision variables created")

    # =========================================================================
    # CONSTRAINTS
    # =========================================================================

    # --- (14a) Objective function ---
    @objective(model, Min, (1/S)*sum(ηhat[s] + ηtilde[s] for s in 1:S) + (1/S)*w * nu)

    # --- (14b) Initial resource and domain constraints ---
    if isnothing(λ_fixed)
        @constraint(model, resource_budget, sum(h) <= λ * w)
        @constraint(model, sum(x) <= γ) 
        # x must be binary, and only interdictable arcs can be selected
        for i in 1:num_arcs
            if !network.interdictable_arcs[i]
                @constraint(model, x[i] == 0)
                println("Arc $i is not interdictable")
            end
        end
    end

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
        D_s = diagm(xi_bar[s])
        Q_hat_s = (xi_bar[s])'*(Φhat_L[s,:,:] - v*Ψhat_L[s,:,:])*(xi_bar[s]) + (Φhat_0[s,:] - v*Ψhat_0[s,:])'*xi_bar[s]
        Mhat_11 = Mhat[s, 1:num_arcs, 1:num_arcs]
        Mhat_12 = Mhat[s, 1:num_arcs, end]
        Mhat_21 = Mhat[s, end, 1:num_arcs]
        Mhat_22 = Mhat[s, end, end]
        @constraint(model, Mhat_11.== ϑhat[s]*Matrix{Float64}(I, num_arcs, num_arcs) - adjoint(D_s)*(Φhat_L[s,:,:] - v*Ψhat_L[s,:,:]))
        @constraint(model, Mhat_12.== -(1/2)*(Φhat_L[s,:,:]-v*Ψhat_L[s,:,:]-v*Ψhat_L[s,:,:])*xi_bar[s] + adjoint(D_s)*(Φhat_0[s,:]-v*Ψhat_0[s,:]))
        @constraint(model, Mhat_22.== ηhat[s] - (Φhat_0[s,:]-v*Ψhat_0[s,:])'*xi_bar[s] - ϑhat[s]*(epsilon^2))
        Mtilde_11 = Mtilde[s, 1:num_arcs, 1:num_arcs]
        Mtilde_12 = Mtilde[s, 1:num_arcs, end]
        Mtilde_21 = Mtilde[s, end, 1:num_arcs]
        Mtilde_22 = Mtilde[s, end, end]
        # @constraint(model, ηtilde[s] >= 0.0) # TODO:: 지우기
        @constraint(model, Mtilde_11.== ϑtilde[s]*Matrix{Float64}(I, num_arcs, num_arcs) - adjoint(D_s)*(Φtilde_L[s,:,:] - v*Ψtilde_L[s,:,:]))
        @constraint(model, Mtilde_12.== -(1/2)*(Φtilde_L[s,:,:]-v*Ψtilde_L[s,:,:])*xi_bar[s] + adjoint(D_s)*(Φtilde_0[s,:]-v*Ψtilde_0[s,:])-Yts_tilde_L[s,1,:].data)
        @constraint(model, Mtilde_22.== ηtilde[s] + Yts_tilde_0[s] - Q_tilde_s - ϑtilde[s]*(epsilon^2))
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
        @constraint(model, sum(μtilde[s,k] + μhat[s,k] for s in 1:S) <= nu)
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
        D_s = diagm(xi_bar[s])
        # =====================================================================
        # Leader's Lambda_hat1 constraint 1: Λˆs_1 * R = [Qˆs; Πˆs; Φˆs]
        # =====================================================================
        Q_hat = adjoint(N) * Πhat_L[s, :, :] + adjoint(I_0) * Φhat_L[s, :, :]
        lhs_mat = vcat(Q_hat, Πhat_L[s, :, :], Φhat_L[s, :, :])
        @constraint(model, Λhat1[s, :, :] * R[s] - lhs_mat .== 0.0) ##TODO:: 주석 풀기
        # =====================================================================
        # Leader's Lambda_hat1 constraint 2
        # Λˆs_1 * r̄ ≥ [d0; 0; 0]
        rhs_vec = vcat(d0-adjoint(N)*Πhat_0[s, :]-adjoint(I_0)*Φhat_0[s, :], -Πhat_0[s,:], -Φhat_0[s,:])
        @constraint(model, Λhat1[s, :, :] * r_dict[s] .>= rhs_vec) ##TODO:: 주석 풀기
        # =====================================================================
        # Leader's Lambda_hat2 constraint: Λˆs_2 * R = -Φˆs
        # =====================================================================
        @constraint(model, Λhat2[s, :, :] * R[s] .== -Φhat_L[s, :, :]) ##TODO:: 주석 풀기
        # Λˆs_2 * r̄ ≥ -μˆs + phi_hat_0
        @constraint(model, Λhat2[s, :, :] * r_dict[s] .- Φhat_0[s, :] .+ μhat[s, :] .>= 0.0) ##TODO:: 주석 풀기
        # =====================================================================
        # Follower's Lambda_tilde1 constraint 1: Λ˜s_1 * R = [Q˜s; Π˜s; Φ˜s]
        # =====================================================================
        # Block 1: Q˜s
        Q_tilde_col = adjoint(N) * Πtilde_L[s, :, :] + adjoint(I_0) * Φtilde_L[s, :, :]
        # Block 2: -N_y * Y˜s - N_ts * Y˜s_ts
        block2 = -N_y * Ytilde_L[s, :, :] - N_ts * Yts_tilde_L[s, :,:]
        # Block 3: -Y˜s
        block3 = -Ytilde_L[s, :, :] + diagm(λ*nu*ones(num_arcs)- v*ψ0)*D_s
        # Block 4: Π˜s
        block4 = Πtilde_L[s, :, :]
        # Block 5: Φ˜s
        block5 = Φtilde_L[s, :, :]
        # Block 6: Y˜s
        block6 = Ytilde_L[s, :, :]
        lhs_mat = vcat(Q_tilde_col, block2, block3, block4, block5, block6)
        @constraint(model, Λtilde1[s, :, :] * R[s] .- lhs_mat .== 0.0) ##TODO:: 주석 풀기
        # Λ˜s_1 * r̄ ≥ [λ*d0; 0; -h; 0; 0; 0]
        rhs_vec_1 =  λ*d0 - adjoint(N)*Πtilde_0[s, :] - adjoint(I_0)*Φtilde_0[s, :]
        rhs_vec_2 = N_y * Ytilde_0[s,:] + N_ts * Yts_tilde_0[s]
        rhs_vec_3 = -h + Ytilde_0[s,:] - diagm(λ*ones(num_arcs)- v*ψ0)*xi_bar[s]
        rhs_vec_4 = -Πtilde_0[s,:]
        rhs_vec_5 = -Φtilde_0[s,:]
        rhs_vec_6 = -Ytilde_0[s,:]
        rhs_vec = vcat(rhs_vec_1, rhs_vec_2, rhs_vec_3, rhs_vec_4, rhs_vec_5, rhs_vec_6)
        
        @constraint(model, Λtilde1[s, :, :] * r_dict[s] .>= rhs_vec) ##TODO:: 주석 풀기
        # 
        # =====================================================================
        # (14p) Follower's capacity dual: Λ˜s_2 * R = -Φ˜s
        # =====================================================================
        @constraint(model, Λtilde2[s, :, :] * R[s] + Φtilde_L[s, :, :] .== 0.0) ##TODO:: 주석 풀기
        # Λ˜s_2 * r̄ ≥ -μ˜s
        @constraint(model, Λtilde2[s, :, :] * r_dict[s] - Φtilde_0[s, :] + μtilde[s, :] .>= 0.0) ##TODO:: 주석 풀기
    end
    println("  ✓ Dual constraints (14m-14p) added for all scenarios")
        # 
    # --- (14q) Linearization constraints for ψ0 ---
    # These linearize the product λ * x
    if isnothing(λ_fixed)
        for k in 1:num_arcs
            @constraint(model, ψ0[k] <= λU * x[k])
            @constraint(model, ψ0[k] <= λ)
            @constraint(model, ψ0[k] >= λ - λU * (1 - x[k]))
            @constraint(model, ψ0[k] >= 0)
        end
    end

    println("  ✓ Linearization constraints (14q) added")

    # =========================================================================
    # Return model and variables
    # =========================================================================
    vars = Dict(
        # Scalar
        # :t => t,
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
        :Λtilde2 => Λtilde2,
        # SDP matrices
        :Mhat => Mhat,
        :Mtilde => Mtilde,
    )

    println("\nModel construction summary:")
    println("  - Variables: $(num_variables(model))")
    println("  - Constraints: $(num_constraints(model, AffExpr, MOI.LessThan{Float64}) + 
                                 num_constraints(model, AffExpr, MOI.EqualTo{Float64}))")
    if x_fixed === nothing
        println("  - Binary variables: $(sum(is_binary(x[i]) for i in 1:num_arcs))")
    end
        println("="^80)

    return model, vars
end

function add_sparsity_constraints!(model, vars, network, S)
    """
    Add sparsity constraints to LDR coefficient matrices based on network adjacency structure.

    This enforces that:
    - Φ_L[s,i,j] = 0 if arcs i and j are NOT adjacent
    - Π_L[s,i,j] = 0 if node i is NOT incident to arc j
    - Similar for Ψ, Y matrices

    # Arguments
    - `model`: JuMP model
    - `vars`: Dictionary containing decision variables
    - `network`: GridNetworkData structure with adjacency information
    - `S`: Number of scenarios

    # Implementation Note
    These constraints reduce the problem dimension by enforcing that LDR coefficients
    are only non-zero for "nearby" arcs/nodes, which is physically reasonable since
    decisions at one location should primarily depend on uncertainty at nearby locations.
    """
    num_arcs = length(network.arcs) - 1  # Exclude dummy arc
    num_nodes = length(network.nodes)

    println("\n" * "="^80)
    println("ADDING SPARSITY CONSTRAINTS")
    println("="^80)

    # Extract LDR coefficient matrices from vars
    Φhat_L = vars[:Φhat][:, :, 1:num_arcs]
    Φtilde_L = vars[:Φtilde][:, :, 1:num_arcs]
    Ψhat_L = vars[:Ψhat][:, :, 1:num_arcs]
    Ψtilde_L = vars[:Ψtilde][:, :, 1:num_arcs]
    Πhat_L = vars[:Πhat][:, :, 1:num_arcs]
    Πtilde_L = vars[:Πtilde][:, :, 1:num_arcs]
    Ytilde_L = vars[:Ytilde][:, :, 1:num_arcs]
    Yts_tilde_L = vars[:Yts_tilde][:, :, 1:num_arcs]

    # Count constraints
    num_arc_sparsity = 0
    num_node_sparsity = 0

    # =========================================================================
    # Arc-to-arc sparsity: Φ, Ψ, Y matrices
    # =========================================================================
    println("\nAdding arc-to-arc sparsity constraints...")

    for s in 1:S, i in 1:num_arcs, j in 1:num_arcs
        if !network.arc_adjacency[i,j]
            # Φhat
            @constraint(model, Φhat_L[s,i,j] == 0)
            # Φtilde
            @constraint(model, Φtilde_L[s,i,j] == 0)
            # Ψhat
            @constraint(model, Ψhat_L[s,i,j] == 0)
            # Ψtilde
            @constraint(model, Ψtilde_L[s,i,j] == 0)
            # Ytilde
            @constraint(model, Ytilde_L[s,i,j] == 0)
            num_arc_sparsity += 5  # 5 constraints per (s,i,j)
        end
    end

    # =========================================================================
    # Node-to-arc sparsity: Π matrices
    # =========================================================================
    println("Adding node-to-arc sparsity constraints...")

    for s in 1:S, i in 1:num_nodes-1, j in 1:num_arcs
        if !network.node_arc_incidence[i,j]
            # Πhat
            @constraint(model, Πhat_L[s,i,j] == 0)
            # Πtilde
            @constraint(model, Πtilde_L[s,i,j] == 0)
            
            num_node_sparsity += 2  # 2 constraints per (s,i,j)
        end
    end

    # =========================================================================
    # Summary
    # =========================================================================
    total_sparsity = num_arc_sparsity + num_node_sparsity

    println("\n" * "="^80)
    println("SPARSITY CONSTRAINTS SUMMARY")
    println("="^80)
    println("Arc-to-arc constraints (Φ, Ψ, Y):  $num_arc_sparsity")
    println("Node-to-arc constraints (Π):       $num_node_sparsity")
    println("Total sparsity constraints:        $total_sparsity")
    println("="^80 * "\n")

    return nothing
end