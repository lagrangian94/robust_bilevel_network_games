using JuMP


"""
RO 말고 2-SP 모형 만들어서 debug. (lambda, tilde quadratic term이 xi(1-v)x-y 가 0이 나오는지 확인)
"""

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
function build_full_2SP_model(network, S, ϕU, λU, γ, w, v, uncertainty_set; optimizer=nothing,
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
    model = Model(Gurobi.Optimizer)


    println("Building 2SP model...")
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

    φhat = [vec(Φhat_L[s,:,:]*xi_bar[s] + Φhat_0[s,:,:]) for s in 1:S]
    ψhat = [vec(Ψhat_L[s,:,:]*xi_bar[s] + Ψhat_0[s,:,:]) for s in 1:S]
    φtilde = [vec(Φtilde_L[s,:,:]*xi_bar[s] + Φtilde_0[s,:,:]) for s in 1:S]
    ψtilde = [vec(Ψtilde_L[s,:,:]*xi_bar[s] + Ψtilde_0[s,:,:]) for s in 1:S]
    πhat = [vec(Πhat_L[s,:,:]*xi_bar[s] + Πhat_0[s,:,:]) for s in 1:S]
    πtilde = [vec(Πtilde_L[s,:,:]*xi_bar[s] + Πtilde_0[s,:,:]) for s in 1:S]
    ytilde = [vec(Ytilde_L[s,:,:]*xi_bar[s] + Ytilde_0[s,:,:]) for s in 1:S]
    yts_tilde = [vec(Yts_tilde_L[s,:,:].data*xi_bar[s] .+ Yts_tilde_0[s]) for s in 1:S]


    """
    아래는 LDR coefficient 0으로 둘 때와 비교. (SP니까 0으로 둬도 objective value는 변하지 않아야 함)
    """
    @constraint(model, Φhat_L[:,:,:] .== 0.0)
    @constraint(model, Ψhat_L[:,:,:] .== 0.0)
    @constraint(model, Φtilde_L[:,:,:] .== 0.0)
    @constraint(model, Ψtilde_L[:,:,:] .== 0.0)
    @constraint(model, Πhat_L[:,:,:] .== 0.0)
    @constraint(model, Πtilde_L[:,:,:] .== 0.0)
    @constraint(model, Ytilde_L[:,:,:] .== 0.0)
    @constraint(model, Yts_tilde_L[:,:,:] .== 0.0)

    @constraint(model, λ.==0.00001 ) ## lambda는 특정 값 이하로는 다 optimal이 나오나? 대충 2이하로는 아무리 작아도 다 optimal이 나오는데... lambda의 ub 찾을수있나?
                                
    # --- Dual variables for inner problems ---
    # These arise from dualizing the inner problems
    # Dimensions are based on the dual formulation structure

    
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
    # xi_bar[s][k]*(Φhat[s,k,:]'*
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
    @constraint(model, epigraph_hat[s=1:S], sum(xi_bar[s][k]*φhat[s][k] - xi_bar[s][k]*v*ψhat[s][k] for k in 1:num_arcs) <= ηhat[s])
    @constraint(model, epigraph_tilde[s=1:S], sum(xi_bar[s][k]*φtilde[s][k] - xi_bar[s][k]*v*ψtilde[s][k] for k in 1:num_arcs) .- yts_tilde[s] .<= ηtilde[s])

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
        # =====================================================================
        # Leader's Lambda_hat1 constraint 1: Λˆs_1 * R = [Qˆs; Πˆs; Φˆs]
        # =====================================================================
        @constraint(model, N_y' * πtilde[s] + φtilde[s] .>= 0.0)
        @constraint(model, N_ts' * πtilde[s] .- λ .>= 0.0)
        @constraint(model, N_y' * πhat[s] + φhat[s] .>= 0.0)
        @constraint(model, N_ts' * πhat[s]  .>= 1.0)

        @constraint(model, [k=1:num_arcs], φhat[s][k] <= μhat[s,k])
        @constraint(model, [k=1:num_arcs], φtilde[s][k] <= μtilde[s,k])
        @constraint(model, N_y* ytilde[s] + N_ts* yts_tilde[s][1] .<= 0.0)

        @constraint(model, [k=1:num_arcs], ytilde[s][k] - h[k] <= xi_bar[s][k]*(λ-v* ψ0[k]))
    end
    @constraint(model, [s=1:S], πhat[s].>=0.0)
    @constraint(model, [s=1:S], πtilde[s].>=0.0)
    @constraint(model, [s=1:S], ytilde[s].>=0.0)
    @constraint(model, [s=1:S], yts_tilde[s].>=0.0)
    @constraint(model, [s=1:S], φhat[s].>=0.0)
    @constraint(model, [s=1:S], φtilde[s].>=0.0)
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
        
        # ldr vars
        :φhat => φhat,
        :ψhat => ψhat,
        :φtilde => φtilde,
        :ψtilde => ψtilde,
        :πhat => πhat,
        :πtilde => πtilde,
        :ytilde => ytilde,
        :yts_tilde => yts_tilde,
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