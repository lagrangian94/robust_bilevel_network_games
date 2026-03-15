using JuMP
using LinearAlgebra
using SparseArrays
using Infiltrator
using Pajarito
using Gurobi
using Mosek, MosekTools
using Hypatia, HiGHS

# Load network generator
include("network_generator.jl")
include("compact_ldr_utils.jl")
using .NetworkGenerator

"""
build_full_2DRNDP_model_compact: Compact LDR version of build_full_2DRNDP_model.

Changes from original:
1. LDR 변수 생성 후 fix()로 비인접 항목 고정 (add_sparsity_constraints! 대체)
2. Big-M 제약조건: 비인접 _L 항목 건너뜀 (redundant constraints 제거)
3. add_sparsity_constraints!() 함수 불필요

Same interface as original — drop-in replacement.
"""
function build_full_2DRNDP_model_compact(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_solver=nothing, conic_solver=nothing,
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
        set_optimizer_attribute(model, "MSK_IPAR_LOG", 10)
        set_optimizer_attribute(model, "MSK_IPAR_LOG_PRESOLVE", 1)
        set_optimizer_attribute(model, "MSK_IPAR_LOG_INFEAS_ANA", 1)
        set_optimizer_attribute(model, "MSK_IPAR_INFEAS_REPORT_AUTO", 1)
    end

    println("Building 2DRNDP model (COMPACT LDR)...")
    println("  Nodes: $num_nodes, Arcs: $num_arcs, Scenarios: $S")
    println("  Interdictable arcs: $num_interdictable")
    println("  Dummy arc index: $dummy_arc_idx")
    println("  Parameters: ϕU = $ϕU, γ = $γ, w = $w, v = $v")

    # Print compact LDR statistics
    print_compact_ldr_stats(network)

    # =========================================================================
    # DECISION VARIABLES
    # =========================================================================

    # --- Scalar variables ---
    @variable(model, nu>= 0)
    if isnothing(λ_fixed)
        @variable(model, λ, lower_bound=0.0, upper_bound=ϕU)  # λ ≤ ϕU: LDR P-bound 조건
    else
        λ=λ_fixed
    end
    # --- Vector variables ---
    if isnothing(x_fixed)
        @variable(model, x[1:num_arcs], Bin)
    else
        x=x_fixed
    end
    if isnothing(h_fixed)
        @variable(model, h[1:num_arcs] >= 0)
    else
        h=h_fixed
    end
    if isnothing(ψ0_fixed)
        @variable(model, ψ0[1:num_arcs] >= 0)
    else
        ψ0=ψ0_fixed
    end

    # --- Scenario-indexed variables ---
    @variable(model, ηhat[1:S]>=0)
    @variable(model, ηtilde[1:S])

    @variable(model, μhat[1:S, 1:num_arcs]>=0)
    @variable(model, μtilde[1:S, 1:num_arcs]>=0)

    # --- LDR coefficient matrices ---
    @variable(model, Φhat[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound= -ϕU, upper_bound = ϕU)
    @variable(model, Ψhat[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound= 0.0)
    @variable(model, Φtilde[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound= -ϕU, upper_bound = ϕU)
    @variable(model, Ψtilde[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound= 0.0)

    @variable(model, Πhat[s=1:S, 1:num_nodes-1, 1:num_arcs+1], lower_bound= -ϕU, upper_bound = ϕU)
    @variable(model, Πtilde[s=1:S, 1:num_nodes-1, 1:num_arcs+1], lower_bound= -ϕU, upper_bound = ϕU)

    @variable(model, Ytilde[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound= -ϕU, upper_bound = ϕU)
    @variable(model, Yts_tilde[s=1:S, 1, 1:num_arcs+1], lower_bound= -ϕU, upper_bound = ϕU)

    # =========================================================================
    # COMPACT LDR: fix() 비인접 항목 → solver presolve에서 제거
    # add_sparsity_constraints!() 대체
    # =========================================================================
    vars_for_fix = Dict(
        :Φhat => Φhat, :Φtilde => Φtilde,
        :Ψhat => Ψhat, :Ψtilde => Ψtilde,
        :Ytilde => Ytilde,
        :Πhat => Πhat, :Πtilde => Πtilde,
    )
    apply_primal_ldr_sparsity!(vars_for_fix, network, S)

    # 변수 따로 정리
    Φhat_L, Ψhat_L, Φtilde_L, Ψtilde_L = Φhat[:,:,1:num_arcs], Ψhat[:,:,1:num_arcs], Φtilde[:,:,1:num_arcs], Ψtilde[:,:,1:num_arcs]
    Πhat_L, Πtilde_L = Πhat[:,:,1:num_arcs], Πtilde[:,:,1:num_arcs]
    Ytilde_L, Yts_tilde_L =  Ytilde[:,:,1:num_arcs], Yts_tilde[:,:,1:num_arcs]
    Φhat_0, Ψhat_0, Φtilde_0, Ψtilde_0 = Φhat[:,:,num_arcs+1], Ψhat[:,:,num_arcs+1], Φtilde[:,:,num_arcs+1], Ψtilde[:,:,num_arcs+1]
    Πhat_0, Πtilde_0 = Πhat[:,:,num_arcs+1], Πtilde[:,:,num_arcs+1]
    Ytilde_0, Yts_tilde_0 =  Ytilde[:,:,num_arcs+1], Yts_tilde[:,1,num_arcs+1]

    # --- Dual variables for inner problems ---
    dim_Λhat1_rows = num_arcs+1 +(num_nodes - 1) + num_arcs
    @variable(model, Λhat1[s=1:S, 1:dim_Λhat1_rows, 1:size(R[1], 1)])
    @variable(model, Λhat2[s=1:S, 1:num_arcs, 1:size(R[1], 1)])

    dim_Λtilde1_rows = num_arcs+1 + (num_nodes - 1) + num_arcs + num_nodes-1 + num_arcs + num_arcs
    @variable(model, Λtilde1[s=1:S, 1:dim_Λtilde1_rows, 1:size(R[1], 1)])
    @variable(model, Λtilde2[s=1:S, 1:num_arcs, 1:size(R[1], 1)])

    # Second order cone constraints
    @constraint(model, [s=1:S, i=1:dim_Λhat1_rows], Λhat1[s, i, :] in SecondOrderCone())
    @constraint(model, [s=1:S, i=1:num_arcs], Λhat2[s, i, :] in SecondOrderCone())
    @constraint(model, [s=1:S, i=1:dim_Λtilde1_rows], Λtilde1[s, i, :] in SecondOrderCone())
    @constraint(model, [s=1:S, i=1:num_arcs], Λtilde2[s, i, :] in SecondOrderCone())

    println("  ✓ Decision variables created (with compact LDR sparsity)")

    # =========================================================================
    # CONSTRAINTS
    # =========================================================================

    # --- (14a) Objective function ---
    @objective(model, Min, (1/S)*sum(ηhat[s] + ηtilde[s] for s in 1:S) + (1/S)*w * nu)

    # --- (14b) Initial resource and domain constraints ---
    if isnothing(λ_fixed)
        @constraint(model, resource_budget, sum(h) <= λ * w)
        @constraint(model, sum(x) <= γ)
        for i in 1:num_arcs
            if !network.interdictable_arcs[i]
                @constraint(model, x[i] == 0)
                println("Arc $i is not interdictable")
            end
        end
    end

    println("  ✓ Constraints (14a-14c) added")

    @variable(model, ϑhat[1:S]>=0)
    @variable(model, ϑtilde[1:S]>=0)

    @variable(model, Mhat[1:S, 1:num_arcs+1, 1:num_arcs+1])
    @variable(model, Mtilde[1:S, 1:num_arcs+1, 1:num_arcs+1])

    # --- SDP constraints (14d, 14g) ---
    for s in 1:S
        D_s = diagm(xi_bar[s])
        Q_hat_s = (xi_bar[s])'*(Φhat_L[s,:,:] - v*Ψhat_L[s,:,:])*(xi_bar[s]) + (Φhat_0[s,:] - v*Ψhat_0[s,:])'*xi_bar[s]
        Mhat_11 = Mhat[s, 1:num_arcs, 1:num_arcs]
        Mhat_12 = Mhat[s, 1:num_arcs, end]
        Mhat_21 = Mhat[s, end, 1:num_arcs]
        Mhat_22 = Mhat[s, end, end]
        @constraint(model, Mhat_11.== ϑhat[s]*Matrix{Float64}(I, num_arcs, num_arcs) - adjoint(D_s)*(Φhat_L[s,:,:] - v*Ψhat_L[s,:,:]))
        @constraint(model, Mhat_12.== -(1/2)*((Φhat_L[s,:,:]-v*Ψhat_L[s,:,:])*xi_bar[s] + adjoint(D_s)*(Φhat_0[s,:]-v*Ψhat_0[s,:])))
        @constraint(model, Mhat_22.== ηhat[s] - (Φhat_0[s,:]-v*Ψhat_0[s,:])'*xi_bar[s] - ϑhat[s]*(epsilon^2))
        Mtilde_11 = Mtilde[s, 1:num_arcs, 1:num_arcs]
        Mtilde_12 = Mtilde[s, 1:num_arcs, end]
        Mtilde_21 = Mtilde[s, end, 1:num_arcs]
        Mtilde_22 = Mtilde[s, end, end]
        @constraint(model, Mtilde_11.== ϑtilde[s]*Matrix{Float64}(I, num_arcs, num_arcs) - adjoint(D_s)*(Φtilde_L[s,:,:] - v*Ψtilde_L[s,:,:]))
        @constraint(model, Mtilde_12.== -(1/2)*((Φtilde_L[s,:,:]-v*Ψtilde_L[s,:,:])*xi_bar[s] + adjoint(D_s)*(Φtilde_0[s,:]-v*Ψtilde_0[s,:])-Yts_tilde_L[s,1,:].data))
        @constraint(model, Mtilde_22.== ηtilde[s] -(Φtilde_0[s,:]-v*Ψtilde_0[s,:])'*xi_bar[s] + Yts_tilde_0[s] - ϑtilde[s]*(epsilon^2))
    end

    @constraint(model, [s=1:S], Mhat[s,:,:] in PSDCone())
    @constraint(model, [s=1:S], Mtilde[s,:,:] in PSDCone())

    # =========================================================================
    # COMPACT Big-M constraints (14j, 14k)
    # 비인접 _L 항목은 이미 fix(0)이므로 Big-M 제약 불필요 (redundant)
    # intercept 열 (j=num_arcs+1)은 항상 포함
    # =========================================================================
    println("  Adding Big-M constraints (14j, 14k) [COMPACT]...")

    num_bigm_created = 0
    for s in 1:S
        for i in 1:num_arcs
            for j in 1:num_arcs+1
                # 비인접 _L 항목 건너뜀 (fix(0)이므로 제약이 자명)
                if j <= num_arcs && !network.arc_adjacency[i,j]
                    continue
                end

                # Leader constraints (14j)
                @constraint(model, Ψhat[s,i,j] <= ϕU * x[i])
                @constraint(model, Ψhat[s,i,j] - Φhat[s,i,j] <= 0)
                @constraint(model, Φhat[s,i,j] - Ψhat[s,i,j] <= ϕU * (1 - x[i]))

                # Follower constraints (14k)
                @constraint(model, Ψtilde[s,i,j] <= ϕU * x[i])
                @constraint(model, Ψtilde[s,i,j] - Φtilde[s,i,j] <= 0)
                @constraint(model, Φtilde[s,i,j] - Ψtilde[s,i,j] <= ϕU * (1 - x[i]))

                num_bigm_created += 6
            end
        end
    end

    num_bigm_full = 6 * S * num_arcs * (num_arcs + 1)
    println("  ✓ Big-M constraints: $num_bigm_created created (full would be $num_bigm_full)")
    println("    Saved: $(num_bigm_full - num_bigm_created) constraints ($(round(100*(num_bigm_full - num_bigm_created)/num_bigm_full, digits=1))%)")

    # --- (14l) Budget constraint ---
    for k in 1:num_arcs
        @constraint(model, sum(μtilde[s,k] + μhat[s,k] for s in 1:S) <= nu)
    end

    println("  ✓ Budget constraint (14l) added")

    # --- (14m-14p) Dual feasibility constraints ---
    println("  Adding dual constraints (14m-14p)...")
    d0 = zeros(num_arcs + 1)
    d0[end] = 1.0
    I_0 = [Matrix{Float64}(I, num_arcs, num_arcs) zeros(num_arcs)]
    N_y = N[:, 1:num_arcs]
    N_ts = N[:, end]

    for s in 1:S
        D_s = diagm(xi_bar[s])
        # Leader's Lambda_hat1 constraint 1
        Q_hat = adjoint(N) * Πhat_L[s, :, :] + adjoint(I_0) * Φhat_L[s, :, :]
        lhs_mat = vcat(Q_hat, Πhat_L[s, :, :], Φhat_L[s, :, :])
        @constraint(model, Λhat1[s, :, :] * R[s] - lhs_mat .== 0.0)
        # Leader's Lambda_hat1 constraint 2
        rhs_vec = vcat(d0-adjoint(N)*Πhat_0[s, :]-adjoint(I_0)*Φhat_0[s, :], -Πhat_0[s,:], -Φhat_0[s,:])
        @constraint(model, Λhat1[s, :, :] * r_dict[s] .>= rhs_vec)
        # Leader's Lambda_hat2 constraint
        @constraint(model, Λhat2[s, :, :] * R[s] .== -Φhat_L[s, :, :])
        @constraint(model, Λhat2[s, :, :] * r_dict[s] .- Φhat_0[s, :] .+ μhat[s, :] .>= 0.0)
        # Follower's Lambda_tilde1 constraint 1
        Q_tilde_col = adjoint(N) * Πtilde_L[s, :, :] + adjoint(I_0) * Φtilde_L[s, :, :]
        block2 = -N_y * Ytilde_L[s, :, :] - N_ts * Yts_tilde_L[s, :,:]
        block3 = -Ytilde_L[s, :, :] + diagm(λ*ones(num_arcs)- v*ψ0)*D_s
        block4 = Πtilde_L[s, :, :]
        block5 = Φtilde_L[s, :, :]
        block6 = Ytilde_L[s, :, :]
        rhs_mat = vcat(Q_tilde_col, block2, block3, block4, block5, block6)
        @constraint(model, Λtilde1[s, :, :] * R[s] .== rhs_mat)
        # Follower's Lambda_tilde1 constraint 2
        rhs_vec_1 =  λ*d0 - adjoint(N)*Πtilde_0[s, :] - adjoint(I_0)*Φtilde_0[s, :]
        rhs_vec_2 = N_y * Ytilde_0[s,:] + N_ts * Yts_tilde_0[s]
        rhs_vec_3 = -h + Ytilde_0[s,:] - diagm(λ*ones(num_arcs)- v*ψ0)*xi_bar[s]
        rhs_vec_4 = -Πtilde_0[s,:]
        rhs_vec_5 = -Φtilde_0[s,:]
        rhs_vec_6 = -Ytilde_0[s,:]
        rhs_vec = vcat(rhs_vec_1, rhs_vec_2, rhs_vec_3, rhs_vec_4, rhs_vec_5, rhs_vec_6)
        @constraint(model, Λtilde1[s, :, :] * r_dict[s] .>= rhs_vec)
        # Follower's Lambda_tilde2 constraint
        @constraint(model, Λtilde2[s, :, :] * R[s] + Φtilde_L[s, :, :] .== 0.0)
        @constraint(model, Λtilde2[s, :, :] * r_dict[s] - Φtilde_0[s, :] + μtilde[s, :] .>= 0.0)
    end
    println("  ✓ Dual constraints (14m-14p) added for all scenarios")

    # --- (14q) Linearization constraints for ψ0 ---
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
        :nu => nu, :λ => λ,
        :x => x, :h => h, :ψ0 => ψ0,
        :ηhat => ηhat, :ηtilde => ηtilde,
        :μhat => μhat, :μtilde => μtilde,
        :Φhat => Φhat, :Ψhat => Ψhat,
        :Φtilde => Φtilde, :Ψtilde => Ψtilde,
        :Πhat => Πhat, :Πtilde => Πtilde,
        :Ytilde => Ytilde, :Yts_tilde => Yts_tilde,
        :Λhat1 => Λhat1, :Λhat2 => Λhat2,
        :Λtilde1 => Λtilde1, :Λtilde2 => Λtilde2,
        :Mhat => Mhat, :Mtilde => Mtilde,
    )

    println("\nModel construction summary (COMPACT):")
    println("  - Variables: $(num_variables(model))")
    println("  - Constraints: $(num_constraints(model, AffExpr, MOI.LessThan{Float64}) +
                                 num_constraints(model, AffExpr, MOI.EqualTo{Float64}))")
    if x_fixed === nothing
        println("  - Binary variables: $(sum(is_binary(x[i]) for i in 1:num_arcs))")
    end
    println("="^80)

    return model, vars
end
