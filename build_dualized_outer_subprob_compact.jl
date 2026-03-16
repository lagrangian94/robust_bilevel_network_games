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
build_dualized_outer_subproblem_compact: Compact LDR version of build_dualized_outer_subproblem.

Changes from original:
1. U, P dual 변수의 비인접 _L 항목을 fix(0)으로 고정
2. 제약조건 필터링은 원본과 동일 (이미 arc_adjacency/node_arc_incidence 체크)
3. 고정된 변수는 objective에서 0으로 기여 → 목적함수 수정 불필요

Same interface as original — drop-in replacement.
"""
function build_dualized_outer_subproblem_compact(network, S, ϕU, λU, γ, w, v, uncertainty_set, optimizer, λ, x, h, ψ0; πU=ϕU, yU=ϕU, ytsU=ϕU)
    # Extract network dimensions
    num_nodes = length(network.nodes)
    num_arcs = length(network.arcs)-1 #dummy arc 제외
    num_interdictable = sum(network.interdictable_arcs)

    # Node-arc incidence matrix (excluding source row)
    N = network.N
    N_y = N[:, 1:num_arcs]  # Regular arcs: (|V|-1) × |A|
    N_ts = N[:, end]         # Dummy arc: (|V|-1) × 1
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    # Dummy arc index (t,s)
    dummy_arc_idx = findfirst(arc -> arc == ("t", "s"), network.arcs)
    E = ones(num_arcs, num_arcs+1) # num_arcs × num_arcs+1 matrix of ones
    I_0 = [Matrix{Float64}(I, num_arcs, num_arcs) zeros(num_arcs)]
    # Create model
    model = Model(optimizer)
    d0 = zeros(num_arcs + 1)
    d0[end] = 1.0
    println("Building dualized outer subproblem (COMPACT)...")
    println("  Nodes: $num_nodes, Arcs: $num_arcs, Scenarios: $S")
    println("  Interdictable arcs: $num_interdictable")
    println("  Dummy arc index: $dummy_arc_idx")
    println("  Parameters: ϕU = $ϕU, λU = $λU, γ = $γ, w = $w, v = $v")

    # =========================================================================
    # DECISION VARIABLES
    # =========================================================================

    # --- Scalar variables ---
    @variable(model, α[1:num_arcs] >= 0)
    # --- Vector variables ---
    dim_Λhat1_rows = (num_arcs + 1) + (num_nodes - 1) + num_arcs
    dim_Λhat2_rows = num_arcs
    dim_Λtilde1_rows = num_arcs+1 + (num_nodes - 1) + num_arcs + num_nodes-1 + num_arcs + num_arcs
    dim_Λtilde2_rows = num_arcs
    @variable(model, βhat1[s=1:S,1:dim_Λhat1_rows]>=0)
    @variable(model, βhat2[s=1:S,1:dim_Λhat2_rows]>=0)
    @variable(model, βtilde1[s=1:S,1:dim_Λtilde1_rows]>=0)
    @variable(model, βtilde2[s=1:S,1:dim_Λtilde2_rows]>=0)

    βhat1_1 = βhat1[:,1:num_arcs+1]
    block2_start = num_arcs+2
    block3_start = block2_start + num_nodes-1
    βhat1_2 = βhat1[:,block2_start:block3_start-1]
    βhat1_3 = βhat1[:,block3_start:end]
    block2_start, block3_start= -1, -1
    @assert sum([size(βhat1_1,2), size(βhat1_2,2), size(βhat1_3,2)]) == dim_Λhat1_rows

    #βtilde1 block 분리
    block2_start = num_arcs+2
    block3_start = block2_start + num_nodes-1
    block4_start = block3_start + num_arcs
    block5_start = block4_start + num_nodes-1
    block6_start = block5_start + num_arcs
    βtilde1_1 = βtilde1[:,1:num_arcs+1]
    βtilde1_2 = βtilde1[:,block2_start:block3_start-1]
    βtilde1_3 = βtilde1[:,block3_start:block4_start-1]
    βtilde1_4 = βtilde1[:,block4_start:block5_start-1]
    βtilde1_5 = βtilde1[:,block5_start:block6_start-1]
    βtilde1_6 = βtilde1[:,block6_start:end]
    block2_start, block3_start, block4_start, block5_start, block6_start= -1, -1, -1, -1, -1
    @assert sum([size(βtilde1_1,2), size(βtilde1_2,2), size(βtilde1_3,2), size(βtilde1_4,2), size(βtilde1_5,2), size(βtilde1_6,2)]) == dim_Λtilde1_rows

    # --- Matrix variables ---
    @variable(model, Mhat[s=1:S,1:num_arcs+1,1:num_arcs+1])
    @variable(model, Mtilde[s=1:S,1:num_arcs+1,1:num_arcs+1])
    @variable(model, Uhat1[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    @variable(model, Uhat2[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    @variable(model, Uhat3[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    @variable(model, Utilde1[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    @variable(model, Utilde2[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    @variable(model, Utilde3[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    dim_R_cols = size(R[1],2)
    @variable(model, Zhat1[s=1:S,1:dim_Λhat1_rows,1:dim_R_cols])
    @variable(model, Zhat2[s=1:S,1:dim_Λhat2_rows,1:dim_R_cols])
    @variable(model, Ztilde1[s=1:S,1:dim_Λtilde1_rows,1:dim_R_cols])
    @variable(model, Ztilde2[s=1:S,1:dim_Λtilde2_rows,1:dim_R_cols])

    # Zhat1 블록 분리
    Zhat1_1 = Zhat1[:,1:num_arcs+1,:]
    block2_start = num_arcs+2
    block3_start = block2_start + num_nodes-1
    Zhat1_2 = Zhat1[:,block2_start:block3_start-1,:]
    Zhat1_3 = Zhat1[:,block3_start:end,:]
    block2_start, block3_start= -1, -1
    @assert sum([size(Zhat1_1,2), size(Zhat1_2,2), size(Zhat1_3,2)]) == dim_Λhat1_rows

    # Ztilde1 블록 분리
    block2_start = num_arcs+2
    block3_start = block2_start + num_nodes-1
    block4_start = block3_start + num_arcs
    block5_start = block4_start + num_nodes-1
    block6_start = block5_start + num_arcs
    Ztilde1_1 = Ztilde1[:,1:num_arcs+1,:]
    Ztilde1_2 = Ztilde1[:,block2_start:block3_start-1,:]
    Ztilde1_3 = Ztilde1[:,block3_start:block4_start-1,:]
    Ztilde1_4 = Ztilde1[:,block4_start:block5_start-1,:]
    Ztilde1_5 = Ztilde1[:,block5_start:block6_start-1,:]
    Ztilde1_6 = Ztilde1[:,block6_start:end,:]
    block2_start, block3_start, block4_start, block5_start, block6_start= -1, -1, -1, -1, -1
    @assert sum([size(Ztilde1_1,2), size(Ztilde1_2,2), size(Ztilde1_3,2), size(Ztilde1_4,2), size(Ztilde1_5,2), size(Ztilde1_6,2)]) == dim_Λtilde1_rows

    @variable(model, Γhat1[s=1:S, 1:dim_Λhat1_rows, 1:size(R[1],1)])
    @variable(model, Γhat2[s=1:S, 1:dim_Λhat2_rows, 1:size(R[1],1)])
    @variable(model, Γtilde1[s=1:S, 1:dim_Λtilde1_rows, 1:size(R[1],1)])
    @variable(model, Γtilde2[s=1:S, 1:dim_Λtilde2_rows, 1:size(R[1],1)])

    @variable(model, Phat1_Φ[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Phat1_Π[s=1:S, 1:num_nodes-1, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Phat2_Φ[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Phat2_Π[s=1:S, 1:num_nodes-1, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Ptilde1_Φ[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Ptilde1_Π[s=1:S, 1:num_nodes-1, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Ptilde2_Φ[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Ptilde2_Π[s=1:S, 1:num_nodes-1, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Ptilde1_Y[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Ptilde2_Y[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Ptilde1_Yts[s=1:S, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Ptilde2_Yts[s=1:S, 1:num_arcs+1], lower_bound=0.0)

    # =========================================================================
    # COMPACT: fix() 비인접 U, P dual 변수
    # =========================================================================
    apply_dual_ldr_sparsity!(model, network, S; has_hat=true, has_tilde=true)

    # =========================================================================
    # OBJECTIVE FUNCTION
    # =========================================================================
    diag_x_E = Diagonal(x) * E
    diag_λ_ψ = Diagonal(λ*ones(num_arcs)-v.*ψ0)
    obj_term1 = [-ϕU * sum((Uhat1[s, :, :] + Utilde1[s, :, :]) .* diag_x_E) for s=1:S]
    obj_term2 = [-ϕU * sum((Uhat3[s, :, :] + Utilde3[s, :, :]) .* (E-diag_x_E)) for s=1:S]
    obj_term3 = [(d0')* βhat1_1[s,:] for s=1:S]
    obj_term4 = [sum(Ztilde1_3[s, :, :] .* (diag_λ_ψ * diagm(xi_bar[s]))) for s=1:S]
    obj_term5 = [(λ*d0')* βtilde1_1[s,:] for s=1:S]
    obj_term6 = [-(h+diag_λ_ψ*xi_bar[s])'* βtilde1_3[s,:] for s=1:S]
    obj_term_ub_hat = [-ϕU * sum(Phat1_Φ[s,:,:]) - πU * sum(Phat1_Π[s,:,:]) for s=1:S]
    obj_term_lb_hat = [-ϕU * sum(Phat2_Φ[s,:,:]) - πU * sum(Phat2_Π[s,:,:]) for s=1:S]
    obj_term_ub_tilde = [-ϕU * sum(Ptilde1_Φ[s,:,:]) - πU * sum(Ptilde1_Π[s,:,:]) - yU * sum(Ptilde1_Y[s,:,:]) - ytsU * sum(Ptilde1_Yts[s,:]) for s=1:S]
    obj_term_lb_tilde = [-ϕU * sum(Ptilde2_Φ[s,:,:]) - πU * sum(Ptilde2_Π[s,:,:]) - yU * sum(Ptilde2_Y[s,:,:]) - ytsU * sum(Ptilde2_Yts[s,:]) for s=1:S]
    @objective(model, Max, sum(obj_term1) + sum(obj_term2) + sum(obj_term3) + sum(obj_term4) + sum(obj_term5) + sum(obj_term6)
    + sum(obj_term_ub_hat) + sum(obj_term_lb_hat) + sum(obj_term_ub_tilde) + sum(obj_term_lb_tilde))

    # =========================================================================
    # CONSTRAINTS (same as original — constraint filtering already in place)
    # =========================================================================

    # --- Semi-definite cone constraints ---
    @constraint(model, [s=1:S], Mhat[s,:,:] in PSDCone())
    @constraint(model, [s=1:S], Mtilde[s,:,:] in PSDCone())
    # --- Second order cone constraints ---
    @constraint(model, [s=1:S, i=1:dim_Λhat1_rows], Γhat1[s, i, :] in SecondOrderCone())
    @constraint(model, [s=1:S, i=1:dim_Λhat2_rows], Γhat2[s, i, :] in SecondOrderCone())
    @constraint(model, [s=1:S, i=1:dim_Λtilde1_rows], Γtilde1[s, i, :] in SecondOrderCone())
    @constraint(model, [s=1:S, i=1:dim_Λtilde2_rows], Γtilde2[s, i, :] in SecondOrderCone())

    # Scalar constraints
    @constraint(model, [s=1:S], Mhat[s, num_arcs+1, num_arcs+1] <= 1/S)
    @constraint(model, [s=1:S], Mtilde[s, num_arcs+1, num_arcs+1] == 1/S)
    @constraint(model, sum(α) <= w*(1/S))
    @constraint(model, [s=1:S], tr(Mhat[s, 1:num_arcs, 1:num_arcs]) - Mhat[s,end,end]*(epsilon^2) <= 0)
    @constraint(model, [s=1:S], tr(Mtilde[s, 1:num_arcs, 1:num_arcs]) - Mtilde[s,end,end]*(epsilon^2) <= 0)

    # --- Matrix Constraints ---
    for s in 1:S
        D_s = diagm(xi_bar[s])
        # --- From Φhat ---
        Mhat_11 = Mhat[s, 1:num_arcs, 1:num_arcs]
        Mhat_12 = Mhat[s, 1:num_arcs, end]
        Mhat_22 = Mhat[s, end, end]
        Mtilde_11 = Mtilde[s, 1:num_arcs, 1:num_arcs]
        Mtilde_12 = Mtilde[s, 1:num_arcs, end]
        Mtilde_22 = Mtilde[s, end, end]
        Adj_L_Mhat_11 = -D_s*Mhat_11
        Adj_L_Mhat_12 = -Mhat_12*adjoint(xi_bar[s])

        Adj_0_Mhat_12 = -D_s * Mhat_12
        Adj_0_Mhat_22 = -xi_bar[s] * Mhat_22

        ## Φhat_L constraint
        lhs_L = Adj_L_Mhat_11+Adj_L_Mhat_12 + Uhat2[s,:,1:num_arcs] - Uhat3[s,:,1:num_arcs]
        -I_0*Zhat1_1[s,:,:] - Zhat1_3[s,:,:] + Zhat2[s,:,:] + Phat1_Φ[s,:,1:num_arcs] - Phat2_Φ[s,:,1:num_arcs]

        for i in 1:num_arcs, j in 1:num_arcs
            if network.arc_adjacency[i,j]
                @constraint(model, lhs_L[i,j] == 0)
            end
        end
        ## Φhat_0 constraint
        @constraint(model, Adj_0_Mhat_12+Adj_0_Mhat_22 + Uhat2[s,:,end] - Uhat3[s,:,end] + I_0*βhat1_1[s,:] + βhat1_3[s,:] - βhat2[s,:] + Phat1_Φ[s,:,end] - Phat2_Φ[s,:,end] .== 0)

        # --- From Ψhat
        Adj_L_Mhat_11 = v*D_s*Mhat_11
        Adj_L_Mhat_12 = v*Mhat_12*adjoint(xi_bar[s])

        Adj_0_Mhat_12 = v*D_s * Mhat_12
        Adj_0_Mhat_22 = xi_bar[s] * Mhat_22 * v
        ## Ψhat_L constraint
        lhs_L = Adj_L_Mhat_11+Adj_L_Mhat_12 -Uhat1[s,:,1:num_arcs] - Uhat2[s,:,1:num_arcs] + Uhat3[s,:,1:num_arcs]
        for i in 1:num_arcs, j in 1:num_arcs
            if network.arc_adjacency[i,j]
                @constraint(model, lhs_L[i,j] <= 0)
            end
        end
        ## Ψhat_0 constraint
        @constraint(model, Adj_0_Mhat_12+Adj_0_Mhat_22 - Uhat1[s,:,end] - Uhat2[s,:,end] + Uhat3[s,:,end] .<= 0.0)

        # --- From Φtilde ---
        Adj_L_Mtilde_11 = -D_s*Mtilde_11
        Adj_L_Mtilde_12 = -Mtilde_12*adjoint(xi_bar[s])

        Adj_0_Mtilde_12 = -D_s * Mtilde_12
        Adj_0_Mtilde_22 = -xi_bar[s] * Mtilde_22
        # --- Φtilde_L constraint
        lhs_L = Adj_L_Mtilde_11+Adj_L_Mtilde_12 + Utilde2[s,:,1:num_arcs] - Utilde3[s,:,1:num_arcs]
        -I_0*Ztilde1_1[s,:,:] - Ztilde1_5[s,:,:] + Ztilde2[s,:,:] + Ptilde1_Φ[s,:,1:num_arcs] - Ptilde2_Φ[s,:,1:num_arcs]
        for i in 1:num_arcs, j in 1:num_arcs
            if network.arc_adjacency[i,j]
                @constraint(model, lhs_L[i,j] == 0)
            end
        end
        # --- Φtilde_0 constraint
        @constraint(model, Adj_0_Mtilde_12+Adj_0_Mtilde_22 + Utilde2[s,:,end] - Utilde3[s,:,end] + I_0*βtilde1_1[s,:] + βtilde1_5[s,:] - βtilde2[s,:] + Ptilde1_Φ[s,:,end] - Ptilde2_Φ[s,:,end] .== 0)

        # --- From Ψtilde ---
        Adj_L_Mtilde_11 = v*D_s*Mtilde_11
        Adj_L_Mtilde_12 = v*(Mtilde_12*adjoint(xi_bar[s]))

        Adj_0_Mtilde_12 = v*D_s * Mtilde_12
        Adj_0_Mtilde_22 = v*xi_bar[s] * Mtilde_22
        # --- Ψtilde_L constraint
        lhs_L = Adj_L_Mtilde_11+Adj_L_Mtilde_12 - Utilde1[s,:,1:num_arcs] - Utilde2[s,:,1:num_arcs] + Utilde3[s,:,1:num_arcs]
        for i in 1:num_arcs, j in 1:num_arcs
            if network.arc_adjacency[i,j]
                @constraint(model, lhs_L[i,j] <= 0.0)
            end
        end
        # --- Ψtilde_0 constraint
        @constraint(model, Adj_0_Mtilde_12+Adj_0_Mtilde_22 - Utilde1[s,:,end] - Utilde2[s,:,end] + Utilde3[s,:,end] .<= 0.0)
        # --- From Ytilde_ts ---
        Adj_L_Mtilde_12 = Mtilde_12

        Adj_0_Mtilde_22 = Mtilde_22
        # --- Ytilde_ts_L constraint
        @constraint(model, adjoint(Adj_L_Mtilde_12) + N_ts' * Ztilde1_2[s,:,:] + Ptilde1_Yts[s,1:num_arcs]' - Ptilde2_Yts[s,1:num_arcs]' .== 0)
        # --- Ytilde_ts_0 constraint
        @constraint(model, Adj_0_Mtilde_22 - N_ts' * βtilde1_2[s,:] + Ptilde1_Yts[s,end]' - Ptilde2_Yts[s,end]' .== 0)
    end
    # --- From μhat ---
    @constraint(model, [s=1:S, k=1:num_arcs], βhat2[s,k] <= α[k])
    # --- From μtilde ---
    @constraint(model, [s=1:S, k=1:num_arcs], βtilde2[s,k] <= α[k])
    # --- From Πhat ---
    # --- Πhat_L constraint
    for i in 1:(num_nodes-1), j in 1:num_arcs
        if network.node_arc_incidence[i,j]
            @constraint(model, [s=1:S], (-N*Zhat1_1[s,:,:])[i,j]-Zhat1_2[s,i,j] + Phat1_Π[s,i,j] - Phat2_Π[s,i,j] == 0.0)
        end
    end

    # --- Πhat_0 constraint
    @constraint(model, [s=1:S], N*βhat1_1[s,:]+ βhat1_2[s,:] + Phat1_Π[s,:,end] - Phat2_Π[s,:,end] .== 0)
    # --- From Πtilde ---
    # --- Πtilde_L constraint
    for i in 1:(num_nodes-1), j in 1:num_arcs
        if network.node_arc_incidence[i,j]
            @constraint(model, [s=1:S], (-N*Ztilde1_1[s,:,:])[i,j]-Ztilde1_4[s,i,j] + Ptilde1_Π[s,i,j] - Ptilde2_Π[s,i,j] == 0.0)
        end
    end

    # --- Πtilde_0 constraint
    @constraint(model, [s=1:S], N*βtilde1_1[s,:]+ βtilde1_4[s,:] + Ptilde1_Π[s,:,end] - Ptilde2_Π[s,:,end] .== 0)
    # --- From Ytilde ---
    # --- From Ytilde_L constraint
    for i in 1:num_arcs, j in 1:num_arcs
        if network.arc_adjacency[i,j]
            @constraint(model, [s=1:S], (N_y' * Ztilde1_2[s,:,:])[i,j]+Ztilde1_3[s,i,j]-Ztilde1_6[s,i,j] + Ptilde1_Y[s,i,j] - Ptilde2_Y[s,i,j] == 0.0)
        end
    end

    # --- Ytilde_0 constraint
    @constraint(model, [s=1:S], -N_y' * βtilde1_2[s,:]-βtilde1_3[s,:]+βtilde1_6[s,:]+ Ptilde1_Y[s,:,end] - Ptilde2_Y[s,:,end] .== 0)
    # --- From Λhat1 ---
    @constraint(model, [s=1:S], Zhat1[s,:,:]*R[s]' + βhat1[s,:]*r_dict[s]' + Γhat1[s,:,:] .== 0.0)
    # --- From Λhat2 ---
    @constraint(model, [s=1:S], Zhat2[s,:,:]*R[s]' + βhat2[s,:]*r_dict[s]' + Γhat2[s,:,:] .== 0.0)
    # --- From Λtilde1 ---
    @constraint(model, [s=1:S], Ztilde1[s,:,:]*R[s]' + βtilde1[s,:]*r_dict[s]' + Γtilde1[s,:,:] .== 0.0)
    # --- From Λtilde2 ---
    @constraint(model, [s=1:S], Ztilde2[s,:,:]*R[s]' + βtilde2[s,:]*r_dict[s]' + Γtilde2[s,:,:] .== 0.0)

    vars = Dict(
        :Mhat => Mhat,
        :Mtilde => Mtilde,
        :Zhat1 => Zhat1,
        :Zhat2 => Zhat2,
        :Ztilde1 => Ztilde1,
        :Ztilde2 => Ztilde2,
        :Γhat1 => Γhat1,
        :Γhat2 => Γhat2,
        :Γtilde1 => Γtilde1,
        :Γtilde2 => Γtilde2,
        :Phat1_Φ => Phat1_Φ,
        :Phat1_Π => Phat1_Π,
        :Phat2_Φ => Phat2_Φ,
        :Phat2_Π => Phat2_Π,
        :Ptilde1_Φ => Ptilde1_Φ,
        :Ptilde1_Π => Ptilde1_Π,
        :Ptilde2_Φ => Ptilde2_Φ,
        :Ptilde2_Π => Ptilde2_Π,
        :Ptilde1_Y => Ptilde1_Y,
        :Ptilde1_Yts => Ptilde1_Yts,
        :Ptilde2_Y => Ptilde2_Y,
        :Ptilde2_Yts => Ptilde2_Yts,
        :Uhat1 => Uhat1,
        :Uhat3 => Uhat3,
        :Utilde1 => Utilde1,
        :Utilde3 => Utilde3,
        :βhat1_1 => βhat1_1,
        :βtilde1_1 => βtilde1_1,
        :βtilde1_3 => βtilde1_3,
        :Ztilde1_3 => Ztilde1_3,
    )

    data = Dict(
        :E => E,
        :v => v,
        :ϕU => ϕU,
        :S => S,
        :d0 => d0,
        :uncertainty_set => uncertainty_set,
    )

    return model, vars, data
end
