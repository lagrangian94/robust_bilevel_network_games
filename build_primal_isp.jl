"""
build_primal_isp.jl — Primal ISP (Inner Subproblem) for nested Benders decomposition.

The primal ISP directly solves the LDR problem with variables (Φ, Ψ, Π, μ, η, ϑ, Λ, M).
This is the dual of the current "dual ISP" in nested_benders_trust_region.jl.

Key advantages:
- Cut coefficients extracted via value(μhat) — no shadow_price needed
- α updated via set_objective_coefficient per inner iteration
- Enables future LDR sparsity (zero-fixing) and chordal SDP decomposition

Architecture (parameter location reversal):
| Current Dual ISP         | Primal ISP                    |
|--------------------------|-------------------------------|
| α in constraint RHS      | α in objective (μhat coeff)   |
| x,h,λ,ψ0 in objective   | x,h,λ,ψ0 in constraints      |
| shadow_price → μ̂         | value(μhat) → μ̂              |
"""

using JuMP
using LinearAlgebra
using Infiltrator


function build_primal_isp_leader(network, S, ϕU, λU, γ, w, v, uncertainty_set, optimizer,
                                  x_sol, λ_sol, h_sol, ψ0_sol, true_S)
    # Extract network dimensions
    num_nodes = length(network.nodes)
    num_arcs = length(network.arcs) - 1  # exclude dummy arc

    # Network data
    N = network.N
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict],
                                  uncertainty_set[:xi_bar], uncertainty_set[:epsilon]

    # Auxiliary structures
    d0 = zeros(num_arcs + 1)
    d0[end] = 1.0
    I_0 = [Matrix{Float64}(I, num_arcs, num_arcs) zeros(num_arcs)]

    # Fixed parameters from outer problem
    x, λ, h, ψ0 = x_sol, λ_sol, h_sol, ψ0_sol

    # Create model
    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))

    println("Building primal ISP leader...")
    println("  Nodes: $num_nodes, Arcs: $num_arcs, Scenarios: $S, true_S: $true_S")

    # =========================================================================
    # VARIABLES (hat part from build_full_model.jl)
    # =========================================================================
    @variable(model, ηhat[1:S] >= 0)
    @variable(model, μhat[1:S, 1:num_arcs] >= 0)
    @variable(model, Φhat[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound=-ϕU, upper_bound=ϕU)
    @variable(model, Ψhat[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Πhat[s=1:S, 1:num_nodes-1, 1:num_arcs+1], lower_bound=-ϕU, upper_bound=ϕU)
    @variable(model, ϑhat[1:S] >= 0)
    @variable(model, Mhat[1:S, 1:num_arcs+1, 1:num_arcs+1])

    dim_Λhat1_rows = num_arcs + 1 + (num_nodes - 1) + num_arcs
    @variable(model, Λhat1[s=1:S, 1:dim_Λhat1_rows, 1:size(R[1], 1)])
    @variable(model, Λhat2[s=1:S, 1:num_arcs, 1:size(R[1], 1)])

    # Split into L (linear part, cols 1:num_arcs) and 0 (constant part, col num_arcs+1)
    Φhat_L = Φhat[:, :, 1:num_arcs]
    Φhat_0 = Φhat[:, :, num_arcs+1]
    Ψhat_L = Ψhat[:, :, 1:num_arcs]
    Ψhat_0 = Ψhat[:, :, num_arcs+1]
    Πhat_L = Πhat[:, :, 1:num_arcs]
    Πhat_0 = Πhat[:, :, num_arcs+1]

    println("  ✓ Decision variables created")

    # =========================================================================
    # LDR SPARSITY: fix non-adjacent entries to zero
    # This matches the dual ISP's selective constraint creation
    # =========================================================================
    for s in 1:S, i in 1:num_arcs, j in 1:num_arcs
        if !network.arc_adjacency[i,j]
            fix(Φhat[s,i,j], 0.0; force=true)
            fix(Ψhat[s,i,j], 0.0; force=true)
        end
    end
    for s in 1:S, i in 1:num_nodes-1, j in 1:num_arcs
        if !network.node_arc_incidence[i,j]
            fix(Πhat[s,i,j], 0.0; force=true)
        end
    end
    println("  ✓ LDR sparsity constraints applied")

    # =========================================================================
    # OBJECTIVE: Min (1/true_S)*Σ_s ηhat[s] + Σ_{s,k} α_k * μhat[s,k]
    # α coefficients initialized to 0, updated via set_objective_coefficient
    # =========================================================================
    @objective(model, Min, (1/true_S) * sum(ηhat[s] for s in 1:S))

    # =========================================================================
    # CONSTRAINTS
    # =========================================================================

    # --- SOC constraints on Λ ---
    @constraint(model, [s=1:S, i=1:dim_Λhat1_rows], Λhat1[s, i, :] in SecondOrderCone())
    @constraint(model, [s=1:S, i=1:num_arcs], Λhat2[s, i, :] in SecondOrderCone())

    # --- PSD constraint on Mhat ---
    @constraint(model, [s=1:S], Mhat[s, :, :] in PSDCone())

    # --- (14e) SDP linking constraints ---
    for s in 1:S
        D_s = diagm(xi_bar[s])
        Mhat_11 = Mhat[s, 1:num_arcs, 1:num_arcs]
        Mhat_12 = Mhat[s, 1:num_arcs, end]
        Mhat_22 = Mhat[s, end, end]

        @constraint(model, Mhat_11 .== ϑhat[s] * Matrix{Float64}(I, num_arcs, num_arcs)
                    - adjoint(D_s) * (Φhat_L[s,:,:] - v * Ψhat_L[s,:,:]))
        @constraint(model, Mhat_12 .== -(1/2) * (
            (Φhat_L[s,:,:] - v*Ψhat_L[s,:,:]) * xi_bar[s]
            + adjoint(D_s) * (Φhat_0[s,:] - v*Ψhat_0[s,:])))
        @constraint(model, Mhat_22 .== ηhat[s]
                    - (Φhat_0[s,:] - v*Ψhat_0[s,:])' * xi_bar[s]
                    - ϑhat[s] * (epsilon^2))
    end

    println("  ✓ SDP constraints (14e) added")

    # --- (14j) Big-M constraints (x is parameter) ---
    for s in 1:S, i in 1:num_arcs, j in 1:num_arcs+1
        @constraint(model, Ψhat[s,i,j] <= ϕU * x[i])
        @constraint(model, Ψhat[s,i,j] - Φhat[s,i,j] <= 0)
        @constraint(model, Φhat[s,i,j] - Ψhat[s,i,j] <= ϕU * (1 - x[i]))
    end

    println("  ✓ Big-M constraints (14j) added")

    # --- (14m) Λhat1 constraints ---
    for s in 1:S
        Q_hat = adjoint(N) * Πhat_L[s, :, :] + adjoint(I_0) * Φhat_L[s, :, :]
        lhs_mat = vcat(Q_hat, Πhat_L[s, :, :], Φhat_L[s, :, :])
        @constraint(model, Λhat1[s, :, :] * R[s] - lhs_mat .== 0.0)

        rhs_vec = vcat(
            d0 - adjoint(N)*Πhat_0[s,:] - adjoint(I_0)*Φhat_0[s,:],
            -Πhat_0[s,:],
            -Φhat_0[s,:])
        @constraint(model, Λhat1[s, :, :] * r_dict[s] .>= rhs_vec)
    end

    println("  ✓ SOC constraints (14m) added")

    # --- (14n) Λhat2 constraints ---
    for s in 1:S
        @constraint(model, Λhat2[s, :, :] * R[s] .== -Φhat_L[s, :, :])
        @constraint(model, Λhat2[s, :, :] * r_dict[s] .- Φhat_0[s,:] .+ μhat[s,:] .>= 0.0)
    end

    println("  ✓ SOC constraints (14n) added")

    vars = Dict(
        :ηhat => ηhat,
        :μhat => μhat,
        :Φhat => Φhat,
        :Ψhat => Ψhat,
        :Πhat => Πhat,
        :ϑhat => ϑhat,
        :Mhat => Mhat,
        :Λhat1 => Λhat1,
        :Λhat2 => Λhat2,
    )

    println("  Variables: $(num_variables(model))")
    return model, vars
end


function build_primal_isp_follower(network, S, ϕU, λU, γ, w, v, uncertainty_set, optimizer,
                                    x_sol, λ_sol, h_sol, ψ0_sol, true_S)
    # Extract network dimensions
    num_nodes = length(network.nodes)
    num_arcs = length(network.arcs) - 1

    # Network data
    N = network.N
    N_y = N[:, 1:num_arcs]
    N_ts = N[:, end]
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict],
                                  uncertainty_set[:xi_bar], uncertainty_set[:epsilon]

    # Auxiliary structures
    d0 = zeros(num_arcs + 1)
    d0[end] = 1.0
    I_0 = [Matrix{Float64}(I, num_arcs, num_arcs) zeros(num_arcs)]

    # Fixed parameters
    x, λ, h, ψ0 = x_sol, λ_sol, h_sol, ψ0_sol

    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))

    println("Building primal ISP follower...")
    println("  Nodes: $num_nodes, Arcs: $num_arcs, Scenarios: $S, true_S: $true_S")

    # =========================================================================
    # VARIABLES (tilde part from build_full_model.jl)
    # =========================================================================
    @variable(model, ηtilde[1:S])  # No lower bound! (important, matches build_full_model.jl)
    @variable(model, μtilde[1:S, 1:num_arcs] >= 0)
    @variable(model, Φtilde[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound=-ϕU, upper_bound=ϕU)
    @variable(model, Ψtilde[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Πtilde[s=1:S, 1:num_nodes-1, 1:num_arcs+1], lower_bound=-ϕU, upper_bound=ϕU)
    @variable(model, Ytilde[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound=-ϕU, upper_bound=ϕU)
    @variable(model, Yts_tilde[s=1:S, 1, 1:num_arcs+1], lower_bound=-ϕU, upper_bound=ϕU)
    @variable(model, ϑtilde[1:S] >= 0)
    @variable(model, Mtilde[1:S, 1:num_arcs+1, 1:num_arcs+1])

    dim_Λtilde1_rows = num_arcs + 1 + (num_nodes - 1) + num_arcs + num_nodes - 1 + num_arcs + num_arcs
    @variable(model, Λtilde1[s=1:S, 1:dim_Λtilde1_rows, 1:size(R[1], 1)])
    @variable(model, Λtilde2[s=1:S, 1:num_arcs, 1:size(R[1], 1)])

    # Split into L and 0 parts
    Φtilde_L = Φtilde[:, :, 1:num_arcs]
    Φtilde_0 = Φtilde[:, :, num_arcs+1]
    Ψtilde_L = Ψtilde[:, :, 1:num_arcs]
    Ψtilde_0 = Ψtilde[:, :, num_arcs+1]
    Πtilde_L = Πtilde[:, :, 1:num_arcs]
    Πtilde_0 = Πtilde[:, :, num_arcs+1]
    Ytilde_L = Ytilde[:, :, 1:num_arcs]
    Ytilde_0 = Ytilde[:, :, num_arcs+1]
    Yts_tilde_L = Yts_tilde[:, :, 1:num_arcs]
    Yts_tilde_0 = Yts_tilde[:, 1, num_arcs+1]

    println("  ✓ Decision variables created")

    # =========================================================================
    # LDR SPARSITY: fix non-adjacent entries to zero
    # This matches the dual ISP's selective constraint creation
    # =========================================================================
    for s in 1:S, i in 1:num_arcs, j in 1:num_arcs
        if !network.arc_adjacency[i,j]
            fix(Φtilde[s,i,j], 0.0; force=true)
            fix(Ψtilde[s,i,j], 0.0; force=true)
            fix(Ytilde[s,i,j], 0.0; force=true)
        end
    end
    for s in 1:S, i in 1:num_nodes-1, j in 1:num_arcs
        if !network.node_arc_incidence[i,j]
            fix(Πtilde[s,i,j], 0.0; force=true)
        end
    end
    println("  ✓ LDR sparsity constraints applied")

    # =========================================================================
    # OBJECTIVE: Min (1/true_S)*Σ_s ηtilde[s] + Σ_{s,k} α_k * μtilde[s,k]
    # =========================================================================
    @objective(model, Min, (1/true_S) * sum(ηtilde[s] for s in 1:S))

    # =========================================================================
    # CONSTRAINTS
    # =========================================================================

    # --- SOC constraints ---
    @constraint(model, [s=1:S, i=1:dim_Λtilde1_rows], Λtilde1[s, i, :] in SecondOrderCone())
    @constraint(model, [s=1:S, i=1:num_arcs], Λtilde2[s, i, :] in SecondOrderCone())

    # --- PSD constraint ---
    @constraint(model, [s=1:S], Mtilde[s, :, :] in PSDCone())

    # --- (14e) SDP linking constraints for tilde ---
    for s in 1:S
        D_s = diagm(xi_bar[s])
        Mtilde_11 = Mtilde[s, 1:num_arcs, 1:num_arcs]
        Mtilde_12 = Mtilde[s, 1:num_arcs, end]
        Mtilde_22 = Mtilde[s, end, end]

        @constraint(model, Mtilde_11 .== ϑtilde[s] * Matrix{Float64}(I, num_arcs, num_arcs)
                    - adjoint(D_s) * (Φtilde_L[s,:,:] - v * Ψtilde_L[s,:,:]))
        @constraint(model, Mtilde_12 .== -(1/2) * (
            (Φtilde_L[s,:,:] - v*Ψtilde_L[s,:,:]) * xi_bar[s]
            + adjoint(D_s) * (Φtilde_0[s,:] - v*Ψtilde_0[s,:])
            - Yts_tilde_L[s,1,:].data))
        @constraint(model, Mtilde_22 .== ηtilde[s]
                    - (Φtilde_0[s,:] - v*Ψtilde_0[s,:])' * xi_bar[s]
                    + Yts_tilde_0[s]
                    - ϑtilde[s] * (epsilon^2))
    end

    println("  ✓ SDP constraints (14e) added")

    # --- (14k) Big-M constraints (x is parameter) ---
    for s in 1:S, i in 1:num_arcs, j in 1:num_arcs+1
        @constraint(model, Ψtilde[s,i,j] <= ϕU * x[i])
        @constraint(model, Ψtilde[s,i,j] - Φtilde[s,i,j] <= 0)
        @constraint(model, Φtilde[s,i,j] - Ψtilde[s,i,j] <= ϕU * (1 - x[i]))
    end

    println("  ✓ Big-M constraints (14k) added")

    # --- (14o) Λtilde1 constraints ---
    for s in 1:S
        D_s = diagm(xi_bar[s])
        diag_λ_ψ = diagm(λ * ones(num_arcs) - v * ψ0)

        # Block structure (6 blocks) — matches build_full_model.jl lines 317-328
        Q_tilde_col = adjoint(N) * Πtilde_L[s, :, :] + adjoint(I_0) * Φtilde_L[s, :, :]
        block2 = -N_y * Ytilde_L[s, :, :] - N_ts * Yts_tilde_L[s, :, :]
        block3 = -Ytilde_L[s, :, :] + diag_λ_ψ * D_s
        block4 = Πtilde_L[s, :, :]
        block5 = Φtilde_L[s, :, :]
        block6 = Ytilde_L[s, :, :]
        rhs_mat = vcat(Q_tilde_col, block2, block3, block4, block5, block6)
        @constraint(model, Λtilde1[s, :, :] * R[s] .== rhs_mat)

        rhs_vec_1 = λ * d0 - adjoint(N)*Πtilde_0[s,:] - adjoint(I_0)*Φtilde_0[s,:]
        rhs_vec_2 = N_y * Ytilde_0[s,:] + N_ts * Yts_tilde_0[s]
        rhs_vec_3 = -h + Ytilde_0[s,:] - diag_λ_ψ * xi_bar[s]
        rhs_vec_4 = -Πtilde_0[s,:]
        rhs_vec_5 = -Φtilde_0[s,:]
        rhs_vec_6 = -Ytilde_0[s,:]
        rhs_vec = vcat(rhs_vec_1, rhs_vec_2, rhs_vec_3, rhs_vec_4, rhs_vec_5, rhs_vec_6)
        @constraint(model, Λtilde1[s, :, :] * r_dict[s] .>= rhs_vec)
    end

    println("  ✓ SOC constraints (14o) added")

    # --- (14p) Λtilde2 constraints ---
    for s in 1:S
        @constraint(model, Λtilde2[s, :, :] * R[s] + Φtilde_L[s, :, :] .== 0.0)
        @constraint(model, Λtilde2[s, :, :] * r_dict[s] - Φtilde_0[s,:] + μtilde[s,:] .>= 0.0)
    end

    println("  ✓ SOC constraints (14p) added")

    vars = Dict(
        :ηtilde => ηtilde,
        :μtilde => μtilde,
        :Φtilde => Φtilde,
        :Ψtilde => Ψtilde,
        :Πtilde => Πtilde,
        :Ytilde => Ytilde,
        :Yts_tilde => Yts_tilde,
        :ϑtilde => ϑtilde,
        :Mtilde => Mtilde,
        :Λtilde1 => Λtilde1,
        :Λtilde2 => Λtilde2,
    )

    println("  Variables: $(num_variables(model))")
    return model, vars
end


"""
Update α coefficients in objective and solve. Extract cuts from variable values directly.
"""
function primal_isp_leader_optimize!(model::Model, vars::Dict;
                                      isp_data=nothing, α_sol=nothing)
    true_S = isp_data[:S]
    num_arcs = length(α_sol)
    μhat = vars[:μhat]

    # Update objective: set α coefficients for μhat
    for k in 1:num_arcs
        set_objective_coefficient(model, μhat[1, k], α_sol[k])
    end

    optimize!(model)
    st = MOI.get(model, MOI.TerminationStatus())

    if (st == MOI.OPTIMAL) || (st == MOI.SLOW_PROGRESS)
        # Extract cuts directly from variable values (no shadow_price!)
        μhat_val = collect(value.(μhat[1, :]))
        ηhat_val = value(vars[:ηhat][1])
        intercept = (1/true_S) * ηhat_val
        subgradient = μhat_val
        obj_val = intercept + α_sol' * subgradient

        # Strong duality check
        @assert abs(obj_val - objective_value(model)) < 1e-4 (
            "Primal ISP leader strong duality failed: " *
            "obj_val=$obj_val, model_obj=$(objective_value(model))")

        cut_coeff = Dict(:μhat => subgradient, :intercept => intercept, :obj_val => obj_val)
        return (:OptimalityCut, cut_coeff)
    else
        @infiltrate
        error("Primal ISP leader not optimal: $(termination_status(model))")
    end
end


"""
Update α coefficients in objective and solve. Extract cuts from variable values directly.
"""
function primal_isp_follower_optimize!(model::Model, vars::Dict;
                                        isp_data=nothing, α_sol=nothing)
    true_S = isp_data[:S]
    num_arcs = length(α_sol)
    μtilde = vars[:μtilde]

    # Update objective: set α coefficients for μtilde
    for k in 1:num_arcs
        set_objective_coefficient(model, μtilde[1, k], α_sol[k])
    end

    optimize!(model)
    st = MOI.get(model, MOI.TerminationStatus())

    if (st == MOI.OPTIMAL) || (st == MOI.SLOW_PROGRESS)
        # Extract cuts directly from variable values
        μtilde_val = collect(value.(μtilde[1, :]))
        ηtilde_val = value(vars[:ηtilde][1])
        intercept = (1/true_S) * ηtilde_val
        subgradient = μtilde_val
        obj_val = intercept + α_sol' * subgradient

        # Strong duality check
        if abs(obj_val - objective_value(model)) > 1e-4
            @warn "Primal ISP follower strong duality gap: obj_val=$obj_val, model_obj=$(objective_value(model))"
            @infiltrate
        end

        cut_coeff = Dict(:μtilde => subgradient, :intercept => intercept, :obj_val => obj_val)
        return (:OptimalityCut, cut_coeff)
    else
        @infiltrate
        error("Primal ISP follower not optimal: $(termination_status(model))")
    end
end


"""
Build per-scenario primal ISP instances. Same pattern as initialize_isp() in nested_benders.jl.
Rebuilds models when x, h, λ, ψ0 change (new outer iteration).
"""
function initialize_primal_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
                                conic_optimizer=nothing, x_sol=nothing, λ_sol=nothing,
                                h_sol=nothing, ψ0_sol=nothing)
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict],
                                  uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    leader_instances = Dict{Int, Tuple{Model, Dict}}()
    follower_instances = Dict{Int, Tuple{Model, Dict}}()

    for s in 1:S
        U_s = Dict(:R => Dict(1=>R[s]), :r_dict => Dict(1=>r_dict[s]),
                    :xi_bar => Dict(1=>xi_bar[s]), :epsilon => epsilon)
        leader_instances[s] = build_primal_isp_leader(
            network, 1, ϕU, λU, γ, w, v, U_s, conic_optimizer,
            x_sol, λ_sol, h_sol, ψ0_sol, S)
        follower_instances[s] = build_primal_isp_follower(
            network, 1, ϕU, λU, γ, w, v, U_s, conic_optimizer,
            x_sol, λ_sol, h_sol, ψ0_sol, S)
    end

    return leader_instances, follower_instances
end
