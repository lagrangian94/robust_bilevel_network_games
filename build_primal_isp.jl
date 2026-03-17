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
    # Store constraint refs for outer cut extraction (dual → Uhat1, Uhat3)
    con_bigM1_hat = Array{ConstraintRef}(undef, num_arcs, num_arcs+1)
    con_bigM3_hat = Array{ConstraintRef}(undef, num_arcs, num_arcs+1)
    for s in 1:S, i in 1:num_arcs, j in 1:num_arcs+1
        con_bigM1_hat[i,j] = @constraint(model, Ψhat[s,i,j] <= ϕU * x[i])
        @constraint(model, Ψhat[s,i,j] - Φhat[s,i,j] <= 0)
        con_bigM3_hat[i,j] = @constraint(model, Φhat[s,i,j] - Ψhat[s,i,j] <= ϕU * (1 - x[i]))
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
        :con_bigM1_hat => con_bigM1_hat,
        :con_bigM3_hat => con_bigM3_hat,
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
    # Store constraint refs for outer cut extraction (dual → Utilde1, Utilde3)
    con_bigM1_tilde = Array{ConstraintRef}(undef, num_arcs, num_arcs+1)
    con_bigM3_tilde = Array{ConstraintRef}(undef, num_arcs, num_arcs+1)
    for s in 1:S, i in 1:num_arcs, j in 1:num_arcs+1
        con_bigM1_tilde[i,j] = @constraint(model, Ψtilde[s,i,j] <= ϕU * x[i])
        @constraint(model, Ψtilde[s,i,j] - Φtilde[s,i,j] <= 0)
        con_bigM3_tilde[i,j] = @constraint(model, Φtilde[s,i,j] - Ψtilde[s,i,j] <= ϕU * (1 - x[i]))
    end

    println("  ✓ Big-M constraints (14k) added")

    # --- (14o) Λtilde1 constraints ---
    # Store constraint refs for outer cut extraction
    # Block sizes: 1=(num_arcs+1), 2=(num_nodes-1), 3=(num_arcs), 4=(num_nodes-1), 5=(num_arcs), 6=(num_arcs)
    block1_size = num_arcs + 1
    block2_size = num_nodes - 1
    block3_size = num_arcs
    total_rows = dim_Λtilde1_rows
    block3_start_idx = block1_size + block2_size + 1
    block3_end_idx = block3_start_idx + block3_size - 1

    con_soc_eq_tilde = nothing    # will store equality constraint refs
    con_soc_ineq_tilde = nothing  # will store inequality constraint refs

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
        con_soc_eq_tilde = @constraint(model, Λtilde1[s, :, :] * R[s] .== rhs_mat)

        rhs_vec_1 = λ * d0 - adjoint(N)*Πtilde_0[s,:] - adjoint(I_0)*Φtilde_0[s,:]
        rhs_vec_2 = N_y * Ytilde_0[s,:] + N_ts * Yts_tilde_0[s]
        rhs_vec_3 = -h + Ytilde_0[s,:] - diag_λ_ψ * xi_bar[s]
        rhs_vec_4 = -Πtilde_0[s,:]
        rhs_vec_5 = -Φtilde_0[s,:]
        rhs_vec_6 = -Ytilde_0[s,:]
        rhs_vec = vcat(rhs_vec_1, rhs_vec_2, rhs_vec_3, rhs_vec_4, rhs_vec_5, rhs_vec_6)
        con_soc_ineq_tilde = @constraint(model, Λtilde1[s, :, :] * r_dict[s] .>= rhs_vec)
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
        :con_bigM1_tilde => con_bigM1_tilde,
        :con_bigM3_tilde => con_bigM3_tilde,
        :con_soc_eq_tilde => con_soc_eq_tilde,
        :con_soc_ineq_tilde => con_soc_ineq_tilde,
        :block3_start_idx => block3_start_idx,
        :block3_end_idx => block3_end_idx,
        :block1_size => block1_size,
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

        # IPM artifact correction: Mosek (IPM) returns analytic center, inflating μ
        # with +ε offset on ZERO-COST components only (where α_k ≈ 0).
        # When α_k > 0, μ_k has nonzero objective coefficient → no offset.
        # See memory/ipm_mu_offset.md for detailed explanation.
        ε = isp_data[:uncertainty_set][:epsilon]
        for k in eachindex(subgradient)
            if α_sol[k] < 1e-8
                subgradient[k] = max(subgradient[k] - ε, 0.0)
            end
        end
        # Recompute intercept so cut remains tight: intercept = obj_val - α'·μ_corrected
        intercept = obj_val - α_sol' * subgradient

        cut_coeff = Dict(:μhat => subgradient, :intercept => intercept, :obj_val => obj_val)
        return (:OptimalityCut, cut_coeff)
    else
        println("FAILED α_sol = ", α_sol)
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

        # IPM artifact correction: subtract ε offset only on zero-cost components (α_k ≈ 0).
        # See memory/ipm_mu_offset.md for detailed explanation.
        ε = isp_data[:uncertainty_set][:epsilon]
        for k in eachindex(subgradient)
            if α_sol[k] < 1e-8
                subgradient[k] = max(subgradient[k] - ε, 0.0)
            end
        end
        intercept = obj_val - α_sol' * subgradient

        cut_coeff = Dict(:μtilde => subgradient, :intercept => intercept, :obj_val => obj_val)
        return (:OptimalityCut, cut_coeff)
    else
        @error "Primal ISP follower not optimal: $(termination_status(model))" α_sol
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


"""
Update (x, h, λ, ψ0) parameters in existing primal ISP instances via set_normalized_rhs.
Avoids rebuilding models each outer iteration, preserving solver warm-start.

Parameter locations in constraints:
  Leader:  Big-M1/3 → x
  Follower: Big-M1/3 → x, SOC eq block3 → λ,ψ0, SOC ineq block1 → λ, SOC ineq block3 → h,λ,ψ0
"""
function update_primal_isp_parameters!(
    primal_leader_instances::Dict, primal_follower_instances::Dict;
    x_sol, h_sol, λ_sol, ψ0_sol, isp_data)

    S = isp_data[:S]
    ϕU = isp_data[:ϕU]
    v = isp_data[:v]
    d0 = isp_data[:d0]
    xi_bar = isp_data[:uncertainty_set][:xi_bar]

    for s in 1:S
        # === Leader: update Big-M constraints (x only) ===
        vars_l = primal_leader_instances[s][2]
        num_arcs = size(vars_l[:con_bigM1_hat], 1)

        for i in 1:num_arcs, j in 1:(num_arcs+1)
            set_normalized_rhs(vars_l[:con_bigM1_hat][i,j], ϕU * x_sol[i])
            set_normalized_rhs(vars_l[:con_bigM3_hat][i,j], ϕU * (1 - x_sol[i]))
        end

        # === Follower: update Big-M + SOC constraints ===
        vars_f = primal_follower_instances[s][2]

        # Big-M (x)
        for i in 1:num_arcs, j in 1:(num_arcs+1)
            set_normalized_rhs(vars_f[:con_bigM1_tilde][i,j], ϕU * x_sol[i])
            set_normalized_rhs(vars_f[:con_bigM3_tilde][i,j], ϕU * (1 - x_sol[i]))
        end

        # SOC equality block3: (λ-v*ψ0[i]) * xi_bar[s][i] * δ(i,j)
        b3s = vars_f[:block3_start_idx]
        b3e = vars_f[:block3_end_idx]
        b1sz = vars_f[:block1_size]
        eq_cons = vars_f[:con_soc_eq_tilde]
        n_eq_cols = size(eq_cons, 2)

        for i in 1:num_arcs
            row = b3s + i - 1
            param_diag = (λ_sol - v * ψ0_sol[i]) * xi_bar[s][i]
            for j in 1:n_eq_cols
                set_normalized_rhs(eq_cons[row, j], i == j ? param_diag : 0.0)
            end
        end

        # SOC inequality block1: λ * d0[row]
        ineq_cons = vars_f[:con_soc_ineq_tilde]
        for row in 1:b1sz
            set_normalized_rhs(ineq_cons[row], λ_sol * d0[row])
        end

        # SOC inequality block3: -h[i] - (λ-v*ψ0[i]) * xi_bar[s][i]
        for i in 1:num_arcs
            row = b3s + i - 1
            new_rhs = -h_sol[i] - (λ_sol - v * ψ0_sol[i]) * xi_bar[s][i]
            set_normalized_rhs(ineq_cons[row], new_rhs)
        end
    end
end


"""
Hybrid inner loop: uses primal ISP for inner α optimization,
then dual ISP for outer cut generation.

Same structure as tr_imp_optimize! (nested_benders_trust_region.jl:279-454),
but calls primal_isp_leader_optimize! / primal_isp_follower_optimize! instead of
isp_leader_optimize! / isp_follower_optimize!.

Key difference from original:
- Primal ISP takes α as objective coefficient (set_objective_coefficient)
- No need for uncertainty_set, λ_sol, x_sol, h_sol, ψ0_sol per ISP call
  (they're baked into the primal model at build time)
"""
function tr_imp_optimize_hybrid!(imp_model::Model, imp_vars::Dict,
    primal_leader_instances::Dict, primal_follower_instances::Dict;
    isp_data=nothing, λ_sol=nothing, x_sol=nothing,
    h_sol=nothing, ψ0_sol=nothing, outer_iter=nothing,
    imp_cuts=nothing, inner_tr=true, tol=1e-4, parallel=false)

    st = MOI.get(imp_model, MOI.TerminationStatus())
    iter = 0
    uncertainty_set = isp_data[:uncertainty_set]
    S_total = isp_data[:S]  # scenario count for /S averaging
    past_obj = []
    past_subprob_obj = []
    past_major_subprob_obj = []
    past_lower_bound = []
    past_upper_bound = []
    major_iter = []
    lower_bound = -Inf
    result = Dict()
    result[:cuts] = Dict()
    if inner_tr
        B_conti_max = isp_data[:w]/isp_data[:S]
        B_conti = B_conti_max * 0.01
        counter = 0
        β_relative = 1e-4
        ρ = 0.0
        centers = Dict(:α=>value.(imp_vars[:α]))
        tr_constraints = Dict(:continuous=>nothing)
    end
    ## Clean up old cuts from previous outer iteration
    if outer_iter>1
        for (cut_name, cut) in imp_cuts[:old_cuts]
            delete(imp_model, cut)
        end
        if inner_tr && imp_cuts[:old_tr_constraints] !== nothing
            for tr_cons in imp_cuts[:old_tr_constraints]
                valid_cons = filter(c -> is_valid(imp_model, c), tr_cons)
                delete.(imp_model, valid_cons)
            end
        end
    end
    ##
    while (st == MOI.DUAL_INFEASIBLE || st == MOI.OPTIMAL)
        iter += 1
        @info "    [Inner-Hybrid] Iteration $iter"
        optimize!(imp_model)
        st = MOI.get(imp_model, MOI.TerminationStatus())
        α_sol = max.(value.(imp_vars[:α]), 0.0)  # clamp: 음수 numerical tolerance → unbounded 방지
        model_estimate = (sum(value.(imp_vars[:t_1_l])) + sum(value.(imp_vars[:t_1_f]))) / S_total  # average over scenarios
        subprob_obj = 0
        dict_cut_info_l, dict_cut_info_f = Dict(), Dict()
        scenario_results, status = solve_scenarios(S; parallel=parallel) do s
            # Primal ISP: only needs isp_data and α_sol (x,h,λ,ψ0 are in constraints)
            (status_l, cut_info_l) = primal_isp_leader_optimize!(
                primal_leader_instances[s][1], primal_leader_instances[s][2];
                isp_data=isp_data, α_sol=α_sol)
            (status_f, cut_info_f) = primal_isp_follower_optimize!(
                primal_follower_instances[s][1], primal_follower_instances[s][2];
                isp_data=isp_data, α_sol=α_sol)
            ok = (status_l == :OptimalityCut) && (status_f == :OptimalityCut)
            return (ok, (cut_info_l, cut_info_f))
        end
        for s in 1:S
            dict_cut_info_l[s] = scenario_results[s][1]
            dict_cut_info_f[s] = scenario_results[s][2]
            subprob_obj += scenario_results[s][1][:obj_val] + scenario_results[s][2][:obj_val]
        end
        subprob_obj /= S_total  # average over scenarios
        lower_bound = max(lower_bound, subprob_obj)
        gap = abs(model_estimate - lower_bound) / max(abs(model_estimate), 1e-10)
        if gap <= tol || lower_bound > model_estimate - 1e-4
            @info "Termination condition met (hybrid)"
            println("model_estimate: ", model_estimate, ", subprob_obj: ", subprob_obj, ", lower_bound: ", lower_bound)
            result[:past_obj] = past_obj
            result[:past_subprob_obj] = past_subprob_obj
            result[:α_sol] = α_sol
            result[:obj_val] = subprob_obj
            result[:past_lower_bound] = past_lower_bound
            result[:iter] = iter
            if inner_tr && tr_constraints[:continuous] !== nothing
                result[:tr_constraints] = tr_constraints[:continuous]
            else
                result[:tr_constraints] = nothing
            end
            return (:OptimalityCut, result)
        else
            if inner_tr
                # Serious Test
                if iter==1
                    push!(past_major_subprob_obj, subprob_obj)
                end
                tr_needs_update = false
                predicted_increase = model_estimate - past_major_subprob_obj[end]
                β_dynamic = max(1e-8, β_relative * predicted_increase)
                improvement = subprob_obj - past_major_subprob_obj[end]
                is_serious_step = (improvement >= β_dynamic)
                if is_serious_step
                    tr_needs_update = true
                    distance = norm(α_sol - centers[:α], Inf)
                    centers[:α] = α_sol
                    push!(major_iter, iter)
                    push!(past_major_subprob_obj, subprob_obj)
                    if (improvement >= 0.5*β_dynamic) && (distance >= B_conti - 1e-6)
                        @info "Very good improvement: Expanding B_conti"
                        B_conti = min(B_conti_max, B_conti * 2.0)
                    else
                        @info "Moderate improvement: Keeping B_conti"
                        B_conti = B_conti
                    end
                    tr_constraints = update_inner_trust_region_constraints!(imp_model, imp_vars, centers, B_conti, tr_constraints, network)
                else
                    @info "Poor improvement: Reducing B_conti"
                    ρ = min(1, B_conti) * improvement / β_dynamic
                    if ρ > 3.0
                        B_conti = B_conti / min(ρ,4)
                        counter = 0
                        tr_needs_update = true
                    elseif (1.0 < ρ) && (counter>=3)
                        B_conti = B_conti / min(ρ,4)
                        counter = 0
                        tr_needs_update = true
                    elseif (1.0 < ρ) && (counter<3)
                        counter += 1
                    elseif (0.0 < ρ) && (ρ <= 1.0)
                        counter += 1
                    else
                        B_conti = B_conti
                        counter = counter
                    end
                    if tr_needs_update
                        tr_constraints = update_inner_trust_region_constraints!(imp_model, imp_vars, centers, B_conti, tr_constraints, network)
                    end
                end
            end
            push!(past_obj, model_estimate)
            push!(past_subprob_obj, subprob_obj)
            push!(past_lower_bound, lower_bound)
            if status == false
                @warn "Subproblem is not optimal"
            end

            # Inner cuts: same format as dual ISP (intercept + α'*subgradient)
            subgradient_l = [dict_cut_info_l[s][:μhat] for s in 1:S]
            subgradient_f = [dict_cut_info_f[s][:μtilde] for s in 1:S]
            intercept_l = [dict_cut_info_l[s][:intercept] for s in 1:S]
            intercept_f = [dict_cut_info_f[s][:intercept] for s in 1:S]

            cut_added_l = @constraint(imp_model, [s=1:S], imp_vars[:t_1_l][s] <= intercept_l[s] + imp_vars[:α]'*subgradient_l[s])
            cut_added_f = @constraint(imp_model, [s=1:S], imp_vars[:t_1_f][s] <= intercept_f[s] + imp_vars[:α]'*subgradient_f[s])
            set_name.(cut_added_l, ["opt_cut_$(iter)_l_s$(s)" for s in 1:S])
            set_name.(cut_added_f, ["opt_cut_$(iter)_f_s$(s)" for s in 1:S])
            result[:cuts]["opt_cut_$(iter)_l"] = cut_added_l
            result[:cuts]["opt_cut_$(iter)_f"] = cut_added_f
            println("subproblem objective (hybrid): ", subprob_obj)
            @info "Optimality cut added (hybrid)"

            # Cut tightness check
            y = Dict(
                [imp_vars[:α][k] => α_sol[k] for k in 1:length(α_sol)]...,
            )
            function evaluate_expr(expr::AffExpr, var_values::Dict)
                eval_result = expr.constant
                for (var, coef) in expr.terms
                    if haskey(var_values, var)
                        eval_result += coef * var_values[var]
                    else
                        error("Variable $var not found in var_values")
                    end
                end
                return eval_result
            end
            opt_cut_val = sum(evaluate_expr(intercept_l[s] + imp_vars[:α]'*subgradient_l[s], y) for s in 1:S) + sum(evaluate_expr(intercept_f[s] + imp_vars[:α]'*subgradient_f[s], y) for s in 1:S)
            if abs(subprob_obj * S_total - opt_cut_val) > 1e-4  # opt_cut_val is raw, subprob_obj is /S
                println("something went wrong (hybrid)")
                @infiltrate
            end
        end
    end
end


"""
Update dual ISP objectives with current (x,h,λ,ψ0) and solve at converged α,
then call evaluate_master_opt_cut to extract outer cut coefficients.

In the original code, isp_leader_optimize!/isp_follower_optimize! update the dual ISP
objectives with (x,h,λ,ψ0) every inner iteration. In hybrid mode, the inner loop uses
primal ISP, so dual ISP objectives are never updated. This function bridges that gap.
"""
function primal_evaluate_master_opt_cut(
    dual_leader_instances::Dict, dual_follower_instances::Dict,
    isp_data::Dict, cut_info::Dict, iter::Int;
    λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing,
    multi_cut_lf=false)

    S = isp_data[:S]
    α_sol = cut_info[:α_sol]
    uncertainty_set = isp_data[:uncertainty_set]
    R = uncertainty_set[:R]
    r_dict = uncertainty_set[:r_dict]
    xi_bar = uncertainty_set[:xi_bar]
    epsilon = uncertainty_set[:epsilon]

    # First, call dual ISP optimize once per scenario to update objectives with (x,h,λ,ψ0)
    for s in 1:S
        U_s = Dict(:R => Dict(:1=>R[s]), :r_dict => Dict(:1=>r_dict[s]),
                    :xi_bar => Dict(:1=>xi_bar[s]), :epsilon => epsilon)
        isp_leader_optimize!(
            dual_leader_instances[s][1], dual_leader_instances[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
        isp_follower_optimize!(
            dual_follower_instances[s][1], dual_follower_instances[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
    end

    # Now dual ISP objectives are updated. Call evaluate_master_opt_cut as usual.
    return evaluate_master_opt_cut(
        dual_leader_instances, dual_follower_instances,
        isp_data, cut_info, iter; multi_cut_lf=multi_cut_lf)
end


"""
Extract outer cut coefficients entirely from primal ISP shadow prices.
No dual ISP needed.

WARNING: This function produces inaccurate outer cuts due to Mosek IPM conic dual degeneracy.
Shadow prices from IPM differ non-uniformly (30-40%) from dual ISP variable values
(the true subgradient). Unlike the inner cut μ offset (uniform +ε, correctable),
this cannot be simply corrected. Invalid outer cuts allow OMP to select extreme parameters
(e.g. λ=λU, h=ϕU) which then cause primal ISP infeasibility.
Use primal_evaluate_master_opt_cut (hybrid: dual ISP for outer cuts) instead.
See memory/ipm_mu_offset.md and debug_test/test_outer_cut_compare.jl for details.

Uses residual intercept: intercept = subprob_obj - eval(cut_terms at x*,h*,λ*,ψ0*)

Sign conventions (MOI, Min problem):
- `<=` constraint: Uhat1 = -dual(con) ≥ 0
- `>=` constraint: βtilde1 = dual(con) ≥ 0
- `==` constraint: Ztilde1 = dual(con) (free)
"""
function evaluate_master_opt_cut_from_primal(
    primal_leader_instances::Dict, primal_follower_instances::Dict,
    isp_data::Dict, cut_info::Dict, iter::Int;
    λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing,
    multi_cut_lf=false)

    S = isp_data[:S]
    α_sol = cut_info[:α_sol]
    uncertainty_set = isp_data[:uncertainty_set]
    R = uncertainty_set[:R]
    r_dict = uncertainty_set[:r_dict]
    xi_bar = uncertainty_set[:xi_bar]
    epsilon = uncertainty_set[:epsilon]
    num_arcs = length(α_sol)
    E = isp_data[:E]
    d0 = isp_data[:d0]
    v = isp_data[:v]
    ϕU = isp_data[:ϕU]

    diag_x_E = Diagonal(x_sol) * E
    diag_λ_ψ = Diagonal(λ_sol * ones(num_arcs) - v .* ψ0_sol)

    # First, re-solve primal ISP at converged α to ensure optimal solution
    status = true
    for s in 1:S
        (st_l, _) = primal_isp_leader_optimize!(
            primal_leader_instances[s][1], primal_leader_instances[s][2];
            isp_data=isp_data, α_sol=α_sol)
        (st_f, _) = primal_isp_follower_optimize!(
            primal_follower_instances[s][1], primal_follower_instances[s][2];
            isp_data=isp_data, α_sol=α_sol)
        status = status && (st_l == :OptimalityCut) && (st_f == :OptimalityCut)
        if !status
            @infiltrate
        end
    end

    # Extract dual values from primal ISP constraints
    # Leader: Uhat1, Uhat3 from Big-M constraints
    Uhat1 = zeros(S, num_arcs, num_arcs+1)
    Uhat3 = zeros(S, num_arcs, num_arcs+1)
    for s in 1:S
        vars_l = primal_leader_instances[s][2]
        # Uhat1 = -dual(Ψhat <= ϕU*x)  (negate for <= in Min)
        Uhat1[s,:,:] = -dual.(vars_l[:con_bigM1_hat])
        # Uhat3 = -dual(Φhat - Ψhat <= ϕU*(1-x))  (negate for <= in Min)
        Uhat3[s,:,:] = -dual.(vars_l[:con_bigM3_hat])
    end

    # Follower: Utilde1, Utilde3, Ztilde1_3, βtilde1_1, βtilde1_3
    Utilde1 = zeros(S, num_arcs, num_arcs+1)
    Utilde3 = zeros(S, num_arcs, num_arcs+1)
    dim_R_cols = size(R[1], 2)
    Ztilde1_3 = zeros(S, num_arcs, dim_R_cols)
    βtilde1_1 = zeros(S, num_arcs+1)
    βtilde1_3 = zeros(S, num_arcs)

    for s in 1:S
        vars_f = primal_follower_instances[s][2]
        # Utilde1, Utilde3 from Big-M (negate for <= in Min)
        Utilde1[s,:,:] = -dual.(vars_f[:con_bigM1_tilde])
        Utilde3[s,:,:] = -dual.(vars_f[:con_bigM3_tilde])

        # Ztilde1_3 from SOC equality constraint (block 3)
        b3s = vars_f[:block3_start_idx]
        b3e = vars_f[:block3_end_idx]
        b1sz = vars_f[:block1_size]
        # dual of equality: direct (free sign)
        eq_duals = dual.(vars_f[:con_soc_eq_tilde])
        Ztilde1_3[s,:,:] = eq_duals[b3s:b3e, :]

        # βtilde1_1 and βtilde1_3 from SOC inequality constraint (>= in Min → direct)
        ineq_duals = dual.(vars_f[:con_soc_ineq_tilde])
        βtilde1_1[s,:] = ineq_duals[1:b1sz]
        βtilde1_3[s,:] = ineq_duals[b3s:b3e]
    end

    # Compute cut terms at current (x*, h*, λ*, ψ0*)
    leader_obj = sum(objective_value(primal_leader_instances[s][1]) for s in 1:S)
    follower_obj = sum(objective_value(primal_follower_instances[s][1]) for s in 1:S)
    avg_obj = (leader_obj + follower_obj) / S  # average over scenarios

    @assert abs(avg_obj - cut_info[:obj_val]) < 1e-3 "obj mismatch: avg=$avg_obj, cut_info=$(cut_info[:obj_val])"

    # Always compute per-scenario intercepts (leader + follower)
    intercept_l = [objective_value(primal_leader_instances[s][1]) -
        (-ϕU * sum(Uhat1[s,:,:] .* diag_x_E) - ϕU * sum(Uhat3[s,:,:] .* (E - diag_x_E)))
        for s in 1:S]
    intercept_f = Float64[]
    for s in 1:S
        ct1 = -ϕU * sum(Utilde1[s,:,:] .* diag_x_E)
        ct2 = -ϕU * sum(Utilde3[s,:,:] .* (E - diag_x_E))
        ct3 = sum(Ztilde1_3[s,:,:] .* (diag_λ_ψ * diagm(xi_bar[s])))
        ct4 = (d0' * βtilde1_1[s,:]) * λ_sol
        ct5 = -(h_sol + diag_λ_ψ * xi_bar[s])' * βtilde1_3[s,:]
        push!(intercept_f, objective_value(primal_follower_instances[s][1]) - (ct1 + ct2 + ct3 + ct4 + ct5))
    end
    intercept = sum(intercept_l) + sum(intercept_f)

    return Dict(:Uhat1=>Uhat1, :Utilde1=>Utilde1, :Uhat3=>Uhat3, :Utilde3=>Utilde3,
                :Ztilde1_3=>Ztilde1_3, :βtilde1_1=>βtilde1_1, :βtilde1_3=>βtilde1_3,
                :intercept=>intercept, :intercept_l=>intercept_l, :intercept_f=>intercept_f)
end
