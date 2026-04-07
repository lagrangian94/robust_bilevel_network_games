"""
build_lp_isp.jl — LP ISP for partial robust cases (ε̂=0 or ε̃=0).

When ε=0 for one side, the S-lemma SDP + SOC robust counterpart + LDR slopes all collapse
to point evaluations at ζ=0. The resulting ISP is a pure LP.

Key advantages over SDP ISP with small ε→0 approximation:
- No ϕU=1/ε blow-up (LDR slope bounds irrelevant)
- LP simplex: exact, no IPM numerical artifacts (no μ offset correction needed)
- Much faster solve

See docs/partial_robust_full_spec.md for full mathematical derivation.

Architecture (same as build_primal_isp.jl):
| Parameter location | LP ISP                      |
|--------------------|---------------------------  |
| α                  | objective (μ coefficient)    |
| x,h,λ,ψ0          | constraint RHS (McCormick)  |
| Cut extraction     | value(μ) → subgradient      |
"""

using JuMP
using LinearAlgebra


# =============================================================================
# Builder functions
# =============================================================================

"""
    build_lp_isp_leader(network, xi_bar_s, ϕU, v_param, true_S, x_sol; optimizer)

Leader LP ISP for ε̂=0 case. Manuscript (16e,16j,16k,16g) at ζ=0.

Variables: ηhat, μhat, π̂₀, ϕ̂₀, ψ̂₀ (intercepts only, no LDR slopes).
Parameters: α (inner, via set_objective_coefficient), x (outer, via McCormick RHS).
"""
function build_lp_isp_leader(network, xi_bar_s, ϕU, v_param, true_S, x_sol; optimizer)
    num_arcs = length(xi_bar_s)
    N_trunc = network.N  # already source-removed: (num_nodes-1) × (num_arcs+1)
    Ny = N_trunc[:, 1:num_arcs]
    Nts = N_trunc[:, end]
    nv1 = size(Ny, 1)  # = num_nodes - 1

    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))

    # --- Variables: intercepts only (no slopes, no SDP/SOC) ---
    @variable(model, ηhat >= 0)
    @variable(model, μhat[1:num_arcs] >= 0)
    @variable(model, Πhat_0[1:nv1] >= 0)
    @variable(model, Φhat_0[1:num_arcs] >= 0)
    @variable(model, Ψhat_0[1:num_arcs] >= 0)

    # --- (16e at ζ=0): epigraph ---
    @constraint(model,
        sum((Φhat_0[k] - v_param * Ψhat_0[k]) * xi_bar_s[k] for k in 1:num_arcs) <= ηhat)

    # --- (16j at ζ=0): flow dual feasibility ---
    @constraint(model, [k=1:num_arcs],
        sum(Ny[j,k] * Πhat_0[j] for j in 1:nv1) + Φhat_0[k] >= 0)
    @constraint(model,
        sum(Nts[j] * Πhat_0[j] for j in 1:nv1) >= 1.0)

    # --- (16k at ζ=0): μ coupling ---
    @constraint(model, [k=1:num_arcs], μhat[k] >= Φhat_0[k])

    # --- (16g intercept): McCormick on ψ̂₀ = ϕ̂₀ · x ---
    @constraint(model, con_bigM1[k=1:num_arcs], Ψhat_0[k] <= ϕU * x_sol[k])
    @constraint(model, [k=1:num_arcs], Ψhat_0[k] <= Φhat_0[k])
    @constraint(model, con_bigM3[k=1:num_arcs], Φhat_0[k] <= Ψhat_0[k] + ϕU * (1 - x_sol[k]))

    # --- Objective: (1/S) * ηhat + Σ α_k * μhat_k ---
    # α coefficients initialized to 0, updated via set_objective_coefficient per inner iter
    @objective(model, Min, (1/true_S) * ηhat)

    vars = Dict(
        :ηhat => ηhat, :μhat => μhat, :Πhat_0 => Πhat_0,
        :Φhat_0 => Φhat_0, :Ψhat_0 => Ψhat_0,
        :con_bigM1 => con_bigM1, :con_bigM3 => con_bigM3,
        :xi_bar_s => xi_bar_s, :v_param => v_param, :true_S => true_S,
    )
    return model, vars
end


"""
    build_lp_isp_follower(network, xi_bar_s, ϕU, v_param, true_S, x_sol, h_sol, λ_sol, ψ0_sol; optimizer)

Follower LP ISP for ε̃=0 case. Manuscript (16f,16l,16m,16h) at ζ=0.

Variables: ηtilde (free!), μtilde, π̃₀, ϕ̃₀, ψ̃₀, ỹ₀, ỹ₀ᵗˢ.
Parameters: α (inner), x,h,λ,ψ⁰ (outer, via McCormick/capacity/ts_dual RHS).
"""
function build_lp_isp_follower(network, xi_bar_s, ϕU, v_param, true_S,
                                x_sol, h_sol, λ_sol, ψ0_sol; optimizer)
    num_arcs = length(xi_bar_s)
    N_trunc = network.N
    Ny = N_trunc[:, 1:num_arcs]
    Nts = N_trunc[:, end]
    nv1 = size(Ny, 1)

    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))

    # --- Variables ---
    @variable(model, ηtilde)                              # FREE (not ≥ 0), manuscript (27)
    @variable(model, μtilde[1:num_arcs] >= 0)
    @variable(model, Πtilde_0[1:nv1] >= 0)
    @variable(model, Φtilde_0[1:num_arcs] >= 0)
    @variable(model, Ψtilde_0[1:num_arcs] >= 0)
    @variable(model, Ytilde_0[1:num_arcs] >= 0)           # ỹ₀ (flow intercept)
    @variable(model, Ytilde_0_ts >= 0)                    # ỹ₀ᵗˢ (total flow intercept)

    # --- (16f at ζ=0): epigraph (ηtilde free) ---
    @constraint(model,
        sum((Φtilde_0[k] - v_param * Ψtilde_0[k]) * xi_bar_s[k] for k in 1:num_arcs) -
        Ytilde_0_ts <= ηtilde)

    # --- (16l at ζ=0): flow dual feasibility ---
    @constraint(model, [k=1:num_arcs],
        sum(Ny[j,k] * Πtilde_0[j] for j in 1:nv1) + Φtilde_0[k] >= 0)

    # ts dual: Ntsᵀ π̃₀ ≥ λ
    @constraint(model, con_ts_dual,
        sum(Nts[j] * Πtilde_0[j] for j in 1:nv1) >= λ_sol)

    # flow conservation: -Ny ỹ₀ - Nts ỹ₀ᵗˢ ≥ 0
    @constraint(model, [j=1:nv1],
        -sum(Ny[j,k] * Ytilde_0[k] for k in 1:num_arcs) - Nts[j] * Ytilde_0_ts >= 0)

    # capacity at ζ=0: ỹ₀ₖ ≤ hₖ + (λ - vₖψ⁰ₖ)ξ̄ₖ
    @constraint(model, con_capacity[k=1:num_arcs],
        Ytilde_0[k] <= h_sol[k] + (λ_sol - v_param * ψ0_sol[k]) * xi_bar_s[k])

    # --- (16m at ζ=0): μ coupling ---
    @constraint(model, [k=1:num_arcs], μtilde[k] >= Φtilde_0[k])

    # --- (16h intercept): McCormick on ψ̃₀ = ϕ̃₀ · x ---
    @constraint(model, con_bigM1[k=1:num_arcs], Ψtilde_0[k] <= ϕU * x_sol[k])
    @constraint(model, [k=1:num_arcs], Ψtilde_0[k] <= Φtilde_0[k])
    @constraint(model, con_bigM3[k=1:num_arcs], Φtilde_0[k] <= Ψtilde_0[k] + ϕU * (1 - x_sol[k]))

    # --- Objective ---
    @objective(model, Min, (1/true_S) * ηtilde)

    vars = Dict(
        :ηtilde => ηtilde, :μtilde => μtilde,
        :Πtilde_0 => Πtilde_0, :Φtilde_0 => Φtilde_0, :Ψtilde_0 => Ψtilde_0,
        :Ytilde_0 => Ytilde_0, :Ytilde_0_ts => Ytilde_0_ts,
        :con_bigM1 => con_bigM1, :con_bigM3 => con_bigM3,
        :con_ts_dual => con_ts_dual, :con_capacity => con_capacity,
        :xi_bar_s => xi_bar_s, :v_param => v_param, :true_S => true_S,
    )
    return model, vars
end


# =============================================================================
# Optimize functions
# =============================================================================

"""
    lp_isp_leader_optimize!(model, vars; isp_data, α_sol)

Update α in objective, solve LP leader ISP, extract inner cut (intercept + α'μ̂).
LP simplex → exact, hard assert with 1e-6 tolerance (no IPM artifact correction needed).
"""
function lp_isp_leader_optimize!(model, vars; isp_data, α_sol)
    true_S = vars[:true_S]
    num_arcs = length(α_sol)

    for k in 1:num_arcs
        set_objective_coefficient(model, vars[:μhat][k], α_sol[k])
    end
    optimize!(model)
    st = MOI.get(model, MOI.TerminationStatus())
    if st != MOI.OPTIMAL
        error("LP leader ISP not optimal: $st")
    end

    μhat_val = [value(vars[:μhat][k]) for k in 1:num_arcs]
    ηhat_val = value(vars[:ηhat])
    intercept = (1/true_S) * ηhat_val
    obj_val = intercept + dot(α_sol, μhat_val)

    @assert abs(obj_val - objective_value(model)) < 1e-6 (
        "LP leader ISP duality gap: obj_val=$obj_val vs model=$(objective_value(model))")

    return (:OptimalityCut, Dict(:μhat => μhat_val, :intercept => intercept, :obj_val => obj_val))
end


"""
    lp_isp_follower_optimize!(model, vars; isp_data, α_sol)

Update α in objective, solve LP follower ISP, extract inner cut (intercept + α'μ̃).
"""
function lp_isp_follower_optimize!(model, vars; isp_data, α_sol)
    true_S = vars[:true_S]
    num_arcs = length(α_sol)

    for k in 1:num_arcs
        set_objective_coefficient(model, vars[:μtilde][k], α_sol[k])
    end
    optimize!(model)
    st = MOI.get(model, MOI.TerminationStatus())
    if st != MOI.OPTIMAL
        @infiltrate
        error("LP follower ISP not optimal: $st")
    end

    μtilde_val = [value(vars[:μtilde][k]) for k in 1:num_arcs]
    ηtilde_val = value(vars[:ηtilde])
    intercept = (1/true_S) * ηtilde_val
    obj_val = intercept + dot(α_sol, μtilde_val)

    @assert abs(obj_val - objective_value(model)) < 1e-6 (
        "LP follower ISP duality gap: obj_val=$obj_val vs model=$(objective_value(model))")

    return (:OptimalityCut, Dict(:μtilde => μtilde_val, :intercept => intercept, :obj_val => obj_val))
end


# =============================================================================
# Parameter update functions (outer iteration)
# =============================================================================

"""
    update_lp_leader_params!(model, vars; x_sol, ϕU)

Update McCormick RHS for new x from OMP. Only x appears in leader LP ISP parameter.
"""
function update_lp_leader_params!(model, vars; x_sol, ϕU)
    for k in 1:length(x_sol)
        set_normalized_rhs(vars[:con_bigM1][k], ϕU * x_sol[k])
        set_normalized_rhs(vars[:con_bigM3][k], ϕU * (1 - x_sol[k]))
    end
end


"""
    update_lp_follower_params!(model, vars; x_sol, h_sol, λ_sol, ψ0_sol, xi_bar_s, v_param, ϕU)

Update McCormick (x) + capacity (h,λ,ψ⁰) + ts_dual (λ) RHS for follower LP ISP.
"""
function update_lp_follower_params!(model, vars; x_sol, h_sol, λ_sol, ψ0_sol, xi_bar_s, v_param, ϕU)
    num_arcs = length(x_sol)
    for k in 1:num_arcs
        set_normalized_rhs(vars[:con_bigM1][k], ϕU * x_sol[k])
        set_normalized_rhs(vars[:con_bigM3][k], ϕU * (1 - x_sol[k]))
        set_normalized_rhs(vars[:con_capacity][k],
            h_sol[k] + (λ_sol - v_param * ψ0_sol[k]) * xi_bar_s[k])
    end
    set_normalized_rhs(vars[:con_ts_dual], λ_sol)
end


# =============================================================================
# Initialize per-scenario instances
# =============================================================================

"""
    initialize_lp_isp_leader(network, S, ϕU, v_param, uncertainty_set, x_sol, true_S; optimizer)

Create per-scenario LP leader ISP instances. Same pattern as initialize_primal_isp.
"""
function initialize_lp_isp_leader(network, S, ϕU, v_param, uncertainty_set, x_sol, true_S; optimizer)
    xi_bar = uncertainty_set[:xi_bar]
    instances = Dict{Int, Tuple{Model, Dict}}()
    for s in 1:S
        model, vars = build_lp_isp_leader(network, xi_bar[s], ϕU, v_param, true_S, x_sol;
                                           optimizer=optimizer)
        instances[s] = (model, vars)
    end
    return instances
end


"""
    initialize_lp_isp_follower(network, S, ϕU, v_param, uncertainty_set,
                                x_sol, h_sol, λ_sol, ψ0_sol, true_S; optimizer)

Create per-scenario LP follower ISP instances.
"""
function initialize_lp_isp_follower(network, S, ϕU, v_param, uncertainty_set,
                                     x_sol, h_sol, λ_sol, ψ0_sol, true_S; optimizer)
    xi_bar = uncertainty_set[:xi_bar]
    instances = Dict{Int, Tuple{Model, Dict}}()
    for s in 1:S
        model, vars = build_lp_isp_follower(network, xi_bar[s], ϕU, v_param, true_S,
                                             x_sol, h_sol, λ_sol, ψ0_sol;
                                             optimizer=optimizer)
        instances[s] = (model, vars)
    end
    return instances
end


# =============================================================================
# Batch update functions (called each outer iteration)
# =============================================================================

"""
    update_lp_isp_leader_parameters!(instances; x_sol, ϕU)

Update all per-scenario LP leader ISP instances with new x from OMP.
"""
function update_lp_isp_leader_parameters!(instances; x_sol, ϕU)
    for (s, (model, vars)) in instances
        update_lp_leader_params!(model, vars; x_sol=x_sol, ϕU=ϕU)
    end
end


"""
    update_lp_isp_follower_parameters!(instances; x_sol, h_sol, λ_sol, ψ0_sol, isp_data)

Update all per-scenario LP follower ISP instances with new (x,h,λ,ψ⁰) from OMP.
"""
function update_lp_isp_follower_parameters!(instances; x_sol, h_sol, λ_sol, ψ0_sol, isp_data)
    ϕU = isp_data[:ϕU_tilde]
    v_param = isp_data[:v]
    xi_bar = isp_data[:uncertainty_set][:xi_bar]
    for (s, (model, vars)) in instances
        update_lp_follower_params!(model, vars;
            x_sol=x_sol, h_sol=h_sol, λ_sol=λ_sol, ψ0_sol=ψ0_sol,
            xi_bar_s=xi_bar[s], v_param=v_param, ϕU=ϕU)
    end
end


# =============================================================================
# Outer cut extraction (from LP ISP shadow prices)
# =============================================================================

"""
    evaluate_lp_leader_outer_cut(model, vars; x_sol, ϕU)

Extract outer cut coefficients from LP leader ISP via shadow_price.
x appears in McCormick RHS only → sensitivity via con_bigM1/con_bigM3 duals.
Returns Dict(:intercept, :coeff_x, :coeff_h, :coeff_λ, :coeff_ψ0).
"""
function evaluate_lp_leader_outer_cut(model, vars; x_sol, ϕU)
    num_arcs = length(x_sol)
    bigM1_duals = [shadow_price(vars[:con_bigM1][k]) for k in 1:num_arcs]
    bigM3_duals = [shadow_price(vars[:con_bigM3][k]) for k in 1:num_arcs]

    # ∂obj/∂xₖ = ϕU · ∂obj/∂(RHS of bigM1) - ϕU · ∂obj/∂(RHS of bigM3)
    coeff_x = [ϕU * bigM1_duals[k] - ϕU * bigM3_duals[k] for k in 1:num_arcs]
    intercept = objective_value(model) - dot(coeff_x, x_sol)

    # Tightness assertion
    cut_value = intercept + dot(coeff_x, x_sol)
    @assert abs(cut_value - objective_value(model)) < 1e-6 (
        "LP leader outer cut tightness: cut=$cut_value vs obj=$(objective_value(model))")

    return Dict(:intercept => intercept, :coeff_x => coeff_x,
                :coeff_h => zeros(num_arcs), :coeff_λ => 0.0, :coeff_ψ0 => zeros(num_arcs))
end


"""
    evaluate_lp_follower_outer_cut(model, vars; x_sol, h_sol, λ_sol, ψ0_sol, xi_bar_s, v_param, ϕU)

Extract outer cut coefficients from LP follower ISP via shadow_price.
Parameters in RHS: x (McCormick), h/λ/ψ⁰ (capacity), λ (ts_dual).
Returns Dict(:intercept, :coeff_x, :coeff_h, :coeff_λ, :coeff_ψ0).
"""
function evaluate_lp_follower_outer_cut(model, vars; x_sol, h_sol, λ_sol, ψ0_sol,
                                         xi_bar_s, v_param, ϕU)
    num_arcs = length(x_sol)
    cap_duals   = [shadow_price(vars[:con_capacity][k]) for k in 1:num_arcs]
    bigM1_duals = [shadow_price(vars[:con_bigM1][k]) for k in 1:num_arcs]
    bigM3_duals = [shadow_price(vars[:con_bigM3][k]) for k in 1:num_arcs]
    ts_dual     = shadow_price(vars[:con_ts_dual])

    # ∂obj/∂xₖ: McCormick RHS sensitivity only (capacity RHS에 x 없음)
    coeff_x = [ϕU * bigM1_duals[k] - ϕU * bigM3_duals[k] for k in 1:num_arcs]
    # ∂obj/∂hₖ: capacity RHS에 hₖ가 coefficient 1로 등장
    coeff_h = copy(cap_duals)
    # ∂obj/∂λ: ts_dual RHS=λ, capacity RHS에 ξ̄ₖ 만큼 등장
    coeff_λ = ts_dual + dot(xi_bar_s, cap_duals)
    # ∂obj/∂ψ⁰ₖ: capacity RHS에 -vₖξ̄ₖ 만큼 등장
    coeff_ψ0 = [-v_param * xi_bar_s[k] * cap_duals[k] for k in 1:num_arcs]

    intercept = objective_value(model) - dot(coeff_x, x_sol) - dot(coeff_h, h_sol) -
                coeff_λ * λ_sol - dot(coeff_ψ0, ψ0_sol)

    # Tightness assertion
    cut_value = intercept + dot(coeff_x, x_sol) + dot(coeff_h, h_sol) +
                coeff_λ * λ_sol + dot(coeff_ψ0, ψ0_sol)
    @assert abs(cut_value - objective_value(model)) < 1e-6 (
        "LP follower outer cut tightness: cut=$cut_value vs obj=$(objective_value(model))")

    return Dict(:intercept => intercept, :coeff_x => coeff_x,
                :coeff_h => coeff_h, :coeff_λ => coeff_λ, :coeff_ψ0 => coeff_ψ0)
end


# =============================================================================
# Partial outer cut extraction (combines LP + SDP sides)
# =============================================================================

"""
    extract_partial_outer_cut(lp_instances, sdp_instances, S, isp_mode;
        x_sol, h_sol, λ_sol, ψ0_sol, isp_data)

Extract combined outer cut info from current LP + SDP ISP solutions.
LP side: shadow_price → linear coefficients.
SDP side: value.() → standard coefficient matrices (Uhat1/3 or Utilde1/3/Z/β).

Returns Dict suitable for add_partial_optimality_cuts!.
"""
function extract_partial_outer_cut(lp_instances, sdp_instances, S, isp_mode;
    x_sol, h_sol, λ_sol, ψ0_sol, isp_data)

    num_arcs = length(x_sol)
    ϕU_hat = isp_data[:ϕU_hat]
    ϕU_tilde = isp_data[:ϕU_tilde]
    v_param = isp_data[:v]
    xi_bar = isp_data[:uncertainty_set][:xi_bar]

    if isp_mode == :partial_hat0
        # Leader = LP, Follower = SDP (dual)
        lp_leader_cuts = Dict{Int, Dict}()
        for s in 1:S
            lp_leader_cuts[s] = evaluate_lp_leader_outer_cut(
                lp_instances[s][1], lp_instances[s][2];
                x_sol=x_sol, ϕU=ϕU_hat)
        end

        # SDP follower: extract standard coefficients
        Utilde1 = cat([value.(sdp_instances[s][2][:Utilde1]) for s in 1:S]...; dims=1)
        Utilde3 = cat([value.(sdp_instances[s][2][:Utilde3]) for s in 1:S]...; dims=1)
        Ztilde1_3 = cat([value.(sdp_instances[s][2][:Ztilde1_3]) for s in 1:S]...; dims=1)
        βtilde1_1 = cat([value.(sdp_instances[s][2][:βtilde1_1]) for s in 1:S]...; dims=1)
        βtilde1_3 = cat([value.(sdp_instances[s][2][:βtilde1_3]) for s in 1:S]...; dims=1)
        intercept_f = [value.(sdp_instances[s][2][:intercept]) for s in 1:S]

        return Dict(
            :mode => :partial_hat0,
            :lp_leader => lp_leader_cuts,
            :Utilde1 => Utilde1, :Utilde3 => Utilde3,
            :Ztilde1_3 => Ztilde1_3, :βtilde1_1 => βtilde1_1, :βtilde1_3 => βtilde1_3,
            :intercept_f => intercept_f,
            # For cut_pool / debug logging
            :intercept_l => [lp_leader_cuts[s][:intercept] for s in 1:S],
        )

    elseif isp_mode == :partial_tilde0
        # Leader = SDP (dual), Follower = LP
        Uhat1 = cat([value.(sdp_instances[s][2][:Uhat1]) for s in 1:S]...; dims=1)
        Uhat3 = cat([value.(sdp_instances[s][2][:Uhat3]) for s in 1:S]...; dims=1)
        intercept_l = [value.(sdp_instances[s][2][:intercept]) for s in 1:S]

        lp_follower_cuts = Dict{Int, Dict}()
        for s in 1:S
            lp_follower_cuts[s] = evaluate_lp_follower_outer_cut(
                lp_instances[s][1], lp_instances[s][2];
                x_sol=x_sol, h_sol=h_sol, λ_sol=λ_sol, ψ0_sol=ψ0_sol,
                xi_bar_s=xi_bar[s], v_param=v_param, ϕU=ϕU_tilde)
        end

        return Dict(
            :mode => :partial_tilde0,
            :Uhat1 => Uhat1, :Uhat3 => Uhat3,
            :intercept_l => intercept_l,
            :lp_follower => lp_follower_cuts,
            # For cut_pool / debug logging
            :intercept_f => [lp_follower_cuts[s][:intercept] for s in 1:S],
        )
    else
        error("extract_partial_outer_cut: invalid isp_mode=$isp_mode")
    end
end


# =============================================================================
# LP-in-IMP: LP ISP를 IMP에 직접 흡수 (dual LP formulation)
# =============================================================================

"""
    build_imp_with_leader_lp(network, S, ϕU, v_param, w, uncertainty_set, x_sol; mip_optimizer)

IMP + leader dual LP (Case 1: ε̂=0). Leader LP 제약을 IMP에 직접 넣어 ISP-L 호출 제거.
t_1_l 제거, leader contribution은 inline dual LP objective로 exact 표현.
t_1_f(=t_f) 유지 (follower SDP cut용).

See docs/lp_in_imp_variant.md §3-4 for derivation.
"""
function build_imp_with_leader_lp(network, S, ϕU, v_param, w, uncertainty_set, x_sol; mip_optimizer)
    num_arcs = length(network.arcs) - 1
    N_trunc = network.N  # already source-removed: (num_nodes-1) × (num_arcs+1)
    Ny = N_trunc[:, 1:num_arcs]
    Nts = N_trunc[:, end]
    nv1 = size(Ny, 1)
    xi_bar = uncertainty_set[:xi_bar]
    flow_upper = sum(sum(xi_bar[s] for s in 1:S))

    model = Model(optimizer_with_attributes(mip_optimizer, MOI.Silent() => true))

    # ===== 기존 IMP 변수 (t_1_l 제거) =====
    @variable(model, t_1_f[s=1:S], upper_bound=flow_upper)   # follower epigraph (SDP cuts)
    @variable(model, α[k=1:num_arcs], lower_bound=0.0, upper_bound=w/S)
    @constraint(model, sum(α) == w*(1/S))

    # ===== Leader dual LP 변수 (per scenario, inline) =====
    @variable(model, τhat[s=1:S] >= 0)
    @variable(model, Zhat1_1[s=1:S, k=1:num_arcs] >= 0)     # flow dual
    @variable(model, βhat1_ts[s=1:S] >= 0)                   # ts flow dual
    @variable(model, Uhat1[s=1:S, k=1:num_arcs] >= 0)        # McCormick dual 1
    @variable(model, Uhat2[s=1:S, k=1:num_arcs] >= 0)        # McCormick dual 2
    @variable(model, Uhat3[s=1:S, k=1:num_arcs] >= 0)        # McCormick dual 3

    for s in 1:S
        ξ̄ = xi_bar[s]
        # (DL1) τ̂ ≤ 1/S
        @constraint(model, τhat[s] <= 1/S)
        # (DL2) Ny · Zhat1_1 + Nts · βhat1_ts ≤ 0  (nv1 constraints)
        @constraint(model, [j=1:nv1],
            sum(Ny[j,k] * Zhat1_1[s,k] for k in 1:num_arcs) + Nts[j] * βhat1_ts[s] <= 0)
        # (DL3) −ξ̄ₖτ̂ + Zhat1_1ₖ + Uhat2ₖ − Uhat3ₖ ≤ αₖ  (α coupling)
        @constraint(model, [k=1:num_arcs],
            -ξ̄[k] * τhat[s] + Zhat1_1[s,k] + Uhat2[s,k] - Uhat3[s,k] <= α[k])
        # (DL4) vξ̄ₖτ̂ − Uhat1ₖ − Uhat2ₖ + Uhat3ₖ = 0
        @constraint(model, [k=1:num_arcs],
            v_param * ξ̄[k] * τhat[s] - Uhat1[s,k] - Uhat2[s,k] + Uhat3[s,k] == 0)
    end

    # ===== 목적함수: leader dual obj + follower epigraph =====
    # Leader: (1/S) Σₛ [βhat1_ts − ϕU·x·Uhat1 − ϕU·(1−x)·Uhat3]
    # Follower: (1/S) Σₛ t_f[s]
    @objective(model, Max,
        (1/S) * sum(
            βhat1_ts[s]
            - ϕU * sum(x_sol[k] * Uhat1[s,k] for k in 1:num_arcs)
            - ϕU * sum((1-x_sol[k]) * Uhat3[s,k] for k in 1:num_arcs)
            + t_1_f[s]
            for s in 1:S))

    vars = Dict(
        :t_1_f => t_1_f, :α => α,
        :τhat => τhat, :Zhat1_1 => Zhat1_1, :βhat1_ts => βhat1_ts,
        :Uhat1 => Uhat1, :Uhat2 => Uhat2, :Uhat3 => Uhat3,
    )
    return model, vars
end


"""
    build_imp_with_follower_lp(network, S, ϕU, v_param, w, uncertainty_set,
                                x_sol, h_sol, λ_sol, ψ0_sol; mip_optimizer)

IMP + follower dual LP (Case 2: ε̃=0). Follower LP 제약을 IMP에 직접 넣어 ISP-F 호출 제거.
t_1_f 제거, follower contribution은 inline dual LP objective로 exact 표현.
t_1_l(=t_l) 유지 (leader SDP cut용).

See docs/lp_in_imp_variant.md §5 for derivation.
"""
function build_imp_with_follower_lp(network, S, ϕU, v_param, w, uncertainty_set,
                                     x_sol, h_sol, λ_sol, ψ0_sol; mip_optimizer)
    num_arcs = length(network.arcs) - 1
    N_trunc = network.N
    Ny = N_trunc[:, 1:num_arcs]
    Nts = N_trunc[:, end]
    nv1 = size(Ny, 1)
    xi_bar = uncertainty_set[:xi_bar]
    flow_upper = sum(sum(xi_bar[s] for s in 1:S))

    model = Model(optimizer_with_attributes(mip_optimizer, MOI.Silent() => true))

    # ===== 기존 IMP 변수 (t_1_f 제거) =====
    @variable(model, t_1_l[s=1:S], upper_bound=flow_upper)   # leader epigraph (SDP cuts)
    @variable(model, α[k=1:num_arcs], lower_bound=0.0, upper_bound=w/S)
    @constraint(model, sum(α) == w*(1/S))

    # ===== Follower dual LP 변수 (per scenario) =====
    @variable(model, τtilde[s=1:S] >= 0)
    @variable(model, Ztilde1_1[s=1:S, k=1:num_arcs] >= 0)    # flow dual
    @variable(model, βtilde1_ts[s=1:S] >= 0)                  # ts flow dual
    @variable(model, Ztilde_cy[s=1:S, j=1:nv1] >= 0)          # flow conservation dual
    @variable(model, Ztilde_cap[s=1:S, k=1:num_arcs] >= 0)    # capacity dual
    @variable(model, Utilde1[s=1:S, k=1:num_arcs] >= 0)       # McCormick dual 1
    @variable(model, Utilde2[s=1:S, k=1:num_arcs] >= 0)       # McCormick dual 2
    @variable(model, Utilde3[s=1:S, k=1:num_arcs] >= 0)       # McCormick dual 3

    for s in 1:S
        ξ̄ = xi_bar[s]
        # (DF1) τ̃ = 1/S  (equality: η̃ free → dual equality)
        @constraint(model, τtilde[s] == 1/S)
        # (DF2) Ny · Ztilde1_1 + Nts · βtilde1_ts ≤ 0
        @constraint(model, [j=1:nv1],
            sum(Ny[j,k] * Ztilde1_1[s,k] for k in 1:num_arcs) + Nts[j] * βtilde1_ts[s] <= 0)
        # (DF3) −ξ̄ₖτ̃ + Ztilde1_1ₖ + Utilde2ₖ − Utilde3ₖ ≤ αₖ  (α coupling)
        @constraint(model, [k=1:num_arcs],
            -ξ̄[k] * τtilde[s] + Ztilde1_1[s,k] + Utilde2[s,k] - Utilde3[s,k] <= α[k])
        # (DF4) vξ̄ₖτ̃ − Utilde1ₖ − Utilde2ₖ + Utilde3ₖ = 0
        @constraint(model, [k=1:num_arcs],
            v_param * ξ̄[k] * τtilde[s] - Utilde1[s,k] - Utilde2[s,k] + Utilde3[s,k] == 0)
        # (DF5) Nyᵀ · Ztilde_cy + Ztilde_cap ≥ 0
        @constraint(model, [k=1:num_arcs],
            sum(Ny[j,k] * Ztilde_cy[s,j] for j in 1:nv1) + Ztilde_cap[s,k] >= 0)
        # (DF6) τ̃ + Ntsᵀ · Ztilde_cy ≥ 0
        @constraint(model, τtilde[s] + sum(Nts[j] * Ztilde_cy[s,j] for j in 1:nv1) >= 0)
    end

    # ===== 목적함수: leader epigraph + follower dual obj =====
    # Leader: (1/S) Σₛ t_l[s]
    # Follower: (1/S) Σₛ [λ·βtilde1_ts − Σₖ Cₖ·Ztilde_cap − ϕU·x·Utilde1 − ϕU·(1−x)·Utilde3]
    @objective(model, Max,
        (1/S) * sum(
            t_1_l[s]
            + λ_sol * βtilde1_ts[s]
            - sum((h_sol[k] + (λ_sol - v_param * ψ0_sol[k]) * xi_bar[s][k]) * Ztilde_cap[s,k]
                  for k in 1:num_arcs)
            - ϕU * sum(x_sol[k] * Utilde1[s,k] for k in 1:num_arcs)
            - ϕU * sum((1-x_sol[k]) * Utilde3[s,k] for k in 1:num_arcs)
            for s in 1:S))

    vars = Dict(
        :t_1_l => t_1_l, :α => α,
        :τtilde => τtilde, :Ztilde1_1 => Ztilde1_1, :βtilde1_ts => βtilde1_ts,
        :Ztilde_cy => Ztilde_cy, :Ztilde_cap => Ztilde_cap,
        :Utilde1 => Utilde1, :Utilde2 => Utilde2, :Utilde3 => Utilde3,
    )
    return model, vars
end


# =============================================================================
# LP-in-IMP: Outer iteration parameter updates
# =============================================================================

"""
    update_imp_leader_lp!(model, vars; x_sol, ϕU, S)

Update IMP(+leader LP) objective coefficients when x changes (outer iteration).
Only Uhat1, Uhat3 coefficients depend on x.
"""
function update_imp_leader_lp!(model, vars; x_sol, ϕU, S)
    num_arcs = length(x_sol)
    for s in 1:S, k in 1:num_arcs
        set_objective_coefficient(model, vars[:Uhat1][s,k], -(1/S) * ϕU * x_sol[k])
        set_objective_coefficient(model, vars[:Uhat3][s,k], -(1/S) * ϕU * (1 - x_sol[k]))
    end
end


"""
    update_imp_follower_lp!(model, vars; x_sol, h_sol, λ_sol, ψ0_sol, ϕU, v_param, xi_bar, S)

Update IMP(+follower LP) objective coefficients when (x,h,λ,ψ⁰) change (outer iteration).
Utilde1, Utilde3 depend on x; βtilde1_ts depends on λ; Ztilde_cap depends on h,λ,ψ⁰.
"""
function update_imp_follower_lp!(model, vars; x_sol, h_sol, λ_sol, ψ0_sol,
                                  ϕU, v_param, xi_bar, S)
    num_arcs = length(x_sol)
    for s in 1:S
        set_objective_coefficient(model, vars[:βtilde1_ts][s], (1/S) * λ_sol)
        for k in 1:num_arcs
            Cₖ = h_sol[k] + (λ_sol - v_param * ψ0_sol[k]) * xi_bar[s][k]
            set_objective_coefficient(model, vars[:Ztilde_cap][s,k], -(1/S) * Cₖ)
            set_objective_coefficient(model, vars[:Utilde1][s,k], -(1/S) * ϕU * x_sol[k])
            set_objective_coefficient(model, vars[:Utilde3][s,k], -(1/S) * ϕU * (1 - x_sol[k]))
        end
    end
end


# =============================================================================
# LP-in-IMP: Outer cut extraction (reads LP vars directly from IMP solution)
# =============================================================================

"""
    extract_lp_in_imp_outer_cut(imp_vars, sdp_instances, S, isp_mode;
        x_sol, h_sol, λ_sol, ψ0_sol, isp_data)

Extract combined outer cut from IMP(+LP) solution + SDP ISP solutions.
LP side: read variable values directly from IMP (no separate LP ISP needed).
SDP side: value.() from SDP ISP instances (same as extract_partial_outer_cut).

Returns Dict suitable for add_partial_optimality_cuts! (reuses :partial_hat0/:partial_tilde0 format).
"""
function extract_lp_in_imp_outer_cut(imp_vars, sdp_instances, S, isp_mode;
    x_sol, h_sol, λ_sol, ψ0_sol, isp_data)

    num_arcs = length(x_sol)
    ϕU_hat = isp_data[:ϕU_hat]
    ϕU_tilde = isp_data[:ϕU_tilde]
    v_param = isp_data[:v]
    xi_bar = isp_data[:uncertainty_set][:xi_bar]

    if isp_mode == :lp_in_imp_hat0
        # Leader LP in IMP → read leader vars from IMP solution
        lp_leader_cuts = Dict{Int, Dict}()
        for s in 1:S
            Uhat1_val = [value(imp_vars[:Uhat1][s,k]) for k in 1:num_arcs]
            Uhat3_val = [value(imp_vars[:Uhat3][s,k]) for k in 1:num_arcs]
            βhat1_ts_val = value(imp_vars[:βhat1_ts][s])

            # Leader outer cut: ∂obj/∂x from McCormick structure
            coeff_x = [ϕU_hat * (Uhat3_val[k] - Uhat1_val[k]) for k in 1:num_arcs]
            # Leader contribution at current x
            leader_obj_s = βhat1_ts_val -
                ϕU_hat * sum(x_sol[k] * Uhat1_val[k] for k in 1:num_arcs) -
                ϕU_hat * sum((1-x_sol[k]) * Uhat3_val[k] for k in 1:num_arcs)
            intercept = leader_obj_s - dot(coeff_x, x_sol)

            lp_leader_cuts[s] = Dict(
                :intercept => intercept, :coeff_x => coeff_x,
                :coeff_h => zeros(num_arcs), :coeff_λ => 0.0, :coeff_ψ0 => zeros(num_arcs))
        end

        # SDP follower: same as extract_partial_outer_cut
        Utilde1 = cat([value.(sdp_instances[s][2][:Utilde1]) for s in 1:S]...; dims=1)
        Utilde3 = cat([value.(sdp_instances[s][2][:Utilde3]) for s in 1:S]...; dims=1)
        Ztilde1_3 = cat([value.(sdp_instances[s][2][:Ztilde1_3]) for s in 1:S]...; dims=1)
        βtilde1_1 = cat([value.(sdp_instances[s][2][:βtilde1_1]) for s in 1:S]...; dims=1)
        βtilde1_3 = cat([value.(sdp_instances[s][2][:βtilde1_3]) for s in 1:S]...; dims=1)
        intercept_f = [value.(sdp_instances[s][2][:intercept]) for s in 1:S]

        return Dict(
            :mode => :partial_hat0,
            :lp_leader => lp_leader_cuts,
            :Utilde1 => Utilde1, :Utilde3 => Utilde3,
            :Ztilde1_3 => Ztilde1_3, :βtilde1_1 => βtilde1_1, :βtilde1_3 => βtilde1_3,
            :intercept_f => intercept_f,
            :intercept_l => [lp_leader_cuts[s][:intercept] for s in 1:S],
        )

    elseif isp_mode == :lp_in_imp_tilde0
        # SDP leader: same as extract_partial_outer_cut
        Uhat1 = cat([value.(sdp_instances[s][2][:Uhat1]) for s in 1:S]...; dims=1)
        Uhat3 = cat([value.(sdp_instances[s][2][:Uhat3]) for s in 1:S]...; dims=1)
        intercept_l = [value.(sdp_instances[s][2][:intercept]) for s in 1:S]

        # Follower LP in IMP → read follower vars from IMP solution
        lp_follower_cuts = Dict{Int, Dict}()
        for s in 1:S
            Utilde1_val = [value(imp_vars[:Utilde1][s,k]) for k in 1:num_arcs]
            Utilde3_val = [value(imp_vars[:Utilde3][s,k]) for k in 1:num_arcs]
            βtilde1_ts_val = value(imp_vars[:βtilde1_ts][s])
            Ztilde_cap_val = [value(imp_vars[:Ztilde_cap][s,k]) for k in 1:num_arcs]

            # ∂obj/∂x: McCormick sensitivity
            coeff_x = [ϕU_tilde * (Utilde3_val[k] - Utilde1_val[k]) for k in 1:num_arcs]
            # ∂obj/∂h: capacity dual (Ztilde_cap coefficient = -1 in obj)
            coeff_h = [-Ztilde_cap_val[k] for k in 1:num_arcs]
            # ∂obj/∂λ: βtilde1_ts + Σₖ ξ̄ₖ·(-Ztilde_cap_k)  (from Cₖ = h + (λ-vψ⁰)ξ̄)
            coeff_λ = βtilde1_ts_val + sum(-xi_bar[s][k] * Ztilde_cap_val[k] for k in 1:num_arcs)
            # ∂obj/∂ψ⁰ₖ: from Cₖ = h + (λ-vψ⁰)ξ̄  → ∂Cₖ/∂ψ⁰ₖ = v·ξ̄ₖ
            coeff_ψ0 = [v_param * xi_bar[s][k] * Ztilde_cap_val[k] for k in 1:num_arcs]

            # Follower contribution at current (x,h,λ,ψ⁰)
            follower_obj_s = λ_sol * βtilde1_ts_val -
                sum((h_sol[k] + (λ_sol - v_param * ψ0_sol[k]) * xi_bar[s][k]) * Ztilde_cap_val[k]
                    for k in 1:num_arcs) -
                ϕU_tilde * sum(x_sol[k] * Utilde1_val[k] for k in 1:num_arcs) -
                ϕU_tilde * sum((1-x_sol[k]) * Utilde3_val[k] for k in 1:num_arcs)
            intercept = follower_obj_s - dot(coeff_x, x_sol) - dot(coeff_h, h_sol) -
                        coeff_λ * λ_sol - dot(coeff_ψ0, ψ0_sol)

            lp_follower_cuts[s] = Dict(
                :intercept => intercept, :coeff_x => coeff_x,
                :coeff_h => coeff_h, :coeff_λ => coeff_λ, :coeff_ψ0 => coeff_ψ0)
        end

        return Dict(
            :mode => :partial_tilde0,
            :Uhat1 => Uhat1, :Uhat3 => Uhat3,
            :intercept_l => intercept_l,
            :lp_follower => lp_follower_cuts,
            :intercept_f => [lp_follower_cuts[s][:intercept] for s in 1:S],
        )
    else
        error("extract_lp_in_imp_outer_cut: invalid isp_mode=$isp_mode")
    end
end
