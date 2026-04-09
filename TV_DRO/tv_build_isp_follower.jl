"""
tv_build_isp_follower.jl — Follower Inner Subproblem (ISP-F) for TV-DRO.

All-scenario pooled LP. Constraints F1-F14 from tv_derivation_revised.md §9.4.

Given α_k from IMP, x̄_k, h̄_k, λ̄, ψ̄⁰_k from OMP:
  Z^F = max  λ̄ Σ_s σ̃^s - Σ_{s,k} [h̄_k + (λ̄ - v_k ψ̄⁰_k) ξ̄_k^s] β_k^s
             - φ^U Σ_{s,k} x̄_k ρ̃¹_{k,s} - φ^U Σ_{s,k} (1-x̄_k) ρ̃³_{k,s}
  s.t. F1-F14

All OMP variables (x̄, h̄, λ̄, ψ̄⁰) appear ONLY in objective → build once, update obj only.
Sensitivity ∂Z^F/∂α_k comes from dual values of F10-F13 (α appears in RHS).
"""

using JuMP
using LinearAlgebra


"""
    build_tv_isp_follower(tv::TVData, x_bar, h_bar, lambda_bar, psi0_bar; optimizer)

Build follower ISP LP (F1-F14). All scenarios pooled.
OMP variables appear only in objective → can update objective without rebuilding constraints.

# Arguments
- `tv`: TVData struct
- `x_bar`: |A| vector, current OMP x
- `h_bar`: |A| vector, current OMP h
- `lambda_bar`: scalar, current OMP λ
- `psi0_bar`: |A| vector, current OMP ψ⁰
- `optimizer`: LP solver (e.g., HiGHS.Optimizer)
"""
function build_tv_isp_follower(tv::TVData, x_bar::Vector{Float64},
                                h_bar::Vector{Float64}, lambda_bar::Float64,
                                psi0_bar::Vector{Float64}; optimizer)
    S = tv.S
    K = tv.num_arcs
    m = tv.nv1
    Ny = tv.Ny
    Nts = tv.Nts
    q = tv.q_hat
    ε = tv.eps_tilde
    ξ = tv.xi_bar
    v = tv.v
    φ_U = tv.phi_U

    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))

    # --- Variables (F14) ---
    @variable(model, σ_tilde[1:S] >= 0)
    @variable(model, u_tilde[1:K, 1:S] >= 0)
    @variable(model, ω[1:m, 1:S] >= 0)
    @variable(model, β[1:K, 1:S] >= 0)
    @variable(model, d[1:S] >= 0)
    @variable(model, e[1:S] >= 0)
    @variable(model, d_nu[1:S, 1:K] >= 0)
    @variable(model, e_nu[1:S, 1:K] >= 0)
    # McCormick duals for ψ̃ ≈ x·φ̃
    @variable(model, ρ_tilde_1[1:K, 1:S] >= 0)
    @variable(model, ρ_tilde_2[1:K, 1:S] >= 0)
    @variable(model, ρ_tilde_3[1:K, 1:S] >= 0)

    # --- (F1): N_y ũ^s + N_ts σ̃^s ≤ 0,  ∀s ---
    @constraint(model, F1[j=1:m, s=1:S],
        sum(Ny[j, k] * u_tilde[k, s] for k in 1:K) + Nts[j] * σ_tilde[s] <= 0)

    # --- (F2): -ξ̄_k^s d_s + ũ_k^s - d_{s,k}^ν + ρ̃²_{k,s} - ρ̃³_{k,s} ≤ 0,  ∀k,s ---
    @constraint(model, F2[k=1:K, s=1:S],
        -ξ[k, s] * d[s] + u_tilde[k, s] - d_nu[s, k]
        + ρ_tilde_2[k, s] - ρ_tilde_3[k, s] <= 0)

    # --- (F3): v_k ξ̄_k^s d_s - ρ̃¹_{k,s} - ρ̃²_{k,s} + ρ̃³_{k,s} ≤ 0,  ∀k,s ---
    @constraint(model, F3[k=1:K, s=1:S],
        v[k] * ξ[k, s] * d[s] - ρ_tilde_1[k, s]
        - ρ_tilde_2[k, s] + ρ_tilde_3[k, s] <= 0)

    # --- (F4): [N_yᵀ ω^s]_k + β_k^s ≥ 0,  ∀k,s ---
    @constraint(model, F4[k=1:K, s=1:S],
        sum(Ny[j, k] * ω[j, s] for j in 1:m) + β[k, s] >= 0)

    # --- (F5): N_tsᵀ ω^s ≥ d_s,  ∀s ---
    @constraint(model, F5[s=1:S],
        sum(Nts[j] * ω[j, s] for j in 1:m) >= d[s])

    # --- (F6): d_s - e_s ≤ q̂_s,  ∀s ---
    @constraint(model, F6[s=1:S], d[s] - e[s] <= q[s])

    # --- (F7): d_s + e_s ≥ q̂_s,  ∀s ---
    @constraint(model, F7[s=1:S], d[s] + e[s] >= q[s])

    # --- (F8): Σ_s e_s ≤ 2ε̃ ---
    @constraint(model, F8, sum(e[s] for s in 1:S) <= 2 * ε)

    # --- (F9): Σ_s d_s = 1 ---
    @constraint(model, F9, sum(d[s] for s in 1:S) == 1)

    # --- (F10): d_{s,k}^ν - e_{s,k}^ν ≤ q̂_s α_k,  ∀s,k ---
    @constraint(model, F10[s=1:S, k=1:K], d_nu[s, k] - e_nu[s, k] <= 0)

    # --- (F11): d_{s,k}^ν + e_{s,k}^ν ≥ q̂_s α_k,  ∀s,k ---
    @constraint(model, F11[s=1:S, k=1:K], d_nu[s, k] + e_nu[s, k] >= 0)

    # --- (F12): Σ_s e_{s,k}^ν ≤ 2ε̃ α_k,  ∀k ---
    @constraint(model, F12[k=1:K], sum(e_nu[s, k] for s in 1:S) <= 0)

    # --- (F13): Σ_s d_{s,k}^ν = α_k,  ∀k ---
    @constraint(model, F13[k=1:K], sum(d_nu[s, k] for s in 1:S) == 0)

    # --- Objective (expanded, all OMP vars in obj only) ---
    @objective(model, Max,
        lambda_bar * sum(σ_tilde[s] for s in 1:S)
        - sum((h_bar[k] + (lambda_bar - v[k] * psi0_bar[k]) * ξ[k, s]) * β[k, s]
              for k in 1:K, s in 1:S)
        - φ_U * sum(x_bar[k] * ρ_tilde_1[k, s] for k in 1:K, s in 1:S)
        - φ_U * sum((1.0 - x_bar[k]) * ρ_tilde_3[k, s] for k in 1:K, s in 1:S))

    vars = Dict(
        :σ_tilde => σ_tilde, :u_tilde => u_tilde,
        :ω => ω, :β => β,
        :d => d, :e => e, :d_nu => d_nu, :e_nu => e_nu,
        :ρ_tilde_1 => ρ_tilde_1, :ρ_tilde_2 => ρ_tilde_2, :ρ_tilde_3 => ρ_tilde_3,
        :F10 => F10, :F11 => F11, :F12 => F12, :F13 => F13,
    )
    return model, vars
end


"""
    update_tv_isp_follower_alpha!(model, vars, tv::TVData, α_sol::Vector{Float64})

Update RHS of F10-F13 for new α from IMP.
"""
function update_tv_isp_follower_alpha!(model, vars, tv::TVData, α_sol::Vector{Float64})
    S = tv.S
    K = tv.num_arcs
    q = tv.q_hat
    ε = tv.eps_tilde

    for s in 1:S, k in 1:K
        set_normalized_rhs(vars[:F10][s, k], q[s] * α_sol[k])
        set_normalized_rhs(vars[:F11][s, k], q[s] * α_sol[k])
    end
    for k in 1:K
        set_normalized_rhs(vars[:F12][k], 2 * ε * α_sol[k])
        set_normalized_rhs(vars[:F13][k], α_sol[k])
    end
end


"""
    update_tv_isp_follower_objective!(model, vars, tv::TVData,
                                       x_bar_new, h_bar_new, lambda_bar_new, psi0_bar_new)

Update objective for new OMP solution. No constraint rebuild needed.
"""
function update_tv_isp_follower_objective!(model, vars, tv::TVData,
                                            x_bar_new::Vector{Float64},
                                            h_bar_new::Vector{Float64},
                                            lambda_bar_new::Float64,
                                            psi0_bar_new::Vector{Float64})
    S = tv.S
    K = tv.num_arcs
    ξ = tv.xi_bar
    v = tv.v
    φ_U = tv.phi_U

    σ_tilde = vars[:σ_tilde]
    β = vars[:β]
    ρ_tilde_1 = vars[:ρ_tilde_1]
    ρ_tilde_3 = vars[:ρ_tilde_3]

    @objective(model, Max,
        lambda_bar_new * sum(σ_tilde[s] for s in 1:S)
        - sum((h_bar_new[k] + (lambda_bar_new - v[k] * psi0_bar_new[k]) * ξ[k, s]) * β[k, s]
              for k in 1:K, s in 1:S)
        - φ_U * sum(x_bar_new[k] * ρ_tilde_1[k, s] for k in 1:K, s in 1:S)
        - φ_U * sum((1.0 - x_bar_new[k]) * ρ_tilde_3[k, s] for k in 1:K, s in 1:S))
end


"""
    tv_isp_follower_optimize!(model, vars, tv::TVData, α_sol::Vector{Float64})

Update α, solve ISP-F, extract inner cut info + outer cut materials.

Returns (status, cut_info) where cut_info contains:
- `:obj_val`: Z^F*
- `:subgradient`: ∂Z^F/∂α (from dual of F10-F13)
- `:intercept`: Z^F* - subgradient' * α_sol
- `:β_val`: β_k^s values (for outer cut)
- `:σ_tilde_val`: σ̃^s values (for outer cut)
- `:ρ_tilde_1_val`, `:ρ_tilde_3_val`: K×S matrices for outer cut π_x
"""
function tv_isp_follower_optimize!(model, vars, tv::TVData, α_sol::Vector{Float64})
    S = tv.S
    K = tv.num_arcs
    q = tv.q_hat
    ε = tv.eps_tilde

    # Update α in RHS
    update_tv_isp_follower_alpha!(model, vars, tv, α_sol)

    optimize!(model)
    st = termination_status(model)
    if st != MOI.OPTIMAL
        error("TV ISP-F not optimal: $st")
    end

    obj_val = objective_value(model)

    # Extract subgradient ∂Z^F/∂α_k from dual values of F10-F13
    # NOTE: shadow_price for ≥ constraints = ∂obj/∂(relaxation) = -∂obj/∂RHS → negate.
    subgradient = zeros(K)
    for k in 1:K
        sg_k = 0.0
        for s in 1:S
            sg_k += q[s] * shadow_price(vars[:F10][s, k])        # F10 ≤ → direct
            sg_k += q[s] * (-shadow_price(vars[:F11][s, k]))     # F11 ≥ → negate
        end
        sg_k += 2 * ε * shadow_price(vars[:F12][k])              # F12 ≤ → direct
        sg_k += 1.0 * shadow_price(vars[:F13][k])                # F13 == → direct
        subgradient[k] = sg_k
    end

    intercept = obj_val - dot(subgradient, α_sol)

    # Extract outer cut materials
    β_val = [value(vars[:β][k, s]) for k in 1:K, s in 1:S]
    σ_tilde_val = [value(vars[:σ_tilde][s]) for s in 1:S]

    # Extract McCormick dual values for outer cut π_x computation
    ρ_tilde_1_val = [value(vars[:ρ_tilde_1][k, s]) for k in 1:K, s in 1:S]
    ρ_tilde_3_val = [value(vars[:ρ_tilde_3][k, s]) for k in 1:K, s in 1:S]

    cut_info = Dict(
        :obj_val => obj_val,
        :subgradient => subgradient,
        :intercept => intercept,
        :β_val => β_val,
        :σ_tilde_val => σ_tilde_val,
        :ρ_tilde_1_val => ρ_tilde_1_val,
        :ρ_tilde_3_val => ρ_tilde_3_val,
    )
    return :OptimalityCut, cut_info
end
