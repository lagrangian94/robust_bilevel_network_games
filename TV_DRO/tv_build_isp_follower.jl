"""
tv_build_isp_follower.jl — Follower Inner Subproblem (ISP-F) for TV-DRO.

All-scenario pooled LP. Constraints F1-F13 from tv_derivation.md §9.4.

Given α_k from IMP, c_k^s and r_k^s from OMP, λ̄ from OMP:
  Z^F = max  λ̄ Σ_s σ̃^s  -  Σ_{s,k} r_k^s β_k^s
  s.t. F1-F13

Sensitivity ∂Z^F/∂α_k comes from dual values of F9-F12 (α appears in RHS).
Also provides outer cut materials: β_k^s, σ̃^s values.
"""

using JuMP
using LinearAlgebra


"""
    build_tv_isp_follower(tv::TVData, c::Matrix{Float64}, r::Matrix{Float64},
                          lambda_bar::Float64; optimizer)

Build follower ISP LP (F1-F13). All scenarios pooled.

# Arguments
- `tv`: TVData struct
- `c`: |A| × S, c_k^s = ξ̄_k^s(1 - v_k x_k)
- `r`: |A| × S, r_k^s = h_k + λ c_k^s
- `lambda_bar`: fixed λ from OMP
- `optimizer`: LP solver (e.g., HiGHS.Optimizer)
"""
function build_tv_isp_follower(tv::TVData, c::Matrix{Float64}, r::Matrix{Float64},
                                lambda_bar::Float64; optimizer)
    S = tv.S
    K = tv.num_arcs
    m = tv.nv1
    Ny = tv.Ny
    Nts = tv.Nts
    q = tv.q_hat
    ε = tv.eps_tilde

    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))

    # --- Variables (F13) ---
    @variable(model, σ_tilde[1:S] >= 0)
    @variable(model, u_tilde[1:K, 1:S] >= 0)
    @variable(model, ω[1:m, 1:S] >= 0)
    @variable(model, β[1:K, 1:S] >= 0)
    @variable(model, d[1:S] >= 0)
    @variable(model, e[1:S] >= 0)
    @variable(model, d_nu[1:S, 1:K] >= 0)
    @variable(model, e_nu[1:S, 1:K] >= 0)

    # --- (F1): N_y ũ^s + N_ts σ̃^s ≤ 0,  ∀s ---
    @constraint(model, F1[j=1:m, s=1:S],
        sum(Ny[j, k] * u_tilde[k, s] for k in 1:K) + Nts[j] * σ_tilde[s] <= 0)

    # --- (F2): -c_k^s d_s + ũ_k^s - d_{s,k}^ν ≤ 0,  ∀k,s ---
    @constraint(model, F2[k=1:K, s=1:S],
        -c[k, s] * d[s] + u_tilde[k, s] - d_nu[s, k] <= 0)

    # --- (F3): [N_yᵀ ω^s]_k + β_k^s ≥ 0,  ∀k,s ---
    @constraint(model, F3[k=1:K, s=1:S],
        sum(Ny[j, k] * ω[j, s] for j in 1:m) + β[k, s] >= 0)

    # --- (F4): N_tsᵀ ω^s ≥ d_s,  ∀s ---
    @constraint(model, F4[s=1:S],
        sum(Nts[j] * ω[j, s] for j in 1:m) >= d[s])

    # --- (F5): d_s - e_s ≤ q̂_s,  ∀s ---
    @constraint(model, F5[s=1:S], d[s] - e[s] <= q[s])

    # --- (F6): d_s + e_s ≥ q̂_s,  ∀s ---
    @constraint(model, F6[s=1:S], d[s] + e[s] >= q[s])

    # --- (F7): Σ_s e_s ≤ 2ε̃ ---
    @constraint(model, F7, sum(e[s] for s in 1:S) <= 2 * ε)

    # --- (F8): Σ_s d_s = 1 ---
    @constraint(model, F8, sum(d[s] for s in 1:S) == 1)

    # --- (F9): d_{s,k}^ν - e_{s,k}^ν ≤ q̂_s α_k,  ∀s,k ---
    @constraint(model, F9[s=1:S, k=1:K], d_nu[s, k] - e_nu[s, k] <= 0)

    # --- (F10): d_{s,k}^ν + e_{s,k}^ν ≥ q̂_s α_k,  ∀s,k ---
    @constraint(model, F10[s=1:S, k=1:K], d_nu[s, k] + e_nu[s, k] >= 0)

    # --- (F11): Σ_s e_{s,k}^ν ≤ 2ε̃ α_k,  ∀k ---
    @constraint(model, F11[k=1:K], sum(e_nu[s, k] for s in 1:S) <= 0)

    # --- (F12): Σ_s d_{s,k}^ν = α_k,  ∀k ---
    @constraint(model, F12[k=1:K], sum(d_nu[s, k] for s in 1:S) == 0)

    # --- Objective: max λ̄ Σ_s σ̃^s - Σ_{s,k} r_k^s β_k^s ---
    @objective(model, Max,
        lambda_bar * sum(σ_tilde[s] for s in 1:S)
        - sum(r[k, s] * β[k, s] for k in 1:K, s in 1:S))

    vars = Dict(
        :σ_tilde => σ_tilde, :u_tilde => u_tilde,
        :ω => ω, :β => β,
        :d => d, :e => e, :d_nu => d_nu, :e_nu => e_nu,
        :F2 => F2,
        :F9 => F9, :F10 => F10, :F11 => F11, :F12 => F12,
    )
    return model, vars
end


"""
    update_tv_isp_follower_alpha!(model, vars, tv::TVData, α_sol::Vector{Float64})

Update RHS of F9-F12 for new α from IMP.
"""
function update_tv_isp_follower_alpha!(model, vars, tv::TVData, α_sol::Vector{Float64})
    S = tv.S
    K = tv.num_arcs
    q = tv.q_hat
    ε = tv.eps_tilde

    for s in 1:S, k in 1:K
        set_normalized_rhs(vars[:F9][s, k], q[s] * α_sol[k])
        set_normalized_rhs(vars[:F10][s, k], q[s] * α_sol[k])
    end
    for k in 1:K
        set_normalized_rhs(vars[:F11][k], 2 * ε * α_sol[k])
        set_normalized_rhs(vars[:F12][k], α_sol[k])
    end
end


"""
    update_tv_isp_follower_outer!(model, vars, tv::TVData,
                                   c_new::Matrix{Float64}, r_new::Matrix{Float64},
                                   lambda_bar_new::Float64)

Update F2 coefficients and objective when outer solution changes.
Rebuild F2 + reset objective.
"""
function update_tv_isp_follower_outer!(model, vars, tv::TVData,
                                       c_new::Matrix{Float64}, r_new::Matrix{Float64},
                                       lambda_bar_new::Float64)
    S = tv.S
    K = tv.num_arcs

    # Delete old F2 constraints
    for k in 1:K, s in 1:S
        delete(model, vars[:F2][k, s])
    end

    # Add new F2 with updated c
    d_var = vars[:d]
    u_tilde = vars[:u_tilde]
    d_nu = vars[:d_nu]
    F2_new = @constraint(model, [k=1:K, s=1:S],
        -c_new[k, s] * d_var[s] + u_tilde[k, s] - d_nu[s, k] <= 0)
    vars[:F2] = F2_new

    # Update objective with new r and λ̄
    σ_tilde = vars[:σ_tilde]
    β = vars[:β]
    @objective(model, Max,
        lambda_bar_new * sum(σ_tilde[s] for s in 1:S)
        - sum(r_new[k, s] * β[k, s] for k in 1:K, s in 1:S))
end


"""
    tv_isp_follower_optimize!(model, vars, tv::TVData, α_sol::Vector{Float64})

Update α, solve ISP-F, extract inner cut info + outer cut materials.

Returns (status, cut_info) where cut_info contains:
- `:obj_val`: Z^F*
- `:subgradient`: ∂Z^F/∂α (from dual of F9-F12)
- `:intercept`: Z^F* - subgradient' * α_sol
- `:β_val`: β_k^s values (for outer cut)
- `:σ_tilde_val`: σ̃^s values (for outer cut)
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

    # Extract subgradient ∂Z^F/∂α_k from dual values of F9-F12
    # NOTE: shadow_price for ≥ constraints = ∂obj/∂(relaxation) = -∂obj/∂RHS → negate.
    subgradient = zeros(K)
    for k in 1:K
        sg_k = 0.0
        for s in 1:S
            sg_k += q[s] * shadow_price(vars[:F9][s, k])        # F9 ≤ → direct
            sg_k += q[s] * (-shadow_price(vars[:F10][s, k]))    # F10 ≥ → negate
        end
        sg_k += 2 * ε * shadow_price(vars[:F11][k])              # F11 ≤ → direct
        sg_k += 1.0 * shadow_price(vars[:F12][k])                # F12 == → direct
        subgradient[k] = sg_k
    end

    intercept = obj_val - dot(subgradient, α_sol)

    # Extract outer cut materials
    β_val = [value(vars[:β][k, s]) for k in 1:K, s in 1:S]
    σ_tilde_val = [value(vars[:σ_tilde][s]) for s in 1:S]

    # Extract outer cut materials: d_s values and φ̃_k^s = shadow_price(F2)
    d_val = [value(vars[:d][s]) for s in 1:S]
    # φ̃_k^s: OSP primal capacity dual = shadow_price of F2 (≤ in Max → sp ≥ 0)
    φ_tilde_val = [shadow_price(vars[:F2][k, s]) for k in 1:K, s in 1:S]

    cut_info = Dict(
        :obj_val => obj_val,
        :subgradient => subgradient,
        :intercept => intercept,
        :β_val => β_val,
        :σ_tilde_val => σ_tilde_val,
        :d_val => d_val,
        :φ_tilde_val => φ_tilde_val,
    )
    return :OptimalityCut, cut_info
end
