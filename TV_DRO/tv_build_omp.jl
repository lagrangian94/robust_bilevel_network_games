"""
tv_build_omp.jl — Outer Master Problem (OMP) for TV-DRO.

  min  t₀
  s.t. 1ᵀh ≤ λw,  x ∈ X,  McCormick(ψ⁰ = λx)
       t₀ ≥ (optimality cuts)

Outer cut: t₀ ≥ Z₀* + π_h'(h-h̄) + π_λ(λ-λ̄) + π_{ψ⁰}'(ψ⁰-ψ̄⁰) + π_x'(x-x̄)

From OSP sensitivity:
  π_{h_k}   = -Σ_s β_k^s                                    (P16 RHS)
  π_λ       = Σ_s σ̃^s - Σ_{s,k} ξ̄_k^s β_k^s               (P9 + P16 RHS)
  π_{ψ⁰_k}  = v_k Σ_s ξ̄_k^s β_k^s                          (P16 RHS)
  π_{x_k}   = -v_k Σ_s ξ̄_k^s (a_s φ̂_k^s + d_s φ̃_k^s)     (P2 + P4 coefficient)
"""

using JuMP
using LinearAlgebra


"""
    build_tv_omp(tv::TVData; optimizer)

Build OMP MILP.

# Returns
- `(model, vars)` with vars[:t_0], vars[:x], vars[:h], vars[:λ], vars[:ψ0]
"""
function build_tv_omp(tv::TVData; optimizer)
    K = tv.num_arcs
    w = tv.w
    γ = tv.gamma
    λU = tv.lambda_U

    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))

    # --- Variables ---
    @variable(model, t_0 >= 0)
    @variable(model, x[1:K], Bin)
    @variable(model, h[1:K] >= 0)
    @variable(model, λ >= 0)
    @variable(model, ψ0[1:K] >= 0)

    # --- Constraints ---
    # Recovery budget (T17): 1ᵀh ≤ λw
    @constraint(model, sum(h[k] for k in 1:K) <= λ * w)

    # Interdiction budget
    @constraint(model, sum(x[k] for k in 1:K) <= γ)

    # Fix non-interdictable arcs to 0
    for k in 1:K
        if !tv.interdictable_arcs[k]
            @constraint(model, x[k] == 0)
        end
    end

    # λ upper bound
    @constraint(model, λ <= λU)

    # McCormick: ψ⁰_k = λ · x_k
    @constraint(model, [k=1:K], ψ0[k] <= λU * x[k])     # MC1
    @constraint(model, [k=1:K], ψ0[k] <= λ)              # MC2
    @constraint(model, [k=1:K], ψ0[k] >= λ - λU * (1 - x[k]))  # MC3

    # Objective
    @objective(model, Min, t_0)

    vars = Dict(:t_0 => t_0, :x => x, :h => h, :λ => λ, :ψ0 => ψ0)
    return model, vars
end


"""
    compute_tv_outer_cut_coeffs(tv::TVData, leader_cut_info, follower_cut_info,
                                 c::Matrix{Float64}, x_sol, h_sol, λ_sol, ψ0_sol)

Compute outer cut coefficients from converged inner loop.

Returns Dict with :intercept, :π_h, :π_λ, :π_ψ0, :π_x, :Z0_val.
"""
function compute_tv_outer_cut_coeffs(tv::TVData, leader_cut_info, follower_cut_info,
                                      c::Matrix{Float64}, x_sol, h_sol, λ_sol, ψ0_sol)
    S = tv.S
    K = tv.num_arcs
    ξ = tv.xi_bar
    v = tv.v

    β = follower_cut_info[:β_val]       # K × S
    σ̃ = follower_cut_info[:σ_tilde_val]  # S

    # For x-gradient: ISP primal values and φ duals
    a_val = leader_cut_info[:a_val]           # S (ISP-L primal a_s)
    φ_hat = leader_cut_info[:φ_hat_val]       # K × S (shadow_price of L2)
    d_val = follower_cut_info[:d_val]         # S (ISP-F primal d_s)
    φ_tilde = follower_cut_info[:φ_tilde_val] # K × S (shadow_price of F2)

    Z_L = leader_cut_info[:obj_val]
    Z_F = follower_cut_info[:obj_val]
    Z0 = Z_L + Z_F

    # π_{h_k} = -Σ_s β_k^s
    π_h = [-sum(β[k, s] for s in 1:S) for k in 1:K]

    # π_λ = Σ_s σ̃^s - Σ_{s,k} ξ̄_k^s β_k^s
    π_λ = sum(σ̃) - sum(ξ[k, s] * β[k, s] for k in 1:K, s in 1:S)

    # π_{ψ⁰_k} = v_k · Σ_s ξ̄_k^s · β_k^s
    π_ψ0 = [v[k] * sum(ξ[k, s] * β[k, s] for s in 1:S) for k in 1:K]

    # π_{x_k} = -v_k Σ_s ξ̄_k^s (a_s φ̂_k^s + d_s φ̃_k^s)
    # From P2/P4 coefficient sensitivity: ∂Z₀/∂x_k from c_k^s = ξ(1-vx) change
    π_x = [-v[k] * sum(ξ[k, s] * (a_val[s] * φ_hat[k, s] + d_val[s] * φ_tilde[k, s])
                        for s in 1:S) for k in 1:K]

    # intercept = Z₀ - π_h'h̄ - π_λ λ̄ - π_{ψ⁰}'ψ̄⁰ - π_x'x̄
    intercept = Z0 - dot(π_h, h_sol) - π_λ * λ_sol - dot(π_ψ0, ψ0_sol) - dot(π_x, x_sol)

    return Dict(
        :intercept => intercept,
        :π_h => π_h,
        :π_λ => π_λ,
        :π_ψ0 => π_ψ0,
        :π_x => π_x,
        :Z0_val => Z0,
    )
end


"""
    add_tv_optimality_cut!(omp_model, omp_vars, outer_cut, iter)

Add optimality cut to OMP:
  t₀ ≥ intercept + π_h'h + π_λ λ + π_{ψ⁰}'ψ⁰ + π_x'x
"""
function add_tv_optimality_cut!(omp_model, omp_vars, outer_cut, iter)
    K = length(omp_vars[:h])
    h = omp_vars[:h]
    λ = omp_vars[:λ]
    ψ0 = omp_vars[:ψ0]
    x = omp_vars[:x]
    t_0 = omp_vars[:t_0]

    intercept = outer_cut[:intercept]
    π_h = outer_cut[:π_h]
    π_λ = outer_cut[:π_λ]
    π_ψ0 = outer_cut[:π_ψ0]
    π_x = outer_cut[:π_x]

    cut_expr = intercept +
        sum(π_h[k] * h[k] for k in 1:K) +
        π_λ * λ +
        sum(π_ψ0[k] * ψ0[k] for k in 1:K) +
        sum(π_x[k] * x[k] for k in 1:K)

    c = @constraint(omp_model, t_0 >= cut_expr)
    set_name(c, "tv_opt_cut_$iter")
    return c
end
