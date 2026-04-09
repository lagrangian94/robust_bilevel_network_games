"""
tv_build_omp.jl — Outer Master Problem (OMP) for TV-DRO.

  min  t₀
  s.t. 1ᵀh ≤ λw,  x ∈ X,  McCormick(ψ⁰ = λx)
       t₀ ≥ (optimality cuts)

Outer cut: t₀ ≥ Z₀* + π_h'(h-h̄) + π_λ(λ-λ̄) + π_{ψ⁰}'(ψ⁰-ψ̄⁰) + π_x'(x-x̄)

From OSP dual objective (§10.1 in tv_derivation_revised.md):
  π_{h_k}   = -Σ_s β_k^s
  π_λ       = Σ_s σ̃^s - Σ_{s,k} ξ̄_k^s β_k^s
  π_{ψ⁰_k}  = v_k Σ_s ξ̄_k^s β_k^s
  π_{x_k}   = -φ^U Σ_s (ρ̂¹_{k,s} + ρ̃¹_{k,s}) + φ^U Σ_s (ρ̂³_{k,s} + ρ̃³_{k,s})

Globally valid: dual feasible region is independent of OMP variables.
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
                                 x_sol, h_sol, λ_sol, ψ0_sol)

Compute outer cut coefficients from converged inner loop.
Uses McCormick-based π_x (globally valid).

Returns Dict with :intercept, :π_h, :π_λ, :π_ψ0, :π_x, :Z0_val.
"""
function compute_tv_outer_cut_coeffs(tv::TVData, leader_cut_info, follower_cut_info,
                                      x_sol, h_sol, λ_sol, ψ0_sol)
    S = tv.S
    K = tv.num_arcs
    ξ = tv.xi_bar
    v = tv.v
    φ_U = tv.phi_U

    β = follower_cut_info[:β_val]       # K × S
    σ̃ = follower_cut_info[:σ_tilde_val]  # S

    # McCormick dual values for x-gradient
    ρ̂1 = leader_cut_info[:ρ_hat_1_val]       # K × S
    ρ̂3 = leader_cut_info[:ρ_hat_3_val]       # K × S
    ρ̃1 = follower_cut_info[:ρ_tilde_1_val]   # K × S
    ρ̃3 = follower_cut_info[:ρ_tilde_3_val]   # K × S

    Z_L = leader_cut_info[:obj_val]
    Z_F = follower_cut_info[:obj_val]
    Z0 = Z_L + Z_F

    # π_{h_k} = -Σ_s β_k^s
    π_h = [-sum(β[k, s] for s in 1:S) for k in 1:K]

    # π_λ = Σ_s σ̃^s - Σ_{s,k} ξ̄_k^s β_k^s
    π_λ = sum(σ̃) - sum(ξ[k, s] * β[k, s] for k in 1:K, s in 1:S)

    # π_{ψ⁰_k} = v_k · Σ_s ξ̄_k^s · β_k^s
    π_ψ0 = [v[k] * sum(ξ[k, s] * β[k, s] for s in 1:S) for k in 1:K]

    # π_{x_k} = -φ^U Σ_s (ρ̂¹_{k,s} + ρ̃¹_{k,s}) + φ^U Σ_s (ρ̂³_{k,s} + ρ̃³_{k,s})
    # From dual objective coefficient of x̄_k (§10.1)
    π_x = [(-φ_U * sum(ρ̂1[k, s] + ρ̃1[k, s] for s in 1:S)
             + φ_U * sum(ρ̂3[k, s] + ρ̃3[k, s] for s in 1:S)) for k in 1:K]

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
