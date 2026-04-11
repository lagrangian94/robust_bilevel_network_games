"""
true_dro_build_subproblem.jl — Bilinear subproblem for True-DRO-Exact.

Single merged subproblem combining ISP-L (DL-1~DL-8) and ISP-F (DF-1~DF-ψ)
from true_dro_v5.md §7.2-§7.3. Coupled by α (Lagrangian multiplier of (CC)).

Bilinearity (only 2 places):
  (DL-2): -(ξ̄_k^s + α_k) a_s + ...   →  ζ^L_{ks} := α_k a_s
  (DF-6): -(ξ̄_k^s + α_k) d_s + ...   →  ζ^F_{ks} := α_k d_s

Bilinear terms reified into auxiliary variables (Gurobi multilinear guideline:
https://support.gurobi.com/hc/en-us/articles/360049744691).
Equality `ζ == α*x` is a quadratic constraint, solved with NonConvex=2.
This isolates bilinearity for cleaner spatial B&B / McCormick relaxation later.

x̄ appears ONLY in objective (coefficients of ρ̂¹/ρ̂³/ρ̃¹/ρ̃³/ρ⁰¹/ρ⁰³)
→ build once, update objective per outer Benders iter.

  max  obj^L(x̄) + obj^F(x̄)
  s.t. (DL-1)~(DL-8), (DF-1)~(DF-ψ), 1ᵀα ≤ w, α ≥ 0
"""

using JuMP
using LinearAlgebra


"""
    build_true_dro_subproblem(td::TrueDROData, x_bar::Vector{Float64}; optimizer)

Build merged bilinear subproblem.

Returns (model, vars). Caller must use `Gurobi.Optimizer` (or any QCP solver
supporting non-convex bilinear) and set `NonConvex=2` separately.
"""
function build_true_dro_subproblem(td::TrueDROData, x_bar::Vector{Float64}; optimizer)
    S = td.S
    K = td.num_arcs
    m = td.nv1
    Ny = td.Ny
    Nts = td.Nts
    q = td.q_hat
    ε̂ = td.eps_hat
    ε̃ = td.eps_tilde
    ξ = td.xi_bar
    v = td.v
    w = td.w
    φ̂U = td.phi_hat_U
    φ̃U = td.phi_tilde_U
    λU = td.lambda_U

    model = Model(optimizer)
    set_silent(model)

    # ====================================================================
    # Tight per-scenario TV ball bounds (true_dro_v5.md §10.1)
    # ====================================================================
    a_min = [max(0.0, q[s] - 2 * ε̂) for s in 1:S]
    a_max = [min(1.0, q[s] + 2 * ε̂) for s in 1:S]
    d_min = [max(0.0, q[s] - 2 * ε̃) for s in 1:S]
    d_max = [min(1.0, q[s] + 2 * ε̃) for s in 1:S]

    # ====================================================================
    # α : Lagrangian multiplier (shared between L and F)
    # ====================================================================
    @variable(model, 0 <= α[1:K] <= w)        # individual bound (B&B 성능)
    @constraint(model, sum(α[k] for k in 1:K) <= w)

    # ====================================================================
    # ISP-L variables (true_dro_v5.md §7.3)
    # ====================================================================
    @variable(model, σ_hat[1:S] >= 0)         # σ̂ˢ
    @variable(model, u_hat[1:K, 1:S] >= 0)    # ûᵏˢ
    # a_s ∈ [max(0, q̂-2ε̂), min(1, q̂+2ε̂)]   (§10.1)
    @variable(model, a_min[s] <= a[s=1:S] <= a_max[s])
    # b_s = |a_s - q̂_s|, ≤ 2ε̂ from DL-6
    @variable(model, 0 <= b[1:S] <= 2 * ε̂)
    @variable(model, ρ_hat_1[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_2[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_3[1:K, 1:S] >= 0)

    # Auxiliary bilinear: ζL_{ks} = α_k · a_s, tight per-(k,s) bound (§10.1-§10.2)
    @variable(model, 0 <= ζL[k=1:K, s=1:S] <= w * a_max[s])
    @constraint(model, ζL_def[k=1:K, s=1:S], ζL[k, s] == α[k] * a[s])

    # --- (DL-1): N_y û^s + N_ts σ̂^s = 0  ∀s ---
    @constraint(model, DL1[j=1:m, s=1:S],
        sum(Ny[j, k] * u_hat[k, s] for k in 1:K) + Nts[j] * σ_hat[s] == 0)

    # --- (DL-2): -(ξ̄_k^s + α_k) a_s + û_k^s + ρ̂² - ρ̂³ ≤ 0  ∀k,s   [BILINEAR via ζL] ---
    @constraint(model, DL2[k=1:K, s=1:S],
        -ξ[k, s] * a[s] - ζL[k, s]
        + u_hat[k, s] + ρ_hat_2[k, s] - ρ_hat_3[k, s] <= 0)

    # --- (DL-3): v_k ξ̄_k^s a_s - ρ̂¹ - ρ̂² + ρ̂³ ≤ 0  ∀k,s ---
    @constraint(model, DL3[k=1:K, s=1:S],
        v[k] * ξ[k, s] * a[s]
        - ρ_hat_1[k, s] - ρ_hat_2[k, s] + ρ_hat_3[k, s] <= 0)

    # --- (DL-4): a_s - b_s ≤ q̂_s ---
    @constraint(model, DL4[s=1:S], a[s] - b[s] <= q[s])

    # --- (DL-5): a_s + b_s ≥ q̂_s ---
    @constraint(model, DL5[s=1:S], a[s] + b[s] >= q[s])

    # --- (DL-6): Σ_s b_s ≤ 2ε̂ ---
    @constraint(model, DL6, sum(b[s] for s in 1:S) <= 2 * ε̂)

    # --- (DL-7): Σ_s a_s = 1 ---
    @constraint(model, DL7, sum(a[s] for s in 1:S) == 1)

    # ====================================================================
    # ISP-F variables (true_dro_v5.md §7.2)
    # ====================================================================
    # d_s ∈ [max(0, q̂-2ε̃), min(1, q̂+2ε̃)]   (§10.1)
    @variable(model, d_min[s] <= d[s=1:S] <= d_max[s])
    # e_s = |d_s - q̂_s|, ≤ 2ε̃ from DF-3
    @variable(model, 0 <= e[1:S] <= 2 * ε̃)
    @variable(model, u_tilde[1:K, 1:S] >= 0)  # ũ
    @variable(model, σ_tilde[1:S] >= 0)        # σ̃ˢ
    @variable(model, ω[1:m, 1:S])              # FREE
    @variable(model, β[1:K, 1:S] >= 0)
    @variable(model, δ >= 0)
    @variable(model, ρ_tilde_1[1:K, 1:S] >= 0)
    @variable(model, ρ_tilde_2[1:K, 1:S] >= 0)
    @variable(model, ρ_tilde_3[1:K, 1:S] >= 0)
    @variable(model, ρ_psi0_1[1:K] >= 0)       # ρ⁰¹
    @variable(model, ρ_psi0_2[1:K] >= 0)       # ρ⁰²
    @variable(model, ρ_psi0_3[1:K] >= 0)       # ρ⁰³

    # Auxiliary bilinear: ζF_{ks} = α_k · d_s, tight per-(k,s) bound (§10.1-§10.2)
    @variable(model, 0 <= ζF[k=1:K, s=1:S] <= w * d_max[s])
    @constraint(model, ζF_def[k=1:K, s=1:S], ζF[k, s] == α[k] * d[s])

    # --- (DF-1): d_s - e_s ≤ q̂_s ---
    @constraint(model, DF1[s=1:S], d[s] - e[s] <= q[s])
    # --- (DF-2): d_s + e_s ≥ q̂_s ---
    @constraint(model, DF2[s=1:S], d[s] + e[s] >= q[s])
    # --- (DF-3): Σ_s e_s ≤ 2ε̃ ---
    @constraint(model, DF3, sum(e[s] for s in 1:S) <= 2 * ε̃)
    # --- (DF-4): Σ_s d_s = 1 ---
    @constraint(model, DF4, sum(d[s] for s in 1:S) == 1)

    # --- (DF-5): N_y ũ^s + N_ts σ̃^s = 0  ∀s ---
    @constraint(model, DF5[j=1:m, s=1:S],
        sum(Ny[j, k] * u_tilde[k, s] for k in 1:K) + Nts[j] * σ_tilde[s] == 0)

    # --- (DF-6): -(ξ̄_k^s + α_k) d_s + ũ_k^s + ρ̃² - ρ̃³ ≤ 0  ∀k,s   [BILINEAR via ζF] ---
    @constraint(model, DF6[k=1:K, s=1:S],
        -ξ[k, s] * d[s] - ζF[k, s]
        + u_tilde[k, s] + ρ_tilde_2[k, s] - ρ_tilde_3[k, s] <= 0)

    # --- (DF-7): v_k ξ̄_k^s d_s - ρ̃¹ - ρ̃² + ρ̃³ ≤ 0  ∀k,s ---
    @constraint(model, DF7[k=1:K, s=1:S],
        v[k] * ξ[k, s] * d[s]
        - ρ_tilde_1[k, s] - ρ_tilde_2[k, s] + ρ_tilde_3[k, s] <= 0)

    # --- (DF-8): [N_yᵀ ω^s]_k - β_k^s ≤ 0  ∀k,s ---
    @constraint(model, DF8[k=1:K, s=1:S],
        sum(Ny[j, k] * ω[j, s] for j in 1:m) - β[k, s] <= 0)

    # --- (DF-9): d_s + N_tsᵀ ω^s ≤ 0  ∀s ---
    @constraint(model, DF9[s=1:S],
        d[s] + sum(Nts[j] * ω[j, s] for j in 1:m) <= 0)

    # --- (DF-h): Σ_s β_k^s ≤ δ  ∀k ---
    @constraint(model, DFh[k=1:K], sum(β[k, s] for s in 1:S) <= δ)

    # --- (DF-λ): Σ_s σ̃^s ≥ Σ_{s,k} ξ̄ β + w δ + Σ_k ρ⁰² - Σ_k ρ⁰³ ---
    @constraint(model, DFlam,
        sum(σ_tilde[s] for s in 1:S)
        >= sum(ξ[k, s] * β[k, s] for k in 1:K, s in 1:S)
           + w * δ
           + sum(ρ_psi0_2[k] for k in 1:K)
           - sum(ρ_psi0_3[k] for k in 1:K))

    # --- (DF-ψ): v_k Σ_s ξ̄ β + ρ⁰¹ + ρ⁰² ≥ ρ⁰³  ∀k ---
    @constraint(model, DFpsi[k=1:K],
        v[k] * sum(ξ[k, s] * β[k, s] for s in 1:S) + ρ_psi0_1[k] + ρ_psi0_2[k]
        >= ρ_psi0_3[k])

    # ====================================================================
    # Objective: max obj^L(x̄) + obj^F(x̄)   (linear in vars, x̄ in coefs)
    # ====================================================================
    obj_L = sum(σ_hat[s] for s in 1:S) -
            φ̂U * sum(x_bar[k] * ρ_hat_1[k, s] for k in 1:K, s in 1:S) -
            φ̂U * sum((1.0 - x_bar[k]) * ρ_hat_3[k, s] for k in 1:K, s in 1:S)

    obj_F = -φ̃U * sum(x_bar[k] * ρ_tilde_1[k, s] for k in 1:K, s in 1:S) -
             φ̃U * sum((1.0 - x_bar[k]) * ρ_tilde_3[k, s] for k in 1:K, s in 1:S) -
             λU * sum(x_bar[k] * ρ_psi0_1[k] for k in 1:K) -
             λU * sum((1.0 - x_bar[k]) * ρ_psi0_3[k] for k in 1:K)

    @objective(model, Max, obj_L + obj_F)

    vars = Dict(
        :α => α,
        # Auxiliary bilinear
        :ζL => ζL, :ζF => ζF,
        # ISP-L
        :σ_hat => σ_hat, :u_hat => u_hat, :a => a, :b => b,
        :ρ_hat_1 => ρ_hat_1, :ρ_hat_2 => ρ_hat_2, :ρ_hat_3 => ρ_hat_3,
        # ISP-F
        :d => d, :e => e, :u_tilde => u_tilde, :σ_tilde => σ_tilde,
        :ω => ω, :β => β, :δ => δ,
        :ρ_tilde_1 => ρ_tilde_1, :ρ_tilde_2 => ρ_tilde_2, :ρ_tilde_3 => ρ_tilde_3,
        :ρ_psi0_1 => ρ_psi0_1, :ρ_psi0_2 => ρ_psi0_2, :ρ_psi0_3 => ρ_psi0_3,
    )
    return model, vars
end


"""
    update_true_dro_subproblem_objective!(model, vars, td::TrueDROData,
                                          x_bar_new::Vector{Float64})

Re-set objective for new x̄ from OMP. Constraints are x-independent.
"""
function update_true_dro_subproblem_objective!(model, vars, td::TrueDROData,
                                                x_bar_new::Vector{Float64})
    S = td.S
    K = td.num_arcs
    φ̂U = td.phi_hat_U
    φ̃U = td.phi_tilde_U
    λU = td.lambda_U

    σ_hat = vars[:σ_hat]
    ρ_hat_1 = vars[:ρ_hat_1]
    ρ_hat_3 = vars[:ρ_hat_3]
    ρ_tilde_1 = vars[:ρ_tilde_1]
    ρ_tilde_3 = vars[:ρ_tilde_3]
    ρ_psi0_1 = vars[:ρ_psi0_1]
    ρ_psi0_3 = vars[:ρ_psi0_3]

    obj_L = sum(σ_hat[s] for s in 1:S) -
            φ̂U * sum(x_bar_new[k] * ρ_hat_1[k, s] for k in 1:K, s in 1:S) -
            φ̂U * sum((1.0 - x_bar_new[k]) * ρ_hat_3[k, s] for k in 1:K, s in 1:S)

    obj_F = -φ̃U * sum(x_bar_new[k] * ρ_tilde_1[k, s] for k in 1:K, s in 1:S) -
             φ̃U * sum((1.0 - x_bar_new[k]) * ρ_tilde_3[k, s] for k in 1:K, s in 1:S) -
             λU * sum(x_bar_new[k] * ρ_psi0_1[k] for k in 1:K) -
             λU * sum((1.0 - x_bar_new[k]) * ρ_psi0_3[k] for k in 1:K)

    @objective(model, Max, obj_L + obj_F)
end


"""
    solve_true_dro_subproblem!(model, vars, td::TrueDROData, x_bar::Vector{Float64})

Update obj for x̄, solve, return Dict with:
- :Z0_val
- :α_val
- :rho_hat_1_val, :rho_hat_3_val      (K × S)
- :rho_tilde_1_val, :rho_tilde_3_val  (K × S)
- :rho_psi0_1_val, :rho_psi0_3_val    (K)
"""
function solve_true_dro_subproblem!(model, vars, td::TrueDROData, x_bar::Vector{Float64})
    S = td.S
    K = td.num_arcs

    update_true_dro_subproblem_objective!(model, vars, td, x_bar)

    optimize!(model)
    st = termination_status(model)
    if st != MOI.OPTIMAL
        error("True-DRO subproblem not optimal: $st")
    end

    Z0_val = objective_value(model)
    α_val = [value(vars[:α][k]) for k in 1:K]
    ρ̂1 = [value(vars[:ρ_hat_1][k, s]) for k in 1:K, s in 1:S]
    ρ̂3 = [value(vars[:ρ_hat_3][k, s]) for k in 1:K, s in 1:S]
    ρ̃1 = [value(vars[:ρ_tilde_1][k, s]) for k in 1:K, s in 1:S]
    ρ̃3 = [value(vars[:ρ_tilde_3][k, s]) for k in 1:K, s in 1:S]
    ρ01 = [value(vars[:ρ_psi0_1][k]) for k in 1:K]
    ρ03 = [value(vars[:ρ_psi0_3][k]) for k in 1:K]

    return Dict(
        :Z0_val => Z0_val,
        :α_val => α_val,
        :rho_hat_1_val => ρ̂1, :rho_hat_3_val => ρ̂3,
        :rho_tilde_1_val => ρ̃1, :rho_tilde_3_val => ρ̃3,
        :rho_psi0_1_val => ρ01, :rho_psi0_3_val => ρ03,
    )
end
