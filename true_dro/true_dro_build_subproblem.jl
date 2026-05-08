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
# Arcwise VI 고정 규칙 (obj_F의 각 term = 0을 직접 강제):
#   x_k = 1 → ρ̃₁[k,:] = 0, ρ⁰₁[k] = 0
#   x_k = 0 → ρ̃₃[k,:] = 0, ρ⁰₃[k] = 0
# set_upper_bound(v, 0) / 복원으로 per-iter 전환.
# ⚠️ global opt에서만 타당 — local solve(OptimalityTarget=1)에서 사용 금지.
function _apply_arcwise_vi_fixings!(vars, td, x_bar)
    K = td.num_arcs
    S = td.S
    ρ̃1 = vars[:ρ_tilde_1]
    ρ̃3 = vars[:ρ_tilde_3]
    ρ⁰1 = vars[:ρ_psi0_1]
    ρ⁰3 = vars[:ρ_psi0_3]
    orig_ub = vars[:rho_orig_ub]  # Float64 or nothing

    function _restore!(v)
        if orig_ub === nothing
            if has_upper_bound(v)
                delete_upper_bound(v)
            end
        else
            set_upper_bound(v, orig_ub)
        end
    end

    for k in 1:K
        if x_bar[k] > 0.5  # x_k = 1
            for s in 1:S
                set_upper_bound(ρ̃1[k, s], 0.0)
                _restore!(ρ̃3[k, s])
            end
            set_upper_bound(ρ⁰1[k], 0.0)
            _restore!(ρ⁰3[k])
        else  # x_k = 0
            for s in 1:S
                _restore!(ρ̃1[k, s])
                set_upper_bound(ρ̃3[k, s], 0.0)
            end
            _restore!(ρ⁰1[k])
            set_upper_bound(ρ⁰3[k], 0.0)
        end
    end
end


function build_true_dro_subproblem(td::TrueDROData, x_bar::Vector{Float64};
                                   optimizer, silent=true, rho_upper_bound::Union{Float64,Nothing}=nothing,
                                   add_objF_vi::Bool=false,
                                   add_objF_vi_arcwise::Bool=false)
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
    if silent
        set_silent(model)
    end
    set_optimizer_attribute(model, "DualReductions", 0)

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
    # CVaR risk parameter
    # ====================================================================
    β = td.beta

    # ====================================================================
    # ISP-L variables (true_dro_v5.md §7.3)
    # ====================================================================
    @variable(model, σ_hat[1:S] >= 0)         # σ̂ˢ
    @variable(model, u_hat[1:K, 1:S] >= 0)    # ûᵏˢ
    # a_s ∈ [max(0, q̂-2ε̂), min(1, q̂+2ε̂)]   (§10.1)
    @variable(model, a_min[s] <= a[s=1:S] <= a_max[s])
    # b_s = |a_s - q̂_s|, ≤ 2ε̂ from DL-6
    @variable(model, 0 <= b[1:S] <= 2 * ε̂)
    # r[s]: CVaR reweighting variable, r[s] ∈ [0, a_max[s]/(1-β)]
    r_max = [a_max[s] / (1.0 - β) for s in 1:S]
    @variable(model, 0 <= r[s=1:S] <= r_max[s])
    @variable(model, ρ_hat_1[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_2[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_3[1:K, 1:S] >= 0)

    # ρ upper bound: local opt에서 ρ 폭발 방지 (local_opt_cut_explosion.md 참조)
    if rho_upper_bound !== nothing
        for k in 1:K, s in 1:S
            set_upper_bound(ρ_hat_1[k, s], rho_upper_bound)
            set_upper_bound(ρ_hat_2[k, s], rho_upper_bound)
            set_upper_bound(ρ_hat_3[k, s], rho_upper_bound)
        end
    end

    # Auxiliary bilinear: ζL_{ks} = α_k · r_s, tight per-(k,s) bound
    @variable(model, 0 <= ζL[k=1:K, s=1:S] <= w * r_max[s])
    @constraint(model, ζL_def[k=1:K, s=1:S], ζL[k, s] == α[k] * r[s])

    # --- (DL-1): N_y û^s + N_ts σ̂^s = 0  ∀s ---
    @constraint(model, DL1[j=1:m, s=1:S],
        sum(Ny[j, k] * u_hat[k, s] for k in 1:K) + Nts[j] * σ_hat[s] == 0)

    # --- (DL-RU1): r[s] ≤ a[s]/(1-β)  ∀s ---
    @constraint(model, DL_RU1[s=1:S], r[s] <= a[s] / (1.0 - β))

    # --- (DL-RU2): Σ_s r[s] = 1 ---
    @constraint(model, DL_RU2, sum(r[s] for s in 1:S) == 1)

    # --- (DL-2): -(ξ̄_k^s + α_k) r_s + û_k^s + ρ̂² - ρ̂³ ≤ 0  ∀k,s   [BILINEAR via ζL, a→r] ---
    @constraint(model, DL2[k=1:K, s=1:S],
        -ξ[k, s] * r[s] - ζL[k, s]
        + u_hat[k, s] + ρ_hat_2[k, s] - ρ_hat_3[k, s] <= 0)

    # --- (DL-3): v_k^s ξ̄_k^s r_s - ρ̂¹ - ρ̂² + ρ̂³ ≤ 0  ∀k,s   [a→r] ---
    @constraint(model, DL3[k=1:K, s=1:S],
        v[k, s] * ξ[k, s] * r[s]
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

    if rho_upper_bound !== nothing
        for k in 1:K, s in 1:S
            set_upper_bound(ρ_tilde_1[k, s], rho_upper_bound)
            set_upper_bound(ρ_tilde_2[k, s], rho_upper_bound)
            set_upper_bound(ρ_tilde_3[k, s], rho_upper_bound)
        end
        for k in 1:K
            set_upper_bound(ρ_psi0_1[k], rho_upper_bound)
            set_upper_bound(ρ_psi0_2[k], rho_upper_bound)
            set_upper_bound(ρ_psi0_3[k], rho_upper_bound)
        end
    end

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

    # --- (DF-7): v_k^s ξ̄_k^s d_s - ρ̃¹ - ρ̃² + ρ̃³ ≤ 0  ∀k,s ---
    @constraint(model, DF7[k=1:K, s=1:S],
        v[k, s] * ξ[k, s] * d[s]
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

    # --- (DF-ψ): Σ_s v_k^s ξ̄ β + ρ⁰¹ + ρ⁰² ≥ ρ⁰³  ∀k ---
    @constraint(model, DFpsi[k=1:K],
        sum(v[k, s] * ξ[k, s] * β[k, s] for s in 1:S) + ρ_psi0_1[k] + ρ_psi0_2[k]
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

    # ====================================================================
    # Valid Inequality: obj_F의 **각 term ≥ 0**  (per-term, aggregate ≥ 0 아님).
    #
    # obj_F = Σ(≤0 term). 각 term 자체가 구조적으로 ≤ 0:
    #   -φ̃U·x_k·ρ̃₁[k,s] ≤ 0,  -φ̃U·(1-x_k)·ρ̃₃[k,s] ≤ 0,
    #   -λU·x_k·ρ⁰₁[k] ≤ 0,    -λU·(1-x_k)·ρ⁰₃[k] ≤ 0.
    # global optimum에서는 obj_F(x*) = 0 → 각 term = 0 (≤0 합이 0).
    # 따라서 각 term ≥ 0 을 개별 constraint로 추가 가능.
    #
    # ⚠️ 이 VI는 **global optimum에서만** 타당. Local opt(OptimalityTarget=1)
    #    에서는 obj_F < 0 가능 → cut-off 발생. **local solve에서는 사용 금지**.
    # Total: 2·K·S + 2·K 제약식.
    # ====================================================================
    if add_objF_vi
        @constraint(model, objF_vi_rho_tilde_1[k=1:K, s=1:S],
            -φ̃U * x_bar[k] * ρ_tilde_1[k, s] >= 0)
        @constraint(model, objF_vi_rho_tilde_3[k=1:K, s=1:S],
            -φ̃U * (1.0 - x_bar[k]) * ρ_tilde_3[k, s] >= 0)
        @constraint(model, objF_vi_rho_psi0_1[k=1:K],
            -λU * x_bar[k] * ρ_psi0_1[k] >= 0)
        @constraint(model, objF_vi_rho_psi0_3[k=1:K],
            -λU * (1.0 - x_bar[k]) * ρ_psi0_3[k] >= 0)
    end

    vars = Dict(
        :α => α,
        # Auxiliary bilinear
        :ζL => ζL, :ζF => ζF,
        # ISP-L
        :σ_hat => σ_hat, :u_hat => u_hat, :a => a, :b => b, :r => r,
        :ρ_hat_1 => ρ_hat_1, :ρ_hat_2 => ρ_hat_2, :ρ_hat_3 => ρ_hat_3,
        # ISP-F
        :d => d, :e => e, :u_tilde => u_tilde, :σ_tilde => σ_tilde,
        :ω => ω, :β => β, :δ => δ,
        :ρ_tilde_1 => ρ_tilde_1, :ρ_tilde_2 => ρ_tilde_2, :ρ_tilde_3 => ρ_tilde_3,
        :ρ_psi0_1 => ρ_psi0_1, :ρ_psi0_2 => ρ_psi0_2, :ρ_psi0_3 => ρ_psi0_3,
        # Precomputed bounds (for fix/unfix restoration)
        :a_min => a_min, :a_max => a_max,
        :r_max => r_max,
        :d_min => d_min, :d_max => d_max,
    )
    if add_objF_vi
        vars[:objF_vi_rho_tilde_1] = objF_vi_rho_tilde_1
        vars[:objF_vi_rho_tilde_3] = objF_vi_rho_tilde_3
        vars[:objF_vi_rho_psi0_1] = objF_vi_rho_psi0_1
        vars[:objF_vi_rho_psi0_3] = objF_vi_rho_psi0_3
    end

    # Arcwise per-arc 변수 고정 (obj_F 각 term = 0 강제)
    if add_objF_vi_arcwise
        vars[:rho_orig_ub] = rho_upper_bound  # Float64 or nothing, 복원용
        vars[:add_objF_vi_arcwise] = true
        _apply_arcwise_vi_fixings!(vars, td, x_bar)
    end
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

    # Per-term VI 계수 업데이트: 각 제약에서 해당 변수의 coef만 갱신
    if haskey(vars, :objF_vi_rho_tilde_1)
        c_t1 = vars[:objF_vi_rho_tilde_1]
        c_t3 = vars[:objF_vi_rho_tilde_3]
        c_p1 = vars[:objF_vi_rho_psi0_1]
        c_p3 = vars[:objF_vi_rho_psi0_3]
        for k in 1:K, s in 1:S
            set_normalized_coefficient(c_t1[k, s], ρ_tilde_1[k, s], -φ̃U * x_bar_new[k])
            set_normalized_coefficient(c_t3[k, s], ρ_tilde_3[k, s], -φ̃U * (1.0 - x_bar_new[k]))
        end
        for k in 1:K
            set_normalized_coefficient(c_p1[k], ρ_psi0_1[k], -λU * x_bar_new[k])
            set_normalized_coefficient(c_p3[k], ρ_psi0_3[k], -λU * (1.0 - x_bar_new[k]))
        end
    end

    # Arcwise VI 재적용
    if get(vars, :add_objF_vi_arcwise, false)
        _apply_arcwise_vi_fixings!(vars, td, x_bar_new)
    end
end


"""
    solve_true_dro_subproblem!(model, vars, td::TrueDROData, x_bar::Vector{Float64})

Update obj for x̄, solve, return Dict with:
- :Z0_val
- :α_val
- :rho_hat_1_val, :rho_hat_3_val      (K × S)
- :rho_tilde_1_val, :rho_tilde_3_val  (K × S)
- :rho_psi0_1_val, :rho_psi0_3_val    (K)
- :is_optimal (Bool)

TIME_LIMIT with feasible incumbent → :is_optimal=false, incumbent 값 반환.
Cut은 valid (feasible point의 obj ≤ Z₀*), 다만 UB 갱신에는 사용 불가.
"""
function solve_true_dro_subproblem!(model, vars, td::TrueDROData, x_bar::Vector{Float64};
                                    is_global::Bool=true)
    S = td.S
    K = td.num_arcs

    update_true_dro_subproblem_objective!(model, vars, td, x_bar)

    optimize!(model)
    st = termination_status(model)

    has_solution = (st == MOI.OPTIMAL) ||
                   (st == MOI.LOCALLY_SOLVED) ||
                   ((st == MOI.TIME_LIMIT || st == MOI.ITERATION_LIMIT) && has_values(model))

    if !has_solution
        # @infiltrate
        error("True-DRO subproblem: $st (no feasible solution)")
    end

    is_optimal = (st == MOI.OPTIMAL)

    Z0_val = objective_value(model)
    α_val = max.([value(vars[:α][k]) for k in 1:K], 0.0)
    ρ̂1 = [value(vars[:ρ_hat_1][k, s]) for k in 1:K, s in 1:S]
    ρ̂3 = [value(vars[:ρ_hat_3][k, s]) for k in 1:K, s in 1:S]
    ρ̃1 = [value(vars[:ρ_tilde_1][k, s]) for k in 1:K, s in 1:S]
    ρ̃3 = [value(vars[:ρ_tilde_3][k, s]) for k in 1:K, s in 1:S]
    ρ01 = [value(vars[:ρ_psi0_1][k]) for k in 1:K]
    ρ03 = [value(vars[:ρ_psi0_3][k]) for k in 1:K]

    # global solver의 TIME_LIMIT 시 BestBd = Z₀(x̄) 상한 → Benders UB 갱신 가능
    Z0_bound = if is_global && !is_optimal
        objective_bound(model)
    else
        Z0_val
    end

    # r_val (CVaR reweighting): present in full/single variants, not in nominal
    r_val = haskey(vars, :r) ? [value(vars[:r][s]) for s in 1:S] : nothing

    return Dict(
        :Z0_val => Z0_val,
        :Z0_bound => Z0_bound,
        :α_val => α_val,
        :r_val => r_val,
        :rho_hat_1_val => ρ̂1, :rho_hat_3_val => ρ̂3,
        :rho_tilde_1_val => ρ̃1, :rho_tilde_3_val => ρ̃3,
        :rho_psi0_1_val => ρ01, :rho_psi0_3_val => ρ03,
        :is_optimal => is_optimal,
    )
end


# ====================================================================
# α-step LP: a,d를 파라미터로 고정 → quadratic constraint 없이 순수 LP.
# Gurobi NonConvex=2 + fix() 조합이 infeasible 오판하는 경우의 fallback.
# 한번 빌드하고, update_alpha_step_lp!로 a,d,x 갱신하여 재사용.
# ====================================================================

"""
    build_alpha_step_lp(td, x_bar, a_val, d_val, r_val; optimizer)

a,d,r를 파라미터(상수)로 넣은 α-step LP. bilinear subproblem과 동일 구조이나
ζL=α·r, ζF=α·d가 linear constraint.
Returns (model, vars). vars에 constraint ref 포함 (update용).
"""
function build_alpha_step_lp(td::TrueDROData, x_bar::Vector{Float64},
                              a_val::Vector{Float64}, d_val::Vector{Float64},
                              r_val::Vector{Float64};
                              optimizer)
    S = td.S; K = td.num_arcs; m = td.nv1
    Ny = td.Ny; Nts = td.Nts; q = td.q_hat
    ε̂ = td.eps_hat; ε̃ = td.eps_tilde
    ξ = td.xi_bar; v = td.v; w = td.w
    φ̂U = td.phi_hat_U; φ̃U = td.phi_tilde_U; λU = td.lambda_U
    β = td.beta

    a_max = [min(1.0, q[s] + 2ε̂) for s in 1:S]
    r_max_bound = [a_max[s] / (1.0 - β) for s in 1:S]
    d_max = [min(1.0, q[s] + 2ε̃) for s in 1:S]

    model = Model(optimizer)
    set_silent(model)

    # α
    @variable(model, 0 <= α[1:K] <= w)
    @constraint(model, sum(α) <= w)

    # ζL[k,s] = α[k] * r_val[s]  (linear: r is parameter)
    @variable(model, 0 <= ζL[k=1:K, s=1:S] <= w * r_max_bound[s])
    @constraint(model, ζL_def[k=1:K, s=1:S], ζL[k, s] == α[k] * r_val[s])

    # ISP-L variables
    @variable(model, σ_hat[1:S] >= 0)
    @variable(model, u_hat[1:K, 1:S] >= 0)
    @variable(model, 0 <= b[1:S] <= 2ε̂)
    @variable(model, ρ_hat_1[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_2[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_3[1:K, 1:S] >= 0)

    # (DL-1)
    @constraint(model, DL1[j=1:m, s=1:S],
        sum(Ny[j, k] * u_hat[k, s] for k in 1:K) + Nts[j] * σ_hat[s] == 0)
    # (DL-2): -ξ*r - ζL + u + ρ̂₂ - ρ̂₃ ≤ 0   (r is constant → ξ*r in RHS)
    @constraint(model, DL2[k=1:K, s=1:S],
        -ζL[k, s] + u_hat[k, s] + ρ_hat_2[k, s] - ρ_hat_3[k, s] <= ξ[k, s] * r_val[s])
    # (DL-3): v*ξ*r - ρ̂₁ - ρ̂₂ + ρ̂₃ ≤ 0
    @constraint(model, DL3[k=1:K, s=1:S],
        -ρ_hat_1[k, s] - ρ_hat_2[k, s] + ρ_hat_3[k, s] <= -v[k, s] * ξ[k, s] * r_val[s])
    # (DL-4~7): b ≥ |a-q̂|, Σb ≤ 2ε̂  (a is constant)
    @constraint(model, DL4[s=1:S], -b[s] <= q[s] - a_val[s])
    @constraint(model, DL5[s=1:S],  b[s] >= q[s] - a_val[s])
    @constraint(model, DL6_con, sum(b) <= 2ε̂)

    # ζF[k,s] = α[k] * d_val[s]  (linear)
    @variable(model, 0 <= ζF[k=1:K, s=1:S] <= w * d_max[s])
    @constraint(model, ζF_def[k=1:K, s=1:S], ζF[k, s] == α[k] * d_val[s])

    # ISP-F variables
    @variable(model, u_tilde[1:K, 1:S] >= 0)
    @variable(model, σ_tilde[1:S] >= 0)
    @variable(model, ω[1:m, 1:S])
    @variable(model, β[1:K, 1:S] >= 0)
    @variable(model, δ >= 0)
    @variable(model, 0 <= e_var[1:S] <= 2ε̃)
    @variable(model, ρ_tilde_1[1:K, 1:S] >= 0)
    @variable(model, ρ_tilde_2[1:K, 1:S] >= 0)
    @variable(model, ρ_tilde_3[1:K, 1:S] >= 0)
    @variable(model, ρ_psi0_1[1:K] >= 0)
    @variable(model, ρ_psi0_2[1:K] >= 0)
    @variable(model, ρ_psi0_3[1:K] >= 0)

    # (DF-1~3): e ≥ |d-q̂|, Σe ≤ 2ε̃
    @constraint(model, DF1[s=1:S], -e_var[s] <= q[s] - d_val[s])
    @constraint(model, DF2[s=1:S],  e_var[s] >= q[s] - d_val[s])
    @constraint(model, DF3_con, sum(e_var) <= 2ε̃)
    @constraint(model, DF4_con, sum(d_val) == 1)  # trivially satisfied, for completeness
    # (DF-5)
    @constraint(model, DF5[j=1:m, s=1:S],
        sum(Ny[j, k] * u_tilde[k, s] for k in 1:K) + Nts[j] * σ_tilde[s] == 0)
    # (DF-6): -ξ*d - ζF + ũ + ρ̃₂ - ρ̃₃ ≤ 0
    @constraint(model, DF6[k=1:K, s=1:S],
        -ζF[k, s] + u_tilde[k, s] + ρ_tilde_2[k, s] - ρ_tilde_3[k, s] <= ξ[k, s] * d_val[s])
    # (DF-7): v*ξ*d - ρ̃₁ - ρ̃₂ + ρ̃₃ ≤ 0
    @constraint(model, DF7[k=1:K, s=1:S],
        -ρ_tilde_1[k, s] - ρ_tilde_2[k, s] + ρ_tilde_3[k, s] <= -v[k, s] * ξ[k, s] * d_val[s])
    # (DF-8)
    @constraint(model, DF8[k=1:K, s=1:S],
        sum(Ny[j, k] * ω[j, s] for j in 1:m) - β[k, s] <= 0)
    # (DF-9): d + Nts^T ω ≤ 0  (d is constant)
    @constraint(model, DF9[s=1:S],
        sum(Nts[j] * ω[j, s] for j in 1:m) <= -d_val[s])
    # (DF-h)
    @constraint(model, DFh[k=1:K], sum(β[k, s] for s in 1:S) <= δ)
    # (DF-λ)
    @constraint(model, DFlam,
        sum(σ_tilde) >= sum(ξ[k, s] * β[k, s] for k in 1:K, s in 1:S)
            + w * δ + sum(ρ_psi0_2) - sum(ρ_psi0_3))
    # (DF-ψ)
    @constraint(model, DFpsi[k=1:K],
        sum(v[k, s] * ξ[k, s] * β[k, s] for s in 1:S) + ρ_psi0_1[k] + ρ_psi0_2[k]
        >= ρ_psi0_3[k])

    # Objective
    obj_L = sum(σ_hat) -
            φ̂U * sum(x_bar[k] * ρ_hat_1[k, s] for k in 1:K, s in 1:S) -
            φ̂U * sum((1 - x_bar[k]) * ρ_hat_3[k, s] for k in 1:K, s in 1:S)
    obj_F = -φ̃U * sum(x_bar[k] * ρ_tilde_1[k, s] for k in 1:K, s in 1:S) -
             φ̃U * sum((1 - x_bar[k]) * ρ_tilde_3[k, s] for k in 1:K, s in 1:S) -
             λU * sum(x_bar[k] * ρ_psi0_1[k] for k in 1:K) -
             λU * sum((1 - x_bar[k]) * ρ_psi0_3[k] for k in 1:K)
    @objective(model, Max, obj_L + obj_F)

    vars = Dict(
        :α => α, :ζL => ζL, :ζF => ζF,
        :σ_hat => σ_hat, :u_hat => u_hat, :b => b,
        :ρ_hat_1 => ρ_hat_1, :ρ_hat_2 => ρ_hat_2, :ρ_hat_3 => ρ_hat_3,
        :u_tilde => u_tilde, :σ_tilde => σ_tilde,
        :ω => ω, :β => β, :δ => δ, :e_var => e_var,
        :ρ_tilde_1 => ρ_tilde_1, :ρ_tilde_2 => ρ_tilde_2, :ρ_tilde_3 => ρ_tilde_3,
        :ρ_psi0_1 => ρ_psi0_1, :ρ_psi0_2 => ρ_psi0_2, :ρ_psi0_3 => ρ_psi0_3,
        # Constraint refs for update
        :ζL_def => ζL_def, :ζF_def => ζF_def,
        :DL2 => DL2, :DL3 => DL3, :DL4 => DL4, :DL5 => DL5,
        :DF1 => DF1, :DF2 => DF2, :DF6 => DF6, :DF7 => DF7, :DF9 => DF9,
    )
    return model, vars
end


"""
    update_alpha_step_lp!(model, vars, td, x_bar, a_val, d_val, r_val)

α-step LP의 a,d,r 파라미터 및 x̄ 목적함수 계수를 갱신.
"""
function update_alpha_step_lp!(model, vars, td::TrueDROData,
                                x_bar::Vector{Float64},
                                a_val::Vector{Float64}, d_val::Vector{Float64},
                                r_val::Vector{Float64})
    S = td.S; K = td.num_arcs
    ξ = td.xi_bar; v = td.v
    φ̂U = td.phi_hat_U; φ̃U = td.phi_tilde_U; λU = td.lambda_U

    α = vars[:α]

    # ---- r 관련 계수 갱신 (ISP-L: a→r) ----
    for s in 1:S
        # ζL_def: ζL[k,s] == α[k] * r[s]  →  α[k] coef = r[s]
        for k in 1:K
            set_normalized_coefficient(vars[:ζL_def][k, s], α[k], -r_val[s])
        end
        # DL2: ... ≤ ξ*r  →  RHS = ξ[k,s]*r[s]
        for k in 1:K
            set_normalized_rhs(vars[:DL2][k, s], ξ[k, s] * r_val[s])
        end
        # DL3: ... ≤ -v*ξ*r  →  RHS = -v[k,s]*ξ[k,s]*r[s]
        for k in 1:K
            set_normalized_rhs(vars[:DL3][k, s], -v[k, s] * ξ[k, s] * r_val[s])
        end
        # DL4: -b ≤ q-a  →  RHS = q-a
        set_normalized_rhs(vars[:DL4][s], td.q_hat[s] - a_val[s])
        # DL5: b ≥ q-a  →  RHS = q-a
        set_normalized_rhs(vars[:DL5][s], td.q_hat[s] - a_val[s])
    end

    # ---- d 관련 계수 갱신 ----
    for s in 1:S
        for k in 1:K
            set_normalized_coefficient(vars[:ζF_def][k, s], α[k], -d_val[s])
        end
        for k in 1:K
            set_normalized_rhs(vars[:DF6][k, s], ξ[k, s] * d_val[s])
        end
        for k in 1:K
            set_normalized_rhs(vars[:DF7][k, s], -v[k, s] * ξ[k, s] * d_val[s])
        end
        set_normalized_rhs(vars[:DF1][s], td.q_hat[s] - d_val[s])
        set_normalized_rhs(vars[:DF2][s], td.q_hat[s] - d_val[s])
        # DF9: Nts^T ω ≤ -d  →  RHS = -d[s]
        set_normalized_rhs(vars[:DF9][s], -d_val[s])
    end

    # ---- x̄ 목적함수 갱신 ----
    for k in 1:K, s in 1:S
        set_objective_coefficient(model, vars[:ρ_hat_1][k, s], -φ̂U * x_bar[k])
        set_objective_coefficient(model, vars[:ρ_hat_3][k, s], -φ̂U * (1 - x_bar[k]))
        set_objective_coefficient(model, vars[:ρ_tilde_1][k, s], -φ̃U * x_bar[k])
        set_objective_coefficient(model, vars[:ρ_tilde_3][k, s], -φ̃U * (1 - x_bar[k]))
    end
    for k in 1:K
        set_objective_coefficient(model, vars[:ρ_psi0_1][k], -λU * x_bar[k])
        set_objective_coefficient(model, vars[:ρ_psi0_3][k], -λU * (1 - x_bar[k]))
    end
end


"""
    solve_alpha_step_lp!(model, vars, td, x_bar, a_val, d_val, r_val)

α-step LP 갱신 + solve. 반환 형식은 solve_true_dro_subproblem!과 동일.
"""
function solve_alpha_step_lp!(model, vars, td::TrueDROData,
                               x_bar::Vector{Float64},
                               a_val::Vector{Float64}, d_val::Vector{Float64},
                               r_val::Vector{Float64})
    S = td.S; K = td.num_arcs

    update_alpha_step_lp!(model, vars, td, x_bar, a_val, d_val, r_val)
    optimize!(model)
    st = termination_status(model)

    if st != MOI.OPTIMAL
        error("α-step LP: $st")
    end

    Z0_val = objective_value(model)
    α_val = max.([value(vars[:α][k]) for k in 1:K], 0.0)
    ρ̂1 = [value(vars[:ρ_hat_1][k, s]) for k in 1:K, s in 1:S]
    ρ̂3 = [value(vars[:ρ_hat_3][k, s]) for k in 1:K, s in 1:S]
    ρ̃1 = [value(vars[:ρ_tilde_1][k, s]) for k in 1:K, s in 1:S]
    ρ̃3 = [value(vars[:ρ_tilde_3][k, s]) for k in 1:K, s in 1:S]
    ρ01 = [value(vars[:ρ_psi0_1][k]) for k in 1:K]
    ρ03 = [value(vars[:ρ_psi0_3][k]) for k in 1:K]

    return Dict(
        :Z0_val => Z0_val,
        :Z0_bound => Z0_val,
        :α_val => α_val,
        :rho_hat_1_val => ρ̂1, :rho_hat_3_val => ρ̂3,
        :rho_tilde_1_val => ρ̃1, :rho_tilde_3_val => ρ̃3,
        :rho_psi0_1_val => ρ01, :rho_psi0_3_val => ρ03,
        :is_optimal => true,
    )
end


# ====================================================================
# Single-layer compact subproblem (ε̃=0)
#
# ε̃=0 → d[s] = q̂[s] (고정), e=0, ζF[k,s] = α[k]·q̂[s] (linear).
# Bilinear term이 ζL (K×S개)만 남음. 기존 대비 bilinear constraint 절반 제거.
#
# ISP-F의 d 관련 변수/제약 전부 제거:
#   - d, e 변수 제거
#   - ζF 변수 + ζF_def quadratic constraint 제거
#   - DF1~DF4 (d의 TV ball) 제거
#   - DF6, DF7, DF9에서 d[s] → q[s] 상수 대입
#
# vars dict는 기존과 호환: update_true_dro_subproblem_objective!,
# solve_true_dro_subproblem!, _apply_arcwise_vi_fixings! 모두 그대로 사용 가능.
# (이들은 d, e, ζF를 참조하지 않음)
# ====================================================================

"""
    build_true_dro_subproblem_single(td::TrueDROData, x_bar::Vector{Float64}; optimizer, ...)

ε̃=0 전용 compact subproblem. ISP-F의 bilinear term (ζF = α·d) 제거.
Bilinear은 ζL = α·a (K×S개)만 남음.

Returns (model, vars). 기존 solve/update 함수와 호환.
"""
function build_true_dro_subproblem_single(td::TrueDROData, x_bar::Vector{Float64};
                                          optimizer, silent=true,
                                          rho_upper_bound::Union{Float64,Nothing}=nothing,
                                          add_objF_vi::Bool=false,
                                          add_objF_vi_arcwise::Bool=false)
    @assert td.eps_tilde == 0.0 "build_true_dro_subproblem_single requires ε̃=0 (got $(td.eps_tilde))"

    S = td.S
    K = td.num_arcs
    m = td.nv1
    Ny = td.Ny
    Nts = td.Nts
    q = td.q_hat
    ε̂ = td.eps_hat
    ξ = td.xi_bar
    v = td.v
    w = td.w
    φ̂U = td.phi_hat_U
    φ̃U = td.phi_tilde_U
    λU = td.lambda_U
    β = td.beta

    model = Model(optimizer)
    if silent
        set_silent(model)
    end
    set_optimizer_attribute(model, "DualReductions", 0)

    # ====================================================================
    # ISP-L: a, r, ζL=α·r bilinear 유지
    # ====================================================================
    a_min = [max(0.0, q[s] - 2 * ε̂) for s in 1:S]
    a_max = [min(1.0, q[s] + 2 * ε̂) for s in 1:S]
    r_max = [a_max[s] / (1.0 - β) for s in 1:S]

    @variable(model, 0 <= α[1:K] <= w)
    @constraint(model, sum(α[k] for k in 1:K) <= w)

    @variable(model, σ_hat[1:S] >= 0)
    @variable(model, u_hat[1:K, 1:S] >= 0)
    @variable(model, a_min[s] <= a[s=1:S] <= a_max[s])
    @variable(model, 0 <= b[1:S] <= 2 * ε̂)
    @variable(model, 0 <= r[s=1:S] <= r_max[s])
    @variable(model, ρ_hat_1[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_2[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_3[1:K, 1:S] >= 0)

    if rho_upper_bound !== nothing
        for k in 1:K, s in 1:S
            set_upper_bound(ρ_hat_1[k, s], rho_upper_bound)
            set_upper_bound(ρ_hat_2[k, s], rho_upper_bound)
            set_upper_bound(ρ_hat_3[k, s], rho_upper_bound)
        end
    end

    # Bilinear: ζL_{ks} = α_k · r_s (유일한 bilinear term)
    @variable(model, 0 <= ζL[k=1:K, s=1:S] <= w * r_max[s])
    @constraint(model, ζL_def[k=1:K, s=1:S], ζL[k, s] == α[k] * r[s])

    # (DL-1)
    @constraint(model, DL1[j=1:m, s=1:S],
        sum(Ny[j, k] * u_hat[k, s] for k in 1:K) + Nts[j] * σ_hat[s] == 0)
    # (DL-RU1): r[s] ≤ a[s]/(1-β)
    @constraint(model, DL_RU1[s=1:S], r[s] <= a[s] / (1.0 - β))
    # (DL-RU2): Σ r[s] = 1
    @constraint(model, DL_RU2, sum(r[s] for s in 1:S) == 1)
    # (DL-2): bilinear via ζL, a→r
    @constraint(model, DL2[k=1:K, s=1:S],
        -ξ[k, s] * r[s] - ζL[k, s]
        + u_hat[k, s] + ρ_hat_2[k, s] - ρ_hat_3[k, s] <= 0)
    # (DL-3): a→r
    @constraint(model, DL3[k=1:K, s=1:S],
        v[k, s] * ξ[k, s] * r[s]
        - ρ_hat_1[k, s] - ρ_hat_2[k, s] + ρ_hat_3[k, s] <= 0)
    # (DL-4~7)
    @constraint(model, DL4[s=1:S], a[s] - b[s] <= q[s])
    @constraint(model, DL5[s=1:S], a[s] + b[s] >= q[s])
    @constraint(model, DL6, sum(b[s] for s in 1:S) <= 2 * ε̂)
    @constraint(model, DL7, sum(a[s] for s in 1:S) == 1)

    # ====================================================================
    # ISP-F: d[s] = q[s] 상수 대입, ζF = α·q (linear), d/e/DF1~4 제거
    # ====================================================================
    @variable(model, u_tilde[1:K, 1:S] >= 0)
    @variable(model, σ_tilde[1:S] >= 0)
    @variable(model, ω[1:m, 1:S])              # FREE
    @variable(model, β[1:K, 1:S] >= 0)
    @variable(model, δ >= 0)
    @variable(model, ρ_tilde_1[1:K, 1:S] >= 0)
    @variable(model, ρ_tilde_2[1:K, 1:S] >= 0)
    @variable(model, ρ_tilde_3[1:K, 1:S] >= 0)
    @variable(model, ρ_psi0_1[1:K] >= 0)
    @variable(model, ρ_psi0_2[1:K] >= 0)
    @variable(model, ρ_psi0_3[1:K] >= 0)

    if rho_upper_bound !== nothing
        for k in 1:K, s in 1:S
            set_upper_bound(ρ_tilde_1[k, s], rho_upper_bound)
            set_upper_bound(ρ_tilde_2[k, s], rho_upper_bound)
            set_upper_bound(ρ_tilde_3[k, s], rho_upper_bound)
        end
        for k in 1:K
            set_upper_bound(ρ_psi0_1[k], rho_upper_bound)
            set_upper_bound(ρ_psi0_2[k], rho_upper_bound)
            set_upper_bound(ρ_psi0_3[k], rho_upper_bound)
        end
    end

    # (DF-5): flow conservation (동일)
    @constraint(model, DF5[j=1:m, s=1:S],
        sum(Ny[j, k] * u_tilde[k, s] for k in 1:K) + Nts[j] * σ_tilde[s] == 0)

    # (DF-6): -(ξ_k^s + α_k)·q_s + ũ + ρ̃₂ - ρ̃₃ ≤ 0   [LINEAR in α]
    @constraint(model, DF6[k=1:K, s=1:S],
        -ξ[k, s] * q[s] - α[k] * q[s]
        + u_tilde[k, s] + ρ_tilde_2[k, s] - ρ_tilde_3[k, s] <= 0)

    # (DF-7): v_k^s·ξ_k^s·q_s - ρ̃₁ - ρ̃₂ + ρ̃₃ ≤ 0   [상수 RHS]
    @constraint(model, DF7[k=1:K, s=1:S],
        v[k, s] * ξ[k, s] * q[s]
        - ρ_tilde_1[k, s] - ρ_tilde_2[k, s] + ρ_tilde_3[k, s] <= 0)

    # (DF-8)
    @constraint(model, DF8[k=1:K, s=1:S],
        sum(Ny[j, k] * ω[j, s] for j in 1:m) - β[k, s] <= 0)

    # (DF-9): q_s + N_ts^T ω ≤ 0
    @constraint(model, DF9[s=1:S],
        q[s] + sum(Nts[j] * ω[j, s] for j in 1:m) <= 0)

    # (DF-h)
    @constraint(model, DFh[k=1:K], sum(β[k, s] for s in 1:S) <= δ)

    # (DF-λ)
    @constraint(model, DFlam,
        sum(σ_tilde[s] for s in 1:S)
        >= sum(ξ[k, s] * β[k, s] for k in 1:K, s in 1:S)
           + w * δ
           + sum(ρ_psi0_2[k] for k in 1:K)
           - sum(ρ_psi0_3[k] for k in 1:K))

    # (DF-ψ)
    @constraint(model, DFpsi[k=1:K],
        sum(v[k, s] * ξ[k, s] * β[k, s] for s in 1:S) + ρ_psi0_1[k] + ρ_psi0_2[k]
        >= ρ_psi0_3[k])

    # ====================================================================
    # Objective (동일 구조)
    # ====================================================================
    obj_L = sum(σ_hat[s] for s in 1:S) -
            φ̂U * sum(x_bar[k] * ρ_hat_1[k, s] for k in 1:K, s in 1:S) -
            φ̂U * sum((1.0 - x_bar[k]) * ρ_hat_3[k, s] for k in 1:K, s in 1:S)

    obj_F = -φ̃U * sum(x_bar[k] * ρ_tilde_1[k, s] for k in 1:K, s in 1:S) -
             φ̃U * sum((1.0 - x_bar[k]) * ρ_tilde_3[k, s] for k in 1:K, s in 1:S) -
             λU * sum(x_bar[k] * ρ_psi0_1[k] for k in 1:K) -
             λU * sum((1.0 - x_bar[k]) * ρ_psi0_3[k] for k in 1:K)

    @objective(model, Max, obj_L + obj_F)

    # Valid Inequality (동일)
    if add_objF_vi
        @constraint(model, objF_vi_rho_tilde_1[k=1:K, s=1:S],
            -φ̃U * x_bar[k] * ρ_tilde_1[k, s] >= 0)
        @constraint(model, objF_vi_rho_tilde_3[k=1:K, s=1:S],
            -φ̃U * (1.0 - x_bar[k]) * ρ_tilde_3[k, s] >= 0)
        @constraint(model, objF_vi_rho_psi0_1[k=1:K],
            -λU * x_bar[k] * ρ_psi0_1[k] >= 0)
        @constraint(model, objF_vi_rho_psi0_3[k=1:K],
            -λU * (1.0 - x_bar[k]) * ρ_psi0_3[k] >= 0)
    end

    vars = Dict(
        :α => α,
        :ζL => ζL,
        # ISP-L
        :σ_hat => σ_hat, :u_hat => u_hat, :a => a, :b => b, :r => r,
        :ρ_hat_1 => ρ_hat_1, :ρ_hat_2 => ρ_hat_2, :ρ_hat_3 => ρ_hat_3,
        # ISP-F (d, e, ζF 없음)
        :u_tilde => u_tilde, :σ_tilde => σ_tilde,
        :ω => ω, :β => β, :δ => δ,
        :ρ_tilde_1 => ρ_tilde_1, :ρ_tilde_2 => ρ_tilde_2, :ρ_tilde_3 => ρ_tilde_3,
        :ρ_psi0_1 => ρ_psi0_1, :ρ_psi0_2 => ρ_psi0_2, :ρ_psi0_3 => ρ_psi0_3,
        # Bounds (a만, d는 없음)
        :a_min => a_min, :a_max => a_max,
        :r_max => r_max,
        :is_single_compact => true,
    )
    if add_objF_vi
        vars[:objF_vi_rho_tilde_1] = objF_vi_rho_tilde_1
        vars[:objF_vi_rho_tilde_3] = objF_vi_rho_tilde_3
        vars[:objF_vi_rho_psi0_1] = objF_vi_rho_psi0_1
        vars[:objF_vi_rho_psi0_3] = objF_vi_rho_psi0_3
    end

    if add_objF_vi_arcwise
        vars[:rho_orig_ub] = rho_upper_bound
        vars[:add_objF_vi_arcwise] = true
        _apply_arcwise_vi_fixings!(vars, td, x_bar)
    end

    return model, vars
end


"""
    build_true_dro_subproblem_nominal(td, x_bar; optimizer, ...)

ε̂=ε̃=0 전용 compact subproblem (nominal SP).
a[s]=d[s]=q̂[s] 상수 대입 → ζL, ζF 모두 제거 → **순수 LP**.
bilinear constraint 0개.
"""
function build_true_dro_subproblem_nominal(td::TrueDROData, x_bar::Vector{Float64};
                                          optimizer, silent=true,
                                          rho_upper_bound::Union{Float64,Nothing}=nothing,
                                          add_objF_vi::Bool=false,
                                          add_objF_vi_arcwise::Bool=false)
    @assert td.eps_hat == 0.0 "build_true_dro_subproblem_nominal requires ε̂=0 (got $(td.eps_hat))"
    @assert td.eps_tilde == 0.0 "build_true_dro_subproblem_nominal requires ε̃=0 (got $(td.eps_tilde))"

    S = td.S
    K = td.num_arcs
    m = td.nv1
    Ny = td.Ny
    Nts = td.Nts
    q = td.q_hat
    ξ = td.xi_bar
    v = td.v
    w = td.w
    φ̂U = td.phi_hat_U
    φ̃U = td.phi_tilde_U
    λU = td.lambda_U

    model = Model(optimizer)
    if silent
        set_silent(model)
    end
    set_optimizer_attribute(model, "DualReductions", 0)

    # ====================================================================
    # α (linear로만 등장 — bilinear 없음)
    # ====================================================================
    @variable(model, 0 <= α[1:K] <= w)
    @constraint(model, sum(α[k] for k in 1:K) <= w)

    # ====================================================================
    # ISP-L: a[s]=q̂[s] 대입, a,b,ζL 제거
    # ====================================================================
    @variable(model, σ_hat[1:S] >= 0)
    @variable(model, u_hat[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_1[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_2[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_3[1:K, 1:S] >= 0)

    if rho_upper_bound !== nothing
        for k in 1:K, s in 1:S
            set_upper_bound(ρ_hat_1[k, s], rho_upper_bound)
            set_upper_bound(ρ_hat_2[k, s], rho_upper_bound)
            set_upper_bound(ρ_hat_3[k, s], rho_upper_bound)
        end
    end

    # (DL-1): flow conservation
    @constraint(model, DL1[j=1:m, s=1:S],
        sum(Ny[j, k] * u_hat[k, s] for k in 1:K) + Nts[j] * σ_hat[s] == 0)

    # (DL-2): -(ξ_k^s + α_k)·q_s + û + ρ̂₂ - ρ̂₃ ≤ 0  [LINEAR in α]
    @constraint(model, DL2[k=1:K, s=1:S],
        -ξ[k, s] * q[s] - α[k] * q[s]
        + u_hat[k, s] + ρ_hat_2[k, s] - ρ_hat_3[k, s] <= 0)

    # (DL-3): v_k^s·ξ_k^s·q_s - ρ̂₁ - ρ̂₂ + ρ̂₃ ≤ 0  [상수]
    @constraint(model, DL3[k=1:K, s=1:S],
        v[k, s] * ξ[k, s] * q[s]
        - ρ_hat_1[k, s] - ρ_hat_2[k, s] + ρ_hat_3[k, s] <= 0)

    # DL-4~7 (TV ball): a=q̂ 고정이므로 자동 만족 → 제거

    # ====================================================================
    # ISP-F: d[s]=q̂[s] 대입, d,e,ζF 제거 (single compact과 동일)
    # ====================================================================
    @variable(model, u_tilde[1:K, 1:S] >= 0)
    @variable(model, σ_tilde[1:S] >= 0)
    @variable(model, ω[1:m, 1:S])              # FREE
    @variable(model, β[1:K, 1:S] >= 0)
    @variable(model, δ >= 0)
    @variable(model, ρ_tilde_1[1:K, 1:S] >= 0)
    @variable(model, ρ_tilde_2[1:K, 1:S] >= 0)
    @variable(model, ρ_tilde_3[1:K, 1:S] >= 0)
    @variable(model, ρ_psi0_1[1:K] >= 0)
    @variable(model, ρ_psi0_2[1:K] >= 0)
    @variable(model, ρ_psi0_3[1:K] >= 0)

    if rho_upper_bound !== nothing
        for k in 1:K, s in 1:S
            set_upper_bound(ρ_tilde_1[k, s], rho_upper_bound)
            set_upper_bound(ρ_tilde_2[k, s], rho_upper_bound)
            set_upper_bound(ρ_tilde_3[k, s], rho_upper_bound)
        end
        for k in 1:K
            set_upper_bound(ρ_psi0_1[k], rho_upper_bound)
            set_upper_bound(ρ_psi0_2[k], rho_upper_bound)
            set_upper_bound(ρ_psi0_3[k], rho_upper_bound)
        end
    end

    # (DF-5)
    @constraint(model, DF5[j=1:m, s=1:S],
        sum(Ny[j, k] * u_tilde[k, s] for k in 1:K) + Nts[j] * σ_tilde[s] == 0)

    # (DF-6): -(ξ_k^s + α_k)·q_s + ũ + ρ̃₂ - ρ̃₃ ≤ 0  [LINEAR in α]
    @constraint(model, DF6[k=1:K, s=1:S],
        -ξ[k, s] * q[s] - α[k] * q[s]
        + u_tilde[k, s] + ρ_tilde_2[k, s] - ρ_tilde_3[k, s] <= 0)

    # (DF-7): v_k^s·ξ_k^s·q_s - ρ̃₁ - ρ̃₂ + ρ̃₃ ≤ 0  [상수]
    @constraint(model, DF7[k=1:K, s=1:S],
        v[k, s] * ξ[k, s] * q[s]
        - ρ_tilde_1[k, s] - ρ_tilde_2[k, s] + ρ_tilde_3[k, s] <= 0)

    # (DF-8)
    @constraint(model, DF8[k=1:K, s=1:S],
        sum(Ny[j, k] * ω[j, s] for j in 1:m) - β[k, s] <= 0)

    # (DF-9): q_s + N_ts^T ω ≤ 0
    @constraint(model, DF9[s=1:S],
        q[s] + sum(Nts[j] * ω[j, s] for j in 1:m) <= 0)

    # (DF-h)
    @constraint(model, DFh[k=1:K], sum(β[k, s] for s in 1:S) <= δ)

    # (DF-λ)
    @constraint(model, DFlam,
        sum(σ_tilde[s] for s in 1:S)
        >= sum(ξ[k, s] * β[k, s] for k in 1:K, s in 1:S)
           + w * δ
           + sum(ρ_psi0_2[k] for k in 1:K)
           - sum(ρ_psi0_3[k] for k in 1:K))

    # (DF-ψ)
    @constraint(model, DFpsi[k=1:K],
        sum(v[k, s] * ξ[k, s] * β[k, s] for s in 1:S) + ρ_psi0_1[k] + ρ_psi0_2[k]
        >= ρ_psi0_3[k])

    # ====================================================================
    # Objective
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
        # ISP-L (a, b, ζL 없음)
        :σ_hat => σ_hat, :u_hat => u_hat,
        :ρ_hat_1 => ρ_hat_1, :ρ_hat_2 => ρ_hat_2, :ρ_hat_3 => ρ_hat_3,
        # ISP-F (d, e, ζF 없음)
        :u_tilde => u_tilde, :σ_tilde => σ_tilde,
        :ω => ω, :β => β, :δ => δ,
        :ρ_tilde_1 => ρ_tilde_1, :ρ_tilde_2 => ρ_tilde_2, :ρ_tilde_3 => ρ_tilde_3,
        :ρ_psi0_1 => ρ_psi0_1, :ρ_psi0_2 => ρ_psi0_2, :ρ_psi0_3 => ρ_psi0_3,
        :is_nominal_compact => true,
        :is_single_compact => true,  # d 없음 guard 호환
    )

    return model, vars
end
