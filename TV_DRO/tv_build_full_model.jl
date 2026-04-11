"""
tv_build_full_model.jl — Full TV-DRO MILP (T1-T18) for verification.

Small instance용 직접 풀기. Benders 결과와 비교하여 검증.
McCormick linearization for x·φ̂ and x·φ̃ bilinearity in g_s^L, g_s^F.
"""

using JuMP
using LinearAlgebra


"""
    build_full_tv_model(tv::TVData; optimizer)

Build complete TV-DRO MILP (T1-T18 + McCormick for bilinear terms).

# Returns
- `(model, vars)` with all decision variables
"""
function build_full_tv_model(tv::TVData; optimizer)
    S = tv.S
    K = tv.num_arcs
    m = tv.nv1
    Ny = tv.Ny
    Nts = tv.Nts
    q = tv.q_hat
    ε̂ = tv.eps_hat
    ε̃ = tv.eps_tilde
    ξ = tv.xi_bar
    v = tv.v
    γ = tv.gamma
    w = tv.w
    λU = tv.lambda_U

    # Big-M for McCormick on x·φ̂, x·φ̃
    φ̂_U = tv.phi_hat_U
    φ̃_U = tv.phi_tilde_U

    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => false))

    # =============================================
    # First-stage variables
    # =============================================
    @variable(model, x[1:K], Bin)
    @variable(model, h[1:K] >= 0)
    @variable(model, λ >= 0)
    @variable(model, ψ0[1:K] >= 0)  # = λ·x (McCormick)
    @variable(model, t)              # epigraph (free)
    @variable(model, ν >= 0)

    # =============================================
    # TV dual variables — objective coupling
    # =============================================
    # Leader
    @variable(model, σ_Lp[1:S] >= 0)
    @variable(model, σ_Lm[1:S] >= 0)
    @variable(model, μ_L >= 0)
    @variable(model, η_L)  # free

    # Follower
    @variable(model, σ_Fp[1:S] >= 0)
    @variable(model, σ_Fm[1:S] >= 0)
    @variable(model, μ_F >= 0)
    @variable(model, η_F)  # free

    # =============================================
    # TV dual variables — ν coupling
    # =============================================
    # Leader per k
    @variable(model, σ_Lνp[1:S, 1:K] >= 0)
    @variable(model, σ_Lνm[1:S, 1:K] >= 0)
    @variable(model, μ_Lν[1:K] >= 0)
    @variable(model, η_Lν[1:K])  # free

    # Follower per k
    @variable(model, σ_Fνp[1:S, 1:K] >= 0)
    @variable(model, σ_Fνm[1:S, 1:K] >= 0)
    @variable(model, μ_Fν[1:K] >= 0)
    @variable(model, η_Fν[1:K])  # free

    # =============================================
    # Scenario recourse — Leader
    # =============================================
    @variable(model, π_hat[1:m, 1:S] >= 0)
    @variable(model, φ_hat[1:K, 1:S] >= 0)

    # =============================================
    # Scenario recourse — Follower
    # =============================================
    @variable(model, π_tilde[1:m, 1:S] >= 0)
    @variable(model, φ_tilde[1:K, 1:S] >= 0)
    @variable(model, y_tilde[1:K, 1:S] >= 0)
    @variable(model, yts_tilde[1:S] >= 0)

    # =============================================
    # McCormick for bilinear: ψ̂_k^s = x_k · φ̂_k^s
    # =============================================
    @variable(model, ψ_hat_mc[1:K, 1:S] >= 0)
    @constraint(model, [k=1:K, s=1:S], ψ_hat_mc[k, s] <= φ̂_U * x[k])
    @constraint(model, [k=1:K, s=1:S], ψ_hat_mc[k, s] <= φ_hat[k, s])
    @constraint(model, [k=1:K, s=1:S], ψ_hat_mc[k, s] >= φ_hat[k, s] - φ̂_U * (1 - x[k]))

    # McCormick for bilinear: ψ̃_k^s = x_k · φ̃_k^s
    @variable(model, ψ_tilde_mc[1:K, 1:S] >= 0)
    @constraint(model, [k=1:K, s=1:S], ψ_tilde_mc[k, s] <= φ̃_U * x[k])
    @constraint(model, [k=1:K, s=1:S], ψ_tilde_mc[k, s] <= φ_tilde[k, s])
    @constraint(model, [k=1:K, s=1:S], ψ_tilde_mc[k, s] >= φ_tilde[k, s] - φ̃_U * (1 - x[k]))

    # =============================================
    # Constraints
    # =============================================

    # g_s^L = Σ_k ξ̄_k^s (φ̂_k^s - v_k ψ̂_mc_k^s)
    # g_s^F = Σ_k ξ̄_k^s (φ̃_k^s - v_k ψ̃_mc_k^s) - ỹ_ts^s

    # --- (T1): Epigraph ---
    @constraint(model,
        sum(q[s] * (σ_Lp[s] - σ_Lm[s]) for s in 1:S) + 2ε̂ * μ_L + η_L
        + sum(q[s] * (σ_Fp[s] - σ_Fm[s]) for s in 1:S) + 2ε̃ * μ_F + η_F
        <= t)

    # --- (T2): σ_s^{L+} - σ_s^{L-} + η^L ≥ g_s^L ---
    @constraint(model, [s=1:S],
        σ_Lp[s] - σ_Lm[s] + η_L >=
        sum(ξ[k, s] * (φ_hat[k, s] - v[k] * ψ_hat_mc[k, s]) for k in 1:K))

    # --- (T3): σ_s^{L+} + σ_s^{L-} ≤ μ^L ---
    @constraint(model, [s=1:S], σ_Lp[s] + σ_Lm[s] <= μ_L)

    # --- (T4): σ_s^{F+} - σ_s^{F-} + η^F ≥ g_s^F ---
    @constraint(model, [s=1:S],
        σ_Fp[s] - σ_Fm[s] + η_F >=
        sum(ξ[k, s] * (φ_tilde[k, s] - v[k] * ψ_tilde_mc[k, s]) for k in 1:K)
        - yts_tilde[s])

    # --- (T5): σ_s^{F+} + σ_s^{F-} ≤ μ^F ---
    @constraint(model, [s=1:S], σ_Fp[s] + σ_Fm[s] <= μ_F)

    # --- (T6): Leader dual feasibility ---
    @constraint(model, [k=1:K, s=1:S],
        sum(Ny[j, k] * π_hat[j, s] for j in 1:m) + φ_hat[k, s] >= 0)

    # --- (T7): N_tsᵀ π̂^s ≥ 1 ---
    @constraint(model, [s=1:S],
        sum(Nts[j] * π_hat[j, s] for j in 1:m) >= 1)

    # --- (T8): Follower dual feasibility ---
    @constraint(model, [k=1:K, s=1:S],
        sum(Ny[j, k] * π_tilde[j, s] for j in 1:m) + φ_tilde[k, s] >= 0)

    # --- (T9): N_tsᵀ π̃^s ≥ λ ---
    @constraint(model, [s=1:S],
        sum(Nts[j] * π_tilde[j, s] for j in 1:m) >= λ)

    # --- (T10): ν coupling ---
    @constraint(model, [k=1:K],
        sum(q[s] * (σ_Lνp[s, k] - σ_Lνm[s, k]) for s in 1:S) + 2ε̂ * μ_Lν[k] + η_Lν[k]
        + sum(q[s] * (σ_Fνp[s, k] - σ_Fνm[s, k]) for s in 1:S) + 2ε̃ * μ_Fν[k] + η_Fν[k]
        <= ν)

    # --- (T11): σ_{s,k}^{Lν+} - σ_{s,k}^{Lν-} + η_k^{Lν} ≥ φ̂_k^s ---
    @constraint(model, [s=1:S, k=1:K],
        σ_Lνp[s, k] - σ_Lνm[s, k] + η_Lν[k] >= φ_hat[k, s])

    # --- (T12): σ_{s,k}^{Lν+} + σ_{s,k}^{Lν-} ≤ μ_k^{Lν} ---
    @constraint(model, [s=1:S, k=1:K],
        σ_Lνp[s, k] + σ_Lνm[s, k] <= μ_Lν[k])

    # --- (T13): σ_{s,k}^{Fν+} - σ_{s,k}^{Fν-} + η_k^{Fν} ≥ φ̃_k^s ---
    @constraint(model, [s=1:S, k=1:K],
        σ_Fνp[s, k] - σ_Fνm[s, k] + η_Fν[k] >= φ_tilde[k, s])

    # --- (T14): σ_{s,k}^{Fν+} + σ_{s,k}^{Fν-} ≤ μ_k^{Fν} ---
    @constraint(model, [s=1:S, k=1:K],
        σ_Fνp[s, k] + σ_Fνm[s, k] <= μ_Fν[k])

    # --- (T15): Follower primal feasibility: N_y ỹ^s + N_ts ỹ_ts^s ≤ 0 ---
    @constraint(model, [j=1:m, s=1:S],
        sum(Ny[j, k] * y_tilde[k, s] for k in 1:K) + Nts[j] * yts_tilde[s] <= 0)

    # --- (T16): ỹ_k^s ≤ h_k + (λ - v_k ψ_k⁰) ξ̄_k^s ---
    @constraint(model, [k=1:K, s=1:S],
        y_tilde[k, s] <= h[k] + (λ - v[k] * ψ0[k]) * ξ[k, s])

    # --- (T17): Budget ---
    @constraint(model, sum(h[k] for k in 1:K) <= λ * w)

    # --- Integrality + budget ---
    @constraint(model, sum(x[k] for k in 1:K) <= γ)
    for k in 1:K
        if !tv.interdictable_arcs[k]
            @constraint(model, x[k] == 0)
        end
    end
    @constraint(model, λ <= λU)
    # @constraint(model, λ >= 0.01)
    # --- McCormick for ψ⁰ = λ·x ---
    @constraint(model, [k=1:K], ψ0[k] <= λU * x[k])
    @constraint(model, [k=1:K], ψ0[k] <= λ)
    @constraint(model, [k=1:K], ψ0[k] >= λ - λU * (1 - x[k]))

    # --- Objective ---
    @objective(model, Min, t + w * ν)

    vars = Dict(
        :x => x, :h => h, :λ => λ, :ψ0 => ψ0, :t => t, :ν => ν,
        :π_hat => π_hat, :φ_hat => φ_hat,
        :π_tilde => π_tilde, :φ_tilde => φ_tilde,
        :y_tilde => y_tilde, :yts_tilde => yts_tilde,
        :σ_Lp => σ_Lp, :σ_Lm => σ_Lm, :μ_L => μ_L, :η_L => η_L,
        :σ_Fp => σ_Fp, :σ_Fm => σ_Fm, :μ_F => μ_F, :η_F => η_F,
        :σ_Lνp => σ_Lνp, :σ_Lνm => σ_Lνm, :μ_Lν => μ_Lν, :η_Lν => η_Lν,
        :σ_Fνp => σ_Fνp, :σ_Fνm => σ_Fνm, :μ_Fν => μ_Fν, :η_Fν => η_Fν,
        :ψ_hat_mc => ψ_hat_mc, :ψ_tilde_mc => ψ_tilde_mc,
    )
    return model, vars
end
