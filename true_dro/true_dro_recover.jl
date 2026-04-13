"""
true_dro_recover.jl — Primal recovery: x* → α* → (h*, λ*, ψ⁰*).

Benders 수렴 후 primal follower decisions 복구.

Step 1: x* 고정, subproblem 풀기 → α* (이미 Benders에서 수행)
Step 2: α*, x* 고정, Piece-F primal LP (§7.1) 풀기 → h*, λ*, ψ⁰*

Piece-F primal (§7.1, true_dro_v5.md):
  min  Σ_s q̂_s(σ^{F+}_s - σ^{F-}_s) + 2ε̃·μ^F + η^F
  s.t. (1)-(7) + McCormick(ψ̃≈x̄·φ̃) + McCormick(ψ⁰≈λ·x̄)

α is parameter in (1): coefficient -(ξ̄+α_k). With α fixed → LP.
"""

using JuMP
using LinearAlgebra
using Printf


"""
    build_primal_piece_F(td::TrueDROData, x_bar, α_bar; optimizer)

Build Piece-F primal LP (§7.1) with fixed ᾱ, x̄. All linear.

Returns (model, vars) with vars containing :h, :λ, :ψ0, :y_tilde, :y_ts, etc.
"""
function build_primal_piece_F(td::TrueDROData, x_bar::Vector{Float64},
                               α_bar::Vector{Float64}; optimizer)
    S = td.S
    K = td.num_arcs
    m = td.nv1
    Ny = td.Ny
    Nts = td.Nts
    q = td.q_hat
    ε̃ = td.eps_tilde
    ξ = td.xi_bar
    v = td.v
    w = td.w
    φ̃U = td.phi_tilde_U
    λU = td.lambda_U

    model = Model(optimizer)
    set_silent(model)

    # --- TV envelope variables ---
    @variable(model, σ_Fp[1:S] >= 0)   # σ^{F+}
    @variable(model, σ_Fm[1:S] >= 0)   # σ^{F-}
    @variable(model, μ_F >= 0)
    @variable(model, η_F)              # free

    # --- Follower dual variables ---
    @variable(model, π_tilde[1:m, 1:S])          # free
    @variable(model, φ_tilde[1:K, 1:S] >= 0)
    @variable(model, ψ_tilde[1:K, 1:S] >= 0)     # McCormick ψ̃ ≈ x̄·φ̃

    # --- Follower primal variables ---
    @variable(model, y_tilde[1:K, 1:S] >= 0)
    @variable(model, y_ts[1:S] >= 0)

    # --- Budget/coupling variables ---
    @variable(model, h[1:K] >= 0)
    @variable(model, λ >= 0)
    @variable(model, ψ0[1:K] >= 0)               # McCormick ψ⁰ ≈ λ·x̄

    # ================================================================
    # Constraints (§7.1)
    # ================================================================

    # (1) σ^{F+}_s - σ^{F-}_s + η^F
    #     - Σ_k (ξ̄_k^s + α_k) φ̃_k^s + Σ_k v_k ξ̄_k^s ψ̃_k^s + ỹ_ts^s ≥ 0   ∀s
    @constraint(model, C1[s=1:S],
        σ_Fp[s] - σ_Fm[s] + η_F
        - sum((ξ[k, s] + α_bar[k]) * φ_tilde[k, s] for k in 1:K)
        + sum(v[k] * ξ[k, s] * ψ_tilde[k, s] for k in 1:K)
        + y_ts[s] >= 0)

    # (2) μ^F - σ^{F+}_s - σ^{F-}_s ≥ 0   ∀s
    @constraint(model, C2[s=1:S], μ_F - σ_Fp[s] - σ_Fm[s] >= 0)

    # (3) [N_yᵀ π̃ˢ]_k + φ̃_k^s ≥ 0   ∀k,s
    @constraint(model, C3[k=1:K, s=1:S],
        sum(Ny[j, k] * π_tilde[j, s] for j in 1:m) + φ_tilde[k, s] >= 0)

    # (4) N_tsᵀ π̃ˢ - λ ≥ 0   ∀s
    @constraint(model, C4[s=1:S],
        sum(Nts[j] * π_tilde[j, s] for j in 1:m) - λ >= 0)

    # (5) N_y ỹˢ + N_ts ỹ_ts^s = 0   ∀s
    @constraint(model, C5[j=1:m, s=1:S],
        sum(Ny[j, k] * y_tilde[k, s] for k in 1:K) + Nts[j] * y_ts[s] == 0)

    # (6) -ỹ_k^s + h_k + λ ξ̄_k^s - v_k ψ⁰_k ξ̄_k^s ≥ 0   ∀k,s
    @constraint(model, C6[k=1:K, s=1:S],
        -y_tilde[k, s] + h[k] + λ * ξ[k, s] - v[k] * ψ0[k] * ξ[k, s] >= 0)

    # (7) λw - Σ_k h_k ≥ 0
    @constraint(model, C7, λ * w - sum(h[k] for k in 1:K) >= 0)

    # McCormick: ψ̃_k^s ≈ x̄_k · φ̃_k^s  (8)-(10)
    @constraint(model, MC_psi_1[k=1:K, s=1:S], ψ_tilde[k, s] <= φ̃U * x_bar[k])
    @constraint(model, MC_psi_2[k=1:K, s=1:S], ψ_tilde[k, s] <= φ_tilde[k, s])
    @constraint(model, MC_psi_3[k=1:K, s=1:S],
        ψ_tilde[k, s] >= φ_tilde[k, s] - φ̃U * (1 - x_bar[k]))

    # McCormick: ψ⁰_k ≈ λ · x̄_k  (11)-(13)
    @constraint(model, MC_psi0_1[k=1:K], ψ0[k] <= λU * x_bar[k])
    @constraint(model, MC_psi0_2[k=1:K], ψ0[k] <= λ)
    @constraint(model, MC_psi0_3[k=1:K], ψ0[k] >= λ - λU * (1 - x_bar[k]))

    # λ upper bound
    @constraint(model, λ <= λU)

    # ================================================================
    # Objective: min
    # ================================================================
    @objective(model, Min,
        sum(q[s] * (σ_Fp[s] - σ_Fm[s]) for s in 1:S)
        + 2 * ε̃ * μ_F + η_F)

    vars = Dict(
        :σ_Fp => σ_Fp, :σ_Fm => σ_Fm, :μ_F => μ_F, :η_F => η_F,
        :π_tilde => π_tilde, :φ_tilde => φ_tilde, :ψ_tilde => ψ_tilde,
        :y_tilde => y_tilde, :y_ts => y_ts,
        :h => h, :λ => λ, :ψ0 => ψ0,
    )
    return model, vars
end


"""
    recover_follower_decisions(td::TrueDROData, x_sol, α_sol; optimizer)

x*, α* 고정 → Piece-F primal LP 풀어서 h*, λ*, ψ⁰* 복구.

Returns Dict with:
  :h, :λ, :ψ0, :obj_F (Piece-F value),
  :y_tilde (K×S flow), :y_ts (S sink flow)
"""
function recover_follower_decisions(td::TrueDROData,
                                     x_sol::Vector{Float64},
                                     α_sol::Vector{Float64};
                                     optimizer)
    S = td.S
    K = td.num_arcs

    model, vars = build_primal_piece_F(td, x_sol, α_sol; optimizer=optimizer)
    optimize!(model)

    st = termination_status(model)
    if st != MOI.OPTIMAL
        error("Piece-F primal not optimal: $st")
    end

    obj_F = objective_value(model)
    h_sol = [value(vars[:h][k]) for k in 1:K]
    λ_sol = value(vars[:λ])
    ψ0_sol = [value(vars[:ψ0][k]) for k in 1:K]
    y_tilde_sol = [value(vars[:y_tilde][k, s]) for k in 1:K, s in 1:S]
    y_ts_sol = [value(vars[:y_ts][s]) for s in 1:S]

    return Dict(
        :h => h_sol,
        :λ => λ_sol,
        :ψ0 => ψ0_sol,
        :obj_F => obj_F,
        :y_tilde => y_tilde_sol,
        :y_ts => y_ts_sol,
    )
end


"""
    recover_and_print(td::TrueDROData, benders_result; optimizer)

Benders 결과로부터 full primal recovery + 출력.
"""
function recover_and_print(td::TrueDROData, benders_result::Dict; optimizer)
    x_sol = benders_result[:x]
    α_sol = benders_result[:α]
    K = td.num_arcs
    S = td.S

    println("=" ^ 60)
    println("Primal Recovery: x* → α* → (h*, λ*, ψ⁰*)")
    println("=" ^ 60)

    x_int = round.(Int, x_sol)
    @printf("  x* = %s\n", string(x_int))
    α_str = join([@sprintf("%.4f", a) for a in α_sol], ", ")
    @printf("  α* = [%s]\n", α_str)

    rec = recover_follower_decisions(td, x_sol, α_sol; optimizer=optimizer)

    @printf("\n  Piece-F obj = %.6f\n", rec[:obj_F])
    @printf("  λ*   = %.6f\n", rec[:λ])
    @printf("  Σh*  = %.6f  (budget: λw = %.6f)\n", sum(rec[:h]), rec[:λ] * td.w)

    # h per arc (nonzero only)
    println("\n  h* (nonzero arcs):")
    for k in 1:K
        if rec[:h][k] > 1e-6
            @printf("    arc %2d: h=%.6f\n", k, rec[:h][k])
        end
    end

    # ψ⁰ per arc (nonzero only)
    println("  ψ⁰* (nonzero arcs):")
    for k in 1:K
        if rec[:ψ0][k] > 1e-6
            @printf("    arc %2d: ψ⁰=%.6f  (x=%d)\n", k, rec[:ψ0][k], round(Int, x_sol[k]))
        end
    end

    # Per-scenario sink flow
    println("\n  Per-scenario sink flow ỹ_ts:")
    for s in 1:S
        @printf("    s=%d: ỹ_ts=%.6f\n", s, rec[:y_ts][s])
    end

    # Verification: Z₀ = Piece-L + Piece-F
    # Piece-L을 직접 검증하려면 ISP-L primal도 필요하지만, subproblem의 Z₀와 비교 가능
    Z0_benders = benders_result[:Z0]
    @printf("\n  Z₀(Benders) = %.6f\n", Z0_benders)
    @printf("  Piece-F(recovered) = %.6f\n", rec[:obj_F])

    return rec
end
