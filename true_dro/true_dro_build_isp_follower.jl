"""
true_dro_build_isp_follower.jl — ISP-F LP for mini-Benders (true_dro_v5.md §7.2).

α를 파라미터로 고정하면 bilinear term ζF = α·d 가 linear → LP.
ISP-F: max -φ̃U·Σx̄ρ̃¹ - φ̃U·Σ(1-x̄)ρ̃³ - λU·Σx̄ρ⁰¹ - λU·Σ(1-x̄)ρ⁰³
  s.t. (DF-1)~(DF-ψ)  with α fixed

Build once, update α via set_normalized_coefficient, update x̄ via objective.
"""

using JuMP


"""
    build_true_dro_isp_follower(td::TrueDROData, x_bar, α_bar; optimizer)

Build ISP-F LP with α_bar fixed. Returns (model, vars).
vars[:DF6] stores constraint refs for α update.
"""
function build_true_dro_isp_follower(td::TrueDROData, x_bar::Vector{Float64},
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

    # Tight per-scenario TV ball bounds (§10.1)
    d_lo = [max(0.0, q[s] - 2 * ε̃) for s in 1:S]
    d_hi = [min(1.0, q[s] + 2 * ε̃) for s in 1:S]

    # ---- Variables ----
    @variable(model, d_lo[s] <= d[s=1:S] <= d_hi[s])
    @variable(model, 0 <= e[1:S] <= 2 * ε̃)
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

    # ---- Constraints ----

    # (DF-1): d_s - e_s ≤ q̂_s
    @constraint(model, DF1[s=1:S], d[s] - e[s] <= q[s])
    # (DF-2): d_s + e_s ≥ q̂_s
    @constraint(model, DF2[s=1:S], d[s] + e[s] >= q[s])
    # (DF-3): Σ_s e_s ≤ 2ε̃
    @constraint(model, DF3, sum(e[s] for s in 1:S) <= 2 * ε̃)
    # (DF-4): Σ_s d_s = 1
    @constraint(model, DF4, sum(d[s] for s in 1:S) == 1)

    # (DF-5): N_y ũ^s + N_ts σ̃^s = 0
    @constraint(model, DF5[j=1:m, s=1:S],
        sum(Ny[j, k] * u_tilde[k, s] for k in 1:K) + Nts[j] * σ_tilde[s] == 0)

    # (DF-6): -(ξ̄_k^s + ᾱ_k) d_s + ũ + ρ̃² - ρ̃³ ≤ 0   [α fixed → linear]
    @constraint(model, DF6[k=1:K, s=1:S],
        -(ξ[k, s] + α_bar[k]) * d[s]
        + u_tilde[k, s] + ρ_tilde_2[k, s] - ρ_tilde_3[k, s] <= 0)

    # (DF-7): v_k ξ̄_k^s d_s - ρ̃¹ - ρ̃² + ρ̃³ ≤ 0
    @constraint(model, DF7[k=1:K, s=1:S],
        v[k] * ξ[k, s] * d[s]
        - ρ_tilde_1[k, s] - ρ_tilde_2[k, s] + ρ_tilde_3[k, s] <= 0)

    # (DF-8): [N_yᵀ ω^s]_k - β_k^s ≤ 0
    @constraint(model, DF8[k=1:K, s=1:S],
        sum(Ny[j, k] * ω[j, s] for j in 1:m) - β[k, s] <= 0)

    # (DF-9): d_s + N_tsᵀ ω^s ≤ 0
    @constraint(model, DF9[s=1:S],
        d[s] + sum(Nts[j] * ω[j, s] for j in 1:m) <= 0)

    # (DF-h): Σ_s β_k^s ≤ δ
    @constraint(model, DFh[k=1:K], sum(β[k, s] for s in 1:S) <= δ)

    # (DF-λ): Σ_s σ̃^s ≥ Σ_{s,k} ξ̄ β + w δ + Σ_k ρ⁰² - Σ_k ρ⁰³
    @constraint(model, DFlam,
        sum(σ_tilde[s] for s in 1:S)
        >= sum(ξ[k, s] * β[k, s] for k in 1:K, s in 1:S)
           + w * δ
           + sum(ρ_psi0_2[k] for k in 1:K)
           - sum(ρ_psi0_3[k] for k in 1:K))

    # (DF-ψ): v_k Σ_s ξ̄ β + ρ⁰¹ + ρ⁰² ≥ ρ⁰³
    @constraint(model, DFpsi[k=1:K],
        v[k] * sum(ξ[k, s] * β[k, s] for s in 1:S)
        + ρ_psi0_1[k] + ρ_psi0_2[k] >= ρ_psi0_3[k])

    # ---- Objective ----
    @objective(model, Max,
        -φ̃U * sum(x_bar[k] * ρ_tilde_1[k, s] for k in 1:K, s in 1:S)
        - φ̃U * sum((1.0 - x_bar[k]) * ρ_tilde_3[k, s] for k in 1:K, s in 1:S)
        - λU * sum(x_bar[k] * ρ_psi0_1[k] for k in 1:K)
        - λU * sum((1.0 - x_bar[k]) * ρ_psi0_3[k] for k in 1:K))

    vars = Dict(
        :d => d, :e => e, :u_tilde => u_tilde, :σ_tilde => σ_tilde,
        :ω => ω, :β => β, :δ => δ,
        :ρ_tilde_1 => ρ_tilde_1, :ρ_tilde_2 => ρ_tilde_2, :ρ_tilde_3 => ρ_tilde_3,
        :ρ_psi0_1 => ρ_psi0_1, :ρ_psi0_2 => ρ_psi0_2, :ρ_psi0_3 => ρ_psi0_3,
        :DF6 => DF6,
    )
    return model, vars
end


"""
    update_isp_follower_alpha!(model, vars, td, α_new)

DF-6의 d[s] 계수를 -(ξ̄_k^s + α_new_k) 으로 갱신.
"""
function update_isp_follower_alpha!(model, vars, td::TrueDROData, α_new::Vector{Float64})
    S = td.S
    K = td.num_arcs
    ξ = td.xi_bar
    d = vars[:d]
    DF6 = vars[:DF6]

    for k in 1:K, s in 1:S
        set_normalized_coefficient(DF6[k, s], d[s], -(ξ[k, s] + α_new[k]))
    end
end


"""
    update_isp_follower_objective!(model, vars, td, x_bar_new)

목적함수의 x̄ 계수 갱신.
"""
function update_isp_follower_objective!(model, vars, td::TrueDROData, x_bar_new::Vector{Float64})
    S = td.S
    K = td.num_arcs
    φ̃U = td.phi_tilde_U
    λU = td.lambda_U

    for k in 1:K, s in 1:S
        set_objective_coefficient(model, vars[:ρ_tilde_1][k, s], -φ̃U * x_bar_new[k])
        set_objective_coefficient(model, vars[:ρ_tilde_3][k, s], -φ̃U * (1.0 - x_bar_new[k]))
    end
    for k in 1:K
        set_objective_coefficient(model, vars[:ρ_psi0_1][k], -λU * x_bar_new[k])
        set_objective_coefficient(model, vars[:ρ_psi0_3][k], -λU * (1.0 - x_bar_new[k]))
    end
end


"""
    solve_isp_follower!(model, vars, td)

Solve ISP-F LP. Returns Dict(:obj_val, :d_val, :rho_tilde_1/3_val, :rho_psi0_1/3_val).
"""
function solve_isp_follower!(model, vars, td::TrueDROData)
    optimize!(model)
    st = termination_status(model)
    if st != MOI.OPTIMAL
        error("ISP-F not optimal: $st")
    end

    S = td.S
    K = td.num_arcs

    return Dict(
        :obj_val => objective_value(model),
        :d_val => [value(vars[:d][s]) for s in 1:S],
        :rho_tilde_1_val => [value(vars[:ρ_tilde_1][k, s]) for k in 1:K, s in 1:S],
        :rho_tilde_3_val => [value(vars[:ρ_tilde_3][k, s]) for k in 1:K, s in 1:S],
        :rho_psi0_1_val => [value(vars[:ρ_psi0_1][k]) for k in 1:K],
        :rho_psi0_3_val => [value(vars[:ρ_psi0_3][k]) for k in 1:K],
    )
end
