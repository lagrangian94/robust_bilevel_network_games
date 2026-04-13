"""
true_dro_build_isp_leader.jl — ISP-L LP for mini-Benders (true_dro_v5.md §7.3).

α를 파라미터로 고정하면 bilinear term ζL = α·a 가 linear → LP.
ISP-L: max Σσ̂ - φ̂U·Σx̄ρ̂¹ - φ̂U·Σ(1-x̄)ρ̂³
  s.t. (DL-1)~(DL-7)  with α fixed

Build once, update α via set_normalized_coefficient, update x̄ via objective.
"""

using JuMP


"""
    build_true_dro_isp_leader(td::TrueDROData, x_bar, α_bar; optimizer)

Build ISP-L LP with α_bar fixed. Returns (model, vars).
vars[:DL2] stores constraint refs for α update.
"""
function build_true_dro_isp_leader(td::TrueDROData, x_bar::Vector{Float64},
                                    α_bar::Vector{Float64}; optimizer)
    S = td.S
    K = td.num_arcs
    m = td.nv1
    Ny = td.Ny
    Nts = td.Nts
    q = td.q_hat
    ε̂ = td.eps_hat
    ξ = td.xi_bar
    v = td.v
    φ̂U = td.phi_hat_U

    model = Model(optimizer)
    set_silent(model)

    # Tight per-scenario TV ball bounds (§10.1)
    a_lo = [max(0.0, q[s] - 2 * ε̂) for s in 1:S]
    a_hi = [min(1.0, q[s] + 2 * ε̂) for s in 1:S]

    # ---- Variables ----
    @variable(model, σ_hat[1:S] >= 0)
    @variable(model, u_hat[1:K, 1:S] >= 0)
    @variable(model, a_lo[s] <= a[s=1:S] <= a_hi[s])
    @variable(model, 0 <= b[1:S] <= 2 * ε̂)
    @variable(model, ρ_hat_1[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_2[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_3[1:K, 1:S] >= 0)

    # ---- Constraints ----

    # (DL-1): N_y û^s + N_ts σ̂^s = 0
    @constraint(model, DL1[j=1:m, s=1:S],
        sum(Ny[j, k] * u_hat[k, s] for k in 1:K) + Nts[j] * σ_hat[s] == 0)

    # (DL-2): -(ξ̄_k^s + ᾱ_k) a_s + û + ρ̂² - ρ̂³ ≤ 0   [α fixed → linear]
    @constraint(model, DL2[k=1:K, s=1:S],
        -(ξ[k, s] + α_bar[k]) * a[s]
        + u_hat[k, s] + ρ_hat_2[k, s] - ρ_hat_3[k, s] <= 0)

    # (DL-3): v_k ξ̄_k^s a_s - ρ̂¹ - ρ̂² + ρ̂³ ≤ 0
    @constraint(model, DL3[k=1:K, s=1:S],
        v[k] * ξ[k, s] * a[s]
        - ρ_hat_1[k, s] - ρ_hat_2[k, s] + ρ_hat_3[k, s] <= 0)

    # (DL-4)~(DL-7)
    @constraint(model, DL4[s=1:S], a[s] - b[s] <= q[s])
    @constraint(model, DL5[s=1:S], a[s] + b[s] >= q[s])
    @constraint(model, DL6, sum(b[s] for s in 1:S) <= 2 * ε̂)
    @constraint(model, DL7, sum(a[s] for s in 1:S) == 1)

    # ---- Objective ----
    @objective(model, Max,
        sum(σ_hat[s] for s in 1:S)
        - φ̂U * sum(x_bar[k] * ρ_hat_1[k, s] for k in 1:K, s in 1:S)
        - φ̂U * sum((1.0 - x_bar[k]) * ρ_hat_3[k, s] for k in 1:K, s in 1:S))

    vars = Dict(
        :σ_hat => σ_hat, :u_hat => u_hat, :a => a, :b => b,
        :ρ_hat_1 => ρ_hat_1, :ρ_hat_2 => ρ_hat_2, :ρ_hat_3 => ρ_hat_3,
        :DL2 => DL2,
    )
    return model, vars
end


"""
    update_isp_leader_alpha!(model, vars, td, α_new)

DL-2의 a[s] 계수를 -(ξ̄_k^s + α_new_k) 으로 갱신.
"""
function update_isp_leader_alpha!(model, vars, td::TrueDROData, α_new::Vector{Float64})
    S = td.S
    K = td.num_arcs
    ξ = td.xi_bar
    a = vars[:a]
    DL2 = vars[:DL2]

    for k in 1:K, s in 1:S
        set_normalized_coefficient(DL2[k, s], a[s], -(ξ[k, s] + α_new[k]))
    end
end


"""
    update_isp_leader_objective!(model, vars, td, x_bar_new)

목적함수의 x̄ 계수 갱신: ρ̂¹ coef = -φ̂U·x̄_k, ρ̂³ coef = -φ̂U·(1-x̄_k).
"""
function update_isp_leader_objective!(model, vars, td::TrueDROData, x_bar_new::Vector{Float64})
    S = td.S
    K = td.num_arcs
    φ̂U = td.phi_hat_U

    for k in 1:K, s in 1:S
        set_objective_coefficient(model, vars[:ρ_hat_1][k, s], -φ̂U * x_bar_new[k])
        set_objective_coefficient(model, vars[:ρ_hat_3][k, s], -φ̂U * (1.0 - x_bar_new[k]))
    end
end


"""
    solve_isp_leader!(model, vars, td)

Solve ISP-L LP. Returns Dict(:obj_val, :a_val, :rho_hat_1_val, :rho_hat_3_val).
"""
function solve_isp_leader!(model, vars, td::TrueDROData)
    optimize!(model)
    st = termination_status(model)
    if st != MOI.OPTIMAL
        error("ISP-L not optimal: $st")
    end

    S = td.S
    K = td.num_arcs

    return Dict(
        :obj_val => objective_value(model),
        :a_val => [value(vars[:a][s]) for s in 1:S],
        :rho_hat_1_val => [value(vars[:ρ_hat_1][k, s]) for k in 1:K, s in 1:S],
        :rho_hat_3_val => [value(vars[:ρ_hat_3][k, s]) for k in 1:K, s in 1:S],
    )
end
