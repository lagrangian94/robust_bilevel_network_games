"""
oos_evaluate.jl — Follower weighted SP + nested Dirichlet OOS evaluation.

OOS evaluation for True-DRO experiment:
  - solve_follower_weighted: q_weights 기반 follower 2-stage SP
  - compute_maxflow_per_scenario: scenario별 deterministic max-flow (template 재사용)
  - oos_evaluate: nested Dirichlet sampling (outer M × inner L)
"""

using JuMP
using HiGHS
using LinearAlgebra
using Statistics
using Printf
using Random
using Distributions


"""
    solve_follower_weighted(network, x_star, v, w, capacity_scenarios, q_weights) -> h_star

Follower의 weighted 2-stage SP:
    max Σ_k q_k · y_ts(k)
    s.t. N·y(k) = 0,  ∀k
         y_a(k) ≤ ξ_a(k)·(1 - v·x_a) + h_a,  ∀a,k
         y(k) ≥ 0, h ≥ 0, Σh ≤ w

solve_follower_response (oos_evaluation.jl)와 동일하되 uniform (1/K) 대신 q_weights 사용.

Args:
- network: GridNetworkData or RealWorldNetworkData
- x_star: Vector{Float64} — leader interdiction (num_arcs, dummy 제외)
- v: Float64 — interdiction effectiveness
- w: Float64 — recovery budget
- capacity_scenarios: Matrix{Float64} — (num_arcs_with_dummy × K)
- q_weights: Vector{Float64} — probability weights, length K

Returns:
- h_star: Vector{Float64} — optimal recovery (num_arcs, dummy 제외)
"""
function solve_follower_weighted(network, x_star::Vector{Float64}, v::Float64, w::Float64,
                                  capacity_scenarios::Matrix{Float64}, q_weights::Vector{Float64})
    num_arcs_total = length(network.arcs)       # dummy 포함
    num_arcs = num_arcs_total - 1               # dummy 제외
    K = size(capacity_scenarios, 2)
    N = network.N

    @assert length(q_weights) == K "q_weights length ($(length(q_weights))) != K ($K)"

    dummy_idx = findfirst(a -> a == ("t", "s"), network.arcs)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, h[1:num_arcs] >= 0)
    @variable(model, y[1:num_arcs_total, 1:K] >= 0)

    # Objective: max Σ_k q_k · y_ts(k)
    @objective(model, Max, sum(q_weights[k] * y[dummy_idx, k] for k in 1:K))

    # Recovery budget
    @constraint(model, sum(h) <= w)

    # Flow conservation
    for k in 1:K
        @constraint(model, N * y[:, k] .== 0.0)
    end

    # Capacity constraints
    for k in 1:K
        for a in 1:num_arcs
            cap = capacity_scenarios[a, k] * (1.0 - v * x_star[a]) + h[a]
            @constraint(model, y[a, k] <= cap)
        end
    end

    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        @warn "solve_follower_weighted status: $(termination_status(model))"
        return zeros(num_arcs)
    end

    return value.(h)
end


"""
    compute_maxflow_per_scenario(network, x_star, h_star, v, capacity_scenarios) -> Vector{Float64}

각 scenario k에 대해 deterministic max-flow. build_maxflow_template 재사용.

Returns: length-K vector of max-flow values.
"""
function compute_maxflow_per_scenario(network, x_star::Vector{Float64}, h_star::Vector{Float64},
                                       v::Float64, capacity_scenarios::Matrix{Float64})
    num_arcs = length(network.arcs) - 1
    K = size(capacity_scenarios, 2)

    mf_model, y_var, cap_con, dummy_idx, na = build_maxflow_template(network)

    flows = Vector{Float64}(undef, K)
    for k in 1:K
        xi_k = capacity_scenarios[1:num_arcs, k]
        flows[k] = solve_deterministic_maxflow!(mf_model, y_var, cap_con, xi_k,
                                                 x_star, h_star, v, dummy_idx, na)
    end

    return flows
end


"""
    oos_evaluate(x_star, network, capacity_scenarios, β, v, w;
                 M=100, L=1000, seed=nothing) -> Dict

Nested Dirichlet OOS evaluation.

Outer loop (j=1..M):
  q̃_j ~ Dir(β·1_K) → solve_follower_weighted(q̃_j) → h*_j
  flows_j = compute_maxflow_per_scenario(h*_j)

Inner loop (ℓ=1..L):
  q_true_ℓ ~ Dir(β·1_K) → Y_jℓ = dot(q_true_ℓ, flows_j)

Statistics:
  Y_bar_j = mean(Y_jℓ over ℓ)          — inner mean per outer sample
  Y_bar   = mean(Y_bar_j over j)        — grand mean
  Var_outer = var(Y_bar_j over j)       — variance from follower decision
  Var_inner = mean(var(Y_jℓ) over j)    — variance from true distribution

Returns Dict with :mean, :p95, :var_outer, :var_inner, :follower_share, :Y_bar, :evals
"""
function oos_evaluate(x_star::Vector{Float64}, network, capacity_scenarios::Matrix{Float64},
                       β::Float64, v::Float64, w::Float64;
                       M::Int=100, L::Int=1000, seed=nothing)
    K = size(capacity_scenarios, 2)

    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    dir_dist = Dirichlet(K, β)

    # Storage
    Y_bar_outer = Vector{Float64}(undef, M)      # inner mean per outer sample
    Y_all = Matrix{Float64}(undef, M, L)          # all evaluations
    var_inner_per_j = Vector{Float64}(undef, M)   # inner variance per outer sample

    for j in 1:M
        # Outer: follower's belief
        q_follower = rand(rng, dir_dist)
        h_star_j = solve_follower_weighted(network, x_star, v, w, capacity_scenarios, q_follower)

        # Compute max-flow for each scenario given h*_j
        flows_j = compute_maxflow_per_scenario(network, x_star, h_star_j, v, capacity_scenarios)

        # Inner: true distribution realizations
        for ℓ in 1:L
            q_true = rand(rng, dir_dist)
            Y_all[j, ℓ] = dot(q_true, flows_j)
        end

        Y_bar_outer[j] = mean(Y_all[j, :])
        var_inner_per_j[j] = var(Y_all[j, :])

        if j % 20 == 0
            @printf("  OOS outer %d/%d: Y_bar=%.4f\n", j, M, Y_bar_outer[j])
        end
    end

    Y_bar = mean(Y_bar_outer)
    var_outer = var(Y_bar_outer)
    var_inner = mean(var_inner_per_j)
    total_var = var_outer + var_inner
    follower_share = total_var > 0 ? var_outer / total_var : 0.0

    # p95: 5th percentile of outer means (worst-case for leader = low flow is bad for follower,
    # but leader wants to MINIMIZE flow → high flow is bad for leader → use 95th percentile)
    p95 = quantile(Y_bar_outer, 0.95)

    @printf("  OOS result: mean=%.4f, p95=%.4f, Var_outer=%.2e, Var_inner=%.2e, follower_share=%.4f\n",
            Y_bar, p95, var_outer, var_inner, follower_share)

    return Dict(
        :mean => Y_bar,
        :p95 => p95,
        :var_outer => var_outer,
        :var_inner => var_inner,
        :follower_share => follower_share,
        :Y_bar => Y_bar_outer,
        :evals => Y_all,
    )
end


"""
    oos_evaluate_scenario_L(x_star, network, capacity_scenarios, β_L, β_H, v, w;
                             M=100, L=1000, seed=nothing) -> Dict

Scenario L (Leader Advantage): P* ≈ q̂, P̃ 부정확.
  Outer: q̃ ~ Dir(β_L·1_K) → follower 부정확
  Inner: q_true ~ Dir(β_H·1_K) → truth ≈ q̂

Variance decomposition 활성: 양쪽 loop 모두 stochastic.
"""
function oos_evaluate_scenario_L(x_star::Vector{Float64}, network, capacity_scenarios::Matrix{Float64},
                                   β_L::Float64, β_H::Float64, v::Float64, w::Float64;
                                   M::Int=100, L::Int=1000, seed=nothing)
    K = size(capacity_scenarios, 2)

    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    dir_follower = Dirichlet(K, β_L)     # q̃ ~ Dir(β_L·1) — follower 부정확
    dir_true = Dirichlet(K, β_H)          # q_true ~ Dir(β_H·1) — truth ≈ q̂

    Y_bar_outer = Vector{Float64}(undef, M)
    Y_all = Matrix{Float64}(undef, M, L)
    var_inner_per_j = Vector{Float64}(undef, M)

    for j in 1:M
        q_follower = rand(rng, dir_follower)
        h_star_j = solve_follower_weighted(network, x_star, v, w, capacity_scenarios, q_follower)
        flows_j = compute_maxflow_per_scenario(network, x_star, h_star_j, v, capacity_scenarios)

        for ℓ in 1:L
            q_true = rand(rng, dir_true)
            Y_all[j, ℓ] = dot(q_true, flows_j)
        end

        Y_bar_outer[j] = mean(Y_all[j, :])
        var_inner_per_j[j] = var(Y_all[j, :])

        if j % 20 == 0
            @printf("  OOS-L outer %d/%d: Y_bar=%.4f\n", j, M, Y_bar_outer[j])
        end
    end

    Y_bar = mean(Y_bar_outer)
    var_outer = var(Y_bar_outer)
    var_inner = mean(var_inner_per_j)
    total_var = var_outer + var_inner
    follower_share = total_var > 0 ? var_outer / total_var : 0.0
    p95 = quantile(Y_bar_outer, 0.95)

    @printf("  OOS-L result: mean=%.4f, p95=%.4f, Var_outer=%.2e, Var_inner=%.2e, follower_share=%.4f\n",
            Y_bar, p95, var_outer, var_inner, follower_share)

    return Dict(
        :mean => Y_bar, :p95 => p95,
        :var_outer => var_outer, :var_inner => var_inner,
        :follower_share => follower_share,
        :Y_bar => Y_bar_outer, :evals => Y_all,
    )
end


"""
    oos_evaluate_scenario_F(x_star, network, capacity_scenarios, β_L, κ, v, w;
                             M=100, seed=nothing) -> Dict

Scenario F (Follower Advantage): q̃ ≈ q_true, q_true ≠ q̂.
  Outer: q_true ~ Dir(β_L·1_K), q̃ ~ Dir(κ·q_true) → q̃ ≈ q_true
  Inner: deterministic (q_true 고정) → L=1, nature effect = 0

Variance decomposition: nature effect = 0 by design (follower가 truth를 알면
nature uncertainty가 follower의 recovery를 통해 이미 흡수됨).
"""
function oos_evaluate_scenario_F(x_star::Vector{Float64}, network, capacity_scenarios::Matrix{Float64},
                                   β_L::Float64, κ::Float64, v::Float64, w::Float64;
                                   M::Int=100, seed=nothing)
    K = size(capacity_scenarios, 2)

    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    dir_true = Dirichlet(K, β_L)

    Y_outer = Vector{Float64}(undef, M)

    for j in 1:M
        q_true = rand(rng, dir_true)
        q_follower = rand(rng, Dirichlet(κ * q_true))

        h_star_j = solve_follower_weighted(network, x_star, v, w, capacity_scenarios, q_follower)
        flows_j = compute_maxflow_per_scenario(network, x_star, h_star_j, v, capacity_scenarios)

        # Inner loop deterministic: eval = dot(q_true, flows)
        Y_outer[j] = dot(q_true, flows_j)

        if j % 20 == 0
            @printf("  OOS-F outer %d/%d: Y=%.4f\n", j, M, Y_outer[j])
        end
    end

    Y_bar = mean(Y_outer)
    var_outer = var(Y_outer)
    p95 = quantile(Y_outer, 0.95)

    @printf("  OOS-F result: mean=%.4f, p95=%.4f, Var=%.2e\n", Y_bar, p95, var_outer)

    return Dict(
        :mean => Y_bar, :p95 => p95,
        :var_outer => var_outer, :var_inner => 0.0,
        :follower_share => 1.0,    # nature effect = 0 by design
        :Y_bar => Y_outer,
    )
end
