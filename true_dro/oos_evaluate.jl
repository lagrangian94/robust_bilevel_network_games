"""
oos_evaluate.jl — Follower weighted SP + nested Dirichlet OOS evaluation.

OOS evaluation for True-DRO experiment:
  - solve_follower_weighted: q_weights 기반 follower 2-stage SP
  - compute_maxflow_per_scenario: scenario별 deterministic max-flow (template 재사용)
  - oos_evaluate: Phase A — symmetric Dirichlet (outer M × inner L), tail metrics
  - oos_evaluate_phase_b: Phase B — asymmetric Dirichlet, OOS mean gap
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
                 M=100, L=100, seed=nothing) -> Dict

Phase A: Symmetric Dirichlet OOS evaluation with tail metrics.
(Ben-Tal et al. 2013, Section 6.4 protocol)

Outer loop (j=1..M):
  q̃_j ~ Dir(β·1_K) → solve_follower_weighted(q̃_j) → h*_j
  flows_j = compute_maxflow_per_scenario(h*_j)

Inner loop (ℓ=1..L):
  q_true_ℓ ~ Dir(β·1_K) → Y_jℓ = dot(q_true_ℓ, flows_j)

Statistics (flat M×L evals):
  mean, p5, p95, min, max — tail metrics for insurance effect
  Variance decomposition: Var_outer (follower belief), Var_inner (nature)

Returns Dict with :mean, :p5, :p95, :min, :max,
                   :var_outer, :var_inner, :follower_share,
                   :Y_bar (outer means), :evals (M×L flat), :h_all, :flows_all
"""
function oos_evaluate(x_star::Vector{Float64}, network, capacity_scenarios::Matrix{Float64},
                       β::Float64, v::Float64, w::Float64;
                       M::Int=100, L::Int=100, seed=nothing)
    K = size(capacity_scenarios, 2)

    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    dir_dist = Dirichlet(K, β)

    # Storage
    Y_bar_outer = Vector{Float64}(undef, M)      # inner mean per outer sample
    Y_all = Matrix{Float64}(undef, M, L)          # all evaluations
    var_inner_per_j = Vector{Float64}(undef, M)   # inner variance per outer sample
    p5_per_j  = Vector{Float64}(undef, M)         # inner p5 per outer sample
    p95_per_j = Vector{Float64}(undef, M)         # inner p95 per outer sample
    h_all = Vector{Vector{Float64}}(undef, M)     # h*_j per outer sample
    flows_all = Matrix{Float64}(undef, M, K)      # flows_j per outer sample
    q_follower_all = Matrix{Float64}(undef, M, K) # follower belief per outer sample
    q_true_all = Array{Float64, 3}(undef, M, L, K) # true distribution per (outer, inner)

    for j in 1:M
        # Outer: follower's belief
        q_follower = rand(rng, dir_dist)
        h_star_j = solve_follower_weighted(network, x_star, v, w, capacity_scenarios, q_follower)

        # Compute max-flow for each scenario given h*_j
        flows_j = compute_maxflow_per_scenario(network, x_star, h_star_j, v, capacity_scenarios)

        h_all[j] = copy(h_star_j)
        flows_all[j, :] .= flows_j
        q_follower_all[j, :] .= q_follower

        # Inner: true distribution realizations
        for ℓ in 1:L
            q_true = rand(rng, dir_dist)
            q_true_all[j, ℓ, :] .= q_true
            Y_all[j, ℓ] = dot(q_true, flows_j)
        end

        Y_bar_outer[j] = mean(Y_all[j, :])
        var_inner_per_j[j] = var(Y_all[j, :])
        p5_per_j[j]  = quantile(Y_all[j, :], 0.05)
        p95_per_j[j] = quantile(Y_all[j, :], 0.95)

        if j % 20 == 0
            @printf("  OOS-A outer %d/%d: Y_bar=%.4f\n", j, M, Y_bar_outer[j])
        end
    end

    # Block-level statistics with CI (M independent blocks)
    _ci(v) = (m = mean(v); se = std(v)/sqrt(M); (m, m - 1.96*se, m + 1.96*se))

    Y_bar, ci_lo, ci_hi = _ci(Y_bar_outer)
    p5, p5_ci_lo, p5_ci_hi = _ci(p5_per_j)
    p95, p95_ci_lo, p95_ci_hi = _ci(p95_per_j)
    y_min = minimum(Y_bar_outer)
    y_max = maximum(Y_bar_outer)

    # Variance decomposition
    var_outer = var(Y_bar_outer)
    var_inner = mean(var_inner_per_j)
    total_var = var_outer + var_inner
    follower_share = total_var > 0 ? var_outer / total_var : 0.0

    @printf("  OOS-A result: mean=%.4f [%.4f,%.4f], p5=%.4f [%.4f,%.4f], p95=%.4f [%.4f,%.4f], f_share=%.4f\n",
            Y_bar, ci_lo, ci_hi, p5, p5_ci_lo, p5_ci_hi, p95, p95_ci_lo, p95_ci_hi, follower_share)

    return Dict(
        :mean => Y_bar,
        :ci_lo => ci_lo,
        :ci_hi => ci_hi,
        :p5 => p5,
        :p5_ci_lo => p5_ci_lo,
        :p5_ci_hi => p5_ci_hi,
        :p95 => p95,
        :p95_ci_lo => p95_ci_lo,
        :p95_ci_hi => p95_ci_hi,
        :min => y_min,
        :max => y_max,
        :var_outer => var_outer,
        :var_inner => var_inner,
        :follower_share => follower_share,
        :Y_bar => Y_bar_outer,
        :p5_per_j => p5_per_j,
        :p95_per_j => p95_per_j,
        :evals => Y_all,
        :h_all => h_all,
        :flows_all => flows_all,
        :q_follower_all => q_follower_all,
        :q_true_all => q_true_all,
    )
end


"""
    compute_win_rate(evals_a, evals_b) -> Float64

evals_a가 evals_b보다 작은 비율 (minimization 문제에서 a가 이기는 비율).
evals_a, evals_b는 같은 길이의 Vector.
"""
function compute_win_rate(evals_a::AbstractVector{Float64}, evals_b::AbstractVector{Float64})
    @assert length(evals_a) == length(evals_b)
    return mean(evals_a .< evals_b)
end


"""
    oos_evaluate_phase_b(x_stars, network, capacity_scenarios, β, v, w;
                          M=100, R=100, noise_scale=0.5, seed=nothing,
                          use_shortcut=false) -> Dict

Phase B: Asymmetric Dirichlet OOS — E[q_true] ≠ uniform.
(Section 7.3 of experiment design v4)

x_stars: Dict{Symbol, Vector{Float64}} with keys :nominal, :single, :two_layer

Outer loop (m=1..M): noise realization 고정
  α = β .* (1.0 .+ noise_scale * randn(K))
  α = max.(α, 0.01)
  p_center = α / sum(α)   # 이 세상의 true mean ≠ uniform

Inner loop (r=1..R): Dir(α) sampling
  p_true ~ Dir(α), q_tilde ~ Dir(α)
  for each model: h* = follower(x*, q_tilde), eval = dot(p_true, flows)

use_shortcut=true: f_share≈0일 때 inner loop 생략, p_center로 직접 계산 (Section 7.4).

Returns Dict with :gap_two_vs_nom, :gap_two_vs_single,
                   :gap_mean, :gap_p5, :gap_p95, :dro_wins, :costs
"""
function oos_evaluate_phase_b(x_stars::Dict{Symbol, Vector{Float64}},
                                network, capacity_scenarios::Matrix{Float64},
                                β::Float64, v::Float64, w::Float64;
                                M::Int=100, R::Int=100, noise_scale::Float64=0.5,
                                seed=nothing, use_shortcut::Bool=false)
    K = size(capacity_scenarios, 2)
    models = [:nominal, :single, :two_layer]

    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    # Storage: model × M
    if use_shortcut
        # Shortcut: p_center로 직접 계산, inner loop 불필요
        costs = Dict(m => Vector{Float64}(undef, M) for m in models)

        for outer_m in 1:M
            # Generate asymmetric α (noise 한 번 고정)
            α = β .* (1.0 .+ noise_scale * randn(rng, K))
            α = max.(α, 0.01)
            p_center = α / sum(α)

            for m in models
                h_star = solve_follower_weighted(network, x_stars[m], v, w,
                                                  capacity_scenarios, p_center)
                flows = compute_maxflow_per_scenario(network, x_stars[m], h_star, v,
                                                       capacity_scenarios)
                costs[m][outer_m] = dot(p_center, flows)
            end

            if outer_m % 20 == 0
                @printf("  OOS-B(shortcut) outer %d/%d\n", outer_m, M)
            end
        end
    else
        # Full inner loop
        costs = Dict(m => Matrix{Float64}(undef, M, R) for m in models)

        for outer_m in 1:M
            # Generate asymmetric α (noise 한 번 고정)
            α = β .* (1.0 .+ noise_scale * randn(rng, K))
            α = max.(α, 0.01)
            dir_α = Dirichlet(α)

            for r in 1:R
                p_true  = rand(rng, dir_α)
                q_tilde = rand(rng, dir_α)

                for m in models
                    h_star = solve_follower_weighted(network, x_stars[m], v, w,
                                                      capacity_scenarios, q_tilde)
                    flows = compute_maxflow_per_scenario(network, x_stars[m], h_star, v,
                                                           capacity_scenarios)
                    costs[m][outer_m, r] = dot(p_true, flows)
                end
            end

            if outer_m % 20 == 0
                @printf("  OOS-B outer %d/%d\n", outer_m, M)
            end
        end

        # Reduce inner loop to mean per outer
        costs = Dict(m => vec(mean(costs[m], dims=2)) for m in models)
    end

    # Gap: two_layer - nominal (< 0 이면 DRO 승, minimization)
    gap_two_vs_nom    = costs[:two_layer] .- costs[:nominal]
    gap_two_vs_single = costs[:two_layer] .- costs[:single]

    results = Dict(
        :gap_two_vs_nom => gap_two_vs_nom,
        :gap_two_vs_single => gap_two_vs_single,
        # Gap stats (two-layer vs nominal)
        :gap_mean => mean(gap_two_vs_nom),
        :gap_p5   => quantile(gap_two_vs_nom, 0.05),
        :gap_p95  => quantile(gap_two_vs_nom, 0.95),
        :dro_wins => mean(gap_two_vs_nom .< 0),
        # Gap stats (two-layer vs single)
        :gap_vs_single_mean => mean(gap_two_vs_single),
        :gap_vs_single_wins => mean(gap_two_vs_single .< 0),
        # Per-model costs
        :costs => Dict(m => costs[m] for m in models),
    )

    @printf("  OOS-B result: gap_mean=%.4f, gap[p5,p95]=[%.4f,%.4f], DRO_wins=%.1f%%\n",
            results[:gap_mean], results[:gap_p5], results[:gap_p95], results[:dro_wins]*100)

    return results
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


"""
    oos_phase_b_generic(x_dict, network, capacities, v, w;
                        β, M, noise_scale, seed, mode) -> Dict{Symbol, Vector{Float64}}

Generic Phase B OOS. 임의의 x_dict (Dict{Symbol, Vector}) 지원.

mode:
  :same_alpha — q_tilde ~ Dir(α), p_true ~ Dir(α).  같은 α, 독립 샘플.
  :diff_alpha — α_tilde 별도 생성, q_tilde ~ Dir(α_tilde), p_true ~ Dir(α).
  :symmetric — q_tilde ~ Dir(β·1_K), p_true ~ Dir(β·1_K).  symmetric Dirichlet, 독립 샘플. (Phase A 방식)

각 outer sample m:
  1. α = β*(1 + noise_scale*N(0,1))  → p_true ~ Dir(α)
  2. mode에 따라 q_tilde 생성
  3. h* = follower(x, q_tilde), flows = maxflow(x, h*)
  4. cost = dot(p_true, flows)

Returns: Dict(model_key => Vector{Float64} of length M)
"""
function oos_phase_b_generic(x_dict::Dict{Symbol, Vector{Float64}},
                              network, capacities::Matrix{Float64},
                              v::Float64, w::Float64;
                              β::Float64=0.5, M::Int=500,
                              noise_scale::Float64=0.5, seed::Int=42,
                              mode::Symbol=:same_alpha)
    K = size(capacities, 2)
    rng = MersenneTwister(seed)
    model_keys = collect(keys(x_dict))

    costs = Dict(k => Vector{Float64}(undef, M) for k in model_keys)

    dir_sym = Dirichlet(K, β)  # symmetric Dir(β·1_K) for :symmetric mode

    for m in 1:M
        if mode == :symmetric
            # Symmetric Dirichlet — no perturbation
            p_true = rand(rng, dir_sym)
        else
            # Asymmetric — perturbed α
            α = β .* (1.0 .+ noise_scale * randn(rng, K))
            α = max.(α, 0.01)
            dir_α = Dirichlet(α)
            p_true = rand(rng, dir_α)
        end

        # Follower's belief
        if mode == :same_alpha
            # (1) Same α, independent sample
            q_tilde = rand(rng, dir_α)
        elseif mode == :diff_alpha
            # (2) Separate α_tilde, then sample
            α_tilde = β .* (1.0 .+ noise_scale * randn(rng, K))
            α_tilde = max.(α_tilde, 0.01)
            q_tilde = rand(rng, Dirichlet(α_tilde))
        elseif mode == :symmetric
            # (3) Symmetric Dirichlet — Phase A style, no perturbation
            q_tilde = rand(rng, dir_sym)
        else
            error("Unknown mode: $mode. Use :same_alpha, :diff_alpha, or :symmetric")
        end

        for k in model_keys
            h = solve_follower_weighted(network, x_dict[k], v, w, capacities, q_tilde)
            flows = compute_maxflow_per_scenario(network, x_dict[k], h, v, capacities)
            costs[k][m] = dot(p_true, flows)
        end

        if m % 100 == 0
            @printf("  OOS-B(%s) outer %d/%d\n", mode, m, M)
        end
    end

    return costs
end
