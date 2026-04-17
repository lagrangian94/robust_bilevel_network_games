"""
run_oos_paired.jl — Bayraksan-Love style paired OOS evaluation.

기존 nested design (inner loop에서 q_true 평균 → E[q_true]=uniform 수렴) 대신,
각 replication에서 (q_true, q̃) 한 쌍을 독립 draw하고 세 모델을 paired 비교.

Protocol:
  for rep = 1, ..., R:
      q_true ~ Dir(β·1_S)      # 고정
      q̃ ~ Dir(β·1_S)           # follower belief (independent)
      for model ∈ {Nominal, Single-DRO, Two-layer DRO}:
          h* = solve_follower(x*_model, q̃)
          flows = maxflow_per_scenario(x*_model, h*)
          true_cost[model] = dot(q_true, flows)

Usage (Julia REPL):
    include("run_oos_paired.jl")
"""

using Printf
using Statistics
using Serialization
using Dates
using LinearAlgebra
using Random
using Distributions
using JuMP
using HiGHS

# ---- Load modules ----
if !@isdefined(NetworkGenerator)
    include("../network_generator.jl")
end
using .NetworkGenerator

# OOS evaluation helpers (build_maxflow_template, solve_deterministic_maxflow!)
include("../oos_evaluation.jl")
# solve_follower_weighted
include("oos_evaluate.jl")


# ============================================================
# 1. Network regeneration (run_oos_from_logs.jl과 동일)
# ============================================================

oos_network_configs = Dict(
    :grid_5x5    => Dict(:type => :grid, :m => 5, :n => 5),
    :sioux_falls => Dict(:type => :real_world, :generator => generate_sioux_falls_network),
    :nobel_us    => Dict(:type => :real_world, :generator => generate_nobel_us_network),
    :abilene     => Dict(:type => :real_world, :generator => generate_abilene_network),
    :polska      => Dict(:type => :real_world, :generator => generate_polska_network),
)

function compute_interdict_budget_oos(config_key::Symbol, num_interdictable::Int, γ_ratio::Float64)
    if config_key in (:sioux_falls, :abilene)
        return 2
    end
    return ceil(Int, γ_ratio * num_interdictable)
end

function regenerate_network_data(config_key::Symbol;
        S::Int=20, γ_ratio::Float64=0.10, ρ::Float64=0.2, v::Float64=1.0, seed::Int=42)
    config = oos_network_configs[config_key]
    if config[:type] == :grid
        network = generate_grid_network(config[:m], config[:n]; seed=seed)
    else
        network = config[:generator]()
    end

    num_arcs = length(network.arcs) - 1
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = compute_interdict_budget_oos(config_key, num_interdictable, γ_ratio)

    capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
        interdictable_arcs=network.interdictable_arcs, seed=seed)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(ρ * γ * c_bar; digits=4)

    return network, capacities, w, γ
end


# ============================================================
# 2. Paired OOS evaluation
# ============================================================

"""
    oos_paired(x_stars, network, capacities, β, v, w; R=1000, seed=42)

Bayraksan-Love style paired OOS.

Args:
- x_stars: Dict{Symbol, Vector{Float64}} — model_name => x* (e.g. :nominal, :single, :true_dro)
- Returns: Dict with per-model stats + paired comparison

Each rep: (q_true, q̃) 독립 draw → 모든 모델에 동일 (q_true, q̃) 적용.
"""
function oos_paired(x_stars::Dict{Symbol, Vector{Float64}},
                     network, capacities::Matrix{Float64},
                     β::Float64, v::Float64, w::Float64;
                     R::Int=1000, seed::Int=42)
    K = size(capacities, 2)  # num scenarios
    rng = MersenneTwister(seed)
    dir_dist = Dirichlet(K, β)

    model_names = collect(keys(x_stars))
    n_models = length(model_names)

    # Storage: R × n_models
    true_costs = Dict{Symbol, Vector{Float64}}()
    for m in model_names
        true_costs[m] = Vector{Float64}(undef, R)
    end
    winners = Vector{Symbol}(undef, R)

    for rep in 1:R
        q_true = rand(rng, dir_dist)
        q_tilde = rand(rng, dir_dist)

        best_cost = Inf
        best_model = model_names[1]

        for m in model_names
            x_star = x_stars[m]

            # Follower responds to q̃
            h_star = solve_follower_weighted(network, x_star, v, w, capacities, q_tilde)

            # Deterministic max-flow per scenario given (x*, h*)
            flows = compute_maxflow_per_scenario(network, x_star, h_star, v, capacities)

            # True cost = dot(q_true, flows)
            cost = dot(q_true, flows)
            true_costs[m][rep] = cost

            if cost < best_cost
                best_cost = cost
                best_model = m
            end
        end
        winners[rep] = best_model

        if rep % 200 == 0
            @printf("  rep %d/%d done\n", rep, R)
        end
    end

    # ---- Statistics ----
    stats = Dict{Symbol, Dict}()
    for m in model_names
        c = true_costs[m]
        stats[m] = Dict(
            :mean => mean(c),
            :std => std(c),
            :median => median(c),
            :p05 => quantile(c, 0.05),
            :p95 => quantile(c, 0.95),
            :min => minimum(c),
            :max => maximum(c),
        )
    end

    # Win rates
    win_counts = Dict{Symbol, Int}()
    for m in model_names
        win_counts[m] = count(==(m), winners)
    end

    # Pairwise: true_dro vs nominal, true_dro vs single, etc.
    pairwise = Dict{Tuple{Symbol,Symbol}, Dict}()
    for i in 1:n_models
        for j in (i+1):n_models
            m1, m2 = model_names[i], model_names[j]
            diffs = true_costs[m1] .- true_costs[m2]
            m1_wins = count(d -> d < -1e-10, diffs)
            m2_wins = count(d -> d > 1e-10, diffs)
            ties = R - m1_wins - m2_wins
            pairwise[(m1, m2)] = Dict(
                :mean_diff => mean(diffs),
                :std_diff => std(diffs),
                :m1_wins => m1_wins,
                :m2_wins => m2_wins,
                :ties => ties,
            )
        end
    end

    return Dict(
        :true_costs => true_costs,
        :stats => stats,
        :win_counts => win_counts,
        :winners => winners,
        :pairwise => pairwise,
        :R => R,
    )
end


# ============================================================
# 3. Load saved results & run
# ============================================================

jls_path = joinpath(@__DIR__, "oos_3variant_results.jls")
if !isfile(jls_path)
    error("Saved results not found: $jls_path\nRun run_oos_from_logs.jl first.")
end
raw_results = deserialize(jls_path)
@info "Loaded $(length(raw_results)) entries from $jls_path"

oos_beta_values = [0.1, 0.3, 0.5, 0.8]
oos_net_keys = [:grid_5x5, :polska, :abilene, :nobel_us, :sioux_falls]
R = 1000

println("=" ^ 80)
println("Paired OOS Evaluation (Bayraksan-Love protocol)")
println("  R=$R reps per (β, network)")
println("=" ^ 80)

paired_results = Dict{Tuple{Symbol, Float64}, Dict}()
summary_rows = []

for β in oos_beta_values
    @printf("\n### β = %.1f ###\n", β)

    for net_key in oos_net_keys
        key = (net_key, β)
        if !haskey(raw_results, key)
            @warn "Missing results" net_key β
            continue
        end
        entry = raw_results[key]

        # Collect x* for each available model
        x_stars = Dict{Symbol, Vector{Float64}}()
        z0_insample = Dict{Symbol, Float64}()

        if haskey(entry, :true_dro) && entry[:true_dro] !== nothing
            x_stars[:true_dro] = entry[:true_dro][:x]
            z0_insample[:true_dro] = entry[:true_dro][:Z0]
        end
        if haskey(entry, :nominal) && entry[:nominal] !== nothing
            x_stars[:nominal] = entry[:nominal][:x]
            z0_insample[:nominal] = entry[:nominal][:Z0]
        end
        if haskey(entry, :single_layer) && entry[:single_layer] !== nothing
            x_stars[:single_layer] = entry[:single_layer][:x]
            z0_insample[:single_layer] = entry[:single_layer][:Z0]
        end

        if length(x_stars) < 2
            @warn "Less than 2 models available" net_key β
            continue
        end

        # Regenerate network
        network, capacities, w, γ = regenerate_network_data(net_key; S=20)
        num_arcs = length(network.arcs) - 1
        @printf("\n  %s (|A|=%d, γ=%d, w=%.4f)\n", net_key, num_arcs, γ, w)

        # Run paired OOS
        result = oos_paired(x_stars, network, capacities, β, 1.0, w; R=R, seed=42)
        paired_results[key] = result

        # Print stats
        @printf("  %-14s  %10s  %10s  %10s  %10s  %8s\n",
                "Model", "Mean", "Std", "P05", "P95", "WinRate")
        @printf("  %s\n", "-" ^ 70)
        for m in [:nominal, :single_layer, :true_dro]
            if !haskey(result[:stats], m)
                continue
            end
            s = result[:stats][m]
            wr = result[:win_counts][m] / R * 100
            @printf("  %-14s  %10.4f  %10.4f  %10.4f  %10.4f  %7.1f%%\n",
                    m, s[:mean], s[:std], s[:p05], s[:p95], wr)

            # Summary row
            push!(summary_rows, (
                beta = β,
                network = string(net_key),
                variant = string(m),
                Z0_insample = get(z0_insample, m, NaN),
                oos_mean = s[:mean],
                oos_std = s[:std],
                oos_p05 = s[:p05],
                oos_p95 = s[:p95],
                win_rate = result[:win_counts][m] / R,
            ))
        end

        # Pairwise
        println("  Pairwise:")
        for ((m1, m2), pw) in result[:pairwise]
            @printf("    %s vs %s: mean_diff=%.4f, wins=%d/%d/%d (m1/tie/m2)\n",
                    m1, m2, pw[:mean_diff], pw[:m1_wins], pw[:ties], pw[:m2_wins])
        end
    end
end


# ============================================================
# 4. Save results
# ============================================================

# Save raw paired results
paired_jls_path = joinpath(@__DIR__, "oos_paired_results.jls")
serialize(paired_jls_path, paired_results)
println("\nPaired results saved → $paired_jls_path")

# CSV summary
csv_path = joinpath(@__DIR__, "oos_paired_summary.csv")
open(csv_path, "w") do io
    println(io, "beta,network,variant,Z0_insample,oos_mean,oos_std,oos_p05,oos_p95,win_rate")
    for r in summary_rows
        @printf(io, "%.1f,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f\n",
                r.beta, r.network, r.variant, r.Z0_insample,
                r.oos_mean, r.oos_std, r.oos_p05, r.oos_p95, r.win_rate)
    end
end
println("Summary CSV saved → $csv_path")

# Print final summary table
println("\n" * "=" ^ 110)
println("Paired OOS Summary (R=$R)")
println("=" ^ 110)
@printf("%-5s %-12s  %-14s  %10s  %10s  %10s  %10s  %8s\n",
        "β", "Network", "Variant", "Z₀", "OOS_mean", "OOS_p05", "OOS_p95", "WinRate")
println("-" ^ 110)
for r in summary_rows
    @printf("%-5.1f %-12s  %-14s  %10.4f  %10.4f  %10.4f  %10.4f  %7.1f%%\n",
            r.beta, r.network, r.variant, r.Z0_insample,
            r.oos_mean, r.oos_p05, r.oos_p95, r.win_rate * 100)
end
println("=" ^ 110)
println("\nDone! $(now())")
