"""
run_oos_eval_polska_detailed.jl — Polska calibrated ε=0.148148 vs nominal, 상세 OOS 비교.
x_nom=[18], x_cal=[33]
"""

using JuMP, HiGHS, Gurobi, Printf, Dates, Statistics, Random, Distributions, LinearAlgebra

include("../../network_generator.jl")
using .NetworkGenerator

include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")

function oos_detailed(x_dict, network, capacities, v, w;
                      β=0.5, M=500, noise_scale=0.5, seed=42)
    K = size(capacities, 2)
    rng = MersenneTwister(seed)
    model_keys = collect(keys(x_dict))
    costs = Dict(k => Vector{Float64}(undef, M) for k in model_keys)
    for m in 1:M
        α = β .* (1.0 .+ noise_scale * randn(rng, K))
        α = max.(α, 0.01)
        p_center = α / sum(α)
        for k in model_keys
            h = solve_follower_weighted(network, x_dict[k], v, w, capacities, p_center)
            flows = compute_maxflow_per_scenario(network, x_dict[k], h, v, capacities)
            costs[k][m] = dot(p_center, flows)
        end
        if m % 100 == 0; @printf("  OOS outer %d/%d\n", m, M); end
    end
    return costs
end

function print_detailed(name, costs)
    @printf("  %-12s: mean=%.6f, std=%.6f, range=%.6f, [min,max]=[%.6f,%.6f]\n",
            name, mean(costs), std(costs), maximum(costs)-minimum(costs), minimum(costs), maximum(costs))
    for p in [90, 95, 99]
        @printf("    p%02d=%.6f\n", p, quantile(costs, p/100))
    end
end

function print_gap_detail(name_a, name_b, costs_a, costs_b)
    gap = costs_a .- costs_b
    M = length(gap)
    @printf("\n  --- %s vs %s (gap = %s - %s, <0 → %s wins) ---\n", name_a, name_b, name_a, name_b, name_a)
    @printf("  Mean:   %+.6f\n", mean(gap))
    @printf("  Std:    %.6f\n", std(gap))
    @printf("  Min:    %+.6f\n", minimum(gap))
    @printf("  Max:    %+.6f\n", maximum(gap))
    @printf("  Median: %+.6f\n", median(gap))
    for p in [1, 5, 10, 25, 75, 90, 95, 99]
        @printf("  p%02d:    %+.6f\n", p, quantile(gap, p/100))
    end
    wins_a = sum(gap .< 0)
    wins_b = sum(gap .> 0)
    @printf("  %s wins: %d/%d (%.1f%%)\n", name_a, wins_a, M, 100*wins_a/M)
    @printf("  %s wins: %d/%d (%.1f%%)\n", name_b, wins_b, M, 100*wins_b/M)
    sorted_gap = sort(gap)
    tail_low = sorted_gap[1:div(M,10)]
    tail_high = sorted_gap[end-div(M,10)+1:end]
    @printf("  Tail (worst 10%% for %s): mean=%+.6f\n", name_a, mean(tail_low))
    @printf("  Tail (worst 10%% for %s): mean=%+.6f\n", name_b, mean(tail_high))
end

# ============================================================
net = generate_polska_network()
num_arcs = length(net.arcs) - 1
all_intd = fill(true, length(net.arcs))
net_mod = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
    net.N, all_intd, net.arc_adjacency, net.node_arc_incidence)

caps, _ = generate_capacity_scenarios_factor_sparse(length(net_mod.arcs), 20;
    interdictable_arcs=all_intd, seed=42, num_factors=5)
intd_idx = findall(all_intd[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
v = 1.0

x_nom = zeros(Float64, num_arcs); x_nom[18] = 1.0
x_cal = zeros(Float64, num_arcs); x_cal[33] = 1.0

x_dict = Dict(:nominal => x_nom, :calibrated => x_cal)

println("=" ^ 80)
println("Polska Detailed OOS — all-intd, γ=1, λU=10.0")
println("  x_nom=[18], x_cal=[33] (ε=0.148)")
println("  M=500, noise_scale=0.5")
println("=" ^ 80)

for β in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    println("\n" * "=" ^ 60)
    @printf("β = %.1f\n", β)
    println("=" ^ 60)
    flush(stdout)

    costs = oos_detailed(x_dict, net_mod, caps, v, w;
                         β=β, M=500, noise_scale=0.5, seed=42)

    print_detailed("nominal", costs[:nominal])
    print_detailed("calibrated", costs[:calibrated])
    print_gap_detail("calibrated", "nominal", costs[:calibrated], costs[:nominal])
    flush(stdout)
end

println("\n\nDone! $(now())")
