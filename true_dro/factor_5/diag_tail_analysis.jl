"""
diag_tail_analysis.jl — Paired tail analysis: [3,6] vs [6,18]
  CVaR comparison, conditional win rates at tail quantiles
  Dirichlet β = 0.1, 0.3, 0.5, 1.0, 5.0
"""

using JuMP, HiGHS, Gurobi, Printf, Statistics, Random, Distributions, LinearAlgebra
using Plots

include("../../network_generator.jl")
using .NetworkGenerator

include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")

# ── network setup ──
net = generate_polska_network()
intd_arcs = fill(true, length(net.arcs))
net = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
    net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)

num_arcs = length(net.arcs) - 1
S = 20

caps, _ = generate_capacity_scenarios_factor_additive(length(net.arcs), S;
    interdictable_arcs=intd_arcs, seed=42, num_factors=5)
intd_idx = findall(intd_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)

# ── two solutions ──
x_nom = zeros(Float64, num_arcs); x_nom[3] = 1.0; x_nom[6] = 1.0
x_dro = zeros(Float64, num_arcs); x_dro[6] = 1.0; x_dro[18] = 1.0

q_hat = fill(1.0 / S, S)

# ── compute h* and flows (follower with q̂) ──
h_nom = solve_follower_weighted(net, x_nom, 1.0, w, caps, q_hat)
h_dro = solve_follower_weighted(net, x_dro, 1.0, w, caps, q_hat)
flows_nom = compute_maxflow_per_scenario(net, x_nom, h_nom, 1.0, caps)
flows_dro = compute_maxflow_per_scenario(net, x_dro, h_dro, 1.0, caps)

println("Per-scenario flows:")
@printf("  %-5s  %10s  %10s  %10s\n", "s", "[3,6]", "[6,18]", "Δ(dro-nom)")
for s in 1:S
    @printf("  s=%-3d  %10.4f  %10.4f  %+10.4f\n", s, flows_nom[s], flows_dro[s], flows_dro[s] - flows_nom[s])
end
@printf("  Mean:  %10.4f  %10.4f\n", mean(flows_nom), mean(flows_dro))
flush(stdout)

# ── OOS analysis for each β ──
β_list = [0.1, 0.3, 0.5, 1.0, 5.0]
M = 5000  # large sample for stable tail estimates

x_dict = Dict(:nom => x_nom, :dro => x_dro)

for β in β_list
    println("\n" * "=" ^ 70)
    @printf("β = %.1f,  M = %d\n", β, M)
    println("=" ^ 70)

    costs = oos_phase_b_generic(x_dict, net, caps, 1.0, w;
                                 β=β, M=M, seed=42, mode=:symmetric)

    c_nom = costs[:nom]
    c_dro = costs[:dro]
    gap = c_dro .- c_nom  # negative = DRO wins

    # ── basic stats ──
    @printf("  [3,6]  mean=%.4f  std=%.4f  p05=%.4f  p50=%.4f  p95=%.4f\n",
            mean(c_nom), std(c_nom), quantile(c_nom, 0.05), median(c_nom), quantile(c_nom, 0.95))
    @printf("  [6,18] mean=%.4f  std=%.4f  p05=%.4f  p50=%.4f  p95=%.4f\n",
            mean(c_dro), std(c_dro), quantile(c_dro, 0.05), median(c_dro), quantile(c_dro, 0.95))
    @printf("  Overall: DRO wins %d/%d (%.1f%%)\n", sum(gap .< 0), M, 100*mean(gap .< 0))
    flush(stdout)

    # ── CVaR at various α (upper tail = worst case for defender) ──
    println("\n  CVaR (upper tail, higher = worse):")
    for α in [0.05, 0.10, 0.20, 0.50]
        k = max(1, Int(ceil(M * (1 - α))))
        sorted_nom = sort(c_nom)
        sorted_dro = sort(c_dro)
        cvar_nom = mean(sorted_nom[k:end])
        cvar_dro = mean(sorted_dro[k:end])
        @printf("    α=%.2f: CVaR_nom=%.4f  CVaR_dro=%.4f  Δ=%+.4f  %s\n",
                α, cvar_nom, cvar_dro, cvar_dro - cvar_nom,
                cvar_dro < cvar_nom ? "DRO better" : "NOM better")
    end
    flush(stdout)

    # ── Conditional win rate: when [3,6] is at its worst ──
    println("\n  Conditional analysis (when [3,6] cost is high):")
    nom_order = sortperm(c_nom, rev=true)  # worst first
    for frac in [0.05, 0.10, 0.20]
        k = max(1, Int(ceil(M * frac)))
        idx = nom_order[1:k]
        wins = sum(c_dro[idx] .< c_nom[idx])
        mean_nom_cond = mean(c_nom[idx])
        mean_dro_cond = mean(c_dro[idx])
        @printf("    Top %.0f%% of [3,6] cost (n=%d): nom_mean=%.4f  dro_mean=%.4f  dro_wins=%d/%d (%.1f%%)\n",
                100*frac, k, mean_nom_cond, mean_dro_cond, wins, k, 100*wins/k)
    end
    flush(stdout)

    # ── Conditional win rate: when [6,18] is at its worst ──
    println("\n  Conditional analysis (when [6,18] cost is high):")
    dro_order = sortperm(c_dro, rev=true)
    for frac in [0.05, 0.10, 0.20]
        k = max(1, Int(ceil(M * frac)))
        idx = dro_order[1:k]
        wins = sum(c_dro[idx] .< c_nom[idx])
        mean_nom_cond = mean(c_nom[idx])
        mean_dro_cond = mean(c_dro[idx])
        @printf("    Top %.0f%% of [6,18] cost (n=%d): nom_mean=%.4f  dro_mean=%.4f  dro_wins=%d/%d (%.1f%%)\n",
                100*frac, k, mean_nom_cond, mean_dro_cond, wins, k, 100*wins/k)
    end
    flush(stdout)

    # ── Paired CVaR: sort by gap ──
    println("\n  Paired gap analysis (gap = dro - nom, negative = DRO better):")
    sorted_gap = sort(gap)
    for α in [0.05, 0.10, 0.20, 0.50]
        k_lo = Int(ceil(M * α))
        k_hi = max(1, Int(ceil(M * (1 - α))))
        best_dro = mean(sorted_gap[1:k_lo])       # DRO's best α tail
        worst_dro = mean(sorted_gap[k_hi:end])     # DRO's worst α tail
        @printf("    α=%.2f: best_gap=%.4f  worst_gap=%+.4f  mean_gap=%+.4f\n",
                α, best_dro, worst_dro, mean(gap))
    end
    flush(stdout)

    # ── Max cost comparison ──
    @printf("\n  Worst sample: max_nom=%.4f  max_dro=%.4f\n", maximum(c_nom), maximum(c_dro))
    @printf("  Best sample:  min_nom=%.4f  min_dro=%.4f\n", minimum(c_nom), minimum(c_dro))
    flush(stdout)
end

println("\n\nDone!")
