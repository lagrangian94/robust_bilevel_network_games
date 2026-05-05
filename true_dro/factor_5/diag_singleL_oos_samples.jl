"""
diag_singleL_oos_samples.jl — Sample별 Nominal vs Single-L (ε=0.1, 0.3) OOS 비교
  q̃ = q̂ (sample별 고정), q_true ~ Dir(β)
  Polska, factor_additive, γ=2, S=20

  h*는 q̂로 한 번만 풀고, q_true만 M번 샘플링.

  x* solutions (from Benders logs):
    sample1: NOM=[6,34], ε01=[6,34], ε03=[6,18]  → NOM=ε01, 비교: [6,34] vs [6,18]
    sample3: NOM=[6,34], ε01=[6,34], ε03=[6,18]  → NOM=ε01, 비교: [6,34] vs [6,18]
    sample8: NOM=[3,6],  ε01=[6,18], ε03=[6,18]  → ε01=ε03, 비교: [3,6] vs [6,18]
"""

using JuMP, HiGHS, Printf, Statistics, Random, Distributions, LinearAlgebra

include("../../network_generator.jl")
using .NetworkGenerator

include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")
include("qhat_samples.jl")

# ── network setup ──
net = generate_polska_network()
num_arcs = length(net.arcs) - 1
all_intd = fill(true, length(net.arcs))
net = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
    net.N, all_intd, net.arc_adjacency, net.node_arc_incidence)

S = 20
caps, _ = generate_capacity_scenarios_factor_additive(length(net.arcs), S;
    interdictable_arcs=all_intd, seed=42, num_factors=5)
intd_idx = findall(all_intd[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
v_eff = 1.0

@printf("polska factor_additive: arcs=%d, w=%.4f\n\n", num_arcs, w)

function make_x(arcs, n)
    x = zeros(n); for a in arcs; x[a] = 1.0; end; x
end

# ── x* configurations per sample (unique pairs only) ──
sample_configs = Dict(
    "sample1" => [
        ("NOM/ε01 [6,34]", [6, 34]),
        ("ε03 [6,18]",     [6, 18]),
    ],
    "sample3" => [
        ("NOM/ε01 [6,34]", [6, 34]),
        ("ε03 [6,18]",     [6, 18]),
    ],
    "sample8" => [
        ("NOM [3,6]",      [3, 6]),
        ("ε01/ε03 [6,18]", [6, 18]),
    ],
)

M = 5000
βs = [0.1, 0.3, 0.5, 1.0, 5.0]

for sample_name in ["sample1", "sample3", "sample8"]
    configs = sample_configs[sample_name]
    q_hat = QHAT_SAMPLES[sample_name]

    println("\n" * "=" ^ 70)
    @printf("SAMPLE: %s  (TV from uniform = %.4f)\n", sample_name,
            0.5 * sum(abs.(q_hat .- 1.0/S)))
    println("=" ^ 70)

    # ── Phase 1: h* with q̃ = q̂ (한 번만) ──
    println("\nPhase 1: Follower recovery h* with q̃ = q̂")
    println("-" ^ 50)

    flows_fixed = Dict{String, Vector{Float64}}()

    for (label, arcs) in configs
        x_vec = make_x(arcs, num_arcs)
        h_star = solve_follower_weighted(net, x_vec, v_eff, w, caps, q_hat)
        flows = compute_maxflow_per_scenario(net, x_vec, h_star, v_eff, caps)
        flows_fixed[label] = flows

        @printf("  %s:\n", label)
        @printf("    E_{q̂}[maxflow] = %.6f\n", dot(q_hat, flows))
        h_nz = findall(h_star .> 1e-6)
        @printf("    h* arcs: ")
        for k in h_nz
            @printf("%d(%s→%s, %.4f) ", k, net.arcs[k][1], net.arcs[k][2], h_star[k])
        end
        println()
        @printf("    Per-scenario: min=%.4f(s=%d), max=%.4f(s=%d), std=%.4f\n",
                minimum(flows), argmin(flows), maximum(flows), argmax(flows), std(flows))
    end

    # Per-scenario flow table (β-independent)
    labels = [l for (l, _) in configs]
    la, lb = labels[1], labels[2]
    fa, fb = flows_fixed[la], flows_fixed[lb]
    println("\n  Per-scenario maxflow comparison (h* fixed at q̂):")
    @printf("    s   %-16s  %-16s  Δ(B−A)\n", la, lb)
    for s in 1:S
        marker = fb[s] < fa[s] ? "←B better" : ""
        @printf("    %2d  %.4f           %.4f           %+.4f  %s\n",
                s, fa[s], fb[s], fb[s]-fa[s], marker)
    end

    # ── Phase 2: OOS with q_true ~ Dir(β), h* 고정 ──
    println("\nPhase 2: OOS evaluation (q̃=q̂ fixed, q_true ~ Dir(β))")
    println("-" ^ 50)

    rng = MersenneTwister(42)  # reset per sample

    for β in βs
        @printf("\n--- β = %.1f, M = %d ---\n", β, M)
        dir = Dirichlet(S, β)

        costs_a = Vector{Float64}(undef, M)
        costs_b = Vector{Float64}(undef, M)
        tv_dists = Vector{Float64}(undef, M)

        for m in 1:M
            q_true = rand(rng, dir)
            tv_dists[m] = 0.5 * sum(abs.(q_true .- q_hat))
            costs_a[m] = dot(q_true, fa)
            costs_b[m] = dot(q_true, fb)
        end

        # Absolute costs
        @printf("  %-16s: mean=%.4f, std=%.4f, p95=%.4f, p99=%.4f, max=%.4f\n",
                la, mean(costs_a), std(costs_a), quantile(costs_a, 0.95),
                quantile(costs_a, 0.99), maximum(costs_a))
        @printf("  %-16s: mean=%.4f, std=%.4f, p95=%.4f, p99=%.4f, max=%.4f\n",
                lb, mean(costs_b), std(costs_b), quantile(costs_b, 0.95),
                quantile(costs_b, 0.99), maximum(costs_b))

        # Pairwise Δ = B − A (negative = B better)
        gap = costs_b .- costs_a
        wins_a = sum(gap .> 0)

        @printf("  TV(q_true, q̂) mean = %.4f\n", mean(tv_dists))
        @printf("  Δ (%s − %s):\n", lb, la)
        @printf("    Δmean=%+.4f, Δp90=%+.4f, Δp95=%+.4f, Δp99=%+.4f, Δmax=%+.4f\n",
                mean(gap), quantile(gap, 0.90), quantile(gap, 0.95),
                quantile(gap, 0.99), maximum(gap))
        @printf("    Δmin=%+.4f, Δp01=%+.4f, Δp05=%+.4f\n",
                minimum(gap), quantile(gap, 0.01), quantile(gap, 0.05))
        @printf("    %s wins: %d/%d (%.1f%%)\n", la, wins_a, M, 100*wins_a/M)
        @printf("    Paired gap: p01=%+.4f, p05=%+.4f, med=%+.4f, p95=%+.4f, p99=%+.4f\n",
                quantile(gap, 0.01), quantile(gap, 0.05), median(gap),
                quantile(gap, 0.95), quantile(gap, 0.99))
    end
end

println("\n\nDone!")
