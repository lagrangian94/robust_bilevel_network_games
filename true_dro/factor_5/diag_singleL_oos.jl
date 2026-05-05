"""
diag_singleL_oos.jl — Nominal [3,6] vs Single-L [6,18] OOS 비교
  q̃ = q̂ (follower ambiguity 없음), q_true ~ Dir(β)
  Polska, factor_additive, γ=2, S=20

  h*는 q̂로 한 번만 풀고, q_true만 M번 샘플링.
"""

using JuMP, HiGHS, Printf, Statistics, Random, Distributions, LinearAlgebra

include("../../network_generator.jl")
using .NetworkGenerator

include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")

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
q_hat = fill(1.0/S, S)

@printf("polska factor_additive: arcs=%d, w=%.4f\n\n", num_arcs, w)

# ── x solutions ──
x_configs = [
    ("Nominal [3,6]",   [3, 6]),
    ("Single-L [6,18]", [6, 18]),
]

function make_x(arcs, n)
    x = zeros(n); for a in arcs; x[a] = 1.0; end; x
end

# ── Phase 1: h* with q̃ = q̂ (한 번만) ──
println("=" ^ 60)
println("Phase 1: Follower recovery h* with q̃ = q̂")
println("=" ^ 60)

h_stars = Dict{String, Vector{Float64}}()
flows_fixed = Dict{String, Vector{Float64}}()

for (label, arcs) in x_configs
    x_vec = make_x(arcs, num_arcs)
    h_star = solve_follower_weighted(net, x_vec, v_eff, w, caps, q_hat)
    flows = compute_maxflow_per_scenario(net, x_vec, h_star, v_eff, caps)

    h_stars[label] = h_star
    flows_fixed[label] = flows

    @printf("\n%s:\n", label)
    @printf("  E_{q̂}[maxflow] = %.6f\n", dot(q_hat, flows))
    h_nz = findall(h_star .> 1e-6)
    @printf("  h* arcs: ")
    for k in h_nz
        @printf("%d(%s→%s, %.4f) ", k, net.arcs[k][1], net.arcs[k][2], h_star[k])
    end
    println()
    @printf("  Per-scenario: min=%.4f(s=%d), max=%.4f(s=%d), std=%.4f\n",
            minimum(flows), argmin(flows), maximum(flows), argmax(flows), std(flows))
end

# ── Phase 2: OOS with q_true ~ Dir(β), h* 고정 ──
println("\n" * "=" ^ 60)
println("Phase 2: OOS evaluation (q̃=q̂ fixed, q_true ~ Dir(β))")
println("=" ^ 60)

M = 5000
rng = MersenneTwister(42)

for β in [0.5, 1.0, 5.0]
    @printf("\n--- β = %.1f, M = %d ---\n", β, M)
    dir = Dirichlet(S, β)

    costs = Dict(label => Vector{Float64}(undef, M) for (label, _) in x_configs)

    for m in 1:M
        q_true = rand(rng, dir)
        for (label, _) in x_configs
            costs[label][m] = dot(q_true, flows_fixed[label])
        end
    end

    # stats
    for (label, _) in x_configs
        c = costs[label]
        @printf("  %s: mean=%.6f, std=%.4f, [p05,p95]=[%.4f, %.4f]\n",
                label, mean(c), std(c), quantile(c, 0.05), quantile(c, 0.95))
    end

    # pairwise
    nom_label = x_configs[1][1]
    rob_label = x_configs[2][1]
    gap = costs[rob_label] .- costs[nom_label]
    nom_wins = sum(gap .> 0)  # lower is better, so rob > nom means nom wins
    @printf("  Gap (%s - %s): mean=%+.6f\n", rob_label, nom_label, mean(gap))
    @printf("  %s wins: %d/%d (%.1f%%)\n", nom_label, nom_wins, M, 100*nom_wins/M)

    # per-scenario flow comparison
    if β == 1.0
        println("\n  Per-scenario maxflow comparison (h* fixed at q̂):")
        f_nom = flows_fixed[nom_label]
        f_rob = flows_fixed[rob_label]
        @printf("    s   flow_nom  flow_rob  Δ(rob-nom)\n")
        for s in 1:S
            @printf("    %2d  %.4f    %.4f    %+.4f  %s\n",
                    s, f_nom[s], f_rob[s], f_rob[s]-f_nom[s],
                    f_rob[s] < f_nom[s] ? "←rob better" : "")
        end
    end
end

println("\nDone!")
