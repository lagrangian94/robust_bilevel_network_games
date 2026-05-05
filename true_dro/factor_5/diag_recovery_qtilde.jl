"""
diag_recovery_qtilde.jl — q̃ 변동이 follower recovery h*에 미치는 영향 분석
  q_true = q̂ 고정 (분포 불확실성 제거), q̃ ~ Dir(β) 샘플링 (follower 정보 왜곡)
  Polska, factor_additive, S=20

  핵심 질문: q̃가 왜곡되면 h*가 suboptimal해지는데,
  [3,6]과 [6,18] 중 어느 쪽이 더 많이 손해보나?
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

function make_x(arcs, n)
    x = zeros(n); for a in arcs; x[a] = 1.0; end; x
end

x_configs = [
    ("Nominal [3,6]",   [3, 6]),
    ("Single-L [6,18]", [6, 18]),
]

# ── Phase 0: baseline (q̃ = q̂, q_true = q̂) ──
println("=" ^ 60)
println("Phase 0: Baseline — q̃ = q̂, q_true = q̂")
println("=" ^ 60)

baseline = Dict{String, Float64}()
baseline_flows = Dict{String, Vector{Float64}}()

for (label, arcs) in x_configs
    x_vec = make_x(arcs, num_arcs)
    h_star = solve_follower_weighted(net, x_vec, v_eff, w, caps, q_hat)
    flows = compute_maxflow_per_scenario(net, x_vec, h_star, v_eff, caps)
    cost = dot(q_hat, flows)
    baseline[label] = cost
    baseline_flows[label] = flows

    @printf("  %s: E_{q̂}[maxflow] = %.6f\n", label, cost)

    h_nz = findall(h_star .> 1e-6)
    @printf("    h* arcs: ")
    for k in h_nz
        @printf("%d(%s→%s, %.4f) ", k, net.arcs[k][1], net.arcs[k][2], h_star[k])
    end
    println()
end
@printf("  Gap (nom - rob) = %.4f  (negative = nom wins)\n\n",
        baseline[x_configs[1][1]] - baseline[x_configs[2][1]])

# ── Phase 1: q̃ ~ Dir(β), q_true = q̂ ──
println("=" ^ 60)
println("Phase 1: q̃ ~ Dir(β), q_true = q̂ 고정")
println("  → h*(q̃) suboptimality가 [3,6] vs [6,18]에 미치는 영향")
println("=" ^ 60)

M = 1000
rng = MersenneTwister(42)

for β in [0.5, 1.0, 5.0]
    println()
    println("-" ^ 50)
    @printf("β = %.1f, M = %d\n", β, M)
    println("-" ^ 50)

    dir = Dirichlet(S, β)
    rng_local = MersenneTwister(42)

    costs = Dict(label => Vector{Float64}(undef, M) for (label, _) in x_configs)
    h_diffs = Dict(label => Vector{Float64}(undef, M) for (label, _) in x_configs)

    for m in 1:M
        q_tilde = rand(rng_local, dir)

        for (label, arcs) in x_configs
            x_vec = make_x(arcs, num_arcs)
            h_star = solve_follower_weighted(net, x_vec, v_eff, w, caps, q_tilde)
            flows = compute_maxflow_per_scenario(net, x_vec, h_star, v_eff, caps)
            costs[label][m] = dot(q_hat, flows)  # q_true = q̂

            # h*의 baseline 대비 변화 (L1)
            h_base = solve_follower_weighted(net, x_vec, v_eff, w, caps, q_hat)
            h_diffs[label][m] = sum(abs.(h_star .- h_base))
        end
    end

    for (label, _) in x_configs
        c = costs[label]
        bl = baseline[label]
        degradation = c .- bl  # positive = q̃ 왜곡으로 성능 악화 (maxflow 증가 = attacker에게 나쁨)
        hd = h_diffs[label]
        @printf("  %s:\n", label)
        @printf("    baseline = %.4f\n", bl)
        @printf("    E[cost|q̃~Dir] = %.4f (std=%.4f)\n", mean(c), std(c))
        @printf("    degradation: mean=%+.4f, std=%.4f, [p05,p95]=[%+.4f, %+.4f]\n",
                mean(degradation), std(degradation),
                quantile(degradation, 0.05), quantile(degradation, 0.95))
        @printf("    h* L1-shift: mean=%.4f, std=%.4f\n", mean(hd), std(hd))
    end

    # pairwise
    gap = costs[x_configs[2][1]] .- costs[x_configs[1][1]]
    nom_wins = sum(gap .> 0)  # rob > nom means nom wins (lower maxflow = better for attacker)
    @printf("\n  Gap (Single-L - Nominal): mean=%+.4f, std=%.4f\n", mean(gap), std(gap))
    @printf("  Nominal wins: %d/%d (%.1f%%)\n", nom_wins, M, 100*nom_wins/M)

    # baseline gap이 q̃ 왜곡으로 줄어드는지?
    bl_gap = baseline[x_configs[2][1]] - baseline[x_configs[1][1]]
    @printf("  Baseline gap = %+.4f, OOS mean gap = %+.4f, shift = %+.4f\n",
            bl_gap, mean(gap), mean(gap) - bl_gap)
end

# ── Phase 2: q̃ 왜곡 시 h*가 어느 arc로 이동하는지 ──
println("\n" * "=" ^ 60)
println("Phase 2: h* allocation shift under q̃ perturbation (β=0.5)")
println("=" ^ 60)

rng2 = MersenneTwister(42)
dir05 = Dirichlet(S, 0.5)
n_show = 5

for (label, arcs) in x_configs
    x_vec = make_x(arcs, num_arcs)
    h_base = solve_follower_weighted(net, x_vec, v_eff, w, caps, q_hat)

    @printf("\n  %s (h* baseline):\n", label)
    h_nz = findall(h_base .> 1e-6)
    for k in h_nz
        @printf("    arc %d (%s→%s): h=%.4f\n", k, net.arcs[k][1], net.arcs[k][2], h_base[k])
    end

    # 몇 개 샘플에서 h* 변화
    rng_show = MersenneTwister(42)
    @printf("    Samples (q̃ ~ Dir(0.5)):\n")
    @printf("    %4s  ", "arc→")
    for k in h_nz
        @printf("  h[%2d]  ", k)
    end
    @printf("  E_{q̂}  degrad\n")

    for i in 1:n_show
        q_tilde = rand(rng_show, dir05)
        h_new = solve_follower_weighted(net, x_vec, v_eff, w, caps, q_tilde)
        flows = compute_maxflow_per_scenario(net, x_vec, h_new, v_eff, caps)
        cost = dot(q_hat, flows)
        deg = cost - baseline[label]

        @printf("    m=%d   ", i)
        for k in h_nz
            @printf(" %+.4f  ", h_new[k] - h_base[k])
        end
        @printf("  %.4f  %+.4f\n", cost, deg)
    end
end

println("\nDone!")
