"""
diag_adversarial_direction.jl — OOS에서 adversarial 방향(s=2 과대가중)이 존재하지만 평균에 묻히는지 검증
  Polska, factor_additive, γ=2, S=20
  q̃ = q̂ 고정, q_true ~ Dir(β), M=5000

  핵심 검증:
  - q_true[2]가 높은 상위 10%에서 rob 승률이 높은가?
  - q_true[2] vs gap의 상관관계
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

# ── h* with q̃ = q̂ (한 번만) ──
x_nom = make_x([3, 6], num_arcs)
x_rob = make_x([6, 18], num_arcs)   # Single-L solution

h_nom = solve_follower_weighted(net, x_nom, v_eff, w, caps, q_hat)
h_rob = solve_follower_weighted(net, x_rob, v_eff, w, caps, q_hat)

f_nom = compute_maxflow_per_scenario(net, x_nom, h_nom, v_eff, caps)
f_rob = compute_maxflow_per_scenario(net, x_rob, h_rob, v_eff, caps)

g = f_nom .- f_rob  # g[s] > 0 이면 nom이 s에서 rob보다 나쁨 (maxflow 높음 = attacker에게 나쁨)

@printf("Per-scenario g = f_nom - f_rob:\n")
@printf("  s   f_nom    f_rob    g(=f_nom-f_rob)\n")
for s in 1:S
    marker = g[s] > 0 ? " ← nom worse" : ""
    @printf("  %2d  %7.4f  %7.4f  %+7.4f%s\n", s, f_nom[s], f_rob[s], g[s], marker)
end
@printf("\nE_{q̂}[g] = %.4f  (negative = nom wins)\n", dot(q_hat, g))
@printf("g[2] = %+.4f  (adversarial target)\n\n", g[2])

# ── OOS with stratification by q_true[2] ──
M = 5000
rng = MersenneTwister(42)

for β in [0.5, 1.0, 5.0]
    println("=" ^ 60)
    @printf("β = %.1f, M = %d\n", β, M)
    println("=" ^ 60)

    dir = Dirichlet(S, β)

    gaps = Vector{Float64}(undef, M)      # dot(q_true, g) — positive = nom worse
    q2_vals = Vector{Float64}(undef, M)    # q_true[2]

    for m in 1:M
        q_true = rand(rng, dir)
        gaps[m] = dot(q_true, g)
        q2_vals[m] = q_true[2]
    end

    # overall
    nom_wins = sum(gaps .< 0)
    @printf("\n  Overall: nom wins %d/%d (%.1f%%), mean gap = %+.4f\n",
            nom_wins, M, 100*nom_wins/M, mean(gaps))

    # stratify by q_true[2] deciles
    perm = sortperm(q2_vals)
    n_bin = M ÷ 10

    @printf("\n  Decile  q2_range          mean_gap   nom_win%%\n")
    for d in 1:10
        idx = perm[(d-1)*n_bin+1 : d*n_bin]
        q2_lo = minimum(q2_vals[idx])
        q2_hi = maximum(q2_vals[idx])
        mg = mean(gaps[idx])
        nw = sum(gaps[idx] .< 0) / n_bin * 100
        marker = mg > 0 ? " ← rob wins" : ""
        @printf("  D%02d     [%.4f, %.4f]  %+.4f    %.1f%%%s\n",
                d, q2_lo, q2_hi, mg, nw, marker)
    end

    # correlation
    corr_q2_gap = cor(q2_vals, gaps)
    @printf("\n  corr(q_true[2], gap) = %.4f\n", corr_q2_gap)

    # top-1 adversarial scenario별 상관관계
    # g가 가장 큰 상위 3개 scenario
    g_sorted = sortperm(g; rev=true)
    @printf("  Top-3 g scenarios: s=%d(g=%+.2f), s=%d(g=%+.2f), s=%d(g=%+.2f)\n",
            g_sorted[1], g[g_sorted[1]], g_sorted[2], g[g_sorted[2]], g_sorted[3], g[g_sorted[3]])

    # 상위 3개 scenario weight 합 vs gap 상관관계
    adv_weights = [sum(rand(MersenneTwister(42+m), dir)[g_sorted[1:3]]) for m in 1:100]  # 별도 검증용

    # 실제 데이터로
    rng2 = MersenneTwister(42)
    adv_sum = Vector{Float64}(undef, M)
    for m in 1:M
        q_true = rand(rng2, dir)
        adv_sum[m] = sum(q_true[g_sorted[1:3]])
    end
    # 주의: rng2와 rng이 같은 seed이므로 동일한 샘플
    @printf("  corr(Σq_adv_top3, gap) = %.4f\n", cor(adv_sum, gaps))

    println()
end

println("Done!")
