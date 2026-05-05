"""
diag_tail_tvball.jl
  Part 1: p95/p99에서 DRO가 이길 때 qtrue 특성
  Part 2: TV ball 안에 들어오는 qtrue만으로 OOS evaluation
"""

using JuMP, HiGHS, Gurobi, Printf, Statistics, Random, Distributions, LinearAlgebra

include("../../network_generator.jl")
using .NetworkGenerator
include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")

# ── setup ──
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

x_nom = zeros(Float64, num_arcs); x_nom[3] = 1.0; x_nom[6] = 1.0
x_dro = zeros(Float64, num_arcs); x_dro[6] = 1.0; x_dro[18] = 1.0
q_hat = fill(1.0 / S, S)

# flows (fixed, follower uses q̂)
h_nom = solve_follower_weighted(net, x_nom, 1.0, w, caps, q_hat)
h_dro = solve_follower_weighted(net, x_dro, 1.0, w, caps, q_hat)
flows_nom = compute_maxflow_per_scenario(net, x_nom, h_nom, 1.0, caps)
flows_dro = compute_maxflow_per_scenario(net, x_dro, h_dro, 1.0, caps)

# DRO adversary가 주목하는 시나리오 (flow 차이 큰 순서)
diff = flows_nom .- flows_dro  # positive = nom이 더 나쁨
top_diff_idx = sortperm(diff, rev=true)
println("Per-scenario flow difference (nom - dro), sorted:")
for i in 1:S
    s = top_diff_idx[i]
    @printf("  s=%2d: nom=%.4f  dro=%.4f  diff=%+.4f\n", s, flows_nom[s], flows_dro[s], diff[s])
end
flush(stdout)

β_list = [0.1, 0.3, 0.5, 1.0, 5.0]
M = 5000

# ══════════════════════════════════════════════════════════════
# Part 1: p95/p99에서 DRO 이길 때 qtrue 특성
# ══════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("PART 1: qtrue characteristics when DRO wins at tail")
println("=" ^ 70)

for β in β_list
    costs = oos_phase_b_generic(Dict(:nom => x_nom, :dro => x_dro), net, caps, 1.0, w;
                                 β=β, M=M, seed=42, mode=:symmetric)
    c_nom = costs[:nom]; c_dro = costs[:dro]

    # 각 OOS sample의 qtrue 복원 (같은 seed로 재생성)
    rng = MersenneTwister(42)
    qtrue_all = Matrix{Float64}(undef, S, M)
    for j in 1:M
        alpha_vec = fill(β, S)
        d = Dirichlet(alpha_vec)
        qtrue_all[:, j] = rand(rng, d)
    end

    # TV distance from q̂
    tv_all = [0.5 * sum(abs.(qtrue_all[:, j] .- q_hat)) for j in 1:M]

    @printf("\n--- β=%.1f ---\n", β)

    for (pctl_name, pctl) in [("p95", 0.95), ("p99", 0.99)]
        thresh_nom = quantile(c_nom, pctl)
        thresh_dro = quantile(c_dro, pctl)

        # samples above p95/p99 for each
        tail_nom = findall(c_nom .>= thresh_nom)
        tail_dro = findall(c_dro .>= thresh_dro)

        # DRO wins in nom's tail
        dro_wins_in_nom_tail = [i for i in tail_nom if c_dro[i] < c_nom[i]]
        # DRO loses in dro's tail
        dro_loses_in_dro_tail = [i for i in tail_dro if c_dro[i] >= c_nom[i]]

        @printf("  %s (threshold: nom=%.4f, dro=%.4f)\n", pctl_name, thresh_nom, thresh_dro)
        @printf("    nom tail: %d samples, DRO wins %d (%.1f%%)\n",
                length(tail_nom), length(dro_wins_in_nom_tail),
                100*length(dro_wins_in_nom_tail)/max(1,length(tail_nom)))

        if !isempty(dro_wins_in_nom_tail)
            idx = dro_wins_in_nom_tail
            tv_win = tv_all[idx]
            q_win = mean(qtrue_all[:, idx], dims=2)[:]
            @printf("    When DRO wins in nom-tail: TV mean=%.4f, [min,max]=[%.4f,%.4f]\n",
                    mean(tv_win), minimum(tv_win), maximum(tv_win))
            # 어떤 시나리오에 weight이 쏠렸나
            q_dev = q_win .- q_hat
            top3 = sortperm(q_dev, rev=true)[1:3]
            bot3 = sortperm(q_dev)[1:3]
            @printf("    Mean qtrue (top3 overweight): ")
            for s in top3
                @printf("s%d=%+.4f ", s, q_dev[s])
            end
            @printf("\n    Mean qtrue (top3 underweight): ")
            for s in bot3
                @printf("s%d=%+.4f ", s, q_dev[s])
            end
            println()
        end
        flush(stdout)
    end
end

# ══════════════════════════════════════════════════════════════
# Part 2: TV ball 안에 들어오는 qtrue만으로 evaluation
# ══════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("PART 2: OOS restricted to TV ball (ε_hat = 0.1, 0.3)")
println("=" ^ 70)

eps_list = [0.1, 0.3]  # TV radius

for β in β_list
    for ε_tv in eps_list
        # 많이 샘플링해서 TV ball 안에 들어오는 것만 필터
        rng = MersenneTwister(42)
        M_try = 50000  # oversample
        c_nom_in = Float64[]
        c_dro_in = Float64[]
        n_total = 0
        n_inside = 0

        alpha_vec = fill(β, S)
        d = Dirichlet(alpha_vec)

        for j in 1:M_try
            qtrue = rand(rng, d)
            n_total += 1
            tv = 0.5 * sum(abs.(qtrue .- q_hat))
            if tv <= ε_tv
                n_inside += 1
                push!(c_nom_in, dot(qtrue, flows_nom))
                push!(c_dro_in, dot(qtrue, flows_dro))
            end
        end

        n = length(c_nom_in)
        if n < 10
            @printf("β=%.1f, ε_tv=%.1f: only %d/%d inside TV ball — skip\n", β, ε_tv, n, M_try)
            continue
        end

        gap = c_dro_in .- c_nom_in
        dro_wins = sum(gap .< 0)

        @printf("\nβ=%.1f, ε_tv=%.1f: %d/%d inside (%.2f%%)\n", β, ε_tv, n, M_try, 100*n/M_try)
        @printf("  [3,6]  mean=%.4f  p95=%.4f  p99=%.4f  max=%.4f\n",
                mean(c_nom_in), quantile(c_nom_in, 0.95), quantile(c_nom_in, 0.99), maximum(c_nom_in))
        @printf("  [6,18] mean=%.4f  p95=%.4f  p99=%.4f  max=%.4f\n",
                mean(c_dro_in), quantile(c_dro_in, 0.95), quantile(c_dro_in, 0.99), maximum(c_dro_in))
        @printf("  DRO wins: %d/%d (%.1f%%)\n", dro_wins, n, 100*dro_wins/n)
        @printf("  Δmean=%+.4f  Δp95=%+.4f  Δp99=%+.4f  Δmax=%+.4f\n",
                mean(c_dro_in)-mean(c_nom_in),
                quantile(c_dro_in,0.95)-quantile(c_nom_in,0.95),
                quantile(c_dro_in,0.99)-quantile(c_nom_in,0.99),
                maximum(c_dro_in)-maximum(c_nom_in))
        flush(stdout)
    end
end

println("\n\nDone!")
