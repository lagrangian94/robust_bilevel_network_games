"""
diag_follower_recovery.jl — [3,6] vs [6,34] follower recovery 분석
  Polska, factor_additive, γ=2, S=20, λU=10.0

  x를 고정하고, 다양한 q̃에서 follower primal LP를 풀어:
  - h* (recovery 배분)
  - scenario별 max-flow
  - E_{q̃}[maxflow]
  를 비교.
"""

using JuMP, HiGHS, Gurobi, Printf, Dates, LinearAlgebra, Statistics, Random, Distributions

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
v_eff = 1.0  # interdiction effectiveness

q_hat = fill(1.0/S, S)

@printf("polska factor_additive: arcs=%d, w=%.4f\n\n", num_arcs, w)

# ── x solutions ──
x_configs = [
    ("x_nom [3,6]",  [3, 6]),
    ("x_rob [6,34]", [6, 34]),
]

function make_x_vec(arcs, n)
    x = zeros(n)
    for a in arcs; x[a] = 1.0; end
    x
end

# ======================================================================
# 1. q̃ = q̂ (nominal) 에서 h*, scenario별 max-flow
# ======================================================================
println("=" ^ 70)
println("1. Follower recovery under q̃ = q̂ (uniform 1/$S)")
println("=" ^ 70)

for (label, arcs) in x_configs
    x_vec = make_x_vec(arcs, num_arcs)
    h_star = solve_follower_weighted(net, x_vec, v_eff, w, caps, q_hat)
    flows = compute_maxflow_per_scenario(net, x_vec, h_star, v_eff, caps)

    @printf("\n%s:\n", label)
    @printf("  E_{q̂}[maxflow] = %.6f\n", dot(q_hat, flows))
    @printf("  h* total = %.4f / w = %.4f\n", sum(h_star), w)

    # h* 배분 (non-zero)
    h_nz = findall(h_star .> 1e-6)
    @printf("  h* non-zero arcs: ")
    for k in h_nz
        arc = net.arcs[k]
        @printf("%d(%s→%s, h=%.4f) ", k, arc[1], arc[2], h_star[k])
    end
    println()

    # scenario별 max-flow
    @printf("  Per-scenario maxflow:\n")
    @printf("    mean=%.4f, std=%.4f, min=%.4f(s=%d), max=%.4f(s=%d)\n",
            mean(flows), std(flows), minimum(flows), argmin(flows), maximum(flows), argmax(flows))
    for s in 1:S
        @printf("    s=%2d: maxflow=%.4f", s, flows[s])
        # interdicted arc capacities
        for a in arcs
            @printf("  cap[%d]=%.4f→eff=%.4f", a, caps[a,s], caps[a,s]*(1-v_eff*x_vec[a])+h_star[a])
        end
        println()
    end
end

# ======================================================================
# 2. Capacity 구조 비교: interdicted arcs
# ======================================================================
println("\n" * "=" ^ 70)
println("2. Interdicted arc capacity structure")
println("=" ^ 70)

@printf("\n  s   cap[3]   cap[6]   cap[34]  sum[3,6]  sum[6,34]\n")
for s in 1:S
    c3 = caps[3, s]; c6 = caps[6, s]; c34 = caps[34, s]
    @printf("  %2d  %.4f   %.4f   %.4f   %.4f    %.4f\n",
            s, c3, c6, c34, c3+c6, c6+c34)
end
@printf("\n  mean: cap[3]=%.4f, cap[6]=%.4f, cap[34]=%.4f\n",
        mean(caps[3,:]), mean(caps[6,:]), mean(caps[34,:]))
@printf("  std:  cap[3]=%.4f, cap[6]=%.4f, cap[34]=%.4f\n",
        std(caps[3,:]), std(caps[6,:]), std(caps[34,:]))
@printf("  corr(3,6)=%.4f, corr(6,34)=%.4f, corr(3,34)=%.4f\n",
        cor(caps[3,:], caps[6,:]), cor(caps[6,:], caps[34,:]), cor(caps[3,:], caps[34,:]))

# ======================================================================
# 3. q̃ 변화에 따른 h* 및 E[maxflow] 변화
# ======================================================================
println("\n" * "=" ^ 70)
println("3. Follower recovery under sampled q̃ (Dirichlet)")
println("=" ^ 70)

rng = MersenneTwister(42)
n_samples = 10

for β in [0.5, 1.0, 5.0]
    @printf("\n--- β = %.1f ---\n", β)
    dir = Dirichlet(S, β)

    for (label, arcs) in x_configs
        x_vec = make_x_vec(arcs, num_arcs)
        e_flows_qtilde = Float64[]   # follower가 보는 기대값
        e_flows_qtrue  = Float64[]   # 실제 q_true로 평가
        h_totals = Float64[]
        h_arcs_all = Vector{Vector{Int}}()

        rng_copy = MersenneTwister(42)
        for i in 1:n_samples
            q_tilde = rand(rng_copy, dir)
            q_true  = rand(rng_copy, dir)

            h_star = solve_follower_weighted(net, x_vec, v_eff, w, caps, q_tilde)
            flows = compute_maxflow_per_scenario(net, x_vec, h_star, v_eff, caps)

            push!(e_flows_qtilde, dot(q_tilde, flows))
            push!(e_flows_qtrue, dot(q_true, flows))
            push!(h_totals, sum(h_star))
            push!(h_arcs_all, findall(h_star .> 1e-6))
        end

        @printf("  %s:\n", label)
        @printf("    E_{q̃}[maxflow]    mean=%.4f (std=%.4f)\n", mean(e_flows_qtilde), std(e_flows_qtilde))
        @printf("    E_{q_true}[maxflow] mean=%.4f (std=%.4f)\n", mean(e_flows_qtrue), std(e_flows_qtrue))
        @printf("    h_total mean=%.4f\n", mean(h_totals))

        # h가 어느 arc에 주로 가는지
        h_arc_freq = Dict{Int,Int}()
        for arcs_h in h_arcs_all
            for a in arcs_h
                h_arc_freq[a] = get(h_arc_freq, a, 0) + 1
            end
        end
        top_arcs = sort(collect(h_arc_freq), by=x->-x[2])
        @printf("    h* frequent arcs: ")
        for (a, cnt) in top_arcs[1:min(5, length(top_arcs))]
            @printf("%d(%s→%s, %d/%d) ", a, net.arcs[a][1], net.arcs[a][2], cnt, n_samples)
        end
        println()
    end

    # pairwise comparison (같은 q̃로 h 결정, 같은 q_true로 평가)
    rng_copy = MersenneTwister(42)
    nom_wins = 0
    for i in 1:n_samples
        q_tilde = rand(rng_copy, dir)
        q_true  = rand(rng_copy, dir)

        x_nom = make_x_vec([3,6], num_arcs)
        x_rob = make_x_vec([6,34], num_arcs)

        h_nom = solve_follower_weighted(net, x_nom, v_eff, w, caps, q_tilde)
        h_rob = solve_follower_weighted(net, x_rob, v_eff, w, caps, q_tilde)

        f_nom = compute_maxflow_per_scenario(net, x_nom, h_nom, v_eff, caps)
        f_rob = compute_maxflow_per_scenario(net, x_rob, h_rob, v_eff, caps)

        e_nom = dot(q_true, f_nom)
        e_rob = dot(q_true, f_rob)

        if e_nom < e_rob
            nom_wins += 1
        end
    end
    @printf("  [3,6] wins: %d/%d (%.0f%%)\n", nom_wins, n_samples, 100*nom_wins/n_samples)
end

# ======================================================================
# 4. Extreme q̃ test: 각 scenario에 weight 집중
# ======================================================================
println("\n" * "=" ^ 70)
println("4. Extreme q̃: weight concentrated on single scenario")
println("=" ^ 70)

@printf("\n  s   E_nom[3,6]  E_rob[6,34]  Δ(rob-nom)  winner\n")
for s in 1:S
    q_ext = zeros(S)
    q_ext[s] = 1.0

    x_nom = make_x_vec([3,6], num_arcs)
    x_rob = make_x_vec([6,34], num_arcs)

    h_nom = solve_follower_weighted(net, x_nom, v_eff, w, caps, q_ext)
    h_rob = solve_follower_weighted(net, x_rob, v_eff, w, caps, q_ext)

    f_nom = compute_maxflow_per_scenario(net, x_nom, h_nom, v_eff, caps)
    f_rob = compute_maxflow_per_scenario(net, x_rob, h_rob, v_eff, caps)

    e_nom = f_nom[s]  # q concentrated on s
    e_rob = f_rob[s]

    winner = e_nom < e_rob ? "[3,6]" : (e_nom > e_rob ? "[6,34]" : "tie")
    @printf("  %2d  %.4f      %.4f       %+.4f     %s\n", s, e_nom, e_rob, e_rob - e_nom, winner)
end

println("\nDone! $(now())")
