"""
plot_grid5x5_boxplot.jl — Grid 5x5 γ=1 λU=10.0 Nominal vs Robust boxplot.
p_center 방식 그대로 (run_oos_eval_grid_lambdaU10.jl과 동일 로직).
"""

using JuMP, HiGHS, Gurobi, Printf, Dates, Statistics, Random, Distributions, LinearAlgebra
using Plots

include("../../network_generator.jl")
using .NetworkGenerator

include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")

function make_boxplot(costs_a, costs_b, label_a, label_b, title_str; savepath=nothing)
    stats = Dict()
    for (name, data) in [(label_a, costs_a), (label_b, costs_b)]
        stats[name] = (
            median = median(data),
            q25 = quantile(data, 0.25),
            q75 = quantile(data, 0.75),
            p05 = quantile(data, 0.05),
            p95 = quantile(data, 0.95),
            mean_val = mean(data),
            min_val = minimum(data),
            max_val = maximum(data),
        )
    end

    println("\n" * "=" ^ 60)
    println(title_str)
    println("=" ^ 60)
    for name in [label_a, label_b]
        s = stats[name]
        @printf("  %-12s: median=%.6f, mean=%.6f\n", name, s.median, s.mean_val)
        @printf("    [p05, p95] = [%.6f, %.6f]  (range=%.6f)\n", s.p05, s.p95, s.p95 - s.p05)
        @printf("    [q25, q75] = [%.6f, %.6f]  (IQR=%.6f)\n", s.q25, s.q75, s.q75 - s.q25)
        @printf("    [min, max] = [%.6f, %.6f]  (range=%.6f)\n", s.min_val, s.max_val, s.max_val - s.min_val)
    end

    gap = costs_b .- costs_a
    @printf("  Gap (%s-%s): mean=%+.6f, median=%+.6f\n", label_b, label_a, mean(gap), median(gap))
    @printf("    %s wins: %d/%d (%.1f%%)\n", label_b, sum(gap .< 0), length(gap), 100*mean(gap .< 0))

    # y축: 두 분포 모두 포함하도록
    all_min = min(stats[label_a].min_val, stats[label_b].min_val)
    all_max = max(stats[label_a].max_val, stats[label_b].max_val)
    y_pad = 0.05 * (all_max - all_min)
    p = plot(size=(600, 450), legend=false, grid=true, gridalpha=0.3,
             title=title_str, ylabel="OOS Expected Max-Flow (lower = better)",
             titlefontsize=11, guidefontsize=10,
             ylims=(all_min - y_pad, all_max + 3*y_pad))

    colors = [:steelblue, :darkorange]
    labels = [label_a, label_b]
    all_data = [costs_a, costs_b]

    for (i, (data, col, lab)) in enumerate(zip(all_data, colors, labels))
        s = stats[lab]
        x = i
        plot!(p, [x, x], [s.p05, s.q25], color=col, linewidth=1.5, linestyle=:dash)
        plot!(p, [x, x], [s.q75, s.p95], color=col, linewidth=1.5, linestyle=:dash)
        cap_w = 0.15
        plot!(p, [x - cap_w, x + cap_w], [s.p05, s.p05], color=col, linewidth=2)
        plot!(p, [x - cap_w, x + cap_w], [s.p95, s.p95], color=col, linewidth=2)
        box_w = 0.3
        box_x = [x - box_w, x + box_w, x + box_w, x - box_w, x - box_w]
        box_y = [s.q25, s.q25, s.q75, s.q75, s.q25]
        plot!(p, Shape(box_x, box_y), fillcolor=col, fillalpha=0.3, linecolor=col, linewidth=2)
        plot!(p, [x - box_w, x + box_w], [s.median, s.median], color=col, linewidth=3)
        scatter!(p, [x], [s.mean_val], color=col, marker=:diamond, markersize=6)
        scatter!(p, [x], [s.min_val], color=col, marker=:x, markersize=5)
        scatter!(p, [x], [s.max_val], color=col, marker=:x, markersize=5)
    end

    xticks!(p, [1, 2], labels)
    sa = stats[label_a]
    sb = stats[label_b]
    annotate!(p, 1.5, max(sa.p95, sb.p95) + 0.02 * abs(sa.p95 - sa.p05),
              text(@sprintf("p95: %s=%.4f, %s=%.4f\nmax: %s=%.4f, %s=%.4f",
                            label_a, sa.p95, label_b, sb.p95,
                            label_a, sa.max_val, label_b, sb.max_val),
                   8, :center))

    if savepath !== nothing
        savefig(p, savepath)
        println("  Saved: $savepath")
    end
    return p
end

# ============================================================
# Grid 5x5 setup (same as run_oos_eval_grid_lambdaU10.jl)
# ============================================================
net = generate_grid_network(5, 5)
num_arcs = length(net.arcs) - 1
caps, _ = generate_capacity_scenarios_factor_sparse(length(net.arcs), 20;
    interdictable_arcs=net.interdictable_arcs, seed=42, num_factors=5)
intd_idx = findall(net.interdictable_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
v = 1.0
K = 20

x_nom = zeros(Float64, num_arcs); x_nom[22] = 1.0
x_rob = zeros(Float64, num_arcs); x_rob[25] = 1.0

M = 500

for β in [0.5, 1.0, 5.0]
    @printf("\n--- Grid5x5 β=%.1f ---\n", β)
    flush(stdout)

    rng = MersenneTwister(42)
    costs_nom = Vector{Float64}(undef, M)
    costs_rob = Vector{Float64}(undef, M)

    for m in 1:M
        α = β .* (1.0 .+ 0.5 * randn(rng, K))
        α = max.(α, 0.01)
        p_center = α / sum(α)

        h_nom = solve_follower_weighted(net, x_nom, v, w, caps, p_center)
        flows_nom = compute_maxflow_per_scenario(net, x_nom, h_nom, v, caps)
        costs_nom[m] = dot(p_center, flows_nom)

        h_rob = solve_follower_weighted(net, x_rob, v, w, caps, p_center)
        flows_rob = compute_maxflow_per_scenario(net, x_rob, h_rob, v, caps)
        costs_rob[m] = dot(p_center, flows_rob)

        if m % 100 == 0; @printf("  OOS %d/%d\n", m, M); end
    end

    fname = @sprintf("true_dro/factor_5/plots/grid5x5_boxplot_beta%.1f.png", β)
    title = @sprintf("Grid 5x5 OOS (β=%.1f, γ=1, λU=10)", β)
    make_boxplot(costs_nom, costs_rob, "Nominal", "Robust", title; savepath=fname)
    flush(stdout)
end

println("\nDone! $(now())")
