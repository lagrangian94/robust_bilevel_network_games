"""
plot_grid5x5_seed42_boxplot.jl — Grid 5x5 seed=42 γ=2 λU=10.0
x_nom=[24,33] vs x_rob=[32,33], oos_phase_b_generic.
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
    annotate!(p, 1.5, max(sa.p95, sb.p95) + 0.02 * abs(all_max - all_min),
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
# Grid 5x5 seed=42, γ=2
# ============================================================
net = generate_grid_network(5, 5; seed=42)
num_arcs = length(net.arcs) - 1
caps, _ = generate_capacity_scenarios_factor_sparse(length(net.arcs), 20;
    interdictable_arcs=net.interdictable_arcs, seed=42, num_factors=5)
intd_idx = findall(net.interdictable_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)

x_nom = zeros(Float64, num_arcs); x_nom[24] = 1.0; x_nom[33] = 1.0
x_rob = zeros(Float64, num_arcs); x_rob[32] = 1.0; x_rob[33] = 1.0
x_dict = Dict(:nominal => x_nom, :robust => x_rob)

@printf("Grid 5x5 seed=42: arcs=%d, γ=2, w=%.4f, x_nom=[24,33], x_rob=[32,33]\n", num_arcs, w)

β_list = [0.5, 1.0, 5.0]

for mode in [:symmetric, :same_alpha, :diff_alpha]
    println("\n" * "=" ^ 80)
    @printf(">>> Grid 5x5 seed=42 — mode=%s\n", mode)
    println("=" ^ 80)

    for β in β_list
        @printf("\n--- Grid5x5s42 β=%.1f mode=%s ---\n", β, mode)
        flush(stdout)

        costs = oos_phase_b_generic(x_dict, net, caps, 1.0, w;
                                     β=β, M=500, seed=42, mode=mode)

        fname = @sprintf("true_dro/factor_5/plots/grid5x5_seed42_boxplot_%s_beta%.1f.png", mode, β)
        title = @sprintf("Grid 5x5 (seed=42) OOS [%s] (β=%.1f, γ=2)", mode, β)
        make_boxplot(costs[:nominal], costs[:robust], "Nominal", "Robust", title; savepath=fname)
        flush(stdout)
    end
end

println("\n\nDone! $(now())")
