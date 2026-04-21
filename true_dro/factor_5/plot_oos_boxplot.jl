"""
plot_oos_boxplot.jl — Nominal vs Calibrated OOS cost distribution boxplots.

두 가지 mode:
  :same_alpha  — q_tilde ~ Dir(α), p_true ~ Dir(α). 같은 α, 독립 샘플.
  :diff_alpha  — α_tilde 별도 생성, q_tilde ~ Dir(α_tilde), p_true ~ Dir(α).

Whiskers at 5%/95% quantiles. Box at 25%/75%. Median line.
Leader minimizes → lower is better.
"""

using JuMP, HiGHS, Gurobi, Printf, Dates, Statistics, Random, Distributions, LinearAlgebra
using Plots

include("../../network_generator.jl")
using .NetworkGenerator

include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")

function make_boxplot(costs_nom, costs_cal, title_str; savepath=nothing)
    stats = Dict()
    for (name, data) in [("Nominal", costs_nom), ("Calibrated", costs_cal)]
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
    for name in ["Nominal", "Calibrated"]
        s = stats[name]
        @printf("  %-12s: median=%.6f, mean=%.6f\n", name, s.median, s.mean_val)
        @printf("    [p05, p95] = [%.6f, %.6f]  (range=%.6f)\n", s.p05, s.p95, s.p95 - s.p05)
        @printf("    [q25, q75] = [%.6f, %.6f]  (IQR=%.6f)\n", s.q25, s.q75, s.q75 - s.q25)
        @printf("    [min, max] = [%.6f, %.6f]  (range=%.6f)\n", s.min_val, s.max_val, s.max_val - s.min_val)
    end

    # Gap stats
    gap = costs_cal .- costs_nom
    @printf("  Gap (cal-nom): mean=%+.6f, median=%+.6f\n", mean(gap), median(gap))
    @printf("    cal wins: %d/%d (%.1f%%)\n", sum(gap .< 0), length(gap), 100*mean(gap .< 0))

    p = plot(size=(600, 450), legend=false, grid=true, gridalpha=0.3,
             title=title_str, ylabel="OOS Expected Max-Flow (lower = better)",
             titlefontsize=11, guidefontsize=10)

    colors = [:steelblue, :darkorange]
    labels = ["Nominal", "Calibrated"]
    all_data = [costs_nom, costs_cal]

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

    sn = stats["Nominal"]
    sc = stats["Calibrated"]
    annotate!(p, 1.5, sn.p95 + 0.02 * (sn.p95 - sn.p05),
              text(@sprintf("p95: Nom=%.4f, Cal=%.4f\nmax: Nom=%.4f, Cal=%.4f",
                            sn.p95, sc.p95, sn.max_val, sc.max_val),
                   8, :center))

    if savepath !== nothing
        savefig(p, savepath)
        println("  Saved: $savepath")
    end
    return p
end

# ============================================================
# Network setup
# ============================================================
function setup_network(gen_func)
    net = gen_func()
    num_arcs = length(net.arcs) - 1
    all_intd = fill(true, length(net.arcs))
    net_mod = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
        net.N, all_intd, net.arc_adjacency, net.node_arc_incidence)
    caps, _ = generate_capacity_scenarios_factor_sparse(length(net_mod.arcs), 20;
        interdictable_arcs=all_intd, seed=42, num_factors=5)
    intd_idx = findall(all_intd[1:num_arcs])
    w = round(maximum(caps[intd_idx, :]); digits=4)
    return net_mod, caps, w, num_arcs
end

# ============================================================
# Run both modes for both networks
# ============================================================
β_list = [0.5, 1.0, 5.0]

for (net_name, gen_func, nom_arc, cal_arc, eps_str) in [
    ("Abilene", generate_abilene_network, 11, 5, "0.566"),
    ("Polska",  generate_polska_network,  18, 33, "0.148"),
]
    net_mod, caps, w, num_arcs = setup_network(gen_func)

    x_nom = zeros(Float64, num_arcs); x_nom[nom_arc] = 1.0
    x_cal = zeros(Float64, num_arcs); x_cal[cal_arc] = 1.0
    x_dict = Dict(:nominal => x_nom, :calibrated => x_cal)

    for mode in [:same_alpha, :diff_alpha]
        println("\n" * "=" ^ 80)
        @printf(">>> %s — mode=%s\n", net_name, mode)
        println("=" ^ 80)

        for β in β_list
            @printf("\n--- %s β=%.1f mode=%s ---\n", net_name, β, mode)
            flush(stdout)

            costs = oos_phase_b_generic(x_dict, net_mod, caps, 1.0, w;
                                         β=β, M=500, seed=42, mode=mode)

            fname = @sprintf("true_dro/factor_5/plots/%s_boxplot_%s_beta%.1f.png",
                             lowercase(net_name), mode, β)
            title = @sprintf("%s OOS [%s] (β=%.1f, ε=%s)", net_name, mode, β, eps_str)

            make_boxplot(costs[:nominal], costs[:calibrated], title; savepath=fname)
            flush(stdout)
        end
    end
end

println("\n\nDone! $(now())")
