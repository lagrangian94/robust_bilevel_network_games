"""
plot_polska_gamma2_boxplot.jl — Nominal vs Single vs Double OOS boxplots.
  polska, γ=2, uniform, S=20, β=0.5/1.0/5.0
  Phase A symmetric Dirichlet (oos_phase_b_generic with mode=:symmetric)
  Nominal [3,6], Single [6,14], Double [18,33]
"""

using JuMP, HiGHS, Gurobi, Printf, Dates, Statistics, Random, Distributions, LinearAlgebra
using Plots

include("../../network_generator.jl")
using .NetworkGenerator

include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")

function make_boxplot_3way(costs_dict, title_str; savepath=nothing)
    names_order = [:nominal, :single, :double]
    display_names = Dict(:nominal => "Nominal [3,6]", :single => "Single [6,14]", :double => "Double [18,33]")
    colors_map = Dict(:nominal => :steelblue, :single => :seagreen, :double => :darkorange)

    stats = Dict()
    for k in names_order
        data = costs_dict[k]
        stats[k] = (
            median = median(data),
            q25 = quantile(data, 0.25),
            q75 = quantile(data, 0.75),
            p05 = quantile(data, 0.05),
            p95 = quantile(data, 0.95),
            mean_val = mean(data),
        )
    end

    println("\n" * "=" ^ 60)
    println(title_str)
    println("=" ^ 60)
    for k in names_order
        s = stats[k]
        @printf("  %-20s: median=%.6f, mean=%.6f\n", display_names[k], s.median, s.mean_val)
        @printf("    [p05, p95] = [%.6f, %.6f]  [q25, q75] = [%.6f, %.6f]\n", s.p05, s.p95, s.q25, s.q75)
    end

    # Pairwise gaps
    for (a, b) in [(:single, :nominal), (:double, :nominal), (:double, :single)]
        gap = costs_dict[a] .- costs_dict[b]
        @printf("  Gap (%s - %s): mean=%+.6f, wins=%d/%d (%.1f%%)\n",
                String(a), String(b), mean(gap), sum(gap .< 0), length(gap), 100*mean(gap .< 0))
    end

    p = plot(size=(750, 450), legend=false, grid=true, gridalpha=0.3,
             title=title_str, ylabel="OOS Expected Max-Flow (lower = better)",
             titlefontsize=11, guidefontsize=10)

    for (i, k) in enumerate(names_order)
        s = stats[k]
        col = colors_map[k]
        x = i

        plot!(p, [x, x], [s.p05, s.q25], color=col, linewidth=1.5, linestyle=:dash)
        plot!(p, [x, x], [s.q75, s.p95], color=col, linewidth=1.5, linestyle=:dash)

        cap_w = 0.12
        plot!(p, [x - cap_w, x + cap_w], [s.p05, s.p05], color=col, linewidth=2)
        plot!(p, [x - cap_w, x + cap_w], [s.p95, s.p95], color=col, linewidth=2)

        box_w = 0.25
        box_x = [x - box_w, x + box_w, x + box_w, x - box_w, x - box_w]
        box_y = [s.q25, s.q25, s.q75, s.q75, s.q25]
        plot!(p, Shape(box_x, box_y), fillcolor=col, fillalpha=0.3, linecolor=col, linewidth=2)

        plot!(p, [x - box_w, x + box_w], [s.median, s.median], color=col, linewidth=3)
        scatter!(p, [x], [s.mean_val], color=col, marker=:diamond, markersize=6)
    end

    xticks!(p, [1, 2, 3], [display_names[k] for k in names_order])

    if savepath !== nothing
        savefig(p, savepath)
        println("  Saved: $savepath")
    end
    return p
end

# ── Network setup ──
net = generate_polska_network()
num_arcs = length(net.arcs) - 1
all_intd = fill(true, length(net.arcs))
net_mod = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
    net.N, all_intd, net.arc_adjacency, net.node_arc_incidence)
caps, _ = generate_capacity_scenarios_uniform_model(length(net_mod.arcs), 20;
    interdictable_arcs=all_intd, seed=42)
intd_idx = findall(all_intd[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)

x_nom = zeros(Float64, num_arcs); x_nom[3] = 1.0; x_nom[6] = 1.0
x_sin = zeros(Float64, num_arcs); x_sin[6] = 1.0; x_sin[14] = 1.0
x_dbl = zeros(Float64, num_arcs); x_dbl[18] = 1.0; x_dbl[33] = 1.0
x_dict = Dict(:nominal => x_nom, :single => x_sin, :double => x_dbl)

β_list = [0.5, 1.0, 5.0]
mkpath(joinpath(@__DIR__, "plots"))

for β in β_list
    @printf("\n--- Polska γ=2 β=%.1f ---\n", β)
    flush(stdout)

    costs = oos_phase_b_generic(x_dict, net_mod, caps, 1.0, w;
                                 β=β, M=500, seed=42, mode=:symmetric)

    fname = joinpath(@__DIR__, "plots",
                     @sprintf("polska_gamma2_3way_beta%.1f.png", β))
    title = @sprintf("Polska γ=2 OOS (β=%.1f, uniform)", β)

    make_boxplot_3way(costs, title; savepath=fname)
    flush(stdout)
end

println("\n\nDone! $(now())")
