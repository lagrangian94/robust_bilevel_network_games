"""
diag_paired_boxplot.jl — Paired difference (dro - nom) boxplot
  Row 1: overall
  Row 2-4: conditional on nom cost top 20%, 10%, 5%
  Columns: β = 0.1, 0.5, 1.0, 5.0
"""

using JuMP, HiGHS, Gurobi, Printf, Statistics, Random, Distributions, LinearAlgebra
using Plots

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
x_dict = Dict(:nom => x_nom, :dro => x_dro)

M = 5000
β_list = [0.1, 0.5, 1.0, 5.0]
cond_labels = ["All", "Top 20%", "Top 10%", "Top 5%"]
cond_fracs = [1.0, 0.20, 0.10, 0.05]

n_rows = length(cond_fracs)
n_cols = length(β_list)

# ── collect data ──
all_gaps = Dict{Tuple{Float64,Float64}, Vector{Float64}}()  # (β, frac) => gaps

for β in β_list
    @printf("Computing β=%.1f ...\n", β)
    flush(stdout)
    costs = oos_phase_b_generic(x_dict, net, caps, 1.0, w;
                                 β=β, M=M, seed=42, mode=:symmetric)
    c_nom = costs[:nom]; c_dro = costs[:dro]
    gap = c_dro .- c_nom

    # overall
    all_gaps[(β, 1.0)] = gap

    # conditional on nom tail
    nom_order = sortperm(c_nom, rev=true)
    for frac in [0.20, 0.10, 0.05]
        k = max(1, Int(ceil(M * frac)))
        idx = nom_order[1:k]
        all_gaps[(β, frac)] = gap[idx]
    end
end

# ── print summary table ──
println("\n" * "=" ^ 80)
println("Paired gap (dro - nom): negative = DRO better")
println("=" ^ 80)
@printf("%-10s", "Cond\\β")
for β in β_list
    @printf(" │ %25s", @sprintf("β=%.1f", β))
end
println()
println("-" ^ 80)

for (label, frac) in zip(cond_labels, cond_fracs)
    @printf("%-10s", label)
    for β in β_list
        g = all_gaps[(β, frac)]
        med = median(g)
        mn = mean(g)
        wins = 100 * mean(g .< 0)
        @printf(" │ med=%+.3f win=%.0f%% n=%d", med, wins, length(g))
    end
    println()
end

# ── plot: 4x4 grid ──
plt = plot(layout=(n_rows, n_cols), size=(1200, 900),
           plot_title="Paired Δ (DRO[6,18] − NOM[3,6])",
           left_margin=5Plots.mm, bottom_margin=3Plots.mm)

for (ci, β) in enumerate(β_list)
    for (ri, (label, frac)) in enumerate(zip(cond_labels, cond_fracs))
        g = all_gaps[(β, frac)]
        sp = (ri - 1) * n_cols + ci

        # horizontal histogram
        histogram!(plt, g, subplot=sp, orientation=:vertical,
                   bins=50, color=:steelblue, fillalpha=0.5, linecolor=:steelblue,
                   legend=false, grid=true, gridalpha=0.3)
        vline!(plt, [0.0], subplot=sp, color=:red, linewidth=2, linestyle=:dash)
        vline!(plt, [median(g)], subplot=sp, color=:darkorange, linewidth=2)

        # labels
        if ri == 1
            title!(plt, @sprintf("β=%.1f", β), subplot=sp)
        end
        if ci == 1
            ylabel!(plt, label, subplot=sp)
        end

        # annotate win rate
        wins = round(100 * mean(g .< 0); digits=1)
        med = round(median(g); digits=3)
        annotate!(plt, [(minimum(g) + 0.1*(maximum(g)-minimum(g)),
                        0.85 * ylims(plt[sp])[2],
                        text(@sprintf("win=%.0f%%\nmed=%+.3f", wins, med), 8, :left))],
                  subplot=sp)
    end
end

mkpath(joinpath(@__DIR__, "plots"))
savepath = joinpath(@__DIR__, "plots", "polska_factor_paired_gap.png")
savefig(plt, savepath)
println("\nSaved: $savepath")

# ── 추가: 단순 수평 boxplot 버전 (compact) ──
plt2 = plot(layout=(1, n_cols), size=(1200, 400),
            plot_title="Paired Δ by condition (orange=median, red=zero)",
            bottom_margin=8Plots.mm)

box_colors = [:steelblue, :seagreen, :darkorange, :purple]

for (ci, β) in enumerate(β_list)
    sp = ci
    for (ri, (label, frac)) in enumerate(zip(reverse(cond_labels), reverse(cond_fracs)))
        g = all_gaps[(β, frac)]
        y = ri
        s = (
            median = median(g),
            q25 = quantile(g, 0.25),
            q75 = quantile(g, 0.75),
            p05 = quantile(g, 0.05),
            p95 = quantile(g, 0.95),
        )
        col = box_colors[length(cond_fracs) - ri + 1]

        # whiskers
        plot!(plt2, [s.p05, s.q25], [y, y], subplot=sp, color=col, linewidth=1.5, linestyle=:dash, legend=false)
        plot!(plt2, [s.q75, s.p95], [y, y], subplot=sp, color=col, linewidth=1.5, linestyle=:dash)
        # caps
        cap_h = 0.15
        plot!(plt2, [s.p05, s.p05], [y-cap_h, y+cap_h], subplot=sp, color=col, linewidth=2)
        plot!(plt2, [s.p95, s.p95], [y-cap_h, y+cap_h], subplot=sp, color=col, linewidth=2)
        # box
        box_h = 0.3
        bx = [s.q25, s.q75, s.q75, s.q25, s.q25]
        by = [y-box_h, y-box_h, y+box_h, y+box_h, y-box_h]
        plot!(plt2, Shape(bx, by), subplot=sp, fillcolor=col, fillalpha=0.3, linecolor=col, linewidth=2)
        # median
        plot!(plt2, [s.median, s.median], [y-box_h, y+box_h], subplot=sp, color=col, linewidth=3)
    end
    # zero line
    vline!(plt2, [0.0], subplot=sp, color=:red, linewidth=2, linestyle=:dash)
    yticks!(plt2, 1:length(cond_labels), reverse(cond_labels), subplot=sp)
    title!(plt2, @sprintf("β=%.1f", β), subplot=sp)
    xlabel!(plt2, "Δ (dro − nom)", subplot=sp)
end

savepath2 = joinpath(@__DIR__, "plots", "polska_factor_paired_hbox.png")
savefig(plt2, savepath2)
println("Saved: $savepath2")

println("\nDone!")
