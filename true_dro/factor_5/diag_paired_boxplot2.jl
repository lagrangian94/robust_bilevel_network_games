"""
diag_paired_boxplot2.jl — Paired Δ boxplot, conditioning on BOTH nom-tail and dro-tail
  Left: condition on nom cost top 20%/10%/5% (기존)
  Right: condition on dro cost top 20%/10%/5%
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

# ── collect data ──
# gaps_nom[(β, frac)] = paired gap conditioned on nom tail
# gaps_dro[(β, frac)] = paired gap conditioned on dro tail
gaps_nom = Dict{Tuple{Float64,Float64}, Vector{Float64}}()
gaps_dro = Dict{Tuple{Float64,Float64}, Vector{Float64}}()

for β in β_list
    @printf("Computing β=%.1f ...\n", β)
    flush(stdout)
    costs = oos_phase_b_generic(x_dict, net, caps, 1.0, w;
                                 β=β, M=M, seed=42, mode=:symmetric)
    c_nom = costs[:nom]; c_dro = costs[:dro]
    gap = c_dro .- c_nom

    # overall
    gaps_nom[(β, 1.0)] = gap
    gaps_dro[(β, 1.0)] = gap

    nom_order = sortperm(c_nom, rev=true)
    dro_order = sortperm(c_dro, rev=true)

    for frac in [0.20, 0.10, 0.05]
        k = max(1, Int(ceil(M * frac)))
        gaps_nom[(β, frac)] = gap[nom_order[1:k]]
        gaps_dro[(β, frac)] = gap[dro_order[1:k]]
    end
end

# ── print summary ──
println("\n" * "=" ^ 80)
println("Condition on NOM tail (nom이 나쁠 때)")
println("=" ^ 80)
@printf("%-10s", "")
for β in β_list; @printf(" │ β=%-22.1f", β); end; println()
for (label, frac) in zip(cond_labels, cond_fracs)
    @printf("%-10s", label)
    for β in β_list
        g = gaps_nom[(β, frac)]
        @printf(" │ med=%+.3f win=%4.0f%% n=%d", median(g), 100*mean(g .< 0), length(g))
    end
    println()
end

println("\n" * "=" ^ 80)
println("Condition on DRO tail (dro가 나쁠 때)")
println("=" ^ 80)
@printf("%-10s", "")
for β in β_list; @printf(" │ β=%-22.1f", β); end; println()
for (label, frac) in zip(cond_labels, cond_fracs)
    @printf("%-10s", label)
    for β in β_list
        g = gaps_dro[(β, frac)]
        @printf(" │ med=%+.3f win=%4.0f%% n=%d", median(g), 100*mean(g .< 0), length(g))
    end
    println()
end

# ── plot: 2 rows x 4 cols ──
# Row 1: condition on nom tail (4 boxes per panel)
# Row 2: condition on dro tail (4 boxes per panel)

box_colors = [:steelblue, :seagreen, :darkorange, :purple]

plt = plot(layout=(2, length(β_list)), size=(1400, 500),
           bottom_margin=5Plots.mm, left_margin=5Plots.mm, top_margin=3Plots.mm)

for (ci, β) in enumerate(β_list)
    for (row, (gaps_dict, row_label)) in enumerate([(gaps_nom, "Cond. on NOM tail"),
                                                      (gaps_dro, "Cond. on DRO tail")])
        sp = (row - 1) * length(β_list) + ci

        for (ri, (label, frac)) in enumerate(zip(reverse(cond_labels), reverse(cond_fracs)))
            g = gaps_dict[(β, frac)]
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
            plot!(plt, [s.p05, s.q25], [y, y], subplot=sp, color=col, linewidth=1.5, linestyle=:dash, legend=false)
            plot!(plt, [s.q75, s.p95], [y, y], subplot=sp, color=col, linewidth=1.5, linestyle=:dash)
            cap_h = 0.15
            plot!(plt, [s.p05, s.p05], [y-cap_h, y+cap_h], subplot=sp, color=col, linewidth=2)
            plot!(plt, [s.p95, s.p95], [y-cap_h, y+cap_h], subplot=sp, color=col, linewidth=2)
            box_h = 0.3
            bx = [s.q25, s.q75, s.q75, s.q25, s.q25]
            by = [y-box_h, y-box_h, y+box_h, y+box_h, y-box_h]
            plot!(plt, Shape(bx, by), subplot=sp, fillcolor=col, fillalpha=0.3, linecolor=col, linewidth=2)
            plot!(plt, [s.median, s.median], [y-box_h, y+box_h], subplot=sp, color=col, linewidth=3)
        end

        vline!(plt, [0.0], subplot=sp, color=:red, linewidth=2, linestyle=:dash)
        yticks!(plt, 1:length(cond_labels), reverse(cond_labels), subplot=sp)
        xlabel!(plt, "Δ (dro − nom)", subplot=sp)

        if row == 1
            title!(plt, @sprintf("β=%.1f", β), subplot=sp)
        end
        if ci == 1
            ylabel!(plt, row_label, subplot=sp)
        end
    end
end

mkpath(joinpath(@__DIR__, "plots"))
savepath = joinpath(@__DIR__, "plots", "polska_factor_paired_hbox_both.png")
savefig(plt, savepath)
println("\nSaved: $savepath")
println("Done!")
