"""
plot_oos_boxplot.jl — Generic OOS boxplot for arbitrary networks and x solutions.
  Phase A symmetric Dirichlet, β=0.5/1.0/5.0, M=500, seed=42

Usage:
  julia plot_oos_boxplot.jl <network> <scenario> <x1> [x2] [x3] [x4]
  network:  grid5x5, abilene, nobel_us, sioux_falls, polska
  scenario: uniform or factor
  x:        comma-separated arc indices, e.g. "3,6" or "3,6,7"
            γ (budget) is inferred from the number of arcs in each x.

Examples:
  julia plot_oos_boxplot.jl polska uniform "3,6" "18,33"
  julia plot_oos_boxplot.jl polska uniform "3,6" "6,14" "18,33"
  julia plot_oos_boxplot.jl grid5x5 factor "5,12,18"
"""

using JuMP, HiGHS, Gurobi, Printf, Dates, Statistics, Random, Distributions, LinearAlgebra
using Plots

include("../../network_generator.jl")
using .NetworkGenerator

include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")

# ── parse arguments ──
if length(ARGS) < 3
    error("Usage: julia plot_oos_boxplot.jl <network> <scenario> <x1> [x2] [x3] [x4]")
end

network_name = lowercase(ARGS[1])
scenario = lowercase(ARGS[2])
scenario in ("uniform", "factor") || error("Unknown scenario: $scenario (uniform or factor)")

# Parse x vectors from args
x_arc_lists = Vector{Vector{Int}}()
for i in 3:length(ARGS)
    arcs = parse.(Int, split(ARGS[i], ","))
    push!(x_arc_lists, arcs)
end
num_solutions = length(x_arc_lists)
num_solutions >= 1 || error("At least one x solution required")
num_solutions <= 4 || error("At most 4 x solutions supported")

# ── generate network ──
if network_name == "grid5x5"
    net = generate_grid_network(5, 5; seed=42)
    intd_arcs = net.interdictable_arcs
elseif network_name == "abilene"
    net = generate_abilene_network()
    intd_arcs = fill(true, length(net.arcs))
    net = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
        net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
elseif network_name == "nobel_us"
    net = generate_nobel_us_network()
    intd_arcs = fill(true, length(net.arcs))
    net = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
        net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
elseif network_name == "sioux_falls"
    net = generate_sioux_falls_network()
    intd_arcs = fill(true, length(net.arcs))
    net = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
        net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
elseif network_name == "polska"
    net = generate_polska_network()
    intd_arcs = fill(true, length(net.arcs))
    net = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
        net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
else
    error("Unknown network: $network_name")
end

num_arcs = length(net.arcs) - 1
S = 20

if scenario == "factor"
    caps, _ = generate_capacity_scenarios_factor_additive(length(net.arcs), S;
        interdictable_arcs=intd_arcs, seed=42, num_factors=5)
else
    caps, _ = generate_capacity_scenarios_uniform_model(length(net.arcs), S;
        interdictable_arcs=intd_arcs, seed=42)
end
intd_idx = findall(intd_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)

# Build x vectors and labels
colors_pool = [:steelblue, :seagreen, :darkorange, :purple]
x_dict = Dict{Symbol, Vector{Float64}}()
labels = Dict{Symbol, String}()
colors_map = Dict{Symbol, Symbol}()
keys_order = Vector{Symbol}()

for (i, arcs) in enumerate(x_arc_lists)
    arc_str = join(arcs, ",")
    key = Symbol("x$i")
    push!(keys_order, key)
    labels[key] = "[$arc_str]"
    colors_map[key] = colors_pool[i]

    xv = zeros(Float64, num_arcs)
    for a in arcs
        1 <= a <= num_arcs || error("Arc $a out of range (1..$num_arcs)")
        xv[a] = 1.0
    end
    x_dict[key] = xv
end

println("Network: $network_name, Scenario: $scenario, w=$w")
for (i, arcs) in enumerate(x_arc_lists)
    println("  x$i = [$(join(arcs, ","))]  (budget=$(length(arcs)))")
end
flush(stdout)

# ── boxplot function ──
function make_boxplot(costs_dict, keys_order, labels, colors_map, title_str; savepath=nothing)
    n = length(keys_order)

    stats = Dict()
    for k in keys_order
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
    for k in keys_order
        s = stats[k]
        @printf("  %-20s: median=%.6f, mean=%.6f\n", labels[k], s.median, s.mean_val)
        @printf("    [p05, p95] = [%.6f, %.6f]  [q25, q75] = [%.6f, %.6f]\n", s.p05, s.p95, s.q25, s.q75)
    end

    # Pairwise gaps (each vs first)
    for i in 2:n
        gap = costs_dict[keys_order[i]] .- costs_dict[keys_order[1]]
        @printf("  Gap (%s - %s): mean=%+.6f, wins=%d/%d (%.1f%%)\n",
                labels[keys_order[i]], labels[keys_order[1]],
                mean(gap), sum(gap .< 0), length(gap), 100*mean(gap .< 0))
    end

    p = plot(size=(200 + 180*n, 450), legend=false, grid=true, gridalpha=0.3,
             title=title_str, ylabel="OOS Expected Max-Flow (lower = better)",
             titlefontsize=11, guidefontsize=10)

    for (i, k) in enumerate(keys_order)
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

    xticks!(p, collect(1:n), [labels[k] for k in keys_order])

    if savepath !== nothing
        savefig(p, savepath)
        println("  Saved: $savepath")
    end
    return p
end

# ── run OOS for each β (Dirichlet) ──
β_list = [0.5, 1.0, 5.0]
mkpath(joinpath(@__DIR__, "plots"))

x_tag = join([join(arcs, "") for arcs in x_arc_lists], "_vs_")

for β in β_list
    @printf("\n--- %s %s β=%.1f ---\n", network_name, scenario, β)
    flush(stdout)

    costs = oos_phase_b_generic(x_dict, net, caps, 1.0, w;
                                 β=β, M=500, seed=42, mode=:symmetric)

    fname = joinpath(@__DIR__, "plots",
                     @sprintf("%s_%s_%s_beta%.1f.png", network_name, scenario, x_tag, β))
    title = @sprintf("%s %s OOS (β=%.1f)", network_name, scenario, β)

    make_boxplot(costs, keys_order, labels, colors_map, title; savepath=fname)
    flush(stdout)
end

# ── run OOS Ben-Tal style (Normal sampling, Option b' nested) ──
N_cal_list = [30, 100, 300]

for N_cal in N_cal_list
    ε_oos = weissman_epsilon(N_cal, S)
    @printf("\n--- %s %s BenTal N_cal=%d ε_oos=%.4f ---\n", network_name, scenario, N_cal, ε_oos)
    flush(stdout)

    result = oos_phase_bental(x_dict, net, caps, 1.0, w;
                               ε_oos=ε_oos, M=100, L=1000, seed=42)
    costs = result[:costs]  # Dict(key => Vector{Float64}(M)) of outer means Ȳ^(j)

    fname = joinpath(@__DIR__, "plots",
                     @sprintf("%s_%s_%s_bental_N%d.png", network_name, scenario, x_tag, N_cal))
    title = @sprintf("%s %s BenTal OOS (N=%d, ε=%.3f)", network_name, scenario, N_cal, ε_oos)

    make_boxplot(costs, keys_order, labels, colors_map, title; savepath=fname)

    # Print variance decomposition
    vd = result[:var_decomp]
    println("  Variance decomposition:")
    for k in keys_order
        d = vd[k]
        @printf("    %-20s: Var_outer=%.2e, Var_inner=%.2e, f_share=%.4f\n",
                labels[k], d.var_outer, d.var_inner, d.follower_share)
    end
    flush(stdout)
end

println("\n\nDone! $(now())")
