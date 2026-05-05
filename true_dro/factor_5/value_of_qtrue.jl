"""
value_of_qtrue.jl — Follower belief = q̂ (fixed uniform), p_true만 sampling.
  Leader의 x 선택이 p_true 변동에 얼마나 robust한지 비교.

  h* = follower(x*, q̂) 한 번 계산 (follower ambiguity 없음)
  p_true^(r) ~ N(q̂, σ) in TV ball → Y^(r) = dot(p_true, flows)

Usage:
  julia value_of_qtrue.jl <network> <scenario> <x1> [x2] [x3] [x4]
  e.g. julia value_of_qtrue.jl polska factor "3,6" "6,18"
"""

using JuMP, HiGHS, Gurobi, Printf, Dates, Statistics, Random, Distributions, LinearAlgebra
using Plots

include("../../network_generator.jl")
using .NetworkGenerator

include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")

# ── parse arguments ──
if length(ARGS) < 3
    error("Usage: julia value_of_qtrue.jl <network> <scenario> <x1> [x2] [x3] [x4]")
end

network_name = lowercase(ARGS[1])
scenario = lowercase(ARGS[2])
scenario in ("uniform", "factor") || error("Unknown scenario: $scenario")

x_arc_lists = Vector{Vector{Int}}()
for i in 3:length(ARGS)
    arcs = parse.(Int, split(ARGS[i], ","))
    push!(x_arc_lists, arcs)
end
num_solutions = length(x_arc_lists)
num_solutions >= 1 || error("At least one x solution required")
num_solutions <= 4 || error("At most 4 x solutions supported")

# ── generate network ──
if network_name == "polska"
    net = generate_polska_network()
    intd_arcs = fill(true, length(net.arcs))
    net = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
        net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
elseif network_name == "abilene"
    net = generate_abilene_network()
    intd_arcs = fill(true, length(net.arcs))
    net = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
        net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
elseif network_name == "grid5x5"
    net = generate_grid_network(5, 5; seed=42)
    intd_arcs = net.interdictable_arcs
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

# ── Phase 1: Compute h* and flows with q̂ = uniform (once per x) ──
q_hat = fill(1.0 / S, S)

flows_dict = Dict{Symbol, Vector{Float64}}()
for k in keys_order
    h_star = solve_follower_weighted(net, x_dict[k], 1.0, w, caps, q_hat)
    flows = compute_maxflow_per_scenario(net, x_dict[k], h_star, 1.0, caps)
    flows_dict[k] = flows
    @printf("  %s: h* computed, E_q̂[flow]=%.4f, flows range=[%.4f, %.4f]\n",
            labels[k], dot(q_hat, flows), minimum(flows), maximum(flows))
end
flush(stdout)

# ── Phase 2: Sample p_true, compute Y = dot(p_true, flows) ──
N_cal_list = [30, 100, 300, 850]
M = 1000

mkpath(joinpath(@__DIR__, "plots"))
x_tag = join([join(arcs, "") for arcs in x_arc_lists], "_vs_")

# ── boxplot function (reused) ──
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

# ── Run for each N_cal ──
for N_cal in N_cal_list
    ε_oos = weissman_epsilon(N_cal, S)
    σ = ε_oos / (2 * S)
    @printf("\n--- %s %s value_of_qtrue N_cal=%d ε_oos=%.4f σ=%.4f ---\n",
            network_name, scenario, N_cal, ε_oos, σ)
    flush(stdout)

    rng = MersenneTwister(42)
    costs = Dict(k => Vector{Float64}(undef, M) for k in keys_order)

    for r in 1:M
        p_true = sample_bental_normal(S, ε_oos, rng)
        for k in keys_order
            costs[k][r] = dot(p_true, flows_dict[k])
        end
    end

    fname = joinpath(@__DIR__, "plots",
                     @sprintf("%s_%s_%s_vqtrue_N%d.png", network_name, scenario, x_tag, N_cal))
    title = @sprintf("%s %s value_of_qtrue (N=%d, ε=%.3f)", network_name, scenario, N_cal, ε_oos)

    make_boxplot(costs, keys_order, labels, colors_map, title; savepath=fname)
    flush(stdout)
end

println("\n\nDone! $(now())")
