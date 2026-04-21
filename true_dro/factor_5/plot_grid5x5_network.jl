"""
plot_grid5x5_network.jl — Grid 5x5 network visualization.

(1) Network graph with node positions
(2) Nominal vs Robust: interdicted arc = dashed red, recovery = circle markers
(3) One instance where robust wins
"""

using JuMP, HiGHS, Gurobi, Printf, Dates, Statistics, Random, Distributions, LinearAlgebra
using Plots

include("../../network_generator.jl")
using .NetworkGenerator

include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")

# ============================================================
# Setup — hardcoded network from original experiment (log)
# ============================================================
arcs_log = [
    ("s","node_1_1"), ("s","node_2_1"), ("s","node_3_1"), ("s","node_4_1"), ("s","node_5_1"),
    ("node_1_1","node_2_1"), ("node_2_1","node_1_1"), ("node_3_1","node_4_1"), ("node_4_1","node_5_1"),
    ("node_2_2","node_3_2"), ("node_3_2","node_4_2"), ("node_4_2","node_3_2"),
    ("node_1_3","node_2_3"), ("node_2_3","node_3_3"), ("node_3_3","node_2_3"), ("node_4_3","node_5_3"),
    ("node_1_4","node_2_4"), ("node_2_4","node_1_4"), ("node_3_4","node_2_4"), ("node_4_4","node_5_4"),
    ("node_1_5","node_2_5"), ("node_2_5","node_1_5"), ("node_3_5","node_2_5"), ("node_4_5","node_5_5"),
    ("node_1_1","node_1_2"), ("node_2_1","node_2_2"), ("node_3_1","node_3_2"), ("node_4_1","node_4_2"), ("node_5_1","node_5_2"),
    ("node_1_2","node_1_3"), ("node_2_2","node_2_3"), ("node_3_2","node_3_3"), ("node_4_2","node_4_3"), ("node_5_2","node_5_3"),
    ("node_1_3","node_1_4"), ("node_2_3","node_2_4"), ("node_3_3","node_3_4"), ("node_4_3","node_4_4"), ("node_5_3","node_5_4"),
    ("node_1_4","node_1_5"), ("node_2_4","node_2_5"), ("node_3_4","node_3_5"), ("node_4_4","node_4_5"), ("node_5_4","node_5_5"),
    ("node_1_5","t"), ("node_2_5","t"), ("node_3_5","t"), ("node_4_5","t"), ("node_5_5","t"),
    ("t","s")
]
intd_log = [
    false, false, false, false, false,
    false, false, false, false,
    true, true, true,
    true, true, true, true,
    true, true, true, true,
    false, false, false, false,
    true, true, true, true, true,
    true, true, true, true, true,
    true, true, true, true, true,
    false, false, false, false, false,
    false, false, false, false, false,
    false
]

nodes_log = ["s"]
for col in 1:5, row in 1:5
    push!(nodes_log, "node_$(row)_$(col)")
end
push!(nodes_log, "t")

num_nodes = length(nodes_log)
num_arcs_total = length(arcs_log)
num_regular_arcs = num_arcs_total - 1

# Node-incidence matrix N: (|V|-1) × |A|
N_mat = zeros(Float64, num_nodes - 1, num_arcs_total)
node_to_idx = Dict(nd => idx for (idx, nd) in enumerate(nodes_log))
for (arc_idx, (from_nd, to_nd)) in enumerate(arcs_log)
    fi = node_to_idx[from_nd]
    ti = node_to_idx[to_nd]
    if fi > 1; N_mat[fi - 1, arc_idx] = 1.0; end
    if ti > 1; N_mat[ti - 1, arc_idx] = -1.0; end
end

arc_adj = NetworkGenerator.generate_arc_adjacency(arcs_log, num_regular_arcs)
node_arc_inc = NetworkGenerator.generate_node_arc_incidence(nodes_log, arcs_log, num_nodes - 1, num_regular_arcs)

net = GridNetworkData(5, 5, nodes_log, arcs_log, N_mat, intd_log, Int[], arc_adj, node_arc_inc)
num_arcs = num_regular_arcs

caps, _ = generate_capacity_scenarios_factor_sparse(length(net.arcs), 20;
    interdictable_arcs=net.interdictable_arcs, seed=42, num_factors=5)
intd_idx = findall(net.interdictable_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
v = 1.0

x_nom = zeros(Float64, num_arcs); x_nom[22] = 1.0
x_rob = zeros(Float64, num_arcs); x_rob[25] = 1.0

# Print arc info
println("Arc 22: $(net.arcs[22]), interdictable=$(net.interdictable_arcs[22])")
println("Arc 25: $(net.arcs[25]), interdictable=$(net.interdictable_arcs[25])")
println()

# Print all arcs
println("All arcs:")
for (i, arc) in enumerate(net.arcs[1:num_arcs])
    intd = net.interdictable_arcs[i] ? " [INTD]" : ""
    nom = x_nom[i] > 0.5 ? " ← NOM" : ""
    rob = x_rob[i] > 0.5 ? " ← ROB" : ""
    println("  Arc $i: $(arc[1]) → $(arc[2])$intd$nom$rob")
end

# ============================================================
# Find one instance where robust wins
# ============================================================
function find_rob_win(net, x_nom, x_rob, v, w, caps)
    K = size(caps, 2)
    num_arcs = length(net.arcs) - 1

    # Try p_center approach (robust always wins here)
    rng = MersenneTwister(42)
    for m in 1:500
        α = 1.0 .* (1.0 .+ 0.5 * randn(rng, K))
        α = max.(α, 0.01)
        p_center = α / sum(α)

        h_nom = solve_follower_weighted(net, x_nom, v, w, caps, p_center)
        flows_nom = compute_maxflow_per_scenario(net, x_nom, h_nom, v, caps)
        cost_nom = dot(p_center, flows_nom)

        h_rob = solve_follower_weighted(net, x_rob, v, w, caps, p_center)
        flows_rob = compute_maxflow_per_scenario(net, x_rob, h_rob, v, caps)
        cost_rob = dot(p_center, flows_rob)

        if cost_rob < cost_nom
            return (idx=m, qtilde=p_center, ptrue=p_center,
                    h_nom=h_nom, h_rob=h_rob,
                    cost_nom=cost_nom, cost_rob=cost_rob)
        end
    end
    error("No robust win found in 500 samples")
end

win = find_rob_win(net, x_nom, x_rob, v, w, caps)

@printf("Robust wins at sample %d: cost_nom=%.4f, cost_rob=%.4f, gap=%.4f\n",
        win.idx, win.cost_nom, win.cost_rob, win.cost_rob - win.cost_nom)
println("h_nom (nonzero): ", [(i, round(win.h_nom[i]; digits=3)) for i in 1:num_arcs if win.h_nom[i] > 0.01])
println("h_rob (nonzero): ", [(i, round(win.h_rob[i]; digits=3)) for i in 1:num_arcs if win.h_rob[i] > 0.01])

# ============================================================
# Node positions
# ============================================================
function get_node_pos(net::GridNetworkData)
    pos = Dict{String, Tuple{Float64, Float64}}()
    m, n = net.m, net.n
    pos["s"] = (0.0, (m + 1) / 2.0)
    pos["t"] = (Float64(n + 1), (m + 1) / 2.0)
    for col in 1:n, row in 1:m
        pos["node_$(row)_$(col)"] = (Float64(col), Float64(m + 1 - row))
    end
    return pos
end

pos = get_node_pos(net)

# ============================================================
# Draw network
# ============================================================
function draw_network(net, pos, x_star, h_star, title_str;
                       savepath=nothing)
    num_arcs = length(net.arcs) - 1
    m, n = net.m, net.n

    p = plot(size=(650, 480), legend=false, grid=false,
             xlims=(-0.7, n + 1.7), ylims=(-0.2, m + 1.2),
             aspect_ratio=:equal, axis=false, ticks=false,
             background_color=:white, foreground_color=:black)

    # Draw arcs (edges first, nodes on top)
    for i in 1:num_arcs
        from, to = net.arcs[i]
        x1, y1 = pos[from]
        x2, y2 = pos[to]

        is_interdicted = x_star[i] > 0.5

        # Shorten line slightly so it doesn't overlap node circles
        dx = x2 - x1; dy = y2 - y1
        len = sqrt(dx^2 + dy^2)
        if len > 0
            shrink = 0.18
            ux, uy = dx/len, dy/len
            ax1 = x1 + shrink * ux; ay1 = y1 + shrink * uy
            ax2 = x2 - shrink * ux; ay2 = y2 - shrink * uy
        else
            ax1, ay1, ax2, ay2 = x1, y1, x2, y2
        end

        if is_interdicted
            plot!(p, [ax1, ax2], [ay1, ay2], color=:red, linewidth=2.5, linestyle=:dash)
        else
            plot!(p, [ax1, ax2], [ay1, ay2], color=:black, linewidth=0.8)
        end

        # Arrowhead at 60% along the arc
        if len > 0
            t = 0.6
            mx = ax1 + t * (ax2 - ax1)
            my = ay1 + t * (ay2 - ay1)
            arrow_len = 0.10
            udx = ux * arrow_len
            udy = uy * arrow_len
            ang = pi/7
            px = udx * cos(ang) - udy * sin(ang)
            py = udx * sin(ang) + udy * cos(ang)
            qx = udx * cos(-ang) - udy * sin(-ang)
            qy = udx * sin(-ang) + udy * cos(-ang)

            acol = is_interdicted ? :red : :black
            plot!(p, Shape([mx, mx - px, mx - qx], [my, my - py, my - qy]),
                  fillcolor=acol, linecolor=acol)
        end
    end

    # Draw recovery (h > 0) — filled circle on arc midpoint
    for i in 1:num_arcs
        if h_star[i] > 0.01
            from, to = net.arcs[i]
            x1, y1 = pos[from]
            x2, y2 = pos[to]
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2

            ms = 3 + 6 * h_star[i] / maximum(h_star)
            scatter!(p, [mx], [my], color=:black, markersize=ms,
                     markerstrokewidth=0, marker=:circle, alpha=0.6)
        end
    end

    # Draw nodes — clean filled circles
    for node in net.nodes
        if node == "t" || node == "s"
            continue
        end
        x, y = pos[node]
        scatter!(p, [x], [y], color=:white, markersize=16,
                 markerstrokewidth=1.0, markerstrokecolor=:black)
    end

    # Source and sink
    sx, sy = pos["s"]
    tx, ty = pos["t"]
    scatter!(p, [sx], [sy], color=:white, markersize=18,
             markerstrokewidth=1.0, markerstrokecolor=:black)
    annotate!(p, sx, sy, text("s", 12, :center))

    scatter!(p, [tx], [ty], color=:white, markersize=18,
             markerstrokewidth=1.0, markerstrokecolor=:black)
    annotate!(p, tx, ty, text("t", 12, :center))

    if savepath !== nothing
        savefig(p, savepath)
        println("  Saved: $savepath")
    end
    return p
end

# ============================================================
# Draw both
# ============================================================
p_nom = draw_network(net, pos, x_nom, win.h_nom,
    @sprintf("Nominal (x=[22], cost=%.2f)", win.cost_nom);
    savepath="true_dro/factor_5/plots/grid5x5_network_nominal.png")

p_rob = draw_network(net, pos, x_rob, win.h_rob,
    @sprintf("Robust (x=[25], cost=%.2f)", win.cost_rob);
    savepath="true_dro/factor_5/plots/grid5x5_network_robust.png")

println("\nDone! $(now())")
