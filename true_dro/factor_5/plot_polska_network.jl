"""
plot_polska_network.jl — Polska network visualization.
  Nominal([3,6]) vs Robust([3,34]): interdicted arc = dashed red, recovery = circle
  γ=2, uniform, S=20
"""

using JuMP, HiGHS, Gurobi, Printf, Dates, Statistics, Random, Distributions, LinearAlgebra
using Plots

include("../../network_generator.jl")
using .NetworkGenerator

include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")

# ── Network setup ──
net = generate_polska_network()
num_arcs_total = length(net.arcs)
num_arcs = num_arcs_total - 1
all_intd = fill(true, num_arcs_total)
net = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
    net.N, all_intd, net.arc_adjacency, net.node_arc_incidence)

caps, _ = generate_capacity_scenarios_uniform_model(num_arcs_total, 20;
    interdictable_arcs=all_intd, seed=42)
intd_idx = findall(all_intd[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
v = 1.0

x_nom = zeros(Float64, num_arcs); x_nom[3] = 1.0; x_nom[6] = 1.0
x_rob = zeros(Float64, num_arcs); x_rob[18] = 1.0; x_rob[33] = 1.0

# Print arc info
println("Interdicted arcs:")
for (label, xv) in [("Nominal", x_nom), ("Robust", x_rob)]
    idxs = findall(xv .> 0.5)
    for i in idxs
        println("  $label arc $i: $(net.arcs[i])")
    end
end
println()

# ── Find one instance where robust wins ──
function find_rob_win(net, x_nom, x_rob, v, w, caps)
    K = size(caps, 2)
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
println("h_nom (nonzero): ", [(i, net.arcs[i], round(win.h_nom[i]; digits=3)) for i in 1:num_arcs if win.h_nom[i] > 0.01])
println("h_rob (nonzero): ", [(i, net.arcs[i], round(win.h_rob[i]; digits=3)) for i in 1:num_arcs if win.h_rob[i] > 0.01])

# ── Node positions (approximate geographic coordinates of Polish cities) ──
# x ~ longitude (E), y ~ latitude (N), scaled for nice layout
node_pos = Dict{String, Tuple{Float64, Float64}}(
    "s"           => (0.5, 5.5),    # Szczecin (source) — far left
    "Kolobrzeg"   => (2.0, 6.0),
    "Gdansk"      => (3.5, 6.5),
    "Bydgoszcz"   => (3.0, 5.0),
    "Poznan"      => (2.0, 4.0),
    "Szczecin"    => (0.5, 5.5),    # same as "s"
    "Bialystok"   => (5.5, 5.5),
    "Warsaw"      => (4.5, 4.0),
    "Lodz"        => (3.5, 3.0),
    "Wroclaw"     => (2.0, 2.0),
    "Katowice"    => (3.0, 1.0),
    "Krakow"      => (4.0, 0.5),
    "Rzeszow"     => (5.5, 0.5),
    "t"           => (6.5, 0.5),    # Rzeszow (sink) — far right
)

# ── Draw function ──
function draw_polska(net, pos, x_star, h_star, title_str; savepath=nothing)
    num_arcs = length(net.arcs) - 1

    p = plot(size=(750, 550), legend=false, grid=false,
             xlims=(-0.5, 7.5), ylims=(-0.8, 7.5),
             aspect_ratio=:equal, axis=false, ticks=false,
             background_color=:white, foreground_color=:black,
             title=title_str, titlefontsize=12)

    # Draw arcs
    for i in 1:num_arcs
        from, to = net.arcs[i]
        if !haskey(pos, from) || !haskey(pos, to)
            continue
        end
        x1, y1 = pos[from]
        x2, y2 = pos[to]

        is_interdicted = x_star[i] > 0.5

        dx = x2 - x1; dy = y2 - y1
        len = sqrt(dx^2 + dy^2)
        if len > 0
            shrink = 0.25
            ux, uy = dx/len, dy/len
            ax1 = x1 + shrink * ux; ay1 = y1 + shrink * uy
            ax2 = x2 - shrink * ux; ay2 = y2 - shrink * uy
        else
            ax1, ay1, ax2, ay2 = x1, y1, x2, y2
            ux, uy = 0.0, 0.0
        end

        if is_interdicted
            plot!(p, [ax1, ax2], [ay1, ay2], color=:red, linewidth=3.0, linestyle=:dash)
        else
            plot!(p, [ax1, ax2], [ay1, ay2], color=:gray60, linewidth=0.8)
        end

        # Arrowhead
        if len > 0
            t = 0.55
            mx = ax1 + t * (ax2 - ax1)
            my = ay1 + t * (ay2 - ay1)
            arrow_len = 0.12
            udx = ux * arrow_len; udy = uy * arrow_len
            ang = pi/7
            px = udx * cos(ang) - udy * sin(ang)
            py = udx * sin(ang) + udy * cos(ang)
            qx = udx * cos(-ang) - udy * sin(-ang)
            qy = udx * sin(-ang) + udy * cos(-ang)
            acol = is_interdicted ? :red : :gray60
            plot!(p, Shape([mx, mx - px, mx - qx], [my, my - py, my - qy]),
                  fillcolor=acol, linecolor=acol)
        end
    end

    # Draw recovery (h > 0)
    h_max = maximum(h_star)
    for i in 1:num_arcs
        if h_star[i] > 0.01
            from, to = net.arcs[i]
            if !haskey(pos, from) || !haskey(pos, to); continue; end
            x1, y1 = pos[from]
            x2, y2 = pos[to]
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            ms = h_max > 0 ? 4 + 8 * h_star[i] / h_max : 6
            scatter!(p, [mx], [my], color=:green, markersize=ms,
                     markerstrokewidth=0.5, markerstrokecolor=:darkgreen,
                     marker=:circle, alpha=0.7)
        end
    end

    # Draw nodes
    orig = net.original_node_names
    for node in net.nodes
        if !haskey(pos, node); continue; end
        x, y = pos[node]
        if node == "s"
            scatter!(p, [x], [y], color=:lightblue, markersize=20,
                     markerstrokewidth=1.5, markerstrokecolor=:black)
            annotate!(p, x, y - 0.45, text("s (Szczecin)", 7, :center, :bold))
        elseif node == "t"
            scatter!(p, [x], [y], color=:lightyellow, markersize=20,
                     markerstrokewidth=1.5, markerstrokecolor=:black)
            annotate!(p, x, y - 0.45, text("t (Rzeszow)", 7, :center, :bold))
        else
            scatter!(p, [x], [y], color=:white, markersize=16,
                     markerstrokewidth=1.0, markerstrokecolor=:black)
            label = haskey(orig, node) ? orig[node] : node
            annotate!(p, x, y - 0.4, text(label, 7, :center))
        end
    end

    if savepath !== nothing
        savefig(p, savepath)
        println("  Saved: $savepath")
    end
    return p
end

# ── Draw both ──
mkpath(joinpath(@__DIR__, "plots"))

p_nom = draw_polska(net, node_pos, x_nom, win.h_nom,
    @sprintf("Nominal x=[3,6] (cost=%.2f)", win.cost_nom);
    savepath=joinpath(@__DIR__, "plots", "polska_interdiction_nom.png"))

p_rob = draw_polska(net, node_pos, x_rob, win.h_rob,
    @sprintf("Robust x=[18,33] (cost=%.2f)", win.cost_rob);
    savepath=joinpath(@__DIR__, "plots", "polska_interdiction_rob.png"))

println("\nDone! $(now())")
