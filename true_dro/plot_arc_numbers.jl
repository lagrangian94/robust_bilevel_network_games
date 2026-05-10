# plot_arc_numbers.jl — grid3x3 arc numbering visualization
using Plots, Printf
include("../network_generator.jl"); NG = NetworkGenerator

net = NG.generate_grid_network(3, 3; seed=42)
K = length(net.arcs) - 1

function get_node_pos(net)
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

p = plot(size=(700, 550), legend=false, grid=false,
         xlims=(-0.8, 4.8), ylims=(-0.3, 4.5),
         aspect_ratio=:equal, axis=false, ticks=false,
         background_color=:white, foreground_color=:black,
         title="Grid 3x3 — Arc Numbering (K=$K)", titlefontsize=13)

# Draw arcs with numbers
for i in 1:K
    from, to = net.arcs[i]
    x1, y1 = pos[from]
    x2, y2 = pos[to]

    dx = x2 - x1; dy = y2 - y1
    len = sqrt(dx^2 + dy^2)
    if len > 0
        shrink = 0.25
        ux, uy = dx/len, dy/len
        ax1 = x1 + shrink * ux; ay1 = y1 + shrink * uy
        ax2 = x2 - shrink * ux; ay2 = y2 - shrink * uy
    else
        continue
    end

    plot!(p, [ax1, ax2], [ay1, ay2], color=:gray50, linewidth=1.0)

    # Arrowhead
    t = 0.55
    mx = ax1 + t * (ax2 - ax1)
    my = ay1 + t * (ay2 - ay1)
    arrow_len = 0.08
    udx = ux * arrow_len; udy = uy * arrow_len
    ang = pi/7
    px2 = udx * cos(ang) - udy * sin(ang)
    py2 = udx * sin(ang) + udy * cos(ang)
    qx = udx * cos(-ang) - udy * sin(-ang)
    qy = udx * sin(-ang) + udy * cos(-ang)
    plot!(p, Shape([mx, mx - px2, mx - qx], [my, my - py2, my - qy]),
          fillcolor=:gray50, linecolor=:gray50)

    # Arc number label — offset perpendicular
    lx = (x1 + x2) / 2
    ly = (y1 + y2) / 2
    offx = -uy * 0.20
    offy = ux * 0.20
    annotate!(p, lx + offx, ly + offy,
              text(string(i), 9, :center, :red))
end

# Draw nodes
for node in net.nodes
    if node == "t" || node == "s"; continue; end
    x, y = pos[node]
    scatter!(p, [x], [y], color=:white, markersize=16,
             markerstrokewidth=1.2, markerstrokecolor=:black)
    short = replace(replace(node, "node_" => ""), "_" => ",")
    annotate!(p, x, y, text(short, 8, :center))
end

# Source and sink
sx, sy = pos["s"]; tx, ty = pos["t"]
scatter!(p, [sx], [sy], color=:lightyellow, markersize=20,
         markerstrokewidth=1.5, markerstrokecolor=:black)
annotate!(p, sx, sy, text("s", 13, :center))
scatter!(p, [tx], [ty], color=:lightyellow, markersize=20,
         markerstrokewidth=1.5, markerstrokecolor=:black)
annotate!(p, tx, ty, text("t", 13, :center))

savefig(p, "plot_arc_numbers.png")
println("Saved: plot_arc_numbers.png")
