# plot_free_vs_tied.jl — grid3x3 free vs tied α visualization
# x=[1,9,10,11,14], γ=5, random v, β=0.95

using Plots, Printf
include("../network_generator.jl"); NG = NetworkGenerator

net = NG.generate_grid_network(3, 3; seed=42)
K = length(net.arcs) - 1

# 모든 arc interdictable
intd_all = fill(true, length(net.arcs))
net = NG.GridNetworkData(net.m, net.n, net.nodes, net.arcs, net.N,
    intd_all, net.arc_directions, net.arc_adjacency, net.node_arc_incidence)

x_bar = zeros(K); for k in [1,9,10,11,14]; x_bar[k] = 1.0; end

# Results from experiment
α_free = zeros(K); α_free[1]=1.0; α_free[11]=4.0; α_free[14]=4.0
α_tied = zeros(K); α_tied[1]=0.46; α_tied[2]=0.41; α_tied[9]=1.36; α_tied[10]=6.51; α_tied[11]=0.13; α_tied[14]=0.13

# Node positions
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

function draw_network_alpha(net, pos, x_star, α_vals, title_str;
                            savepath=nothing, w=9.0)
    num_arcs = length(net.arcs) - 1
    m, n = net.m, net.n

    p = plot(size=(550, 420), legend=false, grid=false,
             xlims=(-0.8, n + 1.8), ylims=(-0.5, m + 1.5),
             aspect_ratio=:equal, axis=false, ticks=false,
             background_color=:white, foreground_color=:black,
             title=title_str, titlefontsize=11)

    # Draw arcs
    for i in 1:num_arcs
        from, to = net.arcs[i]
        x1, y1 = pos[from]
        x2, y2 = pos[to]

        is_interdicted = x_star[i] > 0.5
        has_alpha = α_vals[i] > 0.01

        dx = x2 - x1; dy = y2 - y1
        len = sqrt(dx^2 + dy^2)
        if len > 0
            shrink = 0.22
            ux, uy = dx/len, dy/len
            ax1 = x1 + shrink * ux; ay1 = y1 + shrink * uy
            ax2 = x2 - shrink * ux; ay2 = y2 - shrink * uy
        else
            ax1, ay1, ax2, ay2 = x1, y1, x2, y2
        end

        if is_interdicted && has_alpha
            # interdicted + α > 0: thick blue (recovery)
            lw = 1.5 + 3.0 * α_vals[i] / w
            plot!(p, [ax1, ax2], [ay1, ay2], color=:blue, linewidth=lw, linestyle=:solid)
        elseif is_interdicted
            # interdicted, no recovery: red dashed
            plot!(p, [ax1, ax2], [ay1, ay2], color=:red, linewidth=2.0, linestyle=:dash)
        else
            # normal arc
            plot!(p, [ax1, ax2], [ay1, ay2], color=:gray70, linewidth=0.7)
        end

        # Arrowhead
        if len > 0
            t = 0.55
            mx = ax1 + t * (ax2 - ax1)
            my = ay1 + t * (ay2 - ay1)
            arrow_len = 0.10
            udx = ux * arrow_len; udy = uy * arrow_len
            ang = pi/7
            px = udx * cos(ang) - udy * sin(ang)
            py = udx * sin(ang) + udy * cos(ang)
            qx = udx * cos(-ang) - udy * sin(-ang)
            qy = udx * sin(-ang) + udy * cos(-ang)

            acol = is_interdicted ? (has_alpha ? :blue : :red) : :gray70
            plot!(p, Shape([mx, mx - px, mx - qx], [my, my - py, my - qy]),
                  fillcolor=acol, linecolor=acol)
        end

        # α label on arc midpoint (if > 0.01)
        if α_vals[i] > 0.01
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            # offset perpendicular to arc
            if len > 0
                offx = -uy * 0.18
                offy = ux * 0.18
            else
                offx, offy = 0.0, 0.15
            end
            annotate!(p, mx + offx, my + offy,
                      text(@sprintf("%.1f", α_vals[i]), 8, :center, :blue))
        end
    end

    # Draw nodes
    for node in net.nodes
        if node == "t" || node == "s"; continue; end
        x, y = pos[node]
        scatter!(p, [x], [y], color=:white, markersize=14,
                 markerstrokewidth=1.0, markerstrokecolor=:black)
        # label
        short = replace(replace(node, "node_" => ""), "_" => ",")
        annotate!(p, x, y, text(short, 7, :center))
    end

    # Source and sink
    sx, sy = pos["s"]; tx, ty = pos["t"]
    scatter!(p, [sx], [sy], color=:lightyellow, markersize=18,
             markerstrokewidth=1.5, markerstrokecolor=:black)
    annotate!(p, sx, sy, text("s", 12, :center, :bold))
    scatter!(p, [tx], [ty], color=:lightyellow, markersize=18,
             markerstrokewidth=1.5, markerstrokecolor=:black)
    annotate!(p, tx, ty, text("t", 12, :center, :bold))

    if savepath !== nothing
        savefig(p, savepath)
        println("Saved: $savepath")
    end
    return p
end

# Draw both
p1 = draw_network_alpha(net, pos, x_bar, α_free,
    "FREE (a≠d): Z=14, α=[k1=1, k11=4, k14=4]";
    savepath="plot_free_alpha.png")

p2 = draw_network_alpha(net, pos, x_bar, α_tied,
    "TIED (a=d): Z=10, α=[k1=0.5, k9=1.4, k10=6.5, ...]";
    savepath="plot_tied_alpha.png")

# Combined
p_both = plot(p1, p2, layout=(1, 2), size=(1100, 450))
savefig(p_both, "plot_free_vs_tied.png")
println("Saved: plot_free_vs_tied.png")
