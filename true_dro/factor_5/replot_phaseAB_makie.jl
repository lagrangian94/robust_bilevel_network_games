"""
replot_phaseAB_makie.jl — Re-plot phaseAB boxplots from JLS using CairoMakie (with hatching)

Usage:
  julia replot_phaseAB_makie.jl <jls_path> [key=value...]

Keyword args:
  out_folder=<str>  : output folder (default: same as JLS folder)

Patterns:
  Nominal = empty (no fill)
  Partial = /  (diagonal lines)
  Full    = // (dense diagonal lines)
"""

using CairoMakie, LaTeXStrings, Serialization, Statistics, Printf

# ── parse arguments ──
_kw_args = Dict{String,String}()
_pos_args = String[]
for arg in ARGS
    if occursin("=", arg)
        k, v = split(arg, "="; limit=2)
        _kw_args[lowercase(k)] = v
    else
        push!(_pos_args, arg)
    end
end

length(_pos_args) >= 1 || error("Usage: julia replot_phaseAB_makie.jl <jls_path> [key=value...]")
jls_path = _pos_args[1]
isfile(jls_path) || error("JLS file not found: $jls_path")

# Load data
d = deserialize(jls_path)
network_name = d[:network]
scenario = d[:scenario]
β_risk = d[:β_risk]
risk_tag = d[:risk_tag]
sol_labels = d[:sol_labels]
β_dirs = d[:β_dirs]
noises = get(d, :noises, [0.5, 1.0, 2.0, 5.0])
M = d[:M]
results = d[:results]

# Output folder
out_dir = get(_kw_args, "out_folder", dirname(jls_path))
mkpath(out_dir)

# Display labels
legend_labels = ["Nominal", "Partial", "Full"]
sol_colors = [Makie.wong_colors()[1], Makie.wong_colors()[2], Makie.wong_colors()[3]]

# Patterns: empty, sparse /, dense /
sol_patterns = [
    nothing,                                                                                          # Nominal: no pattern
    Makie.LinePattern(direction=Vec2f(1, 1); width=1, tilesize=(8, 8), linecolor=sol_colors[2]),      # Partial: / (orange)
    Makie.LinePattern(direction=Vec2f(1, 1); width=2, tilesize=(5, 5), linecolor=sol_colors[3]),      # Full: // (green)
]

# ── helper: draw vertical boxplot ──
function draw_vbox!(ax, xi, data, col, pattern;
                    whisker_q=(0.05, 0.95), box_w=0.25, cap_w=0.12)
    med = median(data)
    q25 = quantile(data, 0.25)
    q75 = quantile(data, 0.75)
    wlo = quantile(data, whisker_q[1])
    whi = quantile(data, whisker_q[2])
    mean_val = mean(data)

    # Whiskers (dashed)
    lines!(ax, [xi, xi], [wlo, q25], color=col, linewidth=1.5, linestyle=:dash)
    lines!(ax, [xi, xi], [q75, whi], color=col, linewidth=1.5, linestyle=:dash)
    # Caps
    lines!(ax, [xi-cap_w, xi+cap_w], [wlo, wlo], color=col, linewidth=2)
    lines!(ax, [xi-cap_w, xi+cap_w], [whi, whi], color=col, linewidth=2)
    # Box
    box_xs = [xi-box_w, xi+box_w, xi+box_w, xi-box_w]
    box_ys = [q25, q25, q75, q75]
    if pattern === nothing
        poly!(ax, Point2f.(zip(box_xs, box_ys)), color=(col, 0.15), strokecolor=col, strokewidth=2)
    else
        poly!(ax, Point2f.(zip(box_xs, box_ys)), color=pattern, strokecolor=col, strokewidth=2)
    end
    # Median line
    lines!(ax, [xi-box_w, xi+box_w], [med, med], color=col, linewidth=3)
    # Mean diamond
    scatter!(ax, [xi], [mean_val], color=col, marker=:diamond, markersize=10)

    return (; median=med, q25, q75, wlo, whi, mean_val)
end

# ── plot per β_dir ──
spacing = 3.5
for β_dir in β_dirs
    # Display labels (γ/ζ); detect JLS key format (old: "PhaseA"/"n=X", new: "β=X"/"ζ=X")
    display_labels = [latexstring("\\gamma=", @sprintf("%.1f", β_dir)); [latexstring("\\zeta_{\\mathrm{dir}}=", @sprintf("%.1f", n)) for n in noises]]
    β_result = results[β_dir]
    first_key = first(keys(β_result))
    if startswith(first_key, "PhaseA") || startswith(first_key, "n=")
        jls_keys = ["PhaseA"; [@sprintf("n=%.1f", n) for n in noises]]
    else
        jls_keys = [@sprintf("β=%.1f", β_dir); [@sprintf("ζ=%.1f", n) for n in noises]]
    end
    n_phases = length(display_labels)

    fig = Figure(size=(250*n_phases + 200, 500))
    ax = Axis(fig[1, 1],
        xlabel="", ylabel="Out-of-sample CVaR",
        xticks=([(i-1)*spacing for i in 1:n_phases], display_labels),
    )

    sol_offset = [-0.5, 0.0, 0.5]
    for (pi, plabel) in enumerate(jls_keys)
        xc = (pi - 1) * spacing
        phase_dict = β_result[plabel]
        for sol in 1:3
            data = phase_dict[sol_labels[sol]]
            draw_vbox!(ax, xc + sol_offset[sol], data, sol_colors[sol], sol_patterns[sol];
                       box_w=0.2, cap_w=0.1)
        end
    end

    # Legend (manual)
    legend_elements = [
        [PolyElement(color=(sol_colors[1], 0.15), strokecolor=sol_colors[1], strokewidth=2)],
        [PolyElement(color=sol_patterns[2], strokecolor=sol_colors[2], strokewidth=2)],
        [PolyElement(color=sol_patterns[3], strokecolor=sol_colors[3], strokewidth=2)],
    ]
    Legend(fig[1, 2], legend_elements, legend_labels, framevisible=true, labelsize=11)

    β_str = replace(@sprintf("%.1f", β_dir), "." => "p")
    savepath = joinpath(out_dir, @sprintf("%s_%s_beta%s_phaseAB.png", network_name, scenario, β_str))
    save(savepath, fig, px_per_unit=2)
    @printf("  Saved: %s\n", savepath)
end

println("\nDone!")
