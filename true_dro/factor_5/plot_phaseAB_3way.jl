"""
plot_phaseAB_3way.jl — 3-solution phaseAB boxplot (nominal / single_l / double)
  PhaseA: symmetric Dir(β_dir), PhaseB: asymmetric α = β_dir·(1 + noise·randn)
  매 sample: q_true~Dir, q̃~Dir (독립), h*=follower(x,q̃), flows=maxflow(x,h*), cost=CVaR(q_true,flows)
  β_dir = 0.1, 0.3, 0.5, 1.0 각각 파일 1개
  CairoMakie로 hatching 패턴 boxplot 생성

Usage:
  julia plot_phaseAB_3way.jl <network> <scenario> <x1> <x2> <x3> [key=value...]

Keyword args:
  beta_risk=<float>  : CVaR risk level (default 0.4)
  eps=<float>        : ε value used in optimization (for output folder naming)
  oos_seed=<int>     : OOS sampling seed (default 42)
  out_folder=<str>   : custom output subfolder (overrides eps_X_beta_Y naming)
  label1/2/3=<str>   : solution labels

Example:
  julia plot_phaseAB_3way.jl grid5x5 factor "28,37" "28,31" "27,28" beta_risk=0.4 label1=nominal label2=single_l label3=double
"""

using JuMP, Printf, Statistics, Random, Distributions, LinearAlgebra, Serialization
using CairoMakie, LaTeXStrings

include("../../network_generator.jl")
using .NetworkGenerator
include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")

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

length(_pos_args) >= 5 || error("Usage: julia plot_phaseAB_3way.jl <network> <scenario> <x1> <x2> <x3> [key=value...]")

network_name = lowercase(_pos_args[1])
scenario = lowercase(_pos_args[2])
scenario in ("uniform", "factor") || error("Unknown scenario: $scenario")

β_risk = haskey(_kw_args, "beta_risk") ? parse(Float64, _kw_args["beta_risk"]) : 0.4
ε_val = haskey(_kw_args, "eps") ? parse(Float64, _kw_args["eps"]) : 0.2
oos_seed = haskey(_kw_args, "oos_seed") ? parse(Int, _kw_args["oos_seed"]) : 42
risk_tag = β_risk > 0.0 ? @sprintf("CVaR%.2f", β_risk) : "E"

# Output folder: eps_<X>_beta_<Y> (aligned with logs folder naming), or custom
function fmt_val(v)
    return replace(@sprintf("%.2g", v), "." => "p")
end
if haskey(_kw_args, "out_folder")
    out_subfolder = _kw_args["out_folder"]
else
    eps_str = fmt_val(ε_val)
    beta_str_folder = fmt_val(β_risk)
    out_subfolder = "eps_$(eps_str)_beta_$(beta_str_folder)"
end

# Parse 3 x solutions
x_arc_lists = [parse.(Int, split(_pos_args[i], ",")) for i in 3:5]
sol_labels = [get(_kw_args, "label$i", "[$(join(x_arc_lists[i], ","))]") for i in 1:3]

# ── visual settings (CairoMakie) ──
legend_labels = ["Nominal", "Partial", "Full"]
sol_colors = [Makie.wong_colors()[1], Makie.wong_colors()[2], Makie.wong_colors()[3]]
sol_patterns = [
    nothing,                                                                                          # Nominal: no pattern
    Makie.LinePattern(direction=Vec2f(1, 1); width=1, tilesize=(8, 8), linecolor=sol_colors[2]),      # Partial: / (orange)
    Makie.LinePattern(direction=Vec2f(1, 1); width=2, tilesize=(5, 5), linecolor=sol_colors[3]),      # Full: // (green)
]

# ── network setup (aligned with run_baseline_batch) ──
if network_name == "grid5x5"
    net = generate_grid_network(5, 5; seed=42)
    intd_arcs = net.interdictable_arcs
else
    gen = Dict(
        "abilene"     => generate_abilene_network,
        "nobel_us"    => generate_nobel_us_network,
        "sioux_falls" => generate_sioux_falls_network,
        "polska"      => generate_polska_network,
    )
    haskey(gen, network_name) || error("Unknown network: $network_name")
    net = gen[network_name]()
    intd_arcs = fill(true, length(net.arcs))
    net = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
        net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
end

num_arcs = length(net.arcs) - 1
γ = 2
S = 10

if scenario == "factor"
    caps, _ = generate_capacity_scenarios_factor_additive(length(net.arcs), S;
        interdictable_arcs=intd_arcs, seed=42, num_factors=5)
else
    caps, _ = generate_capacity_scenarios_uniform_model(length(net.arcs), S;
        interdictable_arcs=intd_arcs, seed=42)
end
intd_idx = findall(intd_arcs[1:num_arcs])
w = round(0.5 * γ * median(caps[intd_idx, :]); digits=4)
q_hat = fill(1.0 / S, S)

# v_scenarios: Bernoulli(0.75) per arc per scenario (aligned with run_baseline_batch)
Random.seed!(42)
v_rand = zeros(num_arcs, S)
for k in 1:num_arcs, s in 1:S
    v_rand[k, s] = intd_arcs[k] ? (rand() < 0.75 ? 1.0 : 0.0) : 0.0
end

# Build x vectors
xs = Vector{Vector{Float64}}(undef, 3)
for (i, arcs) in enumerate(x_arc_lists)
    xv = zeros(num_arcs)
    for a in arcs; xv[a] = 1.0; end
    xs[i] = xv
end

@printf("Network: %s, Scenario: %s, S=%d, γ=%d, w=%.4f, risk=%s\n", network_name, scenario, S, γ, w, risk_tag)
for i in 1:3
    @printf("  x%d (%s) = [%s]\n", i, sol_labels[i], join(x_arc_lists[i], ","))
end
flush(stdout)

# ── helper: vertical boxplot (CairoMakie) ──
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

# ── cost function (CVaR or expectation) ──
function compute_cost(q_true, flows, β_risk)
    if β_risk > 0.0
        return compute_cvar(q_true, flows, β_risk)
    else
        return dot(q_true, flows)
    end
end

# ── helper: compute flows for one (x, q̃) pair ──
function compute_flows_for_sample(net, x, v_rand, w, caps, q_tilde)
    h = solve_follower_weighted(net, x, v_rand, w, caps, q_tilde)
    return compute_maxflow_per_scenario(net, x, h, v_rand, caps)
end

# ── main loop ──
M = 500
β_dirs = [0.1, 0.3, 0.5, 1.0]
noises = [0.5, 1.0, 2.0, 5.0]
n_phases = 1 + length(noises)
spacing = 3.5   # x-axis spacing between phases

mkpath(joinpath(@__DIR__, "plots", out_subfolder))

# Collect all results for JLS serialization
all_results = Dict{Float64, Dict}()  # β_dir => {phase_label => {sol_label => Vector{Float64}}}

for β_dir in β_dirs
    phase_labels = [@sprintf("β=%.1f", β_dir); [@sprintf("ζ=%.1f", n) for n in noises]]
    display_labels = [latexstring("\\gamma=", @sprintf("%.1f", β_dir)); [latexstring("\\zeta_{\\mathrm{dir}}=", @sprintf("%.1f", n)) for n in noises]]
    @printf("\n--- β_dir=%.1f ---\n", β_dir)
    flush(stdout)

    # costs_phases[phase][solution] = Vector{Float64}(M)
    costs_phases = [Vector{Vector{Float64}}(undef, 3) for _ in 1:n_phases]

    # Phase A: symmetric Dirichlet — q_true, q̃ 독립 샘플
    rng = MersenneTwister(oos_seed)
    dir = Dirichlet(S, β_dir)
    for sol in 1:3
        costs_phases[1][sol] = Vector{Float64}(undef, M)
    end
    for m in 1:M
        q_true = rand(rng, dir)
        q_tilde = rand(rng, dir)
        for sol in 1:3
            flows = compute_flows_for_sample(net, xs[sol], v_rand, w, caps, q_tilde)
            costs_phases[1][sol][m] = compute_cost(q_true, flows, β_risk)
        end
        m % 100 == 0 && @printf("  PhaseA %d/%d\n", m, M)
    end

    # Phase B: asymmetric — q_true, q̃ 각각 독립 α 생성
    for (ni, noise) in enumerate(noises)
        rng = MersenneTwister(oos_seed)
        for sol in 1:3
            costs_phases[ni+1][sol] = Vector{Float64}(undef, M)
        end
        for m in 1:M
            # q_true
            α = β_dir .* (1.0 .+ noise * randn(rng, S))
            α = max.(α, 0.01)
            q_true = rand(rng, Dirichlet(α))
            # q̃ (독립 α)
            α_tilde = β_dir .* (1.0 .+ noise * randn(rng, S))
            α_tilde = max.(α_tilde, 0.01)
            q_tilde = rand(rng, Dirichlet(α_tilde))
            for sol in 1:3
                flows = compute_flows_for_sample(net, xs[sol], v_rand, w, caps, q_tilde)
                costs_phases[ni+1][sol][m] = compute_cost(q_true, flows, β_risk)
            end
            m % 100 == 0 && @printf("  ζ=%.1f %d/%d\n", noise, m, M)
        end
    end

    # ── Plot (CairoMakie) ──
    fig = Figure(size=(250*n_phases + 200, 500))
    ax = Axis(fig[1, 1],
        xlabel="", ylabel="Out-of-sample CVaR",
        xticks=([(i-1)*spacing for i in 1:n_phases], display_labels),
    )

    sol_offset = [-0.5, 0.0, 0.5]
    for (pi, _) in enumerate(phase_labels)
        xc = (pi - 1) * spacing
        for sol in 1:3
            draw_vbox!(ax, xc + sol_offset[sol], costs_phases[pi][sol], sol_colors[sol], sol_patterns[sol];
                       box_w=0.2, cap_w=0.1)
        end
    end

    # Legend
    legend_elements = [
        [PolyElement(color=(sol_colors[1], 0.15), strokecolor=sol_colors[1], strokewidth=2)],
        [PolyElement(color=sol_patterns[2], strokecolor=sol_colors[2], strokewidth=2)],
        [PolyElement(color=sol_patterns[3], strokecolor=sol_colors[3], strokewidth=2)],
    ]
    Legend(fig[1, 2], legend_elements, legend_labels, framevisible=true, labelsize=11)

    β_str = replace(@sprintf("%.1f", β_dir), "." => "p")
    savepath = joinpath(@__DIR__, "plots", out_subfolder,
                        @sprintf("%s_%s_beta%s_phaseAB.png", network_name, scenario, β_str))
    save(savepath, fig, px_per_unit=2)
    @printf("  Saved: %s\n", savepath)

    # Save to all_results
    β_result = Dict{String, Dict{String, Vector{Float64}}}()
    for (pi, plabel) in enumerate(phase_labels)
        phase_dict = Dict{String, Vector{Float64}}()
        for sol in 1:3
            phase_dict[sol_labels[sol]] = costs_phases[pi][sol]
        end
        β_result[plabel] = phase_dict
    end
    all_results[β_dir] = β_result

    # Print stats
    for (pi, plabel) in enumerate(display_labels)
        @printf("  [%s] ", plabel)
        for sol in 1:3
            @printf("%s=%.4f  ", sol_labels[sol], mean(costs_phases[pi][sol]))
        end
        println()
    end
    flush(stdout)
end

# Save JLS
jls_path = joinpath(@__DIR__, "plots", out_subfolder,
                     "$(network_name)_$(scenario)_phaseAB.jls")
serialize(jls_path, Dict(
    :network => network_name,
    :scenario => scenario,
    :β_risk => β_risk,
    :ε => ε_val,
    :risk_tag => risk_tag,
    :x_arcs => x_arc_lists,
    :sol_labels => sol_labels,
    :β_dirs => β_dirs,
    :noises => noises,
    :M => M,
    :oos_seed => oos_seed,
    :results => all_results,
))
@printf("Saved JLS: %s\n", jls_path)

println("\nDone!")
