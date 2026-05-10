"""
plot_phaseAB_3way.jl — 3-solution phaseAB boxplot (nominal / single_l / double)
  PhaseA: symmetric Dir(β_dir), PhaseB: asymmetric α = β_dir·(1 + noise·randn)
  매 sample: q_true~Dir, q̃~Dir (독립), h*=follower(x,q̃), flows=maxflow(x,h*), cost=CVaR(q_true,flows)
  β_dir = 0.1, 0.3, 0.5, 1.0 각각 파일 1개

Usage:
  julia plot_phaseAB_3way.jl <network> <scenario> <x1> <x2> <x3> [key=value...]

Keyword args:
  beta_risk=<float>  : CVaR risk level (default 0.4)
  eps=<float>        : ε value used in optimization (for output folder naming)
  label1/2/3=<str>   : solution labels

Example:
  julia plot_phaseAB_3way.jl grid5x5 factor "28,37" "28,31" "27,28" beta_risk=0.4 label1=nominal label2=single_l label3=double
"""

using JuMP, HiGHS, Printf, Statistics, Random, Distributions, LinearAlgebra
using Plots

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
risk_tag = @sprintf("CVaR%.2f", β_risk)

# Output folder: eps_<X>_beta_<Y> (aligned with logs folder naming)
eps_str = replace(@sprintf("%.1f", ε_val), "." => "p")
beta_str_folder = replace(@sprintf("%.1f", β_risk), "." => "p")
out_subfolder = "eps_$(eps_str)_beta_$(beta_str_folder)"

# Parse 3 x solutions
x_arc_lists = [parse.(Int, split(_pos_args[i], ",")) for i in 3:5]
sol_labels = [get(_kw_args, "label$i", "[$(join(x_arc_lists[i], ","))]") for i in 1:3]
sol_colors = [:steelblue, :darkorange, :seagreen]

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

# ── helper: vertical boxplot ──
function draw_vbox!(plt, sp, xi, data, col; whisker_q=(0.05, 0.95), box_w=0.25, cap_w=0.12)
    s = (
        median = median(data), q25 = quantile(data, 0.25), q75 = quantile(data, 0.75),
        wlo = quantile(data, whisker_q[1]), whi = quantile(data, whisker_q[2]),
        mean_val = mean(data),
    )
    plot!(plt, [xi, xi], [s.wlo, s.q25], subplot=sp, color=col, lw=1.5, ls=:dash, label=false)
    plot!(plt, [xi, xi], [s.q75, s.whi], subplot=sp, color=col, lw=1.5, ls=:dash, label=false)
    plot!(plt, [xi-cap_w, xi+cap_w], [s.wlo, s.wlo], subplot=sp, color=col, lw=2, label=false)
    plot!(plt, [xi-cap_w, xi+cap_w], [s.whi, s.whi], subplot=sp, color=col, lw=2, label=false)
    bx = [xi-box_w, xi+box_w, xi+box_w, xi-box_w, xi-box_w]
    by = [s.q25, s.q25, s.q75, s.q75, s.q25]
    plot!(plt, Shape(bx, by), subplot=sp, fillcolor=col, fillalpha=0.3, linecolor=col, lw=2, label=false)
    plot!(plt, [xi-box_w, xi+box_w], [s.median, s.median], subplot=sp, color=col, lw=3, label=false)
    scatter!(plt, [xi], [s.mean_val], subplot=sp, color=col, marker=:diamond, ms=6, label=false)
    return s
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
phase_labels = ["PhaseA"; [@sprintf("n=%.1f", n) for n in noises]]
n_phases = length(phase_labels)
spacing = 3.5   # x-axis spacing between phases

mkpath(joinpath(@__DIR__, "plots", out_subfolder))

for β_dir in β_dirs
    @printf("\n--- β_dir=%.1f ---\n", β_dir)
    flush(stdout)

    # costs_phases[phase][solution] = Vector{Float64}(M)
    costs_phases = [Vector{Vector{Float64}}(undef, 3) for _ in 1:n_phases]

    # Phase A: symmetric Dirichlet — q_true, q̃ 독립 샘플
    rng = MersenneTwister(42)
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
        rng = MersenneTwister(42)
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
            m % 100 == 0 && @printf("  n=%.1f %d/%d\n", noise, m, M)
        end
    end

    # ── Plot: 2 rows ──
    plt = plot(layout=(2, 1), size=(250*n_phases + 150, 850),
               left_margin=12Plots.mm, bottom_margin=8Plots.mm, top_margin=5Plots.mm)

    # Row 1: absolute costs — 3 solutions side by side
    sol_offset = [-0.5, 0.0, 0.5]
    for (pi, plabel) in enumerate(phase_labels)
        xc = (pi - 1) * spacing
        for sol in 1:3
            draw_vbox!(plt, 1, xc + sol_offset[sol], costs_phases[pi][sol], sol_colors[sol];
                       box_w=0.2, cap_w=0.1)
        end
    end
    xticks!(plt, [(i-1)*spacing for i in 1:n_phases], phase_labels, subplot=1)
    ylabel!(plt, "OOS $risk_tag Max-Flow", subplot=1)
    title!(plt, @sprintf("%s %s β_dir=%.1f — %s", network_name, scenario, β_dir, risk_tag), subplot=1)
    for sol in 1:3
        plot!(plt, [NaN], [NaN], subplot=1, color=sol_colors[sol], lw=4, label=sol_labels[sol])
    end
    plot!(plt, legend=:topright, subplot=1, legendfontsize=8)

    # Row 2: paired Δ vs nominal (solution 1)
    delta_colors = [:darkorange, :seagreen]
    delta_labels = ["$(sol_labels[2])−$(sol_labels[1])", "$(sol_labels[3])−$(sol_labels[1])"]
    for (pi, plabel) in enumerate(phase_labels)
        xc = (pi - 1) * spacing
        for di in 1:2
            gap = costs_phases[pi][di+1] .- costs_phases[pi][1]
            draw_vbox!(plt, 2, xc + (di == 1 ? -0.35 : 0.35), gap, delta_colors[di];
                       whisker_q=(0.01, 0.99), box_w=0.28, cap_w=0.12)
        end
    end
    hline!(plt, [0.0], subplot=2, color=:red, lw=2, ls=:dash, label=false)
    xticks!(plt, [(i-1)*spacing for i in 1:n_phases], phase_labels, subplot=2)
    ylabel!(plt, "Δ (vs $(sol_labels[1]))", subplot=2)
    title!(plt, @sprintf("β_dir=%.1f Paired Δ, whiskers=p01/p99", β_dir), subplot=2)
    for di in 1:2
        plot!(plt, [NaN], [NaN], subplot=2, color=delta_colors[di], lw=4, label=delta_labels[di])
    end
    plot!(plt, legend=:topright, subplot=2, legendfontsize=8)

    β_str = replace(@sprintf("%.1f", β_dir), "." => "p")
    savepath = joinpath(@__DIR__, "plots", out_subfolder,
                        @sprintf("%s_%s_beta%s_phaseAB.png", network_name, scenario, β_str))
    savefig(plt, savepath)
    @printf("  Saved: %s\n", savepath)

    # Print stats
    for (pi, plabel) in enumerate(phase_labels)
        @printf("  [%s] ", plabel)
        for sol in 1:3
            @printf("%s=%.4f  ", sol_labels[sol], mean(costs_phases[pi][sol]))
        end
        println()
    end
    flush(stdout)
end

println("\nDone!")
