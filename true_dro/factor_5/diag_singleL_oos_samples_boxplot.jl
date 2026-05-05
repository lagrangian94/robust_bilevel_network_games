"""
diag_singleL_oos_samples_boxplot.jl — Sample별 phaseAB 스타일 boxplot
  β별 파일 1개: x축 = PhaseA, n=0.5, n=1.0, n=2.0, n=5.0
  상단: A(blue) / B(orange) 나란히, 하단: paired Δ (purple)
  plots/<sample>/beta<X>_phaseAB.png

  PhaseA: q_true ~ Dir(β·1_S), q̃ = q̂ (= sample-specific, fixed)
  PhaseB: α = β·(1 + n·randn(S)), α = max(α, 0.01), q_true ~ Dir(α), q̃ = q̂ (fixed)
  h*는 q̂로 1회 계산, flows 고정 → cost = dot(q_true, flows)
"""

using JuMP, HiGHS, Printf, Statistics, Random, Distributions, LinearAlgebra
using Plots

include("../../network_generator.jl")
using .NetworkGenerator
include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")
include("qhat_samples.jl")

# ── network setup ──
net = generate_polska_network()
num_arcs = length(net.arcs) - 1
all_intd = fill(true, length(net.arcs))
net = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
    net.N, all_intd, net.arc_adjacency, net.node_arc_incidence)

S = 20
caps, _ = generate_capacity_scenarios_factor_additive(length(net.arcs), S;
    interdictable_arcs=all_intd, seed=42, num_factors=5)
intd_idx = findall(all_intd[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
v_eff = 1.0

function make_x(arcs, n)
    x = zeros(n); for a in arcs; x[a] = 1.0; end; x
end

# ── helper: draw vertical boxplot ──
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

# ── configurations ──
sample_configs = [
    ("sample1", "NOM/ε01 [6,34]", [6,34], "ε03 [6,18]", [6,18]),
    ("sample3", "NOM/ε01 [6,34]", [6,34], "ε03 [6,18]", [6,18]),
    ("sample8", "NOM [3,6]",      [3,6],  "ε01/ε03 [6,18]", [6,18]),
]

M = 5000
βs = [0.1, 0.3, 0.5, 1.0, 5.0]
noises = [0.5, 1.0, 2.0, 5.0]
phase_labels = ["PhaseA"; [@sprintf("n=%.1f", n) for n in noises]]
n_phases = length(phase_labels)
spacing = 3.0

for (sample_name, label_a, arcs_a, label_b, arcs_b) in sample_configs
    q_hat = QHAT_SAMPLES[sample_name]
    tv_from_unif = 0.5 * sum(abs.(q_hat .- 1.0/S))

    @printf("\n=== %s (TV=%.3f): %s vs %s ===\n", sample_name, tv_from_unif, label_a, label_b)
    flush(stdout)

    x_a = make_x(arcs_a, num_arcs)
    x_b = make_x(arcs_b, num_arcs)

    # h* once with q̃ = q̂ (fixed)
    h_a = solve_follower_weighted(net, x_a, v_eff, w, caps, q_hat)
    h_b = solve_follower_weighted(net, x_b, v_eff, w, caps, q_hat)
    flows_a = compute_maxflow_per_scenario(net, x_a, h_a, v_eff, caps)
    flows_b = compute_maxflow_per_scenario(net, x_b, h_b, v_eff, caps)

    outdir = joinpath(@__DIR__, "plots", sample_name)
    mkpath(outdir)

    for β in βs
        @printf("  β=%.1f ...\n", β)
        flush(stdout)

        # Collect costs for each phase
        costs_a_phases = Vector{Vector{Float64}}(undef, n_phases)
        costs_b_phases = Vector{Vector{Float64}}(undef, n_phases)

        # Phase A: symmetric Dirichlet
        rng = MersenneTwister(42)
        dir = Dirichlet(S, β)
        ca = Vector{Float64}(undef, M)
        cb = Vector{Float64}(undef, M)
        for m in 1:M
            q_true = rand(rng, dir)
            ca[m] = dot(q_true, flows_a)
            cb[m] = dot(q_true, flows_b)
        end
        costs_a_phases[1] = ca
        costs_b_phases[1] = cb

        # Phase B: asymmetric α = β·(1 + noise·randn(S))
        for (ni, noise) in enumerate(noises)
            rng = MersenneTwister(42)
            ca = Vector{Float64}(undef, M)
            cb = Vector{Float64}(undef, M)
            for m in 1:M
                α = β .* (1.0 .+ noise * randn(rng, S))
                α = max.(α, 0.01)
                dir_α = Dirichlet(α)
                q_true = rand(rng, dir_α)
                ca[m] = dot(q_true, flows_a)
                cb[m] = dot(q_true, flows_b)
            end
            costs_a_phases[ni + 1] = ca
            costs_b_phases[ni + 1] = cb
        end

        # ── Plot: 2 rows (absolute + Δ) ──
        plt = plot(layout=(2, 1), size=(250*n_phases + 100, 800),
                   left_margin=10Plots.mm, bottom_margin=8Plots.mm, top_margin=5Plots.mm)

        # Row 1: absolute costs
        for (pi, plabel) in enumerate(phase_labels)
            xc = (pi - 1) * spacing
            draw_vbox!(plt, 1, xc - 0.4, costs_a_phases[pi], :steelblue)
            draw_vbox!(plt, 1, xc + 0.4, costs_b_phases[pi], :darkorange)
        end
        xticks!(plt, [(i-1)*spacing for i in 1:n_phases], phase_labels, subplot=1)
        ylabel!(plt, "OOS Expected Max-Flow", subplot=1)
        title!(plt, @sprintf("β=%.1f Single-L OOS: %s vs %s", β, label_a, label_b), subplot=1)
        # legend
        plot!(plt, [NaN], [NaN], subplot=1, color=:steelblue, lw=4, label=label_a)
        plot!(plt, [NaN], [NaN], subplot=1, color=:darkorange, lw=4, label=label_b)
        plot!(plt, legend=:topright, subplot=1, legendfontsize=8)

        # Row 2: paired Δ
        for (pi, plabel) in enumerate(phase_labels)
            xc = (pi - 1) * spacing
            gap = costs_b_phases[pi] .- costs_a_phases[pi]
            s = draw_vbox!(plt, 2, xc, gap, :purple; whisker_q=(0.01, 0.99), box_w=0.35, cap_w=0.15)
            scatter!(plt, [xc], [mean(gap)], subplot=2, color=:red, marker=:diamond, ms=6, label=false)
        end
        hline!(plt, [0.0], subplot=2, color=:red, lw=2, ls=:dash, label=false)
        xticks!(plt, [(i-1)*spacing for i in 1:n_phases], phase_labels, subplot=2)
        ylabel!(plt, "Δ (B − A)", subplot=2)
        title!(plt, @sprintf("β=%.1f Paired Δ (B − A), whiskers=p01/p99", β), subplot=2)

        # β string for filename
        β_str = replace(@sprintf("%.1f", β), "." => "p")
        savepath = joinpath(outdir, @sprintf("beta%s_phaseAB.png", β_str))
        savefig(plt, savepath)
        @printf("    Saved: %s\n", savepath)
    end
end

println("\nDone!")
