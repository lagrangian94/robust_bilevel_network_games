"""
run_oos_eval_grid.jl — Grid 5x5 γ=3 (original intd) OOS Phase B 평가.

x_nom=[25,33,37] vs x_rob=[30,33,37]
"""

using JuMP, HiGHS, Gurobi, Printf, Dates, Statistics, Random, Distributions, LinearAlgebra

include("../../network_generator.jl")
using .NetworkGenerator

include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")

# oos_phase_b_two_models from run_oos_eval.jl
function oos_phase_b_two_models(x_nom, x_rob, network, capacities, v, w;
                                  β=0.5, M=200, noise_scale=0.5, seed=42)
    K = size(capacities, 2)
    rng = MersenneTwister(seed)

    costs_nom = Vector{Float64}(undef, M)
    costs_rob = Vector{Float64}(undef, M)

    for m in 1:M
        α = β .* (1.0 .+ noise_scale * randn(rng, K))
        α = max.(α, 0.01)
        p_center = α / sum(α)

        h_nom = solve_follower_weighted(network, x_nom, v, w, capacities, p_center)
        flows_nom = compute_maxflow_per_scenario(network, x_nom, h_nom, v, capacities)
        costs_nom[m] = dot(p_center, flows_nom)

        h_rob = solve_follower_weighted(network, x_rob, v, w, capacities, p_center)
        flows_rob = compute_maxflow_per_scenario(network, x_rob, h_rob, v, capacities)
        costs_rob[m] = dot(p_center, flows_rob)

        if m % 50 == 0
            @printf("  OOS outer %d/%d\n", m, M)
        end
    end

    gap = costs_rob .- costs_nom

    return Dict(
        :gap_mean   => mean(gap),
        :gap_p5     => quantile(gap, 0.05),
        :gap_p95    => quantile(gap, 0.95),
        :rob_wins   => mean(gap .< 0),    # leader minimizes: rob < nom → rob wins
        :nom_mean   => mean(costs_nom),
        :rob_mean   => mean(costs_rob),
        :costs_nom  => costs_nom,
        :costs_rob  => costs_rob,
    )
end

# ============================================================
# Grid 5x5 γ=3, original interdictable arcs
# ============================================================
net = generate_grid_network(5, 5)
num_arcs = length(net.arcs) - 1

S = 20
caps, _ = generate_capacity_scenarios_factor_sparse(length(net.arcs), S;
    interdictable_arcs=net.interdictable_arcs, seed=42, num_factors=5)
intd_idx = findall(net.interdictable_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
v = 1.0

x_nom = zeros(Float64, num_arcs)
for a in [25, 33, 37]; x_nom[a] = 1.0; end
x_rob = zeros(Float64, num_arcs)
for a in [30, 33, 37]; x_rob[a] = 1.0; end

println("=" ^ 80)
println("OOS Phase B — Grid 5x5 γ=3 (original intd)")
@printf("  x_nom=[25,33,37], x_rob=[30,33,37], w=%.4f\n", w)
println("  β values: [0.1, 0.3, 0.5, 1.0], M=200")
println("=" ^ 80)
flush(stdout)

for β in [0.1, 0.3, 0.5, 1.0]
    @printf("\n  β=%.1f:\n", β)
    flush(stdout)

    result = oos_phase_b_two_models(x_nom, x_rob, net, caps, v, w;
                                      β=β, M=200, noise_scale=0.5, seed=42)

    @printf("    nom_mean=%.4f, rob_mean=%.4f\n", result[:nom_mean], result[:rob_mean])
    @printf("    gap(rob-nom): mean=%.4f, [p5,p95]=[%.4f,%.4f], rob_wins=%.1f%%\n",
            result[:gap_mean], result[:gap_p5], result[:gap_p95], result[:rob_wins]*100)
    flush(stdout)
end

println("\nDone! $(now())")
