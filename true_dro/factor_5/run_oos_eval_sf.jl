using JuMP, HiGHS, Gurobi, Printf, Dates, Statistics, Random, Distributions, LinearAlgebra

include("../../network_generator.jl")
using .NetworkGenerator
include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")

# reuse helper from run_oos_eval.jl
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
        if m % 50 == 0; @printf("  OOS outer %d/%d\n", m, M); end
    end
    gap = costs_rob .- costs_nom
    return Dict(:gap_mean=>mean(gap), :gap_p5=>quantile(gap,0.05), :gap_p95=>quantile(gap,0.95),
                :rob_wins=>mean(gap .< 0), :nom_mean=>mean(costs_nom), :rob_mean=>mean(costs_rob))
end

net = generate_sioux_falls_network()
num_arcs = length(net.arcs) - 1
intd_flags = fill(true, length(net.arcs))
net = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
    net.N, intd_flags, net.arc_adjacency, net.node_arc_incidence)

caps, _ = generate_capacity_scenarios_factor_sparse(length(net.arcs), 20;
    interdictable_arcs=net.interdictable_arcs, seed=42, num_factors=5)
intd_idx = findall(net.interdictable_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)

x_nom = zeros(Float64, num_arcs); x_nom[72] = 1.0; x_nom[73] = 1.0
x_rob = zeros(Float64, num_arcs); x_rob[35] = 1.0; x_rob[73] = 1.0

println("sioux_falls all_intd γ=2: x_nom=[72,73], x_rob=[35,73], w=$w")

for β in [0.1, 0.3, 0.5, 1.0]
    @printf("\nβ=%.1f:\n", β)
    flush(stdout)
    r = oos_phase_b_two_models(x_nom, x_rob, net, caps, 1.0, w; β=β, M=200)
    @printf("  nom=%.4f, rob=%.4f, gap=%.4f, [p5,p95]=[%.4f,%.4f], rob_wins=%.1f%%\n",
            r[:nom_mean], r[:rob_mean], r[:gap_mean], r[:gap_p5], r[:gap_p95], r[:rob_wins]*100)
end
