"""
diag_polska_oos.jl — polska nom(arc 6) vs rob(arc 33) OOS evaluation
"""

using JuMP, HiGHS, Gurobi, Printf, Dates, Statistics, Random, Distributions, LinearAlgebra

include("../../network_generator.jl")
using .NetworkGenerator

include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")

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
v = 1.0

# x vectors from solver results
x_nom = zeros(Float64, num_arcs); x_nom[6] = 1.0
x_rob = zeros(Float64, num_arcs); x_rob[33] = 1.0

x_dict = Dict(:nominal => x_nom, :robust => x_rob)

@printf("polska: arcs=%d, S=%d, w=%.4f, x_nom=[6], x_rob=[33]\n", num_arcs, S, w)

println("\n" * "="^80)
println("OOS Phase B — nom vs rob (M=500)")
println("="^80)

κ_mults = [0.5, 1.0, 5.0]
for κ_mult in κ_mults
    κ = κ_mult * S
    β = κ / S   # = κ * q̂, where q̂ = 1/S (uniform)
    @printf("\nκ=%.1f·S (κ=%g, β=κ·q̂=%.2f):\n", κ_mult, κ, β)
    flush(stdout)

    costs = oos_phase_b_generic(x_dict, net, caps, v, w;
        β=β, M=500, noise_scale=0.5, seed=42, mode=:same_alpha)

    gap = costs[:robust] .- costs[:nominal]
    @printf("  nom_mean=%.4f, rob_mean=%.4f\n", mean(costs[:nominal]), mean(costs[:robust]))
    @printf("  gap(rob-nom): mean=%.4f, [p5,p95]=[%.4f,%.4f]\n",
            mean(gap), quantile(gap, 0.05), quantile(gap, 0.95))
    @printf("  rob_wins=%.1f%% (gap<0 → rob wins, leader minimizes)\n", mean(gap .< 0)*100)
    flush(stdout)
end

println("\n" * "="^80)
println("Done! $(now())")
