# Quick Phase B comparison: x_dro (ε=0.8867) vs x_nominal on grid_5x5
# Asymmetric Dirichlet: E[q_true] ≠ uniform
using Printf, Statistics, LinearAlgebra, Random, Distributions

include("../network_generator.jl")
using .NetworkGenerator
include("oos_dirichlet.jl")
include("../oos_evaluation.jl")
include("oos_evaluate.jl")

# --- Network setup ---
network = generate_grid_network(5, 5; seed=42)
num_arcs = length(network.arcs) - 1
S = 20
capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
    interdictable_arcs=network.interdictable_arcs, seed=42)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
w = round(maximum(capacities[interdictable_idx, :]); digits=4)

# --- x vectors ---
x_nom = Float64.([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
x_dro = Float64.([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])

@printf("x_nom: arcs %s\n", findall(x_nom .> 0))
@printf("x_dro: arcs %s\n", findall(x_dro .> 0))

# Phase B requires Dict with :nominal, :single, :two_layer
# :single = x_nom (dummy), :two_layer = x_dro
x_stars = Dict(
    :nominal   => x_nom,
    :single    => x_nom,      # same as nominal (dummy)
    :two_layer => x_dro,
)

beta_values = [0.1, 0.3, 0.5]
M = 100
R = 100
seed = 42

println("\n" * "=" ^ 90)
println("Phase B: Asymmetric Dirichlet (E[q_true] ≠ uniform)")
println("=" ^ 90)

for β in beta_values
    @printf("\nβ=%.1f  (M=%d, R=%d, noise_scale=0.5)\n", β, M, R)
    println("-" ^ 90)

    result = oos_evaluate_phase_b(x_stars, network, capacities, β, 1.0, w;
                                   M=M, R=R, noise_scale=0.5, seed=seed)

    gap = result[:gap_two_vs_nom]
    gap_mean = mean(gap)
    gap_se = std(gap) / sqrt(M)
    gap_ci_lo = gap_mean - 1.96 * gap_se
    gap_ci_hi = gap_mean + 1.96 * gap_se
    gap_p5  = quantile(gap, 0.05)
    gap_p95 = quantile(gap, 0.95)
    dro_wins = mean(gap .< 0)

    # Per-model cost stats
    costs_nom = result[:costs][:nominal]
    costs_dro = result[:costs][:two_layer]
    nom_mean = mean(costs_nom)
    nom_se = std(costs_nom) / sqrt(M)
    dro_mean = mean(costs_dro)
    dro_se = std(costs_dro) / sqrt(M)

    @printf("  Nominal:  mean=%.4f [%.4f, %.4f]\n", nom_mean, nom_mean-1.96*nom_se, nom_mean+1.96*nom_se)
    @printf("  DRO:      mean=%.4f [%.4f, %.4f]\n", dro_mean, dro_mean-1.96*dro_se, dro_mean+1.96*dro_se)
    @printf("  Gap (DRO-Nom): mean=%.4f [%.4f, %.4f]\n", gap_mean, gap_ci_lo, gap_ci_hi)
    @printf("  Gap quantiles: p5=%.4f, p95=%.4f\n", gap_p5, gap_p95)
    @printf("  DRO wins: %.1f%%\n", dro_wins * 100)
end
println("\n" * "=" ^ 90)
