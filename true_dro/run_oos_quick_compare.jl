# Quick Phase A comparison: x_dro (ε=0.8867) vs x_nominal on grid_5x5
using Printf, Statistics, LinearAlgebra, Random, Distributions, Serialization

include("../network_generator.jl")
using .NetworkGenerator
include("oos_dirichlet.jl")
include("../oos_evaluation.jl")
include("oos_evaluate.jl")

# --- Network setup (same as in-sample) ---
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

# --- Phase A OOS for multiple β ---
beta_values = [0.1, 0.3, 0.5]
M = 100
L = 100
seed = 42

all_oos = Dict{Tuple{Float64, Symbol}, Dict}()

println("\n" * "=" ^ 130)
@printf("%-6s  %-8s  %21s  %21s  %21s  %7s  %7s\n",
        "β", "Model", "Mean [95% CI]", "p5 [95% CI]", "p95 [95% CI]", "f_share", "Win%")
println("-" ^ 130)

for β in beta_values
    oos_nom = oos_evaluate(x_nom, network, capacities, β, 1.0, w; M=M, L=L, seed=seed)
    oos_dro = oos_evaluate(x_dro, network, capacities, β, 1.0, w; M=M, L=L, seed=seed)

    all_oos[(β, :nominal)] = oos_nom
    all_oos[(β, :dro)] = oos_dro

    wr_dro = compute_win_rate(oos_dro[:Y_bar], oos_nom[:Y_bar])

    ci_mean_nom = @sprintf("%.4f [%.4f,%.4f]", oos_nom[:mean], oos_nom[:ci_lo], oos_nom[:ci_hi])
    ci_p5_nom   = @sprintf("%.4f [%.4f,%.4f]", oos_nom[:p5], oos_nom[:p5_ci_lo], oos_nom[:p5_ci_hi])
    ci_p95_nom  = @sprintf("%.4f [%.4f,%.4f]", oos_nom[:p95], oos_nom[:p95_ci_lo], oos_nom[:p95_ci_hi])

    ci_mean_dro = @sprintf("%.4f [%.4f,%.4f]", oos_dro[:mean], oos_dro[:ci_lo], oos_dro[:ci_hi])
    ci_p5_dro   = @sprintf("%.4f [%.4f,%.4f]", oos_dro[:p5], oos_dro[:p5_ci_lo], oos_dro[:p5_ci_hi])
    ci_p95_dro  = @sprintf("%.4f [%.4f,%.4f]", oos_dro[:p95], oos_dro[:p95_ci_lo], oos_dro[:p95_ci_hi])

    @printf("%-6.2f  %-8s  %21s  %21s  %21s  %7.4f  %7s\n",
            β, "nominal", ci_mean_nom, ci_p5_nom, ci_p95_nom, oos_nom[:follower_share], "---")
    @printf("%-6.2f  %-8s  %21s  %21s  %21s  %7.4f  %5.1f%%\n",
            β, "dro", ci_mean_dro, ci_p5_dro, ci_p95_dro, oos_dro[:follower_share], wr_dro * 100)
    flush(stdout)
    println()
end
println("=" ^ 130)

# --- Save ---
save_path = joinpath(@__DIR__, "oos_quick_compare_phase_a.jls")
serialize(save_path, Dict(
    :all_oos => all_oos,
    :x_nom => x_nom,
    :x_dro => x_dro,
    :beta_values => beta_values,
    :M => M, :L => L, :seed => seed,
))
println("Results saved → $save_path")
