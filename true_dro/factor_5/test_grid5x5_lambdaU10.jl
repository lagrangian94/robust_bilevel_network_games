"""
test_grid5x5_lambdaU10.jl — grid 5x5, original interdictable arcs, additive factor k=5, γ=1, λU=10.0
Nominal: Benders ε=0, Robust: Benders ε=1.0
"""

using JuMP, Gurobi, Printf, Dates, Serialization, LinearAlgebra

include("../../network_generator.jl")
using .NetworkGenerator

include("../true_dro_data.jl")
include("../true_dro_build_omp.jl")
include("../true_dro_build_subproblem.jl")
include("../true_dro_build_isp_leader.jl")
include("../true_dro_build_isp_follower.jl")
include("../true_dro_benders.jl")
include("../true_dro_mincut_vi.jl")

net = generate_grid_network(5, 5)
num_arcs = length(net.arcs) - 1

# Grid: 기존 interdictable arcs 보존
caps, F_mat = generate_capacity_scenarios_factor_sparse(length(net.arcs), 20;
    interdictable_arcs=net.interdictable_arcs, seed=42, num_factors=5)
intd_idx = findall(net.interdictable_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
γ = 1
S = 20
λU = 10.0
q_hat = fill(1.0/S, S)

@printf("grid_5x5 k=5: arcs=%d, intd=%d, γ=%d, λU=%.1f, w=%.4f\n", num_arcs, length(intd_idx), γ, λU, w)
flush(stdout)

# Nominal (Benders ε=0)
println("\n" * "="^60)
println("Nominal (Benders ε=0, λU=10.0)")
println("="^60)
flush(stdout)

td_0 = make_true_dro_data(net, caps, q_hat, 0.0, 0.0; w=w, lambda_U=λU, gamma=γ)
t0 = time()
res_nom = true_dro_benders_optimize!(td_0;
    mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
    max_iter=500, tol=1e-4, verbose=true, sub_time_limit=30.0,
    mini_benders=true, max_mini_benders_iter=5,
    strengthen_cuts=:mw, valid_inequality=:mincut,
    inexact=true, nonconvex_attr=("NonConvex" => 2))
wt_nom = time() - t0

x_nom_int = round.(Int, res_nom[:x])
Z0_nom = res_nom[:Z0]

@printf("\nNominal: Z₀=%.6f, iters=%d, time=%.2fs\n", Z0_nom, res_nom[:iters], wt_nom)
println("x_nom arcs = $(findall(x_nom_int .> 0))")
flush(stdout)

# Robust DRO (ε=1.0)
println("\n" * "="^60)
println("Robust DRO (ε=1.0, λU=10.0)")
println("="^60)
flush(stdout)

td_1 = make_true_dro_data(net, caps, q_hat, 1.0, 1.0; w=w, lambda_U=λU, gamma=γ)
t0 = time()
res_rob = true_dro_benders_optimize!(td_1;
    mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
    max_iter=500, tol=1e-4, verbose=true, sub_time_limit=30.0,
    mini_benders=true, max_mini_benders_iter=5,
    strengthen_cuts=:mw, valid_inequality=:mincut,
    inexact=true, nonconvex_attr=("NonConvex" => 2))
wt_rob = time() - t0
x_rob = round.(Int, res_rob[:x])

@printf("\nRobust: Z₀=%.6f, iters=%d, time=%.1fs\n", res_rob[:Z0], res_rob[:iters], wt_rob)
println("x_rob arcs = $(findall(x_rob .> 0))")
flush(stdout)

println("\n" * "="^60)
println("x_nom arcs: $(findall(x_nom_int .> 0))")
println("x_rob arcs: $(findall(x_rob .> 0))")
println("SAME? $(x_nom_int == x_rob)")
println("="^60)
