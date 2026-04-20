# ε_recommended = 0.8867 에서 x* solve (grid_5x5).
using Revise
using JuMP, Gurobi, Printf, LinearAlgebra

if !@isdefined(NetworkGenerator)
    include("../network_generator.jl")
end
using .NetworkGenerator

includet("true_dro_data.jl")
includet("true_dro_build_omp.jl")
includet("true_dro_build_subproblem.jl")
includet("true_dro_build_isp_leader.jl")
includet("true_dro_build_isp_follower.jl")
includet("true_dro_benders.jl")
includet("true_dro_mincut_vi.jl")

network = generate_grid_network(5, 5; seed=42)
num_arcs = length(network.arcs) - 1
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, 0.10 * num_interdictable)
S = 20
capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
    interdictable_arcs=network.interdictable_arcs, seed=42)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
w = round(maximum(capacities[interdictable_idx, :]); digits=4)
q_hat = fill(1.0/S, S)
λU = 2.0

ε = 0.8867
@printf("Solving DRO at ε=%.4f (grid_5x5)\n", ε)

td = make_true_dro_data(network, capacities, q_hat, ε, ε; w=w, lambda_U=λU, gamma=γ)

result = true_dro_benders_optimize!(td;
    mip_optimizer = Gurobi.Optimizer,
    nlp_optimizer = Gurobi.Optimizer,
    lp_optimizer  = Gurobi.Optimizer,
    max_iter = 500,
    tol = 1e-4,
    verbose = true,
    sub_time_limit = 30.0,
    mini_benders = true,
    max_mini_benders_iter = 5,
    strengthen_cuts = :mw,
    valid_inequality = :mincut,
    inexact = true,
    nonconvex_attr = ("NonConvex" => 2))

@printf("\nResult: status=%s, Z₀=%.6f, iters=%d, time=%.1fs\n",
        result[:status], result[:Z0], result[:iters], result[:wall_time])
println("x* = $(round.(Int, result[:x]))")
