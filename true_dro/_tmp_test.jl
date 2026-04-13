root = dirname(@__DIR__)
using JuMP, Gurobi, HiGHS, Printf, LinearAlgebra
include(joinpath(root, "network_generator.jl"))
using .NetworkGenerator
include(joinpath(root, "true_dro/true_dro_data.jl"))
include(joinpath(root, "true_dro/true_dro_build_omp.jl"))
include(joinpath(root, "true_dro/true_dro_build_subproblem.jl"))
include(joinpath(root, "true_dro/true_dro_benders.jl"))
include(joinpath(root, "true_dro/true_dro_recover.jl"))
network = generate_grid_network(3,3;seed=42)
S=3
scenarios,_ = generate_capacity_scenarios_uniform_model(length(network.arcs),S;seed=42)
q_hat = fill(1.0/S,S)
td = make_true_dro_data(network,scenarios,q_hat,0.1,0.1;w=1.0,lambda_U=10.0,gamma=2)
result = true_dro_benders_optimize!(td; mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, max_iter=50, tol=1e-4, verbose=false)
rec = recover_and_print(td, result; optimizer=HiGHS.Optimizer)
