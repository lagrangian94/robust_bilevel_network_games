# Baseline: original solver, inner_tr=false
using JuMP, Gurobi, Mosek, MosekTools, HiGHS, LinearAlgebra, Infiltrator, Revise
includet("../network_generator.jl")
includet("../build_uncertainty_set.jl")
includet("../strict_benders.jl")
includet("../nested_benders_trust_region.jl")
using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model

S=1; λU=10.0; γ_ratio=0.10; ρ_param=0.2; v=1.0; seed=42; epsilon=0.5; ϕU=1/epsilon
network = generate_grid_network(4, 4, seed=seed)
num_arcs = length(network.arcs) - 1
γ = ceil(Int, γ_ratio * sum(network.interdictable_arcs[1:num_arcs]))
caps, _ = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
idx = findall(network.interdictable_arcs[1:num_arcs])
w = ρ_param * γ * sum(caps[idx, :]) / (length(idx) * S)
println("γ=$γ, w=$(round(w,digits=4))")

R, r_dict, xi_bar = build_robust_counterpart_matrices(caps[1:end-1,:], epsilon)
uset = Dict(:R=>R, :r_dict=>r_dict, :xi_bar=>xi_bar, :epsilon=>epsilon)

omp, vars = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
res = tr_nested_benders_optimize!(omp, vars, network, ϕU, λU, γ, w, uset;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    multi_cut=true, outer_tr=true, inner_tr=false)

println("\n=== BASELINE (inner_tr=false) ===")
println("Time: $(round(res[:solution_time],digits=2))s")
println("Outer: $(length(res[:inner_iter])), Inner: $(sum(res[:inner_iter]))")
obj = haskey(res,:past_local_lower_bound) ? minimum(res[:past_local_lower_bound]) : NaN
println("Objective: $obj")
haskey(res,:opt_sol) && println("x*=$(res[:opt_sol][:x])")
haskey(res,:opt_sol) && println("λ*=$(res[:opt_sol][:λ])")
