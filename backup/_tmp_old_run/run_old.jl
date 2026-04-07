# OLD code: 3x3 grid S=1 nested Benders → 결과를 stdout으로 출력
using JuMP, Gurobi, Mosek, MosekTools, HiGHS, Hypatia, Pajarito
using LinearAlgebra, SparseArrays, Infiltrator, Revise

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("parallel_utils.jl")
includet("strict_benders.jl")
includet("nested_benders_trust_region.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

epsilon = 0.5; S = 1; seed = 42; γ_ratio = 0.10; ρ = 0.2; v_param = 1.0; v = 1.0

network = generate_grid_network(3, 3, seed=seed)
num_arcs = length(network.arcs) - 1
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, γ_ratio * num_interdictable)

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = round(ρ * γ * c_bar, digits=4)

capacity_scenarios_regular = capacities[1:end-1, :]
ϕU = 1/epsilon; λU = ϕU

source_arc_idx = findall(a -> a[1] == "s", network.arcs[1:num_arcs])
max_flow_ub = maximum(sum(capacities[source_arc_idx, s] for s in 1:S) / S)
max_cap = maximum(capacity_scenarios_regular)
πU = ϕU; yU = min(max_cap, ϕU); ytsU = min(max_flow_ub, ϕU)

R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

println("="^60)
println("[OLD] TR Nested Benders: 3x3, S=$S, ε=$epsilon")
println("="^60)

omp_model, omp_vars = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, S=S)
result = tr_nested_benders_optimize!(omp_model, omp_vars, network, ϕU, λU, γ, w, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    outer_tr=true, inner_tr=true,
    πU=πU, yU=yU, ytsU=ytsU, strengthen_cuts=:none,
    parallel=false, mini_benders=false)

ub = result[:past_upper_bound][end]
lb = result[:past_lower_bound][end]
x_sol = round.(result[:opt_sol][:x])
println("\n>>> OLD_UB=$ub")
println(">>> OLD_LB=$lb")
println(">>> OLD_X=$(findall(x_sol .> 0.5))")
println(">>> OLD_ITERS=$(length(result[:past_upper_bound]))")

# 파일로도 저장
open("old_result.txt", "w") do f
    println(f, "UB=$ub")
    println(f, "LB=$lb")
    println(f, "X=$(findall(x_sol .> 0.5))")
end
