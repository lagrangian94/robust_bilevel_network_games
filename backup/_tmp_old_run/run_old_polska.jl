# OLD code: Polska S=2,10 nested Benders
using JuMP, Gurobi, Mosek, MosekTools, HiGHS, Hypatia, Pajarito
using LinearAlgebra, SparseArrays, Infiltrator, Revise

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("parallel_utils.jl")
includet("strict_benders.jl")
includet("nested_benders_trust_region.jl")

using .NetworkGenerator: generate_polska_network, generate_capacity_scenarios_uniform_model, print_realworld_network_summary

epsilon = 0.5; seed = 42; γ_ratio = 0.10; ρ = 0.2; v = 1.0

network = generate_polska_network()
print_realworld_network_summary(network)
num_arcs = length(network.arcs) - 1
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, γ_ratio * num_interdictable)

results_all = Dict()

for S_val in [2, 10]
    global S = S_val
    println("\n" * "="^70)
    println("[OLD] TR Nested Benders: Polska, S=$S, ε=$epsilon")
    println("="^70)

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

    GC.gc()
    omp_model, omp_vars = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, S=S)
    t_start = time()
    result = tr_nested_benders_optimize!(omp_model, omp_vars, network, ϕU, λU, γ, w, uncertainty_set;
        mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
        outer_tr=true, inner_tr=true,
        πU=πU, yU=yU, ytsU=ytsU, strengthen_cuts=:none,
        parallel=false, mini_benders=false, max_outer_iter=50)
    elapsed = time() - t_start

    ub = result[:past_upper_bound][end]
    lb = result[:past_lower_bound][end]
    x_sol = round.(result[:opt_sol][:x])
    n_iters = length(result[:past_upper_bound])
    gap = abs(ub - lb) / max(abs(ub), 1e-10)

    println("\n>>> OLD_S$(S)_UB=$ub")
    println(">>> OLD_S$(S)_LB=$lb")
    println(">>> OLD_S$(S)_GAP=$gap")
    println(">>> OLD_S$(S)_X=$(findall(x_sol .> 0.5))")
    println(">>> OLD_S$(S)_ITERS=$n_iters")
    println(">>> OLD_S$(S)_TIME=$(round(elapsed, digits=1))s")

    results_all[S] = Dict(:ub=>ub, :lb=>lb, :x=>findall(x_sol .> 0.5), :iters=>n_iters, :time=>elapsed)
end

open("old_result_polska.txt", "w") do f
    for S in [2, 10]
        r = results_all[S]
        println(f, "S$(S)_UB=$(r[:ub])")
        println(f, "S$(S)_LB=$(r[:lb])")
        println(f, "S$(S)_X=$(r[:x])")
        println(f, "S$(S)_ITERS=$(r[:iters])")
    end
end
