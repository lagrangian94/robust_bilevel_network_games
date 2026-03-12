"""
test_isp_solve_time.jl — Benchmark per-solve time of 4 ISP variants:
  dual ISP leader, dual ISP follower, primal ISP leader, primal ISP follower.

Reports mean, median, variance for each.
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using Hypatia
using LinearAlgebra
using Statistics
using Infiltrator
using Revise

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("strict_benders.jl")
includet("nested_benders_trust_region.jl")
includet("build_primal_isp.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

# ===== Parameters =====
S = 1
ϕU = 10.0
λU = 10.0
γ = 2.0
w = 1.0
v = 1.0
seed = 42
epsilon = 0.5

# ===== Generate Network =====
println("Generating 4×4 grid network...")
network = generate_grid_network(5, 5, seed=seed)
print_network_summary(network)

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

num_arcs = length(network.arcs) - 1
E = ones(num_arcs, num_arcs+1)
d0 = zeros(num_arcs + 1)
d0[end] = 1.0

isp_data = Dict(:E => E, :network => network, :ϕU => ϕU, :λU => λU, :γ => γ,
    :w => w, :v => v, :uncertainty_set => uncertainty_set, :d0 => d0, :S => S)

# ===== Initialize OMP =====
omp_model, omp_vars = build_omp(network, ϕU, λU, γ, w;
    optimizer=Gurobi.Optimizer, multi_cut=true)
st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars)

# ===== Initialize IMP =====
imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer)
st, α_sol = initialize_imp(imp_model, imp_vars)

# ===== Initialize ISP instances =====
dual_leader_instances, dual_follower_instances = initialize_isp(
    network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)

primal_leader_instances, primal_follower_instances = initialize_primal_isp(
    network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)

# ===== Benchmark: run inner loop manually, timing each ISP solve =====

# Storage for solve times
times_dual_leader = Float64[]
times_dual_follower = Float64[]
times_primal_leader = Float64[]
times_primal_follower = Float64[]

# We'll run the full original Benders, recording ISP solve times
# Timed wrappers for ISP solves
function timed_isp_leader_optimize!(model, vars; isp_data=nothing, uncertainty_set=nothing,
        λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing, α_sol=nothing)
    t = @elapsed begin
        result = isp_leader_optimize!(model, vars; isp_data=isp_data, uncertainty_set=uncertainty_set,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
    end
    push!(times_dual_leader, t)
    return result
end

function timed_isp_follower_optimize!(model, vars; isp_data=nothing, uncertainty_set=nothing,
        λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing, α_sol=nothing)
    t = @elapsed begin
        result = isp_follower_optimize!(model, vars; isp_data=isp_data, uncertainty_set=uncertainty_set,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
    end
    push!(times_dual_follower, t)
    return result
end

function timed_primal_isp_leader_optimize!(model, vars; isp_data=nothing, α_sol=nothing)
    t = @elapsed begin
        result = primal_isp_leader_optimize!(model, vars; isp_data=isp_data, α_sol=α_sol)
    end
    push!(times_primal_leader, t)
    return result
end

function timed_primal_isp_follower_optimize!(model, vars; isp_data=nothing, α_sol=nothing)
    t = @elapsed begin
        result = primal_isp_follower_optimize!(model, vars; isp_data=isp_data, α_sol=α_sol)
    end
    push!(times_primal_follower, t)
    return result
end

# ===== Run dual ISP inner loop manually (mimicking tr_imp_optimize!) =====
# We run the full nested benders but intercept ISP calls

# ===== Run both methods side by side, same (x,h,λ,ψ0) =====

println("\n" * "="^80)
println("MANUAL BENCHMARK: Same (x,h,λ,ψ0), multiple inner loops")
println("="^80)

# Reset timing arrays
empty!(times_dual_leader)
empty!(times_dual_follower)
empty!(times_primal_leader)
empty!(times_primal_follower)

# Use the initial OMP solution
R_unc = uncertainty_set[:R]
r_dict_unc = uncertainty_set[:r_dict]
xi_bar_unc = uncertainty_set[:xi_bar]

# --- Dual ISP inner loop ---
println("\n--- Dual ISP inner loop ---")
imp_d, iv_d = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_optimizer=Gurobi.Optimizer)
st_d, α_d = initialize_imp(imp_d, iv_d)
dl_d, df_d = initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)

inner_iter_dual = 0
imp_cuts_d = Dict{Symbol, Any}(:old_tr_constraints => nothing)
st_d = MOI.get(imp_d, MOI.TerminationStatus())
lower_bound_d = -Inf

while (st_d == MOI.DUAL_INFEASIBLE || st_d == MOI.OPTIMAL)
    global inner_iter_dual, st_d, α_d, lower_bound_d
    inner_iter_dual += 1
    optimize!(imp_d)
    st_d = MOI.get(imp_d, MOI.TerminationStatus())
    α_d = value.(iv_d[:α])
    model_estimate = sum(value.(iv_d[:t_1_l])) + sum(value.(iv_d[:t_1_f]))
    subprob_obj = 0.0
    dict_cut_info_l, dict_cut_info_f = Dict(), Dict()

    for s in 1:S
        U_s = Dict(:R => Dict(:1=>R_unc[s]), :r_dict => Dict(:1=>r_dict_unc[s]),
                    :xi_bar => Dict(:1=>xi_bar_unc[s]), :epsilon => epsilon)
        t_l = @elapsed begin
            (status_l, cut_info_l) = isp_leader_optimize!(dl_d[s][1], dl_d[s][2];
                isp_data=isp_data, uncertainty_set=U_s,
                λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_d)
        end
        push!(times_dual_leader, t_l)

        t_f = @elapsed begin
            (status_f, cut_info_f) = isp_follower_optimize!(df_d[s][1], df_d[s][2];
                isp_data=isp_data, uncertainty_set=U_s,
                λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_d)
        end
        push!(times_dual_follower, t_f)

        dict_cut_info_l[s] = cut_info_l
        dict_cut_info_f[s] = cut_info_f
        subprob_obj += cut_info_l[:obj_val] + cut_info_f[:obj_val]
    end

    lower_bound_d = max(lower_bound_d, subprob_obj)
    gap = model_estimate - lower_bound_d
    if gap <= 1e-4
        println("Dual inner converged in $inner_iter_dual iterations")
        break
    end

    # Add inner cuts
    subgradient_l = [dict_cut_info_l[s][:μhat] for s in 1:S]
    subgradient_f = [dict_cut_info_f[s][:μtilde] for s in 1:S]
    intercept_l = [dict_cut_info_l[s][:intercept] for s in 1:S]
    intercept_f = [dict_cut_info_f[s][:intercept] for s in 1:S]
    @constraint(imp_d, [s=1:S], iv_d[:t_1_l][s] <= intercept_l[s] + iv_d[:α]'*subgradient_l[s])
    @constraint(imp_d, [s=1:S], iv_d[:t_1_f][s] <= intercept_f[s] + iv_d[:α]'*subgradient_f[s])

    if inner_iter_dual > 100
        println("Dual inner loop exceeded 100 iterations, stopping")
        break
    end
end

# --- Primal ISP inner loop ---
println("\n--- Primal ISP inner loop ---")
imp_p, iv_p = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_optimizer=Gurobi.Optimizer)
st_p, α_p = initialize_imp(imp_p, iv_p)
pl_p, pf_p = initialize_primal_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)

inner_iter_primal = 0
lower_bound_p = -Inf
st_p = MOI.get(imp_p, MOI.TerminationStatus())

while (st_p == MOI.DUAL_INFEASIBLE || st_p == MOI.OPTIMAL)
    global inner_iter_primal, st_p, α_p, lower_bound_p
    inner_iter_primal += 1
    optimize!(imp_p)
    st_p = MOI.get(imp_p, MOI.TerminationStatus())
    α_p = max.(value.(iv_p[:α]), 0.0)
    model_estimate = sum(value.(iv_p[:t_1_l])) + sum(value.(iv_p[:t_1_f]))
    subprob_obj = 0.0
    dict_cut_info_l, dict_cut_info_f = Dict(), Dict()

    for s in 1:S
        t_l = @elapsed begin
            (status_l, cut_info_l) = primal_isp_leader_optimize!(pl_p[s][1], pl_p[s][2];
                isp_data=isp_data, α_sol=α_p)
        end
        push!(times_primal_leader, t_l)

        t_f = @elapsed begin
            (status_f, cut_info_f) = primal_isp_follower_optimize!(pf_p[s][1], pf_p[s][2];
                isp_data=isp_data, α_sol=α_p)
        end
        push!(times_primal_follower, t_f)

        dict_cut_info_l[s] = cut_info_l
        dict_cut_info_f[s] = cut_info_f
        subprob_obj += cut_info_l[:obj_val] + cut_info_f[:obj_val]
    end

    lower_bound_p = max(lower_bound_p, subprob_obj)
    gap = model_estimate - lower_bound_p
    if gap <= 1e-4
        println("Primal inner converged in $inner_iter_primal iterations")
        break
    end

    # Add inner cuts
    subgradient_l = [dict_cut_info_l[s][:μhat] for s in 1:S]
    subgradient_f = [dict_cut_info_f[s][:μtilde] for s in 1:S]
    intercept_l = [dict_cut_info_l[s][:intercept] for s in 1:S]
    intercept_f = [dict_cut_info_f[s][:intercept] for s in 1:S]
    @constraint(imp_p, [s=1:S], iv_p[:t_1_l][s] <= intercept_l[s] + iv_p[:α]'*subgradient_l[s])
    @constraint(imp_p, [s=1:S], iv_p[:t_1_f][s] <= intercept_f[s] + iv_p[:α]'*subgradient_f[s])

    if inner_iter_primal > 100
        println("Primal inner loop exceeded 100 iterations, stopping")
        break
    end
end

# ===== Report =====
println("\n" * "="^80)
println("ISP SOLVE TIME BENCHMARK (4×4 grid, S=$S)")
println("="^80)

function report_stats(name, times)
    if isempty(times)
        println("  $name: no data")
        return
    end
    println("  $name (n=$(length(times))):")
    println("    mean   = $(round(mean(times)*1000, digits=3)) ms")
    println("    median = $(round(median(times)*1000, digits=3)) ms")
    println("    std    = $(round(std(times)*1000, digits=3)) ms")
    println("    min    = $(round(minimum(times)*1000, digits=3)) ms")
    println("    max    = $(round(maximum(times)*1000, digits=3)) ms")
end

report_stats("Dual ISP Leader", times_dual_leader)
report_stats("Dual ISP Follower", times_dual_follower)
report_stats("Primal ISP Leader", times_primal_leader)
report_stats("Primal ISP Follower", times_primal_follower)

println("\nInner iterations: dual=$inner_iter_dual, primal=$inner_iter_primal")
println("Total ISP solve time:")
println("  Dual:   $(round((sum(times_dual_leader)+sum(times_dual_follower))*1000, digits=1)) ms")
println("  Primal: $(round((sum(times_primal_leader)+sum(times_primal_follower))*1000, digits=1)) ms")
