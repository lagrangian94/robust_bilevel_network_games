"""
test_inner_cut_quality.jl — Compare inner Benders cut quality between dual ISP and primal ISP.

5×5 grid, S=1. Fix (x,h,λ,ψ0) from first OMP solve, run inner loop for both methods.
Record per-iteration: model_estimate, subprob_obj, gap, μ values, intercepts.
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

includet("../network_generator.jl")
includet("../build_uncertainty_set.jl")
includet("../strict_benders.jl")
includet("../nested_benders_trust_region.jl")
includet("../build_primal_isp.jl")

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
println("Generating 5×5 grid network...")
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

# ===== Initialize OMP → get fixed (x,h,λ,ψ0) =====
omp_model, omp_vars = build_omp(network, ϕU, λU, γ, w;
    optimizer=Gurobi.Optimizer, multi_cut=true)
st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars)

println("\nFixed OMP solution:")
println("  x = $x_sol")
println("  λ = $λ_sol")
println("  h = $h_sol")

R_unc = uncertainty_set[:R]
r_dict_unc = uncertainty_set[:r_dict]
xi_bar_unc = uncertainty_set[:xi_bar]

# ===== Storage for per-iteration data =====
mutable struct InnerIterData
    iter::Int
    model_estimate::Float64
    subprob_obj::Float64
    gap::Float64
    lower_bound::Float64
    α::Vector{Float64}
    μ_leader::Vector{Float64}
    μ_follower::Vector{Float64}
    intercept_leader::Float64
    intercept_follower::Float64
end

# =====================================================================
# DUAL ISP inner loop
# =====================================================================
println("\n" * "="^80)
println("DUAL ISP inner loop")
println("="^80)

imp_d, iv_d = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_optimizer=Gurobi.Optimizer)
st_d, α_d = initialize_imp(imp_d, iv_d)
dl_d, df_d = initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_d)

dual_data = InnerIterData[]
inner_iter_dual = 0
lower_bound_d = -Inf
st_d = MOI.get(imp_d, MOI.TerminationStatus())

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
        (status_l, cut_info_l) = isp_leader_optimize!(dl_d[s][1], dl_d[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_d)
        (status_f, cut_info_f) = isp_follower_optimize!(df_d[s][1], df_d[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_d)
        dict_cut_info_l[s] = cut_info_l
        dict_cut_info_f[s] = cut_info_f
        subprob_obj += cut_info_l[:obj_val] + cut_info_f[:obj_val]
    end

    lower_bound_d = max(lower_bound_d, subprob_obj)
    gap = model_estimate - lower_bound_d

    # Record data
    push!(dual_data, InnerIterData(
        inner_iter_dual, model_estimate, subprob_obj, gap, lower_bound_d,
        copy(α_d),
        copy(dict_cut_info_l[1][:μhat]),
        copy(dict_cut_info_f[1][:μtilde]),
        dict_cut_info_l[1][:intercept],
        dict_cut_info_f[1][:intercept]
    ))

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

# =====================================================================
# PRIMAL ISP inner loop
# =====================================================================
println("\n" * "="^80)
println("PRIMAL ISP inner loop")
println("="^80)

imp_p, iv_p = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_optimizer=Gurobi.Optimizer)
st_p, α_p = initialize_imp(imp_p, iv_p)
pl_p, pf_p = initialize_primal_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)

primal_data = InnerIterData[]
inner_iter_primal = 0
lower_bound_p = -Inf
st_p = MOI.get(imp_p, MOI.TerminationStatus())

while (st_p == MOI.DUAL_INFEASIBLE || st_p == MOI.OPTIMAL)
    global inner_iter_primal, st_p, α_p, lower_bound_p
    inner_iter_primal += 1
    optimize!(imp_p)
    st_p = MOI.get(imp_p, MOI.TerminationStatus())
    α_p_raw = value.(iv_p[:α])
    α_p = max.(α_p_raw, 0.0)
    model_estimate = sum(value.(iv_p[:t_1_l])) + sum(value.(iv_p[:t_1_f]))
    subprob_obj = 0.0
    dict_cut_info_l, dict_cut_info_f = Dict(), Dict()

    for s in 1:S
        (status_l, cut_info_l) = primal_isp_leader_optimize!(pl_p[s][1], pl_p[s][2];
            isp_data=isp_data, α_sol=α_p)
        (status_f, cut_info_f) = primal_isp_follower_optimize!(pf_p[s][1], pf_p[s][2];
            isp_data=isp_data, α_sol=α_p)
        dict_cut_info_l[s] = cut_info_l
        dict_cut_info_f[s] = cut_info_f
        subprob_obj += cut_info_l[:obj_val] + cut_info_f[:obj_val]
    end

    lower_bound_p = max(lower_bound_p, subprob_obj)
    gap = model_estimate - lower_bound_p

    # Record data
    push!(primal_data, InnerIterData(
        inner_iter_primal, model_estimate, subprob_obj, gap, lower_bound_p,
        copy(α_p),
        copy(dict_cut_info_l[1][:μhat]),
        copy(dict_cut_info_f[1][:μtilde]),
        dict_cut_info_l[1][:intercept],
        dict_cut_info_f[1][:intercept]
    ))

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

# =====================================================================
# REPORT
# =====================================================================
println("\n" * "="^80)
println("INNER CUT QUALITY COMPARISON (5×5 grid, S=$S, first outer iteration)")
println("="^80)

println("\n--- Dual ISP ($inner_iter_dual iterations) ---")
println("Iter | model_est  | subprob_obj | gap        | LB         | #nonzero(μ_l) | #nonzero(μ_f)")
for d in dual_data
    nz_l = count(x -> abs(x) > 1e-6, d.μ_leader)
    nz_f = count(x -> abs(x) > 1e-6, d.μ_follower)
    println("$(lpad(d.iter, 4)) | $(lpad(round(d.model_estimate, digits=4), 10)) | $(lpad(round(d.subprob_obj, digits=4), 11)) | $(lpad(round(d.gap, digits=6), 10)) | $(lpad(round(d.lower_bound, digits=4), 10)) | $(lpad(nz_l, 13)) | $(lpad(nz_f, 13))")
end

println("\n--- Primal ISP ($inner_iter_primal iterations) ---")
println("Iter | model_est  | subprob_obj | gap        | LB         | #nonzero(μ_l) | #nonzero(μ_f)")
for d in primal_data
    nz_l = count(x -> abs(x) > 1e-6, d.μ_leader)
    nz_f = count(x -> abs(x) > 1e-6, d.μ_follower)
    println("$(lpad(d.iter, 4)) | $(lpad(round(d.model_estimate, digits=4), 10)) | $(lpad(round(d.subprob_obj, digits=4), 11)) | $(lpad(round(d.gap, digits=6), 10)) | $(lpad(round(d.lower_bound, digits=4), 10)) | $(lpad(nz_l, 13)) | $(lpad(nz_f, 13))")
end

# Compare μ at same α: pick iteration 1 (both start from same α)
println("\n--- Iteration 1: μ comparison at same initial α ---")
println("α₁ (dual):  ", round.(dual_data[1].α, digits=4))
println("α₁ (primal):", round.(primal_data[1].α, digits=4))
println()
println("μ_leader (dual):  ", round.(dual_data[1].μ_leader, digits=4))
println("μ_leader (primal):", round.(primal_data[1].μ_leader, digits=4))
println("μ_leader diff:    ", round.(dual_data[1].μ_leader - primal_data[1].μ_leader, digits=6))
println()
println("μ_follower (dual):  ", round.(dual_data[1].μ_follower, digits=4))
println("μ_follower (primal):", round.(primal_data[1].μ_follower, digits=4))
println("μ_follower diff:    ", round.(dual_data[1].μ_follower - primal_data[1].μ_follower, digits=6))
println()
println("intercept_l (dual):  ", round(dual_data[1].intercept_leader, digits=6))
println("intercept_l (primal):", round(primal_data[1].intercept_leader, digits=6))
println("intercept_f (dual):  ", round(dual_data[1].intercept_follower, digits=6))
println("intercept_f (primal):", round(primal_data[1].intercept_follower, digits=6))

# μ norm comparison across iterations
println("\n--- μ norm evolution ---")
println("Dual ISP:")
for d in dual_data
    println("  iter $(d.iter): ||μ_l||=$(round(norm(d.μ_leader), digits=4)), ||μ_f||=$(round(norm(d.μ_follower), digits=4))")
end
println("Primal ISP:")
for d in primal_data
    println("  iter $(d.iter): ||μ_l||=$(round(norm(d.μ_leader), digits=4)), ||μ_f||=$(round(norm(d.μ_follower), digits=4))")
end
