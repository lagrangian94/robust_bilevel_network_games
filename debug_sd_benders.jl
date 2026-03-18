"""
Debug script: Scenario-Decomposed Benders UB issue on Abilene.
Strict Benders 결과는 하드코딩 (이미 검증 완료).
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using LinearAlgebra
using Infiltrator
using Revise
using Logging

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("build_full_model.jl")
includet("parallel_utils.jl")
includet("strict_benders.jl")
includet("nested_benders_trust_region.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary
using .NetworkGenerator: generate_abilene_network, print_realworld_network_summary

# ===== Setup (global scope) =====
S = 2
seed = 42
epsilon = 0.5
γ_ratio = 0.10
ρ = 0.2
v = 1.0

network = generate_abilene_network()
num_arcs = length(network.arcs) - 1

ϕU = 1/epsilon
λU = ϕU
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, γ_ratio * num_interdictable)

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = round(ρ * γ * c_bar, digits=4)

capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

source_arc_idx = findall(a -> a[1] == "s", network.arcs[1:num_arcs])
max_flow_ub = maximum(sum(capacities[source_arc_idx, s] for s in 1:S) / S)
max_cap = maximum(capacity_scenarios_regular)
πU = ϕU
yU = min(max_cap, ϕU)
ytsU = min(max_flow_ub, ϕU)

sb_ub = 1.1250970192069125
x_opt = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
λ_opt = 0.0017868755429410966
h_opt = [0.0, 0.0, 0.0, 0.0, 0.0009771437203832643, 0.0, 0.0003849029005363326, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003998580007903694, 0.0, 0.0, 0.0, 0.0]
ψ0_opt = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0017868755429410966, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0017868755429410966, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0017868755429410966, 0.0, 0.0, 0.0, 0.0]

# ===== Run =====
println("Parameters: S=$S, γ=$γ, ϕU=$ϕU, λU=$λU, w=$w, v=$v")
println("LDR bounds: πU=$πU, yU=$yU, ytsU=$ytsU")
println("\nHardcoded Strict Benders: UB = $sb_ub")
println("  x* = $x_opt")

# ===== Step 2: Joint OSP at optimal =====
println("\n" * "="^80)
println("STEP 2: Joint OSP value at (x*, h*, λ*, ψ0*)")
println("="^80)

osp_joint_model, osp_joint_vars, osp_joint_data = build_dualized_outer_subproblem(
    network, S, ϕU, λU, γ, w, v, uncertainty_set, Mosek.Optimizer,
    λ_opt, x_opt, h_opt, ψ0_opt; πU=πU, yU=yU, ytsU=ytsU)

(joint_status, joint_coeff) = osp_optimize!(osp_joint_model, osp_joint_vars, osp_joint_data,
    λ_opt, x_opt, h_opt, ψ0_opt)
joint_obj = joint_coeff[:obj_val]
joint_α = joint_coeff[:α_sol]
println("Joint OSP obj (= UB): $joint_obj")
println("Joint α sum: $(sum(joint_α))")

# ===== Step 6: Full scenario_benders_optimize! with :mw =====
println("\n" * "="^80)
println("STEP 6: scenario_benders_optimize! (strengthen_cuts=:mw)")
println("="^80)

GC.gc()
model_sd, vars_sd = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut_lf=true, S=S)
result_sd = scenario_benders_optimize!(model_sd, vars_sd, network, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, πU=πU, yU=yU, ytsU=ytsU,
    parallel=false, strengthen_cuts=:mw, mip_optimizer=Gurobi.Optimizer)

sd_ub = minimum(result_sd[:past_upper_bound])
sd_lb = result_sd[:past_obj][end]
println("\n  SD LB = $(round(sd_lb, digits=6)), UB = $(round(sd_ub, digits=6))")
println("  UB history: $(round.(result_sd[:past_upper_bound], digits=6))")
println("  Inner iters: $(result_sd[:inner_iter])")

println("\n" * "="^80)
println("SUMMARY")
println("="^80)
println("Strict Benders UB: $sb_ub")
println("Joint OSP obj:     $joint_obj")
println("SD Benders UB:     $sd_ub")
println("SD Benders LB:     $sd_lb")
