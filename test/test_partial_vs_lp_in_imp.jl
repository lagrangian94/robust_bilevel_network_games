"""
Compare partial_hat0 vs lp_in_imp_hat0 on Grid 4×4, S=20, FO (ε̂=0, ε̃=0.5).
두 모드의 수렴값, iteration 수, solve time 비교.

실행: julia -t 8 test/test_partial_vs_lp_in_imp.jl
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using LinearAlgebra
using Revise
using Printf
using Statistics
using Infiltrator

const PROJECT_ROOT = joinpath(@__DIR__, "..")
includet(joinpath(PROJECT_ROOT, "network_generator.jl"))
includet(joinpath(PROJECT_ROOT, "build_uncertainty_set.jl"))
includet(joinpath(PROJECT_ROOT, "build_full_model.jl"))
includet(joinpath(PROJECT_ROOT, "parallel_utils.jl"))
includet(joinpath(PROJECT_ROOT, "strict_benders.jl"))
includet(joinpath(PROJECT_ROOT, "nested_benders_trust_region.jl"))

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model

# ===== 파라미터 (run_experiment1_vfm.jl과 동일) =====
const NET_M, NET_N = 4, 4
const TEST_S = 20
const TEST_EPSILON = 0.5
const TEST_GAMMA_RATIO = 0.1
const TEST_RHO = 0.2
const TEST_V = 1.0
const TEST_SEED = 42
const MAX_TIME = 3600.0  # 1시간

# ===== Instance Setup (setup_instance_vfm 동일 로직) =====
function setup_instance()
    network = generate_grid_network(NET_M, NET_N, seed=TEST_SEED)
    num_arcs = length(network.arcs) - 1

    # FO: ε̂=0, ε̃=ε
    epsilon_hat = 0.0
    epsilon_tilde = TEST_EPSILON
    ϕU_tilde = 1.0 / epsilon_tilde
    ϕU_hat = ϕU_tilde  # LP leader ISP uses this as McCormick big-M
    λU = max(ϕU_hat, ϕU_tilde)
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = ceil(Int, TEST_GAMMA_RATIO * num_interdictable)

    capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), TEST_S, seed=TEST_SEED)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * TEST_S)
    w = round(TEST_RHO * γ * c_bar, digits=4)

    # Uncertainty set
    capacity_scenarios_regular = capacities[1:end-1, :]
    R, r_dict_hat, r_dict_tilde, xi_bar = build_robust_counterpart_matrices(
        capacity_scenarios_regular, epsilon_hat, epsilon_tilde)
    uncertainty_set = Dict(
        :R => R, :r_dict_hat => r_dict_hat, :r_dict_tilde => r_dict_tilde,
        :xi_bar => xi_bar, :epsilon_hat => epsilon_hat, :epsilon_tilde => epsilon_tilde)

    # LDR coefficient bounds
    source_arc_idx = findall(a -> a[1] == "s", network.arcs[1:num_arcs])
    max_flow_ub = maximum(sum(capacities[source_arc_idx, s] for s in 1:TEST_S) / TEST_S)
    max_cap = maximum(capacity_scenarios_regular)
    πU_hat = ϕU_hat
    πU_tilde = ϕU_tilde
    yU = min(max_cap, ϕU_tilde)
    ytsU = min(max_flow_ub, ϕU_tilde)

    params = Dict(
        :S => TEST_S, :γ => γ, :ϕU_hat => ϕU_hat, :ϕU_tilde => ϕU_tilde,
        :λU => λU, :w => w, :v => TEST_V,
        :πU_hat => πU_hat, :πU_tilde => πU_tilde, :yU => yU, :ytsU => ytsU,
    )

    return network, uncertainty_set, params
end

# ===== Solve =====
function solve_mode(network, uncertainty_set, params; isp_mode::Symbol)
    γ = params[:γ]
    ϕU_hat = params[:ϕU_hat]
    ϕU_tilde = params[:ϕU_tilde]
    λU = params[:λU]
    w = params[:w]
    πU_hat = params[:πU_hat]
    πU_tilde = params[:πU_tilde]
    yU = params[:yU]
    ytsU = params[:ytsU]
    S_val = params[:S]

    global v = params[:v]
    global S = S_val
    global network_g = network  # avoid shadowing

    GC.gc()

    model, vars = build_omp(network, ϕU_hat, λU, γ, w; optimizer=Gurobi.Optimizer, S=S_val)

    t_start = time()
    result = tr_nested_benders_optimize!(model, vars, network, ϕU_hat, ϕU_tilde, λU, γ, w, uncertainty_set;
        mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
        outer_tr=true, inner_tr=true, isp_mode=isp_mode, max_time=MAX_TIME,
        πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU,
        strengthen_cuts=:mw,
        parallel=true, mini_benders=true, max_mini_benders_iter=3,
        ldr_mode=:both)
    solve_time = time() - t_start

    ub_vec = result[:past_upper_bound]
    lb_vec = result[:past_lower_bound]
    obj_val = minimum(ub_vec)
    best_lb = isempty(lb_vec) ? -Inf : maximum(lb_vec)
    num_iters = length(ub_vec)
    opt_gap = abs(obj_val) > 1e-8 ? (obj_val - best_lb) / abs(obj_val) : Inf

    return Dict(
        :obj => obj_val,
        :lb => best_lb,
        :gap => opt_gap,
        :iters => num_iters,
        :time => solve_time,
        :x_star => result[:opt_sol][:x],
        :inner_iters => result[:inner_iter],
    )
end

# ===== Main =====
println("="^70)
println("  Grid $(NET_M)×$(NET_N), S=$(TEST_S), FO (ε̂=0, ε̃=$(TEST_EPSILON))")
println("  partial_hat0 vs lp_in_imp_hat0")
println("="^70)

network, uncertainty_set, params = setup_instance()
println("  γ=$(params[:γ]), w=$(params[:w]), ϕU_hat=$(params[:ϕU_hat]), ϕU_tilde=$(params[:ϕU_tilde])")
println()

# # --- partial_hat0 --- (이미 결과 있음, skip)
# println("▶ [1/2] partial_hat0")
# res_partial = solve_mode(network, uncertainty_set, params; isp_mode=:partial_hat0)
# println("  Done: obj=$(round(res_partial[:obj], digits=6)), gap=$(round(res_partial[:gap]*100, digits=4))%, " *
#         "iters=$(res_partial[:iters]), time=$(round(res_partial[:time], digits=1))s")
# println()
# GC.gc()

# --- lp_in_imp_hat0 ---
println("▶ lp_in_imp_hat0")
res_lp = solve_mode(network, uncertainty_set, params; isp_mode=:lp_in_imp_hat0)
println("  Done: obj=$(round(res_lp[:obj], digits=6)), gap=$(round(res_lp[:gap]*100, digits=4))%, " *
        "iters=$(res_lp[:iters]), time=$(round(res_lp[:time], digits=1))s")
println()

# ===== 결과 =====
println("="^70)
println("  lp_in_imp_hat0 RESULT")
println("="^70)
@printf("  %-20s  %15.6f\n", "Objective", res_lp[:obj])
@printf("  %-20s  %15.6f\n", "Best LB", res_lp[:lb])
@printf("  %-20s  %14.4f%%\n", "Gap", res_lp[:gap]*100)
@printf("  %-20s  %15d\n", "Outer iters", res_lp[:iters])
@printf("  %-20s  %14.1fs\n", "Solve time", res_lp[:time])
@printf("  %-20s  %15.1f\n", "Avg inner iters", mean(res_lp[:inner_iters]))
