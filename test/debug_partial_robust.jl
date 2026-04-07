"""
Debug script for partial robust LP ISP modes (:partial_hat0, :partial_tilde0).

FO/TO를 :dual baseline과 비교하여 cut tightness 및 수렴 검증.
MW cut strengthening (SDP side only) 디버깅 포함.

실행:
  julia -t 8 test/debug_partial_robust.jl
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using LinearAlgebra
using Infiltrator
using Revise
using Printf

const PROJECT_ROOT = joinpath(@__DIR__, "..")
includet(joinpath(PROJECT_ROOT, "network_generator.jl"))
includet(joinpath(PROJECT_ROOT, "build_uncertainty_set.jl"))
includet(joinpath(PROJECT_ROOT, "build_full_model.jl"))
includet(joinpath(PROJECT_ROOT, "parallel_utils.jl"))
includet(joinpath(PROJECT_ROOT, "strict_benders.jl"))
includet(joinpath(PROJECT_ROOT, "nested_benders_trust_region.jl"))

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

# ===== Network Config =====
network_configs = Dict(
    :grid_3x3 => Dict(:type => :grid, :m => 3, :n => 3),
    :grid_4x4 => Dict(:type => :grid, :m => 4, :n => 4),
)

# setup_instance: compare_benders.jl 패턴 재사용
function setup_instance(config_key::Symbol;
    S=10, γ_ratio=0.10, ρ=0.2, v=1.0, seed=42,
    epsilon_hat=0.5, epsilon_tilde=epsilon_hat)

    config = network_configs[config_key]
    network = generate_grid_network(config[:m], config[:n], seed=seed)
    print_network_summary(network)

    num_arcs = length(network.arcs) - 1

    # ε=0 처리: run_experiment1_vfm.jl의 setup_instance_vfm 패턴
    if epsilon_hat == 0.0 && epsilon_tilde == 0.0
        ϕU_hat = 10.0; ϕU_tilde = 10.0
    elseif epsilon_hat == 0.0
        ϕU_tilde = 1.0 / epsilon_tilde
        ϕU_hat = ϕU_tilde
    elseif epsilon_tilde == 0.0
        ϕU_hat = 1.0 / epsilon_hat
        ϕU_tilde = ϕU_hat
    else
        ϕU_hat = 1.0 / epsilon_hat
        ϕU_tilde = 1.0 / epsilon_tilde
    end
    λU = max(ϕU_hat, ϕU_tilde)
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = ceil(Int, γ_ratio * num_interdictable)
    println("  Interdiction budget: γ = ceil($γ_ratio × $num_interdictable) = $γ")

    capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(ρ * γ * c_bar, digits=4)
    println("  Recovery budget: w = ρ·γ·c̄ = $ρ × $γ × $(round(c_bar, digits=2)) = $(round(w, digits=4))")

    capacity_scenarios_regular = capacities[1:end-1, :]
    R, r_dict_hat, r_dict_tilde, xi_bar = build_robust_counterpart_matrices(
        capacity_scenarios_regular, epsilon_hat, epsilon_tilde)
    uncertainty_set = Dict(
        :R => R, :r_dict_hat => r_dict_hat, :r_dict_tilde => r_dict_tilde,
        :xi_bar => xi_bar, :epsilon_hat => epsilon_hat, :epsilon_tilde => epsilon_tilde)

    source_arc_idx = findall(a -> a[1] == "s", network.arcs[1:num_arcs])
    max_flow_ub = maximum(sum(capacities[source_arc_idx, s] for s in 1:S) / S)
    max_cap = maximum(capacity_scenarios_regular)
    πU_hat = ϕU_hat
    πU_tilde = ϕU_tilde
    yU = min(max_cap, ϕU_tilde)
    ytsU = min(max_flow_ub, ϕU_tilde)
    println("  LDR bounds: ϕU_hat=$ϕU_hat, ϕU_tilde=$ϕU_tilde, πU_hat=$πU_hat, πU_tilde=$πU_tilde, yU=$yU, ytsU=$ytsU")

    params = Dict(
        :S => S, :γ => γ, :ϕU_hat => ϕU_hat, :ϕU_tilde => ϕU_tilde, :λU => λU, :w => w, :v => v,
        :πU_hat => πU_hat, :πU_tilde => πU_tilde, :yU => yU, :ytsU => ytsU,
        :epsilon_hat => epsilon_hat, :epsilon_tilde => epsilon_tilde,
        :γ_ratio => γ_ratio, :ρ => ρ, :seed => seed,
    )

    return network, uncertainty_set, params
end


# ===== Fixed Parameters =====
NET = :grid_4x4
S = 20
epsilon = 0.5
seed = 42

println("="^80)
println("DEBUG: Partial MW ($(NET), S=$S, ε=$epsilon)")
println("="^80)

# # ===== Test 1: :partial_tilde0 + MW (TO: Leader=SDP, Follower=LP) =====
# # 에러 재현: VariableNotOwned at evaluate_partial_mw_opt_cut
# println("\n" * "="^80)
# println("TEST 1: :partial_tilde0 + MW (TO — ε̂=ε SDP leader, ε̃=0 LP follower)")
# println("  parallel=false 로 먼저 MW 동작 확인")
# println("="^80)

network1, uset1, params1 = setup_instance(NET;
    S=S, epsilon_hat=epsilon, epsilon_tilde=0.0, seed=seed)
γ1 = params1[:γ]; ϕU_hat1 = params1[:ϕU_hat]; ϕU_tilde1 = params1[:ϕU_tilde]
λU1 = params1[:λU]; w1 = params1[:w]
πU_hat1 = params1[:πU_hat]; πU_tilde1 = params1[:πU_tilde]
yU1 = params1[:yU]; ytsU1 = params1[:ytsU]

# GC.gc()
# model1, vars1 = build_omp(network1, ϕU_hat1, λU1, γ1, w1; optimizer=Gurobi.Optimizer, S=S)
# t1 = time()
# println("\n>> Running partial_tilde0 with MW, parallel=false...")
# result1 = tr_nested_benders_optimize!(model1, vars1, network1, ϕU_hat1, ϕU_tilde1, λU1, γ1, w1, uset1;
#     mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
#     outer_tr=true, inner_tr=true,
#     πU_hat=πU_hat1, πU_tilde=πU_tilde1, yU=yU1, ytsU=ytsU1,
#     strengthen_cuts=:mw, parallel=false, ldr_mode=:both,
#     isp_mode=:partial_tilde0,
#     max_outer_iter=5)
# t1 = time() - t1
# obj1 = minimum(result1[:past_upper_bound])
# iter1 = length(result1[:past_upper_bound])
# println("\n>> :partial_tilde0 + MW (serial) — obj=$(round(obj1, digits=6)), iters=$iter1, time=$(round(t1, digits=1))s")

# ===== Test 2: 같은 설정 + parallel=true (에러 재현) =====
println("\n" * "="^80)
println("TEST 2: :partial_tilde0 + MW (parallel=true) — 에러 재현 테스트")
println("="^80)

GC.gc()
model2, vars2 = build_omp(network1, ϕU_hat1, λU1, γ1, w1; optimizer=Gurobi.Optimizer, S=S)
t2 = time()
println("\n>> Running partial_tilde0 with MW, parallel=true...")
try
    global result2 = tr_nested_benders_optimize!(model2, vars2, network1, ϕU_hat1, ϕU_tilde1, λU1, γ1, w1, uset1;
        mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
        outer_tr=true, inner_tr=true,
        πU_hat=πU_hat1, πU_tilde=πU_tilde1, yU=yU1, ytsU=ytsU1,
        strengthen_cuts=:mw, parallel=true, ldr_mode=:both,
        isp_mode=:partial_tilde0,
        max_outer_iter=5)
    global t2 = time() - t2
    obj2 = minimum(result2[:past_upper_bound])
    iter2 = length(result2[:past_upper_bound])
    println("\n>> :partial_tilde0 + MW (parallel) — obj=$(round(obj2, digits=6)), iters=$iter2, time=$(round(t2, digits=1))s")
catch e
    global t2 = time() - t2
    println("\n>> ERROR (parallel=true): ", sprint(showerror, e))
    println(">> Time before error: $(round(t2, digits=1))s")
    rethrow(e)
end

# ===== Test 3: :partial_hat0 + MW (FO) =====
println("\n" * "="^80)
println("TEST 3: :partial_hat0 + MW (FO — ε̂=0 LP leader, ε̃=ε SDP follower)")
println("="^80)

network3, uset3, params3 = setup_instance(NET;
    S=S, epsilon_hat=0.0, epsilon_tilde=epsilon, seed=seed)
γ3 = params3[:γ]; ϕU_hat3 = params3[:ϕU_hat]; ϕU_tilde3 = params3[:ϕU_tilde]
λU3 = params3[:λU]; w3 = params3[:w]
πU_hat3 = params3[:πU_hat]; πU_tilde3 = params3[:πU_tilde]
yU3 = params3[:yU]; ytsU3 = params3[:ytsU]

GC.gc()
model3, vars3 = build_omp(network3, ϕU_hat3, λU3, γ3, w3; optimizer=Gurobi.Optimizer, S=S)
t3 = time()
println("\n>> Running partial_hat0 with MW, parallel=false...")
result3 = tr_nested_benders_optimize!(model3, vars3, network3, ϕU_hat3, ϕU_tilde3, λU3, γ3, w3, uset3;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    outer_tr=true, inner_tr=true,
    πU_hat=πU_hat3, πU_tilde=πU_tilde3, yU=yU3, ytsU=ytsU3,
    strengthen_cuts=:mw, parallel=false, ldr_mode=:both,
    isp_mode=:partial_hat0,
    max_outer_iter=5)
t3 = time() - t3
obj3 = minimum(result3[:past_upper_bound])
iter3 = length(result3[:past_upper_bound])
println("\n>> :partial_hat0 + MW (serial) — obj=$(round(obj3, digits=6)), iters=$iter3, time=$(round(t3, digits=1))s")

println("\n" * "="^80)
println("DEBUG COMPLETE")
println("="^80)
