"""
Debug script for partial robust LP ISP modes (:partial_hat0, :partial_tilde0).

FO/TO를 :dual baseline과 비교하여 cut tightness 및 수렴 검증.

실행:
  julia -t 1 test/debug_partial_robust.jl
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
)

# setup_instance: compare_benders.jl 패턴 재사용
function setup_instance(config_key::Symbol;
    S=2, γ_ratio=0.10, ρ=0.2, v=1.0, seed=42,
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
S = 2
epsilon = 0.5
seed = 42

println("="^80)
println("DEBUG: Partial Robust LP ISP (Grid 3×3, S=$S, ε=$epsilon)")
println("="^80)

# ===== Test 0: :dual baseline =====
println("\n" * "="^80)
println("TEST 0: :dual baseline (both SDP)")
println("="^80)

network, uncertainty_set, params = setup_instance(:grid_3x3;
    S=S, epsilon_hat=epsilon, epsilon_tilde=epsilon, seed=seed)
γ = params[:γ]; ϕU_hat = params[:ϕU_hat]; ϕU_tilde = params[:ϕU_tilde]
λU = params[:λU]; w = params[:w]; v = params[:v]
πU_hat = params[:πU_hat]; πU_tilde = params[:πU_tilde]
yU = params[:yU]; ytsU = params[:ytsU]

GC.gc()
model0, vars0 = build_omp(network, ϕU_hat, λU, γ, w; optimizer=Gurobi.Optimizer, S=S)
t0 = time()
result0 = tr_nested_benders_optimize!(model0, vars0, network, ϕU_hat, ϕU_tilde, λU, γ, w, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    outer_tr=true, inner_tr=true,
    πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU,
    strengthen_cuts=:mw, parallel=false, ldr_mode=:both,
    isp_mode=:dual)
t0 = time() - t0
obj0 = minimum(result0[:past_upper_bound])
x0 = result0[:opt_sol][:x]
iter0 = length(result0[:past_upper_bound])
println("\n>> :dual — obj=$(round(obj0, digits=6)), iters=$iter0, time=$(round(t0, digits=1))s")

# ===== Test 1: :partial_hat0 (FO: Leader=LP, Follower=SDP) =====
println("\n" * "="^80)
println("TEST 1: :partial_hat0 (FO — ε̂=0 LP leader, ε̃=ε SDP follower)")
println("="^80)

network1, uset1, params1 = setup_instance(:grid_3x3;
    S=S, epsilon_hat=0.0, epsilon_tilde=epsilon, seed=seed)
γ1 = params1[:γ]; ϕU_hat1 = params1[:ϕU_hat]; ϕU_tilde1 = params1[:ϕU_tilde]
λU1 = params1[:λU]; w1 = params1[:w]
πU_hat1 = params1[:πU_hat]; πU_tilde1 = params1[:πU_tilde]
yU1 = params1[:yU]; ytsU1 = params1[:ytsU]

GC.gc()
model1, vars1 = build_omp(network1, ϕU_hat1, λU1, γ1, w1; optimizer=Gurobi.Optimizer, S=S)
t1 = time()
result1 = tr_nested_benders_optimize!(model1, vars1, network1, ϕU_hat1, ϕU_tilde1, λU1, γ1, w1, uset1;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    outer_tr=true, inner_tr=true,
    πU_hat=πU_hat1, πU_tilde=πU_tilde1, yU=yU1, ytsU=ytsU1,
    strengthen_cuts=:none, parallel=false, ldr_mode=:both,
    isp_mode=:partial_hat0)
t1 = time() - t1
obj1 = minimum(result1[:past_upper_bound])
x1 = result1[:opt_sol][:x]
iter1 = length(result1[:past_upper_bound])
println("\n>> :partial_hat0 — obj=$(round(obj1, digits=6)), iters=$iter1, time=$(round(t1, digits=1))s")

# ===== Test 2: :partial_tilde0 (TO: Leader=SDP, Follower=LP) =====
println("\n" * "="^80)
println("TEST 2: :partial_tilde0 (TO — ε̂=ε SDP leader, ε̃=0 LP follower)")
println("="^80)

network2, uset2, params2 = setup_instance(:grid_3x3;
    S=S, epsilon_hat=epsilon, epsilon_tilde=0.0, seed=seed)
γ2 = params2[:γ]; ϕU_hat2 = params2[:ϕU_hat]; ϕU_tilde2 = params2[:ϕU_tilde]
λU2 = params2[:λU]; w2 = params2[:w]
πU_hat2 = params2[:πU_hat]; πU_tilde2 = params2[:πU_tilde]
yU2 = params2[:yU]; ytsU2 = params2[:ytsU]

GC.gc()
model2, vars2 = build_omp(network2, ϕU_hat2, λU2, γ2, w2; optimizer=Gurobi.Optimizer, S=S)
t2 = time()
result2 = tr_nested_benders_optimize!(model2, vars2, network2, ϕU_hat2, ϕU_tilde2, λU2, γ2, w2, uset2;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    outer_tr=true, inner_tr=true,
    πU_hat=πU_hat2, πU_tilde=πU_tilde2, yU=yU2, ytsU=ytsU2,
    strengthen_cuts=:none, parallel=false, ldr_mode=:both,
    isp_mode=:partial_tilde0)
t2 = time() - t2
obj2 = minimum(result2[:past_upper_bound])
x2 = result2[:opt_sol][:x]
iter2 = length(result2[:past_upper_bound])
println("\n>> :partial_tilde0 — obj=$(round(obj2, digits=6)), iters=$iter2, time=$(round(t2, digits=1))s")

# ===== Comparison =====
println("\n" * "="^80)
println("COMPARISON TABLE")
println("="^80)
@printf("%-20s %12s %8s %8s\n", "Mode", "Objective", "Iters", "Time(s)")
@printf("%-20s %12s %8s %8s\n", "-"^20, "-"^12, "-"^8, "-"^8)
@printf("%-20s %12.6f %8d %8.1f\n", ":dual (baseline)", obj0, iter0, t0)
@printf("%-20s %12.6f %8d %8.1f\n", ":partial_hat0 (FO)", obj1, iter1, t1)
@printf("%-20s %12.6f %8d %8.1f\n", ":partial_tilde0 (TO)", obj2, iter2, t2)

println("\nExpected: obj(dual) ≥ obj(partial_hat0) and obj(dual) ≥ obj(partial_tilde0)")
println("  dual ≥ FO? $(obj0 >= obj1 - 1e-4 ? "✓" : "✗") ($(round(obj0 - obj1, digits=6)))")
println("  dual ≥ TO? $(obj0 >= obj2 - 1e-4 ? "✓" : "✗") ($(round(obj0 - obj2, digits=6)))")

println("\nx solutions:")
@printf("  %-20s: %s\n", ":dual", x0)
@printf("  %-20s: %s\n", ":partial_hat0", x1)
@printf("  %-20s: %s\n", ":partial_tilde0", x2)

println("\nDone.")
