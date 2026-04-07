"""
Test: EPSILON_NOMINAL=1.0 (ϕU=1) 에서 N, FO, TO variant가
Full Model (Pajarito) 과 Nested Benders 양쪽 다 정상 작동하는지 검증.

Grid 3×3, S=2, ε=0.5 기준.
  N:  ε̂=1.0, ε̃=1.0  (both nominal)
  FO: ε̂=1.0, ε̃=0.5  (leader nominal, follower robust)
  TO: ε̂=0.5, ε̃=1.0  (leader robust, follower nominal)

실행:
  julia -t 4 debug_test/test_epsilon_nominal.jl
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using Hypatia
using Pajarito
using LinearAlgebra
using Printf
using Revise

const PROJECT_ROOT = joinpath(@__DIR__, "..")
includet(joinpath(PROJECT_ROOT, "network_generator.jl"))
includet(joinpath(PROJECT_ROOT, "build_uncertainty_set.jl"))
includet(joinpath(PROJECT_ROOT, "build_full_model.jl"))
includet(joinpath(PROJECT_ROOT, "parallel_utils.jl"))
includet(joinpath(PROJECT_ROOT, "strict_benders.jl"))
includet(joinpath(PROJECT_ROOT, "nested_benders_trust_region.jl"))

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

# ===== 공통 설정 =====
# ε=0 → ISP SDP degenerate. ε=1e-2 (ϕU=100)로 근사.
# ϕU=1/ε 필수: ISP feasibility가 slope bound 1/ε를 요구
const EPSILON_NOMINAL = 1e-2

network_configs = Dict(
    :grid_3x3 => Dict(:type => :grid, :m => 3, :n => 3),
)

function setup_test_instance(config_key::Symbol;
    S=2, γ_ratio=0.10, ρ=0.2, v=1.0, seed=42,
    epsilon_hat=0.5, epsilon_tilde=epsilon_hat)

    config = network_configs[config_key]
    network = generate_grid_network(config[:m], config[:n], seed=seed)
    print_network_summary(network)

    num_arcs = length(network.arcs) - 1

    ϕU_hat = 1.0 / epsilon_hat
    ϕU_tilde = 1.0 / epsilon_tilde
    λU = ϕU_hat
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = ceil(Int, γ_ratio * num_interdictable)

    capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(ρ * γ * c_bar, digits=4)

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

    params = Dict(
        :S => S, :γ => γ, :ϕU_hat => ϕU_hat, :ϕU_tilde => ϕU_tilde, :λU => λU, :w => w, :v => v,
        :πU_hat => πU_hat, :πU_tilde => πU_tilde, :yU => yU, :ytsU => ytsU,
        :epsilon_hat => epsilon_hat, :epsilon_tilde => epsilon_tilde,
    )

    return network, uncertainty_set, params
end

# ===== Test variants =====
ε = 0.5
variants = [
    (:N,  EPSILON_NOMINAL, EPSILON_NOMINAL),
    (:FO, EPSILON_NOMINAL, ε),
    (:TO, ε,               EPSILON_NOMINAL),
    (:FM, ε,               ε),
]

results_table = []

for (name, ε_hat, ε_tilde) in variants
    println("\n" * "="^70)
    println("VARIANT: $(name)  (ε̂=$(ε_hat), ε̃=$(ε_tilde), ϕU_hat=$(1/ε_hat), ϕU_tilde=$(1/ε_tilde))")
    println("="^70)

    global network, uncertainty_set, params
    network, uncertainty_set, params = setup_test_instance(:grid_3x3;
        S=2, epsilon_hat=ε_hat, epsilon_tilde=ε_tilde)

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

    # global v, S, network for nested benders (compare_benders.jl과 동일 패턴)
    global v = params[:v]
    global S = params[:S]

    println("\n  Params: ϕU_hat=$ϕU_hat, ϕU_tilde=$ϕU_tilde, λU=$λU, γ=$γ, w=$w")
    println("  Bounds: πU_hat=$πU_hat, πU_tilde=$πU_tilde, yU=$yU, ytsU=$ytsU")

    # ── 1. Full Model (Pajarito) ──
    println("\n  ── Full Model (Pajarito) ──")
    full_obj = NaN
    full_time = NaN
    full_status = :UNKNOWN
    try
        GC.gc()
        model_f, vars_f = build_full_2DRNDP_model(network, S_val, ϕU_hat, ϕU_tilde, λU, γ, w, v, uncertainty_set;
            mip_solver=Gurobi.Optimizer, conic_solver=Mosek.Optimizer,
            πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU)
        add_sparsity_constraints!(model_f, vars_f, network, S_val)

        set_optimizer_attribute(model_f, "time_limit", 300.0)  # 5분 제한
        t0 = time()
        optimize!(model_f)
        full_time = time() - t0

        full_status = termination_status(model_f)
        if full_status == MOI.OPTIMAL || full_status == MOI.ALMOST_OPTIMAL
            full_obj = objective_value(model_f)
            x_full = round.(value.(vars_f[:x]))
            println("    Status: $(full_status)")
            println("    Obj: $(round(full_obj, digits=6))")
            println("    x*: $(x_full)")
            println("    Time: $(round(full_time, digits=1))s")
        else
            println("    Status: $(full_status) (no optimal)")
            println("    Time: $(round(full_time, digits=1))s")
        end
    catch e
        println("    ERROR: $e")
        full_status = :ERROR
    end

    # ── 2. Nested Benders (TR) ──
    println("\n  ── Nested Benders (TR, dual) ──")
    nb_obj = NaN
    nb_time = NaN
    nb_status = :UNKNOWN
    try
        GC.gc()
        model_b, vars_b = build_omp(network, ϕU_hat, λU, γ, w; optimizer=Gurobi.Optimizer, S=S_val)

        t1 = time()
        result_b = tr_nested_benders_optimize!(model_b, vars_b, network, ϕU_hat, ϕU_tilde, λU, γ, w, uncertainty_set;
            mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
            outer_tr=true, inner_tr=true,
            πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU,
            strengthen_cuts=:none, parallel=false, mini_benders=true, max_mini_benders_iter=3,
            ldr_mode=:both)
        nb_time = time() - t1

        x_nb = result_b[:opt_sol][:x]
        nb_obj = minimum(result_b[:past_upper_bound])
        n_iters = length(result_b[:past_upper_bound])
        nb_status = :OPTIMAL

        println("    Obj: $(round(nb_obj, digits=6))")
        println("    x*: $(x_nb)")
        println("    Iters: $(n_iters)")
        println("    Time: $(round(nb_time, digits=1))s")
    catch e
        println("    ERROR: $e")
        nb_status = :ERROR
    end

    # ── 비교 ──
    gap = NaN
    if !isnan(full_obj) && !isnan(nb_obj)
        gap = abs(full_obj - nb_obj) / max(abs(full_obj), 1e-8)
        println("\n  Gap: |$(round(full_obj, digits=4)) - $(round(nb_obj, digits=4))| / |full| = $(round(gap*100, digits=2))%")
    end

    push!(results_table, (name=name, ε_hat=ε_hat, ε_tilde=ε_tilde,
        full_status=full_status, full_obj=full_obj, full_time=full_time,
        nb_status=nb_status, nb_obj=nb_obj, nb_time=nb_time, gap=gap))
end

# ===== Summary =====
println("\n\n" * "="^90)
println("SUMMARY: EPSILON_NOMINAL=$(EPSILON_NOMINAL) (ϕU=$(1/EPSILON_NOMINAL))")
println("="^90)
println("┌──────┬───────┬───────┬──────────────┬────────────┬──────────────┬────────────┬─────────┐")
println("│ Name │  ε̂   │  ε̃   │  Full Obj    │ Full Time  │  Benders Obj │ Bend Time  │  Gap %  │")
println("├──────┼───────┼───────┼──────────────┼────────────┼──────────────┼────────────┼─────────┤")
for r in results_table
    full_str = r.full_status == :ERROR ? "   ERROR    " : @sprintf("%11.4f ", r.full_obj)
    nb_str   = r.nb_status == :ERROR   ? "   ERROR    " : @sprintf("%11.4f ", r.nb_obj)
    ft_str   = isnan(r.full_time) ? "     -     " : @sprintf("%8.1fs  ", r.full_time)
    bt_str   = isnan(r.nb_time)   ? "     -     " : @sprintf("%8.1fs  ", r.nb_time)
    gap_str  = isnan(r.gap) ? "   -   " : @sprintf("%6.2f%% ", r.gap*100)
    println(@sprintf("│ %-4s │ %5.2f │ %5.2f │%s│%s│%s│%s│%s│",
        r.name, r.ε_hat, r.ε_tilde, full_str, ft_str, nb_str, bt_str, gap_str))
end
println("└──────┴───────┴───────┴──────────────┴────────────┴──────────────┴────────────┴─────────┘")

# ===== Pass/Fail =====
all_ok = all(r -> r.full_status != :ERROR && r.nb_status != :ERROR, results_table)
if all_ok
    println("\n✓ ALL VARIANTS SOLVED SUCCESSFULLY")
    matched = all(r -> !isnan(r.gap) && r.gap < 0.05, results_table)
    if matched
        println("✓ Full Model / Benders gap < 5% for all variants")
    else
        println("⚠ Some gaps > 5% — check convergence")
    end
else
    println("\n✗ SOME VARIANTS FAILED — see above for details")
end
