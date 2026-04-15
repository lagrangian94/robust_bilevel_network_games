"""
run_oos_experiment.jl — True-DRO OOS Experiment: ε calibration + Dirichlet OOS evaluation.

3 models 비교:
  :nominal      → build_full_2SP_model (ε̂=0, ε̃=0)
                  양쪽 다 nominal. 분포 불확실성 무시.
  :single_dro   → true_dro_benders_optimize! (ε̂=ε, ε̃=0)
                  Leader만 ambiguity set 사용 (1-layer DRO).
                  Follower는 nominal q̂=(1/K)·1 그대로 믿고 h 결정.
  :twolayer_dro → true_dro_benders_optimize! (ε̂=ε, ε̃=ε)
                  Leader + Follower 양쪽 다 ambiguity set (2-layer DRO).
                  가장 보수적인 interdiction 결정.

OOS 기대 순서 (leader 관점, leader는 flow를 minimize):
  nominal(낙관적) ≤ single_dro ≤ twolayer_dro(보수적)

Workflow per (network, instance, β):
  1. calibrate_epsilon(K, β) → ε
  2. Solve 3 models → x* each
  3. oos_evaluate(x*, ...) → OOS performance comparison
"""

using Revise
using JuMP
using Gurobi
using HiGHS
using Printf
using LinearAlgebra
using Statistics
using Dates
using Serialization
using Random
using Infiltrator
# ---- Load modules ----
if !@isdefined(NetworkGenerator)
    include("../network_generator.jl")
end
using .NetworkGenerator

# True-DRO
includet("true_dro_data.jl")
includet("true_dro_build_omp.jl")
includet("true_dro_build_subproblem.jl")
includet("true_dro_build_isp_leader.jl")
includet("true_dro_build_isp_follower.jl")
includet("true_dro_benders.jl")
includet("true_dro_mincut_vi.jl")
includet("true_dro_recover.jl")

# Nominal SP
includet("../build_uncertainty_set.jl")
includet("../build_nominal_sp.jl")

# OOS utilities
includet("../oos_evaluation.jl")   # build_maxflow_template, solve_deterministic_maxflow!
includet("oos_dirichlet.jl")
includet("oos_evaluate.jl")


# ===== Network configs =====
network_configs = Dict(
    :grid_3x3   => Dict(:type => :grid, :m => 3, :n => 3),
    :grid_4x4   => Dict(:type => :grid, :m => 4, :n => 4),
    :grid_5x5   => Dict(:type => :grid, :m => 5, :n => 5),
    :abilene    => Dict(:type => :real_world, :generator => NetworkGenerator.generate_abilene_network),
    :polska     => Dict(:type => :real_world, :generator => NetworkGenerator.generate_polska_network),
)


# ===== Default experiment parameters (spec §8) =====
const OOS_NETWORKS = [:grid_3x3, :grid_4x4, :grid_5x5, :abilene, :polska]
const OOS_BETA_VALUES = [0.1, 0.3, 0.5, 0.8]
const OOS_S = 10              # K (=S) scenarios
const OOS_GAMMA_RATIO = 0.10
const OOS_RHO = 0.2
const OOS_V = 1.0             # full interdiction
const OOS_M = 100             # outer OOS samples
const OOS_L = 1000            # inner OOS samples
const OOS_N_CAL = 10000       # ε calibration samples
const OOS_COVERAGE = 0.95
const OOS_N_INSTANCES = 10    # 총 instance 수 (inst_seed=1..10)
const OOS_SEED = 42           # OOS Dirichlet 샘플링 seed (고정)
const OOS_BENDERS_MAX_ITER = 500
const OOS_BENDERS_TOL = 1e-4
const OOS_SUB_TIME_LIMIT = 120.0
const OOS_NOMINAL_TIME_LIMIT = 3600.0


# ===== Checkpoint =====
const OOS_CHECKPOINT_FILE = "true_dro/oos_experiment_checkpoint.jls"

function load_oos_checkpoint()
    isfile(OOS_CHECKPOINT_FILE) ? deserialize(OOS_CHECKPOINT_FILE) : Dict{String,Any}()
end

function save_oos_checkpoint(results)
    serialize(OOS_CHECKPOINT_FILE, results)
end

function oos_result_key(net_key, β, instance_id, model_name)
    "$(net_key)_beta$(β)_inst$(instance_id)_$(model_name)"
end


# ===== Instance setup =====
"""
    setup_oos_instance(config_key; S, γ_ratio, ρ, v, seed) -> (network, capacities)

네트워크 + capacity scenario 생성 (ε는 별도 calibrate).
"""
function setup_oos_instance(config_key::Symbol;
    S=OOS_S, γ_ratio=OOS_GAMMA_RATIO, ρ=OOS_RHO, v=OOS_V, seed=42)

    config = network_configs[config_key]

    if config[:type] == :grid
        network = NetworkGenerator.generate_grid_network(config[:m], config[:n]; seed=seed)
    elseif config[:type] == :real_world
        network = config[:generator]()
    end

    num_arcs = length(network.arcs) - 1
    capacities, _ = NetworkGenerator.generate_capacity_scenarios_factor_model(length(network.arcs), S;
        interdictable_arcs=network.interdictable_arcs, seed=seed)

    # Compute parameters
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = ceil(Int, γ_ratio * num_interdictable)
    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(ρ * γ * c_bar; digits=4)

    return network, capacities, γ, w
end


# ===== Model solvers =====

"""
    solve_model_nominal(network, capacities, γ, w, v; max_time) -> (x_star, obj_val, solve_time)

Nominal SP: build_full_2SP_model with ε=0.
"""
function solve_model_nominal(network, capacities::Matrix{Float64}, γ::Int, w::Float64,
                              v::Float64; max_time=OOS_NOMINAL_TIME_LIMIT)
    num_arcs = length(network.arcs) - 1
    S = size(capacities, 2)

    # Uncertainty set (ε=0 → nominal)
    cap_regular = capacities[1:end-1, :]
    R, r_dict_hat, r_dict_tilde, xi_bar = build_robust_counterpart_matrices(cap_regular, 0.0, 0.0)
    uncertainty_set = Dict(
        :R => R, :r_dict_hat => r_dict_hat, :r_dict_tilde => r_dict_tilde,
        :xi_bar => xi_bar, :epsilon_hat => 0.0, :epsilon_tilde => 0.0)

    ϕU = 10.0
    λU = 10.0

    GC.gc()
    t_start = time()
    model, vars = build_full_2SP_model(network, S, ϕU, λU, γ, w, v, uncertainty_set)
    set_optimizer_attribute(model, "TimeLimit", max_time)
    optimize!(model)
    solve_time = time() - t_start

    st = termination_status(model)
    if st ∉ (MOI.OPTIMAL, MOI.TIME_LIMIT)
        error("Nominal SP failed: $st")
    end

    x_star = value.(vars[:x])
    obj_val = objective_value(model)

    return x_star, obj_val, solve_time
end


"""
    solve_model_dro(network, capacities, γ, w, eps_hat, eps_tilde;
                    max_iter, tol, sub_time_limit) -> (x_star, obj_val, solve_time, iters)

True-DRO via Benders. eps_tilde=0 → single-DRO, else two-layer.
"""
function solve_model_dro(network, capacities::Matrix{Float64}, γ::Int, w::Float64,
                          eps_hat::Float64, eps_tilde::Float64;
                          max_iter=OOS_BENDERS_MAX_ITER, tol=OOS_BENDERS_TOL,
                          sub_time_limit=OOS_SUB_TIME_LIMIT)
    S = size(capacities, 2)
    λU = 2.0
    q_hat = fill(1.0 / S, S)

    td = make_true_dro_data(network, capacities, q_hat, eps_hat, eps_tilde;
                            w=w, lambda_U=λU, gamma=γ)

    GC.gc()
    t_start = time()
    result = true_dro_benders_optimize!(td;
        mip_optimizer=Gurobi.Optimizer,
        nlp_optimizer=Gurobi.Optimizer,
        max_iter=max_iter,
        tol=tol,
        verbose=true,
        sub_verbose=false,
        sub_time_limit=sub_time_limit,
        mini_benders=true,
        lp_optimizer=HiGHS.Optimizer,
        valid_inequality=:mincut)
    solve_time = time() - t_start

    x_star = result[:x]
    obj_val = result[:Z0]

    return x_star, obj_val, solve_time, result[:iters], result[:status]
end


# ===== Main experiment =====

function run_oos_experiment(;
    networks=OOS_NETWORKS,
    β_values=OOS_BETA_VALUES,
    generalize::Bool=false,
    S=OOS_S,
    M=OOS_M,
    L=OOS_L,
    n_cal=OOS_N_CAL,
    coverage=OOS_COVERAGE,
    verbose=true)

    # generalize=false: 첫 instance(seed=42)만 실행
    # generalize=true:  나머지 9개(seed=43..51)도 추가 실행
    n_instances = generalize ? OOS_N_INSTANCES : 1

    results = load_oos_checkpoint()

    model_configs = [
        (:nominal,      0.0, 0.0),
        (:single_dro,   :eps, 0.0),    # eps_hat=ε, eps_tilde=0
        (:twolayer_dro, :eps, :eps),    # eps_hat=ε, eps_tilde=ε
    ]

    total_runs = length(networks) * length(β_values) * n_instances * length(model_configs)
    completed = count(v -> !haskey(v, :error), values(results))

    println("=" ^ 80)
    println("TRUE-DRO OOS EXPERIMENT: ε Calibration + Dirichlet OOS")
    println("=" ^ 80)
    println("  Networks: $networks")
    println("  β values: $β_values")
    println("  generalize=$generalize → n_instances=$n_instances (inst_seed=1..$(n_instances), oos_seed=$(OOS_SEED))")
    println("  S=$S, M=$M, L=$L")
    println("  n_cal=$n_cal, coverage=$coverage")
    println("  Total runs: $total_runs, already done: $completed")
    println("=" ^ 80)

    run_count = completed

    for net_key in networks
        for β in β_values
            # ── ε calibration ──
            ε_cal = calibrate_epsilon(S, β; n_cal=n_cal, coverage=coverage)
            @printf("\n[%s, β=%.2f] calibrated ε = %.6f\n", net_key, β, ε_cal)

            for inst in 1:n_instances
                inst_seed = inst  # capacity scenario seed: 1,2,...,10

                # ── Instance setup ──
                network, capacities, γ, w = setup_oos_instance(net_key;
                    S=S, seed=inst_seed)
                num_arcs = length(network.arcs) - 1

                @printf("  Instance %d: inst_seed=%d, |A|=%d, γ=%d, w=%.4f\n",
                        inst, inst_seed, num_arcs, γ, w)

                for (model_name, eps_hat_spec, eps_tilde_spec) in model_configs
                    rkey = oos_result_key(net_key, β, inst, model_name)

                    # Skip if already done
                    if haskey(results, rkey) && !haskey(results[rkey], :error)
                        run_count += 1
                        continue
                    end

                    # Resolve ε values
                    eps_hat = eps_hat_spec === :eps ? ε_cal : Float64(eps_hat_spec)
                    eps_tilde = eps_tilde_spec === :eps ? ε_cal : Float64(eps_tilde_spec)

                    run_count += 1
                    println("\n" * "─" ^ 70)
                    @printf("[%d/%d] %s | β=%.2f | inst=%d | %s (ε̂=%.4f, ε̃=%.4f)\n",
                            run_count, total_runs, net_key, β, inst, model_name, eps_hat, eps_tilde)
                    println("  $(Dates.format(now(), "HH:MM:SS"))")
                    println("─" ^ 70)

                    try
                        # ── In-sample solve ──
                        x_star, obj_val, solve_time, n_iters, status = if model_name == :nominal
                            x, obj, t = solve_model_nominal(network, capacities, γ, w, OOS_V)
                            (x, obj, t, 0, :Optimal)
                        else
                            solve_model_dro(network, capacities, γ, w, eps_hat, eps_tilde)
                        end

                        x_int = round.(Int, x_star)
                        @printf("  In-sample: obj=%.4f, time=%.1fs, iters=%d, status=%s\n",
                                obj_val, solve_time, n_iters, status)
                        println("  x* = $x_int")

                        # ── OOS evaluation ──
                        # OOS Dirichlet seed 고정 (공정 비교). capacity만 inst_seed로 달라짐.
                        oos_result = oos_evaluate(x_star, network, capacities, β, OOS_V, w;
                                                   M=M, L=L, seed=OOS_SEED)

                        # ── Store ──
                        results[rkey] = Dict(
                            :network => net_key,
                            :beta => β,
                            :instance => inst,
                            :model => model_name,
                            :epsilon_cal => ε_cal,
                            :epsilon_hat => eps_hat,
                            :epsilon_tilde => eps_tilde,
                            :x_star => x_star,
                            :obj_insample => obj_val,
                            :solve_time => solve_time,
                            :n_iters => n_iters,
                            :status => status,
                            :oos_mean => oos_result[:mean],
                            :oos_p95 => oos_result[:p95],
                            :var_outer => oos_result[:var_outer],
                            :var_inner => oos_result[:var_inner],
                            :follower_share => oos_result[:follower_share],
                        )

                    catch e
                        @warn "FAILED: $rkey" exception=(e, catch_backtrace())
                        results[rkey] = Dict(
                            :network => net_key,
                            :beta => β,
                            :instance => inst,
                            :model => model_name,
                            :error => string(e),
                        )
                    end

                    save_oos_checkpoint(results)
                end  # model
            end  # instance
        end  # β
    end  # network

    println("\n" * "=" ^ 80)
    println("EXPERIMENT COMPLETE")
    println("=" ^ 80)

    print_oos_summary(results, networks, β_values, n_instances)

    return results
end


# ===== Summary table =====

function print_oos_summary(results, networks, β_values, n_instances)
    println("\n" * "=" ^ 80)
    println("OOS EXPERIMENT SUMMARY")
    println("=" ^ 80)

    model_names = [:nominal, :single_dro, :twolayer_dro]

    for net_key in networks
        println("\n── $(net_key) ──")
        @printf("  %-6s | %-8s | %-12s | %-12s | %-12s | %-8s | %-8s\n",
                "β", "ε_cal", "Model", "OOS Mean", "OOS p95", "VFR_mean", "VFR_p95")
        println("  " * "-" ^ 78)

        for β in β_values
            # Collect per-model statistics across instances
            model_means = Dict{Symbol, Vector{Float64}}()
            model_p95s = Dict{Symbol, Vector{Float64}}()

            for m_name in model_names
                model_means[m_name] = Float64[]
                model_p95s[m_name] = Float64[]
                for inst in 1:n_instances
                    rkey = oos_result_key(net_key, β, inst, m_name)
                    if haskey(results, rkey) && !haskey(results[rkey], :error)
                        push!(model_means[m_name], results[rkey][:oos_mean])
                        push!(model_p95s[m_name], results[rkey][:oos_p95])
                    end
                end
            end

            # ε_cal from first available result
            ε_cal = NaN
            for inst in 1:n_instances
                rkey = oos_result_key(net_key, β, inst, :nominal)
                if haskey(results, rkey) && haskey(results[rkey], :epsilon_cal)
                    ε_cal = results[rkey][:epsilon_cal]
                    break
                end
            end

            # Nominal baseline
            nom_mean_avg = isempty(model_means[:nominal]) ? NaN : mean(model_means[:nominal])
            nom_p95_avg = isempty(model_p95s[:nominal]) ? NaN : mean(model_p95s[:nominal])

            for m_name in model_names
                avg_mean = isempty(model_means[m_name]) ? NaN : mean(model_means[m_name])
                avg_p95 = isempty(model_p95s[m_name]) ? NaN : mean(model_p95s[m_name])

                # VFR: (model - nominal) / nominal  (leader minimizes → lower is better)
                vfr_mean = isnan(nom_mean_avg) || isnan(avg_mean) ? NaN :
                           (avg_mean - nom_mean_avg) / max(abs(nom_mean_avg), 1e-10)
                vfr_p95 = isnan(nom_p95_avg) || isnan(avg_p95) ? NaN :
                          (avg_p95 - nom_p95_avg) / max(abs(nom_p95_avg), 1e-10)

                @printf("  %-6.2f | %-8.4f | %-12s | %-12.4f | %-12.4f | %-8.4f | %-8.4f\n",
                        β, ε_cal, m_name, avg_mean, avg_p95, vfr_mean, vfr_p95)
            end
            println()
        end
    end
end


# ===== Quick test (고정: 3×3, S=2) =====

function run_oos_quick_test()
    println("=" ^ 60)
    println("OOS Quick Test: Grid 3×3, S=2, β=0.5, M=10, L=100")
    println("=" ^ 60)

    _run_oos_test(:grid_3x3, S=2, β=0.5, M=10, L=100, seed=1)
end


# ===== Custom test (user input) =====

function run_oos_custom_test()
    println("=" ^ 60)
    println("OOS Custom Test")
    println("=" ^ 60)

    # Network
    println("Network options: $(collect(keys(network_configs)))")
    print("Network [grid_3x3]: ")
    net_input = strip(readline())
    net_key = isempty(net_input) ? :grid_3x3 : Symbol(net_input)

    print("S [10]: ")
    S_input = strip(readline())
    S = isempty(S_input) ? 10 : parse(Int, S_input)

    print("β [0.5]: ")
    β_input = strip(readline())
    β = isempty(β_input) ? 0.5 : parse(Float64, β_input)

    print("M (outer OOS) [100]: ")
    M_input = strip(readline())
    M = isempty(M_input) ? 100 : parse(Int, M_input)

    print("L (inner OOS) [1000]: ")
    L_input = strip(readline())
    L = isempty(L_input) ? 1000 : parse(Int, L_input)

    print("inst_seed [1]: ")
    seed_input = strip(readline())
    seed = isempty(seed_input) ? 1 : parse(Int, seed_input)

    _run_oos_test(net_key; S=S, β=β, M=M, L=L, seed=seed)
end


# ===== Shared test logic =====

function _run_oos_test(net_key::Symbol; S=2, β=0.5, M=10, L=100, seed=1)
    # ε calibration
    ε = calibrate_epsilon(S, β; n_cal=10000, coverage=0.95)
    @printf("Calibrated ε = %.6f (S=%d, β=%.2f)\n\n", ε, S, β)

    # Instance
    network, capacities, γ, w = setup_oos_instance(net_key; S=S, seed=seed)
    num_arcs = length(network.arcs) - 1
    @printf("Network: %s, |A|=%d, γ=%d, w=%.4f, inst_seed=%d\n\n", net_key, num_arcs, γ, w, seed)

    # Solve 3 models
    model_results = Dict{Symbol, Any}()

    println("── Nominal ──")
    x_nom, obj_nom, t_nom = solve_model_nominal(network, capacities, γ, w, OOS_V)
    model_results[:nominal] = (x=x_nom, obj=obj_nom, time=t_nom)
    @printf("  obj=%.4f, time=%.1fs, x=%s\n\n", obj_nom, t_nom, round.(Int, x_nom))

    @printf("── Single-DRO (ε̂=%.4f, ε̃=0) ──\n", ε)
    x_s, obj_s, t_s, it_s, st_s = solve_model_dro(network, capacities, γ, w, ε, 0.0;
                                                     max_iter=200)
    model_results[:single_dro] = (x=x_s, obj=obj_s, time=t_s)
    @printf("  obj=%.4f, time=%.1fs, iters=%d, x=%s\n\n", obj_s, t_s, it_s, round.(Int, x_s))

    @printf("── Two-layer DRO (ε̂=ε̃=%.4f) ──\n", ε)
    x_t, obj_t, t_t, it_t, st_t = solve_model_dro(network, capacities, γ, w, ε, ε;
                                                     max_iter=200)
    model_results[:twolayer_dro] = (x=x_t, obj=obj_t, time=t_t)
    @printf("  obj=%.4f, time=%.1fs, iters=%d, x=%s\n\n", obj_t, t_t, it_t, round.(Int, x_t))

    # OOS evaluate all 3
    @printf("── OOS Evaluation (M=%d, L=%d, oos_seed=%d) ──\n", M, L, OOS_SEED)
    for m_name in [:nominal, :single_dro, :twolayer_dro]
        mr = model_results[m_name]
        println("\n  Model: $m_name")
        oos = oos_evaluate(mr.x, network, capacities, β, OOS_V, w;
                            M=M, L=L, seed=OOS_SEED)
        @printf("    OOS mean=%.4f, p95=%.4f, follower_share=%.4f\n",
                oos[:mean], oos[:p95], oos[:follower_share])
    end

    return model_results
end


# ===== Entry point (직접 실행 시에만) =====
if abspath(PROGRAM_FILE) == @__FILE__
    println("\nSelect mode:")
    println("  1) Full experiment (inst_seed=1 only)")
    println("  2) Full experiment + generalize (inst_seed=1..10)")
    println("  3) Quick test (Grid 3x3, S=2, β=0.5)")
    println("  4) Custom test (user input)")
    print("Choice [3]: ")
    choice = strip(readline())

    if choice == "1"
        run_oos_experiment(generalize=false)
    elseif choice == "2"
        run_oos_experiment(generalize=true)
    elseif choice == "4"
        run_oos_custom_test()
    else
        run_oos_quick_test()
    end
end
