"""
Experiment 1: Value of Full Model (VFM)

4가지 모델 variant (N, FO, TO, FM)를 in-sample solve + OOS evaluation으로 비교.
"양쪽 hedge(leader + follower)가 모두 필요한가" 증명.

실행:
  julia -t 8 run_experiment1_vfm.jl

모델 variants:
  N  (Nominal):       ε_hat=0, ε_tilde=0
  FO (Follower-Only): ε_hat=0, ε_tilde=ε
  TO (True-Only):     ε_hat=ε, ε_tilde=0
  FM (Full Model):    ε_hat=ε, ε_tilde=ε
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using LinearAlgebra
using Statistics
using Serialization
using Printf
using Revise
using Dates
using Infiltrator

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("build_full_model.jl")
includet("parallel_utils.jl")
includet("strict_benders.jl")
includet("nested_benders_trust_region.jl")
includet("build_nominal_sp.jl")
includet("oos_evaluation.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary
using .NetworkGenerator: generate_sioux_falls_network, generate_nobel_us_network, generate_abilene_network, generate_polska_network, print_realworld_network_summary

# ===== Network Configs =====
network_configs = Dict(
    :grid_3x3 => Dict(:type => :grid, :m => 3, :n => 3),
    :grid_5x5 => Dict(:type => :grid, :m => 5, :n => 5),
    :sioux_falls => Dict(:type => :real_world, :generator => generate_sioux_falls_network),
    :nobel_us => Dict(:type => :real_world, :generator => generate_nobel_us_network),
    :abilene => Dict(:type => :real_world, :generator => generate_abilene_network),
    :polska => Dict(:type => :real_world, :generator => generate_polska_network),
)

# Model variants: (name, isp_mode)
# ε=0 side uses exact LP ISP (no SDP numerical issues, no ϕU=1/ε blow-up)
const MODEL_VARIANTS = [
    (:N,  :nominal),           # Nominal SP (solve_nominal path)
    (:FO, :partial_hat0),      # ε̂=0 → LP leader ISP, ε̃=ε → SDP follower ISP
    (:TO, :partial_tilde0),    # ε̂=ε → SDP leader ISP, ε̃=0 → LP follower ISP
    (:FM, :dual),              # ε̂=ε, ε̃=ε → both SDP ISP (standard)
]

# ===== Defaults =====
EXP_NETWORKS = [:grid_5x5]
EXP_S = 10
EXP_GAMMA_RATIOS = [0.05, 0.10]
EXP_EPSILONS = [0.25, 0.5, 1.0]
EXP_SEED = 42
EXP_K_TEST = 100
EXP_S_FOLLOWER = 10
EXP_RHO = 0.2
EXP_V = 1.0
CHECKPOINT_FILE = "output/experiment1_vfm_checkpoint.jls"

# ===== Interactive Input (compare_benders.jl과 동일 패턴) =====
# 직접 실행할 때만 interactive input, include()로 불릴 때는 skip
if (abspath(PROGRAM_FILE) == @__FILE__) && !("--test" in ARGS)
    println("="^80)
    println("EXPERIMENT 1: VALUE OF FULL MODEL (VFM)")
    println("="^80)

    println("\n네트워크 인스턴스 선택")
    println("  1. Grid network")
    println("  2. Sioux Falls (24 nodes, 76 arcs)")
    println("  3. Nobel US (14 nodes, 38 arcs)")
    println("  4. Abilene (12 nodes, 30 arcs)")
    println("  5. Polska (12 nodes, 36 arcs)")
    print("선택 (1-5): ")
    net_choice = parse(Int, readline())

    if net_choice == 1
        print("Grid rows (m): "); m = parse(Int, readline())
        print("Grid cols (n): "); n = parse(Int, readline())
        network_configs[Symbol("grid_$(m)x$(n)")] = Dict(:type => :grid, :m => m, :n => n)
        EXP_NETWORKS = [Symbol("grid_$(m)x$(n)")]
    elseif net_choice == 2
        EXP_NETWORKS = [:sioux_falls]
    elseif net_choice == 3
        EXP_NETWORKS = [:nobel_us]
    elseif net_choice == 4
        EXP_NETWORKS = [:abilene]
    elseif net_choice == 5
        EXP_NETWORKS = [:polska]
    else
        error("잘못된 선택: $net_choice")
    end

    print("시나리오 수 S [$EXP_S]: "); s_str = strip(readline())
    if !isempty(s_str); EXP_S = parse(Int, s_str); end

    EXP_GAMMA_RATIOS = [0.1]  # 고정
    EXP_S_FOLLOWER = EXP_S    # in-sample S와 동일

    print("ε values (comma-separated) [$(join(EXP_EPSILONS, ","))]: "); eps_str = strip(readline())
    if !isempty(eps_str); EXP_EPSILONS = parse.(Float64, split(eps_str, ",")); end

    print("OOS test scenarios K_test [$EXP_K_TEST]: "); kt_str = strip(readline())
    if !isempty(kt_str); EXP_K_TEST = parse(Int, kt_str); end

    # 네트워크별 checkpoint 파일
    CHECKPOINT_FILE = "output/experiment1_vfm_$(EXP_NETWORKS[1]).jls"

    println("\n" * "="^80)
    println("설정 확인:")
    println("  Networks:  $(EXP_NETWORKS)")
    println("  S=$(EXP_S), γ_ratio=0.1, ε=$(EXP_EPSILONS)")
    println("  seed=$(EXP_SEED), K_test=$(EXP_K_TEST), S_f=$(EXP_S_FOLLOWER)")
    println("  ρ=$(EXP_RHO), v=$(EXP_V)")
    println("  Checkpoint: $(CHECKPOINT_FILE)")
    println("="^80)
end

function load_checkpoint()
    if isfile(CHECKPOINT_FILE)
        println("Loading checkpoint from $CHECKPOINT_FILE...")
        return deserialize(CHECKPOINT_FILE)
    end
    return Dict{String, Any}()
end

function save_checkpoint(results)
    # atomic write: tmp → rename (JLD2 corruption 방지)
    tmp = CHECKPOINT_FILE * ".tmp"
    serialize(tmp, results)
    mv(tmp, CHECKPOINT_FILE, force=true)
end

function result_key(net_key, γ_ratio, ε, variant)
    return "$(net_key)_γ$(γ_ratio)_ε$(ε)_$(variant)"
end

# ===== Setup Instance (adapted from compare_benders.jl) =====
function setup_instance_vfm(config_key::Symbol;
    S=10, γ_ratio=0.10, ρ=0.2, v=1.0, seed=42,
    epsilon_hat=0.5, epsilon_tilde=epsilon_hat)

    config = network_configs[config_key]

    # Network 생성
    if config[:type] == :grid
        network = generate_grid_network(config[:m], config[:n], seed=seed)
    elseif config[:type] == :real_world
        network = config[:generator]()
    end

    num_arcs = length(network.arcs) - 1  # dummy 제외

    # Parameters: ϕU computation handles ε=0 by borrowing from the nonzero side
    if epsilon_hat == 0.0 && epsilon_tilde == 0.0
        # Both nominal: dummy values (solve_nominal path doesn't use these for Benders)
        ϕU_hat = 10.0
        ϕU_tilde = 10.0
    elseif epsilon_hat == 0.0
        ϕU_tilde = 1.0 / epsilon_tilde
        ϕU_hat = ϕU_tilde  # LP leader ISP uses this as McCormick big-M
    elseif epsilon_tilde == 0.0
        ϕU_hat = 1.0 / epsilon_hat
        ϕU_tilde = ϕU_hat  # LP follower ISP uses this as McCormick big-M
    else
        ϕU_hat = 1.0 / epsilon_hat
        ϕU_tilde = 1.0 / epsilon_tilde
    end
    λU = max(ϕU_hat, ϕU_tilde)
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = ceil(Int, γ_ratio * num_interdictable)

    capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(ρ * γ * c_bar, digits=4)

    # Uncertainty set
    capacity_scenarios_regular = capacities[1:end-1, :]
    R, r_dict_hat, r_dict_tilde, xi_bar = build_robust_counterpart_matrices(
        capacity_scenarios_regular, epsilon_hat, epsilon_tilde)
    uncertainty_set = Dict(
        :R => R, :r_dict_hat => r_dict_hat, :r_dict_tilde => r_dict_tilde,
        :xi_bar => xi_bar, :epsilon_hat => epsilon_hat, :epsilon_tilde => epsilon_tilde)

    # LDR coefficient bounds
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
        :γ_ratio => γ_ratio, :ρ => ρ, :seed => seed,
    )

    return network, uncertainty_set, params
end

# ===== Solve one variant =====
"""
    solve_nominal(network, uncertainty_set, params) -> (x_star, obj_val, solve_time, num_iters)

Nominal SP (N variant): build_full_2SP_model로 직접 풀기. ε 무관.
"""
function solve_nominal(net, uncertainty_set, params; max_time=7200.0)
    γ = params[:γ]
    ϕU_hat = params[:ϕU_hat]
    λU = params[:λU]
    w = params[:w]
    v_param = params[:v]
    S_val = params[:S]

    GC.gc()

    t_start = time()
    model, vars = build_full_2SP_model(net, S_val, ϕU_hat, λU, γ, w, v_param, uncertainty_set)
    set_optimizer_attribute(model, "TimeLimit", max_time)
    optimize!(model)
    solve_time = time() - t_start

    st = termination_status(model)
    if st ∉ (MOI.OPTIMAL, MOI.TIME_LIMIT)
        error("Nominal SP failed: $st")
    end

    time_limit_hit = (st == MOI.TIME_LIMIT)
    x_star = value.(vars[:x])
    obj_val = objective_value(model)
    h_star = value.(vars[:h])
    λ_star = value(vars[:λ])
    opt_gap = time_limit_hit ? relative_gap(model) : 0.0

    return x_star, obj_val, solve_time, 0, h_star, λ_star, opt_gap, time_limit_hit
end

"""
    solve_variant(network, uncertainty_set, params; isp_mode=:dual) -> (x_star, obj_val, solve_time)

Benders decomposition으로 in-sample 문제 풀기.
isp_mode: :dual (standard), :partial_hat0 (ε̂=0), :partial_tilde0 (ε̃=0)
"""
function solve_variant(net, uncertainty_set, params; isp_mode=:dual, max_time=7200.0)
    γ = params[:γ]
    ϕU_hat = params[:ϕU_hat]
    ϕU_tilde = params[:ϕU_tilde]
    λU = params[:λU]
    w = params[:w]
    πU_hat = params[:πU_hat]
    πU_tilde = params[:πU_tilde]
    yU = params[:yU]
    ytsU = params[:ytsU]

    if isp_model != :dual
        max_time = 3600
    end

    # tr_nested_benders_optimize!()가 global v, S, network를 참조 (compare_benders.jl과 동일 패턴)
    global v = params[:v]
    global S = params[:S]
    global network = net

    GC.gc()

    # Build OMP
    model, vars = build_omp(network, ϕU_hat, λU, γ, w; optimizer=Gurobi.Optimizer, S=S)

    # Solve
    t_start = time()
    result = tr_nested_benders_optimize!(model, vars, network, ϕU_hat, ϕU_tilde, λU, γ, w, uncertainty_set;
        mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
        outer_tr=true, inner_tr=true, isp_mode=isp_mode, max_time=max_time,
        πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU,
        strengthen_cuts=:mw,  # partial mode: SDP side only MW (LP side는 기존 cut 유지)
        parallel=true, mini_benders=true, max_mini_benders_iter=3,
        ldr_mode=:both)
    solve_time = time() - t_start

    # Extract best solution (global UB 기준)
    ub_vec = result[:past_upper_bound]
    lb_vec = result[:past_lower_bound]
    obj_val = minimum(ub_vec)
    num_iters = length(ub_vec)

    # Optimality gap
    best_lb = isempty(lb_vec) ? -Inf : maximum(lb_vec)
    opt_gap = abs(obj_val) > 1e-8 ? (obj_val - best_lb) / abs(obj_val) : Inf
    time_limit_hit = solve_time >= max_time

    # Best UB에 해당하는 solution 사용
    x_star = result[:opt_sol][:x]
    h_star = result[:opt_sol][:h]
    λ_star = result[:opt_sol][:λ]

    return x_star, obj_val, solve_time, num_iters, h_star, λ_star, opt_gap, time_limit_hit
end

# ===== Main Experiment Loop =====
function run_experiment()
    results = load_checkpoint()
    total_runs = length(EXP_NETWORKS) * length(EXP_GAMMA_RATIOS) * length(EXP_EPSILONS) *
                 length(MODEL_VARIANTS)
    completed = count(v -> !haskey(v, :error), values(results))
    println("="^80)
    println("EXPERIMENT 1: VALUE OF FULL MODEL (VFM)")
    println("="^80)
    println("  Networks: $(EXP_NETWORKS)")
    println("  Epsilons: $(EXP_EPSILONS)")
    println("  γ ratios: $(EXP_GAMMA_RATIOS)")
    println("  S=$(EXP_S), S_f=$(EXP_S_FOLLOWER), K_test=$(EXP_K_TEST), seed=$(EXP_SEED)")
    println("  Total runs: $(total_runs), already done: $(completed)")
    println("="^80)

    run_count = completed

    for net_key in EXP_NETWORKS
        for γ_ratio in EXP_GAMMA_RATIOS
            for ε in EXP_EPSILONS
                for (variant_name, isp_mode) in MODEL_VARIANTS
                    rkey = result_key(net_key, γ_ratio, ε, variant_name)
                    if haskey(results, rkey) && !haskey(results[rkey], :error)
                        continue  # already done (에러 항목은 재실행)
                    end

                    # Compute ε̂/ε̃ from isp_mode
                    ε_hat = isp_mode in (:nominal, :partial_hat0) ? 0.0 : ε
                    ε_tilde = isp_mode in (:nominal, :partial_tilde0) ? 0.0 : ε

                    run_count += 1
                    println("\n" * "─"^70)
                    println("[$(run_count)/$(total_runs)] $(net_key) | γ=$(γ_ratio) | ε=$(ε) | " *
                            "$(variant_name) (ε̂=$(ε_hat), ε̃=$(ε_tilde))")
                    println("  $(Dates.format(now(), "HH:MM:SS"))")
                    println("─"^70)

                    # === In-sample solve ===
                    try
                        network, uncertainty_set, params = setup_instance_vfm(net_key;
                            S=EXP_S, γ_ratio=γ_ratio, ρ=EXP_RHO, v=EXP_V,
                            seed=EXP_SEED,
                            epsilon_hat=ε_hat, epsilon_tilde=ε_tilde)

                        x_star, obj_val, solve_time, num_iters, h_insample, λ_insample, opt_gap, time_limit_hit = if variant_name == :N
                            solve_nominal(network, uncertainty_set, params)
                        else
                            solve_variant(network, uncertainty_set, params; isp_mode=isp_mode)
                        end

                        println("  In-sample: obj=$(round(obj_val, digits=4)), " *
                                "time=$(round(solve_time, digits=1))s, iters=$(num_iters)" *
                                (time_limit_hit ? ", TIME LIMIT (gap=$(round(opt_gap*100, digits=2))%)" : ""))
                        println("  x* = $(x_star)")

                        # === OOS evaluation ===
                        # Follower belief scenarios (seed+1)
                        follower_caps, _ = generate_capacity_scenarios_uniform_model(
                            length(network.arcs), EXP_S_FOLLOWER, seed=EXP_SEED+1)

                        # Test scenarios (seed+2)
                        test_caps, _ = generate_capacity_scenarios_uniform_model(
                            length(network.arcs), EXP_K_TEST, seed=EXP_SEED+2)

                        println("  Evaluating OOS (K_test=$(EXP_K_TEST))...")
                        mean_flow, std_flow, h_star = evaluate_oos(
                            network, x_star, EXP_V, params[:w], follower_caps, test_caps)

                        println("  OOS: mean=$(round(mean_flow, digits=4)) ± $(round(std_flow, digits=4))")

                        # Store result
                        results[rkey] = Dict(
                            :network => net_key,
                            :gamma_ratio => γ_ratio,
                            :epsilon => ε,
                            :variant => variant_name,
                            :epsilon_hat => ε_hat,
                            :epsilon_tilde => ε_tilde,
                            :x_star => x_star,
                            :h_star => h_star,
                            :h_insample => h_insample,
                            :lambda_insample => λ_insample,
                            :obj_insample => obj_val,
                            :oos_mean => mean_flow,
                            :oos_std => std_flow,
                            :solve_time => solve_time,
                            :num_iters => num_iters,
                            :opt_gap => opt_gap,
                            :time_limit_hit => time_limit_hit,
                        )

                    catch e
                        @warn "FAILED: $(rkey)" exception=(e, catch_backtrace())
                        results[rkey] = Dict(
                            :network => net_key,
                            :gamma_ratio => γ_ratio,
                            :epsilon => ε,
                            :variant => variant_name,
                            :error => string(e),
                        )
                    end

                    # Checkpoint after each run
                    save_checkpoint(results)


                end  # variant
            end  # ε
        end  # γ_ratio
    end  # network

    println("\n" * "="^80)
    println("EXPERIMENT COMPLETE")
    println("="^80)

    # Print summary tables
    print_summary_tables(results)

    return results
end

# ===== Summary Table =====
function print_summary_tables(results)
    println("\n" * "="^80)
    println("SUMMARY TABLES")
    println("="^80)

    for net_key in EXP_NETWORKS
        for γ_ratio in EXP_GAMMA_RATIOS
            println("\n── $(net_key), γ_ratio=$(γ_ratio) ──")
            println("┌────────┬────────────┬────────────┬────────────┬────────────┐")
            println("│   ε    │     N      │     FO     │     TO     │     FM     │")
            println("├────────┼────────────┼────────────┼────────────┼────────────┤")

            for ε in EXP_EPSILONS
                row = @sprintf("│ %5.2f  ", ε)
                for (variant_name, _) in MODEL_VARIANTS
                    rkey = result_key(net_key, γ_ratio, ε, variant_name)
                    if haskey(results, rkey) && haskey(results[rkey], :oos_mean)
                        m = results[rkey][:oos_mean]
                        row *= @sprintf("│ %9.3f  ", m)
                    else
                        row *= "│     ---    "
                    end
                end
                row *= "│"
                println(row)
            end
            println("└────────┴────────────┴────────────┴────────────┴────────────┘")
        end
    end
end

# ===== Quick Test Mode =====
"""
    run_quick_test()

Grid 3×3, S=2, ε=0.5, R=1 파이프라인 검증.
"""
function run_quick_test()
    println("="^60)
    println("QUICK TEST: Grid 3×3, S=2, ε=0.5, seed=42")
    println("="^60)

    net_key = :grid_3x3
    ε = 0.5
    γ_ratio = 0.10
    seed = 42

    test_results = Dict{Symbol, Any}()

    for (variant_name, isp_mode) in MODEL_VARIANTS
        ε_hat = isp_mode in (:nominal, :partial_hat0) ? 0.0 : ε
        ε_tilde = isp_mode in (:nominal, :partial_tilde0) ? 0.0 : ε

        println("\n── $(variant_name) (ε̂=$(ε_hat), ε̃=$(ε_tilde), isp_mode=$(isp_mode)) ──")

        network, uncertainty_set, params = setup_instance_vfm(net_key;
            S=2, γ_ratio=γ_ratio, ρ=EXP_RHO, v=EXP_V,
            seed=seed,
            epsilon_hat=ε_hat, epsilon_tilde=ε_tilde)

        x_star, obj_val, solve_time, num_iters, h_insample, λ_insample, opt_gap, time_limit_hit = if variant_name == :N
            solve_nominal(network, uncertainty_set, params)
        else
            solve_variant(network, uncertainty_set, params; isp_mode=isp_mode)
        end

        println("  In-sample: obj=$(round(obj_val, digits=4)), time=$(round(solve_time, digits=1))s")
        println("  x* = $(x_star)")

        # OOS
        follower_caps, _ = generate_capacity_scenarios_uniform_model(
            length(network.arcs), 10, seed=seed+1)
        test_caps, _ = generate_capacity_scenarios_uniform_model(
            length(network.arcs), 100, seed=seed+2)

        mean_flow, std_flow, h_star = evaluate_oos(
            network, x_star, EXP_V, params[:w], follower_caps, test_caps)

        println("  OOS: mean=$(round(mean_flow, digits=4)) ± $(round(std_flow, digits=4))")

        test_results[variant_name] = Dict(
            :x_star => x_star, :obj => obj_val, :oos_mean => mean_flow, :oos_std => std_flow)
    end

    # Sanity check: OOS_FM ≤ OOS_N (FM이 더 effective interdiction → lower flow)
    if haskey(test_results, :FM) && haskey(test_results, :N)
        fm_flow = test_results[:FM][:oos_mean]
        n_flow = test_results[:N][:oos_mean]
        println("\n── Sanity Check ──")
        println("  OOS_FM = $(round(fm_flow, digits=4))")
        println("  OOS_N  = $(round(n_flow, digits=4))")
        if fm_flow <= n_flow + 1e-4
            println("  ✓ PASS: FM ≤ N (FM is more effective)")
        else
            @warn "  ✗ FAIL: FM > N — check implementation!"
        end
    end

    return test_results
end

# ===== Entry Point =====
if abspath(PROGRAM_FILE) == @__FILE__
    if "--test" in ARGS
        run_quick_test()
    else
        run_experiment()
    end
end
