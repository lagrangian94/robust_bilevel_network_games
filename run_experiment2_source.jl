"""
Experiment 2: Source Decomposition — Exp1 결과 재사용 + OOS re-evaluation

Exp1에서 풀린 x*를 가져와서 OOS만 다른 조건으로 평가.
  Scenario A: Exp1 FO의 x* → test = in-sample caps (P_true = P̂)
  Scenario B: Exp1 FM의 x* → test = pool에서 leader가 안 본 시나리오

solve 없음 — evaluate_oos()만 실행.

실행:
  julia -t 8 run_experiment2_source.jl
"""

using JuMP
using HiGHS
using LinearAlgebra
using Statistics
using Serialization
using Printf
using Revise
using Dates
using Random
using Plots

includet("network_generator.jl")
includet("oos_evaluation.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model
using .NetworkGenerator: generate_sioux_falls_network, generate_nobel_us_network, generate_abilene_network, generate_polska_network

# ===== Network Configs =====
network_configs = Dict(
    :grid_3x3 => Dict(:type => :grid, :m => 3, :n => 3),
    :grid_4x4 => Dict(:type => :grid, :m => 4, :n => 4),
    :grid_5x5 => Dict(:type => :grid, :m => 5, :n => 5),
    :sioux_falls => Dict(:type => :real_world, :generator => generate_sioux_falls_network),
    :nobel_us => Dict(:type => :real_world, :generator => generate_nobel_us_network),
    :abilene => Dict(:type => :real_world, :generator => generate_abilene_network),
    :polska => Dict(:type => :real_world, :generator => generate_polska_network),
)

# ===== Defaults =====
EXP_SEED = 42
EXP_S = 10
EXP_S_FOLLOWER = 10
EXP_RHO = 0.2
EXP_V = 1.0
EXP_GAMMA_RATIO = 0.1
EXP_EPSILONS = [0.5]
EXP_K_POOL = 200          # Scenario B pool size
EXP_NETWORK = :grid_5x5
SEED_POOL = 99             # Scenario B pool 생성용 seed

CHECKPOINT_FILE = "output/experiment2_source_grid_5x5.jls"
EXP1_RESULTS_FILE = ""     # Exp1 결과 파일 경로 (run_all에서 설정)

# ===== Helpers =====

"""네트워크 생성 (solve 없이 network struct만)"""
function make_network(config_key::Symbol; seed=42)
    config = network_configs[config_key]
    if config[:type] == :grid
        return generate_grid_network(config[:m], config[:n], seed=seed)
    else
        return config[:generator]()
    end
end

"""w 계산 (ρ·γ·c̄). solve 불필요, 시나리오 통계만."""
function compute_w(network, S, γ_ratio, ρ, seed)
    num_arcs = length(network.arcs) - 1
    capacities, _ = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = ceil(Int, γ_ratio * num_interdictable)
    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    return round(ρ * γ * c_bar, digits=4)
end

"""Exp1 결과에서 variant의 x* 로드"""
function load_exp1_solution(exp1_file::String, net_key::Symbol, γ_ratio, ε, variant::Symbol)
    if !isfile(exp1_file)
        error("Exp1 결과 파일 없음: $(exp1_file)")
    end
    exp1 = deserialize(exp1_file)
    rkey = "$(net_key)_γ$(γ_ratio)_ε$(ε)_$(variant)"
    if !haskey(exp1, rkey)
        error("Exp1에 키 없음: $(rkey). 가용 키: $(collect(keys(exp1)))")
    end
    r = exp1[rkey]
    if haskey(r, :error)
        error("Exp1 $(rkey) 실패 상태: $(r[:error])")
    end
    return r
end

# ===== Checkpoint =====
function load_checkpoint()
    if isfile(CHECKPOINT_FILE)
        println("Loading checkpoint from $CHECKPOINT_FILE...")
        return deserialize(CHECKPOINT_FILE)
    end
    return Dict{String, Any}()
end

function save_checkpoint(results)
    tmp = CHECKPOINT_FILE * ".tmp"
    serialize(tmp, results)
    mv(tmp, CHECKPOINT_FILE, force=true)
end

# ===== Scenario A: Follower belief isolation =====
"""
Exp1 FO의 x* → OOS with test = in-sample caps (P_true = P̂).
Baseline: Exp1 N의 x*.
"""
function run_scenario_A(results)
    println("\n" * "="^80)
    println("SCENARIO A: Follower Belief Isolation (Exp1 x* reuse)")
    println("  FO x* → test = in-sample caps")
    println("="^80)

    net_key = EXP_NETWORK
    network = make_network(net_key, seed=EXP_SEED)
    num_arcs_total = length(network.arcs)
    w = compute_w(network, EXP_S, EXP_GAMMA_RATIO, EXP_RHO, EXP_SEED)

    # In-sample caps = P_true (P̂)
    in_sample_caps, _ = generate_capacity_scenarios_uniform_model(num_arcs_total, EXP_S, seed=EXP_SEED)
    # Follower belief (독립)
    follower_caps, _ = generate_capacity_scenarios_uniform_model(num_arcs_total, EXP_S_FOLLOWER, seed=EXP_SEED+1)

    # ε=0 baseline + nonzero ε
    all_eps = sort(unique([0.0; EXP_EPSILONS]))

    for ε in all_eps
        rkey = "A_ε$(ε)"
        if haskey(results, rkey) && !haskey(results[rkey], :error)
            println("  [skip] $(rkey) — already done")
            continue
        end

        println("\n" * "─"^70)
        println("[Scenario A] ε=$(ε) | $(Dates.format(now(), "HH:MM:SS"))")
        println("─"^70)

        try
            # Exp1에서 x* 로드
            variant = ε == 0.0 ? :N : :FO
            # FO는 ε̂=0이므로 Exp1에서 ε_hat=0, ε_tilde=ε
            # result_key 형식: Exp1은 ε 파라미터 하나로 키 생성 (N/FO/TO/FM)
            exp1_ε = ε == 0.0 ? EXP_EPSILONS[1] : ε  # N은 ε 무관이지만 Exp1 키에 ε 포함
            # N variant는 모든 ε에서 동일 → Exp1에서 해당 ε로 저장됨
            exp1_result = load_exp1_solution(EXP1_RESULTS_FILE, net_key, EXP_GAMMA_RATIO, exp1_ε, variant)

            x_star = exp1_result[:x_star]
            obj_insample = exp1_result[:obj_insample]

            println("  Loaded Exp1 $(variant) x*: $(x_star)")
            println("  Exp1 in-sample obj: $(round(obj_insample, digits=4))")

            # OOS: test_caps = in_sample_caps (P_true = P̂)
            mean_flow, std_flow, h_star = evaluate_oos(
                network, x_star, EXP_V, w, follower_caps, in_sample_caps)

            println("  OOS: mean=$(round(mean_flow, digits=4)) ± $(round(std_flow, digits=4))")

            results[rkey] = Dict(
                :scenario => :A,
                :epsilon => ε,
                :variant_source => variant,
                :x_star => x_star,
                :h_star => h_star,
                :obj_insample => obj_insample,
                :oos_mean => mean_flow,
                :oos_std => std_flow,
            )
        catch e
            @warn "FAILED: $(rkey)" exception=(e, catch_backtrace())
            results[rkey] = Dict(:scenario => :A, :epsilon => ε, :error => string(e))
        end

        save_checkpoint(results)
    end

    return results
end

# ===== Scenario B: Leader info gap isolation =====
"""
Exp1 FM의 x* → OOS with test = pool에서 leader가 안 본 시나리오.
Baseline: Exp1 N의 x*.
"""
function run_scenario_B(results)
    println("\n" * "="^80)
    println("SCENARIO B: Leader Info Gap Isolation (Exp1 x* reuse)")
    println("  FM x* → test = pool unseen scenarios")
    println("="^80)

    net_key = EXP_NETWORK
    network = make_network(net_key, seed=EXP_SEED)
    num_arcs_total = length(network.arcs)
    w = compute_w(network, EXP_S, EXP_GAMMA_RATIO, EXP_RHO, EXP_SEED)

    # Generate large pool
    pool_caps, _ = generate_capacity_scenarios_uniform_model(num_arcs_total, EXP_K_POOL, seed=SEED_POOL)
    println("  Pool generated: $(EXP_K_POOL) scenarios, seed=$(SEED_POOL)")

    # Leader subsample (Exp1이 본 S개를 시뮬레이션)
    rng = MersenneTwister(EXP_SEED)
    leader_idx = randperm(rng, EXP_K_POOL)[1:EXP_S]
    rest_idx = setdiff(1:EXP_K_POOL, leader_idx)
    println("  Leader subsample: $(length(leader_idx)) scenarios")

    test_caps_B = pool_caps[:, rest_idx]           # leader가 안 본 나머지
    follower_caps_B = pool_caps                    # follower는 전체 pool

    # ε=0 baseline + nonzero ε
    all_eps = sort(unique([0.0; EXP_EPSILONS]))

    for ε in all_eps
        rkey = "B_ε$(ε)"
        if haskey(results, rkey) && !haskey(results[rkey], :error)
            println("  [skip] $(rkey) — already done")
            continue
        end

        println("\n" * "─"^70)
        println("[Scenario B] ε=$(ε) | $(Dates.format(now(), "HH:MM:SS"))")
        println("─"^70)

        try
            # Exp1에서 x* 로드
            variant = ε == 0.0 ? :N : :FM
            exp1_ε = ε == 0.0 ? EXP_EPSILONS[1] : ε
            exp1_result = load_exp1_solution(EXP1_RESULTS_FILE, net_key, EXP_GAMMA_RATIO, exp1_ε, variant)

            x_star = exp1_result[:x_star]
            obj_insample = exp1_result[:obj_insample]

            println("  Loaded Exp1 $(variant) x*: $(x_star)")
            println("  Exp1 in-sample obj: $(round(obj_insample, digits=4))")

            # OOS: test = pool 나머지, follower = pool 전체
            mean_flow, std_flow, h_star = evaluate_oos(
                network, x_star, EXP_V, w, follower_caps_B, test_caps_B)

            println("  OOS: mean=$(round(mean_flow, digits=4)) ± $(round(std_flow, digits=4))")

            results[rkey] = Dict(
                :scenario => :B,
                :epsilon => ε,
                :variant_source => variant,
                :x_star => x_star,
                :h_star => h_star,
                :obj_insample => obj_insample,
                :oos_mean => mean_flow,
                :oos_std => std_flow,
                :leader_idx => leader_idx,
            )
        catch e
            @warn "FAILED: $(rkey)" exception=(e, catch_backtrace())
            results[rkey] = Dict(:scenario => :B, :epsilon => ε, :error => string(e))
        end

        save_checkpoint(results)
    end

    return results
end

# ===== Results Output =====

function print_results(results)
    println("\n" * "="^80)
    println("EXPERIMENT 2: SOURCE DECOMPOSITION RESULTS")
    println("="^80)

    nonzero_eps = filter(e -> e > 0, EXP_EPSILONS)

    # === Scenario A ===
    baseline_A = haskey(results, "A_ε0.0") && haskey(results["A_ε0.0"], :oos_mean) ?
        results["A_ε0.0"][:oos_mean] : nothing

    println("\n── Scenario A (Follower belief isolation) ──")
    if !isnothing(baseline_A)
        println("  Baseline (N): OOS = $(round(baseline_A, digits=4))")
    end

    eps_A = Float64[]
    delta_A = Float64[]

    if !isempty(nonzero_eps)
        println("┌────────┬────────────┬────────────┐")
        println("│   ε    │   OOS_A    │   Δ_A(%)   │")
        println("├────────┼────────────┼────────────┤")

        for ε in nonzero_eps
            rkey = "A_ε$(ε)"
            if haskey(results, rkey) && haskey(results[rkey], :oos_mean)
                oos = results[rkey][:oos_mean]
                if !isnothing(baseline_A) && baseline_A > 1e-8
                    δ = (baseline_A - oos) / baseline_A * 100.0
                    println(@sprintf("│ %5.2f  │ %9.3f  │ %8.2f%%  │", ε, oos, δ))
                    push!(eps_A, ε)
                    push!(delta_A, δ)
                else
                    println(@sprintf("│ %5.2f  │ %9.3f  │    ---     │", ε, oos))
                end
            else
                println(@sprintf("│ %5.2f  │    ---     │    ---     │", ε))
            end
        end
        println("└────────┴────────────┴────────────┘")
    end

    # === Scenario B ===
    baseline_B = haskey(results, "B_ε0.0") && haskey(results["B_ε0.0"], :oos_mean) ?
        results["B_ε0.0"][:oos_mean] : nothing

    println("\n── Scenario B (Leader info gap isolation) ──")
    if !isnothing(baseline_B)
        println("  Baseline (N): OOS = $(round(baseline_B, digits=4))")
    end

    eps_B = Float64[]
    delta_B = Float64[]

    if !isempty(nonzero_eps)
        println("┌────────┬────────────┬────────────┐")
        println("│   ε    │   OOS_B    │   Δ_B(%)   │")
        println("├────────┼────────────┼────────────┤")

        for ε in nonzero_eps
            rkey = "B_ε$(ε)"
            if haskey(results, rkey) && haskey(results[rkey], :oos_mean)
                oos = results[rkey][:oos_mean]
                if !isnothing(baseline_B) && baseline_B > 1e-8
                    δ = (baseline_B - oos) / baseline_B * 100.0
                    println(@sprintf("│ %5.2f  │ %9.3f  │ %8.2f%%  │", ε, oos, δ))
                    push!(eps_B, ε)
                    push!(delta_B, δ)
                else
                    println(@sprintf("│ %5.2f  │ %9.3f  │    ---     │", ε, oos))
                end
            else
                println(@sprintf("│ %5.2f  │    ---     │    ---     │", ε))
            end
        end
        println("└────────┴────────────┴────────────┘")
    end

    # Plot Δ curves
    if !isempty(eps_A) || !isempty(eps_B)
        plot_delta_curves(eps_A, delta_A, eps_B, delta_B)
    end
end

function plot_delta_curves(eps_A, delta_A, eps_B, delta_B)
    p = plot(title="Experiment 2: Source Decomposition ($(EXP_NETWORK))",
             xlabel="ε", ylabel="Δ (%)", legend=:topright,
             size=(700, 450), grid=true)

    if !isempty(eps_A)
        plot!(p, eps_A, delta_A, label="A: Follower belief", color=:blue,
              marker=:circle, linewidth=2, markersize=6)
        idx_best_A = argmax(delta_A)
        annotate!(p, eps_A[idx_best_A], delta_A[idx_best_A] + 0.5,
                  text("ε*_A=$(eps_A[idx_best_A])", 8, :blue))
    end

    if !isempty(eps_B)
        plot!(p, eps_B, delta_B, label="B: Leader info gap", color=:red,
              marker=:square, linewidth=2, markersize=6)
        idx_best_B = argmax(delta_B)
        annotate!(p, eps_B[idx_best_B], delta_B[idx_best_B] + 0.5,
                  text("ε*_B=$(eps_B[idx_best_B])", 8, :red))
    end

    hline!(p, [0.0], color=:gray, linestyle=:dash, label=nothing)

    fig_path = joinpath(dirname(CHECKPOINT_FILE), "exp2_delta_$(EXP_NETWORK).png")
    savefig(p, fig_path)
    println("\n  Figure saved: $(fig_path)")
end

# ===== Interactive Input =====
# 직접 실행할 때만 interactive input, include()로 불릴 때는 skip
if (abspath(PROGRAM_FILE) == @__FILE__) && !("--test" in ARGS)
    println("="^80)
    println("EXPERIMENT 2: SOURCE DECOMPOSITION")
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
        EXP_NETWORK = Symbol("grid_$(m)x$(n)")
    elseif net_choice == 2
        EXP_NETWORK = :sioux_falls
    elseif net_choice == 3
        EXP_NETWORK = :nobel_us
    elseif net_choice == 4
        EXP_NETWORK = :abilene
    elseif net_choice == 5
        EXP_NETWORK = :polska
    else
        error("잘못된 선택: $net_choice")
    end

    print("Exp1 결과 파일 경로: "); EXP1_RESULTS_FILE = strip(readline())

    print("시나리오 수 S [$EXP_S]: "); s_str = strip(readline())
    if !isempty(s_str); EXP_S = parse(Int, s_str); end
    EXP_S_FOLLOWER = EXP_S

    print("ε values (comma-separated) [$(join(EXP_EPSILONS, ","))]: "); eps_str = strip(readline())
    if !isempty(eps_str); EXP_EPSILONS = parse.(Float64, split(eps_str, ",")); end

    print("Pool size K_pool (Scenario B) [$EXP_K_POOL]: "); kp_str = strip(readline())
    if !isempty(kp_str); EXP_K_POOL = parse(Int, kp_str); end

    CHECKPOINT_FILE = "output/experiment2_source_$(EXP_NETWORK).jls"

    println("\n" * "="^80)
    println("설정 확인:")
    println("  Network:   $(EXP_NETWORK)")
    println("  Exp1 file: $(EXP1_RESULTS_FILE)")
    println("  S=$(EXP_S), γ_ratio=$(EXP_GAMMA_RATIO), ε=$(EXP_EPSILONS)")
    println("  K_pool=$(EXP_K_POOL) (Scenario B)")
    println("  seed=$(EXP_SEED), S_f=$(EXP_S_FOLLOWER)")
    println("  ρ=$(EXP_RHO), v=$(EXP_V)")
    println("  Checkpoint: $(CHECKPOINT_FILE)")
    println("="^80)
end

# ===== Main =====
function run_experiment2()
    if isempty(EXP1_RESULTS_FILE) || !isfile(EXP1_RESULTS_FILE)
        error("EXP1_RESULTS_FILE 설정 필요: '$(EXP1_RESULTS_FILE)' (파일 없음)")
    end

    results = load_checkpoint()

    println("="^80)
    println("EXPERIMENT 2: SOURCE DECOMPOSITION (OOS re-eval)")
    println("="^80)
    println("  Network: $(EXP_NETWORK)")
    println("  Exp1: $(EXP1_RESULTS_FILE)")
    println("  S=$(EXP_S), ε=$(EXP_EPSILONS), K_pool=$(EXP_K_POOL)")
    println("="^80)

    run_scenario_A(results)
    run_scenario_B(results)

    print_results(results)

    println("\n" * "="^80)
    println("EXPERIMENT 2 COMPLETE")
    println("="^80)

    return results
end

# ===== Entry Point =====
if abspath(PROGRAM_FILE) == @__FILE__
    if "--test" in ARGS
        # Quick test
        global EXP_NETWORK = :grid_3x3
        global EXP_S = 2
        global EXP_S_FOLLOWER = 2
        global EXP_EPSILONS = [0.5]
        global EXP_K_POOL = 20
        global CHECKPOINT_FILE = "output/experiment2_source_test.jls"
        # EXP1_RESULTS_FILE must be set
        run_experiment2()
    else
        run_experiment2()
    end
end
