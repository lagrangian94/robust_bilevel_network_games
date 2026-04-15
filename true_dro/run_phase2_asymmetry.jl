"""
run_phase2_asymmetry.jl — Phase 2: Information Asymmetry Scenarios (L, F).

Spec §8 Phase 2:
  Scenario S (Symmetric): 기본 실험 β=0.3 결과 재사용 (여기서 안 품)
  Scenario L (Leader Advantage): P* ≈ q̂, P̃ 부정확
    - q_true ~ Dir(β_H·1), q̃ ~ Dir(β_L·1)
    - ε₁: Dir(β_H=50) calibrate → 작음
    - ε₂: Dir(β_L=0.3) calibrate → 큼
    - Two-layer DRO 가치 가장 큼
  Scenario F (Follower Advantage): P̃ ≈ P*, P̂ 부정확
    - q_true ~ Dir(β_L·1), q̃ ~ Dir(κ·q_true)
    - ε₁ = ε₂: Dir(β_L=0.3) calibrate → 큼 (과대추정)
    - Two-layer DRO overconservative 예상

실행:
  include("true_dro/run_oos_experiment.jl")  # 이미 로드된 경우 생략
  include("true_dro/run_phase2_asymmetry.jl")
"""

# ===== 공통 코드 로드 =====
if !@isdefined(run_oos_experiment)
    include("run_oos_experiment.jl")
end

# ===== Phase 2 Constants =====
const P2_BETA_L = 0.3       # low concentration (부정확한 쪽)
const P2_BETA_H = 50.0      # high concentration (정확한 쪽)
const P2_KAPPA = 50.0        # Scenario F: q̃ ~ Dir(κ·q_true)
const P2_CHECKPOINT_FILE = "true_dro/phase2_asymmetry_checkpoint.jls"


# ===== Checkpoint =====

function load_p2_checkpoint()
    isfile(P2_CHECKPOINT_FILE) ? deserialize(P2_CHECKPOINT_FILE) : Dict{String,Any}()
end

function save_p2_checkpoint(results)
    serialize(P2_CHECKPOINT_FILE, results)
end

function p2_result_key(net_key, scenario, instance_id, model_name)
    "$(net_key)_$(scenario)_inst$(instance_id)_$(model_name)"
end


# ===== Main Phase 2 =====

"""
    run_phase2_experiment(; networks, generalize, S, M, L, ...)

Phase 2: Information Asymmetry Scenarios.

Scenario L: ε₁ from Dir(β_H), ε₂ from Dir(β_L) → asymmetric ε
Scenario F: ε₁ = ε₂ from Dir(β_L) → symmetric but overestimate for follower

각 scenario에서 3 models: nominal, single_dro, twolayer_dro
"""
function run_phase2_experiment(;
    networks=OOS_NETWORKS,
    generalize::Bool=false,
    S=OOS_S,
    M=OOS_M,
    L=OOS_L,
    n_cal=OOS_N_CAL,
    coverage=OOS_COVERAGE,
    verbose=true)

    n_instances = generalize ? OOS_N_INSTANCES : 1
    results = load_p2_checkpoint()

    # ── ε calibration (network-independent, S와 β만 의존) ──
    ε_L = calibrate_epsilon(S, P2_BETA_L; n_cal=n_cal, coverage=coverage)  # large
    ε_H = calibrate_epsilon(S, P2_BETA_H; n_cal=n_cal, coverage=coverage)  # small

    @printf("Phase 2 ε calibration: ε(β_L=%.1f) = %.6f, ε(β_H=%.1f) = %.6f\n",
            P2_BETA_L, ε_L, P2_BETA_H, ε_H)

    # Scenario configs: (name, ε_hat, ε_tilde, oos_func_args)
    scenarios = [
        (:L, ε_H, ε_L),    # Leader advantage: ε₁ small, ε₂ large
        (:F, ε_L, ε_L),    # Follower advantage: ε₁=ε₂ large (overestimate)
    ]

    model_configs = [
        (:nominal,      0.0, 0.0),
        (:single_dro,   :eps_hat, 0.0),
        (:twolayer_dro, :eps_hat, :eps_tilde),
    ]

    total_runs = length(networks) * length(scenarios) * n_instances * length(model_configs)
    completed = count(v -> !haskey(v, :error), values(results))

    println("=" ^ 80)
    println("PHASE 2: INFORMATION ASYMMETRY SCENARIOS")
    println("=" ^ 80)
    println("  Networks: $networks")
    println("  Scenarios: L (leader advantage), F (follower advantage)")
    println("  β_L=$(P2_BETA_L), β_H=$(P2_BETA_H), κ=$(P2_KAPPA)")
    println("  ε(β_L)=$(round(ε_L, digits=6)), ε(β_H)=$(round(ε_H, digits=6))")
    println("  generalize=$generalize → n_instances=$n_instances")
    println("  S=$S, M=$M, L=$L")
    println("  Total runs: $total_runs, already done: $completed")
    println("=" ^ 80)

    run_count = completed

    for net_key in networks
        for (scenario_name, ε_hat_base, ε_tilde_base) in scenarios
            @printf("\n══ %s | Scenario %s (ε̂=%.4f, ε̃=%.4f) ══\n",
                    net_key, scenario_name, ε_hat_base, ε_tilde_base)

            for inst in 1:n_instances
                inst_seed = inst
                network, capacities, γ, w_val = setup_oos_instance(net_key; S=S, seed=inst_seed)
                num_arcs = length(network.arcs) - 1

                @printf("  Instance %d: |A|=%d, γ=%d, w=%.4f\n", inst, num_arcs, γ, w_val)

                for (model_name, eps_hat_spec, eps_tilde_spec) in model_configs
                    rkey = p2_result_key(net_key, scenario_name, inst, model_name)

                    if haskey(results, rkey) && !haskey(results[rkey], :error)
                        run_count += 1
                        continue
                    end

                    # Resolve ε
                    eps_hat = eps_hat_spec === :eps_hat ? ε_hat_base : Float64(eps_hat_spec)
                    eps_tilde = eps_tilde_spec === :eps_tilde ? ε_tilde_base : Float64(eps_tilde_spec)

                    run_count += 1
                    println("\n" * "─" ^ 70)
                    @printf("[%d/%d] %s | Scenario %s | inst=%d | %s (ε̂=%.4f, ε̃=%.4f)\n",
                            run_count, total_runs, net_key, scenario_name, inst,
                            model_name, eps_hat, eps_tilde)
                    println("  $(Dates.format(now(), "HH:MM:SS"))")
                    println("─" ^ 70)

                    try
                        # ── In-sample solve ──
                        x_star, obj_val, solve_time, n_iters, status = if model_name == :nominal
                            x, obj, t = solve_model_nominal(network, capacities, γ, w_val, OOS_V)
                            (x, obj, t, 0, :Optimal)
                        else
                            solve_model_dro(network, capacities, γ, w_val, eps_hat, eps_tilde)
                        end

                        @printf("  In-sample: obj=%.4f, time=%.1fs, iters=%d, status=%s\n",
                                obj_val, solve_time, n_iters, status)
                        println("  x* = $(round.(Int, x_star))")

                        # ── OOS evaluation (scenario-specific) ──
                        oos_result = if scenario_name == :L
                            oos_evaluate_scenario_L(x_star, network, capacities,
                                                      P2_BETA_L, P2_BETA_H, OOS_V, w_val;
                                                      M=M, L=L, seed=OOS_SEED)
                        elseif scenario_name == :F
                            oos_evaluate_scenario_F(x_star, network, capacities,
                                                      P2_BETA_L, P2_KAPPA, OOS_V, w_val;
                                                      M=M, seed=OOS_SEED)
                        end

                        # ── Store ──
                        results[rkey] = Dict(
                            :network => net_key, :scenario => scenario_name,
                            :instance => inst, :model => model_name,
                            :epsilon_hat => eps_hat, :epsilon_tilde => eps_tilde,
                            :epsilon_cal_L => ε_L, :epsilon_cal_H => ε_H,
                            :x_star => x_star, :obj_insample => obj_val,
                            :solve_time => solve_time, :n_iters => n_iters, :status => status,
                            :oos_mean => oos_result[:mean], :oos_p95 => oos_result[:p95],
                            :var_outer => oos_result[:var_outer],
                            :var_inner => oos_result[:var_inner],
                            :follower_share => oos_result[:follower_share],
                        )

                    catch e
                        @warn "FAILED: $rkey" exception=(e, catch_backtrace())
                        results[rkey] = Dict(
                            :network => net_key, :scenario => scenario_name,
                            :instance => inst, :model => model_name,
                            :error => string(e),
                        )
                    end

                    save_p2_checkpoint(results)
                end  # model
            end  # instance
        end  # scenario
    end  # network

    println("\n" * "=" ^ 80)
    println("PHASE 2 COMPLETE")
    println("=" ^ 80)

    print_p2_summary(results, networks, n_instances)

    return results
end


# ===== Summary =====

function print_p2_summary(results, networks, n_instances;
                           baseline_file=OOS_CHECKPOINT_FILE)
    println("\n" * "=" ^ 80)
    println("PHASE 2 SUMMARY: Information Asymmetry")
    println("=" ^ 80)

    # Baseline (Scenario S) from basic experiment β=0.3
    baseline_results = isfile(baseline_file) ? deserialize(baseline_file) : Dict{String,Any}()

    model_names = [:nominal, :single_dro, :twolayer_dro]
    scenario_names = [:S, :L, :F]

    for net_key in networks
        println("\n══ $(net_key) ══")
        @printf("  %-10s | %-12s | %-10s | %-10s | %-10s | %-12s\n",
                "Scenario", "Model", "OOS Mean", "OOS p95", "VOR(%)", "Foll. Share")
        println("  " * "-" ^ 70)

        for scenario in scenario_names
            # Nominal baseline for VOR
            nom_mean = NaN

            for m_name in model_names
                means = Float64[]
                p95s = Float64[]
                f_shares = Float64[]

                for inst in 1:n_instances
                    if scenario == :S
                        # Baseline 실험에서 β=0.3 결과 재사용
                        rkey = oos_result_key(net_key, P2_BETA_L, inst, m_name)
                        src = baseline_results
                    else
                        rkey = p2_result_key(net_key, scenario, inst, m_name)
                        src = results
                    end

                    if haskey(src, rkey) && !haskey(src[rkey], :error)
                        push!(means, src[rkey][:oos_mean])
                        push!(p95s, src[rkey][:oos_p95])
                        if haskey(src[rkey], :follower_share)
                            push!(f_shares, src[rkey][:follower_share])
                        end
                    end
                end

                avg_mean = isempty(means) ? NaN : mean(means)
                avg_p95 = isempty(p95s) ? NaN : mean(p95s)
                avg_fs = isempty(f_shares) ? NaN : mean(f_shares)

                if m_name == :nominal
                    nom_mean = avg_mean
                end

                vor = isnan(nom_mean) || isnan(avg_mean) || abs(nom_mean) < 1e-10 ? NaN :
                      (nom_mean - avg_mean) / abs(nom_mean) * 100.0

                @printf("  %-10s | %-12s | %-10.4f | %-10.4f | %-10.2f | %-12.4f\n",
                        scenario, m_name, avg_mean, avg_p95, vor, avg_fs)
            end
            println()
        end
    end
end
