"""
run_vor_sensitivity.jl — VOR (Value of Robustness) ε sensitivity sweep.

Spec §8 Phase 1: 각 β에서 calibrated ε*를 구한 후,
  ε/ε* ∈ {0, 0.25, 0.5, 1.0, 1.5, 2.0} sweep.

기대 결과: inverted-U shape. ε 너무 작으면 overfitting, 너무 크면 overconservatism.

실행:
  include("true_dro/run_oos_experiment.jl")  # 이미 로드된 경우 생략
  include("true_dro/run_vor_sensitivity.jl")
"""

# ===== 공통 코드 로드 =====
if !@isdefined(run_oos_experiment)
    include("run_oos_experiment.jl")
end

# ===== VOR Constants =====
const VOR_EPSILON_RATIOS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
const VOR_CHECKPOINT_FILE = "true_dro/vor_sensitivity_checkpoint.jls"


# ===== Checkpoint =====

function load_vor_checkpoint()
    isfile(VOR_CHECKPOINT_FILE) ? deserialize(VOR_CHECKPOINT_FILE) : Dict{String,Any}()
end

function save_vor_checkpoint(results)
    serialize(VOR_CHECKPOINT_FILE, results)
end

function vor_result_key(net_key, β, instance_id, model_name, eps_ratio)
    "$(net_key)_beta$(β)_inst$(instance_id)_$(model_name)_r$(eps_ratio)"
end


# ===== Main VOR Sensitivity =====

"""
    run_vor_sensitivity(; networks, β_values, ε_ratios, generalize, S, M, L, ...)

VOR ε sweep: 각 (network, β, instance)에 대해
  1. calibrate ε* from Dir(β)
  2. ratio ∈ ε_ratios에 대해 ε = ratio × ε*
  3. Nominal은 ε-independent → ratio=0에서 한번만 solve
  4. single_dro, twolayer_dro는 각 ratio에서 solve + OOS evaluate
"""
function run_vor_sensitivity(;
    networks=OOS_NETWORKS,
    β_values=OOS_BETA_VALUES,
    ε_ratios=VOR_EPSILON_RATIOS,
    generalize::Bool=false,
    S=OOS_S,
    M=OOS_M,
    L=OOS_L,
    n_cal=OOS_N_CAL,
    coverage=OOS_COVERAGE,
    verbose=true)

    n_instances = generalize ? OOS_N_INSTANCES : 1
    results = load_vor_checkpoint()

    # Nominal은 ratio별로 안 바뀌므로 model_configs에서 분리
    dro_models = [
        (:single_dro,   :eps, 0.0),
        (:twolayer_dro, :eps, :eps),
    ]

    total_runs = length(networks) * length(β_values) * n_instances *
                 (1 + length(dro_models) * (length(ε_ratios) - 1))  # nominal 1회 + DRO × (ratios-1)
    completed = count(v -> !haskey(v, :error), values(results))

    println("=" ^ 80)
    println("VOR SENSITIVITY: ε/ε* Sweep")
    println("=" ^ 80)
    println("  Networks: $networks")
    println("  β values: $β_values")
    println("  ε/ε* ratios: $ε_ratios")
    println("  generalize=$generalize → n_instances=$n_instances")
    println("  S=$S, M=$M, L=$L")
    println("  Total runs: ~$total_runs, already done: $completed")
    println("=" ^ 80)

    run_count = 0

    for net_key in networks
        for β in β_values
            # ── ε calibration ──
            ε_cal = calibrate_epsilon(S, β; n_cal=n_cal, coverage=coverage)
            @printf("\n[%s, β=%.2f] calibrated ε* = %.6f\n", net_key, β, ε_cal)

            for inst in 1:n_instances
                inst_seed = inst
                network, capacities, γ, w_val = setup_oos_instance(net_key; S=S, seed=inst_seed)
                num_arcs = length(network.arcs) - 1

                @printf("  Instance %d: inst_seed=%d, |A|=%d, γ=%d, w=%.4f\n",
                        inst, inst_seed, num_arcs, γ, w_val)

                # ── Nominal (ratio-independent, 한번만) ──
                nom_rkey = vor_result_key(net_key, β, inst, :nominal, 0.0)
                if !haskey(results, nom_rkey) || haskey(results[nom_rkey], :error)
                    run_count += 1
                    println("\n" * "─" ^ 70)
                    @printf("[%d] %s | β=%.2f | inst=%d | nominal\n",
                            run_count, net_key, β, inst)
                    println("─" ^ 70)

                    try
                        x_nom, obj_nom, t_nom = solve_model_nominal(network, capacities, γ, w_val, OOS_V)
                        oos_nom = oos_evaluate(x_nom, network, capacities, β, OOS_V, w_val;
                                                M=M, L=L, seed=OOS_SEED)

                        results[nom_rkey] = Dict(
                            :network => net_key, :beta => β, :instance => inst,
                            :model => :nominal, :eps_ratio => 0.0,
                            :epsilon_cal => ε_cal, :epsilon_hat => 0.0, :epsilon_tilde => 0.0,
                            :x_star => x_nom, :obj_insample => obj_nom, :solve_time => t_nom,
                            :oos_mean => oos_nom[:mean], :oos_p95 => oos_nom[:p95],
                            :var_outer => oos_nom[:var_outer], :var_inner => oos_nom[:var_inner],
                            :follower_share => oos_nom[:follower_share],
                        )
                    catch e
                        @warn "FAILED: $nom_rkey" exception=(e, catch_backtrace())
                        results[nom_rkey] = Dict(
                            :network => net_key, :beta => β, :instance => inst,
                            :model => :nominal, :error => string(e),
                        )
                    end
                    save_vor_checkpoint(results)
                end

                # ── DRO models × ε ratios ──
                for ratio in ε_ratios
                    ratio == 0.0 && continue  # ratio=0 → ε=0 → nominal과 동일, skip

                    ε_sweep = ratio * ε_cal

                    for (model_name, eps_hat_spec, eps_tilde_spec) in dro_models
                        rkey = vor_result_key(net_key, β, inst, model_name, ratio)

                        if haskey(results, rkey) && !haskey(results[rkey], :error)
                            continue
                        end

                        eps_hat = eps_hat_spec === :eps ? ε_sweep : Float64(eps_hat_spec)
                        eps_tilde = eps_tilde_spec === :eps ? ε_sweep : Float64(eps_tilde_spec)

                        run_count += 1
                        println("\n" * "─" ^ 70)
                        @printf("[%d] %s | β=%.2f | inst=%d | %s | ε/ε*=%.2f (ε̂=%.4f, ε̃=%.4f)\n",
                                run_count, net_key, β, inst, model_name, ratio, eps_hat, eps_tilde)
                        println("  $(Dates.format(now(), "HH:MM:SS"))")
                        println("─" ^ 70)

                        try
                            x_star, obj_val, solve_time, n_iters, status =
                                solve_model_dro(network, capacities, γ, w_val, eps_hat, eps_tilde)

                            @printf("  In-sample: obj=%.4f, time=%.1fs, iters=%d\n",
                                    obj_val, solve_time, n_iters)

                            oos_result = oos_evaluate(x_star, network, capacities, β, OOS_V, w_val;
                                                       M=M, L=L, seed=OOS_SEED)

                            results[rkey] = Dict(
                                :network => net_key, :beta => β, :instance => inst,
                                :model => model_name, :eps_ratio => ratio,
                                :epsilon_cal => ε_cal,
                                :epsilon_hat => eps_hat, :epsilon_tilde => eps_tilde,
                                :x_star => x_star, :obj_insample => obj_val,
                                :solve_time => solve_time, :n_iters => n_iters, :status => status,
                                :oos_mean => oos_result[:mean], :oos_p95 => oos_result[:p95],
                                :var_outer => oos_result[:var_outer], :var_inner => oos_result[:var_inner],
                                :follower_share => oos_result[:follower_share],
                            )

                        catch e
                            @warn "FAILED: $rkey" exception=(e, catch_backtrace())
                            results[rkey] = Dict(
                                :network => net_key, :beta => β, :instance => inst,
                                :model => model_name, :eps_ratio => ratio,
                                :error => string(e),
                            )
                        end

                        save_vor_checkpoint(results)
                    end  # model
                end  # ratio
            end  # instance
        end  # β
    end  # network

    println("\n" * "=" ^ 80)
    println("VOR SENSITIVITY COMPLETE")
    println("=" ^ 80)

    print_vor_summary(results, networks, β_values, ε_ratios, generalize ? OOS_N_INSTANCES : 1)

    return results
end


# ===== Summary =====

function print_vor_summary(results, networks, β_values, ε_ratios, n_instances)
    println("\n" * "=" ^ 80)
    println("VOR SENSITIVITY SUMMARY")
    println("=" ^ 80)

    model_names = [:nominal, :single_dro, :twolayer_dro]

    for net_key in networks
        println("\n══ $(net_key) ══")

        for β in β_values
            @printf("\n  β = %.2f\n", β)
            @printf("  %-8s | %-12s | %-12s | %-12s | %-10s\n",
                    "ε/ε*", "Model", "OOS Mean", "OOS p95", "VOR(%)")
            println("  " * "-" ^ 65)

            # Nominal baseline (ratio=0)
            nom_means = Float64[]
            for inst in 1:n_instances
                rkey = vor_result_key(net_key, β, inst, :nominal, 0.0)
                if haskey(results, rkey) && !haskey(results[rkey], :error)
                    push!(nom_means, results[rkey][:oos_mean])
                end
            end
            nom_avg = isempty(nom_means) ? NaN : mean(nom_means)

            for ratio in ε_ratios
                for m_name in model_names
                    # Nominal은 ratio=0만
                    if m_name == :nominal && ratio != 0.0
                        continue
                    end
                    # DRO는 ratio>0만
                    if m_name != :nominal && ratio == 0.0
                        continue
                    end

                    r_key = vor_result_key(net_key, β, 1, m_name,
                                           m_name == :nominal ? 0.0 : ratio)

                    # Collect across instances
                    means = Float64[]
                    p95s = Float64[]
                    for inst in 1:n_instances
                        rk = vor_result_key(net_key, β, inst, m_name,
                                            m_name == :nominal ? 0.0 : ratio)
                        if haskey(results, rk) && !haskey(results[rk], :error)
                            push!(means, results[rk][:oos_mean])
                            push!(p95s, results[rk][:oos_p95])
                        end
                    end

                    avg_mean = isempty(means) ? NaN : mean(means)
                    avg_p95 = isempty(p95s) ? NaN : mean(p95s)

                    # VOR = (Nominal - Model) / Nominal × 100%
                    # Leader minimizes → lower OOS mean is better → VOR > 0 means model wins
                    vor = isnan(nom_avg) || isnan(avg_mean) || abs(nom_avg) < 1e-10 ? NaN :
                          (nom_avg - avg_mean) / abs(nom_avg) * 100.0

                    display_ratio = m_name == :nominal ? 0.0 : ratio
                    @printf("  %-8.2f | %-12s | %-12.4f | %-12.4f | %-10.2f\n",
                            display_ratio, m_name, avg_mean, avg_p95, vor)
                end
            end
            println()
        end
    end
end
