"""
run_oos_phase_b.jl — Phase B OOS evaluation (asymmetric Dirichlet, OOS mean gap).

E[q_true] ≠ uniform 상황을 만들어 DRO 가치를 OOS mean으로 측정.
Outer loop: α = β·(1 + noise_scale·randn(K)) 고정 → p_center ≠ uniform.
Inner loop: p_true, q_tilde ~ Dir(α) — 또는 shortcut (p_center 직접 사용).

Usage:
    include("run_oos_phase_b.jl")
    # 또는 run_oos_experiment.jl에서 호출
"""

# Phase A의 공유 함수들 (oos_regenerate_network, parse_all_x_stars 등)
if !@isdefined(oos_regenerate_network)
    include(joinpath(@__DIR__, "run_oos_phase_a.jl"))
end


# ============================================================
# Phase B main
# ============================================================

function run_oos_phase_b(;
        beta_values   = [0.1, 0.3, 0.5],
        net_keys      = [:grid_5x5, :polska, :abilene, :nobel_us, :sioux_falls],
        S::Int        = 20,
        M::Int        = 100,
        R::Int        = 100,
        noise_scale   = 0.5,
        seed::Int     = 42,
        use_shortcut  = true,   # f_share≈0 → p_center 직접 계산
        log_suffix    = "wmax")

    println("=" ^ 80)
    println("Phase B OOS Evaluation (Asymmetric Dirichlet, OOS Mean Gap)")
    println("  Models: Nominal / Single-layer / Two-layer DRO")
    println("  β values: $beta_values")
    println("  Networks: $net_keys")
    println("  S=$S, M=$M, R=$R, noise_scale=$noise_scale, shortcut=$use_shortcut, seed=$seed")
    println("=" ^ 80)

    all_results = Dict{Tuple{Symbol, Float64}, Dict}()
    summary_rows = []

    for net_key in net_keys
        network, capacities, w, γ = oos_regenerate_network(net_key; S=S)
        num_arcs = length(network.arcs) - 1

        for β in beta_values
            ε_hat = round(lookup_epsilon(S, β; coverage=0.95); digits=2)
            @printf("\n[%s] β=%.1f, ε̂=%.2f\n", net_key, β, ε_hat)

            # Parse x* for all 3 models
            x_stars_raw = parse_all_x_stars(net_key, β; S=S, log_suffix=log_suffix)
            for (mkey, (x, z, src)) in x_stars_raw
                @printf("  [%-10s] Z₀=%.4f, Σx=%d, source=%s\n", mkey, z, sum(x), src)
            end

            # x* 동일 여부 체크
            x_vectors = Dict(k => v[1] for (k, v) in x_stars_raw)
            all_same = true
            ref_x = nothing
            for (k, x) in x_vectors
                if ref_x === nothing
                    ref_x = x
                elseif x != ref_x
                    all_same = false
                    break
                end
            end
            if all_same && length(x_vectors) > 1
                @printf("  ⚠ All models have same x* — Phase B gap will be 0. Skipping.\n")
                # 기록은 하되 gap=0 표시
                push!(summary_rows, (
                    network       = string(net_key),
                    beta          = β,
                    gap_mean      = 0.0,
                    gap_p5        = 0.0,
                    gap_p95       = 0.0,
                    dro_wins      = NaN,
                    gap_sl_mean   = 0.0,
                    gap_sl_wins   = NaN,
                    same_x        = true,
                ))
                continue
            end

            # 필요한 3 모델 x* → Dict{Symbol, Vector}
            if !haskey(x_vectors, :nominal) || !haskey(x_vectors, :single) || !haskey(x_vectors, :two_layer)
                missing_keys = filter(k -> !haskey(x_vectors, k), [:nominal, :single, :two_layer])
                @warn "Missing model(s), skipping Phase B" net_key β missing_keys
                continue
            end

            x_stars_dict = Dict{Symbol, Vector{Float64}}(
                :nominal   => x_vectors[:nominal],
                :single    => x_vectors[:single],
                :two_layer => x_vectors[:two_layer],
            )

            @printf("  Running Phase B OOS (M=%d, %s)...\n", M,
                    use_shortcut ? "shortcut" : "R=$R")

            pb_result = oos_evaluate_phase_b(x_stars_dict, network, capacities, β, 1.0, w;
                                               M=M, R=R, noise_scale=Float64(noise_scale),
                                               seed=seed, use_shortcut=use_shortcut)

            all_results[(net_key, β)] = pb_result

            # Pairwise same_x 체크
            td_nom_same = x_vectors[:two_layer] == x_vectors[:nominal]
            td_sl_same  = x_vectors[:two_layer] == x_vectors[:single]

            push!(summary_rows, (
                network       = string(net_key),
                beta          = β,
                gap_mean      = td_nom_same ? 0.0 : pb_result[:gap_mean],
                gap_p5        = td_nom_same ? 0.0 : pb_result[:gap_p5],
                gap_p95       = td_nom_same ? 0.0 : pb_result[:gap_p95],
                dro_wins      = td_nom_same ? NaN : pb_result[:dro_wins],
                gap_sl_mean   = td_sl_same ? 0.0 : pb_result[:gap_vs_single_mean],
                gap_sl_wins   = td_sl_same ? NaN : pb_result[:gap_vs_single_wins],
                same_x        = all_same,
            ))
        end
    end

    # ---- Save ----
    jls_path = joinpath(@__DIR__, "oos_phase_b_results.jls")
    serialize(jls_path, all_results)
    println("\nRaw results saved → $jls_path")

    csv_path = joinpath(@__DIR__, "oos_phase_b_summary.csv")
    open(csv_path, "w") do io
        println(io, "network,beta,gap_mean,gap_p5,gap_p95,dro_wins,gap_sl_mean,gap_sl_wins,same_x")
        for r in summary_rows
            dro_str = isnan(r.dro_wins) ? "" : @sprintf("%.4f", r.dro_wins)
            sl_str  = isnan(r.gap_sl_wins) ? "" : @sprintf("%.4f", r.gap_sl_wins)
            @printf(io, "%s,%.1f,%.6f,%.6f,%.6f,%s,%.6f,%s,%s\n",
                    r.network, r.beta, r.gap_mean, r.gap_p5, r.gap_p95, dro_str,
                    r.gap_sl_mean, sl_str, r.same_x)
        end
    end
    println("Summary CSV saved → $csv_path")

    # ---- Print table ----
    println("\n" * "=" ^ 100)
    println("Phase B OOS Summary (Asymmetric Dirichlet, OOS Mean Gap)")
    println("  gap < 0 → DRO wins (minimization)")
    println("=" ^ 100)
    @printf("%-12s %4s  %10s  %10s  %10s  %8s  %10s  %8s  %6s\n",
            "Network", "β", "Gap Mean", "Gap p5", "Gap p95", "DRO Win%", "Gap(SL)", "SL Win%", "same_x")
    println("-" ^ 100)

    for r in summary_rows
        dro_str = if r.same_x
            " same_x"
        elseif isnan(r.dro_wins)
            "   ---"
        else
            @sprintf("%7.1f%%", r.dro_wins * 100)
        end
        sl_str = if r.same_x
            " same_x"
        elseif isnan(r.gap_sl_wins)
            "   ---"
        else
            @sprintf("%7.1f%%", r.gap_sl_wins * 100)
        end
        @printf("%-12s %4.1f  %10.4f  %10.4f  %10.4f  %s  %10.4f  %s  %6s\n",
                r.network, r.beta, r.gap_mean, r.gap_p5, r.gap_p95, dro_str,
                r.gap_sl_mean, sl_str, r.same_x ? "yes" : "no")
    end
    println("=" ^ 100)

    return all_results, summary_rows
end
