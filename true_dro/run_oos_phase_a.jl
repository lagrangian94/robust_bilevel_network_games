"""
run_oos_phase_a.jl — Phase A OOS evaluation (symmetric Dirichlet, tail metrics).

3-variant 비교 (all from pre-computed logs):
  - Nominal:       S{S}_nominal_wmax/ 에서 x* 파싱
  - Single-layer:  S{S}_beta{β}_wmax/ 에서 *_single*.txt x* 파싱
  - Two-layer DRO: S{S}_beta{β}_wmax/ 에서 *_wmax.txt (non-single) x* 파싱

Bug workaround: final summary x*가 all-zeros인 로그 → last OMP line의 x= 사용.

Usage:
    include("run_oos_phase_a.jl")
    # 또는 run_oos_experiment.jl에서 호출
"""


# ============================================================
# 1. Log 파싱 (last OMP x fallback)
# ============================================================

"""
    parse_benders_log(filepath) -> Dict

로그에서 x*, Z0, status 파싱.
- Nominal: "Nominal SP: Z₀=..." + "x* = [...]"
- True-DRO/Single: summary line + last OMP "x=[...]" (fallback if final x* is all-zeros)
"""
function parse_benders_log(filepath::String)
    lines = readlines(filepath)

    x_star_summary = nothing    # final summary의 x*
    x_star_omp = nothing        # last OMP line의 x
    Z0 = NaN
    status = "Unknown"
    iters = 0
    wall_time = NaN
    eps_hat = NaN
    eps_tilde = NaN
    variant = :unknown

    for line in lines
        # ε values
        m_eps = match(r"ε̂=([\d.]+),\s*ε̃=([\d.]+)", line)
        if m_eps !== nothing
            eps_hat = parse(Float64, m_eps.captures[1])
            eps_tilde = parse(Float64, m_eps.captures[2])
        end

        # True-DRO / Single-layer status
        m_td = match(r"(True-DRO|Single-layer):\s*status=(\w+),\s*Z₀=([\d.]+),\s*iters=(\d+),\s*time=([\d.]+)s", line)
        if m_td !== nothing
            variant = m_td.captures[1] == "Single-layer" ? :single : :two_layer
            status = m_td.captures[2]
            Z0 = parse(Float64, m_td.captures[3])
            iters = parse(Int, m_td.captures[4])
            wall_time = parse(Float64, m_td.captures[5])
        end

        # Nominal SP status
        m_nom = match(r"Nominal SP:\s*Z₀=([\d.]+),\s*time=([\d.]+)s", line)
        if m_nom !== nothing
            variant = :nominal
            status = "Optimal"
            Z0 = parse(Float64, m_nom.captures[1])
            wall_time = parse(Float64, m_nom.captures[2])
        end

        # OMP line: x=[...] — 매번 업데이트 (마지막 iteration 값 유지)
        m_omp = match(r"OMP:.*x=\[([0-9,\s]+)\]", line)
        if m_omp !== nothing
            x_star_omp = parse.(Int, split(m_omp.captures[1], r"[,\s]+"; keepempty=false))
        end

        # Summary x* = [...] (status 파싱 이후에만)
        if !isnan(Z0) && x_star_summary === nothing
            m_x = match(r"^\s*x\*\s*=\s*\[([0-9,\s]+)\]", line)
            if m_x !== nothing
                x_star_summary = parse.(Int, split(m_x.captures[1], r"[,\s]+"; keepempty=false))
            end
        end
    end

    # x* 결정: summary가 all-zeros이면 last OMP 사용
    x_star = x_star_summary
    x_source = :summary
    if x_star !== nothing && all(x_star .== 0) && x_star_omp !== nothing
        x_star = x_star_omp
        x_source = :last_omp
    end
    # summary 자체가 없으면 OMP 사용
    if x_star === nothing && x_star_omp !== nothing
        x_star = x_star_omp
        x_source = :last_omp
    end

    return Dict(
        :x_star => x_star,
        :x_source => x_source,
        :Z0 => Z0,
        :status => status,
        :variant => variant,
        :iters => iters,
        :wall_time => wall_time,
        :eps_hat => eps_hat,
        :eps_tilde => eps_tilde,
    )
end


# ============================================================
# 2. 네트워크 재생성
# ============================================================

const OOS_NET_CONFIGS = Dict(
    :grid_5x5    => Dict(:type => :grid, :m => 5, :n => 5),
    :sioux_falls => Dict(:type => :real_world, :generator => generate_sioux_falls_network),
    :nobel_us    => Dict(:type => :real_world, :generator => generate_nobel_us_network),
    :abilene     => Dict(:type => :real_world, :generator => generate_abilene_network),
    :polska      => Dict(:type => :real_world, :generator => generate_polska_network),
)

function oos_compute_interdict_budget(config_key::Symbol, num_interdictable::Int, γ_ratio::Float64)
    if config_key in (:sioux_falls, :abilene)
        return 2
    end
    return ceil(Int, γ_ratio * num_interdictable)
end

function oos_regenerate_network(config_key::Symbol;
        S::Int=20, γ_ratio::Float64=0.10, seed::Int=42)
    config = OOS_NET_CONFIGS[config_key]
    network = config[:type] == :grid ?
        generate_grid_network(config[:m], config[:n]; seed=seed) :
        config[:generator]()

    num_arcs = length(network.arcs) - 1
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = oos_compute_interdict_budget(config_key, num_interdictable, γ_ratio)

    capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
        interdictable_arcs=network.interdictable_arcs, seed=seed)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_max = maximum(capacities[interdictable_idx, :])
    w = round(c_max; digits=4)   # wmax scheme

    return network, capacities, w, γ
end


# ============================================================
# 3. Log file finder
# ============================================================

function find_log_file(base_dir::String, net_key::Symbol, suffix::String)
    !isdir(base_dir) && return nothing
    pattern_prefix = "log_$(net_key)_"
    files = filter(readdir(base_dir)) do f
        startswith(f, pattern_prefix) && endswith(f, "$(suffix).txt")
    end
    isempty(files) && return nothing
    sort!(files)
    return joinpath(base_dir, files[end])
end


# ============================================================
# 4. x* 파싱: 3 모델 (nominal, single, two-layer)
# ============================================================

"""
    parse_all_x_stars(net_key, β; S, log_suffix) -> Dict{Symbol, Tuple{Vector, Float64, Symbol}}

3 모델의 x*, Z0, source를 로그에서 파싱.
Returns: model_key => (x_star, Z0, x_source)
"""
function parse_all_x_stars(net_key::Symbol, β::Float64;
        S::Int=20, log_suffix::String="wmax")
    β_str = replace(@sprintf("%.1f", β), "." => "p")
    nom_dir = joinpath(@__DIR__, "S$(S)_nominal_$(log_suffix)")
    β_dir   = joinpath(@__DIR__, "S$(S)_beta$(β_str)_$(log_suffix)")

    result = Dict{Symbol, Tuple{Vector{Float64}, Float64, Symbol}}()

    # Nominal
    nom_log = find_log_file(nom_dir, net_key, "nominal_$(log_suffix)")
    if nom_log !== nothing
        p = parse_benders_log(nom_log)
        if p[:x_star] !== nothing
            result[:nominal] = (Float64.(p[:x_star]), p[:Z0], p[:x_source])
        end
    end

    # Single-layer
    for sfx in ["single_compact_$(log_suffix)", "single_$(log_suffix)"]
        sl_log = find_log_file(β_dir, net_key, sfx)
        if sl_log !== nothing
            p = parse_benders_log(sl_log)
            if p[:x_star] !== nothing
                result[:single] = (Float64.(p[:x_star]), p[:Z0], p[:x_source])
                break
            end
        end
    end

    # Two-layer (non-single, non-nominal _wmax files)
    if isdir(β_dir)
        prefix = "log_$(net_key)_"
        td_files = filter(readdir(β_dir)) do f
            startswith(f, prefix) &&
            endswith(f, "_$(log_suffix).txt") &&
            !contains(f, "single") &&
            !contains(f, "nominal")
        end
        if !isempty(td_files)
            sort!(td_files)
            p = parse_benders_log(joinpath(β_dir, td_files[end]))
            if p[:x_star] !== nothing
                result[:two_layer] = (Float64.(p[:x_star]), p[:Z0], p[:x_source])
            end
        end
    end

    return result
end


# ============================================================
# 5. Phase A main
# ============================================================

function run_oos_phase_a(;
        beta_values = [0.1, 0.3, 0.5],
        net_keys    = [:grid_5x5, :polska, :abilene, :nobel_us, :sioux_falls],
        S::Int      = 20,
        M::Int      = 100,
        L::Int      = 100,
        seed::Int   = 42,
        log_suffix  = "wmax")

    println("=" ^ 80)
    println("Phase A OOS Evaluation (Symmetric Dirichlet, Tail Metrics)")
    println("  Models: Nominal / Single-layer / Two-layer DRO")
    println("  β values: $beta_values")
    println("  Networks: $net_keys")
    println("  S=$S, M=$M, L=$L, seed=$seed")
    println("=" ^ 80)

    all_results = Dict{Tuple{Symbol, Float64, Symbol}, Dict}()
    summary_rows = []

    for net_key in net_keys
        network, capacities, w, γ = oos_regenerate_network(net_key; S=S)
        num_arcs = length(network.arcs) - 1

        for β in beta_values
            ε_hat = round(lookup_epsilon(S, β; coverage=0.95); digits=2)
            @printf("\n[%s] β=%.1f, ε̂=%.2f\n", net_key, β, ε_hat)

            # Parse x* for all 3 models
            x_stars = parse_all_x_stars(net_key, β; S=S, log_suffix=log_suffix)
            for (mkey, (x, z, src)) in x_stars
                @printf("  [%-10s] Z₀=%.4f, Σx=%d, source=%s\n", mkey, z, sum(x), src)
            end

            if !haskey(x_stars, :nominal)
                @warn "Nominal x* not found, skipping" net_key
                continue
            end

            # x* 동일 쌍 감지
            x_vectors = Dict(k => v[1] for (k, v) in x_stars)
            same_x_as = Dict{Symbol, Symbol}()
            eval_order = filter(k -> haskey(x_vectors, k), [:nominal, :single, :two_layer])
            for i in 2:length(eval_order)
                for j in 1:(i-1)
                    if x_vectors[eval_order[i]] == x_vectors[eval_order[j]]
                        same_x_as[eval_order[i]] = eval_order[j]
                        @printf("  ⚠ %s x* == %s x* (same_x, OOS 공유)\n", eval_order[i], eval_order[j])
                        break
                    end
                end
            end

            # OOS evaluate (x* 동일하면 재사용)
            model_oos = Dict{Symbol, Dict}()
            for mkey in eval_order
                if haskey(same_x_as, mkey)
                    model_oos[mkey] = model_oos[same_x_as[mkey]]
                    @printf("  [%-10s] OOS reused from %s (same x*)\n", mkey, same_x_as[mkey])
                else
                    @printf("  [%-10s] Running Phase A OOS (M=%d, L=%d)...\n", mkey, M, L)
                    oos_r = oos_evaluate(x_vectors[mkey], network, capacities, β, 1.0, w;
                                          M=M, L=L, seed=seed)
                    model_oos[mkey] = oos_r
                end
                _, Z0, _ = x_stars[mkey]
                all_results[(net_key, β, mkey)] = merge(model_oos[mkey],
                    Dict(:Z0 => Z0, :x => x_vectors[mkey]))
            end

            # Win rate (x* 동일하면 NaN)
            pairs = [(:two_layer, :nominal), (:two_layer, :single), (:single, :nominal)]
            for (a, b) in pairs
                haskey(model_oos, a) && haskey(model_oos, b) || continue
                if x_vectors[a] == x_vectors[b]
                    @printf("  Win rate (%-10s vs %-8s): same_x\n", a, b)
                else
                    wr = compute_win_rate(model_oos[a][:Y_bar], model_oos[b][:Y_bar])
                    @printf("  Win rate (%-10s vs %-8s): %.1f%%\n", a, b, wr * 100)
                end
            end

            # Summary rows
            for mkey in eval_order
                _, Z0, _ = x_stars[mkey]
                oos_r = model_oos[mkey]
                is_same_x_nom = mkey != :nominal && x_vectors[mkey] == x_vectors[:nominal]
                wr_vs_nom = if mkey == :nominal || is_same_x_nom
                    NaN
                else
                    compute_win_rate(oos_r[:Y_bar], model_oos[:nominal][:Y_bar])
                end

                push!(summary_rows, (
                    network    = string(net_key),
                    beta       = β,
                    variant    = string(mkey),
                    eps_hat    = mkey == :nominal ? 0.0 : ε_hat,
                    Z0         = Z0,
                    oos_mean   = oos_r[:mean],
                    ci_lo      = oos_r[:ci_lo],
                    ci_hi      = oos_r[:ci_hi],
                    oos_p5     = oos_r[:p5],
                    p5_ci_lo   = oos_r[:p5_ci_lo],
                    p5_ci_hi   = oos_r[:p5_ci_hi],
                    oos_p95    = oos_r[:p95],
                    p95_ci_lo  = oos_r[:p95_ci_lo],
                    p95_ci_hi  = oos_r[:p95_ci_hi],
                    oos_min    = oos_r[:min],
                    oos_max    = oos_r[:max],
                    f_share    = oos_r[:follower_share],
                    win_vs_nom = wr_vs_nom,
                    same_x_nom = is_same_x_nom,
                ))
            end
        end
    end

    # ---- Save ----
    jls_path = joinpath(@__DIR__, "oos_phase_a_results.jls")
    serialize(jls_path, all_results)
    println("\nRaw results saved → $jls_path")

    csv_path = joinpath(@__DIR__, "oos_phase_a_summary.csv")
    open(csv_path, "w") do io
        println(io, "network,beta,variant,eps_hat,Z0,oos_mean,ci_lo,ci_hi,oos_p5,p5_ci_lo,p5_ci_hi,oos_p95,p95_ci_lo,p95_ci_hi,oos_min,oos_max,f_share,win_vs_nom,same_x_nom")
        for r in summary_rows
            win_str = isnan(r.win_vs_nom) ? "" : @sprintf("%.4f", r.win_vs_nom)
            @printf(io, "%s,%.1f,%s,%.2f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%s,%s\n",
                    r.network, r.beta, r.variant, r.eps_hat, r.Z0,
                    r.oos_mean, r.ci_lo, r.ci_hi,
                    r.oos_p5, r.p5_ci_lo, r.p5_ci_hi,
                    r.oos_p95, r.p95_ci_lo, r.p95_ci_hi,
                    r.oos_min, r.oos_max,
                    r.f_share, win_str, r.same_x_nom)
        end
    end
    println("Summary CSV saved → $csv_path")

    # ---- Print table ----
    println("\n" * "=" ^ 130)
    println("Phase A OOS Summary (Symmetric Dirichlet, Tail Metrics)")
    println("=" ^ 130)
    @printf("%-12s %4s  %-10s  %5s  %9s  %21s  %21s  %21s  %7s  %7s\n",
            "Network", "β", "Variant", "ε̂", "Z₀", "Mean [95% CI]", "p5 [95% CI]", "p95 [95% CI]", "f_share", "Win%")
    println("-" ^ 160)

    for r in summary_rows
        win_str = if r.variant == "nominal"
            "   ---"
        elseif r.same_x_nom
            "same_x"
        elseif isnan(r.win_vs_nom)
            "   ---"
        else
            @sprintf("%5.1f%%", r.win_vs_nom * 100)
        end
        ci_mean = @sprintf("%.4f [%.4f,%.4f]", r.oos_mean, r.ci_lo, r.ci_hi)
        ci_p5   = @sprintf("%.4f [%.4f,%.4f]", r.oos_p5, r.p5_ci_lo, r.p5_ci_hi)
        ci_p95  = @sprintf("%.4f [%.4f,%.4f]", r.oos_p95, r.p95_ci_lo, r.p95_ci_hi)
        @printf("%-12s %4.1f  %-10s  %5.2f  %9.4f  %21s  %21s  %21s  %7.4f  %s\n",
                r.network, r.beta, r.variant, r.eps_hat, r.Z0,
                ci_mean, ci_p5, ci_p95,
                r.f_share, win_str)
    end
    println("=" ^ 160)

    return all_results, summary_rows
end
