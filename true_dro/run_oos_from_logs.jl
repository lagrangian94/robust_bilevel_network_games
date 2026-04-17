"""
run_oos_from_logs.jl — 3-variant OOS 평가:
  1. Nominal (build_full_2SP_model): robustness 없는 2-stage SP
  2. Single-layer (ε̂=calibrated, ε̃=0): leader만 robust
  3. True-DRO (ε̂=ε̃=calibrated): 로그에서 x* 파싱

Usage (Julia REPL):
    include("run_oos_from_logs.jl")
"""

using Printf
using Statistics
using Serialization
using Dates
using DelimitedFiles
using JuMP
using Gurobi
using HiGHS
using LinearAlgebra

# ---- Load modules ----
if !@isdefined(NetworkGenerator)
    include("../network_generator.jl")
end
using .NetworkGenerator

include("oos_dirichlet.jl")

# OOS evaluation
include("../oos_evaluation.jl")
include("oos_evaluate.jl")

# True-DRO Benders machinery (for single-layer solves)
include("true_dro_data.jl")
include("true_dro_build_omp.jl")
include("true_dro_build_subproblem.jl")
include("true_dro_build_isp_leader.jl")
include("true_dro_build_isp_follower.jl")
include("true_dro_benders.jl")
include("true_dro_mincut_vi.jl")

# Nominal SP: build_full_2SP_model (from ../build_nominal_sp.jl)
# 파일 top-level에 Pajarito/Mosek/Hypatia 등 불필요 import → 함수 정의만 추출
if !@isdefined(build_full_2SP_model)
    _nominal_sp_src = read(joinpath(@__DIR__, "..", "build_nominal_sp.jl"), String)
    _func_start = findfirst("function build_full_2SP_model", _nominal_sp_src)
    _func_str = _nominal_sp_src[first(_func_start):end]
    include_string(Main, _func_str)
    @info "build_full_2SP_model loaded from build_nominal_sp.jl"
end


# ============================================================
# 1. Log 파싱
# ============================================================

function parse_log(filepath::String)
    lines = readlines(filepath)

    x_star = nothing
    Z0 = NaN
    status = "Unknown"
    iters = 0
    wall_time = NaN
    eps_hat = NaN
    eps_tilde = NaN

    for (i, line) in enumerate(lines)
        m_eps = match(r"ε̂=([\d.]+),\s*ε̃=([\d.]+)", line)
        if m_eps !== nothing
            eps_hat = parse(Float64, m_eps.captures[1])
            eps_tilde = parse(Float64, m_eps.captures[2])
        end

        m_status = match(r"True-DRO:\s*status=(\w+),\s*Z₀=([\d.]+),\s*iters=(\d+),\s*time=([\d.]+)s", line)
        if m_status !== nothing
            status = m_status.captures[1]
            Z0 = parse(Float64, m_status.captures[2])
            iters = parse(Int, m_status.captures[3])
            wall_time = parse(Float64, m_status.captures[4])
        end

        if x_star === nothing
            m_x = match(r"^\s*x\*\s*=\s*\[([0-9,\s]+)\]", line)
            if m_x !== nothing && !isnan(Z0)
                x_star = parse.(Int, split(m_x.captures[1], r"[,\s]+"; keepempty=false))
            end
        end
    end

    return Dict(
        :x_star => x_star, :Z0 => Z0, :status => status,
        :iters => iters, :wall_time => wall_time,
        :eps_hat => eps_hat, :eps_tilde => eps_tilde,
    )
end


# ============================================================
# 2. 네트워크 재생성
# ============================================================

oos_network_configs = Dict(
    :grid_5x5    => Dict(:type => :grid, :m => 5, :n => 5),
    :sioux_falls => Dict(:type => :real_world, :generator => generate_sioux_falls_network),
    :nobel_us    => Dict(:type => :real_world, :generator => generate_nobel_us_network),
    :abilene     => Dict(:type => :real_world, :generator => generate_abilene_network),
    :polska      => Dict(:type => :real_world, :generator => generate_polska_network),
)

function compute_interdict_budget_oos(config_key::Symbol, num_interdictable::Int, γ_ratio::Float64)
    if config_key in (:sioux_falls, :abilene)
        return 2
    end
    return ceil(Int, γ_ratio * num_interdictable)
end

function regenerate_network_data(config_key::Symbol;
        S::Int=20, γ_ratio::Float64=0.10, ρ::Float64=0.2, v::Float64=1.0, seed::Int=42)

    config = oos_network_configs[config_key]
    if config[:type] == :grid
        network = generate_grid_network(config[:m], config[:n]; seed=seed)
    else
        network = config[:generator]()
    end

    num_arcs = length(network.arcs) - 1
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = compute_interdict_budget_oos(config_key, num_interdictable, γ_ratio)

    capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
        interdictable_arcs=network.interdictable_arcs, seed=seed)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(ρ * γ * c_bar; digits=4)

    return network, capacities, w, γ
end


# ============================================================
# 3. Nominal SP solver
# ============================================================

"""
    solve_nominal_sp(network, capacities, w, γ; S, λU) -> (x_star, Z0)

build_full_2SP_model로 nominal 2-stage SP (no DRO) 풀기.
"""
function solve_nominal_sp(network, capacities, w, γ;
                           S::Int=20, λU::Float64=2.0)
    num_arcs = length(network.arcs) - 1

    # build_full_2SP_model이 요구하는 uncertainty_set 구성
    # 실제 constraints에서는 xi_bar만 사용됨 (R, r_dict, epsilon은 추출만 하고 미사용)
    xi_bar_vecs = [capacities[1:num_arcs, s] for s in 1:S]
    uncertainty_set = Dict(
        :xi_bar => xi_bar_vecs,
        :R => zeros(0, 0),
        :r_dict_hat => Dict(),
        :epsilon_hat => 0.0,
    )

    ϕU = λU  # big-M for LDR coefficients
    v_param = 1.0  # interdiction effectiveness (scalar)

    model, vars = build_full_2SP_model(network, S, ϕU, λU, γ, w, v_param, uncertainty_set)
    set_optimizer_attribute(model, "OutputFlag", 0)
    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        @warn "Nominal SP status: $(termination_status(model))"
        return zeros(num_arcs), NaN
    end

    x_star = Float64.(round.(Int, value.(vars[:x])))
    Z0 = objective_value(model)
    return x_star, Z0
end


# ============================================================
# 4. Single-layer solver (ε̂=calibrated, ε̃=0)
# ============================================================

function solve_single_layer(network, capacities, w, γ, ε_hat;
                             S::Int=20, λU::Float64=2.0, tol::Float64=1e-4,
                             max_iter::Int=1000, sub_time_limit=30.0)
    q_hat = fill(1.0 / S, S)
    td = make_true_dro_data(network, capacities, q_hat, ε_hat, 0.0;
                            w=w, lambda_U=λU, gamma=γ)

    result = true_dro_benders_optimize!(td;
        mip_optimizer=Gurobi.Optimizer,
        nlp_optimizer=Gurobi.Optimizer,
        lp_optimizer=Gurobi.Optimizer,
        inexact=false,
        max_iter=max_iter,
        tol=tol,
        verbose=true,
        sub_verbose=false,
        sub_time_limit=sub_time_limit,
        mini_benders=true,
        max_mini_benders_iter=5,
        strengthen_cuts=:mw,
        valid_inequality=:mincut)

    x_star = Float64.(round.(Int, result[:x]))
    Z0 = result[:Z0]
    iters = result[:iters]
    wt = get(result, :wall_time, NaN)
    return x_star, Z0, iters, wt
end


# ============================================================
# 5. Helpers
# ============================================================

function find_log_file(log_dir::String, net_key::Symbol)
    !isdir(log_dir) && return nothing
    pattern = "log_$(net_key)_"
    files = filter(f -> startswith(f, pattern) && endswith(f, ".txt"), readdir(log_dir))
    isempty(files) && return nothing
    sort!(files)
    return joinpath(log_dir, files[end])
end

function run_oos_and_summarize(x_star, network, capacities, β, w, M, L, seed)
    oos_result = oos_evaluate(x_star, network, capacities, β, 1.0, w;
                               M=M, L=L, seed=seed)
    Y_bar_outer = oos_result[:Y_bar]
    oos_geomean = exp(mean(log.(Y_bar_outer)))
    return oos_result, oos_geomean
end


# ============================================================
# 6. Main loop
# ============================================================

oos_beta_values = [0.1, 0.3, 0.5, 0.8]
oos_net_keys = [:grid_5x5, :polska, :abilene, :nobel_us, :sioux_falls]
oos_S = 20
oos_M = 100
oos_L = 1000
oos_seed = 42

println("=" ^ 70)
println("3-Variant OOS Evaluation")
println("  Nominal:      build_full_2SP_model (no DRO)")
println("  Single-layer: ε̂=calibrated, ε̃=0")
println("  True-DRO:     ε̂=ε̃=calibrated (from logs)")
println("=" ^ 70)
println("  β values: $oos_beta_values")
println("  Networks: $oos_net_keys")
println("  S=$oos_S, M=$oos_M, L=$oos_L, seed=$oos_seed")
println()

raw_results = Dict{Tuple{Symbol, Float64}, Dict}()
summary_rows = []

# ---- Nominal은 β 무관 (DRO 없음) → 네트워크별 1회만 풀기 ----
nominal_cache = Dict{Symbol, Tuple{Vector{Float64}, Float64}}()

for β in oos_beta_values
    β_str = replace(@sprintf("%.1f", β), "." => "p")
    log_dir = joinpath(@__DIR__, "S$(oos_S)_beta$(β_str)_cov95")

    ε_raw = lookup_epsilon(oos_S, β; coverage=0.95)
    ε_hat = round(ε_raw; digits=2)
    @printf("\n# β=%.1f → ε̂=%.2f\n", β, ε_hat)
    println("#" ^ 60)

    for net_key in oos_net_keys
        println("\n" * "-" ^ 50)
        @printf("  Network: %s (β=%.1f)\n", net_key, β)
        println("-" ^ 50)

        # Regenerate network data
        network, capacities, w, γ = regenerate_network_data(net_key; S=oos_S)
        num_arcs = length(network.arcs) - 1
        @printf("  |A|=%d, γ=%d, w=%.4f\n", num_arcs, γ, w)

        entry = Dict{Symbol, Any}()

        # ---- (A) True-DRO: parse from log ----
        log_path = find_log_file(log_dir, net_key)
        if log_path === nothing
            @warn "Log file not found" net_key log_dir
            continue
        end
        parsed = parse_log(log_path)
        if parsed[:x_star] === nothing
            @warn "Failed to parse x* from log" log_path
            continue
        end
        x_td = Float64.(parsed[:x_star])
        Z0_td = parsed[:Z0]
        if length(x_td) != num_arcs
            @error "x* length mismatch" length(x_td) num_arcs
            continue
        end
        @printf("  [True-DRO] Z₀=%.6f, x*=%s\n", Z0_td, string(round.(Int, x_td)))

        println("  [True-DRO] OOS...")
        oos_td, geomean_td = run_oos_and_summarize(x_td, network, capacities, β, w, oos_M, oos_L, oos_seed)
        @printf("  [True-DRO] OOS mean=%.4f, p95=%.4f\n", oos_td[:mean], oos_td[:p95])
        entry[:true_dro] = Dict(:x => x_td, :Z0 => Z0_td, :oos => oos_td, :geomean => geomean_td)

        # ---- (B) Nominal: build_full_2SP_model (β 무관, 캐시) ----
        if !haskey(nominal_cache, net_key)
            println("  [Nominal] Solving 2SP model...")
            t0 = time()
            x_nom, Z0_nom = solve_nominal_sp(network, capacities, w, γ; S=oos_S)
            elapsed_nom = time() - t0
            @printf("  [Nominal] Z₀=%.6f, time=%.1fs, x*=%s\n",
                    Z0_nom, elapsed_nom, string(round.(Int, x_nom)))
            nominal_cache[net_key] = (x_nom, Z0_nom)
        else
            x_nom, Z0_nom = nominal_cache[net_key]
            @printf("  [Nominal] (cached) Z₀=%.6f, x*=%s\n", Z0_nom, string(round.(Int, x_nom)))
        end

        println("  [Nominal] OOS...")
        oos_nom, geomean_nom = run_oos_and_summarize(x_nom, network, capacities, β, w, oos_M, oos_L, oos_seed)
        @printf("  [Nominal] OOS mean=%.4f, p95=%.4f\n", oos_nom[:mean], oos_nom[:p95])
        entry[:nominal] = Dict(:x => x_nom, :Z0 => Z0_nom, :oos => oos_nom, :geomean => geomean_nom)

        # ---- (C) Single-layer: ε̂=calibrated, ε̃=0 ----
        println("  [Single] Solving Benders (ε̂=$ε_hat, ε̃=0)...")
        try
            t0 = time()
            x_sl, Z0_sl, iters_sl, wt_sl = solve_single_layer(
                network, capacities, w, γ, ε_hat; S=oos_S)
            @printf("  [Single] Z₀=%.6f, iters=%d, time=%.1fs, x*=%s\n",
                    Z0_sl, iters_sl, wt_sl, string(round.(Int, x_sl)))

            println("  [Single] OOS...")
            oos_sl, geomean_sl = run_oos_and_summarize(x_sl, network, capacities, β, w, oos_M, oos_L, oos_seed)
            @printf("  [Single] OOS mean=%.4f, p95=%.4f\n", oos_sl[:mean], oos_sl[:p95])
            entry[:single_layer] = Dict(:x => x_sl, :Z0 => Z0_sl, :oos => oos_sl, :geomean => geomean_sl)
        catch e
            @warn "Single-layer failed" net_key β exception=e
            entry[:single_layer] = nothing
        end

        raw_results[(net_key, β)] = entry

        # Summary rows
        _variants = [
            ("true_dro",     entry[:true_dro],     ε_hat,  ε_hat),
            ("nominal",      entry[:nominal],      0.0,    0.0),
        ]
        if get(entry, :single_layer, nothing) !== nothing
            push!(_variants, ("single_layer", entry[:single_layer], ε_hat, 0.0))
        end
        for (variant, vdata, eh, et) in _variants
            oos_r = vdata[:oos]
            Y_bar = oos_r[:Y_bar]
            push!(summary_rows, (
                beta         = β,
                network      = string(net_key),
                variant      = variant,
                S            = oos_S,
                eps_hat      = eh,
                eps_tilde    = et,
                Z0_insample  = vdata[:Z0],
                oos_mean     = oos_r[:mean],
                oos_geomean  = vdata[:geomean],
                oos_std      = std(Y_bar),
                oos_p95      = oos_r[:p95],
                var_outer    = oos_r[:var_outer],
                var_inner    = oos_r[:var_inner],
                follower_share = oos_r[:follower_share],
            ))
        end
    end
end


# ============================================================
# 7. Save results
# ============================================================

jls_path = joinpath(@__DIR__, "oos_3variant_results.jls")
serialize(jls_path, raw_results)
println("\nRaw results saved → $jls_path ($(length(raw_results)) entries)")

csv_path = joinpath(@__DIR__, "oos_3variant_summary.csv")
open(csv_path, "w") do io
    println(io, "beta,network,variant,S,eps_hat,eps_tilde,Z0_insample,oos_mean,oos_geomean,oos_std,oos_p95,var_outer,var_inner,follower_share")
    for r in summary_rows
        @printf(io, "%.1f,%s,%s,%d,%.2f,%.2f,%.6f,%.6f,%.6f,%.6f,%.6f,%.2e,%.2e,%.4f\n",
                r.beta, r.network, r.variant, r.S, r.eps_hat, r.eps_tilde,
                r.Z0_insample, r.oos_mean, r.oos_geomean, r.oos_std, r.oos_p95,
                r.var_outer, r.var_inner, r.follower_share)
    end
end
println("Summary CSV saved → $csv_path")

# Print summary table
println("\n" * "=" ^ 115)
println("3-Variant OOS Summary")
println("=" ^ 115)
@printf("%-5s %-12s  %-13s  %5s %5s  %10s  %10s  %10s  %8s\n",
        "β", "Network", "Variant", "ε̂", "ε̃", "Z₀", "OOS_mean", "OOS_p95", "f_share")
println("-" ^ 115)
for r in summary_rows
    @printf("%-5.1f %-12s  %-13s  %5.2f %5.2f  %10.4f  %10.4f  %10.4f  %8.4f\n",
            r.beta, r.network, r.variant, r.eps_hat, r.eps_tilde,
            r.Z0_insample, r.oos_mean, r.oos_p95, r.follower_share)
end
println("=" ^ 115)
println("\nDone! $(now())")
