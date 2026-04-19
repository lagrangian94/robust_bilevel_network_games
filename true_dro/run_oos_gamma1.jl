"""
run_oos_gamma1.jl — γ=1 실험: polska, sioux_falls.

1. True-DRO x* → γ=1 로그에서 파싱
2. Nominal SP → build_full_2SP_model (γ=1)
3. Single-layer → True-DRO Benders (ε̂=cal, ε̃=0, γ=1)
4. Paired OOS evaluation (Bayraksan-Love protocol)

Usage: include("run_oos_gamma1.jl")
"""

using Printf
using Statistics
using Serialization
using Dates
using LinearAlgebra
using Random
using Distributions
using JuMP
using Gurobi
using HiGHS

# ---- Load modules ----
if !@isdefined(NetworkGenerator)
    include("../network_generator.jl")
end
using .NetworkGenerator

include("oos_dirichlet.jl")
include("../oos_evaluation.jl")
include("oos_evaluate.jl")

# True-DRO Benders (for single-layer)
include("true_dro_data.jl")
include("true_dro_build_omp.jl")
include("true_dro_build_subproblem.jl")
include("true_dro_build_isp_leader.jl")
include("true_dro_build_isp_follower.jl")
include("true_dro_benders.jl")
include("true_dro_mincut_vi.jl")

# Nominal SP
if !@isdefined(build_full_2SP_model)
    _nominal_sp_src = read(joinpath(@__DIR__, "..", "build_nominal_sp.jl"), String)
    _func_start = findfirst("function build_full_2SP_model", _nominal_sp_src)
    _func_str = _nominal_sp_src[first(_func_start):end]
    include_string(Main, _func_str)
    @info "build_full_2SP_model loaded"
end


# ============================================================
# Config
# ============================================================
const G1_S = 20
const G1_GAMMA = 1
const G1_RHO = 0.2
const G1_SEED = 42
const G1_R = 1000  # paired OOS reps

g1_beta_values = [0.1, 0.3, 0.5, 0.8]
g1_net_keys = [:polska, :sioux_falls]

g1_network_configs = Dict(
    :sioux_falls => Dict(:type => :real_world, :generator => generate_sioux_falls_network),
    :polska      => Dict(:type => :real_world, :generator => generate_polska_network),
)


# ============================================================
# Network regeneration (γ=1 고정)
# ============================================================
function regenerate_gamma1(config_key::Symbol; S::Int=G1_S)
    config = g1_network_configs[config_key]
    network = config[:generator]()
    num_arcs = length(network.arcs) - 1

    capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
        interdictable_arcs=network.interdictable_arcs, seed=G1_SEED)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(G1_RHO * G1_GAMMA * c_bar; digits=4)

    return network, capacities, w
end


# ============================================================
# Log parsing (gamma1 logs)
# ============================================================
function parse_log(filepath::String)
    lines = readlines(filepath)
    x_star = nothing; Z0 = NaN; status = "Unknown"
    iters = 0; wall_time = NaN; eps_hat = NaN; eps_tilde = NaN

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
    return Dict(:x_star => x_star, :Z0 => Z0, :status => status,
                :iters => iters, :wall_time => wall_time)
end

function find_gamma1_log(log_dir::String, net_key::Symbol)
    !isdir(log_dir) && return nothing
    pattern = "log_$(net_key)_"
    files = filter(f -> startswith(f, pattern) && endswith(f, ".txt") && occursin("gamma1", f),
                   readdir(log_dir))
    isempty(files) && return nothing
    sort!(files)
    return joinpath(log_dir, files[end])
end


# ============================================================
# Nominal SP (γ=1)
# ============================================================
function solve_nominal_gamma1(network, capacities, w; S::Int=G1_S, λU::Float64=2.0)
    num_arcs = length(network.arcs) - 1
    xi_bar_vecs = [capacities[1:num_arcs, s] for s in 1:S]
    uncertainty_set = Dict(:xi_bar => xi_bar_vecs, :R => zeros(0, 0),
                           :r_dict_hat => Dict(), :epsilon_hat => 0.0)
    model, vars = build_full_2SP_model(network, S, λU, λU, G1_GAMMA, w, 1.0, uncertainty_set)
    set_optimizer_attribute(model, "OutputFlag", 0)
    optimize!(model)
    if termination_status(model) != MOI.OPTIMAL
        @warn "Nominal SP status: $(termination_status(model))"
        return zeros(num_arcs), NaN
    end
    x_star = Float64.(round.(Int, value.(vars[:x])))
    return x_star, objective_value(model)
end


# ============================================================
# Single-layer (ε̂=cal, ε̃=0, γ=1)
# ============================================================
function solve_single_gamma1(network, capacities, w, ε_hat;
                              S::Int=G1_S, λU::Float64=2.0)
    q_hat = fill(1.0 / S, S)
    td = make_true_dro_data(network, capacities, q_hat, ε_hat, 0.0;
                            w=w, lambda_U=λU, gamma=G1_GAMMA)
    result = true_dro_benders_optimize!(td;
        mip_optimizer=Gurobi.Optimizer,
        nlp_optimizer=Gurobi.Optimizer,
        lp_optimizer=Gurobi.Optimizer,
        inexact=false, max_iter=1000, tol=1e-4,
        verbose=true, sub_verbose=false, sub_time_limit=30.0,
        mini_benders=true, max_mini_benders_iter=5,
        strengthen_cuts=:mw, valid_inequality=:mincut)
    x_star = Float64.(round.(Int, result[:x]))
    return x_star, result[:Z0], result[:iters], get(result, :wall_time, NaN)
end


# ============================================================
# Paired OOS (from run_oos_paired.jl)
# ============================================================
function oos_paired(x_stars::Dict{Symbol, Vector{Float64}},
                     network, capacities::Matrix{Float64},
                     β::Float64, v::Float64, w::Float64;
                     R::Int=G1_R, seed::Int=G1_SEED)
    K = size(capacities, 2)
    rng = MersenneTwister(seed)
    dir_dist = Dirichlet(K, β)
    model_names = collect(keys(x_stars))

    true_costs = Dict(m => Vector{Float64}(undef, R) for m in model_names)
    winners = Vector{Symbol}(undef, R)

    for rep in 1:R
        q_true = rand(rng, dir_dist)
        q_tilde = rand(rng, dir_dist)
        best_cost = Inf; best_model = model_names[1]

        for m in model_names
            h_star = solve_follower_weighted(network, x_stars[m], v, w, capacities, q_tilde)
            flows = compute_maxflow_per_scenario(network, x_stars[m], h_star, v, capacities)
            cost = dot(q_true, flows)
            true_costs[m][rep] = cost
            if cost < best_cost; best_cost = cost; best_model = m; end
        end
        winners[rep] = best_model
        rep % 200 == 0 && @printf("  rep %d/%d\n", rep, R)
    end

    stats = Dict(m => Dict(
        :mean => mean(c), :std => std(c), :median => median(c),
        :p05 => quantile(c, 0.05), :p95 => quantile(c, 0.95),
    ) for (m, c) in true_costs)
    win_counts = Dict(m => count(==(m), winners) for m in model_names)

    return Dict(:true_costs => true_costs, :stats => stats,
                :win_counts => win_counts, :R => R)
end


# ============================================================
# Main loop
# ============================================================
println("=" ^ 80)
println("γ=1 OOS Evaluation (polska, sioux_falls)")
println("  Nominal + Single-layer + True-DRO → Paired OOS (R=$G1_R)")
println("=" ^ 80)

all_summary = []
raw_results = Dict{Tuple{Symbol, Float64}, Dict}()

# Nominal은 β 무관 → 캐시
nominal_cache = Dict{Symbol, Tuple{Vector{Float64}, Float64}}()

for β in g1_beta_values
    β_str = replace(@sprintf("%.1f", β), "." => "p")
    log_dir = joinpath(@__DIR__, "S$(G1_S)_beta$(β_str)_cov95")

    ε_raw = lookup_epsilon(G1_S, β; coverage=0.95)
    ε_hat = round(ε_raw; digits=2)
    @printf("\n### β=%.1f → ε̂=%.2f ###\n", β, ε_hat)

    for net_key in g1_net_keys
        println("\n" * "-" ^ 60)
        @printf("  %s (β=%.1f, γ=%d)\n", net_key, β, G1_GAMMA)
        println("-" ^ 60)

        network, capacities, w = regenerate_gamma1(net_key)
        num_arcs = length(network.arcs) - 1
        @printf("  |A|=%d, γ=%d, w=%.4f\n", num_arcs, G1_GAMMA, w)

        x_stars = Dict{Symbol, Vector{Float64}}()
        z0_map = Dict{Symbol, Float64}()

        # ---- (A) True-DRO: parse γ=1 log ----
        log_path = find_gamma1_log(log_dir, net_key)
        if log_path !== nothing
            parsed = parse_log(log_path)
            if parsed[:x_star] !== nothing && length(parsed[:x_star]) == num_arcs
                x_td = Float64.(parsed[:x_star])
                Z0_td = parsed[:Z0]
                x_stars[:true_dro] = x_td
                z0_map[:true_dro] = Z0_td
                @printf("  [True-DRO] Z₀=%.6f, x*=%s\n", Z0_td, string(round.(Int, x_td)))
            else
                @warn "Failed to parse True-DRO log" log_path
            end
        else
            @warn "γ=1 log not found" net_key log_dir
        end

        # ---- (B) Nominal SP (γ=1, 캐시) ----
        if !haskey(nominal_cache, net_key)
            println("  [Nominal] Solving...")
            t0 = time()
            x_nom, Z0_nom = solve_nominal_gamma1(network, capacities, w)
            @printf("  [Nominal] Z₀=%.6f, time=%.1fs, x*=%s\n",
                    Z0_nom, time()-t0, string(round.(Int, x_nom)))
            nominal_cache[net_key] = (x_nom, Z0_nom)
        else
            x_nom, Z0_nom = nominal_cache[net_key]
            @printf("  [Nominal] (cached) Z₀=%.6f\n", Z0_nom)
        end
        x_stars[:nominal] = x_nom
        z0_map[:nominal] = Z0_nom

        # ---- (C) Single-layer (γ=1, ε̃=0) ----
        println("  [Single] Solving (ε̂=$ε_hat, ε̃=0, γ=$G1_GAMMA)...")
        try
            x_sl, Z0_sl, iters_sl, wt_sl = solve_single_gamma1(network, capacities, w, ε_hat)
            @printf("  [Single] Z₀=%.6f, iters=%d, time=%.1fs, x*=%s\n",
                    Z0_sl, iters_sl, wt_sl, string(round.(Int, x_sl)))
            x_stars[:single_layer] = x_sl
            z0_map[:single_layer] = Z0_sl
        catch e
            @warn "Single-layer failed" net_key β exception=e
        end

        if length(x_stars) < 2
            @warn "Not enough models" net_key β
            continue
        end

        # ---- (D) Paired OOS ----
        println("  [OOS] Paired evaluation (R=$G1_R)...")
        oos = oos_paired(x_stars, network, capacities, β, 1.0, w)

        raw_results[(net_key, β)] = Dict(:x_stars => x_stars, :z0 => z0_map, :oos => oos)

        # Print
        @printf("  %-14s  %10s  %10s  %10s  %10s  %8s\n",
                "Model", "Z₀", "OOS_mean", "OOS_p05", "OOS_p95", "WinRate")
        @printf("  %s\n", "-" ^ 70)
        for m in [:nominal, :single_layer, :true_dro]
            !haskey(oos[:stats], m) && continue
            s = oos[:stats][m]
            wr = oos[:win_counts][m] / G1_R * 100
            z0 = get(z0_map, m, NaN)
            @printf("  %-14s  %10.4f  %10.4f  %10.4f  %10.4f  %7.1f%%\n",
                    m, z0, s[:mean], s[:p05], s[:p95], wr)

            push!(all_summary, (
                beta=β, network=string(net_key), variant=string(m), gamma=G1_GAMMA,
                Z0_insample=z0, oos_mean=s[:mean], oos_std=s[:std],
                oos_p05=s[:p05], oos_p95=s[:p95],
                win_rate=oos[:win_counts][m]/G1_R,
            ))
        end
    end
end


# ============================================================
# Save
# ============================================================
jls_path = joinpath(@__DIR__, "oos_gamma1_results.jls")
serialize(jls_path, raw_results)
println("\nRaw results saved → $jls_path")

csv_path = joinpath(@__DIR__, "oos_gamma1_summary.csv")
open(csv_path, "w") do io
    println(io, "beta,network,variant,gamma,Z0_insample,oos_mean,oos_std,oos_p05,oos_p95,win_rate")
    for r in all_summary
        @printf(io, "%.1f,%s,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f\n",
                r.beta, r.network, r.variant, r.gamma, r.Z0_insample,
                r.oos_mean, r.oos_std, r.oos_p05, r.oos_p95, r.win_rate)
    end
end
println("CSV saved → $csv_path")

# Final table
println("\n" * "=" ^ 110)
println("γ=1 Paired OOS Summary (R=$G1_R)")
println("=" ^ 110)
@printf("%-5s %-12s  %-14s  %10s  %10s  %10s  %10s  %8s\n",
        "β", "Network", "Variant", "Z₀", "OOS_mean", "OOS_p05", "OOS_p95", "WinRate")
println("-" ^ 110)
for r in all_summary
    @printf("%-5.1f %-12s  %-14s  %10.4f  %10.4f  %10.4f  %10.4f  %7.1f%%\n",
            r.beta, r.network, r.variant, r.Z0_insample,
            r.oos_mean, r.oos_p05, r.oos_p95, r.win_rate * 100)
end
println("=" ^ 110)
println("\nDone! $(now())")
