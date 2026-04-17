"""
test_benders_batch.jl — test_benders.jl을 모든 네트워크에 대해 순차 실행.

고정 설정:
  S=20, ε̂=ε̃=0.7, sub_tl=30, MW cuts, mincut VI, inexact, mini-benders
네트워크: grid_3x3, abilene, nobel_us, polska, sioux_falls, grid_5x5
각 실행마다 별도 로그 파일, profiles.jls에 누적 저장.
"""

using Revise
using JuMP
using Gurobi
using HiGHS
using Printf
using LinearAlgebra
using Serialization
using Dates
using Logging

if !@isdefined(NetworkGenerator)
    include("../network_generator.jl")
end
using .NetworkGenerator

includet("true_dro_data.jl")
includet("true_dro_build_omp.jl")
includet("true_dro_build_subproblem.jl")
includet("true_dro_build_isp_leader.jl")
includet("true_dro_build_isp_follower.jl")
includet("true_dro_benders.jl")
includet("true_dro_mincut_vi.jl")
includet("true_dro_recover.jl")

# ===== Network configs (test_benders.jl과 동일) =====
network_configs = Dict(
    :grid_3x3    => Dict(:type => :grid, :m => 3, :n => 3),
    :grid_5x5    => Dict(:type => :grid, :m => 5, :n => 5),
    :sioux_falls => Dict(:type => :real_world, :generator => generate_sioux_falls_network),
    :nobel_us    => Dict(:type => :real_world, :generator => generate_nobel_us_network),
    :abilene     => Dict(:type => :real_world, :generator => generate_abilene_network),
    :polska      => Dict(:type => :real_world, :generator => generate_polska_network),
)

function compute_interdict_budget(config_key::Symbol, num_interdictable::Int, γ_ratio::Float64)
    if config_key in (:sioux_falls, :abilene)
        return 2
    end
    return ceil(Int, γ_ratio * num_interdictable)
end

function setup_true_dro_instance(config_key::Symbol;
    S=10, γ_ratio=0.10, ρ=0.2, v=1.0, seed=42,
    epsilon_hat=0.5, epsilon_tilde=epsilon_hat)
    config = network_configs[config_key]
    if config[:type] == :grid
        network = generate_grid_network(config[:m], config[:n]; seed=seed)
        print_network_summary(network)
    else
        network = config[:generator]()
        print_realworld_network_summary(network)
    end
    num_arcs = length(network.arcs) - 1
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = compute_interdict_budget(config_key, num_interdictable, γ_ratio)
    capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
        interdictable_arcs=network.interdictable_arcs, seed=seed)
    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(ρ * γ * c_bar; digits=4)
    λU = 2.0
    q_hat = fill(1.0 / S, S)
    println("  |A|=$num_arcs, S=$S, γ=$γ, w=$(round(w, digits=4)), λU=$λU")
    println("  ε̂=$epsilon_hat, ε̃=$epsilon_tilde")
    td = make_true_dro_data(network, capacities, q_hat, epsilon_hat, epsilon_tilde;
                            w=w, lambda_U=λU, gamma=γ)
    return network, td
end

# ===== 배치 설정 =====
const S_VAL          = 20
const EPSILON        = 0.7
const SUB_TL         = 30.0
const USE_MINI       = true
const MAX_MB_ITER    = 5
const STRENGTHEN     = :mw
const USE_INEXACT    = true
const VI_SYM         = :mincut

# 실행 순서: 작고 쉬운 것 → 크고 어려운 것
const NETWORKS = [:grid_3x3, :abilene, :nobel_us, :polska, :sioux_falls, :grid_5x5]

# ===== 각 네트워크 실행 =====
function run_one(instance_key::Symbol)
    log_timestamp = Dates.format(now(), "yyyymmdd_HHMMss")
    log_filename = joinpath(@__DIR__, "log_batch_$(instance_key)_S$(S_VAL)_$(log_timestamp).txt")
    log_io = open(log_filename, "w")
    original_stdout = stdout
    rd, wr = redirect_stdout()
    log_task = @async begin
        try
            while isopen(rd)
                data = readavailable(rd)
                isempty(data) && break
                write(original_stdout, data)
                flush(original_stdout)
                write(log_io, data)
                flush(log_io)
            end
        catch e
            e isa EOFError || rethrow()
        end
    end
    try
        println("Log file: $log_filename")
        println("Started: $(now())")
        println()
        println("=" ^ 70)
        println("INSTANCE: $instance_key (S=$S_VAL, ε̂=$EPSILON, ε̃=$EPSILON)")
        println("  Settings: MW cuts, mincut VI, inexact, mini-benders")
        println("=" ^ 70)

        network, td = setup_true_dro_instance(instance_key;
            S=S_VAL, epsilon_hat=EPSILON, epsilon_tilde=EPSILON)

        # VI-only OMP diagnostic
        if VI_SYM == :mincut
            println("\n" * "-" ^ 40)
            println("VI-only OMP diagnostic (no Benders cuts)")
            vi_omp, vi_vars = build_true_dro_omp(td; optimizer=Gurobi.Optimizer, silent=false)
            add_phase1_mincut_vi!(vi_omp, vi_vars, td)
            optimize!(vi_omp)
            vi_st = termination_status(vi_omp)
            if vi_st == MOI.OPTIMAL
                vi_lb = objective_value(vi_omp)
                vi_x = round.(Int, [value(vi_vars[:x][k]) for k in 1:td.num_arcs])
                @printf("  VI-only LB = %.6f, x = %s\n", vi_lb, string(vi_x))
            else
                println("  VI-only OMP: $vi_st")
            end
            println("-" ^ 40)
        end

        println("\n" * "=" ^ 70)
        println("True-DRO Benders")
        println("=" ^ 70)

        result = true_dro_benders_optimize!(td;
            mip_optimizer=Gurobi.Optimizer,
            nlp_optimizer=Gurobi.Optimizer,
            inexact=USE_INEXACT,
            max_iter=1000,
            tol=1e-4,
            verbose=true,
            sub_verbose=true,
            sub_time_limit=SUB_TL,
            mini_benders=USE_MINI,
            lp_optimizer=(USE_MINI ? Gurobi.Optimizer : nothing),
            max_mini_benders_iter=MAX_MB_ITER,
            strengthen_cuts=STRENGTHEN,
            valid_inequality=VI_SYM)

        gap = abs(result[:upper_bound] - result[:lower_bound]) /
              max(abs(result[:upper_bound]), 1e-10)

        println("\n" * "-" ^ 40)
        wt = get(result, :wall_time, NaN)
        @printf("%s: status=%s, Z₀=%.6f, iters=%d, time=%.2fs\n",
                instance_key, result[:status], result[:Z0], result[:iters], wt)
        @printf("  LB=%.6f, UB=%.6f, gap=%.2e\n",
                result[:lower_bound], result[:upper_bound], gap)
        x_int = round.(Int, result[:x])
        println("  x* = $x_int")

        # Profile 저장
        profile_key = (net=instance_key, S=S_VAL, εh=EPSILON, εt=EPSILON,
                       tl=SUB_TL, mb=USE_MINI, sc=STRENGTHEN, vi=VI_SYM)
        profile_path = joinpath(@__DIR__, "profiles.jls")
        profiles = isfile(profile_path) ? deserialize(profile_path) : Dict{NamedTuple, Dict}()
        profiles[profile_key] = result
        serialize(profile_path, profiles)
        println("  Profile saved → $profile_path ($(length(profiles)) entries)")

        # Primal recovery
        try
            rec = recover_and_print(td, result; optimizer=HiGHS.Optimizer)
        catch err
            println("  Recovery failed: $err")
        end

        println("\nFinished: $(now())")
        return (status=:ok, key=instance_key, Z0=result[:Z0],
                iters=result[:iters], wall=wt, gap=gap, x=x_int)
    catch err
        println("\nERROR: $err")
        bt = catch_backtrace()
        Base.showerror(stdout, err, bt)
        return (status=:error, key=instance_key, err=err)
    finally
        redirect_stdout(original_stdout)
        close(wr)
        try; wait(log_task); catch; end
        close(log_io)
        println("Log saved → $log_filename")
    end
end

# ===== 배치 실행 =====
println("=" ^ 70)
println("BATCH: True-DRO Benders over all networks")
println("  S=$S_VAL, ε̂=ε̃=$EPSILON, sub_tl=$(SUB_TL)s")
println("  MW + mincut + inexact + mini-benders")
println("  Networks (순서): $(NETWORKS)")
println("=" ^ 70)

batch_start = time()
all_results = []
for net in NETWORKS
    println("\n\n" * "#" ^ 70)
    println("# [$(findfirst(==(net), NETWORKS))/$(length(NETWORKS))] $net")
    println("#" ^ 70)
    flush(stdout)
    r = run_one(net)
    push!(all_results, r)
end
batch_wall = time() - batch_start

# ===== 최종 요약 =====
println("\n\n" * "=" ^ 70)
println("BATCH SUMMARY")
println("=" ^ 70)
@printf("%-14s %-10s %10s %6s %10s %10s\n",
        "network", "status", "Z0", "iters", "wall(s)", "gap")
println("-" ^ 70)
for r in all_results
    if r.status == :ok
        @printf("%-14s %-10s %10.6f %6d %10.2f %10.2e\n",
                r.key, "ok", r.Z0, r.iters, r.wall, r.gap)
    else
        @printf("%-14s %-10s\n", r.key, "ERROR")
    end
end
println("-" ^ 70)
@printf("Total wall time: %.2f s (%.2f min)\n", batch_wall, batch_wall / 60)
