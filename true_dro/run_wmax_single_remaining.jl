"""
run_wmax_single_remaining.jl — β=0.8 single-layer (ε̃=0) 미완료 인스턴스 실행.

compact subproblem (build_true_dro_subproblem_single) 적용.
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
includet("oos_dirichlet.jl")

# ===== Config =====
network_configs = Dict(
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

const S_VAL = 20
const GAMMA_RATIO = 0.10
const SEED = 42
const LAMBDA_U = 2.0
const SUB_TL = 30.0
const USE_MINI = true
const MAX_MB_ITER = 5
const STRENGTHEN = :mw
const USE_INEXACT = true
const VI_SYM = :mincut

const BETA = 0.8
const NETWORKS = [:grid_5x5, :polska, :abilene, :nobel_us, :sioux_falls]


function setup_wmax_instance(config_key::Symbol; S, ε_hat)
    config = network_configs[config_key]
    if config[:type] == :grid
        network = generate_grid_network(config[:m], config[:n]; seed=SEED)
        print_network_summary(network)
    else
        network = config[:generator]()
        print_realworld_network_summary(network)
    end

    num_arcs = length(network.arcs) - 1
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = compute_interdict_budget(config_key, num_interdictable, GAMMA_RATIO)

    capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
        interdictable_arcs=network.interdictable_arcs, seed=SEED)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    w = round(maximum(capacities[interdictable_idx, :]); digits=4)

    q_hat = fill(1.0 / S, S)

    println("  |A|=$num_arcs, S=$S, γ=$γ, w=$(round(w, digits=4)) (=max cap), λU=$LAMBDA_U")
    println("  ε̂=$ε_hat, ε̃=0.0 (single-layer)")

    td = make_true_dro_data(network, capacities, q_hat, ε_hat, 0.0;
                            w=w, lambda_U=LAMBDA_U, gamma=γ)
    return network, td, γ
end


function run_one_single(instance_key::Symbol, ε_hat::Float64)
    β_str = replace(@sprintf("%.1f", BETA), "." => "p")
    log_dir = joinpath(@__DIR__, "S$(S_VAL)_beta$(β_str)_wmax")
    mkpath(log_dir)

    log_timestamp = Dates.format(now(), "yyyymmdd_HHMMSSs")
    log_filename = joinpath(log_dir, "log_$(instance_key)_S$(S_VAL)_$(log_timestamp)_single_compact_wmax.txt")
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
        network, td, γ = setup_wmax_instance(instance_key; S=S_VAL, ε_hat=ε_hat)

        @printf("SINGLE-LAYER (compact): %s (S=%d, β=%.1f, γ=%d, w=%.4f, ε̂=%.2f, ε̃=0.00)\n",
                instance_key, S_VAL, BETA, γ, td.w, ε_hat)
        println("  Settings: MW cuts, mincut VI, inexact=true, mini-benders, w=max(cap)")
        println("  Compact subproblem: ζF bilinear removed (ε̃=0)")
        println("=" ^ 70)

        result = true_dro_benders_optimize!(td;
            mip_optimizer=Gurobi.Optimizer,
            nlp_optimizer=Gurobi.Optimizer,
            lp_optimizer=Gurobi.Optimizer,
            inexact=USE_INEXACT,
            max_iter=1000,
            tol=1e-4,
            verbose=true,
            sub_verbose=true,
            sub_time_limit=SUB_TL,
            mini_benders=USE_MINI,
            max_mini_benders_iter=MAX_MB_ITER,
            strengthen_cuts=STRENGTHEN,
            valid_inequality=VI_SYM)

        gap = abs(result[:upper_bound] - result[:lower_bound]) /
              max(abs(result[:upper_bound]), 1e-10)

        println("\n" * "-" ^ 40)
        wt = get(result, :wall_time, NaN)
        @printf("Single-layer (compact): status=%s, Z₀=%.6f, iters=%d, time=%.2fs\n",
                result[:status], result[:Z0], result[:iters], wt)
        @printf("  LB=%.6f, UB=%.6f, gap=%.2e\n",
                result[:lower_bound], result[:upper_bound], gap)
        x_int = round.(Int, result[:x])
        println("  x* = $x_int")

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


# ===== 실행 =====
ε_raw = lookup_epsilon(S_VAL, BETA; coverage=0.95)
ε_hat = round(ε_raw; digits=2)

println("=" ^ 70)
println("SINGLE-LAYER REMAINING (β=$BETA, ε̂=$ε_hat, ε̃=0, compact subproblem)")
println("  w = max(interdictable cap), S=$S_VAL")
println("  Networks: $NETWORKS")
println("=" ^ 70)

batch_start = time()
all_results = []

for net in NETWORKS
    println("\n" * "#" ^ 70)
    @printf("# SINGLE (compact) %s, β=%.1f\n", net, BETA)
    println("#" ^ 70)
    flush(stdout)
    r = run_one_single(net, ε_hat)
    push!(all_results, r)
end

batch_wall = time() - batch_start

println("\n\n" * "=" ^ 70)
println("SINGLE-LAYER (compact) SUMMARY (β=$BETA, w=max cap)")
println("=" ^ 70)
@printf("%-14s %10s %6s %10s %10s  %s\n",
        "network", "Z0", "iters", "wall(s)", "gap", "x*")
println("-" ^ 90)
for r in all_results
    if r.status == :ok
        @printf("%-14s %10.6f %6d %10.2f %10.2e  %s\n",
                r.key, r.Z0, r.iters, r.wall, r.gap, string(r.x))
    else
        @printf("%-14s  ERROR\n", r.key)
    end
end
println("-" ^ 90)
@printf("Total wall time: %.2f s (%.2f min)\n", batch_wall, batch_wall / 60)
println("\nDone! $(now())")
