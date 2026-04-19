"""
run_gamma1_batch.jl — polska, sioux_falls에 대해 γ=1로 True-DRO Benders 실행.

4개 β × 2개 네트워크 = 8 runs.
로그: S20_beta{β}_cov95/log_{net}_S20_..._gamma1.txt
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
    :sioux_falls => Dict(:type => :real_world, :generator => generate_sioux_falls_network),
    :polska      => Dict(:type => :real_world, :generator => generate_polska_network),
)

const S_VAL = 20
const GAMMA = 1           # γ=1 고정
const RHO = 0.2
const SEED = 42
const LAMBDA_U = 2.0
const SUB_TL = 30.0
const USE_MINI = true
const MAX_MB_ITER = 5
const STRENGTHEN = :mw
const USE_INEXACT = false
const VI_SYM = :mincut

const BETA_VALUES = [0.1, 0.3, 0.5, 0.8]
const NETWORKS = [:polska, :sioux_falls]


function setup_gamma1_instance(config_key::Symbol; S, β, ε_hat, ε_tilde)
    config = network_configs[config_key]
    network = config[:generator]()
    print_realworld_network_summary(network)

    num_arcs = length(network.arcs) - 1
    capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
        interdictable_arcs=network.interdictable_arcs, seed=SEED)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(RHO * GAMMA * c_bar; digits=4)

    q_hat = fill(1.0 / S, S)

    println("  |A|=$num_arcs, S=$S, γ=$GAMMA, w=$(round(w, digits=4)), λU=$LAMBDA_U")
    println("  ε̂=$ε_hat, ε̃=$ε_tilde")

    td = make_true_dro_data(network, capacities, q_hat, ε_hat, ε_tilde;
                            w=w, lambda_U=LAMBDA_U, gamma=GAMMA)
    return network, td
end


function run_one(instance_key::Symbol, β::Float64, ε_hat::Float64)
    β_str = replace(@sprintf("%.1f", β), "." => "p")
    log_dir = joinpath(@__DIR__, "S$(S_VAL)_beta$(β_str)_cov95")
    mkpath(log_dir)

    log_timestamp = Dates.format(now(), "yyyymmdd_HHMMSSs")
    log_filename = joinpath(log_dir, "log_$(instance_key)_S$(S_VAL)_$(log_timestamp)_gamma1.txt")
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
        @printf("INSTANCE: %s (S=%d, β=%.1f, γ=%d, ε̂=%.2f, ε̃=%.2f)\n",
                instance_key, S_VAL, β, GAMMA, ε_hat, ε_hat)
        println("  Settings: MW cuts, mincut VI, inexact=false, mini-benders")
        println("=" ^ 70)

        network, td = setup_gamma1_instance(instance_key;
            S=S_VAL, β=β, ε_hat=ε_hat, ε_tilde=ε_hat)

        # Min-cut VI diagnostic
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

        println("\n" * "=" ^ 70)
        println("True-DRO Benders")
        println("=" ^ 70)

        result = true_dro_benders_optimize!(td;
            mip_optimizer=Gurobi.Optimizer,
            nlp_optimizer=Gurobi.Optimizer,
            lp_optimizer=Gurobi.Optimizer,
            inexact=USE_INEXACT,
            max_iter=1000,
            tol=1e-4,
            verbose=true,
            sub_verbose=false,
            sub_time_limit=SUB_TL,
            mini_benders=USE_MINI,
            max_mini_benders_iter=MAX_MB_ITER,
            strengthen_cuts=STRENGTHEN,
            valid_inequality=VI_SYM)

        gap = abs(result[:upper_bound] - result[:lower_bound]) /
              max(abs(result[:upper_bound]), 1e-10)

        println("\n" * "-" ^ 40)
        wt = get(result, :wall_time, NaN)
        @printf("True-DRO: status=%s, Z₀=%.6f, iters=%d, time=%.2fs\n",
                result[:status], result[:Z0], result[:iters], wt)
        @printf("  LB=%.6f, UB=%.6f, gap=%.2e\n",
                result[:lower_bound], result[:upper_bound], gap)
        x_int = round.(Int, result[:x])
        println("  x* = $x_int")

        # Primal recovery
        try
            rec = recover_and_print(td, result; optimizer=HiGHS.Optimizer)
        catch err
            println("  Recovery failed: $err")
        end

        println("\nFinished: $(now())")
        return (status=:ok, key=instance_key, β=β, Z0=result[:Z0],
                iters=result[:iters], wall=wt, gap=gap, x=x_int)
    catch err
        println("\nERROR: $err")
        bt = catch_backtrace()
        Base.showerror(stdout, err, bt)
        return (status=:error, key=instance_key, β=β, err=err)
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
println("BATCH: γ=1 True-DRO Benders (polska, sioux_falls)")
println("  S=$S_VAL, γ=$GAMMA, ρ=$RHO")
println("  β values: $BETA_VALUES")
println("  MW + mincut + inexact=false + mini-benders")
println("=" ^ 70)

batch_start = time()
all_results = []

for β in BETA_VALUES
    ε_raw = lookup_epsilon(S_VAL, β; coverage=0.95)
    ε_hat = round(ε_raw; digits=2)
    @printf("\n### β=%.1f → ε̂=%.2f ###\n", β, ε_hat)

    for net in NETWORKS
        println("\n" * "#" ^ 70)
        @printf("# %s, β=%.1f, γ=%d\n", net, β, GAMMA)
        println("#" ^ 70)
        flush(stdout)
        r = run_one(net, β, ε_hat)
        push!(all_results, r)
    end
end

batch_wall = time() - batch_start

# ===== 최종 요약 =====
println("\n\n" * "=" ^ 70)
println("BATCH SUMMARY (γ=$GAMMA)")
println("=" ^ 70)
@printf("%-14s %5s %10s %6s %10s %10s  %s\n",
        "network", "β", "Z0", "iters", "wall(s)", "gap", "x*")
println("-" ^ 100)
for r in all_results
    if r.status == :ok
        @printf("%-14s %5.1f %10.6f %6d %10.2f %10.2e  %s\n",
                r.key, r.β, r.Z0, r.iters, r.wall, r.gap, string(r.x))
    else
        @printf("%-14s %5.1f  ERROR\n", r.key, r.β)
    end
end
println("-" ^ 100)
@printf("Total wall time: %.2f s (%.2f min)\n", batch_wall, batch_wall / 60)
