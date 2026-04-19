"""
run_wmax_batch.jl — w=max(capacity) 로 recovery budget 키워서 3-variant 순차 실행.

순서: True-DRO (double) → Single-layer (ε̃=0) → Nominal SP
4개 β × 5개 네트워크.
로그: S20_beta{β}_wmax/log_{net}_S20_..._wmax.txt
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

# Nominal SP
if !@isdefined(build_full_2SP_model)
    _nominal_sp_src = read(joinpath(@__DIR__, "..", "build_nominal_sp.jl"), String)
    _func_start = findfirst("function build_full_2SP_model", _nominal_sp_src)
    _func_str = _nominal_sp_src[first(_func_start):end]
    include_string(Main, _func_str)
    @info "build_full_2SP_model loaded"
end

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

const BETA_VALUES = [0.1, 0.3, 0.5, 0.8]
const NETWORKS = [:grid_5x5, :polska, :abilene, :nobel_us, :sioux_falls]


function setup_wmax_instance(config_key::Symbol; S, β, ε_hat, ε_tilde)
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
    println("  ε̂=$ε_hat, ε̃=$ε_tilde")

    td = make_true_dro_data(network, capacities, q_hat, ε_hat, ε_tilde;
                            w=w, lambda_U=LAMBDA_U, gamma=γ)
    return network, td, γ
end


function run_one(instance_key::Symbol, β::Float64, ε_hat::Float64)
    β_str = replace(@sprintf("%.1f", β), "." => "p")
    log_dir = joinpath(@__DIR__, "S$(S_VAL)_beta$(β_str)_wmax")
    mkpath(log_dir)

    log_timestamp = Dates.format(now(), "yyyymmdd_HHMMSSs")
    log_filename = joinpath(log_dir, "log_$(instance_key)_S$(S_VAL)_$(log_timestamp)_wmax.txt")
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
        network, td, γ = setup_wmax_instance(instance_key;
            S=S_VAL, β=β, ε_hat=ε_hat, ε_tilde=ε_hat)

        @printf("INSTANCE: %s (S=%d, β=%.1f, γ=%d, w=%.4f, ε̂=%.2f, ε̃=%.2f)\n",
                instance_key, S_VAL, β, γ, td.w, ε_hat, ε_hat)
        println("  Settings: MW cuts, mincut VI, inexact=false, mini-benders, w=max(cap)")
        println("=" ^ 70)

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
println("BATCH: w=max(cap) True-DRO Benders (5 networks)")
println("  S=$S_VAL, γ_ratio=$GAMMA_RATIO, w=max(interdictable cap)")
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
        @printf("# %s, β=%.1f\n", net, β)
        println("#" ^ 70)
        flush(stdout)
        r = run_one(net, β, ε_hat)
        push!(all_results, r)
    end
end

batch_wall = time() - batch_start

# ===== True-DRO 요약 =====
println("\n\n" * "=" ^ 70)
println("TRUE-DRO SUMMARY (w=max cap)")
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
@printf("True-DRO total wall time: %.2f s (%.2f min)\n", batch_wall, batch_wall / 60)


# ============================================================
# PHASE 2: Single-layer (ε̃=0)
# ============================================================

function run_one_single(instance_key::Symbol, β::Float64, ε_hat::Float64)
    β_str = replace(@sprintf("%.1f", β), "." => "p")
    log_dir = joinpath(@__DIR__, "S$(S_VAL)_beta$(β_str)_wmax")
    mkpath(log_dir)

    log_timestamp = Dates.format(now(), "yyyymmdd_HHMMSSs")
    log_filename = joinpath(log_dir, "log_$(instance_key)_S$(S_VAL)_$(log_timestamp)_single_wmax.txt")
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
        network, td, γ = setup_wmax_instance(instance_key;
            S=S_VAL, β=β, ε_hat=ε_hat, ε_tilde=0.0)

        @printf("SINGLE-LAYER: %s (S=%d, β=%.1f, γ=%d, w=%.4f, ε̂=%.2f, ε̃=0.00)\n",
                instance_key, S_VAL, β, γ, td.w, ε_hat)
        println("  Settings: MW cuts, mincut VI, inexact=false, mini-benders, w=max(cap)")
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
        @printf("Single-layer: status=%s, Z₀=%.6f, iters=%d, time=%.2fs\n",
                result[:status], result[:Z0], result[:iters], wt)
        @printf("  LB=%.6f, UB=%.6f, gap=%.2e\n",
                result[:lower_bound], result[:upper_bound], gap)
        x_int = round.(Int, result[:x])
        println("  x* = $x_int")

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

println("\n\n" * "=" ^ 70)
println("BATCH PHASE 2: Single-layer (ε̃=0, w=max cap)")
println("=" ^ 70)

single_start = time()
single_results = []

for β in BETA_VALUES
    ε_raw = lookup_epsilon(S_VAL, β; coverage=0.95)
    ε_hat = round(ε_raw; digits=2)
    @printf("\n### β=%.1f → ε̂=%.2f (ε̃=0) ###\n", β, ε_hat)

    for net in NETWORKS
        println("\n" * "#" ^ 70)
        @printf("# SINGLE %s, β=%.1f\n", net, β)
        println("#" ^ 70)
        flush(stdout)
        r = run_one_single(net, β, ε_hat)
        push!(single_results, r)
    end
end

single_wall = time() - single_start

println("\n\n" * "=" ^ 70)
println("SINGLE-LAYER SUMMARY (w=max cap)")
println("=" ^ 70)
@printf("%-14s %5s %10s %6s %10s %10s  %s\n",
        "network", "β", "Z0", "iters", "wall(s)", "gap", "x*")
println("-" ^ 100)
for r in single_results
    if r.status == :ok
        @printf("%-14s %5.1f %10.6f %6d %10.2f %10.2e  %s\n",
                r.key, r.β, r.Z0, r.iters, r.wall, r.gap, string(r.x))
    else
        @printf("%-14s %5.1f  ERROR\n", r.key, r.β)
    end
end
println("-" ^ 100)
@printf("Single-layer total wall time: %.2f s (%.2f min)\n", single_wall, single_wall / 60)


# ============================================================
# PHASE 3: Nominal SP (β-independent)
# ============================================================

function regenerate_wmax_network(config_key::Symbol; S::Int=S_VAL)
    config = network_configs[config_key]
    if config[:type] == :grid
        network = generate_grid_network(config[:m], config[:n]; seed=SEED)
    else
        network = config[:generator]()
    end
    num_arcs = length(network.arcs) - 1
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = compute_interdict_budget(config_key, num_interdictable, GAMMA_RATIO)

    capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
        interdictable_arcs=network.interdictable_arcs, seed=SEED)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    w = round(maximum(capacities[interdictable_idx, :]); digits=4)

    return network, capacities, w, γ
end

function run_one_nominal(instance_key::Symbol)
    log_dir = joinpath(@__DIR__, "S$(S_VAL)_nominal_wmax")
    mkpath(log_dir)

    log_timestamp = Dates.format(now(), "yyyymmdd_HHMMSSs")
    log_filename = joinpath(log_dir, "log_$(instance_key)_S$(S_VAL)_$(log_timestamp)_nominal_wmax.txt")
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

        network, capacities, w, γ = regenerate_wmax_network(instance_key)
        num_arcs = length(network.arcs) - 1

        println("=" ^ 70)
        @printf("NOMINAL SP: %s (S=%d, γ=%d, w=%.4f)\n", instance_key, S_VAL, γ, w)
        println("=" ^ 70)

        xi_bar_vecs = [capacities[1:num_arcs, s] for s in 1:S_VAL]
        uncertainty_set = Dict(:xi_bar => xi_bar_vecs, :R => zeros(0, 0),
                               :r_dict_hat => Dict(), :epsilon_hat => 0.0)
        model, vars = build_full_2SP_model(network, S_VAL, LAMBDA_U, LAMBDA_U, γ, w, 1.0, uncertainty_set)
        set_optimizer_attribute(model, "OutputFlag", 0)

        t0 = time()
        optimize!(model)
        wt = time() - t0

        if termination_status(model) != MOI.OPTIMAL
            @warn "Nominal SP status: $(termination_status(model))"
            println("\nFinished: $(now())")
            return (status=:error, key=instance_key, err="$(termination_status(model))")
        end

        x_star = Float64.(round.(Int, value.(vars[:x])))
        Z0 = objective_value(model)
        x_int = round.(Int, x_star)

        @printf("Nominal SP: Z₀=%.6f, time=%.2fs\n", Z0, wt)
        println("  x* = $x_int")

        println("\nFinished: $(now())")
        return (status=:ok, key=instance_key, Z0=Z0, wall=wt, x=x_int)
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

println("\n\n" * "=" ^ 70)
println("BATCH PHASE 3: Nominal SP (w=max cap, β-independent)")
println("=" ^ 70)

nominal_start = time()
nominal_results = []

for net in NETWORKS
    println("\n" * "#" ^ 70)
    @printf("# NOMINAL %s\n", net)
    println("#" ^ 70)
    flush(stdout)
    r = run_one_nominal(net)
    push!(nominal_results, r)
end

nominal_wall = time() - nominal_start

println("\n\n" * "=" ^ 70)
println("NOMINAL SP SUMMARY (w=max cap)")
println("=" ^ 70)
@printf("%-14s %10s %10s  %s\n", "network", "Z0", "wall(s)", "x*")
println("-" ^ 70)
for r in nominal_results
    if r.status == :ok
        @printf("%-14s %10.6f %10.2f  %s\n", r.key, r.Z0, r.wall, string(r.x))
    else
        @printf("%-14s  ERROR\n", r.key)
    end
end
println("-" ^ 70)
@printf("Nominal total wall time: %.2f s\n", nominal_wall)

total_wall = batch_wall + single_wall + nominal_wall
println("\n" * "=" ^ 70)
@printf("GRAND TOTAL wall time: %.2f s (%.2f min)\n", total_wall, total_wall / 60)
println("=" ^ 70)
println("\nDone! $(now())")
