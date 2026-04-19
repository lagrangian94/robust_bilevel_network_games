"""
run_wmax_double_grid5x5.jl — β=0.8 double-layer grid_5x5 (w=max cap) 1건 실행.

mini-benders timing 비교용 (single-layer compact와 대조).
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
const NET_KEY = :grid_5x5

# ===== Setup =====
network = generate_grid_network(5, 5; seed=SEED)
print_network_summary(network)

num_arcs = length(network.arcs) - 1
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, GAMMA_RATIO * num_interdictable)

capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S_VAL;
    interdictable_arcs=network.interdictable_arcs, seed=SEED)

interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
w = round(maximum(capacities[interdictable_idx, :]); digits=4)

ε_raw = lookup_epsilon(S_VAL, BETA; coverage=0.95)
ε_hat = round(ε_raw; digits=2)

q_hat = fill(1.0 / S_VAL, S_VAL)

println("=" ^ 70)
@printf("DOUBLE-LAYER: grid_5x5 (S=%d, β=%.1f, γ=%d, w=%.4f)\n", S_VAL, BETA, γ, w)
@printf("  ε̂=%.2f, ε̃=%.2f (double-layer)\n", ε_hat, ε_hat)
println("  Settings: MW cuts, mincut VI, inexact=true, mini-benders")
println("=" ^ 70)

td = make_true_dro_data(network, capacities, q_hat, ε_hat, ε_hat;
                        w=w, lambda_U=LAMBDA_U, gamma=γ)

# ===== Logging =====
β_str = replace(@sprintf("%.1f", BETA), "." => "p")
log_dir = joinpath(@__DIR__, "S$(S_VAL)_beta$(β_str)_wmax")
mkpath(log_dir)

log_timestamp = Dates.format(now(), "yyyymmdd_HHMMSSs")
log_filename = joinpath(log_dir, "log_grid_5x5_S$(S_VAL)_$(log_timestamp)_double_wmax.txt")
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
    @printf("Double-layer: status=%s, Z₀=%.6f, iters=%d, time=%.2fs\n",
            result[:status], result[:Z0], result[:iters], wt)
    @printf("  LB=%.6f, UB=%.6f, gap=%.2e\n",
            result[:lower_bound], result[:upper_bound], gap)
    x_int = round.(Int, result[:x])
    println("  x* = $x_int")
    println("\nFinished: $(now())")
catch err
    println("\nERROR: $err")
    bt = catch_backtrace()
    Base.showerror(stdout, err, bt)
finally
    redirect_stdout(original_stdout)
    close(wr)
    try; wait(log_task); catch; end
    close(log_io)
    println("Log saved → $log_filename")
end
