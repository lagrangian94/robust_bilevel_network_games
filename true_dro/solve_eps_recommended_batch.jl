"""
solve_eps_recommended_batch.jl — exp2_calibration_results.jls의 각 네트워크에 대해
eps_recommended에서 x*를 solve하고, :x_star/:Z0_star를 추가하여 재저장.

grid_5x5는 이미 별도 로그에 저장되어 있으므로 건너뛴다.
"""

using Revise
using JuMP
using Gurobi
using Printf
using Serialization
using Dates

# ---- Load modules ----
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

# ---- Config (run_experiment2_calibration.jl과 동일) ----
const EXP2_S = 20
const EXP2_GAMMA_RATIO = 0.10
const EXP2_SEED = 42
const EXP2_LAMBDA_U = 2.0

const EXP2_NET_CONFIGS = Dict(
    :grid_5x5    => Dict(:type => :grid, :m => 5, :n => 5),
    :sioux_falls => Dict(:type => :real_world, :generator => generate_sioux_falls_network),
    :nobel_us    => Dict(:type => :real_world, :generator => generate_nobel_us_network),
    :abilene     => Dict(:type => :real_world, :generator => generate_abilene_network),
    :polska      => Dict(:type => :real_world, :generator => generate_polska_network),
)

function _compute_gamma(config_key::Symbol, num_interdictable::Int)
    if config_key in (:sioux_falls, :abilene)
        return 2
    end
    return ceil(Int, EXP2_GAMMA_RATIO * num_interdictable)
end

function _regenerate_network(config_key::Symbol)
    config = EXP2_NET_CONFIGS[config_key]
    network = config[:type] == :grid ?
        generate_grid_network(config[:m], config[:n]; seed=EXP2_SEED) :
        config[:generator]()

    num_arcs = length(network.arcs) - 1
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = _compute_gamma(config_key, num_interdictable)

    capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), EXP2_S;
        interdictable_arcs=network.interdictable_arcs, seed=EXP2_SEED)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    w = round(maximum(capacities[interdictable_idx, :]); digits=4)

    return network, capacities, w, γ
end

# ---- Load existing results ----
jls_path = joinpath(@__DIR__, "exp2_calibration_results.jls")
data = deserialize(jls_path)

# ---- Log directory ----
log_dir = joinpath(@__DIR__, "exp2_eps_recommended_solves")
mkpath(log_dir)
println("Log directory: $log_dir")

# ---- Networks to solve (grid_5x5 제외) ----
targets = [:polska, :abilene, :nobel_us, :sioux_falls]

for net_key in targets
    r = data[net_key]
    ε_rec = r[:eps_recommended]

    # 이미 x_star가 있으면 건너뛰기
    if haskey(r, :x_star) && r[:x_star] !== nothing
        @printf("%-14s: x_star already exists, skipping.\n", net_key)
        continue
    end

    println("\n" * "#" ^ 70)
    @printf("# %s: solving DRO at ε_recommended = %.6f\n", net_key, ε_rec)
    println("#" ^ 70)
    flush(stdout)

    # Network setup
    network, capacities, w, γ = _regenerate_network(net_key)
    num_arcs = length(network.arcs) - 1
    q_hat = fill(1.0 / EXP2_S, EXP2_S)

    @printf("  arcs=%d, γ=%d, w=%.4f, ε=%.6f\n", num_arcs, γ, w, ε_rec)

    td = make_true_dro_data(network, capacities, q_hat, ε_rec, ε_rec;
                             w=w, lambda_U=EXP2_LAMBDA_U, gamma=γ)

    # Log file setup (stdout tee)
    log_filename = joinpath(log_dir, "log_$(net_key)_eps$(round(ε_rec; digits=4))_$(Dates.format(now(), "yyyymmdd_HHMMSSs")).txt")
    log_io = open(log_filename, "w")
    original_stdout = stdout
    rd, wr = redirect_stdout()
    log_task = @async begin
        try
            while isopen(rd)
                buf = readavailable(rd)
                isempty(buf) && break
                write(original_stdout, buf)
                flush(original_stdout)
                write(log_io, buf)
                flush(log_io)
            end
        catch e
            e isa EOFError || rethrow()
        end
    end

    try
        println("Log file: $log_filename")
        println("Started: $(now())")
        @printf("DRO solve: %s, S=%d, ε̂=ε̃=%.6f, γ=%d, w=%.4f, λU=%.1f\n",
                net_key, EXP2_S, ε_rec, γ, w, EXP2_LAMBDA_U)
        println()

        t0 = time()
        result = true_dro_benders_optimize!(td;
            mip_optimizer = Gurobi.Optimizer,
            nlp_optimizer = Gurobi.Optimizer,
            lp_optimizer  = Gurobi.Optimizer,
            max_iter = 500,
            tol = 1e-4,
            verbose = true,
            sub_time_limit = 30.0,
            mini_benders = true,
            max_mini_benders_iter = 5,
            strengthen_cuts = :mw,
            valid_inequality = :mincut,
            inexact = true,
            nonconvex_attr = ("NonConvex" => 2))
        wt = time() - t0

        x_star = result[:x]
        Z0 = result[:Z0]
        x_int = round.(Int, x_star)

        @printf("\nTrue-DRO: status=%s, Z₀=%.6f, iters=%d, time=%.2fs\n",
                result[:status], Z0, result[:iters], wt)
        println("  x* = $x_int")
        println("  interdicted arcs: $(findall(x_int .> 0))")
        println("\nFinished: $(now())")

        # 결과 저장
        data[net_key][:x_star] = x_star
        data[net_key][:Z0_star] = Z0
        serialize(jls_path, data)
        @printf("  Updated %s in %s\n", net_key, jls_path)

    catch err
        println("\nERROR: $err")
        bt = catch_backtrace()
        Base.showerror(stdout, err, bt)
    finally
        redirect_stdout(original_stdout)
        close(wr)
        try; wait(log_task); catch; end
        close(log_io)
        println("  Log saved → $log_filename")
    end

    flush(stdout)
end

# ---- grid_5x5: dro_solves에서 가장 가까운 ε의 x* 사용 ----
if !haskey(data[:grid_5x5], :x_star) || data[:grid_5x5][:x_star] === nothing
    r = data[:grid_5x5]
    # grid_5x5는 이미 별도로 풀었음 (exp2_grid5x5_console_log.txt)
    # dro_solves에서 eps_recommended에 가장 가까운 결과 사용
    ε_rec = r[:eps_recommended]
    closest_ε = sort(collect(keys(r[:dro_solves])); by=e -> abs(e - ε_rec))[1]
    x_star = r[:dro_solves][closest_ε][:x]
    Z0_star = r[:dro_solves][closest_ε][:Z0]

    data[:grid_5x5][:x_star] = x_star
    data[:grid_5x5][:Z0_star] = Z0_star
    serialize(jls_path, data)
    @printf("\ngrid_5x5: used dro_solves[ε=%.6f] → x*=%s, Z₀=%.6f\n",
            closest_ε, string(round.(Int, x_star)), Z0_star)
end

# ---- Summary ----
println("\n" * "=" ^ 80)
println("Summary: x_star at ε_recommended")
println("=" ^ 80)
@printf("%-14s  %10s  %10s  %s\n", "Network", "ε_rec", "Z₀_star", "x* (nonzero arcs)")
println("-" ^ 80)
for net_key in [:grid_5x5, :polska, :abilene, :nobel_us, :sioux_falls]
    r = data[net_key]
    x_int = round.(Int, r[:x_star])
    @printf("%-14s  %10.6f  %10.6f  %s\n",
            net_key, r[:eps_recommended],
            r[:Z0_star], string(findall(x_int .> 0)))
end
println("-" ^ 80)
println("\nAll results saved → $jls_path")
println("Done! $(now())")
