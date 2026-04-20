"""
run_wmax_nominal.jl — Nominal SP (build_full_2SP_model) w=max cap, 5 networks.

직접 LP로 풀어서 x*, Z₀ 산출.
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

# Nominal SP (build_full_2SP_model 함수만 로드)
if !@isdefined(build_full_2SP_model)
    _nominal_sp_src = read(joinpath(@__DIR__, "..", "build_nominal_sp.jl"), String)
    _func_start = findfirst("function build_full_2SP_model", _nominal_sp_src)
    _func_str = _nominal_sp_src[first(_func_start):end]
    include_string(Main, _func_str)
    @info "build_full_2SP_model loaded"
end

# # Benders framework (주석처리)
# includet("true_dro_build_omp.jl")
# includet("true_dro_build_subproblem.jl")
# includet("true_dro_build_isp_leader.jl")
# includet("true_dro_build_isp_follower.jl")
# includet("true_dro_benders.jl")
# includet("true_dro_mincut_vi.jl")
# includet("true_dro_recover.jl")
# includet("oos_dirichlet.jl")

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
const NETWORKS = [:grid_5x5, :polska, :abilene, :nobel_us, :sioux_falls]


function regenerate_wmax_network(config_key::Symbol; S::Int=S_VAL)
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
        set_optimizer_attribute(model, "OutputFlag", 1)
        set_optimizer_attribute(model, "LogFile", "nominal_gurobi.log")

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


# ===== 실행 =====
println("=" ^ 70)
println("NOMINAL SP BATCH (w=max cap, S=$S_VAL)")
println("  Networks: $NETWORKS")
println("=" ^ 70)

batch_start = time()
all_results = []

for net in NETWORKS
    println("\n" * "#" ^ 70)
    @printf("# NOMINAL %s\n", net)
    println("#" ^ 70)
    flush(stdout)
    r = run_one_nominal(net)
    push!(all_results, r)
end

batch_wall = time() - batch_start

println("\n\n" * "=" ^ 70)
println("NOMINAL SP SUMMARY (w=max cap)")
println("=" ^ 70)
@printf("%-14s %10s %10s  %s\n", "network", "Z0", "wall(s)", "x*")
println("-" ^ 70)
for r in all_results
    if r.status == :ok
        @printf("%-14s %10.6f %10.2f  %s\n", r.key, r.Z0, r.wall, string(r.x))
    else
        @printf("%-14s  ERROR\n", r.key)
    end
end
println("-" ^ 70)
@printf("Total wall time: %.2f s (%.2f min)\n", batch_wall, batch_wall / 60)
println("\nDone! $(now())")
