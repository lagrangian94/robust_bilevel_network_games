"""
run_eps_calibration.jl — Factor 5 additive model에서 ε calibration (two-phase bisection).

Phase 1: PO-PP bisection → ε^S (no DRO solve, cheap)
Phase 2: NR-WR bisection → ε^D (DRO solve needed, expensive)

Usage:
    julia true_dro/factor_5/run_eps_calibration.jl
"""

using JuMP, Gurobi, Printf, Dates, Serialization, LinearAlgebra

include("../../network_generator.jl")
using .NetworkGenerator

include("../true_dro_data.jl")
include("../true_dro_build_omp.jl")
include("../true_dro_build_subproblem.jl")
include("../true_dro_build_isp_leader.jl")
include("../true_dro_build_isp_follower.jl")
include("../true_dro_benders.jl")
include("../true_dro_mincut_vi.jl")

# ============================================================
# Config
# ============================================================
const F5_S = 20
const F5_SEED = 42
const F5_LAMBDA_U = 10.0
const F5_NUM_FACTORS = 5

# Known x* from previous solves (additive factor, all intd)
const KNOWN_X = Dict(
    :polska   => (γ=1, nom_arcs=[18], rob_arcs=[33], all_intd=true),
    :abilene  => (γ=1, nom_arcs=[11], rob_arcs=[5],  all_intd=true),
)

const NET_GENERATORS = Dict(
    :polska   => generate_polska_network,
    :abilene  => generate_abilene_network,
)

function f5_setup_network(net_key::Symbol)
    net = NET_GENERATORS[net_key]()
    num_arcs = length(net.arcs) - 1
    kx = KNOWN_X[net_key]

    if kx.all_intd
        intd_flags = fill(true, length(net.arcs))
        net = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
            net.N, intd_flags, net.arc_adjacency, net.node_arc_incidence)
    end

    caps, _ = generate_capacity_scenarios_factor_sparse(length(net.arcs), F5_S;
        interdictable_arcs=net.interdictable_arcs, seed=F5_SEED, num_factors=F5_NUM_FACTORS)
    intd_idx = findall(net.interdictable_arcs[1:num_arcs])
    w = round(maximum(caps[intd_idx, :]); digits=4)
    γ = kx.γ

    x_nom = zeros(Float64, num_arcs)
    for a in kx.nom_arcs; x_nom[a] = 1.0; end
    x_rob = zeros(Float64, num_arcs)
    for a in kx.rob_arcs; x_rob[a] = 1.0; end

    q_hat = fill(1.0/F5_S, F5_S)

    return net, caps, w, γ, q_hat, x_nom, x_rob, num_arcs
end

# ============================================================
# evaluate_f: subproblem 평가
# ============================================================
function build_evaluate_f_model(td::TrueDROData;
        optimizer=Gurobi.Optimizer,
        nonconvex_attr=("NonConvex" => 2))
    K = td.num_arcs
    x_dummy = zeros(K)
    if td.eps_hat == 0.0 && td.eps_tilde == 0.0
        model, vars = build_true_dro_subproblem_nominal(td, x_dummy;
            optimizer=optimizer_with_attributes(optimizer, "OutputFlag" => 1, "LogToConsole" => 1),
            silent=false, rho_upper_bound=10.0)
        return model, vars
    else
        model, vars = build_true_dro_subproblem(td, x_dummy;
            optimizer=optimizer_with_attributes(optimizer, "OutputFlag" => 1, "LogToConsole" => 1, nonconvex_attr),
            silent=false, rho_upper_bound=10.0)
        return model, vars
    end
end

function evaluate_f(model, vars, td::TrueDROData, x::Vector{Float64};
        sub_time_limit::Float64=1800.0,
        mip_gap::Float64=0.005,
        log_file::String="")
    set_optimizer_attribute(model, "TimeLimit", sub_time_limit)
    set_optimizer_attribute(model, "MIPGap", mip_gap)
    if !isempty(log_file)
        set_optimizer_attribute(model, "LogFile", log_file)
    end
    @printf("    evaluate_f: solving (TimeLimit=%.0fs)... \n", sub_time_limit)
    flush(stdout)
    t0 = time()
    sub_info = solve_true_dro_subproblem!(model, vars, td, x)
    @printf("    done (%.1fs, Z₀=%.6f, status=%s)\n", time()-t0, sub_info[:Z0_val], termination_status(model))
    flush(stdout)
    return sub_info[:Z0_val]
end

# ============================================================
# solve_dro wrapper
# ============================================================
function solve_dro_f5(td::TrueDROData; verbose=true)
    result = true_dro_benders_optimize!(td;
        mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
        max_iter=500, tol=1e-4, verbose=verbose, sub_time_limit=30.0,
        mini_benders=true, max_mini_benders_iter=5,
        strengthen_cuts=:mw, valid_inequality=:mincut,
        inexact=true, nonconvex_attr=("NonConvex" => 2))
    return result
end

# ============================================================
# Two-phase bisection
# ============================================================
function find_epsilon_range_f5(net_key::Symbol;
        eps_max=1.0, tol=0.01, max_iter_phase1=15, max_iter_phase2=10, verbose=true)

    net, caps, w, γ, q_hat, x_neut, x_rob, num_arcs = f5_setup_network(net_key)
    kx = KNOWN_X[net_key]

    if verbose
        println("=" ^ 70)
        @printf("ε calibration: %s (γ=%d, all_intd=%s)\n", net_key, γ, kx.all_intd)
        @printf("  x_neut=%s, x_rob=%s, w=%.4f\n", string(kx.nom_arcs), string(kx.rob_arcs), w)
        println("=" ^ 70)
    end

    # Baseline models
    td_0 = make_true_dro_data(net, caps, q_hat, 0.0, 0.0; w=w, lambda_U=F5_LAMBDA_U, gamma=γ)
    td_1 = make_true_dro_data(net, caps, q_hat, eps_max, eps_max; w=w, lambda_U=F5_LAMBDA_U, gamma=γ)

    model_0, vars_0 = build_evaluate_f_model(td_0)
    model_1, vars_1 = build_evaluate_f_model(td_1)

    log_dir = joinpath(@__DIR__, "logs")
    mkpath(log_dir)
    gurobi_log = joinpath(log_dir, "gurobi_$(net_key).log")

    f0_neut = evaluate_f(model_0, vars_0, td_0, x_neut; log_file=gurobi_log)
    f1_rob  = evaluate_f(model_1, vars_1, td_1, x_rob; log_file=gurobi_log)

    if verbose
        @printf("  f_0(x_neut)=%.6f, f_1(x_rob)=%.6f\n", f0_neut, f1_rob)
    end

    history = Dict[]

    # ================================================================
    # Phase 1: PO-PP bisection → ε^S
    # ================================================================
    if verbose
        println("\n" * "=" ^ 70)
        println("Phase 1: PO-PP bisection (no DRO solves)")
        println("=" ^ 70)
    end

    lo1, hi1 = 0.0, eps_max
    ε1 = (lo1 + hi1) / 2.0

    for iter in 1:max_iter_phase1
        td_ε = make_true_dro_data(net, caps, q_hat, ε1, ε1; w=w, lambda_U=F5_LAMBDA_U, gamma=γ)
        model_ε, vars_ε = build_evaluate_f_model(td_ε)

        f_ε_neut = evaluate_f(model_ε, vars_ε, td_ε, x_neut; log_file=gurobi_log)
        f_ε_rob  = evaluate_f(model_ε, vars_ε, td_ε, x_rob; log_file=gurobi_log)
        po_pp = f_ε_neut - f_ε_rob

        if verbose
            @printf("  P1 iter %2d: ε=%.6f, PO-PP=%.6f [lo=%.6f, hi=%.6f]\n",
                    iter, ε1, po_pp, lo1, hi1)
        end

        push!(history, Dict(:phase=>1, :iter=>iter, :eps=>ε1, :po_pp=>po_pp))

        if po_pp < 0
            lo1 = ε1
        else
            hi1 = ε1
        end
        ε1 = (lo1 + hi1) / 2.0

        (hi1 - lo1) < tol && break
    end

    eps_S = (lo1 + hi1) / 2.0
    if verbose
        @printf("\n  Phase 1 result: ε^S ≈ %.6f [%.6f, %.6f]\n", eps_S, lo1, hi1)
    end

    # ================================================================
    # Phase 2: NR-WR bisection → ε^D (from ε^S rightward)
    # ================================================================
    if verbose
        println("\n" * "=" ^ 70)
        println("Phase 2: NR-WR bisection (DRO solves)")
        @printf("  Search range: [%.6f, %.6f]\n", hi1, eps_max)
        println("=" ^ 70)
    end

    # Pre-check: NR-WR at eps_max
    f0_rob = evaluate_f(model_0, vars_0, td_0, x_rob)
    nr_max = f0_rob - f0_neut
    wr_max = 0.0  # f_1(x_rob) - f_1(x_rob) = 0
    nr_wr_max = nr_max - wr_max

    if verbose
        @printf("  NR-WR at ε=%.4f: NR=%.6f, WR=0, NR-WR=%.6f\n", eps_max, nr_max, nr_wr_max)
    end

    lo2, hi2 = hi1, eps_max
    eps_D = eps_max
    dro_solves = Dict{Float64, Any}()

    if nr_wr_max < 0
        if verbose; println("  NR-WR(ε_max) < 0 → ε^D ≥ ε_max"); end
        eps_D = eps_max
    else
        ε2 = (lo2 + hi2) / 2.0
        for iter in 1:max_iter_phase2
            if verbose
                @printf("\n  P2 iter %2d: ε=%.6f [lo=%.6f, hi=%.6f]\n", iter, ε2, lo2, hi2)
            end

            td_ε = make_true_dro_data(net, caps, q_hat, ε2, ε2; w=w, lambda_U=F5_LAMBDA_U, gamma=γ)

            t0 = time()
            res_star = solve_dro_f5(td_ε; verbose=verbose)
            t_dro = time() - t0
            x_star = res_star[:x]
            dro_solves[ε2] = res_star

            if verbose
                @printf("  x*(ε=%.6f): Z₀=%.6f, iters=%d, time=%.1fs, x=%s\n",
                        ε2, res_star[:Z0], res_star[:iters], t_dro, string(findall(round.(Int, x_star) .> 0)))
            end

            f0_star = evaluate_f(model_0, vars_0, td_0, x_star)
            f1_star = evaluate_f(model_1, vars_1, td_1, x_star)

            nr = f0_star - f0_neut
            wr = f1_star - f1_rob
            nr_wr = nr - wr

            if verbose
                @printf("  NR=%.6f, WR=%.6f, NR-WR=%.6f\n", nr, wr, nr_wr)
            end

            push!(history, Dict(:phase=>2, :iter=>iter, :eps=>ε2, :nr=>nr, :wr=>wr, :nr_wr=>nr_wr))

            if nr_wr < 0
                lo2 = ε2
            else
                hi2 = ε2
            end
            ε2 = (lo2 + hi2) / 2.0

            (hi2 - lo2) < tol && break
        end
        eps_D = (lo2 + hi2) / 2.0
    end

    if verbose
        println("\n" * "=" ^ 70)
        @printf("Result: ε^S ≈ %.6f, ε^D ≈ %.6f\n", eps_S, eps_D)
        @printf("Recommended: ε = %.6f (midpoint of [ε^S, ε^D])\n", (eps_S + eps_D) / 2.0)
        println("=" ^ 70)
    end

    return Dict(
        :net_key => net_key,
        :eps_S => eps_S, :eps_S_lo => lo1, :eps_S_hi => hi1,
        :eps_D => eps_D, :eps_D_lo => lo2, :eps_D_hi => hi2,
        :eps_recommended => (eps_S + eps_D) / 2.0,
        :x_neut => x_neut, :x_rob => x_rob,
        :f0_neut => f0_neut, :f1_rob => f1_rob,
        :history => history, :dro_solves => dro_solves,
    )
end

# ============================================================
# Main: run for polska and abilene
# ============================================================
net_key = Symbol(ARGS[1])  # :polska or :abilene
println("Running ε calibration for $net_key")
result = find_epsilon_range_f5(net_key)

# Save
jls_path = joinpath(@__DIR__, "eps_calibration_$(net_key).jls")
serialize(jls_path, result)
println("\nSaved → $jls_path")
