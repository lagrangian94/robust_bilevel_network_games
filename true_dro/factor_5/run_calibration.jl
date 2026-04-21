"""
factor_3/run_calibration.jl — Factor k=3 실험: Two-phase ε calibration.

nominal/robust 결과를 results/에서 로드한 뒤 two-phase bisection 실행.
Phase 1: PO-PP bisection → ε^S (DRO solve 없이)
Phase 2: NR-WR bisection → ε^D (ε^S부터, DRO solve 필요)

Usage:
    include("true_dro/factor_3/run_calibration.jl")
"""

using Revise
using JuMP
using Gurobi
using Printf
using Serialization
using Dates

if !@isdefined(NetworkGenerator)
    include("../../network_generator.jl")
end
using .NetworkGenerator

includet("../true_dro_data.jl")
includet("../true_dro_build_omp.jl")
includet("../true_dro_build_subproblem.jl")
includet("../true_dro_build_isp_leader.jl")
includet("../true_dro_build_isp_follower.jl")
includet("../true_dro_benders.jl")
includet("../true_dro_mincut_vi.jl")
includet("../oos_dirichlet.jl")

include("config.jl")


# ============================================================
# evaluate_f, solve_dro (run_experiment2_calibration.jl에서 가져옴)
# ============================================================

function build_evaluate_f_model(td::TrueDROData;
        optimizer=Gurobi.Optimizer,
        nonconvex_attr=("NonConvex" => 2))
    K = td.num_arcs
    x_dummy = zeros(K)
    if td.eps_hat == 0.0 && td.eps_tilde == 0.0
        model, vars = build_true_dro_subproblem_nominal(td, x_dummy;
            optimizer=optimizer_with_attributes(optimizer, MOI.Silent() => true),
            rho_upper_bound=10.0)
        return model, vars, :nominal
    elseif td.eps_tilde == 0.0
        model, vars = build_true_dro_subproblem_single(td, x_dummy;
            optimizer=optimizer_with_attributes(optimizer, MOI.Silent() => true,
                                                nonconvex_attr),
            rho_upper_bound=10.0)
        return model, vars, :single
    else
        model, vars = build_true_dro_subproblem(td, x_dummy;
            optimizer=optimizer_with_attributes(optimizer, MOI.Silent() => true,
                                                nonconvex_attr),
            rho_upper_bound=10.0)
        return model, vars, :full
    end
end

function evaluate_f(model, vars, td::TrueDROData, x::Vector{Float64};
        sub_time_limit::Float64=3600.0, mip_gap::Float64=0.005)
    set_optimizer_attribute(model, "TimeLimit", sub_time_limit)
    set_optimizer_attribute(model, "MIPGap", mip_gap)
    sub_info = solve_true_dro_subproblem!(model, vars, td, x)
    if !sub_info[:is_optimal]
        @printf("  ⚠ evaluate_f: TIME_LIMIT, Z0_val=%.6f, gap=%.2e\n",
                sub_info[:Z0_val],
                abs(sub_info[:Z0_val] - sub_info[:Z0_bound]) / max(abs(sub_info[:Z0_val]), 1e-10))
    end
    return sub_info[:Z0_val]
end

function solve_dro(td::TrueDROData;
        sub_time_limit::Float64=30.0, max_iter::Int=500, verbose::Bool=true)
    result = true_dro_benders_optimize!(td;
        mip_optimizer = Gurobi.Optimizer,
        nlp_optimizer = Gurobi.Optimizer,
        lp_optimizer  = Gurobi.Optimizer,
        max_iter = max_iter, tol = 1e-4, verbose = verbose,
        sub_time_limit = sub_time_limit,
        mini_benders = true, max_mini_benders_iter = 5,
        strengthen_cuts = :mw, valid_inequality = :mincut,
        inexact = true, nonconvex_attr = ("NonConvex" => 2))
    return Dict(:x => result[:x], :α => result[:α], :Z0 => result[:Z0],
                :status => result[:status], :iters => result[:iters],
                :wall_time => result[:wall_time])
end


# ============================================================
# Two-phase bisection
# ============================================================

function find_epsilon_range_f3(network, capacities, q_hat, w, γ,
        x_neut::Vector{Float64}, x_rob::Vector{Float64}, net_key::Symbol;
        eps_max::Float64 = 1.0,
        lambda_U::Float64 = F3_LAMBDA_U,
        tol::Float64 = 0.01,
        max_iter_phase1::Int = 15,
        max_iter_phase2::Int = 10,
        benders_sub_time_limit::Float64 = 30.0,
        benders_max_iter::Int = 500,
        verbose::Bool = true)

    S = size(capacities, 2)

    # Baseline models
    td_0 = make_true_dro_data(network, capacities, q_hat, 0.0, 0.0;
                               w=w, lambda_U=lambda_U, gamma=γ)
    td_1 = make_true_dro_data(network, capacities, q_hat, eps_max, eps_max;
                               w=w, lambda_U=lambda_U, gamma=γ)

    model_0, vars_0, _ = build_evaluate_f_model(td_0)
    f0_neut = evaluate_f(model_0, vars_0, td_0, x_neut)

    model_1, vars_1, _ = build_evaluate_f_model(td_1)
    f1_rob = evaluate_f(model_1, vars_1, td_1, x_rob)

    if verbose
        @printf("Baselines: f_0(x_neut)=%.6f, f_1(x_rob)=%.6f\n", f0_neut, f1_rob)
    end

    history = Vector{Dict}()
    dro_solves = Dict{Float64, Dict}()

    # Checkpoint
    checkpoint_dir = joinpath(@__DIR__, "results")
    checkpoint_path = joinpath(checkpoint_dir, "checkpoint_$(net_key).jls")

    function _save_checkpoint(phase, lo, hi, ε)
        serialize(checkpoint_path, Dict(
            :net_key => net_key, :phase => phase,
            :lo => lo, :hi => hi, :next_eps => ε,
            :history => history, :dro_solves => dro_solves,
        ))
    end

    # ================================================================
    # Phase 1: PO-PP bisection → ε^S
    # ================================================================
    if verbose
        println("\n" * "=" ^ 70)
        println("Phase 1: Find ε^S via PO-PP bisection (no DRO solves)")
        println("=" ^ 70)
    end

    lo1 = 0.0; hi1 = eps_max; ε1 = (lo1 + hi1) / 2.0

    for iter in 1:max_iter_phase1
        if verbose
            @printf("\n--- P1 iter %d: ε=%.6f [%.6f, %.6f] ---\n", iter, ε1, lo1, hi1)
        end

        td_ε = make_true_dro_data(network, capacities, q_hat, ε1, ε1;
                                   w=w, lambda_U=lambda_U, gamma=γ)
        model_ε, vars_ε, _ = build_evaluate_f_model(td_ε)

        f_ε_neut = evaluate_f(model_ε, vars_ε, td_ε, x_neut)
        f_ε_rob  = evaluate_f(model_ε, vars_ε, td_ε, x_rob)
        po_pp = f_ε_neut - f_ε_rob

        region = po_pp < 0 ? :below_eps_S : :above_eps_S
        push!(history, Dict(:phase => 1, :iter => iter, :eps => ε1,
                            :lo => lo1, :hi => hi1, :po_pp => po_pp,
                            :nr => NaN, :wr => NaN, :nr_wr => NaN,
                            :region => region))

        if verbose
            @printf("  PO-PP=%.6f → %s\n", po_pp, region)
        end

        if po_pp < 0
            lo1 = ε1; ε1 = (ε1 + hi1) / 2.0
        else
            hi1 = ε1; ε1 = (lo1 + ε1) / 2.0
        end

        _save_checkpoint(1, lo1, hi1, ε1)

        if (hi1 - lo1) < tol
            if verbose; @printf("  Phase 1 converged: ε^S ∈ [%.6f, %.6f]\n", lo1, hi1); end
            break
        end
    end

    eps_S_lo = lo1; eps_S_hi = hi1; eps_S_mid = (lo1 + hi1) / 2.0
    if verbose; @printf("\nε^S ≈ %.6f  [%.6f, %.6f]\n", eps_S_mid, eps_S_lo, eps_S_hi); end

    # ================================================================
    # Phase 2: NR-WR bisection → ε^D
    # ================================================================
    if verbose
        println("\n" * "=" ^ 70)
        println("Phase 2: Find ε^D via NR-WR bisection (DRO solves)")
        println("=" ^ 70)
    end

    lo2 = eps_S_hi; hi2 = eps_max; ε2 = (lo2 + hi2) / 2.0
    found_region = :unknown
    eps_recommended = eps_S_mid

    # Pre-check: ε=eps_max에서 NR-WR
    f0_rob = evaluate_f(model_0, vars_0, td_0, x_rob)
    nr_max = f0_rob - f0_neut
    wr_max = 0.0  # f1(x_rob) - f1(x_rob) = 0
    nr_wr_max = nr_max - wr_max

    if verbose
        @printf("  Pre-check at ε=%.4f: NR=%.6f, NR-WR=%.6f\n", eps_max, nr_max, nr_wr_max)
    end

    if nr_wr_max < 0
        found_region = :in_range
        eps_recommended = eps_S_hi
        if verbose; @printf("  ε^D ≥ %.4f → eps_recommended = %.6f\n", eps_max, eps_recommended); end
    else
        for iter in 1:max_iter_phase2
            if verbose
                @printf("\n--- P2 iter %d: ε=%.6f [%.6f, %.6f] ---\n", iter, ε2, lo2, hi2)
            end

            td_ε = make_true_dro_data(network, capacities, q_hat, ε2, ε2;
                                       w=w, lambda_U=lambda_U, gamma=γ)

            # DRO solve
            result = f3_tee_solve(net_key, "calib_p2_eps$(round(ε2;digits=4))") do
                solve_dro(td_ε; sub_time_limit=benders_sub_time_limit,
                          max_iter=benders_max_iter, verbose=verbose)
            end
            x_star = result[:x]
            dro_solves[ε2] = result

            f0_star = evaluate_f(model_0, vars_0, td_0, x_star)
            f1_star = evaluate_f(model_1, vars_1, td_1, x_star)
            nr = f0_star - f0_neut
            wr = f1_star - f1_rob
            nr_wr = nr - wr

            # PO-PP 참고
            model_ε2, vars_ε2, _ = build_evaluate_f_model(td_ε)
            f_ε_neut = evaluate_f(model_ε2, vars_ε2, td_ε, x_neut)
            f_ε_rob  = evaluate_f(model_ε2, vars_ε2, td_ε, x_rob)
            po_pp = f_ε_neut - f_ε_rob

            region = nr_wr < 0 ? :in_range : :above_eps_D
            push!(history, Dict(:phase => 2, :iter => iter, :eps => ε2,
                                :lo => lo2, :hi => hi2, :po_pp => po_pp,
                                :nr => nr, :wr => wr, :nr_wr => nr_wr,
                                :f0_star => f0_star, :f1_star => f1_star,
                                :x_star => round.(Int, x_star), :region => region))

            if verbose
                @printf("  NR=%.6f, WR=%.6f, NR-WR=%.6f → %s\n", nr, wr, nr_wr, region)
            end

            if nr_wr < 0
                lo2 = ε2; ε2 = (ε2 + hi2) / 2.0
            else
                hi2 = ε2; ε2 = (lo2 + ε2) / 2.0
            end

            _save_checkpoint(2, lo2, hi2, ε2)

            if (hi2 - lo2) < tol
                found_region = :converged
                eps_recommended = (lo2 + hi2) / 2.0
                if verbose; @printf("  Phase 2 converged: ε^D ∈ [%.6f, %.6f]\n", lo2, hi2); end
                break
            end
        end

        if found_region == :unknown
            found_region = :max_iter_reached
            eps_recommended = (lo2 + hi2) / 2.0
        end
    end

    eps_D_lo = lo2; eps_D_hi = hi2

    # ---- x*(ε_recommended) ----
    x_star_rec = nothing; Z0_star_rec = NaN
    if !haskey(dro_solves, eps_recommended)
        if verbose; @printf("\n--- Solving x*(ε_rec=%.6f) ---\n", eps_recommended); end
        td_rec = make_true_dro_data(network, capacities, q_hat, eps_recommended, eps_recommended;
                                     w=w, lambda_U=lambda_U, gamma=γ)
        res_rec = f3_tee_solve(net_key, "calib_eps_rec") do
            solve_dro(td_rec; sub_time_limit=benders_sub_time_limit,
                       max_iter=benders_max_iter, verbose=verbose)
        end
        dro_solves[eps_recommended] = res_rec
        x_star_rec = res_rec[:x]; Z0_star_rec = res_rec[:Z0]
    else
        x_star_rec = dro_solves[eps_recommended][:x]
        Z0_star_rec = dro_solves[eps_recommended][:Z0]
    end

    # Checkpoint 정리
    isfile(checkpoint_path) && rm(checkpoint_path)

    return Dict(
        :eps_recommended => eps_recommended,
        :eps_S_lo => eps_S_lo, :eps_S_hi => eps_S_hi,
        :eps_D_lo => eps_D_lo, :eps_D_hi => eps_D_hi,
        :region => found_region,
        :history => history, :dro_solves => dro_solves,
        :x_neut => x_neut, :x_rob => x_rob,
        :x_star => x_star_rec, :Z0_star => Z0_star_rec,
        :f0_neut => f0_neut, :f1_rob => f1_rob,
    )
end


# ============================================================
# Main
# ============================================================

# Load nominal & robust results
nom_path = joinpath(@__DIR__, "results", "nominal_results.jls")
rob_path = joinpath(@__DIR__, "results", "robust_results.jls")

if !isfile(nom_path)
    error("Nominal results not found. Run run_nominal_batch.jl first.")
end
if !isfile(rob_path)
    error("Robust results not found. Run run_robust_batch.jl first.")
end

all_nominal = deserialize(nom_path)
all_robust  = deserialize(rob_path)

println("=" ^ 70)
println("Factor k=$(F3_NUM_FACTORS) Experiment: Two-Phase ε Calibration")
println("  Networks: $F3_NETWORKS")
println("  S=$(F3_S), λU=$(F3_LAMBDA_U)")
println("=" ^ 70)

batch_start = time()
all_calibration = Dict{Symbol, Dict}()
summary_rows = []

for net_key in F3_NETWORKS
    println("\n\n" * "#" ^ 70)
    @printf("# %s — ε calibration\n", net_key)
    println("#" ^ 70)
    flush(stdout)

    # x_neut == x_rob인 네트워크는 건너뛰기
    x_neut = all_nominal[net_key][:x]
    x_rob  = all_robust[net_key][:x]
    if round.(Int, x_neut) == round.(Int, x_rob)
        @printf("  x_neut == x_rob → ε^S = ε_max (calibration 불필요)\n")
        @printf("  x_neut = %s\n", string(findall(round.(Int, x_neut) .> 0)))
        all_calibration[net_key] = Dict(
            :eps_recommended => NaN, :region => :same_x,
            :x_neut => x_neut, :x_rob => x_rob,
            :x_star => x_neut, :Z0_star => all_nominal[net_key][:Z0],
            :eps_S_lo => 1.0, :eps_S_hi => 1.0,
            :eps_D_lo => 1.0, :eps_D_hi => 1.0,
            :history => Dict[], :dro_solves => Dict{Float64,Dict}(),
            :f0_neut => NaN, :f1_rob => NaN,
        )
        push!(summary_rows, (network=string(net_key), eps_recommended=NaN,
              eps_S="[1.0,1.0]", eps_D="[1.0,1.0]", region="same_x",
              iters_p1=0, iters_p2=0, n_dro=0, wall_time=0.0))
        continue
    end

    network, capacities, w, γ, _ = f3_regenerate_network(net_key)
    q_hat = fill(1.0 / F3_S, F3_S)

    t0 = time()
    result = find_epsilon_range_f3(network, capacities, q_hat, w, γ,
        x_neut, x_rob, net_key;
        eps_max = 1.0, lambda_U = F3_LAMBDA_U,
        tol = 0.01, max_iter_phase1 = 15, max_iter_phase2 = 10,
        benders_sub_time_limit = 30.0, benders_max_iter = 500,
        verbose = true)
    t_net = time() - t0

    all_calibration[net_key] = result

    # Summary
    n_p1 = count(h -> h[:phase] == 1, result[:history])
    n_p2 = count(h -> h[:phase] == 2, result[:history])
    push!(summary_rows, (
        network = string(net_key),
        eps_recommended = result[:eps_recommended],
        eps_S = @sprintf("[%.4f,%.4f]", result[:eps_S_lo], result[:eps_S_hi]),
        eps_D = @sprintf("[%.4f,%.4f]", result[:eps_D_lo], result[:eps_D_hi]),
        region = string(result[:region]),
        iters_p1 = n_p1, iters_p2 = n_p2,
        n_dro = length(result[:dro_solves]),
        wall_time = t_net,
    ))

    # 즉시 저장
    serialize(joinpath(@__DIR__, "results", "calibration_results.jls"), all_calibration)
    @printf("  Saved (%d networks done), time=%.1fs\n", length(all_calibration), t_net)
    flush(stdout)
end

batch_wall = time() - batch_start

# ---- Final summary ----
println("\n\n" * "=" ^ 120)
println("Factor k=$(F3_NUM_FACTORS) — Calibration Summary")
println("=" ^ 120)
@printf("%-14s  %10s  %16s  %16s  %-12s  %4s  %4s  %4s  %10s\n",
        "Network", "ε_rec", "ε^S", "ε^D", "Region", "P1", "P2", "DRO", "Time(s)")
println("-" ^ 120)
for r in summary_rows
    @printf("%-14s  %10s  %16s  %16s  %-12s  %4d  %4d  %4d  %10.1f\n",
            r.network,
            isnan(r.eps_recommended) ? "N/A" : @sprintf("%.4f", r.eps_recommended),
            r.eps_S, r.eps_D, r.region,
            r.iters_p1, r.iters_p2, r.n_dro, r.wall_time)
end
println("-" ^ 120)
@printf("Total wall time: %.1f s (%.1f min)\n", batch_wall, batch_wall / 60)
println("\nDone! $(now())")
