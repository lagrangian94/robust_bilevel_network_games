"""
factor_3/run_robust_batch.jl — Factor k=3 실험: 5개 네트워크 robust solve (ε=1.0).

γ: grid_5x5=3, real-world=1. w=max interdictable capacity.
결과: logs/ 에 로그, results/ 에 JLS.

Usage:
    include("true_dro/factor_3/run_robust_batch.jl")
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

include("config.jl")

# ============================================================

println("=" ^ 70)
println("Factor k=$(F3_NUM_FACTORS) Experiment: Robust Batch Solve (ε=1.0)")
println("  Networks: $F3_NETWORKS")
println("  S=$(F3_S), λU=$(F3_LAMBDA_U), seed=$(F3_SEED)")
println("=" ^ 70)

all_robust = Dict{Symbol, Dict}()

for net_key in F3_NETWORKS
    println("\n" * "#" ^ 70)
    @printf("# %s — Robust solve (ε=1.0)\n", net_key)
    println("#" ^ 70)
    flush(stdout)

    network, capacities, w, γ, F_mat = f3_regenerate_network(net_key)
    num_arcs = length(network.arcs) - 1
    q_hat = fill(1.0 / F3_S, F3_S)

    @printf("  arcs=%d, interdictable=%d, γ=%d, w=%.4f, k=%d\n",
            num_arcs, sum(network.interdictable_arcs[1:num_arcs]), γ, w, F3_NUM_FACTORS)

    td = make_true_dro_data(network, capacities, q_hat, 1.0, 1.0;
                             w=w, lambda_U=F3_LAMBDA_U, gamma=γ)

    result = f3_tee_solve(net_key, "robust") do
        t0 = time()
        res = true_dro_benders_optimize!(td;
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

        x_star = res[:x]
        Z0 = res[:Z0]
        x_int = round.(Int, x_star)

        @printf("\nTrue-DRO: status=%s, Z₀=%.6f, iters=%d, time=%.2fs\n",
                res[:status], Z0, res[:iters], wt)
        println("  x* = $x_int")
        println("  interdicted arcs: $(findall(x_int .> 0))")

        Dict(:x => x_star, :Z0 => Z0, :status => res[:status],
             :iters => res[:iters], :wall_time => wt)
    end

    all_robust[net_key] = result

    # 네트워크별 즉시 저장
    save_path = joinpath(@__DIR__, "results", "robust_results.jls")
    serialize(save_path, all_robust)
    @printf("  Results saved (%d networks done)\n", length(all_robust))
    flush(stdout)
end

# ---- Summary ----
println("\n" * "=" ^ 80)
println("Robust Batch Summary (k=$(F3_NUM_FACTORS), ε=1.0)")
println("=" ^ 80)
@printf("%-14s  %3s  %10s  %6s  %10s  %s\n",
        "Network", "γ", "Z₀", "Iters", "Time(s)", "x* arcs")
println("-" ^ 80)
for net_key in F3_NETWORKS
    r = all_robust[net_key]
    x_int = round.(Int, r[:x])
    _, _, _, γ, _ = f3_regenerate_network(net_key)
    @printf("%-14s  %3d  %10.4f  %6d  %10.1f  %s\n",
            net_key, γ, r[:Z0], r[:iters], r[:wall_time],
            string(findall(x_int .> 0)))
end
println("-" ^ 80)

# ---- x_neut vs x_rob 비교 ----
nom_path = joinpath(@__DIR__, "results", "nominal_results.jls")
if isfile(nom_path)
    all_nominal = deserialize(nom_path)
    println("\n--- x_neut vs x_rob comparison ---")
    @printf("%-14s  %-20s  %-20s  %s\n", "Network", "x_neut", "x_rob", "Same?")
    println("-" ^ 70)
    for net_key in F3_NETWORKS
        haskey(all_nominal, net_key) || continue
        x_neut = round.(Int, all_nominal[net_key][:x])
        x_rob = round.(Int, all_robust[net_key][:x])
        same = x_neut == x_rob
        @printf("%-14s  %-20s  %-20s  %s\n",
                net_key,
                string(findall(x_neut .> 0)),
                string(findall(x_rob .> 0)),
                same ? "SAME" : "DIFF")
    end
end

println("\nDone! $(now())")
