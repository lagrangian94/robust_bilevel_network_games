"""
true_dro_run.jl — True-DRO-Exact entry script.

기본 검증: 작은 그리드에서 Benders 수렴 확인.
Full model이 없으므로 v1 검증은 다음으로 진행:
  1. Benders 수렴 (LB ≈ UB)
  2. (선택) ε̂=ε̃=0 일 때 nominal 2SP 값과 비교
  3. (선택) TV_DRO (V^Dir)와 sandwich: V*(true) ≤ V^Dir
"""

using Revise
using JuMP
using Gurobi
using Printf
using LinearAlgebra

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
includet("true_dro_recover.jl")


"""
    run_true_dro(; m=2, n=2, S=2, seed=42,
                 eps_hat=0.1, eps_tilde=0.1,
                 gamma=2, w=1.0, lambda_U=10.0,
                 max_iter=30, tol=1e-4, verbose=true,
                 sub_verbose=false,
                 mini_benders=false, lp_optimizer=nothing,
                 sub_time_limit=nothing)

Build True-DRO instance and run outer Benders.
`sub_verbose=true` 로 Gurobi NonConvex subproblem 로그 출력.
`mini_benders=true` 로 §9.4 mini-Benders 추가 cut 생성 (lp_optimizer 필요).
`sub_time_limit` — bilinear subproblem time limit (sec). nothing=무제한.
"""
function run_true_dro(; m=2, n=2, S=2, seed=42,
                       eps_hat=0.1, eps_tilde=0.1,
                       gamma=2, w=1.0, lambda_U=10.0,
                       max_iter=30, tol=1e-4, verbose=true,
                       sub_verbose=true,
                       mini_benders=false, lp_optimizer=nothing,
                       sub_time_limit=nothing)
    println("=" ^ 60)
    println("True-DRO-Exact: $(m)×$(n) grid, S=$S, ε̂=$eps_hat, ε̃=$eps_tilde")
    println("=" ^ 60)

    network = generate_grid_network(m, n; seed=seed)
    num_arcs = length(network.arcs) - 1
    println("Network: $(network.m)×$(network.n), |A|=$num_arcs")

    num_arcs_with_dummy = length(network.arcs)
    scenarios, _ = generate_capacity_scenarios_uniform_model(num_arcs_with_dummy, S; seed=seed)

    q_hat = fill(1.0 / S, S)

    td = make_true_dro_data(network, scenarios, q_hat, eps_hat, eps_tilde;
                            w=w, lambda_U=lambda_U, gamma=gamma)

    println("\n--- Outer Benders (OMP ↔ bilinear subproblem via Gurobi NonConvex=2) ---")
    result = true_dro_benders_optimize!(td;
        mip_optimizer=Gurobi.Optimizer,
        nlp_optimizer=Gurobi.Optimizer,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        sub_verbose=sub_verbose,
        sub_time_limit=sub_time_limit,
        mini_benders=mini_benders,
        lp_optimizer=lp_optimizer)

    @printf("\nResult: status=%s, Z₀=%.6f, iters=%d\n",
            result[:status], result[:Z0], result[:iters])
    @printf("LB=%.6f, UB=%.6f, gap=%.2e\n",
            result[:lower_bound], result[:upper_bound],
            abs(result[:upper_bound] - result[:lower_bound]) /
                max(abs(result[:upper_bound]), 1e-10))
    x_int = round.(Int, result[:x])
    α_str = join([@sprintf("%.3f", a) for a in result[:α]], ",")
    @printf("x* = %s\nα* = [%s]\n", string(x_int), α_str)

    return result
end


# ============================================================
# Run if executed directly
# ============================================================
if abspath(PROGRAM_FILE) == @__FILE__
    run_true_dro()
end
