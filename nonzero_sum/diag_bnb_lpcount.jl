"""
diag_bnb_lpcount.jl — α-B&B 안의 노드 LP 호출 횟수·벽시계·Gurobi 시간 (barrier 가 따로 풀 때보다 노드당 훨씬 느린 원인).
환경변수: BC_S = 200  BC_TL = 300  BC_VAR = "bar" (bar | base | bard | bardx | based)
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
S = parse(Int, get(ENV, "BC_S", "200")); TL = parse(Float64, get(ENV, "BC_TL", "300")); var = get(ENV, "BC_VAR", "bar")
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=1,
                                           eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
nd = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
x̄ = zeros(nd.nx); x̄[[1, 2]] .= 1
kw = Dict("bar" => (lp_method=2, lp_presolve=-1), "base" => NamedTuple(), "bard" => (lp_method=2, lp_presolve=-1, defer_rows=true),
          "bardx" => (lp_method=2, lp_presolve=-1, defer_rows=true, lp_crossover=0), "based" => (defer_rows=true,))[var]
r = nz_alpha_bnb(nd, x̄; nworkers=12, time_limit=TL, rel_gap=1e-3, verbose=true, log_every=60.0, kw...)
@printf("RESULT %s S=%d: nodes %d, relax(worker 합) %.0fs | LP 호출 %d, LP 벽시계 합 %.0fs, Gurobi solve_time 합 %.0fs, 수치오류 %d (실패 %d)\n",
        var, S, r[:nodes], r[:t_relax], _NZ_LPN[], _NZ_LPWALL[], _NZ_LPGRB[], r[:numerr], r[:numerr_fail])
