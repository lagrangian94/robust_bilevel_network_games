# PADM 진단: x 고정 블록 LP 의 쌍대 pd 가 닫힌 거점 행에서 어떤 값인지 (2026-11-26 밤, 읽기 전용 진단)
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_belief_ccg.jl")); include(joinpath(root, "nonzero_sum", "nz_belief_ccg_tree.jl"))
nd = make_loc_tkp(; g=13, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2 / 3, beta=0.4, lambdaU=1e4, NB=3)
K = nd.meta[:K]; nx = nd.nx; m, ny = nz_m(nd), nz_ny(nd); A = nd.A; XE = nz_xentries(nd); S = nd.S
envs = _nz_envs(1)
lp_opt = () -> (o = Gurobi.Optimizer(envs[1]); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
C = (nd = nd, eta = 1e-4, vub = false, wd = false, XE = XE, lp_opt = lp_opt,
     rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m], colA = [[(k, A[k, j]) for k in 1:m if A[k, j] != 0] for j in 1:ny],
     ents_of_row = [[e for e in XE if e.k == k] for k in 1:m])
d = copy(nd.q_hat); r = copy(nd.q_hat)
P = Model(lp_opt); xv = @variable(P, [1:nx]); tP = @variable(P); fixc = @constraint(P, [i=1:nx], xv[i] == 0.0)
bk = _bccg_add_block!(merge(C, (M = P, x = xv, t0 = tP)), d, r, 1e4); @objective(P, Min, tP)
x0 = zeros(nx); x0[5] = 1; x0[21] = 1; println("x0 = ", loc_tkp_xstr(nd, x0))
for i in 1:nx; set_normalized_rhs(fixc[i], x0[i]); end
optimize!(P); println("status ", termination_status(P), " L = ", objective_value(P))
pd = value.(bk.pd)
@printf("x 항목 %d 개, 행 수 m = %d, S = %d\n", length(XE), m, S)
for e in XE
    U = [upper_bound(bk.pd[e.k, s]) for s in 1:S]
    @printf("  x[%2d] (거점 %2d 단계 %d) = %d  행 %3d  u 평균 %8.2f  g 평균 %8.2f  pd 평균 %10.3f  U 평균 %10.3f  pd/U %.2f\n",
            e.i, (e.i - 1) ÷ K + 1, (e.i - 1) % K + 1, Int(x0[e.i]), e.k, sum(nd.u[e.k, :]) / S, sum(e.g) / S, sum(pd[e.k, :]) / S, sum(U) / S, sum(pd[e.k, :]) / max(1e-12, sum(U)))
end
