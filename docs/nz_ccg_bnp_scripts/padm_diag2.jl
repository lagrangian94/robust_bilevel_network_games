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
# --- 닫힌 행 쌍대 최소화 (사전식 두 번째 LP)
tstar = objective_value(P)
closed = [k for k in 1:m if !isempty(C.ents_of_row[k]) && all(x0[e.i] < 0.5 for e in C.ents_of_row[k])]
cc = @constraint(P, tP <= tstar + 1e-7 * max(1.0, abs(tstar)))
@objective(P, Min, sum(bk.pd[k, s] for k in closed, s in 1:S)); optimize!(P)
println("2 단계 status ", termination_status(P), ", 닫힌 행 ", length(closed))
p̂ = value.(bk.pd)
for k in closed[1:min(6, length(closed))]
    @printf("  행 %d: 첫 LP pd 평균 %.2f → 최소 pd 평균 %.2f (U %.2f)\n", k, sum(pd[k, :]) / S, sum(p̂[k, :]) / S, upper_bound(bk.pd[k, 1]))
end
delete(P, cc); @objective(P, Min, tP)
# --- x 단계 (w = 1, π = 0, 노드 고정 없음): 벌점마다 고른 x
nh = nz_nh(nd)
M = Model(() -> (o = Gurobi.Optimizer(envs[1]); MOI.set(o, MOI.Silent(), true); o))
@variable(M, xb[1:nx], Bin); @constraint(M, sum(xb) <= nd.gamma)
for (ix, rhs) in get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[]); @constraint(M, sum(xb[i] for i in ix) <= rhs); end
for i in 1:nx; nd.x_allowed[i] || fix(xb[i], 0.0; force=true); end
h = @variable(M, [i=1:nh], lower_bound = 0.0, upper_bound = nd.hU[i]); @constraint(M, nd.W * h .<= nd.wvec)
y = @variable(M, [1:ny, 1:S], lower_bound = 0.0)
for k in 1:m, s in 1:S
    @constraint(M, sum(a * y[j, s] for (j, a) in C.rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0) - sum(e.g[s] * xb[e.i] for e in C.ents_of_row[k]; init=0.0) <= nd.u[k, s])
end
CF = -(dot(nd.c0, h) + sum(d[s] * nd.c[j] * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0 && d[s] != 0; init=AffExpr(0.0)))
CL = sum(r[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0; init=AffExpr(0.0))
CLp = sum(r[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0 && d[s] > 0; init=AffExpr(0.0))
for (lab, pp) in (("첫 LP 쌍대", pd), ("최소 쌍대", p̂))
    coef = zeros(nx); for e in XE, s in 1:S; coef[e.i] += e.g[s] * pp[e.k, s]; end
    for ρ in (1.0, 10.0, 100.0, 1e3, 1e4)
        @objective(M, Min, CL + ρ * (CF - 1e-4 * CLp) + ρ * sum(coef[i] * xb[i] for i in 1:nx))
        t = @elapsed optimize!(M)
        @printf("  %s ρ=%-7g → x = %-28s (%s, %.2fs)\n", lab, ρ, loc_tkp_xstr(nd, round.(value.(xb))), termination_status(M), t)
    end
end
