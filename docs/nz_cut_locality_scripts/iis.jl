# combinatorial Benders cut 이 비국소인가: belief 블록 LP 에서 x 를 x̂ 로 고정하고 "q·x + t0 ≤ UB − ε" 를 넣어 실행불가로 만든 뒤,
# 최소 충돌 집합 (IIS) 에 x 고정 조건이 몇 개 들어가는지 본다. 일부만 들어가면 그 부분 배치를 공유하는 모든 x 를 자르는 cut 이 된다.
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random
const GRB_ENV = Gurobi.Env()
include(joinpath(root, "nonzero_sum", "loc_tkp.jl"))
nd = make_loc_tkp(; g=1, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
K, nB, S = nd.meta[:K], nd.meta[:nB], nd.S; qh = nd.q_hat
m, ny, nh, nx = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.nx; nW = size(nd.W, 1); A = nd.A; XE = nz_xentries(nd)
colA = [[(k, A[k, j]) for k in 1:m if A[k, j] != 0] for j in 1:ny]; rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m]
ents_of_row = [[e for e in XE if e.k == k] for k in 1:m]
lam = 10.0; UBv = parse(Float64, get(ENV, "UB", "-1627.6")); d = qh; r = qh
"x 를 연속 변수로 두고 등식으로 고정한 belief 블록 LP (bccg.jl 의 비관적 블록과 같음)"
function build(xh; cardfree::Bool=false)
    M = Model(() -> (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); o))
    @variable(M, 0 <= x[1:nx] <= 1); @variable(M, t0 >= -1e5)
    fx = @constraint(M, [i=1:nx], x[i] == xh[i])
    h = @variable(M, [i=1:nh], lower_bound = 0.0, upper_bound = nd.hU[i]); @constraint(M, nd.W * h .<= nd.wvec)
    y = @variable(M, [1:ny, 1:S], lower_bound = 0.0)
    @constraint(M, [k=1:m, s=1:S], sum(a * y[j, s] for (j, a) in rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0)
                                   - sum(e.g[s] * x[e.i] for e in ents_of_row[k]; init=0.0) <= nd.u[k, s])
    mu = @variable(M, [1:m, 1:S], lower_bound = 0.0); kp = @variable(M, [1:nW], lower_bound = 0.0); om = @variable(M, [1:nh], lower_bound = 0.0)
    @constraint(M, [j=1:ny, s=1:S], sum(a * mu[k, s] for (k, a) in colA[j]) >= r[s] * nd.ell[j] + lam * d[s] * nd.c[j])
    @constraint(M, [i=1:nh], -sum(nd.hcoef[k] * mu[k, s] for k in 1:m, s in 1:S if nd.hrow[k] == i) + sum(nd.W[l, i] * kp[l] for l in 1:nW) + om[i] >= lam * nd.c0[i])
    psi = @variable(M, [1:length(XE), 1:S], lower_bound = 0.0)
    for (ei, e) in enumerate(XE), s in 1:S
        U = lam * d[s] * nd.piFU[e.k] + 2.0 * r[s] * maximum(abs, nd.ell)
        @constraint(M, psi[ei, s] >= mu[e.k, s] - U * (1 - x[e.i]))
    end
    for k in unique(e.k for e in XE), s in 1:S; set_upper_bound(mu[k, s], lam * d[s] * nd.piFU[k] + 2.0 * r[s] * maximum(abs, nd.ell)); end
    @constraint(M, t0 >= sum(nd.u[k, s] * mu[k, s] for k in 1:m, s in 1:S if nd.u[k, s] != 0)
                         + sum(e.g[s] * psi[ei, s] for (ei, e) in enumerate(XE), s in 1:S) + dot(nd.wvec, kp) + dot(nd.hU, om)
                         - lam * (dot(nd.c0, h) + sum(d[s] * nd.c[j] * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0)))
    return M, x, t0, fx
end
xof(sel) = (x = zeros(nx); for (st, k) in sel; x[(st - 1) * K + k] = 1; end; x)
rng = MersenneTwister(4)
tests = Vector{Vector{Tuple{Int,Int}}}()
for t in 1:10; st = sort(randperm(rng, nB)[1:3]); push!(tests, [(s, rand(rng, 1:K)) for s in st]); end
for t in 1:4; st = sort(randperm(rng, nB)[1:2]); push!(tests, [(s, rand(rng, 1:K)) for s in st]); end
push!(tests, [(rand(rng, 1:nB), 2)]); push!(tests, [(rand(rng, 1:nB), 3)])
for sel in tests
    xh = xof(sel); M, x, t0, fx = build(xh)
    @objective(M, Min, dot(nd.qx, x) + t0); optimize!(M); fval = objective_value(M)
    if fval <= UBv + 1.0
        @printf("x=%-26s 하한 %.1f ≤ UB: 이 belief 로는 못 쳐냄 (건너뜀)\n", string(sel), fval); continue
    end
    @constraint(M, dot(nd.qx, x) + t0 <= UBv - 1.0)
    @objective(M, Min, 0.0); optimize!(M)
    compute_conflict!(M)
    inC = [i for i in 1:nx if MOI.get(M, MOI.ConstraintConflictStatus(), fx[i]) == MOI.IN_CONFLICT]
    open_in = [i for i in inC if xh[i] > 0.5]; closed_sites = unique((i - 1) ÷ K + 1 for i in inC if xh[i] < 0.5)
    all_closed = nB - length(sel)
    @printf("x=%-26s 하한 %.1f | 충돌 집합의 x 조건 %2d 개: 연 거점 조건 %d / %d, 닫힌 거점 %2d / %d 곳\n", string(sel), fval, length(inC), length(open_in), length(sel), length(closed_sites), all_closed)
    flush(stdout)
end
