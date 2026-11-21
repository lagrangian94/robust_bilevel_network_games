# 조인 용량 쌍대 상한 U′_i = max_j (max_A c_Aj − c_ij)⁺ (loc_tkp.jl 머리말 논증) 이 정수 x 에서 강쌍대를 지키는지 검사 (2026-11-21).
#   follower 섭동 최대화 문제 −(C_F − ε C_L) 의 primal 값 vs 용량 쌍대에 상한 d_s U′ + ε r_s max|ℓ| 를 건 쌍대 최소값.
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random
const GRB_ENV = Gurobi.Env()
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "loc_tkp.jl"))
envf(k, d) = get(ENV, k, d)
nd = make_loc_tkp(; g=parse(Int, envf("GEO", "1")), S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
MM = nd.meta; nA, nB, K, S, ncu = MM[:nA], MM[:nB], MM[:K], nd.S, MM[:ncu]
m, ny, nh = nz_m(nd), nz_ny(nd), nz_nh(nd); A = nd.A; XE = nz_xentries(nd); nW = size(nd.W, 1)
rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m]; colA = [[(k, A[k, j]) for k in 1:m if A[k, j] != 0] for j in 1:ny]
dist = MM[:dist]; println("dist 크기 ", size(dist), "  (점포 ", nA + nB, ", 고객 ", ncu, ")")
cA = [maximum(dist[a, j] for a in 1:nA) for j in 1:ncu]
UF = zeros(m); for st in 1:nB; i = nA + st; UF[MM[:rcap](i)] = maximum(max(cA[j] - dist[i, j], 0.0) for j in 1:ncu); end
xr = unique(e.k for e in XE)
@printf("상한 비교 (점포별): 기존 piFU %s\n                     조인 U′  %s\n", string(round.(Int, nd.piFU[xr])), string(round.(Int, UF[xr])))
EPSP = parse(Float64, envf("EPSP", "0.1"))
bels = Any[(nd.q_hat, nd.q_hat)]
dd = [0.12, 0.1, 0.153, 0.163, 0.037, 0.13, 0.134, 0.037, 0.04, 0.087]; rr = [0.0, 0.0, 0.255, 0.272, 0.0, 0.217, 0.0, 0.0, 0.0, 0.256]
push!(bels, (dd ./ sum(dd), rr ./ sum(rr)))
rng = MersenneTwister(7)
xs = Vector{Vector{Float64}}()
for t in 1:parse(Int, envf("NX", "150"))
    xv = zeros(nd.nx); n = rand(rng, 1:3); for st in randperm(rng, nB)[1:n]; xv[(st - 1) * K + rand(rng, 1:K)] = 1.0; end; push!(xs, xv)
end
worst = 0.0; nbad = 0; ntot = 0
for (bi, (d, r)) in enumerate(bels)
    cε(j, s) = d[s] * nd.c[j] + EPSP * r[s] * nd.ell[j]
    for xv in xs
        gx(k, s) = sum(e.g[s] * xv[e.i] for e in XE if e.k == k; init=0.0)
        P = Model(GRB1)
        @variable(P, 0 <= h[i=1:nh] <= nd.hU[i]); @constraint(P, nd.W * h .<= nd.wvec); @variable(P, y[1:ny, 1:S] >= 0)
        @constraint(P, [k=1:m, s=1:S], sum(a * y[j, s] for (j, a) in rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0) <= nd.u[k, s] + gx(k, s))
        @objective(P, Max, dot(nd.c0, h) + sum(cε(j, s) * y[j, s] for j in 1:ny, s in 1:S if cε(j, s) != 0)); optimize!(P); vp = objective_value(P)
        Dm = Model(GRB1)
        @variable(Dm, pd[1:m, 1:S] >= 0); @variable(Dm, kp[1:nW] >= 0); @variable(Dm, om[1:nh] >= 0)
        @constraint(Dm, [j=1:ny, s=1:S], sum(a * pd[k, s] for (k, a) in colA[j]) >= cε(j, s))
        @constraint(Dm, [i=1:nh], -sum(nd.hcoef[k] * pd[k, s] for k in 1:m, s in 1:S if nd.hrow[k] == i) + sum(nd.W[l, i] * kp[l] for l in 1:nW) + om[i] >= nd.c0[i])
        for k in xr, s in 1:S; set_upper_bound(pd[k, s], d[s] * UF[k] + EPSP * r[s] * maximum(abs, nd.ell)); end
        @objective(Dm, Min, sum((nd.u[k, s] + gx(k, s)) * pd[k, s] for k in 1:m, s in 1:S) + dot(nd.wvec, kp) + dot(nd.hU, om)); optimize!(Dm)
        vd = termination_status(Dm) == MOI.OPTIMAL ? objective_value(Dm) : Inf
        rel = (vd - vp) / max(1.0, abs(vp)); global ntot += 1
        rel > 1e-7 && (global nbad += 1); global worst = max(worst, rel)
    end
    @printf("belief %d 끝: 누적 %d 건 중 쌍대 > 원 (1e-7 넘음) %d, 최대 상대차 %.2e\n", bi, ntot, nbad, worst); flush(stdout)
end
