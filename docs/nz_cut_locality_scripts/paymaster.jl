# 지불 상계 마스터: min q·x + t0,  t0 ≥ (V_F⁰(x) − V_F(x)) / κ   (κ = 120, 비용 부호)
#   V_F⁰ (리더 거점 지불 0 인 follower 최소 비용): 원문제 사본 (h0, y0), x 는 우변. 마스터가 최소화하므로 정확
#   −V_F (= follower 최대화 값 V_F^max): 쌍대 (π, κ, ω), x·π 는 McCormick (ψ ≥ π − U(1−x)). 마스터가 최소화하므로 약쌍대로 유효
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Statistics
const GRB_ENV = Gurobi.Env()
include(joinpath(root, "nonzero_sum", "loc_tkp.jl"))
envf(k, d) = get(ENV, k, d)
nd = make_loc_tkp(; g=parse(Int, envf("GEO", "1")), S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
MM = nd.meta; nA, ncu, nB, K, S = MM[:nA], MM[:ncu], MM[:nB], MM[:K], nd.S; tiers = MM[:tiers]
m, ny, nh, nx = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.nx; nW = size(nd.W, 1); A = nd.A; XE = nz_xentries(nd); d = nd.q_hat
rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m]; colA = [[(k, A[k, j]) for k in 1:m if A[k, j] != 0] for j in 1:ny]
hrow_of = [[k for k in 1:m if nd.hrow[k] == i] for i in 1:nh]
κ = 120.0
c0F = copy(nd.c0); cF = copy(nd.c)
for st in 1:nB
    ib = nA + st; c0F[MM[:hidx](ib, 1)] = 0.0
    for j in 1:ncu; cF[MM[:iyS](ib, j)] = nd.c[MM[:iyR](ib, j)]; end
end
REL = envf("REL", "0") == "1"
M = Model(() -> (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 8); o))
set_optimizer_attribute(M, "OutputFlag", 1); set_optimizer_attribute(M, "LogToConsole", 0); set_optimizer_attribute(M, "LogFile", joinpath(@__DIR__, "paymaster_grb.log"))
set_optimizer_attribute(M, "MIPGap", 1e-5); set_optimizer_attribute(M, "TimeLimit", parse(Float64, envf("TL", "1800")))
REL ? @variable(M, 0 <= x[1:nx] <= 1) : @variable(M, x[1:nx], Bin)
@variable(M, t0)
@constraint(M, sum(x) <= nd.gamma)
for (idx, rhs) in get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[]); @constraint(M, sum(x[i] for i in idx) <= rhs); end
capterm(k, s) = sum(e.g[s] * x[e.i] for e in XE if e.k == k; init=AffExpr(0.0))
# V_F⁰: 원문제 사본
@variable(M, 0 <= h0[i=1:nh] <= nd.hU[i]); @constraint(M, nd.W * h0 .<= nd.wvec); @variable(M, y0[1:ny, 1:S] >= 0)
@constraint(M, [k=1:m, s=1:S], sum(a * y0[j, s] for (j, a) in rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h0[nd.hrow[k]] : 0.0) - capterm(k, s) <= nd.u[k, s])
if envf("VUB", "1") == "1"                                     # 호 단위 제약 (V_F⁰ 사본에만: 정수 x 에서 중복)
    for st in 1:nB, j in 1:ncu, s in 1:S
        @constraint(M, y0[MM[:iyR](nA + st, j), s] + y0[MM[:iyS](nA + st, j), s] <= -nd.u[MM[:rdem](j), s] * sum(x[(st - 1) * K + k] for k in 1:K))
    end
end
obj0 = @expression(M, dot(c0F, h0) + sum(d[s] * cF[j] * y0[j, s] for j in 1:ny, s in 1:S if cF[j] != 0))     # 최대화 값 (= −비용)
# V_F^max 의 쌍대 (최소화)
@variable(M, π[1:m, 1:S] >= 0); @variable(M, κv[1:nW] >= 0); @variable(M, ω[1:nh] >= 0)
@constraint(M, [j=1:ny, s=1:S], sum(a * π[k, s] for (k, a) in colA[j]) >= d[s] * nd.c[j])
@constraint(M, [i=1:nh], -sum(nd.hcoef[k] * π[k, s] for k in hrow_of[i], s in 1:S) + sum(nd.W[l, i] * κv[l] for l in 1:nW) + ω[i] >= nd.c0[i])
dualobj = @expression(M, sum(nd.u[k, s] * π[k, s] for k in 1:m, s in 1:S if nd.u[k, s] != 0) + dot(nd.wvec, κv) + dot(nd.hU, ω))
for e in XE, s in 1:S
    U = d[s] * nd.piFU[e.k]; set_upper_bound(π[e.k, s], U)
    ψ = @variable(M, lower_bound = 0.0); @constraint(M, ψ >= π[e.k, s] - U * (1 - x[e.i])); add_to_expression!(dualobj, e.g[s], ψ)
end
# t0 ≥ (V_F⁰ − V_F)/κ  =  (−obj0 + V_F^max)/κ,  V_F^max ≤ dualobj
@constraint(M, t0 >= (-obj0 + dualobj) / κ)
@objective(M, Min, dot(nd.qx, x) + t0)
t = @elapsed optimize!(M)
@printf("PAY REL=%s VUB=%s: 상태 %s, 목적 %.2f, 하한 %.2f, 노드 %d, %.1fs  (참: 정수점 상계 최소 −2482.81, 실제 최적 −2095.27)\n", REL, envf("VUB", "1"),
        termination_status(M), objective_value(M), objective_bound(M), REL ? 0 : Int(MOI.get(M, MOI.NodeCount())), t)
x̄ = value.(x); println("   x: ", [(round(x̄[i]; digits=2), i) for i in 1:nx if x̄[i] > 1e-6])
REL || println("   루트: ", join(filter(l -> occursin("Root relaxation", l), readlines(joinpath(@__DIR__, "paymaster_grb.log"))), " | "))
