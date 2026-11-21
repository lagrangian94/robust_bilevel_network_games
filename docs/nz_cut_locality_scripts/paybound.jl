# 리더 수익 상계  Rev ≤ (V_F(∅) − V_F⁰(x)) / 120  의 조임: V_F⁰ = 리더 거점 지불이 0 일 때 follower 최소 비용 (약정비·현장 할증 0)
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Statistics
const GRB_ENV = Gurobi.Env()
include(joinpath(root, "nonzero_sum", "loc_tkp.jl"))
nd = make_loc_tkp(; g=1, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
MM = nd.meta; nA, ncu, nB, K, S = MM[:nA], MM[:ncu], MM[:nB], MM[:K], nd.S
m, ny, nh = nz_m(nd), nz_ny(nd), nz_nh(nd); A = nd.A; XE = nz_xentries(nd); d = nd.q_hat; p = MM[:p]
rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m]
# 비용 (최소화 부호): follower 목적은 max c0ᵀh + Σ d cᵀy  → 비용 = −(…)
c0F = copy(nd.c0); cF = copy(nd.c)
for st in 1:nB
    ib = nA + st; c0F[MM[:hidx](ib, 1)] = 0.0                        # 리더 거점 약정비 0
    for j in 1:ncu; cF[MM[:iyS](ib, j)] = nd.c[MM[:iyR](ib, j)]; end  # 리더 거점 현장 할증 0
end
F = Model(() -> (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); o))
@variable(F, 0 <= h[i=1:nh] <= nd.hU[i]); @constraint(F, nd.W * h .<= nd.wvec); @variable(F, y[1:ny, 1:S] >= 0)
@constraint(F, con[k=1:m, s=1:S], sum(a * y[j, s] for (j, a) in rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0) <= nd.u[k, s])
xrows = unique(e.k for e in XE)
function setx!(lev)
    for k in xrows, s in 1:S; set_normalized_rhs(con[k, s], nd.u[k, s]); end
    for (st, l) in enumerate(lev); l == 0 && continue; k = MM[:rcap](nA + st); for s in 1:S; set_normalized_rhs(con[k, s], nd.u[k, s] + MM[:tiers][l]); end; end
end
val(c0v, cv) = (@objective(F, Max, dot(c0v, h) + sum(d[s] * cv[j] * y[j, s] for j in 1:ny, s in 1:S if cv[j] != 0)); optimize!(F); -objective_value(F))
rows = []
for line in eachline(joinpath(@__DIR__, "vtable_g1_nominal.tsv"))
    startswith(line, "set") && continue
    p_ = split(line, '\t'); push!(rows, (p_[1], parse(Float64, p_[2]), parse(Float64, p_[3]), -parse(Float64, p_[4])))
end
setx!(zeros(Int, nB)); VF∅ = val(nd.c0, nd.c)
@printf("V_F(∅) = %.1f (비용)\n", VF∅)
ratio = Float64[]; viol = 0; out = Tuple{String,Float64,Float64}[]
t = @elapsed for (sel, q, V, VFcost) in rows
    isempty(sel) && continue
    lev = zeros(Int, nB); for tk in split(sel, ';'); a, b = parse.(Int, split(tk, ':')); lev[a] = b; end
    setx!(lev); V0 = val(c0F, cF)
    rev = -V; bnd = (VF∅ - V0) / 120
    bnd < rev - 1e-6 * max(1.0, rev) && (global viol += 1)
    push!(ratio, bnd / rev); push!(out, (sel, rev, bnd))
end
@printf("x %d 개 (%.0fs): 상계 위반 %d | 상계/실제 수익: 중앙 %.2f, 10%% %.2f, 90%% %.2f, 최소 %.2f, 최대 %.2f\n", length(ratio), t, viol,
        median(ratio), quantile(ratio, 0.1), quantile(ratio, 0.9), minimum(ratio), maximum(ratio))
for n in 1:3
    rr = [b / r for (s, r, b) in out if count(==(';'), s) == n - 1]
    @printf("  %d 곳: 상계/실제 중앙 %.2f (10%% %.2f, 90%% %.2f)\n", n, median(rr), quantile(rr, 0.1), quantile(rr, 0.9))
end
# 이 상계로 만든 하한 q + (V_F⁰ − V_F(∅))/120 이 최적 근처를 얼마나 쳐내나
qs = Dict(s => q for (s, q, V, _) in rows)
best = minimum(q + V for (s, q, V, _) in rows if !isempty(s))
lbv = [(qs[s] - b) for (s, r, b) in out]
@printf("최적 %.2f. 이 상계로 만든 하한의 최솟값 %.2f, 하한 < 최적 + 0.5%% 인 x %d / %d\n", best, minimum(lbv), count(<(best + 0.005 * abs(best)), lbv), length(lbv))
