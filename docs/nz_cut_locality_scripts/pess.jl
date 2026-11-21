# 같은 belief (d = r = q̂) 에서 낙관적 V_opt(x) vs 비관적 V_pess(x), 그리고 섭동 (follower 비용 − ε·리더 비용) 이 비관적 선택을 재현하는가
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Statistics
const GRB_ENV = Gurobi.Env()
include(joinpath(root, "nonzero_sum", "loc_tkp.jl"))
nd = make_loc_tkp(; g=parse(Int, get(ENV, "GEO", "1")), S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
MM = nd.meta; nA, nB, K, S = MM[:nA], MM[:nB], MM[:K], nd.S
m, ny, nh = nz_m(nd), nz_ny(nd), nz_nh(nd); A = nd.A; XE = nz_xentries(nd); d = nd.q_hat; r = nd.q_hat
rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m]
F = Model(() -> (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); o))
@variable(F, 0 <= h[i=1:nh] <= nd.hU[i]); @constraint(F, nd.W * h .<= nd.wvec); @variable(F, y[1:ny, 1:S] >= 0)
@constraint(F, con[k=1:m, s=1:S], sum(a * y[j, s] for (j, a) in rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0) <= nd.u[k, s])
costF = @expression(F, -(dot(nd.c0, h) + sum(d[s] * nd.c[j] * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0)))
costL = @expression(F, sum(r[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0))
opt_c = @constraint(F, costF <= 1e30)
xrows = unique(e.k for e in XE)
function setx!(lev)
    for k in xrows, s in 1:S; set_normalized_rhs(con[k, s], nd.u[k, s]); end
    for (st, l) in enumerate(lev); l == 0 && continue; k = MM[:rcap](nA + st); for s in 1:S; set_normalized_rhs(con[k, s], nd.u[k, s] + MM[:tiers][l]); end; end
end
levs = Vector{Vector{Int}}(); qs = Float64[]
push!(levs, zeros(Int, nB)); push!(qs, 0.0)
for n in 1:3, comb in Iterators.filter(c -> issorted(c) && allunique(c), Iterators.product(ntuple(_ -> 1:nB, n)...)), lv in Iterators.product(ntuple(_ -> 1:K, n)...)
    l = zeros(Int, nB); for (a, b) in zip(comb, lv); l[a] = b; end; push!(levs, l); push!(qs, sum(nd.qx[(a - 1) * K + b] for (a, b) in zip(comb, lv)))
end
EPS = (1e-1, 1e-2, 1e-3, 1e-4)
Vo = zeros(length(levs)); Vp = zeros(length(levs)); Ve = zeros(length(levs), length(EPS))
t = @elapsed for (n, lev) in enumerate(levs)
    setx!(lev)
    set_normalized_rhs(opt_c, 1e30); @objective(F, Min, costF); optimize!(F); vf = objective_value(F)
    set_normalized_rhs(opt_c, vf + 1e-7 * abs(vf))
    @objective(F, Min, costL); optimize!(F); Vo[n] = objective_value(F)
    @objective(F, Max, costL); optimize!(F); Vp[n] = objective_value(F)
    set_normalized_rhs(opt_c, 1e30)
    for (ie, ε) in enumerate(EPS)
        @objective(F, Min, costF - ε * costL); optimize!(F); Ve[n, ie] = value(costL)
    end
end
gap = (Vp .- Vo) ./ max.(1.0, abs.(Vo))
@printf("x %d 개 (%.0fs). 비관 − 낙관 차이: 0 (< 1e-6) %d, 0.5%% 넘음 %d, 5%% 넘음 %d, 최대 %.3f\n", length(levs), t, count(<(1e-6), gap), count(>(0.005), gap), count(>(0.05), gap), maximum(gap))
fo = qs .+ Vo; fp = qs .+ Vp
@printf("최적: 낙관 %.2f (x %s), 비관 %.2f (x %s)\n", minimum(fo), string(levs[argmin(fo)]), minimum(fp), string(levs[argmin(fp)]))
for (ie, ε) in enumerate(EPS)
    e = abs.(Ve[:, ie] .- Vp) ./ max.(1.0, abs.(Vp))
    @printf("섭동 ε = %.0e: 비관 값 재현 (1e-6 안) %d / %d, 최대 오차 %.2e, 섭동 해가 비관보다 큰 (최적 집합 밖) 경우 %d\n", ε, count(<(1e-6), e), length(e), maximum(e),
            count(i -> Ve[i, ie] > Vp[i] + 1e-6 * max(1.0, abs(Vp[i])), eachindex(Vp)))
end
open(joinpath(@__DIR__, "vtable_g1_pess.tsv"), "w") do io
    println(io, "lev\tq\tVopt\tVpess")
    for n in eachindex(levs); println(io, join(levs[n], ","), "\t", qs[n], "\t", Vo[n], "\t", Vp[n]); end
end
