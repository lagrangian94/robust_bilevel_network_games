# x̂ 고정: 싼 탐색으로 (v, d, r) 를 얻고, 블록 LB_μ(x̂; d, r) 를 μ 별로 정확 LP 로 계산해 v 와 비교
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(@__DIR__, "bgen_core.jl"))
nd = make_loc_tkp(; g=1, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
_BXROWS[] = unique(e.k for e in nz_xentries(nd)); MM = nd.meta; nA, nB, K, S = MM[:nA], MM[:nB], MM[:K], nd.S
m, ny, nh = nz_m(nd), nz_ny(nd), nz_nh(nd); A = nd.A; qh = nd.q_hat; bx = MM[:box]
rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m]
x̂ = zeros(nd.nx); x̂[(7 - 1) * K + 2] = 1; x̂[(11 - 1) * K + 2] = 1
ev = BEv(nd, GRB_ENV); bev_setx!(ev, nd, x̂)
inset(dd) = all(dd .>= bx.dl .- 1e-12) && all(dd .<= bx.du .+ 1e-12) && sum(abs.(dd .- qh)) <= 2nd.eps_tilde + 1e-12
cands = [copy(qh)]
for i in 1:S, j in 1:S
    i == j && continue; mv = min(qh[i] - bx.dl[i], bx.du[j] - qh[j], nd.eps_tilde); dd = copy(qh); dd[i] -= mv; dd[j] += mv; inset(dd) && push!(cands, dd)
end
best = (-Inf, nothing, nothing, nothing)
for dc in cands
    v, dd, α = bev_g!(ev, nd, dc); v > best[1] && (global best = (v, dd, copy(ev.lastr), α))
end
v, d, r, α = best
@printf("x̂ = 점포 {9:300; 13:300} (거점 7, 11): 싼 탐색 최고 v = %.3f\n   d = %s\n   r = %s\n   max r/d = %.2f\n", v, string(round.(d; digits=3)), string(round.(r; digits=3)), maximum(r ./ d))
F = Model(GRB1)
@variable(F, 0 <= h[i=1:nh] <= nd.hU[i]); @constraint(F, nd.W * h .<= nd.wvec); @variable(F, y[1:ny, 1:S] >= 0)
@constraint(F, [k=1:m, s=1:S], sum(a * y[j, s] for (j, a) in rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0) <= nd.u[k, s] + nz_gx(nd, k, s, x̂))
CF = @expression(F, -(dot(nd.c0, h) + sum(d[s] * nd.c[j] * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0)))
CL = @expression(F, sum(r[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0))
for ε in (0.0, 1e-1)
    @objective(F, Min, CF - ε * CL); optimize!(F); VFe = objective_value(F); CLpess = value(CL); hp = value.(h)
    @printf("ε = %.0e: 섭동 follower 최적의 리더 비용 (비관 대리) = %.3f   [정확 평가 v = %.3f], h = %s\n", ε, CLpess, v, string(round.(hp; digits=1)))
    for μ in (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0)
        @objective(F, Min, μ * (CF - ε * CL) + CL); optimize!(F)
        @printf("   μ = %7.2f: 블록 LB = %.3f\n", μ, objective_value(F) - μ * VFe)
    end
end
# 정확 평가의 구조: 그 α 에서 시나리오별 비관 recourse φ_s 와 r 가중합
Q, φ = nz_recourse_values(nd, x̂, α)
@printf("α (oracle 쪽 반응) 의 Σ r φ = %.3f, α = %s\n", dot(r, φ), string(round.(α; digits=1)))
# 낙관 정확값: follower 최적 중 리더 비용 최소
@objective(F, Min, CF); optimize!(F); vf = objective_value(F)
c1 = @constraint(F, CF <= vf + 1e-9 * abs(vf)); @objective(F, Min, CL); optimize!(F); vopt = objective_value(F)
@objective(F, Max, CL); optimize!(F); vpes = objective_value(F)
@printf("낙관 정확 %.3f, 비관 정확 %.3f (차 %.3f%%)\n", vopt, vpes, 100 * (vpes - vopt) / abs(vpes))
