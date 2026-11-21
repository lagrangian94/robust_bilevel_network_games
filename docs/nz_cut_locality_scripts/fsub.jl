# follower 값함수 V_F(x; d) 가 용량 증분에 대해 단조·submodular 인가 (LP 값, 잡음 없음).
#   원소 = 증분 (거점 st, 단계 k): 단계 1 = 0→200, 2 = 200→300, 3 = 300→400. 집합은 아래로 닫힌 것만 (단계 k 를 넣으려면 k−1 이 있어야).
#   검사: f(S+e1+e2) − f(S+e2) ≤ f(S+e1) − f(S)   (e1, e2 는 S 에 각각, 그리고 함께 추가 가능)
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics
const GRB_ENV = Gurobi.Env()
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(@__DIR__, "bgen_core.jl"))
g = parse(Int, get(ENV, "GEO", "1")); NS = parse(Int, get(ENV, "NS", "800")); MAXS = parse(Int, get(ENV, "MAXS", "6"))
nd = make_loc_tkp(; g=g, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
_BXROWS[] = unique(e.k for e in nz_xentries(nd)); K, nB, S = nd.meta[:K], nd.meta[:nB], nd.S; qh = nd.q_hat
ev = BEv(nd, GRB_ENV); cache = Dict{Vector{Int},Float64}()
"lev[st] = 거점 st 의 단계 (0 = 닫힘). follower 2 단계 LP 의 최적값 (최대화 부호)"
function f(lev::Vector{Int}, d)
    get!(cache, copy(lev)) do
        xv = zeros(nd.nx); for st in 1:nB; lev[st] > 0 && (xv[(st - 1) * K + lev[st]] = 1.0); end
        bev_setx!(ev, nd, xv); bev_resp!(ev, nd, d) === nothing && error("follower LP 실패"); objective_value(ev.R)
    end
end
rng = MersenneTwister(8); d = qh
nmono = 0; nsub = 0; vsub = Float64[]; ntest = 0; ncross = 0; nown = 0; vcross = 0; vown = 0
for t in 1:NS
    lev = zeros(Int, nB)
    for _ in 1:rand(rng, 0:MAXS); st = rand(rng, 1:nB); lev[st] < K && (lev[st] += 1); end
    cand = [st for st in 1:nB if lev[st] < K]
    a = rand(rng, cand); l1 = copy(lev); l1[a] += 1
    # e2: 다른 거점의 증분, 또는 같은 거점의 다음 증분
    same = rand(rng) < 0.3 && l1[a] < K
    if same
        l2 = nothing; l12 = copy(l1); l12[a] += 1                 # 같은 거점: 오목성 f(b+2) − f(b+1) ≤ f(b+1) − f(b)
        f0, f1, f12 = f(lev, d), f(l1, d), f(l12, d)
        g1 = f1 - f0; g2 = f12 - f1; global nown += 1
        (g1 < -1e-7 * abs(f0) || g2 < -1e-7 * abs(f0)) && (global nmono += 1)
        if g2 > g1 + 1e-7 * abs(f0); global vown += 1; push!(vsub, (g2 - g1) / max(1.0, abs(g1))); end
    else
        b = rand(rng, [st for st in cand if st != a]); l2 = copy(lev); l2[b] += 1; l12 = copy(l1); l12[b] += 1
        f0, f1, f2, f12 = f(lev, d), f(l1, d), f(l2, d), f(l12, d)
        g1 = f1 - f0; g1b = f12 - f2; global ncross += 1
        (g1 < -1e-7 * abs(f0) || f2 - f0 < -1e-7 * abs(f0)) && (global nmono += 1)
        if g1b > g1 + 1e-7 * abs(f0); global vcross += 1; push!(vsub, (g1b - g1) / max(1.0, abs(g1))); end
    end
    global ntest += 1
end
@printf("f(∅) = %.1f,  한 증분의 이득 예: %.1f\n", f(zeros(Int, nB), d), f([i == 1 ? 1 : 0 for i in 1:nB], d) - f(zeros(Int, nB), d))
@printf("검사 %d 건 (다른 거점 %d, 같은 거점 %d), follower LP %d 개\n", ntest, ncross, nown, length(cache))
@printf("단조성 위반 %d\n", nmono)
@printf("submodular 위반: 다른 거점 %d / %d, 같은 거점 (오목성) %d / %d", vcross, ncross, vown, nown)
isempty(vsub) ? println() : @printf("  | 위반 크기 (이득 대비) 중앙값 %.3f, 최대 %.3f\n", median(vsub), maximum(vsub))
