# 거점별 분해 부등식의 수치 검사 (belief 고정):
#   리더 비용 φ_s(T) = −Σ_{i∈T} (거점 i 의 수익).  후보 부등식: 거점 i 의 수익 ≤ min_{j∈T, j≠i} Rbar_is({i, j})  (경쟁 거점이 생기면 줄어듦)
#   Rbar_is(S) = 배치 S 에서 follower 의 최적 반응 α(S, d) 아래, recourse 최적해들 가운데 거점 i 수익의 최댓값 (LP)
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(@__DIR__, "bgen_core.jl"))
g = parse(Int, get(ENV, "GEO", "1"))
nd = make_loc_tkp(; g=g, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
_BXROWS[] = unique(e.k for e in nz_xentries(nd)); K, nB, S = nd.meta[:K], nd.meta[:nB], nd.S; qh = nd.q_hat
m, ny = nz_m(nd), nz_ny(nd); MM = nd.meta; nA, ncu = MM[:nA], MM[:ncu]
cols(st) = (ib = nA + st; vcat([MM[:iyR](ib, j) for j in 1:ncu], [MM[:iyS](ib, j) for j in 1:ncu], [MM[:ys](ib)]))
ev = BEv(nd, GRB_ENV); yp = all_variables(ev.Mφ)[1:ny]
objL = @expression(ev.Mφ, sum(nd.ell[j] * yp[j] for j in 1:ny))
objS = Dict(st => @expression(ev.Mφ, sum(-nd.ell[j] * yp[j] for j in cols(st))) for st in 1:nB)
xvec(sel) = (x = zeros(nd.nx); for (s, k) in sel; x[(s - 1) * K + k] = 1; end; x)
"배치 sel 과 belief d 에서: (정확 값 v, 재가중 r, 시나리오별 비관 비용 φ, 거점별·시나리오별 최대 수익 Rbar)"
function info(sel, d)
    x = xvec(sel); bev_setx!(ev, nd, x); α = bev_resp!(ev, nd, d)
    v, _ = bev_value!(ev, nd, α); r = copy(ev.lastr)
    φ = zeros(S); Rb = Dict(st => zeros(S) for (st, _) in sel)
    for s in 1:S
        for k in 1:m
            rr = nd.u[k, s] + ev.gx[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0)
            set_normalized_rhs(ev.MQcon[k], rr); set_normalized_rhs(ev.Mφcon[k], rr)
        end
        optimize!(ev.MQ); Q = objective_value(ev.MQ); set_normalized_rhs(ev.MφcQ, Q - 1e-9 * max(1.0, abs(Q)))
        @objective(ev.Mφ, Max, objL); optimize!(ev.Mφ); φ[s] = objective_value(ev.Mφ)
        for (st, _) in sel
            @objective(ev.Mφ, Max, objS[st]); optimize!(ev.Mφ); Rb[st][s] = objective_value(ev.Mφ)
        end
    end
    @objective(ev.Mφ, Max, objL)
    return (v=v, r=r, φ=φ, Rb=Rb)
end
rng = MersenneTwister(3); d = qh
one = Dict{Any,Any}(); two = Dict{Any,Any}()
t1 = @elapsed for i in 1:nB, k in 1:K; one[(i, k)] = info([(i, k)], d); end
t2 = @elapsed for i in 1:nB, j in i+1:nB, k1 in 1:K, k2 in 1:K; two[((i, k1), (j, k2))] = info([(i, k1), (j, k2)], d); end
@printf("한 곳 %d (%.0fs), 두 곳 %d (%.0fs)\n", length(one), t1, length(two), t2); flush(stdout)
pairR(a, b) = a[1] < b[1] ? two[(a, b)].Rb[a[1]] : two[(b, a)].Rb[a[1]]          # 배치 {a, b} 에서 거점 a 의 수익 (시나리오별)
# 검사 1: 두 곳에서 거점 수익 ≤ 단독 수익 (시나리오별)
bad1 = 0; n1 = 0
for ((a, b), inf) in two, (me, _) in ((a, b), (b, a)), s in 1:S
    global n1 += 1; inf.Rb[me[1]][s] > one[me].Rb[me[1]][s] + 1e-6 * max(1.0, one[me].Rb[me[1]][s]) && (global bad1 += 1)
end
@printf("[1] 두 곳 배치의 거점 수익 ≤ 단독 수익: 위반 %d / %d\n", bad1, n1)
# 세 곳 표본
tris = Set{Any}()
while length(tris) < parse(Int, get(ENV, "NTRI", "300"))
    st = sort(randperm(rng, nB)[1:3]); push!(tris, [(s, rand(rng, 1:K)) for s in st])
end
bad2 = 0; n2 = 0; bad3 = 0; n3 = 0; sl_pair = Float64[]; sl_one = Float64[]; worst3 = 0.0
t3 = @elapsed for T in tris
    inf = info(T, d)
    for a in T, s in 1:S
        for b in T
            b == a && continue
            global n2 += 1; inf.Rb[a[1]][s] > pairR(a, b)[s] + 1e-6 * max(1.0, pairR(a, b)[s]) && (global bad2 += 1)
        end
    end
    # 시나리오별 총비용 하한: φ_s(T) ≥ −Σ_i min_j Rbar_is({i, j})
    lbp = [-sum(minimum(pairR(a, b)[s] for b in T if b != a) for a in T) for s in 1:S]
    lbo = [-sum(one[a].Rb[a[1]][s] for a in T) for s in 1:S]
    for s in 1:S
        global n3 += 1
        if inf.φ[s] < lbp[s] - 1e-6 * max(1.0, abs(lbp[s])); global bad3 += 1; global worst3 = max(worst3, (lbp[s] - inf.φ[s]) / abs(inf.φ[s])); end
    end
    push!(sl_pair, (inf.v - dot(inf.r, lbp)) / abs(inf.v)); push!(sl_one, (inf.v - dot(inf.r, lbo)) / abs(inf.v))
end
@printf("[2] 세 곳 배치의 거점 수익 ≤ 두 곳 배치에서의 수익: 위반 %d / %d\n", bad2, n2)
@printf("[3] 시나리오별 하한 φ_s(T) ≥ −Σ_i min_j Rbar: 위반 %d / %d (최악 %.3f)\n", bad3, n3, worst3)
q(v, p) = quantile(v, p)
@printf("[4] 세 곳 %d 개에서 하한의 느슨함 (값 − 하한)/|값|:  단독 기준 중앙값 %.3f (90%% %.3f)  |  두 곳 기준 중앙값 %.3f (90%% %.3f, 최대 %.3f), 0.5%% 안 %d, 2%% 안 %d, 5%% 안 %d  (%.0fs)\n",
        length(tris), median(sl_one), q(sl_one, 0.9), median(sl_pair), q(sl_pair, 0.9), maximum(sl_pair), count(<=(0.005), sl_pair), count(<=(0.02), sl_pair), count(<=(0.05), sl_pair), t3)
