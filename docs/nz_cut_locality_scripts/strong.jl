# 강한 정식화 시험: belief (d, r) 고정 블록  P(x) = D(x) − λ V_F(x)  의 x-완화가 얼마나 조이는가.
#   D(x)   = max_{h,y} Σ_s r_s ℓᵀy_s + λ (c0ᵀh + Σ_s d_s cᵀy_s)   (쌍대로 넣음: x·쌍대 곱)
#   V_F(x) = max_{h,y} c0ᵀh + Σ_s d_s cᵀy_s                        (원문제로 넣음)
# 정식화: W = 지금 (용량 행만, McCormick),  V = + 호 단위 제약 y_ij ≤ ξ_j z_i (VUB),  R = V + 쌍대의 거점별 조건부 사본 (RLT 1 단계)
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics
const GRB_ENV = Gurobi.Env()
include(joinpath(root, "nonzero_sum", "loc_tkp.jl"))
nd = make_loc_tkp(; g=1, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
K, nB, S = nd.meta[:K], nd.meta[:nB], nd.S; qh = nd.q_hat; MM = nd.meta; nA, ncu = MM[:nA], MM[:ncu]
m, ny, nh, nx = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.nx; nW = size(nd.W, 1); A = nd.A; XE = nz_xentries(nd)
colA = [[(k, A[k, j]) for k in 1:m if A[k, j] != 0] for j in 1:ny]; rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m]
ents_of_row = [[e for e in XE if e.k == k] for k in 1:m]
lam = 10.0; d = qh; r = qh
xi(j, s) = -nd.u[MM[:rdem](j), s]                               # 수요
arcs = [(st, j) for st in 1:nB for j in 1:ncu]                 # B 거점 st, 고객 j
colR(st, j) = MM[:iyR](nA + st, j); colS(st, j) = MM[:iyS](nA + st, j)
site_of_col = Dict{Int,Tuple{Int,Int}}(); for (st, j) in arcs; site_of_col[colR(st, j)] = (st, j); site_of_col[colS(st, j)] = (st, j); end
rhs_obj(s) = [r[s] * nd.ell[j] + lam * d[s] * nd.c[j] for j in 1:ny]
Ucap(k, s) = lam * d[s] * nd.piFU[k] + 2.0 * r[s] * maximum(abs, nd.ell)
capsite = Dict(MM[:rcap](nA + st) => st for st in 1:nB)
"정식화 form ∈ (:W, :V, :R) 의 완화 값 P̃(x) (x 는 [0,1]^nx 의 임의 점)"
function Ptilde(xv, form::Symbol)
    M = Model(() -> (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); o))
    z = [sum(xv[(st - 1) * K + k] for k in 1:K) for st in 1:nB]
    gcap = [sum(e.g[s] * xv[e.i] for e in ents_of_row[k]; init=0.0) for k in 1:m, s in 1:S]
    vub = form != :W
    # --- V_F 쪽: follower 원문제 사본 ---
    h = @variable(M, [i=1:nh], lower_bound = 0.0, upper_bound = nd.hU[i]); @constraint(M, nd.W * h .<= nd.wvec)
    y = @variable(M, [1:ny, 1:S], lower_bound = 0.0)
    @constraint(M, [k=1:m, s=1:S], sum(a * y[j, s] for (j, a) in rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0) <= nd.u[k, s] + gcap[k, s])
    vub && @constraint(M, [(st, j) in arcs, s in 1:S], y[colR(st, j), s] + y[colS(st, j), s] <= xi(j, s) * z[st])
    VF = dot(nd.c0, h) + sum(d[s] * nd.c[j] * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0)
    # --- D 쪽: 벌점 LP 의 쌍대 ---
    mu = @variable(M, [1:m, 1:S], lower_bound = 0.0); kp = @variable(M, [1:nW], lower_bound = 0.0); om = @variable(M, [1:nh], lower_bound = 0.0)
    w = vub ? @variable(M, [1:nB, 1:ncu, 1:S], lower_bound = 0.0) : nothing        # 호 단위 제약의 쌍대
    for j in 1:ny, s in 1:S
        extra = (vub && haskey(site_of_col, j)) ? w[site_of_col[j][1], site_of_col[j][2], s] : 0.0
        @constraint(M, sum(a * mu[k, s] for (k, a) in colA[j]) + extra >= rhs_obj(s)[j])
    end
    @constraint(M, [i=1:nh], -sum(nd.hcoef[k] * mu[k, s] for k in 1:m, s in 1:S if nd.hrow[k] == i) + sum(nd.W[l, i] * kp[l] for l in 1:nW) + om[i] >= lam * nd.c0[i])
    Dobj = AffExpr(0.0)
    add_to_expression!(Dobj, sum(nd.u[k, s] * mu[k, s] for k in 1:m, s in 1:S if nd.u[k, s] != 0) + dot(nd.wvec, kp) + dot(nd.hU, om))
    if form != :R
        # McCormick: 곱 ≥ 쌍대 − U (1 − x)
        psi = @variable(M, [1:length(XE), 1:S], lower_bound = 0.0)
        for (ei, e) in enumerate(XE), s in 1:S
            @constraint(M, psi[ei, s] >= mu[e.k, s] - Ucap(e.k, s) * (1 - xv[e.i])); add_to_expression!(Dobj, e.g[s], psi[ei, s])
        end
        for k in keys(capsite), s in 1:S; set_upper_bound(mu[k, s], Ucap(k, s)); end
        if vub
            p2 = @variable(M, [1:nB, 1:ncu, 1:S], lower_bound = 0.0)
            for (st, j) in arcs, s in 1:S
                U2 = Ucap(MM[:rcap](nA + st), s); set_upper_bound(w[st, j, s], U2)
                @constraint(M, p2[st, j, s] >= w[st, j, s] - U2 * (1 - z[st])); add_to_expression!(Dobj, xi(j, s), p2[st, j, s])
            end
        end
    else
        # RLT 1 단계: 거점 st 의 상태 t (0 = 닫힘, 1..K) 마다 쌍대 사본 (mu, w 의 그 거점 관련 성분만이 아니라 전체), Σ_t 사본 = 원본, 쌍대 실행가능 × 상태 변수
        zt(st, t) = t == 0 ? 1 - z[st] : xv[(st - 1) * K + t]
        for st in 1:nB
            muc = @variable(M, [0:K, 1:m, 1:S], lower_bound = 0.0); wc = @variable(M, [0:K, 1:nB, 1:ncu, 1:S], lower_bound = 0.0)
            @constraint(M, [k=1:m, s=1:S], sum(muc[t, k, s] for t in 0:K) == mu[k, s])
            @constraint(M, [a=1:nB, j=1:ncu, s=1:S], sum(wc[t, a, j, s] for t in 0:K) == w[a, j, s])
            # 상한 × 상태 변수: 비활성 상태의 사본은 0 (이게 없으면 용량·호 쌍대를 비활성 사본으로 옮겨 x 항을 피함 → 정수점에서도 틀림)
            for t in 0:K, s in 1:S
                for (kcap, _) in capsite; @constraint(M, muc[t, kcap, s] <= Ucap(kcap, s) * zt(st, t)); end
                for a in 1:nB, j in 1:ncu; @constraint(M, wc[t, a, j, s] <= Ucap(MM[:rcap](nA + a), s) * zt(st, t)); end
            end
            for t in 0:K, j in 1:ny, s in 1:S
                extra = haskey(site_of_col, j) ? wc[t, site_of_col[j][1], site_of_col[j][2], s] : 0.0
                @constraint(M, sum(a * muc[t, k, s] for (k, a) in colA[j]) + extra >= rhs_obj(s)[j] * zt(st, t))
            end
            # 이 거점의 x·쌍대 곱 = 조건부 사본 (용량: 단계 t 의 사본 × 그 단계 용량,  호 단위: 열린 상태 사본들의 합 × 수요)
            kc = MM[:rcap](nA + st)
            for t in 1:K, s in 1:S
                add_to_expression!(Dobj, nd.meta[:tiers][t], muc[t, kc, s])
                for j in 1:ncu; add_to_expression!(Dobj, xi(j, s), wc[t, st, j, s]); end
            end
        end
    end
    @objective(M, Min, Dobj - lam * VF); optimize!(M)
    termination_status(M) == MOI.OPTIMAL || return NaN
    return objective_value(M)
end
xof(sel) = (x = zeros(nx); for (st, k) in sel; x[(st - 1) * K + k] = 1; end; x)
println("m=$m ny=$ny  호 $(length(arcs))")
# (1) 변 하나: x̂ = {(4,2),(9,2)} 에서 거점 13 을 크기 2 로 여는 방향
base = [(4, 2), (9, 2)]; add = (13, 2)
x0 = xof(base); x1 = xof(vcat(base, [add]))
for form in (:W, :V, :R)
    t0 = @elapsed (p0 = Ptilde(x0, form)); p1 = Ptilde(x1, form)
    @printf("[변] %s: P(0)=%.1f P(1)=%.1f (%.0fs/LP) |", form, p0, p1, t0)
    for t in (0.05, 0.25, 0.5, 0.75)
        v = Ptilde(x0 .+ t .* (x1 .- x0), form); chord = p0 + t * (p1 - p0)
        @printf("  t=%.2f: %.0f (현 %.0f)", t, v, chord)
    end
    println(); flush(stdout)
end
# (2) 얇게 퍼진 분수점: 모든 거점의 크기 2 를 3/13 씩 (합 3). 정수점 값의 범위와 비교
xs = zeros(nx); for st in 1:nB; xs[(st - 1) * K + 2] = 3 / nB; end
rng = MersenneTwister(1); pv = Float64[]
for t in 1:12; st = sort(randperm(rng, nB)[1:3]); push!(pv, Ptilde(xof([(s, 2) for s in st]), :W)); end
@printf("[퍼짐] 크기 2 세 곳짜리 정수점 12 개의 P: 최소 %.0f, 중앙 %.0f, 최대 %.0f\n", minimum(pv), median(pv), maximum(pv))
for form in (:W, :V, :R)
    t0 = @elapsed (v = Ptilde(xs, form)); @printf("[퍼짐] %s: 균등 분수점 (거점마다 3/13) 의 완화 값 %.0f  (%.0fs)\n", form, v, t0); flush(stdout)
end
