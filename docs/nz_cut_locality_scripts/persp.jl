# perspective (Sherali–Adams 1 단계, 원·쌍대 모두 분할) cut 의 시험 (2026-11-20).
#   belief (d, r) 고정 블록  P(x) = min_{μ,κ,ω,h,y} D(x; μ,κ,ω) − λ V(h, y)
#     D = Σ uᵀμ + wᵀκ + hUᵀω + Σ_e g_e x_e μ_{k(e)}   (벌점 LP 의 쌍대, x·μ 곱)
#     V = c0ᵀh + Σ_s d_s cᵀy_s,  A y_s − B h ≤ u_s + Σ_e g_e x_e   (follower 대안)
#   W (지금): x·μ 를 McCormick.  SA: 거점 st 의 상태 t (0 = 닫힘, 1..K) 마다 모든 변수의 사본, Σ_t 사본 = 원본,
#     사본은 z_{st,t} 를 곱한 원래 제약을 만족, 다른 거점 용량은 곱 변수 X = z·x (McCormick, 대칭) 로.
#   cut 은 x 를 등식으로 고정한 LP 의 쌍대 (꼭짓점에서는 부분기울기가 유일하지 않아 중심 쪽 ε 이동한 점에서).
#   비교: 정수점 x′ 에서 (P(x′) − ℓ(x′)) / |P(x′) − P(x̂)|  (0 = 정확, 음수면 무효)
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics
const GRB_ENV = Gurobi.Env()
include(joinpath(root, "nonzero_sum", "loc_tkp.jl"))
envf(k, d) = get(ENV, k, d)
nd = make_loc_tkp(; g=1, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
K, nB, S = nd.meta[:K], nd.meta[:nB], nd.S; qh = nd.q_hat; MM = nd.meta; nA = MM[:nA]; tiers = MM[:tiers]
m, ny, nh = nz_m(nd), nz_ny(nd), nz_nh(nd); nW = size(nd.W, 1); A = nd.A
SITES = parse.(Int, split(envf("SITES", "4,6,9,11,13"), ","))          # 쓰는 B 거점 (나머지는 닫힘 고정)
nS = length(SITES); nx = nS * K
xe(a, k) = (a - 1) * K + k                                              # 국소 x 번호: a = SITES 안의 순번
capk(a) = MM[:rcap](nA + SITES[a])
colA = [[(k, A[k, j]) for k in 1:m if A[k, j] != 0] for j in 1:ny]; rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m]
lam = parse(Float64, envf("LAM", "10")); d = qh; r = qh
rhs_obj(s) = [r[s] * nd.ell[j] + lam * d[s] * nd.c[j] for j in 1:ny]
Ucap(k, s) = lam * d[s] * nd.piFU[k] + 2.0 * r[s] * maximum(abs, nd.ell)
hrow_of = [[k for k in 1:m if nd.hrow[k] == i] for i in 1:nh]
mkM() = (M = Model(() -> (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), parse(Int, envf("THR", "8"))); o)); M)
"한 사본 (가중 z: 상수 또는 affine 식) 의 변수와 제약. capfun(k, s) = 그 사본에서 용량 행 k 의 x 항 (affine 식)"
function block!(M, z, capfun)
    mu = @variable(M, [1:m, 1:S], lower_bound = 0.0); kp = @variable(M, [1:nW], lower_bound = 0.0); om = @variable(M, [1:nh], lower_bound = 0.0)
    h = @variable(M, [1:nh], lower_bound = 0.0); y = @variable(M, [1:ny, 1:S], lower_bound = 0.0)
    @constraint(M, [j=1:ny, s=1:S], sum(a * mu[k, s] for (k, a) in colA[j]) >= rhs_obj(s)[j] * z)
    @constraint(M, [i=1:nh], -sum(nd.hcoef[k] * mu[k, s] for k in hrow_of[i], s in 1:S) + sum(nd.W[l, i] * kp[l] for l in 1:nW) + om[i] >= lam * nd.c0[i] * z)
    @constraint(M, [i=1:nh], h[i] <= nd.hU[i] * z); @constraint(M, nd.W * h .<= nd.wvec .* z)
    @constraint(M, [k=1:m, s=1:S], sum(a * y[j, s] for (j, a) in rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0) <= nd.u[k, s] * z + capfun(k, s))
    return (mu=mu, kp=kp, om=om, h=h, y=y)
end
Dbase(b) = sum(nd.u[k, s] * b.mu[k, s] for k in 1:m, s in 1:S if nd.u[k, s] != 0) + dot(nd.wvec, b.kp) + dot(nd.hU, b.om)
Vval(b) = dot(nd.c0, b.h) + sum(d[s] * nd.c[j] * b.y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0)
iscap = Dict(capk(a) => a for a in 1:nS)
"W (McCormick) LP. x 를 변수로 두고 등식 고정 → (값, 기울기)"
function solveW(x̂)
    M = mkM(); @variable(M, xv[1:nx]); fx = @constraint(M, [i=1:nx], xv[i] == x̂[i])
    capf(k, s) = haskey(iscap, k) ? sum(tiers[t] * xv[xe(iscap[k], t)] for t in 1:K) : AffExpr(0.0)
    b = block!(M, 1.0, capf)
    obj = Dbase(b) - lam * Vval(b)
    for a in 1:nS, t in 1:K, s in 1:S
        kc = capk(a); U = Ucap(kc, s); set_upper_bound(b.mu[kc, s], U)
        p = @variable(M, lower_bound = 0.0); @constraint(M, p >= b.mu[kc, s] - U * (1 - xv[xe(a, t)])); add_to_expression!(obj, tiers[t], p)
    end
    @objective(M, Min, obj); optimize!(M)
    termination_status(M) == MOI.OPTIMAL || error("W: $(termination_status(M))")
    return objective_value(M), [dual(fx[i]) for i in 1:nx]
end
"SA 1 단계 LP"
function solveSA(x̂)
    M = mkM(); @variable(M, xv[1:nx]); fx = @constraint(M, [i=1:nx], xv[i] == x̂[i])
    zt(a, t) = t == 0 ? 1 - sum(xv[xe(a, k)] for k in 1:K) : 1.0 * xv[xe(a, t)]
    # 곱 X[a,t,(b,k)] = z_{a,t} x_{b,k}  (b ≠ a), 대칭·McCormick·합 조건
    X = Dict{Tuple{Int,Int,Int,Int},VariableRef}()
    for a in 1:nS, t in 0:K, b in 1:nS, k in 1:K
        b == a && continue
        v = @variable(M, lower_bound = 0.0); X[(a, t, b, k)] = v
        @constraint(M, v <= zt(a, t)); @constraint(M, v <= xv[xe(b, k)]); @constraint(M, v >= zt(a, t) + xv[xe(b, k)] - 1)
    end
    for a in 1:nS, b in 1:nS, k in 1:K
        b == a && continue
        @constraint(M, sum(X[(a, t, b, k)] for t in 0:K) == xv[xe(b, k)])
        for t in 1:K; a < b && @constraint(M, X[(a, t, b, k)] == X[(b, k, a, t)]); end
    end
    for a in 1:nS, t in 0:K, b in 1:nS                                   # 한 거점은 한 상태
        b == a && continue; @constraint(M, sum(X[(a, t, b, k)] for k in 1:K) <= zt(a, t))
    end
    orig = block!(M, 1.0, (k, s) -> haskey(iscap, k) ? sum(tiers[t] * xv[xe(iscap[k], t)] for t in 1:K) : AffExpr(0.0))
    obj = Dbase(orig) - lam * Vval(orig)
    for a in 1:nS
        cps = Dict{Int,Any}()
        for t in 0:K
            capf(k, s) = !haskey(iscap, k) ? AffExpr(0.0) :
                         iscap[k] == a ? (t == 0 ? AffExpr(0.0) : tiers[t] * zt(a, t)) :
                         sum(tiers[kk] * X[(a, t, iscap[k], kk)] for kk in 1:K)
            cps[t] = block!(M, zt(a, t), capf)
        end
        for f in (:mu, :kp, :om, :h, :y)
            orv = getfield(orig, f)
            @constraint(M, [idx in eachindex(orv)], sum(getfield(cps[t], f)[idx] for t in 0:K) == orv[idx])
        end
        kc = capk(a)                                                     # x_{a,t} μ_{cap a} = 사본 t 의 μ
        for t in 1:K, s in 1:S; add_to_expression!(obj, tiers[t], cps[t].mu[kc, s]); end
        # 쌍대 실행가능 영역은 유계가 아니라 z = 0 인 사본도 0 이 아닐 수 있다 (점근 방향). 그러면 용량 행 쌍대를 비활성 사본으로 옮겨
        # 곱 항을 피한다 (다른 거점이 열려 있을 때 현이 깨짐). 쌍대 상한 × 상태 변수로 막는다: 정수 상태에서는 정확, 분수 상태에서만 느슨
        if envf("ZB", "1") == "1"
            for t in 0:K, s in 1:S; @constraint(M, cps[t].mu[kc, s] <= Ucap(kc, s) * zt(a, t)); end
        end
    end
    @objective(M, Min, obj); optimize!(M)
    termination_status(M) == MOI.OPTIMAL || error("SA: $(termination_status(M))")
    return objective_value(M), [dual(fx[i]) for i in 1:nx]
end
xof(sel) = (x = zeros(nx); for (a, k) in sel; x[xe(a, k)] = 1; end; x)
EPS = 1e-3; ctr = fill(1 / (K + 1), nx)
cutat(fn, x̂) = (xε = x̂ .+ EPS .* (ctr .- x̂); (v, gr) = fn(xε); (v + dot(gr, x̂ .- xε), gr))
# 정수점 전부 (작은 거점 집합)
allX = Vector{Vector{Tuple{Int,Int}}}(); push!(allX, Tuple{Int,Int}[])
for n in 1:min(3, nS), comb in Iterators.filter(c -> issorted(c) && allunique(c), Iterators.product(ntuple(_ -> 1:nS, n)...)), lv in Iterators.product(ntuple(_ -> 1:K, n)...)
    push!(allX, collect(zip(comb, lv)))
end
tP = @elapsed (Pv = [solveW(xof(s))[1] for s in allX])
@printf("거점 %s (|X| = %d), λ = %.0f. 정수점 값 P 계산 %.0fs. P 범위 [%.0f, %.0f]\n", string(SITES), length(allX), lam, tP, minimum(Pv), maximum(Pv)); flush(stdout)
for base in (Tuple{Int,Int}[], [(1, 2)], [(2, 1), (3, 2)])
    x̂ = xof(base); Ph = solveW(x̂)[1]
    for (nm, fn) in (("McCormick", solveW), ("perspective", solveSA))
        tc = @elapsed ((v, gr) = cutat(fn, x̂))
        err = Float64[]; nbad = 0; cls = Dict(1 => Float64[], 2 => Float64[], 3 => Float64[])
        for (i, sel) in enumerate(allX)
            sel == base && continue
            ℓ = v + dot(gr, xof(sel) .- x̂); e = (Pv[i] - ℓ) / max(1.0, abs(Pv[i] - Ph))
            e < -1e-4 && (nbad += 1); push!(err, e)
            nd_ = length(symdiff(Set(sel), Set(base))); push!(cls[min(nd_, 3)], e)
        end
        @printf("x̂=%-18s %-11s 값 %.1f (참 %.1f) %5.1fs | 무효 %d | 오차/변화: 한 칸 중앙 %.2f · 두 칸 %.2f · 세 칸 이상 %.2f | 0.1 안 %d%% | 기울기 [%.0f, %.0f]\n",
                string(base), nm, v, Ph, tc, nbad, median(cls[1]), isempty(cls[2]) ? NaN : median(cls[2]), isempty(cls[3]) ? NaN : median(cls[3]),
                round(Int, 100 * mean(err .<= 0.1)), minimum(gr), maximum(gr)); flush(stdout)
    end
end
