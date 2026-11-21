# 고정 ŷ (value-function / disjunctive) cut 시험 (2026-11-20). belief 하나 (d, r) 의 낙관적 bilevel 을 높은점 완화 + lazy cut 으로 푼다.
#   마스터: x 이진, follower 원문제 사본 (h, y) (x 는 우변), t0 ≥ Σ_s r_s ℓᵀy_s.  쌍대·McCormick·벌점 없음.
#   cut (정수해 x̂ 에서): ŷ = follower 최적 반응, V̂ = 그 값, I = ŷ 가 쓰는 거점, k_i = 필요한 최소 크기.
#        F(h, y) ≥ V̂ − Σ_{i∈I} M_i (1 − Σ_{k ≥ k_i} x_i^k)        (F = follower 목적, 최대화 부호)
#        MODE = total: M_i = V̂ − V_F(∅) (항상 유효).  MODE = alone: M_i = 그 거점만 필요한 크기로 열었을 때의 이득 (follower 값이 용량에 submodular 일 때 유효)
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(@__DIR__, "bgen_core.jl"))
envf(k, d) = get(ENV, k, d)
g = parse(Int, envf("GEO", "1")); MODE = Symbol(envf("MODE", "total")); TL = parse(Float64, envf("TL", "1800")); PRE = parse(Int, envf("PRE", "0"))
nd = make_loc_tkp(; g=g, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
_BXROWS[] = unique(e.k for e in nz_xentries(nd)); K, nB, S = nd.meta[:K], nd.meta[:nB], nd.S; qh = nd.q_hat; MM = nd.meta; nA, ncu = MM[:nA], MM[:ncu]
m, ny, nh, nx = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.nx; A = nd.A; XE = nz_xentries(nd); tiers = MM[:tiers]
rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m]; ents_of_row = [[e for e in XE if e.k == k] for k in 1:m]
if haskey(ENV, "SITES")
    keep = parse.(Int, split(ENV["SITES"], ",")); for st in 1:nB, k in 1:K; st in keep || (nd.x_allowed[(st - 1) * K + k] = false); end
end
d = qh; r = qh
cols(st) = (ib = nA + st; vcat([MM[:iyR](ib, j) for j in 1:ncu], [MM[:iyS](ib, j) for j in 1:ncu], [MM[:ys](ib)]))
ev = BEv(nd, GRB_ENV)
"x 에서 follower 최적: (값 V_F, 거점별 필요한 최소 크기 — 안 쓰면 0)"
function follower(xv)
    bev_setx!(ev, nd, xv); bev_resp!(ev, nd, d) === nothing && error("follower LP 실패")
    V = objective_value(ev.R); need = zeros(Int, nB)
    for st in 1:nB
        use = maximum(sum(value(ev.Ry[j, s]) for j in cols(st)) for s in 1:S)
        use <= 1e-6 && continue
        need[st] = findfirst(k -> tiers[k] >= use - 1e-6, 1:K)
    end
    return V, need
end
V0, _ = follower(zeros(nx))
alone = Dict{Tuple{Int,Int},Float64}()
gain_alone(st, k) = get!(alone, (st, k)) do
    xv = zeros(nx); xv[(st - 1) * K + k] = 1; follower(xv)[1] - V0
end
# ---------- 마스터 ----------
NT = Threads.nthreads()
M = Model(() -> (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), NT); o))
set_optimizer_attribute(M, "LazyConstraints", 1); set_optimizer_attribute(M, "MIPGap", 1e-4); set_optimizer_attribute(M, "TimeLimit", TL)
set_optimizer_attribute(M, "OutputFlag", 1); set_optimizer_attribute(M, "LogToConsole", 0); set_optimizer_attribute(M, "LogFile", joinpath(@__DIR__, "hpr_grb_$(MODE)_$(PRE)_$(envf("VUB", "0")).log"))
@variable(M, x[1:nx], Bin); @variable(M, t0 >= -1e5)
@constraint(M, sum(x) <= nd.gamma)
for (idx, rhs) in get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[]); @constraint(M, sum(x[i] for i in idx) <= rhs); end
for i in 1:nx; nd.x_allowed[i] || fix(x[i], 0.0; force=true); end
@variable(M, 0 <= h[i=1:nh] <= nd.hU[i]); @constraint(M, nd.W * h .<= nd.wvec); @variable(M, y[1:ny, 1:S] >= 0)
@constraint(M, [k=1:m, s=1:S], sum(a * y[j, s] for (j, a) in rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0)
                               - sum(e.g[s] * x[e.i] for e in ents_of_row[k]; init=0.0) <= nd.u[k, s])
const VUB = envf("VUB", "0") == "1"
if VUB                                                          # 호 단위 제약: 거점 st 에서 고객 j 로 가는 양 ≤ 수요 × (거점이 열림)
    for st in 1:nB, j in 1:ncu, s in 1:S
        @constraint(M, y[MM[:iyR](nA + st, j), s] + y[MM[:iyS](nA + st, j), s] <= -nd.u[MM[:rdem](j), s] * sum(x[(st - 1) * K + k] for k in 1:K))
    end
end
@constraint(M, t0 >= sum(r[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0))
Fexpr = @expression(M, dot(nd.c0, h) + sum(d[s] * nd.c[j] * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0))
@constraint(M, Fexpr >= V0)                                    # 거점이 늘면 follower 값은 줄지 않음
@objective(M, Min, dot(nd.qx, x) + t0)
"x̂ 에서의 cut 을 (좌변 식 ≥ 우변 식) 으로"
function cut_at(xv)
    V, need = follower(xv)
    rhs = AffExpr(V)
    for st in 1:nB
        need[st] == 0 && continue
        Mi = MODE == :total ? V - V0 : gain_alone(st, need[st])
        add_to_expression!(rhs, -Mi); for k in need[st]:K; add_to_expression!(rhs, Mi, x[(st - 1) * K + k]); end
    end
    return V, rhs
end
ncut = Ref(0); ncb = Ref(0)
if PRE >= 1                                                    # 한 곳 (PRE ≥ 1), 두 곳 (PRE ≥ 2) 배치의 cut 을 미리
    tp = @elapsed begin
        for st in 1:nB, k in 1:K
            nd.x_allowed[(st - 1) * K + k] || continue
            xv = zeros(nx); xv[(st - 1) * K + k] = 1; _, rhs = cut_at(xv); @constraint(M, Fexpr >= rhs); ncut[] += 1
        end
        if PRE >= 2
            for a in 1:nB, b in a+1:nB, k1 in 1:K, k2 in 1:K
                (nd.x_allowed[(a - 1) * K + k1] && nd.x_allowed[(b - 1) * K + k2]) || continue
                xv = zeros(nx); xv[(a - 1) * K + k1] = 1; xv[(b - 1) * K + k2] = 1; _, rhs = cut_at(xv); @constraint(M, Fexpr >= rhs); ncut[] += 1
            end
        end
    end
    @printf("미리 넣은 cut %d 개 (%.0fs)\n", ncut[], tp)
end
function cb(cb_data)
    callback_node_status(cb_data, M) == MOI.CALLBACK_NODE_STATUS_INTEGER || return
    ncb[] += 1
    xv = [callback_value(cb_data, x[i]) > 0.5 ? 1.0 : 0.0 for i in 1:nx]
    Fv = callback_value(cb_data, Fexpr)
    V, rhs = cut_at(xv)
    if Fv < V - 1e-7 * max(1.0, abs(V))
        MOI.submit(M, MOI.LazyConstraint(cb_data), @build_constraint(Fexpr >= rhs)); ncut[] += 1
    end
    return
end
set_attribute(M, MOI.LazyConstraintCallback(), cb)
# 루트 LP 하한 (cut 없이 / 미리 넣은 cut 만)
Mr = nothing
tt = @elapsed optimize!(M)
st = termination_status(M)
xb = has_values(M) ? Float64.(value.(x) .> 0.5) : zeros(nx)
@printf("RESULT hpr vub=$(envf("VUB","0")) g=%d mode=%s pre=%d status=%s x=%s obj=%.2f bound=%.2f nodes=%d callbacks=%d cuts=%d time=%.1f  (V_F(∅)=%.1f)\n",
        g, MODE, PRE, st, loc_tkp_xstr(nd, xb), has_values(M) ? objective_value(M) : NaN, objective_bound(M), Int(MOI.get(M, MOI.NodeCount())), ncb[], ncut[], tt, V0)
if has_values(M)                                               # 검산: 그 x 에서의 정확 평가 (비관적, 같은 belief 의 반응)
    bev_setx!(ev, nd, xb); v, _, _ = bev_g!(ev, nd, d)
    @printf("   검산: x* 에서 명목 belief 반응의 정확 값 g = %.2f → q·x + g = %.2f (마스터는 리더에게 유리한 tie·r = q̂ 라 이보다 낮거나 같아야 함)\n", v, dot(nd.qx, xb) + v)
end
