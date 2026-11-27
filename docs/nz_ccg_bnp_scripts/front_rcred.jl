# front_rcred.jl — 전선 상자 하나를 뿌리로 작은 best-first B&B 를 돌려, 노드 LP 의 쌍대 (변수 경계의 reduced cost) 로 상자를 줄이는 판과
#   α 분기만 한 판의 노드 수·시간을 비교 (2026-11-27, α-B&B 노드 수 튜닝 실험실, 스레드 1). front_dobbt_bb.jl 의 틀을 그대로 씀.
#  근거 (Ryoo & Sahinidis 1996, branch-and-reduce 의 marginals-based range reduction): 상자 B 의 노드 LP R(B) 의 최적값 z, α_k ≤ u_k 경계의
#   쌍대 μ > 0 이면, R(B) 에서 그 경계만 α_k ≤ t 로 바꾼 LP 는 쌍대해가 그대로 실행 가능이라 값 ≤ z − μ (u_k − t). R(B) 의 다른 행은 B 의 부분 상자에서도
#   유효하므로 이것은 B ∩ {α_k ≤ t} 의 상한이다. 그래서 z − μ (u_k − t) ≤ 문턱이면 그 부분은 버려도 됨 → l_k' = u_k − (z − 문턱)/μ. 아래 경계도 같음.
#   LP 를 더 풀지 않고 (이미 푼 LP 의 쌍대만 읽음) 자식 상자를 줄인다. 상자 경계가 계수로 들어간 행 (McCormick·RLT·범위 행) 의 쌍대는 쓰지 않는다 (그 행은 고정).
#  모드 MODE: none (α 분기만), rc (α 경계를 줄여 자식에 물림), rcd (α 와 d 경계 둘 다). RCR (기본 0): 줄인 상자로 그 노드를 다시 풀고 다시 줄이는 횟수
#  점검 CHECK=1: 상자마다 뿌리에서 줄인 경계 하나를 실제로 넣고 (다른 행은 그대로) 다시 풀어 값 ≤ 문턱 + 여유 인지 확인 (아니면 오류)
#  환경 변수: FR (전선 파일), HUCAP 1, K 8 (상자 수), PICK top | spread, NC 40, ROUNDS 30, TB 90 (상자당 초), CUT lb | tau, TAU
#  2026-11-27 추가: BR = viol (기본, 저장소 규칙 = 곱 위반 합 최대 좌표) | strong (모든 후보 좌표의 두 자식을 풀어 곱 점수 max(Δ−,ε)·max(Δ+,ε) 최대,
#   Δ = 부모 상한 − max(자식 상한, 문턱). 고른 좌표의 자식 풀이는 그대로 씀) | sfirst (strong 을 위반 상위 SB_TOP 좌표만). 노드 수와 따로 LP 수를 셈
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Serialization, Statistics
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
envf(k, d) = get(ENV, k, d)
fr = deserialize(envf("FR", ""))
K = parse(Int, envf("K", "8")); NC = parse(Int, envf("NC", "40")); RD = parse(Int, envf("ROUNDS", "30"))
nd = make_loc_tkp(; g=fr.g, S=fr.S, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2 / 3, beta=0.4, lambdaU=fr.lam)
x̄ = fr.x; K_ = nd.meta[:K]; tiers = nd.meta[:tiers]; nA = nd.meta[:nA]
if envf("HUCAP", "1") == "1"
    hU = copy(nd.hU); hidx = nd.meta[:hidx]; sx = nd.meta[:site_x]
    for i in nA+1:nd.meta[:nst]
        b = sum(tiers[(t - 1) % K_ + 1] * x̄[t] for t in sx[i]; init=0.0)
        k = hidx(i, 1); b > 0.5 && b < hU[k] && (hU[k] = b)
    end
    global nd = nz_with_hU(nd, hU)
end
hUp, _ = nz_presolve_hU(nd, x̄); nd = nz_with_hU(nd, hUp)
nh, S = nz_nh(nd), nd.S
LB = fr.LB
env = Gurobi.Env()
mk() = (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
P = _nz_build_sep(nd, x̄; optimizer=mk)
DR = _nz_dr_attach!(P, nd, x̄; optimizer=mk, dd=false)
for a in ("FeasibilityTol", "OptimalityTol"); set_optimizer_attribute(P.O.model, a, 1e-7); end
set_optimizer_attribute(P.O.model, "NumericFocus", 0); P.O.model.ext[:lp_tol] = 1e-7
set_optimizer_attribute(P.O.model, "ScaleFlag", parse(Int, envf("SCALE", "2")))   # 저장소 기본 (lp_scale = 2) 과 같게
CL = [_nz_dr_build_corner(nd, x̄, s; optimizer=mk) for s in 1:S]
lo0, hi0 = _nz_dr_global(nd, DR)
v = P.O.v; Hr = v[:Hr]; dvar = v[:d]; αv_ = v[:α]
dl0 = lower_bound.(dvar); du0 = upper_bound.(dvar)
# d 상자 (front_dobbt_bb 와 같음): 시나리오마다 d 경계 행 둘을 RLT 후보로 더 둔다 (처음엔 원래 경계와 같아 영향 없음)
Jlo = zeros(Int, S); Jhi = zeros(Int, S)
for s in 1:S
    push!(P.rows, _NZRow(-dl0[s], [(:d, s, 1.0)], 0)); push!(P.rowscen, s); Jlo[s] = length(P.rows)
    push!(P.rows, _NZRow(du0[s], [(:d, s, -1.0)], 0)); push!(P.rowscen, s); Jhi[s] = length(P.rows)
end
function set_dbox!(dl, du)
    for s in 1:S
        set_lower_bound(dvar[s], dl[s]); set_upper_bound(dvar[s], du[s])
        P.rows[Jlo[s]] = _NZRow(-dl[s], [(:d, s, 1.0)], 0); P.rows[Jhi[s]] = _NZRow(du[s], [(:d, s, -1.0)], 0)
    end
    for n in eachindex(P.allrows)
        i, islo, b, cref, _, j = P.allrows[n]
        (j in Jlo || j in Jhi) || continue
        c0 = P.rows[j].c0
        set_normalized_coefficient(cref, αv_[i], islo ? c0 : -c0)
        rhs = islo ? b * c0 : -b * c0
        P.active[n] && set_normalized_rhs(cref, rhs)
        P.allrows[n] = (i, islo, b, cref, rhs, j)
    end
end
function solve_box!(l, u, lop, hip, phib, eq; rounds=RD)
    _nz_sticky_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
    lo, hi = _nz_dr_corner_ranges(CL, DR, nd, x̄, l, u, lop, hip)
    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=true, eq=eq)
    z = _nz_sticky_solve!(P; maxrounds=rounds)
    return z, (lo, hi)
end
zr = solve_box!(zeros(nh), copy(nd.hU), lo0, hi0, fill(Inf, S), falses(nh))[1]
@printf("--- 전선 %s: g=%d S=%d x=%s, 열린 상자 %d, incumbent %.2f, 상한 %.2f | 루트 다시 풂 %.2f (저장 %.2f)\n",
        envf("FR", ""), fr.g, S, fr.xs, length(fr.boxes), LB, fr.UB, zr, fr.root_UB); flush(stdout)
fmt(z) = isfinite(z) ? @sprintf("%+8.2f", z - LB) : (z < 0 ? "  불가능" : "     Inf")
act = [i for i in 1:nh if nd.hU[i] > 1e-9]

MODE = envf("MODE", "none"); RCR = parse(Int, envf("RCR", "0")); TB = parse(Float64, envf("TB", "90")); CUTM = envf("CUT", "lb")
MODE in ("none", "rc", "rcd") || error("front_rcred: MODE = none | rc | rcd")
cutv = CUTM == "tau" ? parse(Float64, envf("TAU", "nan")) : LB
isfinite(cutv) || error("front_rcred: CUT=tau 면 TAU 필요")
CHECK = envf("CHECK", "0") == "1"
BR = envf("BR", "viol"); SB_TOP = parse(Int, envf("SB_TOP", "2")); BR in ("viol", "strong", "sfirst") || error("front_rcred: BR = viol | strong | sfirst")

"노드 풀기: (상한, 분기 좌표, 쌍대 정보) — 상한 ≤ cut 이면 좌표 0. 쌍대 정보 = (α̂, α reduced cost, d̂, d reduced cost)"
function evalnode(b, l, u, dl, du)
    set_dbox!(dl, du); z = solve_box!(l, u, b.lo, b.hi, b.phib, b.eq)[1]
    (!isfinite(z) || z <= cutv) && return z, 0, nothing
    termination_status(P.O.model) == MOI.OPTIMAL || error("front_rcred: 노드 LP 상태 $(termination_status(P.O.model))")
    αv = value.(αv_); rcα = reduced_cost.(αv_); dv = value.(dvar); rcd = reduced_cost.(dvar)
    αc = clamp.(αv, l, u); rv = value.(v[:r])
    ζLv = value.(v[:ζL]); ζFv = value.(v[:ζF]); ζWv = value.(v[:ζW])
    viol = zeros(nh)
    for i in 1:nh
        u[i] - l[i] > 1e-6 || continue
        viol[i] = sum(abs(ζLv[i, s] - αc[i] * rv[s]) + abs(ζFv[i, s] - αc[i] * dv[s]) for s in 1:S)
        for (jj, kk) in enumerate(Hr)
            nd.hrow[kk] == i || continue
            viol[i] += sum(abs(ζWv[jj, s] - αc[i] * value(v[:ϖ][kk, s])) for s in 1:S)
        end
    end
    return z, (maximum(viol) > 1e-9 ? argmax(viol) : 0), (αv, rcα, dv, rcd, viol)
end
"경계 줄이기: 값 x̂ 가 위 경계에 있고 쌍대 μ = |rc| > 0 이면 아래 경계를 u − 여유/μ 로 (아래 경계에 있으면 반대로). 반환 (l', u', 줄인 좌표 수, 줄인 폭 비율 합)"
function reduce_box(z, xv, rc, l, u; minw=1e-7)
    gap = (z - cutv) * (1 + 1e-6) + 1e-6 * max(1.0, abs(z))      # 허용오차 여유 (노드 상한의 zpad 와 같은 크기)
    l2 = copy(l); u2 = copy(u); nr = 0; fr_ = 0.0
    for k in eachindex(l)
        w = u[k] - l[k]; w > minw || continue
        μ = abs(rc[k]); μ > 1e-9 || continue
        if xv[k] >= u[k] - 1e-9 * max(1.0, w)
            t = u[k] - gap / μ
            t > l[k] && (l2[k] = t; nr += 1; fr_ += (t - l[k]) / w)
        elseif xv[k] <= l[k] + 1e-9 * max(1.0, w)
            t = l[k] + gap / μ
            t < u[k] && (u2[k] = t; nr += 1; fr_ += (u[k] - t) / w)
        end
    end
    return l2, u2, nr, fr_
end
"점검: 뿌리에서 줄인 α 경계 하나 (가장 많이 줄인 좌표) 를 실제로 넣고 다른 행은 그대로 다시 풂 → 값 ≤ cut + 여유 여야 함"
function check_one(z, l, u, l2, u2)
    ks = [k for k in eachindex(l) if l2[k] > l[k] || u2[k] < u[k]]
    isempty(ks) && return nothing
    k = ks[argmax([max(l2[k] - l[k], u[k] - u2[k]) / (u[k] - l[k]) for k in ks])]
    up = l2[k] > l[k]                                   # 위 경계에 있던 좌표 → 아래 경계를 올린 것: 버린 쪽은 α_k ≤ l2[k]
    lb0, ub0 = lower_bound(αv_[k]), upper_bound(αv_[k])
    up ? set_upper_bound(αv_[k], l2[k]) : set_lower_bound(αv_[k], u2[k])
    optimize!(P.O.model); st = termination_status(P.O.model)
    zc = st == MOI.OPTIMAL ? objective_value(P.O.model) : (st == MOI.INFEASIBLE ? -Inf : NaN)
    set_lower_bound(αv_[k], lb0); set_upper_bound(αv_[k], ub0)
    isnan(zc) && error("front_rcred 점검: 상태 $st")
    zc <= cutv + 2e-6 * max(1.0, abs(cutv)) || error(@sprintf("front_rcred 점검 실패: 좌표 %d 버린 쪽 값 %.6f > 문턱 %.6f (z %.4f)", k, zc, cutv, z))
    return (k, zc - cutv)
end

@printf("    모드 %s, RCR %d, 상자당 예산 %.0fs, 절단값 %s = %.2f, 점검 %s\n", MODE, RCR, TB, CUTM, cutv, CHECK); flush(stdout)
PICK = envf("PICK", "top")
cand = if PICK == "top"
    sort(fr.boxes; by=b -> -b.ub)[1:min(NC, end)]
else                                   # spread: 상한 순으로 고르게 (분위수)
    bs = sort(fr.boxes; by=b -> -b.ub); n = length(bs)
    [bs[clamp(round(Int, q * n), 1, n)] for q in range(0.0, 0.95; length=NC)]
end
tot_sb = Ref(0)
nshow = 0; tall = time(); res = Float64[]; tot_nodes = 0; tot_t = 0.0; red_stat = [0, 0, 0.0]   # (줄인 노드, 줄인 좌표, 폭 비율 합)
for b in cand
    global nshow, tot_nodes, tot_t
    nshow >= K && break
    z0, ia0, du_0 = evalnode(b, b.l, b.u, dl0, du0)
    (isfinite(z0) && z0 > LB + 5) || continue
    nshow += 1
    if CHECK && du_0 !== nothing
        l2, u2, nr, _ = reduce_box(z0, du_0[1], du_0[2], b.l, b.u)
        ck = check_one(z0, b.l, b.u, l2, u2)
        ck === nothing ? println("    점검: 줄인 좌표 없음") : @printf("    점검: 좌표 %d 버린 쪽 값 − 문턱 = %.4f (≤ 0 이면 맞음), 줄인 좌표 %d\n", ck[1], ck[2], nr)
        z0, ia0, du_0 = evalnode(b, b.l, b.u, dl0, du0)          # 모델을 노드 상태로 되돌림
    end
    t0 = time(); nodes = 1; nred = 0; nsb = 0
    exact = Float64[]
    open = Any[(z0, ia0, du_0, copy(b.l), copy(b.u), copy(dl0), copy(du0))]
    while !isempty(open) && time() - t0 < TB
        kk = argmax([o[1] for o in open]); z, ia, dq, l, u, dl, du = open[kk]; deleteat!(open, kk)
        if MODE != "none" && dq !== nothing
            for r in 0:RCR
                l, u, nr, f1 = reduce_box(z, dq[1], dq[2], l, u)
                if MODE == "rcd"
                    dl, du, nr2, f2 = reduce_box(z, dq[3], dq[4], dl, du); nr += nr2; f1 += f2
                end
                nr > 0 && (red_stat[1] += 1; red_stat[2] += nr; red_stat[3] += f1)
                (r == RCR || nr == 0) && break
                z, ia, dq = evalnode(b, l, u, dl, du); nodes += 1
                (!isfinite(z) || z <= cutv) && break
            end
            (!isfinite(z) || z <= cutv) && continue
        end
        ia == 0 && (push!(exact, z); continue)
        kids(k) = (tm = 0.5 * (l[k] + u[k]); u1 = copy(u); u1[k] = tm; l2 = copy(l); l2[k] = tm; ((l, u1), (l2, u)))
        solvekid(lc, uc) = all(nd.W * lc .<= nd.wvec .+ 1e-9) ? evalnode(b, lc, uc, dl, du) : (-Inf, 0, nothing)
        res2 = nothing
        if BR != "viol"
            cands = BR == "strong" ? [k for k in act if u[k] - l[k] > 1e-6] : first(sortperm(dq === nothing ? zeros(nh) : dq[5]; rev=true), SB_TOP)
            cands = [k for k in cands if u[k] - l[k] > 1e-6]
            best = -Inf
            for k in cands
                kk2 = kids(k); r1 = solvekid(kk2[1]...); r2 = solvekid(kk2[2]...); nsb += 2
                Δ1 = z - max(r1[1], cutv); Δ2 = z - max(r2[1], cutv)
                sc = max(Δ1, 1e-3) * max(Δ2, 1e-3)
                sc > best && (best = sc; res2 = (kk2, (r1, r2)))
            end
        end
        if res2 === nothing
            kk2 = kids(ia); res2 = (kk2, (solvekid(kk2[1]...), solvekid(kk2[2]...)))
        end
        for ((lc, uc), (zc, ic, dqc)) in zip(res2[1], res2[2])
            nodes += 1
            (isfinite(zc) && zc > cutv) && push!(open, (zc, ic, dqc, copy(lc), copy(uc), copy(dl), copy(du)))
        end
    end
    zt = max(isempty(open) ? -Inf : maximum(o[1] for o in open), isempty(exact) ? -Inf : maximum(exact)); push!(res, isfinite(zt) ? zt - LB : -Inf)
    tb = time() - t0; tot_nodes += nodes; tot_t += tb; tot_sb[] += nsb
    @printf("  상자 %d (저장 상한 %s, 폭 최대 %.1f): 뿌리 %s → %.1fs 뒤 하위 나무 상한 %s (노드 %d, 열린 %d)\n",
            nshow, fmt(b.ub), maximum(b.u .- b.l), fmt(z0), tb, fmt(zt), nodes, length(open)); nsb > 0 && @printf("      strong LP %d
", nsb)
    flush(stdout)
end
@printf("분기 %s: strong LP 합 %d
", BR, tot_sb[])
@printf("끝 (%.0fs): 상자 %d, 노드 합 %d, 시간 합 %.1fs, 줄인 노드 %d (좌표 %d, 폭 비율 평균 %.3f) | 상자별 상한 − incumbent: %s\n",
        time() - tall, nshow, tot_nodes, tot_t, red_stat[1], red_stat[2], red_stat[2] > 0 ? red_stat[3] / red_stat[2] : 0.0,
        join([isfinite(x) ? @sprintf("%.1f", x) : "죽음" for x in res], " "))
