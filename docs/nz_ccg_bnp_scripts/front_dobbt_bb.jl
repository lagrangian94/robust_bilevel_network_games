# front_dobbt_bb.jl — 전선 상자 하나를 뿌리로 같은 시간 예산의 작은 best-first B&B 를 돌려, d 범위 OBBT 를 넣은 판과 α 분기만 한 판의 하위 나무 상한을 비교 (2026-11-27).
#   front_dobbt.jl (OBBT 한 상자 효과) 의 다음 단계. 자식은 부모의 조인 d 범위를 물려받음. 모드 MODE: none (α 분기만), every (노드마다 OBBT OBR 회), root (뿌리에서만 OBBT OBR 회).
# (아래는 front_dobbt.jl 머리말)
#   근거: 상한 오차는 follower 블록의 곱 ζF = α·d 의 McCormick/RLT 오차이고 (md 11.14 (1)), 그 오차는 (α 폭)·(d 폭) 에 비례한다.
#   α-B&B 는 d 범위를 볼 상자 [0.037, 0.163] 그대로 두는데, front_dsplit 에서 d 를 d̂ 에 고정하면 상자가 모두 죽었다 → d 범위가 실제로는 좁을 수 있음.
#   상자마다: (목적 ≥ 절단값) 을 더한 노드 LP 에서 min/max d_s (시나리오마다 LP 2 개) → d 상자를 교집합으로 조여 다시 풂, OBR 번 반복.
#   절단값 = 저장된 incumbent (CUT=lb, 기본) 또는 목표 TAU (CUT=tau). 실행 불가능이면 그 상자는 절단값 아래 → 가지치기 가능.
#   비교: 같은 상자를 B&B 규칙으로 한 번 나눈 α 자식 2 개의 상한 최댓값 (a1).
#  환경 변수: FR (전선 파일), HUCAP 1, K 6 (상자 수), NC 40, ROUNDS 30, OBR 3, CUT lb, TAU (CUT=tau 일 때), RV 0 (1 이면 리더 재가중 r 도 조임)
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Serialization, Statistics
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
envf(k, d) = get(ENV, k, d)
fr = deserialize(envf("FR", ""))
K = parse(Int, envf("K", "6")); NC = parse(Int, envf("NC", "40")); RD = parse(Int, envf("ROUNDS", "30"))
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
CL = [_nz_dr_build_corner(nd, x̄, s; optimizer=mk) for s in 1:S]
lo0, hi0 = _nz_dr_global(nd, DR)
v = P.O.v; Hr = v[:Hr]; dvar = v[:d]
dl0 = lower_bound.(dvar); du0 = upper_bound.(dvar)
# 시나리오마다 d 경계 행 둘을 더 둔다 (처음엔 원래 경계와 같아 영향 없음)
Jlo = zeros(Int, S); Jhi = zeros(Int, S)
for s in 1:S
    push!(P.rows, _NZRow(-dl0[s], [(:d, s, 1.0)], 0)); push!(P.rowscen, s); Jlo[s] = length(P.rows)
    push!(P.rows, _NZRow(du0[s], [(:d, s, -1.0)], 0)); push!(P.rowscen, s); Jhi[s] = length(P.rows)
end
"d 상자를 [dl, du] 로: 변수 경계 + 더 둔 경계 행의 c0 + 이미 만든 RLT 행의 α 계수·rhs"
function set_dbox!(dl, du)
    α = v[:α]
    for s in 1:S
        set_lower_bound(dvar[s], dl[s]); set_upper_bound(dvar[s], du[s])
        P.rows[Jlo[s]] = _NZRow(-dl[s], [(:d, s, 1.0)], 0); P.rows[Jhi[s]] = _NZRow(du[s], [(:d, s, -1.0)], 0)
    end
    for n in eachindex(P.allrows)
        i, islo, b, cref, _, j = P.allrows[n]
        (j in Jlo || j in Jhi) || continue
        c0 = P.rows[j].c0
        set_normalized_coefficient(cref, α[i], islo ? c0 : -c0)
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
@printf("    d 상자 (볼 상자 ±ε̃/√S): dl %s du %s\n", string(round.(dl0; digits=3)), string(round.(du0; digits=3)))
fmt(z) = isfinite(z) ? @sprintf("%+8.2f", z - LB) : (z < 0 ? "  불가능" : "     Inf")
cand = sort(fr.boxes; by=b -> -b.ub)[1:min(NC, end)]
act = [i for i in 1:nh if nd.hU[i] > 1e-9]

MODE = envf("MODE", "none"); OBR = parse(Int, envf("OBR", "1")); TB = parse(Float64, envf("TB", "90")); CUTM = envf("CUT", "lb")
MODE in ("none", "every", "root") || error("front_dobbt_bb: MODE = none | every | root")
cutv = CUTM == "tau" ? parse(Float64, envf("TAU", "nan")) : LB
isfinite(cutv) || error("front_dobbt_bb: CUT=tau 면 TAU 필요")
"(목적 ≥ cut) 아래에서 변수 묶음 xs 의 범위. 반환 (lo, hi, 실행불가능?, LP 수)"
function obbt(xs, lo, hi, cut)
    m = P.O.model; obj = objective_function(m); sen = objective_sense(m)
    cc = @constraint(m, obj >= cut); lo2 = copy(lo); hi2 = copy(hi); dead = false; nlp = 0
    for s in eachindex(xs)
        for (sn, isl) in ((MOI.MIN_SENSE, true), (MOI.MAX_SENSE, false))
            @objective(m, sn, xs[s]); optimize!(m); nlp += 1
            st = termination_status(m)
            if st == MOI.INFEASIBLE || st == MOI.INFEASIBLE_OR_UNBOUNDED
                dead = true; break
            elseif st == MOI.OPTIMAL
                val = value(xs[s])
                isl ? (lo2[s] = max(lo2[s], val - 1e-7)) : (hi2[s] = min(hi2[s], val + 1e-7))
            end
        end
        dead && break
    end
    delete(m, cc); @objective(m, sen, obj)
    return lo2, hi2, dead, nlp
end

"노드 풀기: (상한, 분기 좌표) — 상한 ≤ cut 이면 좌표 0"
function evalnode(b, l, u, dl, du)
    set_dbox!(dl, du); z = solve_box!(l, u, b.lo, b.hi, b.phib, b.eq)[1]
    (!isfinite(z) || z <= cutv) && return z, 0
    αv = clamp.(value.(v[:α]), l, u); rv = value.(v[:r]); dv = value.(dvar)
    ζLv = value.(v[:ζL]); ζFv = value.(v[:ζF]); ζWv = value.(v[:ζW])
    viol = zeros(nh)
    for i in 1:nh
        u[i] - l[i] > 1e-6 || continue
        viol[i] = sum(abs(ζLv[i, s] - αv[i] * rv[s]) + abs(ζFv[i, s] - αv[i] * dv[s]) for s in 1:S)
        for (jj, kk) in enumerate(Hr)
            nd.hrow[kk] == i || continue
            viol[i] += sum(abs(ζWv[jj, s] - αv[i] * value(v[:ϖ][kk, s])) for s in 1:S)
        end
    end
    return z, (maximum(viol) > 1e-9 ? argmax(viol) : 0)
end
"OBBT 를 OBR 회 (각 회 = d 범위 조임 + 다시 풂). 반환 (z, 분기 좌표, dl, du, LP 수)"
function tighten(b, l, u, dl, du, z, ia)
    nlp = 0
    for k in 1:OBR
        dl2, du2, dead, n1 = obbt(dvar, dl, du, cutv); nlp += n1
        dead && return -Inf, 0, dl, du, nlp
        dl, du = dl2, du2
        z, ia = evalnode(b, l, u, dl, du)
        (!isfinite(z) || z <= cutv) && return z, 0, dl, du, nlp
    end
    return z, ia, dl, du, nlp
end
@printf("    모드 %s, OBBT 회수 %d, 상자당 예산 %.0fs, 절단값 %s = %.2f
", MODE, OBR, TB, CUTM, cutv); flush(stdout)
nshow = 0; tall = time(); res = Float64[]
for b in cand
    global nshow
    nshow >= K && break
    z0, ia0 = evalnode(b, b.l, b.u, dl0, du0)
    (isfinite(z0) && z0 > LB + 20) || continue
    nshow += 1
    t0 = time(); nodes = 1; nlp = 0; nob = 0
    exact = Float64[]
    open = Any[(z0, ia0, copy(b.l), copy(b.u), copy(dl0), copy(du0), MODE != "none")]   # (상한, 좌표, l, u, dl, du, OBBT 할지)
    while !isempty(open) && time() - t0 < TB
        kk = argmax([o[1] for o in open]); z, ia, l, u, dl, du, dob = open[kk]; deleteat!(open, kk)
        if dob
            z, ia, dl, du, n1 = tighten(b, l, u, dl, du, z, ia); nlp += n1; nob += 1
            (!isfinite(z) || z <= cutv) && continue
        end
        ia == 0 && (push!(exact, z); continue)      # 나눌 좌표 없음 (완화가 정확) → 그 값은 따로 둠
        tm = 0.5 * (l[ia] + u[ia]); u1 = copy(u); u1[ia] = tm; l2 = copy(l); l2[ia] = tm
        for (lc, uc) in ((l, u1), (l2, u))
            zc, ic = evalnode(b, lc, uc, dl, du); nodes += 1
            (isfinite(zc) && zc > cutv) && push!(open, (zc, ic, copy(lc), copy(uc), copy(dl), copy(du), MODE == "every"))
        end
    end
    zt = max(isempty(open) ? -Inf : maximum(o[1] for o in open), isempty(exact) ? -Inf : maximum(exact)); push!(res, isfinite(zt) ? zt - LB : -Inf)
    @printf("  상자 %d: 뿌리 %s → %.0fs 뒤 하위 나무 상한 %s (노드 %d, 열린 %d, OBBT %d 번 LP %d)
",
            nshow, fmt(z0), time() - t0, fmt(zt), nodes, length(open), nob, nlp)
    flush(stdout)
end
@printf("끝 (%.0fs), 상자별 상한 − incumbent: %s
", time() - tall, join([isfinite(x) ? @sprintf("%.1f", x) : "죽음" for x in res], " "))
