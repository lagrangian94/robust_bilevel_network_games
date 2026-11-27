# front_dd_inc.jl — 선별 DD 를 몇 개씩 나눠 인증하고 노드 LP 를 다시 풀어, 가지치기되면 바로 멈추면 KKT-MIP 가 얼마나 줄어드는지 (2026-11-27, md 11.24).
#   근거: 지리 5 막힌 점 300 s DD 판 (logs/orep_g5_ddt300.log) 에서 DD 받은 노드 136 개 중 126 개가 DD 뒤 다시 푼 LP 로 가지쳐짐.
#   DD 비용의 78% 가 KKT-MIP (노드당 36 개, 하나 0.29 s), 선별 면 LP 는 8% → 가지치기에 충분한 MIP 만 풀면 비용이 크게 줄 수 있음.
#  모드 MODE:
#   full  지금 패치와 같음 — 꼭짓점 선별을 통과한 (쌍, s) 전부를 MIP 2 개씩, 그 뒤 한 번 다시 풂
#   incR  선별 후보를 "두 꼭짓점 차이 폭 / 함의 폭" 이 작은 순서로 B 개씩 MIP → 다시 풂 → 가지치기면 멈춤
#   incV  같은 방식, 순서는 지금 노드 LP 해의 쌍대 차이가 꼭짓점 구간에서 벗어난 양 (배치마다 다시 계산)
#   incV1 incV 와 같되, 배치마다 LP 해가 벗어난 쪽 MIP 하나만 (반대쪽은 그 LP 해를 자르지 못함)
#  MIP 는 패치와 같게 꼭짓점 LP 범위를 H 행 쌍대 경계로 넣고 (dd_sigbd), 실행 불가능이면 경계를 빼고 다시 풀어 확인. 시간 상한 2 s.
#  문턱 = incumbent + 1e-3·|incumbent| (oracle 재현의 rel_gap 1e-3 과 같음)
#  환경 변수: FR (전선 파일), HUCAP 1, K 8, PICK top | spread, NC 40, ROUNDS 30, MODE, B 2
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Serialization, Statistics
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
include(joinpath(root, "docs", "nz_ccg_bnp_scripts", "abb_dd_patch.jl"))    # _abb_sigbd! (dd_sigbd 경계)
envf(k, d) = get(ENV, k, d)
fr = deserialize(envf("FR", ""))
K = parse(Int, envf("K", "8")); NC = parse(Int, envf("NC", "40")); RD = parse(Int, envf("ROUNDS", "30"))
MODE = envf("MODE", "full"); B = parse(Int, envf("B", "2"))
MODE in ("full", "incR", "incV", "incV1") || error("front_dd_inc: MODE = full | incR | incV | incV1")
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
nh, S, m = nz_nh(nd), nd.S, nz_m(nd)
LB = fr.LB; THR = LB + 1e-3 * max(1.0, abs(LB))
env = Gurobi.Env()
mk() = (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
P = _nz_build_sep(nd, x̄; optimizer=mk)
DR = _nz_dr_attach!(P, nd, x̄; optimizer=mk, dd=true)
for a in ("FeasibilityTol", "OptimalityTol"); set_optimizer_attribute(P.O.model, a, 1e-7); end
set_optimizer_attribute(P.O.model, "NumericFocus", 0); set_optimizer_attribute(P.O.model, "ScaleFlag", 2); P.O.model.ext[:lp_tol] = 1e-7
CL = [_nz_dr_build_corner(nd, x̄, s; optimizer=mk) for s in 1:S]
lo0, hi0 = _nz_dr_global(nd, DR)
v = P.O.v
zpad(z) = isfinite(z) ? z + 1e-6 * max(1.0, abs(z)) : z
function solve_box!(l, u, lop, hip, phib, eq; rounds=RD)
    _nz_sticky_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
    lo, hi = _nz_dr_corner_ranges(CL, DR, nd, x̄, l, u, lop, hip)
    dlo, dhi = _nz_dr_dglobal(nd, DR)
    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=true, eq=eq, dlo=dlo, dhi=dhi)
    return zpad(_nz_sticky_solve!(P; maxrounds=rounds)), lo, hi, dlo, dhi
end
"DD 행만 바꿔 다시 풂 (나무의 패치와 같게 child_rounds = 1)"
function resolve!(l, u, lo, hi, phib, eq, dlo, dhi)
    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=true, eq=eq, dlo=dlo, dhi=dhi)
    return zpad(_nz_sticky_solve!(P; maxrounds=1))
end
"꼭짓점 αc 에서 시나리오 s 의 최적 쌍대 면 위 차이의 [최소, 최대] (패치 _abb_dd_screen! 와 같은 LP)"
function corner_diff(s, αc, ps)
    C = CL[s]; rr = nz_rhs(nd, x̄, αc, s)
    for k in 1:m; nd.hrow[k] > 0 && set_normalized_coefficient(C.opt, C.σ[k], rr[k]); end
    set_normalized_rhs(C.opt, 1e20)
    @objective(C.model, Min, sum(rr[k] * C.σ[k] for k in 1:m if rr[k] != 0)); optimize!(C.model)
    termination_status(C.model) == MOI.OPTIMAL || error("corner_diff: 꼭짓점 LP $(termination_status(C.model))")
    zc = objective_value(C.model); set_normalized_rhs(C.opt, zc + 1e-7 * max(1.0, abs(zc)))
    out = Dict{Int,Tuple{Float64,Float64}}()
    for p in ps
        a, b2 = DR.pairs[p]; e = C.σ[DR.Hk[a]] - C.σ[DR.Hk[b2]]
        @objective(C.model, Min, e); optimize!(C.model); mn = objective_value(C.model)
        @objective(C.model, Max, e); optimize!(C.model); mx = objective_value(C.model)
        out[p] = (mn, mx)
    end
    out
end
"(쌍 p, 시나리오 s) 의 차이 범위 한쪽 (ismax) 을 KKT-MIP 로. 반환 :empty | :ok | :fail, 쓴 MIP 수"
function mip_side!(dlo, dhi, p, s, l, u, lo, hi, ismax)
    M = DR.mips[s]; a, b2 = DR.pairs[p]
    for i in 1:nh; set_lower_bound(M.α[i], l[i]); set_upper_bound(M.α[i], u[i]); end
    _abb_sigbd!(M, DR, nd, lo, hi, s)
    @objective(M.model, ismax ? MAX_SENSE : MIN_SENSE, M.σ[DR.Hk[a]] - M.σ[DR.Hk[b2]])
    set_time_limit_sec(M.model, 2.0); n = 1
    getbd(st) = st == MOI.OPTIMAL ? objective_bound(M.model) : (st == MOI.TIME_LIMIT ? _nz_dr_bound_or_nan(M.model) : NaN)
    ok = _nz_dr_opt!(DR, M, l, u, "inc s=$s"); st = termination_status(M.model); bd = ok ? getbd(st) : NaN
    if ok && st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
        _abb_sigbd!(M, DR, nd, nothing, nothing, s); n += 1
        ok = _nz_dr_opt!(DR, M, l, u, "inc 재확인 s=$s"); st = termination_status(M.model); bd = ok ? getbd(st) : NaN
    end
    _abb_sigbd!(M, DR, nd, nothing, nothing, s); set_time_limit_sec(M.model, nothing)   # 결과를 읽은 뒤에 되돌림 (모델을 바꾸면 결과가 사라짐)
    ok || return :fail, n
    st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED) && return :empty, n
    isfinite(bd) || return :fail, n
    ismax ? (bd + 1e-6 < dhi[p, s] && (dhi[p, s] = bd + 1e-6)) : (bd - 1e-6 > dlo[p, s] && (dlo[p, s] = bd - 1e-6))
    dhi[p, s] < dlo[p, s] && (dhi[p, s] = dlo[p, s] = 0.5 * (dlo[p, s] + dhi[p, s]))
    return :ok, n
end
dist(x, a, b) = x < a ? a - x : (x > b ? x - b : 0.0)
"지금 LP 해의 (쌍, s) 쌍대 차이가 꼭짓점 구간 [c1, c2] 에서 벗어난 양 (× d̂_s) 과 쪽 (+1 = 위로 벗어남 → max MIP 가 자름)"
function lpviol(p, s, c1, c2)
    a, b2 = DR.pairs[p]; ka, kb = DR.Hk[a], DR.Hk[b2]; ds = value(v[:d][s])
    ds > 1e-7 || return 0.0, 0
    δ = (value(v[:π̃][ka, s]) - value(v[:π̃][kb, s])) / ds
    return dist(δ, c1, c2) * ds, (δ > c2 ? 1 : (δ < c1 ? -1 : 0))
end

PICK = envf("PICK", "top")
bs = sort(fr.boxes; by=b -> -b.ub); n_ = length(bs)
cand = PICK == "top" ? bs[1:min(NC, end)] : [bs[clamp(round(Int, q * n_), 1, n_)] for q in range(0.0, 0.95; length=NC)]
@printf("--- 선별 DD 나눠 인증 %s: g=%d S=%d x=%s, 열린 상자 %d, incumbent %.2f, 문턱 %.2f | 모드 %s, 배치 %d, 고르기 %s\n",
        envf("FR", ""), fr.g, S, fr.xs, length(fr.boxes), LB, THR, MODE, B, PICK); flush(stdout)
tot = Dict(:mip => 0, :lp => 0, :t => 0.0, :tscr => 0.0, :pruned => 0, :n => 0)
for b in cand
    tot[:n] >= K && break
    z0, lo, hi, dlo, dhi = solve_box!(b.l, b.u, b.lo, b.hi, b.phib, b.eq)
    (isfinite(z0) && z0 > THR) || continue
    maximum((b.u[i] - b.l[i] for i in 1:nh if nd.hU[i] > 1e-9); init=0.0) <= 25.0 || continue   # 패치의 dd_maxw
    tot[:n] += 1
    t0 = time()
    # 선별 (패치와 같음): 두 성분이 안 고정이고 함의 폭 > 1 인 (쌍, s) 중 두 꼭짓점의 차이가 함의 폭의 절반 안에 모인 것
    cands = Tuple{Float64,Int,Int,Float64,Float64}[]          # (모임 비율, p, s, 꼭짓점 구간)
    for s in 1:S
        ps = Int[]
        for (p, (a, b2)) in enumerate(DR.pairs)
            (hi[a, s] - lo[a, s] > 1e-4 && hi[b2, s] - lo[b2, s] > 1e-4) || continue
            wl = max(dlo[p, s], lo[a, s] - hi[b2, s]); wh = min(dhi[p, s], hi[a, s] - lo[b2, s])
            wh - wl > 1.0 && push!(ps, p)
        end
        isempty(ps) && continue
        cl = corner_diff(s, b.l, ps); cu = corner_diff(s, b.u, ps)
        for p in ps
            a, b2 = DR.pairs[p]
            wl = max(dlo[p, s], lo[a, s] - hi[b2, s]); wh = min(dhi[p, s], hi[a, s] - lo[b2, s])
            c1 = min(cl[p][1], cu[p][1]); c2 = max(cl[p][2], cu[p][2])
            c2 - c1 < 0.5 * (wh - wl) && push!(cands, ((c2 - c1) / (wh - wl), p, s, c1, c2))
        end
    end
    tscr = time() - t0
    # 선별 LP 가 노드 LP 모델을 건드리지 않으므로 노드 LP 해 (v) 는 그대로
    nmip = 0; nlp = 0; z = z0; hist = [z0]; empty = false
    if MODE == "full"
        for (_, p, s, _, _) in cands, side in (false, true)
            r, k = mip_side!(dlo, dhi, p, s, b.l, b.u, lo, hi, side); nmip += k
            r === :empty && (empty = true; break)
        end
        z = empty ? -Inf : resolve!(b.l, b.u, lo, hi, b.phib, b.eq, dlo, dhi); nlp += 1; push!(hist, z)
    else
        left = copy(cands)
        MODE == "incR" && sort!(left; by=c -> c[1])
        while !isempty(left) && z > THR && !empty
            if MODE != "incR"                                   # LP 해 기준으로 다시 줄 세움
                sc = [lpviol(c[2], c[3], c[4], c[5]) for c in left]
                ord = sortperm([x[1] for x in sc]; rev=true); left = left[ord]; sc = sc[ord]
                sc[1][1] <= 1e-6 && MODE == "incV1" && break   # 지금 LP 해를 자를 후보 없음
            end
            batch = left[1:min(B, end)]; left = left[(length(batch)+1):end]
            for c in batch
                _, p, s, c1, c2 = c
                sides = (false, true)
                if MODE == "incV1"
                    _, sd = lpviol(p, s, c1, c2); sd == 0 && continue
                    sides = (sd > 0,)
                end
                for side in sides
                    r, k = mip_side!(dlo, dhi, p, s, b.l, b.u, lo, hi, side); nmip += k
                    r === :empty && (empty = true; break)
                end
                empty && break
            end
            z = empty ? -Inf : resolve!(b.l, b.u, lo, hi, b.phib, b.eq, dlo, dhi); nlp += 1; push!(hist, z)
        end
    end
    tb = time() - t0
    pr = z <= THR
    tot[:mip] += nmip; tot[:lp] += nlp; tot[:t] += tb; tot[:tscr] += tscr; tot[:pruned] += pr
    @printf("  상자 (폭 최대 %4.1f): 상한 − 문턱 %s | 후보 %2d, MIP %3d, 다시 풂 %2d, %5.1fs (선별 %.1fs) %s\n", maximum(b.u .- b.l),
            join([isfinite(h) ? @sprintf("%+.1f", h - THR) : "-Inf" for h in first(hist, 8)], " → "), length(cands), nmip, nlp, tb, tscr, pr ? "가지침" : "남음")
    flush(stdout)
end
@printf("RESULT 모드 %s 배치 %d: 상자 %d 개 중 가지침 %d, MIP 합 %d, 다시 풂 %d, 시간 %.1fs (선별 %.1fs)\n",
        MODE, B, tot[:n], tot[:pruned], tot[:mip], tot[:lp], tot[:t], tot[:tscr])
