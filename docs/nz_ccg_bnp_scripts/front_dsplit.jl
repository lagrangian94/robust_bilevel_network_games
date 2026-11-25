# 저장된 oracle 전선에서 "belief d 를 나누는 것" 과 "α 를 나누는 것" 이 노드 상한을 얼마나 내리는지 비교 (2026-11-26, α-B&B 처음부터 다시 보기).
#   가설: 전선 상자를 살려 두는 것은 follower 블록의 곱 ζF = α·d 의 McCormick/RLT 오차다. 위반 합은 작지만 (≈1.8) 약정 한 단위의 값이
#   최대 p = 800 이라 follower recourse 이익이 ≈500 부풀고 (front_blame 의 Σ(cȳ − d̂Q)), 그만큼 최적이 아닌 α̂ 가 통과한다.
#   ζF 오차는 (α 폭)·(d 폭) 에 비례하는데 α-B&B 는 α 만 나눈다 → d 폭 (볼 상자 ±ε̃/√S) 은 끝까지 그대로.
#   비교 (상자마다, 30 라운드 분리로 다시 풂):
#     z0      : 그 상자
#     dfix    : d 를 LP 해 d̂ 에 고정 (진단용 제한, 상한 아님) — d 쪽 오차를 없앤 상한의 한계
#     d1, d2  : ζF 위반이 큰 시나리오의 d_s 를 d̂_s 에서 나눔 (1 단계 2 자식, 2 단계 4 자식) → 자식 상한의 최댓값
#     a1, a2  : B&B 와 같은 규칙 (곱 위반 합이 큰 좌표, 중점) 으로 α 를 나눔 → 자식 상한의 최댓값
#   d 범위는 _nz_rows 의 경계 행 (d_s − dl ≥ 0, du − d_s ≥ 0) 을 시나리오마다 하나씩 더 두고 c0 와 α 계수를 바꿔 넣는다 (분리가 RLT 곱을 만든다).
#  환경 변수: FR (전선 파일), HUCAP 1, K 6, NC 40, ROUNDS 30
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
nshow = 0
tall = time()
for b in cand
    global nshow
    nshow >= K && break
    set_dbox!(dl0, du0)
    z0, rg = solve_box!(b.l, b.u, b.lo, b.hi, b.phib, b.eq)
    (isfinite(z0) && z0 > LB + 20) || continue          # 30 라운드로 다시 풀어도 살아 있는 상자만
    nshow += 1
    lo, hi = rg
    α̂ = clamp.(value.(v[:α]), b.l, b.u); d̂ = value.(dvar); ζF = value.(v[:ζF])
    vs = [sum(abs(ζF[i, s] - α̂[i] * d̂[s]) for i in 1:nh) for s in 1:S]
    # dfix: d = d̂ 고정 (진단)
    set_dbox!(max.(dl0, d̂ .- 1e-9), min.(du0, d̂ .+ 1e-9)); zf = solve_box!(b.l, b.u, b.lo, b.hi, b.phib, b.eq)[1]
    # d 나누기: 위반 큰 시나리오에서 d̂ 로 (끝에 너무 가까우면 중점)
    function dsplit(dl, du, s)
        t = d̂[s]; w = du[s] - dl[s]
        (t - dl[s] < 0.1w || du[s] - t < 0.1w) && (t = 0.5 * (dl[s] + du[s]))
        du1 = copy(du); du1[s] = t; dl2 = copy(dl); dl2[s] = t
        return ((dl, du1), (dl2, du))
    end
    ord = sortperm(vs; rev=true)
    t1 = time()
    kids1 = dsplit(dl0, du0, ord[1])
    z1 = [(set_dbox!(k[1], k[2]); solve_box!(b.l, b.u, b.lo, b.hi, b.phib, b.eq)[1]) for k in kids1]
    kids2 = [kk for k in kids1 for kk in dsplit(k[1], k[2], ord[2])]
    z2 = [(set_dbox!(k[1], k[2]); solve_box!(b.l, b.u, b.lo, b.hi, b.phib, b.eq)[1]) for k in kids2]
    td = time() - t1
    set_dbox!(dl0, du0)
    # α 나누기: B&B 의 viol 규칙 (ζL + ζF + ζW 위반 합이 큰 좌표, 중점)
    function asplit(l, u)
        z, _ = solve_box!(l, u, b.lo, b.hi, b.phib, b.eq)
        αv = clamp.(value.(v[:α]), l, u); rv = value.(v[:r]); dv = value.(dvar)
        ζLv = value.(v[:ζL]); ζFv = value.(v[:ζF]); ζWv = value.(v[:ζW])
        viol = zeros(nh)
        for i in 1:nh
            u[i] - l[i] > 1e-6 || continue
            viol[i] = sum(abs(ζLv[i, s] - αv[i] * rv[s]) + abs(ζFv[i, s] - αv[i] * dv[s]) for s in 1:S)
            for (jj, k) in enumerate(Hr)
                nd.hrow[k] == i || continue
                viol[i] += sum(abs(ζWv[jj, s] - αv[i] * value(v[:ϖ][k, s])) for s in 1:S)
            end
        end
        i = argmax(viol); t = 0.5 * (l[i] + u[i])
        u1 = copy(u); u1[i] = t; l2 = copy(l); l2[i] = t
        return ((copy(l), u1), (l2, copy(u))), i
    end
    t2 = time()
    akids1, ia = asplit(b.l, b.u)
    za1 = [solve_box!(k[1], k[2], b.lo, b.hi, b.phib, b.eq)[1] for k in akids1]
    akids2 = [kk for k in akids1 for kk in asplit(k[1], k[2])[1]]
    za2 = [solve_box!(k[1], k[2], b.lo, b.hi, b.phib, b.eq)[1] for k in akids2]
    ta = time() - t2
    @printf("  상자 %d: z0 %s | dfix %s | d1 %s d2 %s (%.0fs, 시나리오 %d,%d, ζF 위반 %.2f,%.2f / 합 %.2f) | a1 %s a2 %s (%.0fs, 좌표 %d) | 폭 %s\n",
            nshow, fmt(z0), fmt(zf), fmt(maximum(z1)), fmt(maximum(z2)), td, ord[1], ord[2], vs[ord[1]], vs[ord[2]], sum(vs),
            fmt(maximum(za1)), fmt(maximum(za2)), ta, ia, string(round.((b.u .- b.l)[act]; digits=2)))
    @printf("      d 자식 2 단계: %s | α 자식 2 단계: %s\n", join(fmt.(z2), " "), join(fmt.(za2), " "))
    flush(stdout)
end
@printf("끝 (%.0fs)\n", time() - tall)
