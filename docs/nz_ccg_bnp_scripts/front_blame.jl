# 저장된 oracle 전선 (orcl_replay.jl 의 orep_front_*.jls) 에서 노드 완화 해가 어디서 참 문제를 벗어나는지 (2026-11-25 밤, α-B&B 튜닝 진단).
#   nonzero_sum/lab_blame.jl 의 loc_tkp 판. B&B 와 같게 꼭짓점 LP 범위 (range_lp) + sticky 분리 (30 라운드, 수렴) + 상자의 상보성 분기 기록 (eq).
#   z     : 노드 LP 상한          zE  : α̂ 고정 정확값 (belief 자유) = 참 G(α̂)
#   zEd   : α̂ 와 follower belief d̂ 를 함께 고정한 정확값. 실행 불가능이면 α̂ 가 d̂ 에서 follower 최적이 아님 → follower 쪽 (F) 완화 오차
#   dist  : α̂ 를 follower 최적으로 만드는 belief 중 d̂ 에 가장 가까운 것까지 L1 거리
#   곱 위반, 리더 블록 Σ(ℓŷ − r̂φ(α̂)), follower 블록 Σ(cȳ − d̂Q(α̂))
#  환경 변수: FR (전선 파일), HUCAP 1, K 8, NC 32 (상한 큰 순서 후보, 다시 풀어 상위 K), ROUNDS 30
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Serialization, Statistics
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
envf(k, d) = get(ENV, k, d)
fr = deserialize(envf("FR", ""))
K = parse(Int, envf("K", "8")); NC = parse(Int, envf("NC", "32")); RD = parse(Int, envf("ROUNDS", "30"))
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
nh, S, m, ny = nz_nh(nd), nd.S, nz_m(nd), nz_ny(nd)
LB = fr.LB; sc = max(1.0, abs(LB))
env = Gurobi.Env()
mk() = (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
P = _nz_build_sep(nd, x̄; optimizer=mk)
DR = _nz_dr_attach!(P, nd, x̄; optimizer=mk, dd=false)
for a in ("FeasibilityTol", "OptimalityTol"); set_optimizer_attribute(P.O.model, a, 1e-7); end
set_optimizer_attribute(P.O.model, "NumericFocus", 0); P.O.model.ext[:lp_tol] = 1e-7
CL = [_nz_dr_build_corner(nd, x̄, s; optimizer=mk) for s in 1:S]
F = nz_build_fixed_alpha(nd, x̄; optimizer=mk)
lo0, hi0 = _nz_dr_global(nd, DR)
v = P.O.v; Hr = v[:Hr]
function solve_box!(l, u, lop, hip, phib, eq; rounds=RD)
    _nz_sticky_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
    lo, hi = _nz_dr_corner_ranges(CL, DR, nd, x̄, l, u, lop, hip)
    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=true, eq=eq)
    z = _nz_sticky_solve!(P; maxrounds=rounds)
    return z, (lo, hi)
end
t0 = time(); zr = solve_box!(zeros(nh), copy(nd.hU), lo0, hi0, fill(Inf, S), falses(nh))[1]
@printf("--- 전선 %s: g=%d S=%d x=%s, 열린 상자 %d, incumbent %.2f, 상한 %.2f | 루트 다시 풂 %.2f (저장 %.2f, %.0fs)\n",
        envf("FR", ""), fr.g, S, fr.xs, length(fr.boxes), LB, fr.UB, zr, fr.root_UB, time() - t0); flush(stdout)
cand = sort(fr.boxes; by=b -> -b.ub)[1:min(NC, end)]
zs = [solve_box!(b.l, b.u, b.lo, b.hi, b.phib, b.eq)[1] for b in cand]
@printf("    후보 %d 개: 저장 상한 − inc %.2f ~ %.2f, 다시 푼 (%d 라운드) 상한 − inc %.2f ~ %.2f\n", length(cand),
        minimum(b.ub for b in cand) - LB, maximum(b.ub for b in cand) - LB, RD, minimum(zs) - LB, maximum(zs) - LB); flush(stdout)
ordz = sortperm(zs; rev=true)[1:min(K, end)]
"시나리오 s 의 recourse 값 Q_s(α) 와 pessimistic 리더 평가 φ_s(α)"
function true_vals(α)
    Q = zeros(S); φ = zeros(S)
    for s in 1:S
        rr = nz_rhs(nd, x̄, α, s)
        mdl = Model(mk); @variable(mdl, y[1:ny] >= 0)
        @constraint(mdl, [k = 1:m], sum(nd.A[k, j] * y[j] for j in 1:ny if nd.A[k, j] != 0) <= rr[k])
        @objective(mdl, Max, dot(nd.c, y)); optimize!(mdl)
        termination_status(mdl) == MOI.OPTIMAL || return fill(NaN, S), fill(NaN, S)
        Q[s] = objective_value(mdl)
        @constraint(mdl, dot(nd.c, y) >= Q[s] - 1e-7 * max(1, abs(Q[s])))
        @objective(mdl, Max, dot(nd.ell, y)); optimize!(mdl); φ[s] = objective_value(mdl)
    end
    return Q, φ
end
act = [i for i in 1:nh if nd.hU[i] > 1e-9]
@printf("    활성 좌표 %s (hᵁ %s), incumbent α %s\n", string(act), string(round.(nd.hU[act]; digits=1)), string(round.(fr.α[act]; digits=2)))
for n in ordz
    b = cand[n]
    z, rg = solve_box!(b.l, b.u, b.lo, b.hi, b.phib, b.eq)
    lo, hi = rg
    α̂ = clamp.(value.(v[:α]), b.l, b.u); r̂ = value.(v[:r]); d̂ = value.(v[:d])
    vL = sum(abs(value(v[:ζL][i, s]) - α̂[i] * r̂[s]) for i in 1:nh, s in 1:S)
    vF = sum(abs(value(v[:ζF][i, s]) - α̂[i] * d̂[s]) for i in 1:nh, s in 1:S)
    vW = sum(abs(value(v[:ζW][jj, s]) - α̂[nd.hrow[k]] * value(v[:ϖ][k, s])) for (jj, k) in enumerate(Hr), s in 1:S)
    vP = isempty(DR.zP) ? NaN : sum(abs(value(DR.zP[a, s]) - α̂[nd.hrow[k]] * value(v[:π̃][k, s])) for (a, k) in enumerate(DR.Hk), s in 1:S)
    ℓŷ = [sum(nd.ell[j] * value(v[:ŷ][j, s]) for j in 1:ny) for s in 1:S]
    cŷ = [sum(nd.c[j] * value(v[:ŷ][j, s]) for j in 1:ny) for s in 1:S]
    cȳ = [sum(nd.c[j] * value(v[:ȳ][j, s]) for j in 1:ny) for s in 1:S]
    Q, φ = true_vals(α̂)
    zE = nz_eval_alpha!(F, nd, α̂)
    dv = F.v[:d]; lbs = lower_bound.(dv); ubs = upper_bound.(dv)
    for s in 1:S; set_lower_bound(dv[s], max(lbs[s], d̂[s] - 1e-9)); set_upper_bound(dv[s], min(ubs[s], d̂[s] + 1e-9)); end
    zEd = nz_eval_alpha!(F, nd, α̂)
    for s in 1:S; set_lower_bound(dv[s], lbs[s]); set_upper_bound(dv[s], ubs[s]); end
    dist = NaN
    if zE > -Inf
        obj0 = objective_function(F.model)
        dev = @variable(F.model, [1:S], lower_bound = 0.0)
        c1 = @constraint(F.model, [s = 1:S], dev[s] >= dv[s] - d̂[s]); c2 = @constraint(F.model, [s = 1:S], dev[s] >= d̂[s] - dv[s])
        @objective(F.model, Min, sum(dev)); optimize!(F.model)
        dist = termination_status(F.model) == MOI.OPTIMAL ? objective_value(F.model) : NaN
        delete(F.model, c1); delete(F.model, c2); delete(F.model, dev)
        set_objective(F.model, MAX_SENSE, obj0)
    end
    eL = sum(ℓŷ[s] - r̂[s] * φ[s] for s in 1:S); eQL = sum(cŷ[s] - r̂[s] * Q[s] for s in 1:S); eQF = sum(cȳ[s] - d̂[s] * Q[s] for s in 1:S)
    nun = count(hi .- lo .> 1e-4)
    @printf("  상자: z−inc %+.2f (저장 %+.2f) | G(α̂)−inc %s | d̂ 고정 %s | d 거리 %.3f | 안 고정 쌍대 %d/%d | 상보성 분기 %d | α̂ %s | 폭 %s\n",
            z - LB, b.ub - LB, isfinite(zE) ? @sprintf("%+.2f", zE - LB) : "실행 불가능", isfinite(zEd) ? @sprintf("%+.2f", zEd - LB) : "실행 불가능",
            dist, nun, length(lo), count(b.eq), string(round.(α̂[act]; digits=2)), string(round.((b.u .- b.l)[act]; digits=2)))
    @printf("      곱 위반 ζL %.3f ζF %.3f ζW %.1f ζΠ %.1f | 리더 블록 Σ(ℓŷ − r̂φ) %+.2f, Σ(cŷ − r̂Q) %+.2f | follower 블록 Σ(cȳ − d̂Q) %+.2f | r̂ %s | d̂ %s\n",
            vL, vF, vW, vP, eL, eQL, eQF, string(round.(r̂; digits=3)), string(round.(d̂; digits=3)))
    flush(stdout)
end
