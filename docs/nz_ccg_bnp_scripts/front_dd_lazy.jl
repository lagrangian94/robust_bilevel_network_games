# front_dd_lazy.jl — 상자 DD 를 전부 풀지 않고 노드 LP 해에서 의심스러운 (쌍, 시나리오) 만 골라 KKT-MIP 로 조이는 분리 방식 (lazy DD) 시험.
#   점수 = LP 의 쌍대 차이 (π̃_a − π̃_b)/d̂_s 가 α̂ 에서의 실제 쌍대 (꼭짓점 LP, 상자 [α̂, α̂]) 가 허용하는 차이 구간에서 벗어난 양 × d̂_s (리더 블록은 ϖ, r̂).
#   라운드마다 점수 상위 M 개 (쌍, s) 의 차이 범위를 MIP 2 개씩으로 구해 넣고 다시 풂. 환경 변수 M 8, R 6 (라운드).
# (아래는 front_dd.jl 머리말) # front_dd.jl — 저장된 oracle 전선에서 쌍대 차이 범위 (DD) 를 넣으면 노드 상한이 얼마나 내려가는지 (2026-11-25 밤, α-B&B 튜닝 진단). front_blame.jl 에서 시작.
#   기본 판 (꼭짓점 LP 범위) 은 DD 를 못 씀 (range_lp = true 는 dd = false). 상자마다 (a) 기본, (b) 상자 DD (KKT-MIP), (c) 루트 DD (GDD=1, 전체 정의역에서 한 번) 를 비교.
# (아래는 front_blame.jl 머리말)
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
DR = _nz_dr_attach!(P, nd, x̄; optimizer=mk, dd=true)
for a in ("FeasibilityTol", "OptimalityTol"); set_optimizer_attribute(P.O.model, a, 1e-7); end
set_optimizer_attribute(P.O.model, "NumericFocus", 0); P.O.model.ext[:lp_tol] = 1e-7
CL = [_nz_dr_build_corner(nd, x̄, s; optimizer=mk) for s in 1:S]
F = nz_build_fixed_alpha(nd, x̄; optimizer=mk)
lo0, hi0 = _nz_dr_global(nd, DR)
v = P.O.v; Hr = v[:Hr]
function solve_box!(l, u, lop, hip, phib, eq; rounds=RD, dlo=nothing, dhi=nothing)
    _nz_sticky_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
    lo, hi = _nz_dr_corner_ranges(CL, DR, nd, x̄, l, u, lop, hip)
    dlo === nothing && ((dlo, dhi) = _nz_dr_dglobal(nd, DR))
    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=true, eq=eq, dlo=dlo, dhi=dhi)
    z = _nz_sticky_solve!(P; maxrounds=rounds)
    z == -Inf && (z = _nz_dr_recheck_infeasible!(P.O.model))   # B&B 와 같게: 실행 불가능은 처음부터 다시 풀어 재확인
    return z, (lo, hi)
end


TGT = parse(Float64, envf("TGT", "-3151.60")); MM = parse(Int, envf("M", "8")); RR = parse(Int, envf("R", "6"))
t0 = time(); zr = solve_box!(zeros(nh), copy(nd.hU), lo0, hi0, fill(Inf, S), falses(nh))[1]
@printf("--- lazy DD 시험 %s: g=%d S=%d x=%s, 열린 상자 %d, incumbent %.2f, 목표 %.2f | 루트 %.2f | 쌍 %d × 시나리오 %d, 라운드당 %d 개, 최대 %d 라운드\n",
        envf("FR", ""), fr.g, S, fr.xs, length(fr.boxes), LB, TGT, zr, length(DR.pairs), S, MM, RR); flush(stdout)
cand = sort(fr.boxes; by=b -> -b.ub)[1:min(NC, end)]
zs = [solve_box!(b.l, b.u, b.lo, b.hi, b.phib, b.eq)[1] for b in cand]
live = [n for n in eachindex(cand) if isfinite(zs[n]) && zs[n] > TGT]
ordz = live[sortperm(zs[live]; rev=true)][1:min(K, end)]
@printf("    후보 %d 개 중 자기 LP 상한 > 목표 %d 개, 위에서 %d 개\n", length(cand), length(live), length(ordz)); flush(stdout)
"(쌍 p, 시나리오 s) 의 차이 범위를 상자 [l, u] 에서 KKT-MIP 2 개로 (_nz_dr_dranges! 와 같은 여유)"
function dd_pair!(dlo, dhi, p, s, l, u)
    M = DR.mips[s]; a, b2 = DR.pairs[p]; ka, kb = DR.Hk[a], DR.Hk[b2]
    for i in 1:nh; set_lower_bound(M.α[i], l[i]); set_upper_bound(M.α[i], u[i]); end
    for ismax in (false, true)
        @objective(M.model, ismax ? MAX_SENSE : MIN_SENSE, M.σ[ka] - M.σ[kb])
        _nz_dr_opt!(DR, M, l, u, "lazy s=$s 쌍 ($ka, $kb)") || continue
        st = termination_status(M.model)
        st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED) && return false      # 상자 안에 실행 가능한 α 없음
        st == MOI.OPTIMAL || continue
        bd = objective_bound(M.model)
        ismax ? (dhi[p, s] = min(dhi[p, s], bd + 1e-6)) : (dlo[p, s] = max(dlo[p, s], bd - 1e-6))
    end
    dhi[p, s] < dlo[p, s] && (dhi[p, s] = dlo[p, s] = 0.5 * (dlo[p, s] + dhi[p, s]))
    return true
end
dist(x, a, b) = x < a ? a - x : (x > b ? x - b : 0.0)
SCREEN = envf("SCREEN", "lp")
"꼭짓점 αc 에서 시나리오 s 의 최적 쌍대 면 위 차이 σ_a − σ_b 의 [최소, 최대] (쌍 목록 ps)"
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
tot = Dict(:mip => 0, :t => 0.0, :pruned => 0)
for n in ordz
    b = cand[n]
    z, (lo, hi) = solve_box!(b.l, b.u, b.lo, b.hi, b.phib, b.eq)
    dlo, dhi = _nz_dr_dglobal(nd, DR)
    for (p, (a, b2)) in enumerate(DR.pairs), s in 1:S     # 성분 범위가 함의하는 차이 (B&B 의 _nz_dr_dranges! 처음과 같음)
        dlo[p, s] = max(dlo[p, s], lo[a, s] - hi[b2, s]); dhi[p, s] = min(dhi[p, s], hi[a, s] - lo[b2, s])
    end
    done = Set{Tuple{Int,Int}}(); hist = [z]; nmip = 0
    if SCREEN == "corner"   # 두 꼭짓점의 차이 구간이 같은 상수 근처이고 성분 범위가 함의하는 폭은 넓은 (쌍, s) 만 MIP 로 인증
        tb = @elapsed begin
            ncand = 0; empty_box = false
            for s_ in 1:S
                ps = [p for (p, (a, b2)) in enumerate(DR.pairs) if hi[a, s_] - lo[a, s_] > 1e-4 && hi[b2, s_] - lo[b2, s_] > 1e-4 && dhi[p, s_] - dlo[p, s_] > 1.0]
                isempty(ps) && continue
                cl = corner_diff(s_, b.l, ps); cu = corner_diff(s_, b.u, ps)
                for p in ps
                    (a1, a2), (b1, b2_) = cl[p], cu[p]
                    lo_ = min(a1, b1); hi_ = max(a2, b2_)
                    hi_ - lo_ < 0.5 * (dhi[p, s_] - dlo[p, s_]) || continue      # 두 꼭짓점의 차이가 함의 폭의 절반 안에 모임
                    ncand += 1; nmip += 2
                    dd_pair!(dlo, dhi, p, s_, b.l, b.u) || (empty_box = true; break)
                end
                empty_box && break
            end
            z = empty_box ? -Inf : solve_box!(b.l, b.u, b.lo, b.hi, b.phib, b.eq; dlo=dlo, dhi=dhi)[1]
            push!(hist, z)
        end
        tot[:mip] += nmip; tot[:t] += tb; z <= TGT && (tot[:pruned] += 1)
        @printf("    상자 %3d: z − 목표 %s (후보 %d, MIP %d 개, %.1fs) | 안 고정 쌍대 %d/%d | 폭 %s\n", n, join([isfinite(h) ? @sprintf("%+.2f", h - TGT) : "-Inf" for h in hist], " → "),
                ncand, nmip, tb, count(hi .- lo .> 1e-4), length(lo), string(round.((b.u .- b.l)[nd.hU .> 1e-9]; digits=2)))
        flush(stdout); continue
    end
    tb = @elapsed for rd in 1:RR
        (z <= TGT || !isfinite(z)) && break
        α̂ = clamp.(value.(v[:α]), b.l, b.u); d̂ = value.(v[:d]); r̂ = value.(v[:r])
        la, ha = _nz_dr_corner_ranges(CL, DR, nd, x̄, α̂, α̂, lo, hi)
        sc_ = Tuple{Float64,Int,Int}[]
        for (p, (a, b2)) in enumerate(DR.pairs), s in 1:S
            (p, s) in done && continue
            (hi[a, s] - lo[a, s] > 1e-4 && hi[b2, s] - lo[b2, s] > 1e-4) || continue
            ka, kb = DR.Hk[a], DR.Hk[b2]; I1, I2 = la[a, s] - ha[b2, s], ha[a, s] - la[b2, s]
            vf = d̂[s] > 1e-7 ? dist((value(v[:π̃][ka, s]) - value(v[:π̃][kb, s])) / d̂[s], I1, I2) * d̂[s] : 0.0
            vl = r̂[s] > 1e-7 ? dist((value(v[:ϖ][ka, s]) - value(v[:ϖ][kb, s])) / r̂[s], I1, I2) * r̂[s] : 0.0
            max(vf, vl) > 1e-3 && push!(sc_, (max(vf, vl), p, s))
        end
        isempty(sc_) && break
        sort!(sc_; rev=true)
        empty_box = false
        for (_, p, s) in sc_[1:min(MM, end)]
            push!(done, (p, s)); nmip += 2
            dd_pair!(dlo, dhi, p, s, b.l, b.u) || (empty_box = true; break)
        end
        z = empty_box ? -Inf : solve_box!(b.l, b.u, b.lo, b.hi, b.phib, b.eq; dlo=dlo, dhi=dhi)[1]
        push!(hist, z)
    end
    tot[:mip] += nmip; tot[:t] += tb; z <= TGT && (tot[:pruned] += 1)
    @printf("    상자 %3d: z − 목표 %s (MIP %d 개, %.1fs) | 안 고정 쌍대 %d/%d | 폭 %s\n", n, join([isfinite(h) ? @sprintf("%+.2f", h - TGT) : "-Inf" for h in hist], " → "),
            nmip, tb, count(hi .- lo .> 1e-4), length(lo), string(round.((b.u .- b.l)[nd.hU .> 1e-9]; digits=2)))
    flush(stdout)
end
@printf("RESULT lazy DD: 상자 %d 개 중 가지치기 %d, MIP 합 %d 개, %.0fs (상자당 %.1fs)\n", length(ordz), tot[:pruned], tot[:mip], tot[:t], tot[:t] / max(1, length(ordz)))
