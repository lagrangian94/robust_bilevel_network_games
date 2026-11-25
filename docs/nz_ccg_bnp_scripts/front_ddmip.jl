# front_ddmip.jl — 쌍대 차이 범위 (DD) 의 KKT-MIP 를 싸게 만들 수 있는지 시간 재기 (2026-11-26).
#   DD 문턱 12.5 판의 전선을 다시 보니 남은 상자는 대부분 문턱보다 넓어 DD 가 적용되지 않은 상자였다 (좁은 상자는 82% 잘림).
#   그래서 DD 의 문제는 효과가 아니라 MIP 비용 (개당 약 1 s) 이다. 선별 후보 (쌍, s) 마다 min·max MIP 를 세 가지로 풀어 시간과 bound 를 비교:
#     V0 지금 그대로 (α 상자만)   V1 + 상자의 꼭짓점 LP 범위 [lo, hi] 를 H 행 쌍대의 변수 경계로 (유효: 범위는 상자 안 모든 최적 쌍대를 담음)
#     V2 V1 + MIPFocus 3
#   bound 가 V0 와 1e-5 넘게 다르면 경고 (유효성 점검).
# (아래는 front_dd_lazy.jl 머리말)
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

cand = sort(fr.boxes; by=b -> -b.ub)[1:min(NC, end)]
nshow = 0; T = Dict(k => Float64[] for k in (:V0, :V1, :V2)); nbad = 0
for b in cand
    global nshow, nbad
    nshow >= K && break
    z0, (lo, hi) = solve_box!(b.l, b.u, b.lo, b.hi, b.phib, b.eq)
    (isfinite(z0) && z0 > LB + 20) || continue
    nshow += 1
    dlo0, dhi0 = _nz_dr_dglobal(nd, DR)
    tb = Dict(k => 0.0 for k in (:V0, :V1, :V2)); nm = 0; nconst = 0
    for s in 1:S
        ps = Int[]
        for (p, (a, b2)) in enumerate(DR.pairs)
            (hi[a, s] - lo[a, s] > 1e-4 && hi[b2, s] - lo[b2, s] > 1e-4) || continue
            wl = max(dlo0[p, s], lo[a, s] - hi[b2, s]); wh = min(dhi0[p, s], hi[a, s] - lo[b2, s])
            wh - wl > 1.0 && push!(ps, p)
        end
        isempty(ps) && continue
        ext = Dict{Int,Tuple{Float64,Float64}}()
        for αc in (b.l, b.u)
            for (p, mm) in corner_diff(s, αc, ps); o = get(ext, p, (Inf, -Inf)); ext[p] = (min(o[1], mm[1]), max(o[2], mm[2])); end
        end
        M = DR.mips[s]
        for i in 1:nh; set_lower_bound(M.α[i], b.l[i]); set_upper_bound(M.α[i], b.u[i]); end
        for p in ps
            a, b2 = DR.pairs[p]
            wl = max(dlo0[p, s], lo[a, s] - hi[b2, s]); wh = min(dhi0[p, s], hi[a, s] - lo[b2, s])
            ext[p][2] - ext[p][1] < 0.5 * (wh - wl) || continue
            ka, kb = DR.Hk[a], DR.Hk[b2]
            bds = Dict{Symbol,Vector{Float64}}()
            for V in (:V0, :V1, :V2)
                if V != :V0
                    for (aa, k) in enumerate(DR.Hk)
                        set_lower_bound(M.σ[k], max(0.0, lo[aa, s])); set_upper_bound(M.σ[k], hi[aa, s])
                    end
                end
                V == :V2 && set_optimizer_attribute(M.model, "MIPFocus", 3)
                bb = Float64[]
                for ismax in (false, true)
                    @objective(M.model, ismax ? MAX_SENSE : MIN_SENSE, M.σ[ka] - M.σ[kb])
                    t = @elapsed optimize!(M.model)
                    tb[V] += t; push!(T[V], t)
                    st = termination_status(M.model)
                    push!(bb, st == MOI.OPTIMAL ? objective_bound(M.model) : (st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED) ? -Inf : NaN))
                end
                bds[V] = bb
                V == :V2 && set_optimizer_attribute(M.model, "MIPFocus", 0)
                for k in DR.Hk
                    set_lower_bound(M.σ[k], 0.0); isfinite(nd.varpiU[k]) ? set_upper_bound(M.σ[k], nd.varpiU[k]) : delete_upper_bound(M.σ[k])
                end
            end
            nm += 2
            abs(bds[:V0][2] - bds[:V0][1]) < 1e-4 && (nconst += 1)
            for V in (:V1, :V2)
                for n in 1:2
                    x0, x1 = bds[:V0][n], bds[V][n]
                    (isfinite(x0) && isfinite(x1) && abs(x0 - x1) > 1e-5) && (nbad += 1; @printf("    !! bound 불일치 s=%d 쌍 (%d,%d) %s %s: V0 %.6f %s %.6f\n", s, ka, kb, V, n == 1 ? "min" : "max", x0, V, x1))
                end
            end
        end
    end
    @printf("  상자 %d: z0 %+.2f | 후보 MIP %d 개 (상수 차이 %d 쌍) | 시간 V0 %.1fs V1 %.1fs V2 %.1fs\n", nshow, z0 - LB, nm, nconst, tb[:V0], tb[:V1], tb[:V2])
    flush(stdout)
end
q(x, p) = isempty(x) ? NaN : quantile(x, p)
for V in (:V0, :V1, :V2)
    @printf("%s: MIP %d 개, 평균 %.3fs, 중앙값 %.3fs, 90%% %.3fs\n", V, length(T[V]), mean(T[V]), q(T[V], 0.5), q(T[V], 0.9))
end
@printf("bound 불일치 %d\n", nbad)
