# front_dd.jl — 저장된 oracle 전선에서 쌍대 차이 범위 (DD) 를 넣으면 노드 상한이 얼마나 내려가는지 (2026-11-25 밤, α-B&B 튜닝 진단). front_blame.jl 에서 시작.
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

GDD = envf("GDD", "0") == "1"; TGT = parse(Float64, envf("TGT", "-3151.60"))
t0 = time(); zr = solve_box!(zeros(nh), copy(nd.hU), lo0, hi0, fill(Inf, S), falses(nh))[1]
@printf("--- DD 시험 %s: g=%d S=%d x=%s, 열린 상자 %d, incumbent %.2f, 목표 %.2f | 루트 %.2f (%.0fs), 쌍 %d × 시나리오 %d\n",
        envf("FR", ""), fr.g, S, fr.xs, length(fr.boxes), LB, TGT, zr, time() - t0, length(DR.pairs), S); flush(stdout)
dg = nothing
if GDD
    lr, hr = _nz_dr_corner_ranges(CL, DR, nd, x̄, zeros(nh), copy(nd.hU), lo0, hi0)
    dl0, dh0 = _nz_dr_dglobal(nd, DR)
    tg = @elapsed (dg = _nz_dr_dranges!(DR, nd, zeros(nh), copy(nd.hU), lr, hr, dl0, dh0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), Dict{Tuple{Int,Int,Bool},Vector{Float64}}()))
    w0 = [min(hr[a, s] - lr[b, s], dh0[p, s]) - max(lr[a, s] - hr[b, s], dl0[p, s]) for (p, (a, b)) in enumerate(DR.pairs), s in 1:S]
    w1 = dg[2] .- dg[1]
    zg = solve_box!(zeros(nh), copy(nd.hU), lo0, hi0, fill(Inf, S), falses(nh); dlo=dg[1], dhi=dg[2])[1]
    @printf("    루트 DD: %.0fs, 차이 폭 (성분 범위가 함의) 중앙 %.1f → %.1f, 폭 ≤ 1e-3 인 (쌍, s) %d / %d, 루트 상한 %.2f → %.2f\n",
            tg, median(w0), median(w1), count(<=(1e-3), w1), length(w1), zr, zg); flush(stdout)
end
# 유효성 점검 (VAL=1): incumbent α 를 품은 폭 2w 상자에서 DD 를 넣어도 상한 ≥ incumbent 이어야 함
if envf("VAL", "0") == "1"
    for w in (1.0, 4.0)
        l = clamp.(fr.α .- w, 0.0, nd.hU); u = clamp.(fr.α .+ w, 0.0, nd.hU); l[nd.hU .<= 1e-9] .= 0.0; u[nd.hU .<= 1e-9] .= 0.0
        z0, (lo, hi) = solve_box!(l, u, lo0, hi0, fill(Inf, S), falses(nh))
        dl0, dh0 = _nz_dr_dglobal(nd, DR); dg === nothing || ((dl0, dh0) = dg)
        dd = _nz_dr_dranges!(DR, nd, l, u, lo, hi, dl0, dh0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), Dict{Tuple{Int,Int,Bool},Vector{Float64}}())
        z1 = solve_box!(l, u, lo0, hi0, fill(Inf, S), falses(nh); dlo=dd[1], dhi=dd[2])[1]
        zg = dg === nothing ? NaN : solve_box!(l, u, lo0, hi0, fill(Inf, S), falses(nh); dlo=dg[1], dhi=dg[2])[1]
        @printf("    [유효성] incumbent 중심 폭 %.0f 상자: 기본 z − inc %+.3f, 상자 DD %+.3f, 루트 DD %s (모두 ≥ 0 이어야 함)\n", 2w, z0 - LB, z1 - LB,
                isnan(zg) ? "—" : @sprintf("%+.3f", zg - LB))
        (z1 < LB - 1e-3 * sc || (!isnan(zg) && zg < LB - 1e-3 * sc)) && println("    [유효성] !!! DD 상한이 incumbent 아래 — DD 가 유효하지 않거나 수치 문제")
        flush(stdout)
    end
end
cand = sort(fr.boxes; by=b -> -b.ub)[1:min(NC, end)]
zs = [solve_box!(b.l, b.u, b.lo, b.hi, b.phib, b.eq)[1] for b in cand]
live = [n for n in eachindex(cand) if isfinite(zs[n]) && zs[n] > TGT]
ordz = live[sortperm(zs[live]; rev=true)][1:min(K, end)]
@printf("    후보 %d 개 중 자기 LP 상한 > 목표 %d 개, 위에서 %d 개\n", length(cand), length(live), length(ordz)); flush(stdout)
println("    상자 | 기본 z − 목표 | 상자 DD z − 목표 (MIP 초) | 루트 DD z − 목표 | 안 고정 쌍대 | 폭")
nb = 0; nb_dd = 0; nb_g = 0
for n in ordz
    b = cand[n]
    z0, (lo, hi) = solve_box!(b.l, b.u, b.lo, b.hi, b.phib, b.eq)
    dl0, dh0 = _nz_dr_dglobal(nd, DR)
    dg === nothing || ((dl0, dh0) = dg)
    tdd = 0.0; z1 = NaN
    if envf("BDD", "1") == "1"   # 상자 DD (KKT-MIP, 상자마다)
        tdd = @elapsed (dd = _nz_dr_dranges!(DR, nd, b.l, b.u, lo, hi, dl0, dh0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), Dict{Tuple{Int,Int,Bool},Vector{Float64}}()))
        z1 = solve_box!(b.l, b.u, b.lo, b.hi, b.phib, b.eq; dlo=dd[1], dhi=dd[2])[1]
        # 어떤 쌍이 조여졌나: 성분 범위가 함의하는 차이 폭 대비 DD 폭
        tight = Tuple{Int,Int,Int,Float64,Float64}[]
        for (p, (a, b2)) in enumerate(DR.pairs), s in 1:S
            wi = min(hi[a, s] - lo[b2, s], dh0[p, s]) - max(lo[a, s] - hi[b2, s], dl0[p, s]); wd = dd[2][p, s] - dd[1][p, s]
            wi > 1.0 && wd < 0.5 * wi && push!(tight, (DR.Hk[a], DR.Hk[b2], s, wi, wd))
        end
        rowsite(k) = nd.hrow[k]
        cnt = Dict{Tuple{Int,Int},Int}()
        for t in tight; key = (rowsite(t[1]), rowsite(t[2])); cnt[key] = get(cnt, key, 0) + 1; end
        @printf("      조여진 (쌍, s) %d 개 / %d. 좌표 쌍별: %s\n", length(tight), length(DR.pairs) * S,
                join([@sprintf("(%d,%d)×%d", k[1], k[2], v) for (k, v) in sort(collect(cnt); by=x -> -x[2])][1:min(8, end)], " "))
        for t in sort(tight; by=t -> t[5] / t[4])[1:min(4, end)]
            @printf("        행 %d (좌표 %d) − 행 %d (좌표 %d), s=%d: 함의 폭 %.1f → DD 폭 %.2f\n", t[1], rowsite(t[1]), t[2], rowsite(t[2]), t[3], t[4], t[5])
        end
    end
    zg = dg === nothing ? NaN : solve_box!(b.l, b.u, b.lo, b.hi, b.phib, b.eq; dlo=dg[1], dhi=dg[2])[1]
    global nb += 1; (!isnan(z1) && z1 <= TGT) && (global nb_dd += 1); (isfinite(zg) && zg <= TGT) && (global nb_g += 1)
    @printf("    %3d | %+10.2f | %+10.2f (%.0fs) | %s | %d/%d | %s\n", n, z0 - TGT, z1 - TGT, tdd, isnan(zg) ? "—" : @sprintf("%+10.2f", zg - TGT),
            count(hi .- lo .> 1e-4), length(lo), string(round.((b.u .- b.l)[nd.hU .> 1e-9]; digits=2)))
    flush(stdout)
end
@printf("RESULT DD 시험: 상자 %d 개 중 목표 아래로 (가지치기) 상자 DD %d, 루트 DD %s\n", nb, nb_dd, dg === nothing ? "—" : string(nb_g))
