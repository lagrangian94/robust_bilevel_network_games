"""
diag_pooled_dr.jl — pooled 무작위 d=8 seed 1 x̄={3} 에서 쌍대 범위 조임 (base 구성) 이 α-B&B incumbent α° 를 잘라내는 원인 가르기.
test_dr_validity_pooled.jl 의 상자 검사에서 off (범위 행 없음) 는 정상, base (W·F 범위 + 상자 상보성 C) 부터 실행 불가능이 나왔다.

  α° 와 그 둘레 상자에서 노드 LP 를 네 가지로 푼다:
    range+C : 상자별 범위 + 상자 상보성 (기본)       range : 범위만 (comp=false)
    global+C: 전역 범위 [0, ϖᵁ] + 상자 상보성           global: 전역 범위만
  그리고 α° 에서 시나리오별 recourse LP 의 최적 쌍대 (더 낮출 수 없는 것) 를 직접 구해 범위 [lo, hi] 안에 있는지 본다.
환경변수: DG_NLOC = 8  NZ_SEED = 1  DG_X = "3"  DG_S = 3  DG_POOL = total | store (거점별 한도 300)
실행: julia -t 4,1 nonzero_sum/diag_pooled_dr.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random
const GRB_ENV = Gurobi.Env()
GRB() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 2); MOI.set(o, MOI.Silent(), true); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d)
nloc = parse(Int, envf("DG_NLOC", "8")); seed = parse(Int, envf("NZ_SEED", "1")); S = parse(Int, envf("DG_S", "3"))
xi = parse.(Int, split(envf("DG_X", "3"), ","))
cap = 150.0 * (nloc - 5) / 2
nd1 = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=2, nB=3, NB=3, S=S, seed=seed, bA=cap, bB=cap, qB=cap,
                                          dlo=50.0, dhi=150.0, reservation=:pooled, wres=300.0, pool_limit=Symbol(envf("DG_POOL", "total")), quota=300.0, eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
ndf = nz_with_bounds(nd1; thetaU=nz_theta_circuit_exact(nd1)[1])
x̄ = zeros(ndf.nx); x̄[xi] .= 1
r = nz_alpha_bnb_dr(ndf, x̄; nworkers=2, time_limit=120.0, rel_gap=1e-4, verbose=false, ipopt_time=30.0)
hUp, _ = nz_presolve_hU(ndf, x̄); nd = nz_with_hU(ndf, hUp); nh = nz_nh(nd)
α° = clamp.(r[:α], 0.0, nd.hU)
F = nz_build_fixed_alpha(nd, x̄; optimizer=GRB); G = nz_eval_alpha!(F, nd, α°)
d° = [value(F.v[:d][s]) for s in 1:S]; r° = [value(F.v[:r][s]) for s in 1:S]
@printf("B&B [%.4f, %.4f], α° = %s, G(α°) = %.4f, d° = %s, r° = %s, hU = %s\n", r[:LB], r[:UB], string(round.(α°; digits=3)), G,
        string(round.(d°; digits=4)), string(round.(r°; digits=4)), string(nd.hU))

# α° 에서 follower 가 정말 최적인가: d° 에서 2 단계 LP 의 최적값과 α° 고정 값 비교
fm = Model(GRB); @variable(fm, 0 <= αf[i = 1:nh] <= nd.hU[i]); @variable(fm, yf[1:nz_ny(nd), 1:S] >= 0); @constraint(fm, nd.W * αf .<= nd.wvec)
for s in 1:S, k in 1:nz_m(nd)
    @constraint(fm, sum(nd.A[k, j] * yf[j, s] for j in 1:nz_ny(nd) if nd.A[k, j] != 0) <= nz_rhs(nd, x̄, zeros(nh), s)[k] + (nd.hrow[k] > 0 ? nd.hcoef[k] * αf[nd.hrow[k]] : 0.0))
end
@objective(fm, Max, dot(nd.c0, αf) + sum(d°[s] * nd.c[j] * yf[j, s] for j in 1:nz_ny(nd), s in 1:S)); optimize!(fm); vopt = objective_value(fm)
for i in 1:nh; fix(αf[i], α°[i]; force=true); end; optimize!(fm); vα = objective_value(fm)
@printf("follower 최적성 (d°): 최적 %.6f, α° 고정 %.6f, 차 %.2e\n", vopt, vα, vopt - vα)

# 정확한 최적 반응: d° 에서 follower LP 의 최적 꼭짓점 α* (simplex). 이 점을 기준으로 다시 검사한다 (α° 는 허용오차 안에서만 최적일 수 있음)
for i in 1:nh; unfix(fm[:αf][i]); set_lower_bound(fm[:αf][i], 0.0); set_upper_bound(fm[:αf][i], nd.hU[i]); end
set_optimizer_attribute(fm, "Method", 0); optimize!(fm); αstar = clamp.(value.(fm[:αf]), 0.0, nd.hU)
Gs = nz_eval_alpha!(F, nd, αstar)
@printf("d° 의 정확한 최적 꼭짓점 α* = %s, G(α*) = %.4f (G(α°) = %.4f)
", string(round.(αstar; digits=4)), Gs, G)
P = _nz_build_sep(nd, x̄; optimizer=GRB); DR = _nz_dr_attach!(P, nd, x̄; optimizer=GRB, vcut=false, cslift=false)
lo0, hi0 = _nz_dr_global(nd, DR)
function node(l, u; ranges=true, comp=true)
    _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
    rg = _nz_dr_ranges(DR, nd, l, u, lo0, hi0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), fill(Inf, S))
    rg === nothing && return (-Inf, nothing)
    lo, hi, wt, phib = rg
    ranges || (lo = copy(lo0); hi = copy(hi0))
    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp)
    z = _nz_sep_solve!(P); z == -Inf && (z = _nz_dr_recheck_infeasible!(P.O.model))
    return (z, (lo, hi))
end
function check_ref(αr, Gr, label)
    z0, lh = node(copy(αr), copy(αr)); lo, hi = lh
    println("== 기준점 ", label, ": 점 상자의 범위 (H 행 × 시나리오), Hk 좌표 = ", [nd.hrow[k] for k in DR.Hk])
    for (a, k) in enumerate(DR.Hk)
        @printf("  좌표 %d (α = %.4f): %s
", nd.hrow[k], αr[nd.hrow[k]], join([@sprintf("[%.3f, %.3f]", lo[a, s], hi[a, s]) for s in 1:S], " "))
    end
    for (nm, rg, cp) in (("range+C", true, true), ("range", true, false), ("global+C", false, true), ("global", false, false))
        z, _ = node(copy(αr), copy(αr); ranges=rg, comp=cp)
        @printf("  점 상자 %-8s: 노드 LP = %s (G = %.4f)
", nm, z == -Inf ? "실행 불가능" : @sprintf("%.4f", z), Gr)
    end
    # test_dr_validity 와 같은 상자 생성 (폭 0·한쪽 경계가 기준점에 붙은 상자 포함)
    rng = MersenneTwister(1234 + S); cnt = Dict{String,Vector{Int}}()
    for n in 1:100
        l = zeros(nh); u = copy(nd.hU)
        for i in 1:nh
            nd.hU[i] > 0 || continue
            mode = rand(rng, 1:6); lw = -4.0
            wl = 10.0^(rand(rng) * (log10(nd.hU[i]) - lw) + lw); wu = 10.0^(rand(rng) * (log10(nd.hU[i]) - lw) + lw)
            mode == 1 && (wl = 0.0); mode == 2 && (wu = 0.0); mode == 3 && (wl = 0.0; wu = 0.0); mode == 4 && (wl = Inf; wu = Inf)
            l[i] = max(0.0, αr[i] - wl); u[i] = min(nd.hU[i], αr[i] + wu)
        end
        for (nm, rg, cp) in (("range+C", true, true), ("range", true, false), ("global+C", false, true), ("global", false, false))
            z, _ = node(l, u; ranges=rg, comp=cp)
            c = get!(cnt, nm, [0, 0]); z == -Inf ? (c[1] += 1) : (z < Gr - 1e-5 * max(1, abs(Gr)) && (c[2] += 1))
        end
    end
    println("  상자 100 개 (test_dr_validity 와 같은 생성): (실행 불가능, 상한 < G)")
    for nm in ("range+C", "range", "global+C", "global"); @printf("    %-8s %s
", nm, string(cnt[nm])); end
end
check_ref(α°, G, "α° (B&B incumbent)")
check_ref(αstar, Gs, "α* (d° 의 정확한 최적 꼭짓점)")
