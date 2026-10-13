"""
lab_root_tighten.jl — 루트에서 LP 범위 조임 (목적 ≥ incumbent 인 영역의 좌표별 min/max α_i) 으로 α 상자를 줄인 뒤 α-B&B 를 돌리면 얼마나 빨라지는지.

솔버 (nz_alpha_bnb_dr) 는 고치지 않는다: 줄인 상한을 hᵁ 로 넣은 사본 (nz_with_hU) 을 넘긴다. 그래서 하한 쪽 조임 (l_i > 0) 은 쓰지 못하고 상한 쪽만 쓴다.
  1. incumbent: LT_LB 를 주면 그 값, 아니면 α-B&B 를 LT_T1 초 돌려 얻은 값
  2. 루트 노드 LP (기본 구성의 조임 전부) 를 분리까지 푼 뒤, 목적 ≥ LB 조건을 넣고 좌표마다 min/max α_i (LP 2 × 활성 좌표 개)
  3. 줄인 hᵁ 로 α-B&B 를 처음부터
유효성: 노드 LP 는 Ω 의 완화에 "최적값을 바꾸지 않는 제한" (쌍대 범위 등, α 마다 값이 같은 완성이 남는다) 을 더한 것이라, G(α) ≥ LB 인 α 는 모두
        {목적 ≥ LB} 인 LP 실행가능 영역의 사영 안에 있다. 그 밖의 α 를 버려도 sup 은 max(LB, 남은 영역의 sup) 로 같다.

환경변수: NZ_SEED = 3  LT_S = 20  LT_X = "1,2"  LT_TL = 600  LT_GAP = 1e-3  LT_WORKERS = 12  LT_T1 = 30  LT_LB = (비우면 1 의 짧은 실행)
          LT_BASE = 1 (조임 없는 기준 실행도 함께)
실행: julia -t 14,1 nonzero_sum/lab_root_tighten.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
mk1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d)
S = parse(Int, envf("LT_S", "20")); seed = parse(Int, envf("NZ_SEED", "3")); TL = parse(Float64, envf("LT_TL", "600"))
GAP = parse(Float64, envf("LT_GAP", "1e-3")); nw = parse(Int, envf("LT_WORKERS", "12")); T1 = parse(Float64, envf("LT_T1", "30"))
xi = parse.(Int, split(envf("LT_X", "1,2"), ","))
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=seed,
                                           eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
ndf = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
x̄ = zeros(ndf.nx); x̄[xi] .= 1
hUp, _ = nz_presolve_hU(ndf, x̄); nd = nz_with_hU(ndf, hUp)
nh = nz_nh(nd); act = findall(nd.hU .> 0)
@printf("lab_root_tighten: SGB128 seed=%d S=%d x=%s 활성 좌표 %d, gap %.0e, 제한 %.0fs\n", seed, S, string(xi), length(act), GAP, TL); flush(stdout)
line(tag, r, t) = @printf("RESULT %-10s %s LB=%.4f UB=%.4f gap=%.3e time=%.1f nodes=%d rootUB=%.2f 범위 MIP %d\n", tag, r[:is_exact] ? "Optimal" : "TimeLim",
                          r[:LB], r[:UB], (r[:UB] - r[:LB]) / max(1, abs(r[:LB])), t, r[:nodes], r[:root_UB], r[:range_mips])
if envf("LT_BASE", "1") == "1"
    tb = @elapsed rb = nz_alpha_bnb_dr(nd, x̄; nworkers=nw, time_limit=TL, rel_gap=GAP, verbose=false, ipopt_time=120.0)
    line("기준", rb, tb); flush(stdout)
end
# 1. incumbent
t1 = 0.0
if isempty(envf("LT_LB", ""))
    t1 = @elapsed r1 = nz_alpha_bnb_dr(nd, x̄; nworkers=nw, time_limit=T1, rel_gap=GAP, verbose=false, ipopt_time=120.0)
    LB = r1[:LB]
    @printf("1. 짧은 실행 %.0fs: LB=%.4f UB=%.4f (노드 %d)\n", t1, LB, r1[:UB], r1[:nodes])
else
    LB = parse(Float64, envf("LT_LB", "")); @printf("1. 주어진 LB=%.4f\n", LB)
end
flush(stdout)
# 2. 루트 범위 조임
sc = max(1.0, abs(LB))
t2 = @elapsed begin
    P = _nz_build_sep(nd, x̄; optimizer=mk1); mdl = P.O.model; αv = P.O.v[:α]
    for a in ("FeasibilityTol", "OptimalityTol"); set_optimizer_attribute(mdl, a, 1e-7); end
    set_optimizer_attribute(mdl, "NumericFocus", 0); mdl.ext[:lp_tol] = 1e-7; mdl.ext[:retry_fresh] = true
    DR = _nz_dr_attach!(P, nd, x̄; optimizer=mk1); lo0, hi0 = _nz_dr_global(nd, DR); dl0, dh0 = _nz_dr_dglobal(nd, DR)
    l = zeros(nh); u = copy(nd.hU)
    _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
    lo, hi, wt, phib = _nz_dr_ranges(DR, nd, l, u, lo0, hi0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), fill(Inf, S))
    dl, dh = _nz_dr_dranges!(DR, nd, l, u, lo, hi, dl0, dh0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), wt)
    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; dlo=dl, dhi=dh)
    zroot = _nz_sep_solve!(P; maxrounds=30)
    fobj = objective_function(mdl)
    @constraint(mdl, fobj >= LB - 1e-6 * sc)
    nl = copy(l); nu = copy(u)
    for i in act, ismax in (false, true)
        @objective(mdl, ismax ? MOI.MAX_SENSE : MOI.MIN_SENSE, αv[i]); optimize!(mdl)
        st = termination_status(mdl)
        st == MOI.OPTIMAL || error("범위 조임 LP: $st (좌표 $i)")      # 실행 불가능이면 LB 가 루트 상한보다 큰 것 — 여기서는 나올 수 없다
        ismax ? (nu[i] = min(u[i], objective_value(mdl) + 1e-6)) : (nl[i] = max(l[i], objective_value(mdl) - 1e-6))
    end
end
hU2 = [nu[i] <= 1e-5 ? 0.0 : nu[i] for i in 1:nh]
@printf("2. 루트 상한 %.2f, 범위 조임 %.0fs: 0 으로 고정 %d / %d, 상한 폭 합 %.1f%% (하한 쪽 조임 %.1f%% 는 쓰지 못함)\n   좌표별 [l, u] / hᵁ = %s\n",
        zroot, t2, count(i -> hU2[i] == 0, act), length(act), 100 * sum(hU2) / sum(nd.hU), 100 * sum(nl) / sum(nd.hU),
        join([@sprintf("%.2f~%.2f", nl[i] / nd.hU[i], nu[i] / nd.hU[i]) for i in act], " ")); flush(stdout)
# 3. 줄인 상자에서 α-B&B
nd2 = nz_with_hU(nd, hU2)
t3 = @elapsed r3 = nz_alpha_bnb_dr(nd2, x̄; nworkers=nw, time_limit=TL, rel_gap=GAP, verbose=false, ipopt_time=120.0)
line("조임", r3, t3)
@printf("   전체 %.1fs = incumbent %.1f + 범위 조임 %.1f + α-B&B %.1f | 조임 후의 값이 LB 보다 작으면 최적은 max(LB, ·) = %.4f\n", t1 + t2 + t3, t1, t2, t3, max(LB, r3[:LB]))
