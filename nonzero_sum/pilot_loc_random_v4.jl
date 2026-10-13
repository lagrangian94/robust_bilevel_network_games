"""
pilot_loc_random_v4.jl — 분기 차원이 큰 인스턴스에서 2026-10-13 의 기본 α-B&B 구성이 통하는지 보는 파일럿.

Goyal et al. (2023) §7.3 최소 설정 (pilot_loc_random.jl 과 같은 인스턴스): d = 15, A 2, B 후보 3, N_B = 3, 고객 10.
pair 예약이면 n_h = 5 × 10 = 50 (기본 인스턴스 SGB128 은 15, x̄ = {1,2} 에서 9).
방법은 벤치마크와 같은 ALGL (Algorithm 2 + local cut), 허용오차 0.5%, oracle 목표 gap 1e-3, Ipopt 120 s.
θᵁ 는 make_location_instance 의 기본값 (거리 해상도 기준 circuit 상계 v / 해상도 = 50). 정확한 circuit 값은 경로 수가 많아 계산하지 않는다.

환경변수: PR_SEEDS = "1,2"  PR_S = "3"  PR_RES = "pair"  PR_TL = 1800  PR_QUOTA = 300  PR_WORKERS = 12  PR_BNB_KW = (α-B&B 인자, 기본 비움)
          PR_NLOC = 15 (위치 수 d. 고객 = d − 5, pair 면 n_h = 5 (d − 5), 용량 150 (d − 5) / 2 — 분기 차원을 바꿔 가며 한계를 보는 용도)
          PR_PRE = 0 | 1 (1: 예약 상한을 고객의 최대 수요로 줄이고, 반응 정의역 프리솔브 nz_root_tighten.jl 을 시험용으로 연결. lab_rtight_bench.jl 과 같은 방식)
실행: julia -t 14,1 nonzero_sum/pilot_loc_random_v4.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
include(joinpath(root, "nonzero_sum", "nz_root_tighten.jl"))
envf(k, d) = get(ENV, k, d)
strs(k, d) = String.(strip.(split(envf(k, d), ",")))
seeds = parse.(Int, strs("PR_SEEDS", "1,2")); Ss = parse.(Int, strs("PR_S", "3")); ress = Symbol.(strs("PR_RES", "pair"))
TL = parse(Float64, envf("PR_TL", "1800")); quota = parse(Float64, envf("PR_QUOTA", "300")); nw = parse(Int, envf("PR_WORKERS", "12"))
kw = merge((ipopt_time=120.0,), eval(Meta.parse(envf("PR_BNB_KW", "NamedTuple()"))))
nloc = parse(Int, envf("PR_NLOC", "15")); capv = 150.0 * (nloc - 5) / 2; PRE = envf("PR_PRE", "0") == "1"
function mk(res, S, seed)
    nd = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=2, nB=3, NB=3, S=S, seed=seed,
        bA=capv, bB=capv, qB=capv, dlo=50.0, dhi=150.0, reservation=res, quota=quota, wres=300.0,
        eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
    (PRE && res == :pair) || return nd
    mt = nd.meta; dmaxj = [maximum(mt[:xi][j, :]) for j in 1:mt[:ncu]]            # 예약 상한 ≤ 고객의 최대 수요 (지배 논증, 문서 §10.15)
    return nz_with_hU(nd, [min(nd.hU[mt[:hidx](i, j)], dmaxj[j]) for i in 1:mt[:nst] for j in 1:mt[:ncu]])
end
const _RT_CACHE = Dict{Any,Tuple{Vector{Float64},Int}}()
if PRE          # nz_alpha_bnb_dr 의 프리솔브 호출에 반응 정의역 프리솔브를 끼운다 (기존 파일은 그대로, x̄ 마다 한 번)
    function nz_presolve_hU(nd::NZData, x̄::Vector{Float64})
        key = (nd.name, nd.S, copy(nd.hU), copy(x̄))
        haskey(_RT_CACHE, key) && return _RT_CACHE[key]
        hU1, n1 = invoke(nz_presolve_hU, Tuple{NZData,Any}, nd, x̄)
        hU2, n2, t = nz_response_presolve(nz_with_hU(nd, hU1), x̄; env=_nz_envs(1)[1])
        @printf("    [반응 프리솔브] x=%s: 활성 %d 중 %d 개를 0 으로 고정 (%.1fs)\n", string(findall(x̄ .> 0.5)), count(hU1 .> 0), n2, t); flush(stdout)
        return (_RT_CACHE[key] = (hU2, n1 + n2))
    end
end
xstr(x) = "{" * join(findall(x .> 0.5), ",") * "}"
for res in ress, seed in seeds, S in sort(Ss)
    nd = mk(res, S, seed)
    @printf("\n--- %s d=%d seed=%d S=%d: n_h=%d, θᵁ=%.1f λᵁ=%.1f, TL=%.0fs, 프리솔브 %s\n", res, nloc, seed, S, nz_nh(nd), nd.thetaU, nd.lambdaU, TL, PRE ? "켬" : "끔"); flush(stdout)
    t0 = time()
    r = nz_belief_menu_benders(nd; optimizer=GRB, tol=5e-3, oracle=:alpha_bnb, bnb=:dr, oracle_gap=1e-3, nworkers=nw, time_limit=TL,
                               local_first=true, local_time=60.0, bnb_kw=kw, verbose=true)
    gap = abs(r[:UB] - r[:LB]) / max(1.0, abs(r[:UB]))
    @printf("RESULT res=%s d=%d seed=%d S=%d nh=%d status=%s x=%s LB=%.6f UB=%.6f gap=%.2e iters=%d global=%d local=%s menu=%s time=%.1f\n",
            res, nloc, seed, S, nz_nh(nd), r[:status], xstr(r[:x]), r[:LB], r[:UB], gap, r[:iters], r[:oracle_calls],
            string(get(r, :local_cuts, "-")), string(get(r, :menu_size, "-")), time() - t0)
    flush(stdout)
end
