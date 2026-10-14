"""
pilot_pooled_store.jl — retailer + pooled 예약 (거점별 한도 h_i ≤ w_i) 모형의 3 단계 파일럿. 문서 §10.19.

인스턴스는 Goyal et al. (2023) §7.3 의 생성 방식: 위치 d 개 ([0,1]² 균등), 적격지 m = nA + nB (A 점유 nA, B 후보 nB), 나머지 d − m 이 시장 (수요 U[50,150]),
용량 b = 150 (d − m) / nA (A·B 같음), 출점비 qB (Goyal 750 / 1000), B 출점 수 ≤ NB. n_h = 적격지 수 m (거점별 예약 풀) 이라 d 와 무관하다.
θᵁ 는 정확한 circuit 값 (_nz_theta_circuit_exact_pooled). 방법은 벤치마크와 같은 ALGL (Algorithm 2 + local cut), 허용오차 0.5%, oracle gap 1e-3.

환경변수: PP_NLOC = 15   PP_NA = 2   PP_NB = 3   PP_NBMAX = (기본 nB)   PP_S = "10"   PP_SEEDS = "1"   PP_QUOTA = 300   PP_QB = 750
          PP_F = 0.1 (예약비)   PP_P = 0.3 (현장 프리미엄)   PP_TL = 1800   PP_WORKERS = 12   PP_PRE = 1 (반응 정의역 프리솔브)   PP_EPS = 0.3   PP_DELTA = 0.1
실행: julia -t 14,1 nonzero_sum/pilot_pooled_store.jl
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
envf(k, d) = get(ENV, k, d); pf(k, d) = parse(Float64, envf(k, d)); pint(k, d) = parse(Int, envf(k, d))
strs(k, d) = String.(strip.(split(envf(k, d), ",")))
nloc = pint("PP_NLOC", "15"); nA = pint("PP_NA", "2"); nB = pint("PP_NB", "3"); NB = pint("PP_NBMAX", string(nB))
Ss = parse.(Int, strs("PP_S", "10")); seeds = parse.(Int, strs("PP_SEEDS", "1"))
quota = pf("PP_QUOTA", "300"); qB = pf("PP_QB", "750"); fres = pf("PP_F", "0.1"); prem = pf("PP_P", "0.3")
TL = pf("PP_TL", "1800"); nw = pint("PP_WORKERS", "12"); PRE = envf("PP_PRE", "1") == "1"; ε = pf("PP_EPS", "0.3"); δ = pf("PP_DELTA", "0.1")
m = nA + nB; capv = 150.0 * (nloc - m) / nA
function mk(S, seed)
    nd = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=nA, nB=nB, NB=NB, S=S, seed=seed,
        bA=capv, bB=capv, qB=qB, dlo=50.0, dhi=150.0, p=prem, f=fres, reservation=:pooled, pool_limit=:store, quota=quota,
        eps_hat=ε, eps_tilde=ε, beta=0.4), δ)
    return nz_with_bounds(nd; thetaU=nz_theta_circuit_exact(nd)[1])
end
const _RT_CACHE = Dict{Any,Tuple{Vector{Float64},Int}}()
if PRE          # nz_alpha_bnb_dr 의 프리솔브 호출에 반응 정의역 프리솔브를 끼운다 (pilot_loc_random_v4.jl 과 같은 방식, x̄ 마다 한 번)
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
kw = (ipopt_time=120.0,)
for seed in seeds, S in sort(Ss)
    nd = mk(S, seed)
    @printf("\n--- pooled(거점별 %.0f) d=%d A=%d B=%d (N_B≤%d) seed=%d S=%d: n_h=%d, |X|=%d, 용량 %.0f, 출점비 %.0f, f=%.2f p=%.2f, θᵁ=%.1f λᵁ=%.1f, TL=%.0fs, 프리솔브 %s\n",
            quota, nloc, nA, nB, NB, seed, S, nz_nh(nd), sum(binomial(nB, k) for k in 0:NB), capv, qB, fres, prem, nd.thetaU, nd.lambdaU, TL, PRE ? "켬" : "끔")
    flush(stdout)
    t0 = time()
    r = nz_belief_menu_benders(nd; optimizer=GRB, tol=5e-3, oracle=:alpha_bnb, bnb=:dr, oracle_gap=1e-3, nworkers=nw, time_limit=TL,
                               local_first=true, local_time=60.0, bnb_kw=kw, verbose=true)
    gap = abs(r[:UB] - r[:LB]) / max(1.0, abs(r[:UB]))
    @printf("RESULT pooled-store d=%d m=%d nA=%d nB=%d seed=%d S=%d nh=%d status=%s x=%s LB=%.6f UB=%.6f gap=%.2e iters=%d global=%d local=%s menu=%s time=%.1f\n",
            nloc, m, nA, nB, seed, S, nz_nh(nd), r[:status], xstr(r[:x]), r[:LB], r[:UB], gap, r[:iters], r[:oracle_calls],
            string(get(r, :local_cuts, "-")), string(get(r, :menu_size, "-")), time() - t0)
    flush(stdout)
end
