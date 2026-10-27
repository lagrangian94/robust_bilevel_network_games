"""
run_bd_kkt_pooled.jl — 벤치마크 BD-KKT / BD-G (nz_bd_kkt.jl, PP_METHOD = kkt | omega) 를 본 실험 목록 L1~L5 의 인스턴스로 (pilot_pooled_store.jl 과 같은 생성·환경변수).

결정 (2026-10-21 사용자): 먼저 S = 10 만. 잘 안 풀리면 S = 20 부터는 "--" 로 보고. 그다음 BD-G 도 같은 방식.
시간 한도·허용오차·λᵁ·θᵁ·인스턴스는 Algorithm 2 (ALGL) 실행과 같다. KKT 모형의 Gurobi 스레드 = PP_WORKERS (우리 B&B worker 수와 같게).

환경변수: pilot_pooled_store.jl 과 같음 (PP_NLOC, PP_NA, PP_NB, PP_NBMAX, PP_S, PP_SEEDS, PP_QUOTA, PP_QB, PP_LAMBDA, PP_F, PP_P, PP_TL, PP_WORKERS, PP_EPS, PP_DELTA)
실행: julia -t 2 nonzero_sum/run_bd_kkt_pooled.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
envf(k, d) = get(ENV, k, d); pf(k, d) = parse(Float64, envf(k, d)); pint(k, d) = parse(Int, envf(k, d))
strs(k, d) = String.(strip.(split(envf(k, d), ",")))
nw = pint("PP_WORKERS", "12")
meth = Symbol(envf("PP_METHOD", "kkt")); meth in (:kkt, :omega, :ccg) || error("PP_METHOD = kkt | omega | ccg")
MNAME = meth == :kkt ? "BD-KKT" : meth == :omega ? "BD-G" : "CCG"
GRBK() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), nw); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
include(joinpath(root, "nonzero_sum", "nz_bd_kkt.jl"))
if meth == :ccg       # CCG 의 subproblem = 우리 spatial B&B (pilot_pooled_store.jl 의 Ours 실행과 같은 설정·반응 정의역 프리솔브)
    isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
    include(joinpath(root, "nonzero_sum", "nz_root_tighten.jl"))
    include(joinpath(root, "nonzero_sum", "nz_ccg.jl"))
    const _RT_CACHE = Dict{Any,Tuple{Vector{Float64},Int}}()
    function nz_presolve_hU(nd::NZData, x̄::Vector{Float64})
        key = (nd.name, nd.S, copy(nd.hU), copy(x̄))
        haskey(_RT_CACHE, key) && return _RT_CACHE[key]
        hU1, n1 = invoke(nz_presolve_hU, Tuple{NZData,Any}, nd, x̄)
        hU2, n2, t = nz_response_presolve(nz_with_hU(nd, hU1), x̄; env=_nz_envs(1)[1])
        return (_RT_CACHE[key] = (hU2, n1 + n2))
    end
end
nloc = pint("PP_NLOC", "15"); nA = pint("PP_NA", "2"); nB = pint("PP_NB", "3"); NB = pint("PP_NBMAX", string(nB))
Ss = parse.(Int, strs("PP_S", "10")); seeds = parse.(Int, strs("PP_SEEDS", "1"))
quota = pf("PP_QUOTA", "300"); lam = pf("PP_LAMBDA", "2000"); fres = pf("PP_F", "0.1"); prem = pf("PP_P", "0.3")
TL = pf("PP_TL", "3600"); ε = pf("PP_EPS", "0.3"); δ = pf("PP_DELTA", "0.1")
m = nA + nB; capv = 150.0 * (nloc - m) / nA
qB = envf("PP_QB", "750") == "cap" ? capv : pf("PP_QB", "750")
function mk(S, seed)
    nd = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=nA, nB=nB, NB=NB, S=S, seed=seed,
        bA=capv, bB=capv, qB=qB, dlo=50.0, dhi=150.0, p=prem, f=fres, reservation=:pooled, pool_limit=:store, quota=quota,
        eps_hat=ε, eps_tilde=ε, beta=0.4), δ)
    return nz_with_bounds(nd; thetaU=nz_theta_circuit_exact(nd)[1], lambdaU=lam)
end
xstr(x) = "{" * join(findall(x .> 0.5), ",") * "}"
for seed in seeds, S in sort(Ss)
    nd = mk(S, seed)
    @printf("\n--- %s pooled(거점별 %.0f) d=%d A=%d B=%d (N_B≤%d) seed=%d S=%d: n_h=%d, |X|=%d, 용량 %.0f, 출점비 %.0f, θᵁ=%.1f λᵁ=%.1f, TL=%.0fs, Gurobi 스레드 %d\n",
            MNAME, quota, nloc, nA, nB, NB, seed, S, nz_nh(nd), sum(binomial(nB, k) for k in 0:NB), capv, qB, nd.thetaU, nd.lambdaU, TL, nw)
    flush(stdout)
    t0 = time()
    r = meth == :ccg ?
        nz_ccg(nd; optimizer=GRBK, tol=5e-3, oracle_gap=1e-3, time_limit=TL, nworkers=nw, bnb_kw=(ipopt_time=120.0,), verbose=true) :
        nz_bd_kkt_benders(nd; optimizer=GRB, kkt_optimizer=GRBK, tol=5e-3, oracle_gap=1e-3, time_limit=TL, local_time=60.0, verbose=true, oracle=meth)
    gap = abs(r[:UB] - r[:LB]) / max(1.0, abs(r[:UB]))
    @printf("RESULT method=%s λ=%.0f qB=%.0f d=%d m=%d nA=%d nB=%d seed=%d S=%d nh=%d status=%s x=%s LB=%.6f UB=%.6f gap=%.2e iters=%d global=%d local=%d time=%.1f\n",
            MNAME, lam, qB, nloc, m, nA, nB, seed, S, nz_nh(nd), r[:status], xstr(r[:x]), r[:LB], r[:UB], gap, r[:iters], get(r, :oracle_calls, get(r, :scenarios, 0)),
            get(r, :local_cuts, 0), time() - t0)
    flush(stdout)
end
