"""
pilot_loc_random_v4.jl — 분기 차원이 큰 인스턴스에서 2026-10-13 의 기본 α-B&B 구성이 통하는지 보는 파일럿.

Goyal et al. (2023) §7.3 최소 설정 (pilot_loc_random.jl 과 같은 인스턴스): d = 15, A 2, B 후보 3, N_B = 3, 고객 10.
pair 예약이면 n_h = 5 × 10 = 50 (기본 인스턴스 SGB128 은 15, x̄ = {1,2} 에서 9).
방법은 벤치마크와 같은 ALGL (Algorithm 2 + local cut), 허용오차 0.5%, oracle 목표 gap 1e-3, Ipopt 120 s.
θᵁ 는 make_location_instance 의 기본값 (거리 해상도 기준 circuit 상계 v / 해상도 = 50). 정확한 circuit 값은 경로 수가 많아 계산하지 않는다.

환경변수: PR_SEEDS = "1,2"  PR_S = "3"  PR_RES = "pair"  PR_TL = 1800  PR_QUOTA = 300  PR_WORKERS = 12  PR_BNB_KW = (α-B&B 인자, 기본 비움)
실행: julia -t 14,1 nonzero_sum/pilot_loc_random_v4.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
envf(k, d) = get(ENV, k, d)
strs(k, d) = String.(strip.(split(envf(k, d), ",")))
seeds = parse.(Int, strs("PR_SEEDS", "1,2")); Ss = parse.(Int, strs("PR_S", "3")); ress = Symbol.(strs("PR_RES", "pair"))
TL = parse(Float64, envf("PR_TL", "1800")); quota = parse(Float64, envf("PR_QUOTA", "300")); nw = parse(Int, envf("PR_WORKERS", "12"))
kw = merge((ipopt_time=120.0,), eval(Meta.parse(envf("PR_BNB_KW", "NamedTuple()"))))
mk(res, S, seed) = nz_with_delta(make_location_instance(; coords=:random, n_loc=15, nA=2, nB=3, NB=3, S=S, seed=seed,
    bA=750.0, bB=750.0, qB=750.0, dlo=50.0, dhi=150.0, reservation=res, quota=quota, wres=300.0,
    eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
xstr(x) = "{" * join(findall(x .> 0.5), ",") * "}"
for res in ress, seed in seeds, S in sort(Ss)
    nd = mk(res, S, seed)
    @printf("\n--- %s seed=%d S=%d: n_h=%d, θᵁ=%.1f λᵁ=%.1f, TL=%.0fs\n", res, seed, S, nz_nh(nd), nd.thetaU, nd.lambdaU, TL); flush(stdout)
    t0 = time()
    r = nz_belief_menu_benders(nd; optimizer=GRB, tol=5e-3, oracle=:alpha_bnb, bnb=:dr, oracle_gap=1e-3, nworkers=nw, time_limit=TL,
                               local_first=true, local_time=60.0, bnb_kw=kw, verbose=true)
    gap = abs(r[:UB] - r[:LB]) / max(1.0, abs(r[:UB]))
    @printf("RESULT res=%s seed=%d S=%d nh=%d status=%s x=%s LB=%.6f UB=%.6f gap=%.2e iters=%d global=%d local=%s menu=%s time=%.1f\n",
            res, seed, S, nz_nh(nd), r[:status], xstr(r[:x]), r[:LB], r[:UB], gap, r[:iters], r[:oracle_calls],
            string(get(r, :local_cuts, "-")), string(get(r, :menu_size, "-")), time() - t0)
    flush(stdout)
end
