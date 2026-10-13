"""
scale_oracle_random.jl — 무작위 location 인스턴스 (Goyal §7.3 최소 설정, pilot_loc_random_v4.jl 과 같음) 에서 고정 x̄ 의 oracle 을
α-B&B (nz_alpha_bnb_dr) 로 풀고 진행 로그와 시간 분해를 본다. scale_oracle.jl 의 무작위 인스턴스판 (분기 차원 n_h = 50).

환경변수: NZ_SEED = 1  SR_S = 3  SR_X = "1"  SR_NLOC = 15 (위치 수 d, 고객 = d − 5, pair 면 n_h = 5 (d − 5), 용량 150 (d − 5) / 2)  SR_RES = pair  SR_TL = 600  SR_GAP = 1e-3  SR_WORKERS = 12  SR_QUOTA = 300
          SR_KW = (α-B&B 인자, 예 "(dd=false,)")
실행: julia -t 14,1 nonzero_sum/scale_oracle_random.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d)
seed = parse(Int, envf("NZ_SEED", "1")); S = parse(Int, envf("SR_S", "3")); res = Symbol(envf("SR_RES", "pair"))
TL = parse(Float64, envf("SR_TL", "600")); GAP = parse(Float64, envf("SR_GAP", "1e-3")); nw = parse(Int, envf("SR_WORKERS", "12"))
xi = isempty(strip(envf("SR_X", "1"))) ? Int[] : parse.(Int, split(envf("SR_X", "1"), ","))
kw = eval(Meta.parse(envf("SR_KW", "NamedTuple()")))
nloc = parse(Int, envf("SR_NLOC", "15")); cap = 150.0 * (nloc - 5) / 2
nd = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=2, nB=3, NB=3, S=S, seed=seed,
    bA=cap, bB=cap, qB=cap, dlo=50.0, dhi=150.0, reservation=res, quota=parse(Float64, envf("SR_QUOTA", "300")), wres=300.0,
    eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
x̄ = zeros(nd.nx); x̄[xi] .= 1
nact = count(nz_presolve_hU(nd, x̄)[1] .> 0)
@printf("scale_oracle_random: %s d=%d seed=%d S=%d x=%s n_h=%d (활성 %d) θᵁ=%.1f TL=%.0fs gap=%.0e kw=%s\n", res, nloc, seed, S, string(xi), nz_nh(nd), nact,
        nd.thetaU, TL, GAP, string(kw)); flush(stdout)
t = @elapsed r = nz_alpha_bnb_dr(nd, x̄; nworkers=nw, time_limit=TL, rel_gap=GAP, verbose=true, log_every=30.0, ipopt_time=120.0, kw...)
n = max(r[:nodes], 1)
@printf("RESULT %s LB=%.4f UB=%.4f gap=%.3e time=%.1f nodes=%d rootUB=%.2f | worker 합 %.0fs = 완화 LP %.0f + 범위 MIP %.0f (MIP %d, 노드당 %d) + 재확인 %.0f + α 고정 LP %.0f | 고정비율 %.2f 실행불가능 %d (되살아남 %d) 수치오류 %d (실패 %d) 상보성 분기 %d GC 대기 %.0fs | 느린 노드 %s\n",
        r[:is_exact] ? "Optimal" : "TimeLim", r[:LB], r[:UB], (r[:UB] - r[:LB]) / max(1, abs(r[:LB])), t, r[:nodes], r[:root_UB],
        t * nw, r[:t_relax], r[:t_range], r[:range_mips], round(Int, r[:range_mips] / n), r[:t_recheck], r[:t_eval],
        r[:fixfrac], r[:infeas], r[:infeas_recovered], r[:numerr], r[:numerr_fail], r[:cbranch], r[:gc_wait],
        string([round.(t; digits=0) for t in r[:slow_nodes][1:min(4, end)]]))
