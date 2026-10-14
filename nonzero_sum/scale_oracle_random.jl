"""
scale_oracle_random.jl — 무작위 location 인스턴스 (Goyal §7.3 최소 설정, pilot_loc_random_v4.jl 과 같음) 에서 고정 x̄ 의 oracle 을
α-B&B (nz_alpha_bnb_dr) 로 풀고 진행 로그와 시간 분해를 본다. scale_oracle.jl 의 무작위 인스턴스판 (분기 차원 n_h = 50).

환경변수: NZ_SEED = 1  SR_S = 3  SR_X = "1"  SR_NLOC = 15 (위치 수 d, 고객 = d − 5, pair 면 n_h = 5 (d − 5), 용량 150 (d − 5) / 2)  SR_RES = pair  SR_TL = 600  SR_GAP = 1e-3  SR_WORKERS = 12  SR_QUOTA = 300
          SR_KW = (α-B&B 인자, 예 "(dd=false,)")
          SR_SOLVER = dr (기본: nz_alpha_bnb_dr) | legacy (옛 구현 nz_alpha_bnb, 쌍대 범위 없음. SR_KW = "(varpi_branch=true,)" 로 ϖ 분기)
          SR_PRE = none (기본) | resp (반응 정의역 프리솔브 nz_root_tighten.jl 을 먼저 적용해 0 으로 고정된 좌표를 뺀다)
          SR_ZS = 0 | 1 (1: 같은 데이터에서 리더 손실만 follower 의 recourse 값으로 바꾼 대조군: ℓ = c, θᵁ = 0. 리더 블록에 recourse 쌍대해 ϖ 와의 곱이 없어진다
                  — zero-sum 과 같은 구조. 분기 차원·시나리오·follower 블록은 그대로라 "비제로섬에서만 안 닫히는 원인" 을 가른다)
          SR_HU = quota (기본: 예약 상한 = 점포별 할당량) | demand (예약 상한을 고객의 최대 수요로 줄임: h_ij > max_s ξ_j^s 인 예약은 어느 시나리오에서도
                  쓰이지 않고 예약비 f > 0 만 내므로 follower 의 최적 반응이 될 수 없다. nz_presolve_hU 와 같은 종류의 지배 논증)
실행: julia -t 14,1 nonzero_sum/scale_oracle_random.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
include(joinpath(root, "nonzero_sum", "nz_root_tighten.jl"))
envf(k, d) = get(ENV, k, d)
seed = parse(Int, envf("NZ_SEED", "1")); S = parse(Int, envf("SR_S", "3")); res = Symbol(envf("SR_RES", "pair"))
TL = parse(Float64, envf("SR_TL", "600")); GAP = parse(Float64, envf("SR_GAP", "1e-3")); nw = parse(Int, envf("SR_WORKERS", "12"))
xi = isempty(strip(envf("SR_X", "1"))) ? Int[] : parse.(Int, split(envf("SR_X", "1"), ","))
kw = eval(Meta.parse(envf("SR_KW", "NamedTuple()")))
nloc = parse(Int, envf("SR_NLOC", "15")); cap = 150.0 * (nloc - 5) / 2
nd = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=2, nB=3, NB=3, S=S, seed=seed,
    bA=cap, bB=cap, qB=cap, dlo=50.0, dhi=150.0, reservation=res, quota=parse(Float64, envf("SR_QUOTA", "300")), wres=300.0,
    eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
humode = envf("SR_HU", "quota"); humode in ("quota", "demand") || error("SR_HU = quota | demand")
if humode == "demand"
    mt = nd.meta; mt[:reservation] == :pair || error("SR_HU = demand 는 pair 예약 전용")
    dmaxj = [maximum(mt[:xi][j, :]) for j in 1:mt[:ncu]]
    global nd = nz_with_hU(nd, [min(nd.hU[mt[:hidx](i, j)], dmaxj[j]) for i in 1:mt[:nst] for j in 1:mt[:ncu]])
end
if envf("SR_ZS", "0") == "1"
    # ℓ = c 면 리더 복사본이 cᵀŷ 를 최대화하므로 recourse 최적성이 저절로 성립 → θ 벌점 (과 α·ϖ 곱) 이 필요 없다. 리더 쪽 x 행 상계는 follower 의 것과 같다
    global nd = NZData(nd.name * "_zs", nd.A, nd.c, copy(nd.c), nd.u, nd.hrow, nd.hcoef, nd.xrow, nd.g, nd.W, nd.wvec, nd.c0, nd.hU, nd.nx, nd.x_allowed, nd.gamma, nd.qx,
                       nd.S, nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta, 0.0, copy(nd.piFU), nd.piFU, nd.lambdaU, nd.varpiU, nd.meta)
end
x̄ = zeros(nd.nx); x̄[xi] .= 1
solver = envf("SR_SOLVER", "dr"); solver in ("dr", "legacy") || error("SR_SOLVER = dr | legacy")
if envf("SR_PRE", "none") == "resp"
    hU1, _ = nz_presolve_hU(nd, x̄)
    hU2, nz0, tp = nz_response_presolve(nz_with_hU(nd, hU1), x̄; env=_nz_envs(1)[1])
    @printf("반응 정의역 프리솔브: 활성 %d 중 %d 개를 0 으로 고정 (%.1fs)\n", count(hU1 .> 0), nz0, tp)
    global nd = nz_with_hU(nd, hU2)
end
nact = count(nz_presolve_hU(nd, x̄)[1] .> 0)
@printf("scale_oracle_random: %s d=%d seed=%d S=%d x=%s n_h=%d (활성 %d) θᵁ=%.1f Σhᵁ=%.0f (%s) TL=%.0fs gap=%.0e solver=%s kw=%s\n", res, nloc, seed, S, string(xi), nz_nh(nd), nact,
        nd.thetaU, sum(nz_presolve_hU(nd, x̄)[1]), humode, TL, GAP, solver, string(kw)); flush(stdout)
if solver == "legacy"
    tl = @elapsed rl = nz_alpha_bnb(nd, x̄; nworkers=nw, time_limit=TL, rel_gap=GAP, verbose=true, log_every=30.0, ipopt_time=120.0, kw...)
    @printf("RESULT %s LB=%.4f UB=%.4f gap=%.3e time=%.1f nodes=%d rootUB=%.2f (legacy)\n", rl[:is_exact] ? "Optimal" : "TimeLim", rl[:LB], rl[:UB],
            (rl[:UB] - rl[:LB]) / max(1, abs(rl[:LB])), tl, rl[:nodes], rl[:root_UB])
    exit()
end
t = @elapsed r = nz_alpha_bnb_dr(nd, x̄; nworkers=nw, time_limit=TL, rel_gap=GAP, verbose=true, log_every=30.0, ipopt_time=120.0, kw...)
n = max(r[:nodes], 1)
@printf("RESULT %s LB=%.4f UB=%.4f gap=%.3e time=%.1f nodes=%d rootUB=%.2f | worker 합 %.0fs = 완화 LP %.0f + 범위 MIP %.0f (MIP %d, 노드당 %d) + 재확인 %.0f + α 고정 LP %.0f | 고정비율 %.2f 실행불가능 %d (되살아남 %d) 수치오류 %d (실패 %d) 상보성 분기 %d GC 대기 %.0fs | 느린 노드 %s\n",
        r[:is_exact] ? "Optimal" : "TimeLim", r[:LB], r[:UB], (r[:UB] - r[:LB]) / max(1, abs(r[:LB])), t, r[:nodes], r[:root_UB],
        t * nw, r[:t_relax], r[:t_range], r[:range_mips], round(Int, r[:range_mips] / n), r[:t_recheck], r[:t_eval],
        r[:fixfrac], r[:infeas], r[:infeas_recovered], r[:numerr], r[:numerr_fail], r[:cbranch], r[:gc_wait],
        string([round.(t; digits=0) for t in r[:slow_nodes][1:min(4, end)]]))
