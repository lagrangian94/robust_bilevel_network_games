"""
사본 (2026-11-06): 저장소 nonzero_sum/sgb15_insample.jl 에 지역 요인 수요 (fac_builder.jl 의 make_sgb15_fac) 만 끼움.
  SG_DEM = region (기본) | iid. make_sgb15_fac 는 dist = euclid, 해상도 10, 한도 300 고정이라 SG_DIST·SG_DISTRES·SG_QUOTA 는 그 값이어야 함.
  나머지 (Benders·α-B&B·프리솔브·허용오차) 는 저장소 그대로.

sgb15_insample.jl — SGB15 (Yanıkoğlu–Kuhn 2018 §7) 모형 가치 인스턴스의 in-sample 풀이 (2026-11-05, docs/robustness_value_todo.md).

인스턴스: make_sgb15_instance (nz_data.jl). 인구 상위 15 개 도시, A = Saint Louis·San Jose, B 는 나머지 13 곳 중 ≤ 3 (|𝒳| = 378),
  수요 U[μ/2, 3μ/2] 독립 S 개 (q̂ 균등), 점포별 예약 풀 (한도 SG_QUOTA), 예약비 f = 50·현장 프리미엄 p = 200 (SGB 척도 기본값), 거리 SG_DIST.
모형 (SG_MODEL): nominal (ε̂ = ε̃ = 0) | datadr (ε, 0) | product (ε, ε, δ = ∞) | coupled (ε, ε, δ = SG_DELTA). 리더 CVaR 수준 β = SG_BETA.
방법: 계산 실험 (pilot_pooled_store.jl) 과 같은 Algorithm 2 — belief cut + local cut + α-B&B (기본 구성) + 반응 정의역 프리솔브.
  허용오차 SG_TOL (기본 5e-3 = 계산 실험과 같음). oracle gap 1e-3 은 Benders 가 tol 에 맞춰 더 조인다.
  거리 해상도 SG_DISTRES 마일 (기본 10). 1 마일이면 follower 비용이 거의 같은 대안이 많아 θᵁ = v/해상도 = 5 가 되고
  ε̃ > 0 모형의 α-B&B 상한이 매우 느슨함 (2026-11-05 스모크: Product·Coupled 가 1 시간에 gap 1~2%, 같은 활성 차원의 L1·L3·L5 는 30~54 초).
  θᵁ = 정확한 circuit 값 (nz_theta_circuit_exact). λᵁ = SG_LAMBDA (기본 auto = 40·θᵁ, L1~L6 의 2000/50 과 같은 비율). λ 무관 인증은 따로.

환경변수: SG_MODEL = nominal,datadr,product,coupled (쉼표로 여러 개, 차례로 풂)   SG_SEEDS = 1   SG_S = 10   SG_EPS = 0.3   SG_DELTA = 0.1
          SG_BETA = 0.4   SG_DIST = euclid | road   SG_DISTRES = 10   SG_QUOTA = 300   SG_TOL = 5e-3   SG_TL = 3600   SG_WORKERS = 12
          SG_LAMBDA = auto (40·θᵁ) | 숫자   SG_PRE = 1
          SG_MAXITER = 20000 (Benders 반복 한도. menu cut 이 x̄ 마다 붙어 반복 수가 |𝒳| × belief 수 정도라, 기본 500 이면 |𝒳| = 378 에서 oracle 전에 멈춤 — 2026-11-05 스모크)
출력: 표준출력의 RESULT 줄 (모형·시드마다 한 줄, x 는 B 후보 번호와 도시 이름). 큐에서는 모형·시드마다 logs/sgb15/S<S>/<model>_s<seed>.log 로 받는다.
실행: julia -t 14,1 nonzero_sum/sgb15_insample.jl
"""
root = dirname(@__DIR__)
const SPD = joinpath(@__DIR__, "logs", "sgb15_fac") * "/"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Distributions
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
include(joinpath(root, "nonzero_sum", "nz_root_tighten.jl"))
include(joinpath(@__DIR__, "fac_builder.jl"))
envf(k, d) = get(ENV, k, d); pf(k, d) = parse(Float64, envf(k, d)); pint(k, d) = parse(Int, envf(k, d))
strs(k, d) = String.(strip.(split(envf(k, d), ",")))
models = strs("SG_MODEL", "nominal,datadr,product,coupled")
all(m -> m in ("nominal", "datadr", "product", "coupled"), models) || error("SG_MODEL = nominal | datadr | product | coupled (쉼표로 여러 개)")
seeds = parse.(Int, strs("SG_SEEDS", "1")); S = pint("SG_S", "10")
ε = pf("SG_EPS", "0.3"); δ = pf("SG_DELTA", "0.1"); βL = pf("SG_BETA", "0.4")
DIST = Symbol(envf("SG_DIST", "euclid")); DIST in (:euclid, :road) || error("SG_DIST = euclid | road")
quota = pf("SG_QUOTA", "300"); tol = pf("SG_TOL", "5e-3"); TL = pf("SG_TL", "3600"); nw = pint("SG_WORKERS", "12")
dres = pf("SG_DISTRES", "10"); LAM = envf("SG_LAMBDA", "auto"); PRE = envf("SG_PRE", "1") == "1"; maxit = pint("SG_MAXITER", "20000")
DEM = Symbol(envf("SG_DEM", "region")); DEM in (:region, :iid) || error("SG_DEM = region | iid")
(DIST == :euclid && dres == 10.0 && quota == 300.0) || error("make_sgb15_fac 는 euclid·해상도 10·한도 300 고정")
GRBK() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), nw); o)
const _RT_CACHE = Dict{Any,Tuple{Vector{Float64},Int}}()
if PRE          # pilot_pooled_store.jl 과 같음: nz_alpha_bnb_dr 의 프리솔브 호출에 반응 정의역 프리솔브를 끼운다 (x̄ 마다 한 번)
    function nz_presolve_hU(nd::NZData, x̄::Vector{Float64})
        key = (nd.name, nd.S, copy(nd.hU), copy(x̄))
        haskey(_RT_CACHE, key) && return _RT_CACHE[key]
        hU1, n1 = invoke(nz_presolve_hU, Tuple{NZData,Any}, nd, x̄)
        hU2, n2, t = nz_response_presolve(nz_with_hU(nd, hU1), x̄; env=_nz_envs(1)[1])
        @printf("    [반응 프리솔브] x=%s: 활성 %d 중 %d 개를 0 으로 고정 (%.1fs)\n", string(findall(x̄ .> 0.5)), count(hU1 .> 0), n2, t); flush(stdout)
        return (_RT_CACHE[key] = (hU2, n1 + n2))
    end
end

"모형 이름 → (ε̂, ε̃, δ)"
model_eps(m) = m == "nominal" ? (0.0, 0.0, Inf) : m == "datadr" ? (ε, 0.0, Inf) : m == "product" ? (ε, ε, Inf) : (ε, ε, δ)
function mk(model, seed)
    eh, et, d = model_eps(model)
    nd = nz_with_delta(make_sgb15_fac(; S=S, seed=seed, dem=DEM, eps_hat=eh, eps_tilde=et, beta=βL), d)
    θ = nz_theta_circuit_exact(nd)[1]
    return nz_with_bounds(nd; thetaU=θ, lambdaU=(LAM == "auto" ? 40θ : parse(Float64, LAM)))
end
xstr(x) = "{" * join(findall(x .> 0.5), ",") * "}"
"B 후보 번호 → 도시 이름 (빈칸은 _, RESULT 줄을 공백으로 나눌 수 있게)"
xcity(nd, x) = isempty(findall(x .> 0.5)) ? "-" :
    join([replace(nd.meta[:cities][nd.meta[:nA] + i], " " => "_") for i in findall(x .> 0.5)], ";")
for seed in seeds, model in models
    nd = mk(model, seed); eh, et, d = model_eps(model)
    @printf("\n--- SGB15 %s (ε̂=%.2f ε̃=%.2f δ=%s β=%.2f) seed=%d S=%d: 거리 %s (해상도 %.0f 마일), n_h=%d, |X|=%d, A 용량 %.1f, B 용량 %.1f, 출점비 %.0f, f=%.0f p=%.0f, 예약 한도 %.0f, θᵁ=%.1f λᵁ=%.1f, tol=%.0e, TL=%.0fs, 프리솔브 %s\n",
            model, eh, et, string(d), βL, seed, S, DIST, dres, nz_nh(nd), sum(binomial(nd.nx, k) for k in 0:nd.gamma), nd.meta[:bA], nd.meta[:bB],
            nd.qx[1], nd.meta[:f], nd.meta[:p], quota, nd.thetaU, nd.lambdaU, tol, TL, PRE ? "켬" : "끔")
    println("    B 후보 (x 번호 순): ", join(nd.meta[:cities][nd.meta[:nA]+1:nd.meta[:nst]], " / ")); flush(stdout)
    t0 = time()
    r = nz_belief_menu_benders(nd; optimizer=GRB, tol=tol, max_iter=maxit, oracle=:alpha_bnb, bnb=:dr, oracle_gap=1e-3, nworkers=nw, time_limit=TL,
                               local_first=true, local_time=60.0, bnb_kw=(ipopt_time=120.0,), verbose=true, global_optimizer=GRBK, belief_cuts=true)
    gap = abs(r[:UB] - r[:LB]) / max(1.0, abs(r[:UB]))
    @printf("RESULT sgb15fac dem=%s model=%s epsh=%.2f epst=%.2f delta=%s beta=%.2f dist=%s distres=%.0f quota=%.0f S=%d seed=%d status=%s x=%s cities=%s LB=%.6f UB=%.6f gap=%.2e iters=%d global=%d local=%s menu=%s theta=%.2f lambda=%.1f tol=%.0e time=%.1f\n",
            string(DEM), model, eh, et, string(d), βL, DIST, dres, quota, S, seed, r[:status], xstr(r[:x]), xcity(nd, r[:x]), r[:LB], r[:UB], gap, r[:iters],
            r[:oracle_calls], string(get(r, :local_cuts, "-")), string(get(r, :menu_size, "-")), nd.thetaU, nd.lambdaU, tol, time() - t0)
    flush(stdout)
end
