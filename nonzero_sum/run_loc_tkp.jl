"""
run_loc_tkp.jl — 원고 6.1 입지 인스턴스 (loc_tkp.jl) 를 Algorithm 1 로 푼다 (2026-11-20, 계산 성능 표 L1·L2 용).

방법은 pilot_pooled_store.jl (L1~L6 의 Ours) 과 같다: nz_belief_menu_benders, local cut 먼저, α-B&B (bnb = :dr), 허용오차 0.5%,
oracle gap 1e-3, 반응 정의역 프리솔브. 모형은 원고 표 tab:loc-models:
  nominal  ε̂ = ε̃ = 0        data  ε̂ = r, ε̃ = 0        belief  ε̂ = 0, ε̃ = r
  coupled  ε̂ = ε̃ = r, δ = LT_DELTA (기본 r/3)          product  ε̂ = ε̃ = r, δ 없음
  상자: 볼마다 |q_s − q̂_s| ≤ ε/√S (LT_BOX = 0 이면 끔)

환경변수: LT_G = "1" (지리, 쉼표 목록)   LT_SEEDS = "1"   LT_S = "10"   LT_MODEL = coupled   LT_R = 0.2   LT_DELTA = (r/3)
          LT_BETA = 0.4   LT_TIERS = "200,300,400"   LT_NB = 3   LT_LAMBDA = 100   LT_TL = 3600   LT_WORKERS = 12   LT_PRE = 1   LT_BOX = 1
          LT_ORACLE = alpha_bnb | omega_grb (Ours+G: 전역 단계를 Gurobi Ω 로)   LT_BELIEF = 1 | 0 (Ours−B)   LT_BNBKW = "" (α-B&B 인자, pilot 과 같은 형식)
실행: julia -t 14,1 nonzero_sum/run_loc_tkp.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
include(joinpath(root, "nonzero_sum", "nz_root_tighten.jl"))
envf(k, d) = get(ENV, k, d); pf(k, d) = parse(Float64, envf(k, d)); pint(k, d) = parse(Int, envf(k, d))
strs(k, d) = String.(strip.(split(envf(k, d), ",")))
ORC = Symbol(envf("LT_ORACLE", "alpha_bnb")); ORC in (:alpha_bnb, :omega_grb) || error("LT_ORACLE = alpha_bnb | omega_grb")
ORC == :omega_grb && include(joinpath(root, "nonzero_sum", "nz_bd_kkt.jl"))
BEL = envf("LT_BELIEF", "1"); BEL in ("0", "1") || error("LT_BELIEF = 1 | 0"); BEL = BEL == "1"
MODEL = Symbol(envf("LT_MODEL", "coupled")); MODEL in (:nominal, :data, :belief, :coupled, :product) || error("LT_MODEL = nominal|data|belief|coupled|product")
geos = parse.(Int, strs("LT_G", "1")); seeds = parse.(Int, strs("LT_SEEDS", "1")); Ss = parse.(Int, strs("LT_S", "10"))
r = pf("LT_R", "0.2"); δc = haskey(ENV, "LT_DELTA") ? pf("LT_DELTA", "0") : r / 3; βL = pf("LT_BETA", "0.4")
tiers = parse.(Float64, strs("LT_TIERS", "200,300,400")); NB = pint("LT_NB", "3"); lam = pf("LT_LAMBDA", "100")
TL = pf("LT_TL", "3600"); nw = pint("LT_WORKERS", "12"); PRE = envf("LT_PRE", "1") == "1"; BOX = envf("LT_BOX", "1") == "1"
GRBK() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), nw); o)
ε̂, ε̃, δ = MODEL == :nominal ? (0.0, 0.0, Inf) : MODEL == :data ? (r, 0.0, Inf) : MODEL == :belief ? (0.0, r, Inf) :
          MODEL == :coupled ? (r, r, δc) : (r, r, Inf)
mk(g, S, seed) = make_loc_tkp(; g=g, S=S, seed=seed, eps_hat=ε̂, eps_tilde=ε̃, delta=δ, beta=βL, box=BOX, tiers=tiers, NB=NB, lambdaU=lam)

const _RT_CACHE = Dict{Any,Tuple{Vector{Float64},Int}}()
if PRE          # pilot_pooled_store.jl 과 같은 반응 정의역 프리솔브 (x̄ 마다 한 번)
    function nz_presolve_hU(nd::NZData, x̄::Vector{Float64})
        key = (nd.name, nd.S, copy(nd.hU), copy(x̄))
        haskey(_RT_CACHE, key) && return _RT_CACHE[key]
        hU1, n1 = invoke(nz_presolve_hU, Tuple{NZData,Any}, nd, x̄)
        hU2, n2, t = nz_response_presolve(nz_with_hU(nd, hU1), x̄; env=_nz_envs(1)[1])
        @printf("    [반응 프리솔브] x=%s: 활성 %d 중 %d 개를 0 으로 고정 (%.1fs)\n", loc_tkp_xstr(nd, x̄), count(hU1 .> 0), n2, t); flush(stdout)
        return (_RT_CACHE[key] = (hU2, n1 + n2))
    end
end
function _lt_kw(str)
    kv = Pair{Symbol,Any}[]
    for t in split(str, ","; keepempty=false)
        a, b = strip.(split(t, "="))
        push!(kv, Symbol(a) => (b == "true" ? true : b == "false" ? false : occursin(r"^[0-9]+$", b) ? parse(Int, b) : parse(Float64, b)))
    end
    return (; kv...)
end
kw = merge((ipopt_time=120.0,), _lt_kw(envf("LT_BNBKW", "")))
nX(nd) = sum(binomial(nd.meta[:nB], n) * nd.meta[:K]^n for n in 0:nd.gamma)
for g in geos, seed in seeds, S in sort(Ss)
    nd = mk(g, S, seed)
    @printf("\n--- loc_tkp g=%d seed=%d S=%d model=%s r=%.2f (ε̂=%.2f ε̃=%.2f δ=%s) β=%.2f 상자 %s, 크기 %s, N_B≤%d, |X|=%d, n_h=%d, θᵁ=%.4f λᵁ=%.0f, TL=%.0fs, oracle=%s, belief cut %s, 프리솔브 %s\n",
            g, seed, S, MODEL, r, ε̂, ε̃, isfinite(δ) ? @sprintf("%.4f", δ) : "없음", βL, BOX ? "켬" : "끔", join(Int.(tiers), "·"), NB, nX(nd),
            nz_nh(nd), nd.thetaU, nd.lambdaU, TL, ORC, BEL ? "켬" : "끔", PRE ? "켬" : "끔")
    flush(stdout)
    t0 = time()
    res = nz_belief_menu_benders(nd; optimizer=GRB, tol=5e-3, oracle=ORC, bnb=:dr, oracle_gap=1e-3, nworkers=nw, time_limit=TL,
                                 local_first=true, local_time=60.0, bnb_kw=kw, verbose=true, global_optimizer=GRBK, belief_cuts=BEL)
    gap = abs(res[:UB] - res[:LB]) / max(1.0, abs(res[:UB]))
    @printf("RESULT loc_tkp g=%d seed=%d S=%d model=%s r=%.2f delta=%s beta=%.2f status=%s x=%s LB=%.6f UB=%.6f gap=%.2e iters=%d global=%d local=%s menu=%s time=%.1f oracle=%s belief=%d\n",
            g, seed, S, MODEL, r, isfinite(δ) ? @sprintf("%.4f", δ) : "Inf", βL, res[:status], loc_tkp_xstr(nd, res[:x]), res[:LB], res[:UB], gap,
            res[:iters], res[:oracle_calls], string(get(res, :local_cuts, "-")), string(get(res, :menu_size, "-")), time() - t0, ORC, BEL ? 1 : 0)
    flush(stdout)
end
