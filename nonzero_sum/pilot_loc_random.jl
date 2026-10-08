"""
pilot_loc_random.jl — 원고 6.1 계산 실험 (Table loc-comp) 의 설정을 정하기 위한 파일럿. Goyal et al. (2023) §7.3 무작위 인스턴스.

Goyal §7.3 최소 설정: 위치 d = 15 ([0,1]² 균등), 적격 후보지 5 곳 중 A 점유 2 곳 → B 후보 3, N_B = 3 (|𝒳| = 8),
고객 = 나머지 10 곳 (수요 U[50,150]), 적격지 용량 b_i = 150 (d − 5) / 2 = 750, 출점비 750.
예약 확장: 거리 해상도 0.1, p = 0.3, f = 0.1 (무작위 좌표용 기본값), pair (n_h = 5 × 10 = 50) 또는 pooled (n_h = 5).

방법: E (모든 x 를 KKT 독립 평가, x 하나라도 시간 제한이면 미해결) 과 Algorithm 2 (belief-menu + α-B&B).
작은 S 부터. 어떤 (예약 방식, seed, 방법) 이 시간 제한에 걸리면 그보다 큰 S 는 그 방법으로 돌리지 않는다.

환경변수
  PL_SEEDS = "1,2"   PL_S = "3,5,10"   PL_RES = "pooled,pair"   PL_METHODS = "E,ALG"
  PL_QUOTA = 300 (pair 의 점포별 선판매 한도)   PL_WRES = 300 (pooled 의 예약 총한도)
  PL_EPS = 0.3   PL_BETA = 0.4   PL_DELTA = 0.1 (Inf = product)
  PL_E_TL = 300 (E 의 x 하나당)   PL_TL = 3600 (Algorithm 2 전체)   PL_ORACLE_TIME = 600   PL_BOOST = 2400
  PL_LOCAL_WLS = 0 (기본: PL_WORKERS=12, 휴리스틱 켬) | 1 이면 worker 1, 휴리스틱 끔 (worker 제한 PC)
실행: julia -t 14,1 nonzero_sum/pilot_loc_random.jl      (PL_LOCAL_WLS=1 이면 -t 3,1)
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
envf(k, d) = get(ENV, k, d)
pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
strs(k, d) = String.(strip.(split(envf(k, d), ",")))

seeds   = parse.(Int, strs("PL_SEEDS", "1,2"))
Ss      = parse.(Int, strs("PL_S", "3,5,10"))
ress    = Symbol.(strs("PL_RES", "pooled,pair"))
methods = strs("PL_METHODS", "E,ALG")
quota   = parse(Float64, envf("PL_QUOTA", "300")); wres = parse(Float64, envf("PL_WRES", "300"))
ε = parse(Float64, envf("PL_EPS", "0.3")); β = parse(Float64, envf("PL_BETA", "0.4")); δ = pd(envf("PL_DELTA", "0.1"))
E_TL = parse(Float64, envf("PL_E_TL", "300")); TL = parse(Float64, envf("PL_TL", "3600"))
local_wls = envf("PL_LOCAL_WLS", "0") == "1"
nw = local_wls ? 1 : parse(Int, envf("PL_WORKERS", "12"))
alg_kw = (optimizer=GRB, tol=1e-4, oracle=:alpha_bnb, nworkers=nw, time_limit=TL,
          oracle_time=min(parse(Float64, envf("PL_ORACLE_TIME", "600")), TL),
          boost_time=min(parse(Float64, envf("PL_BOOST", "2400")), TL),
          bnb_kw=local_wls ? (envs=[GRB_ENV], heuristic=false) : NamedTuple(), verbose=false)

mk(res, S, seed) = make_location_instance(; coords=:random, n_loc=15, nA=2, nB=3, NB=3, S=S, seed=seed,
    bA=750.0, bB=750.0, qB=750.0, dlo=50.0, dhi=150.0, reservation=res, quota=quota, wres=wres,
    eps_hat=ε, eps_tilde=ε, beta=β)
all_x(nd) = [Float64.(collect(bits)) for bits in Iterators.product(fill(0:1, nd.nx)...)
             if sum(bits) <= nd.gamma && all(nd.x_allowed[i] || bits[i] == 0 for i in 1:nd.nx)]
xstr(x) = "{" * join(findall(x .> 0.5), ",") * "}"

println("="^100)
@printf("pilot (Goyal §7.3 최소: d=15, A 2, B 후보 3, N_B=3)  ε̂=ε̃=%.2f β=%.2f δ=%s  E TL/x=%.0fs  ALG TL=%.0fs  workers=%d\n",
        ε, β, string(δ), E_TL, TL, nw)
println("="^100); flush(stdout)

given_up = Set{Tuple{Symbol,Int,String}}()          # (예약 방식, seed, 방법) — 더 큰 S 는 건너뜀
for res in ress, seed in seeds, S in sort(Ss)
    nd = nz_with_delta(mk(res, S, seed), δ)
    @printf("\n--- %s seed=%d S=%d: n_h=%d, |X|=%d, θᵁ=%.1f λᵁ=%.1f\n", res, seed, S, nz_nh(nd), length(all_x(nd)),
            nd.thetaU, nd.lambdaU); flush(stdout)
    if "E" in methods
        if (res, seed, "E") in given_up
            println("  E: 작은 S 에서 시간 제한 → 건너뜀")
        else
            t0 = time(); best = (Inf, Float64[]); solved = true
            for x in all_x(nd)
                r = try nz_kkt_value(nd, x; optimizer=GRB, time_limit=E_TL) catch e; (println("  KKT 오류: ", e); nothing) end
                if r === nothing || r[:status] != MOI.OPTIMAL
                    solved = false
                    @printf("  E: x=%s 에서 %s (%.0fs) → 미해결, 이 seed 의 더 큰 S 는 E 생략\n", xstr(x),
                            r === nothing ? "오류" : string(r[:status]), time() - t0)
                    break
                end
                v = dot(nd.qx, x) + r[:value]
                v < best[1] && (best = (v, x))
            end
            t = time() - t0
            solved || push!(given_up, (res, seed, "E"))
            @printf("RESULT method=E res=%s seed=%d S=%d nh=%d solved=%s x=%s value=%.6f time=%.1f\n", res, seed, S, nz_nh(nd),
                    solved, solved ? xstr(best[2]) : "-", solved ? best[1] : NaN, t)
            flush(stdout)
        end
    end
    if "ALG" in methods
        if (res, seed, "ALG") in given_up
            println("  ALG: 작은 S 에서 시간 제한 → 건너뜀")
        else
            t0 = time()
            r = nz_belief_menu_benders(nd; alg_kw...)
            t = time() - t0
            solved = r[:status] == :Optimal
            solved || push!(given_up, (res, seed, "ALG"))
            gap = abs(r[:UB] - r[:LB]) / max(1.0, abs(r[:UB]))
            @printf("RESULT method=ALG res=%s seed=%d S=%d nh=%d status=%s x=%s LB=%.6f UB=%.6f gap=%.2e iters=%d oracle=%d menu=%s time=%.1f\n",
                    res, seed, S, nz_nh(nd), r[:status], xstr(r[:x]), r[:LB], r[:UB], gap, r[:iters], r[:oracle_calls],
                    string(get(r, :menu_size, "-")), t)
            flush(stdout)
        end
    end
end
