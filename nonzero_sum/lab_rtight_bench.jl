"""
lab_rtight_bench.jl — 반응 정의역 프리솔브 (nz_root_tighten.jl) 를 α-B&B 에 **시험용으로** 연결해 (1) 유효성 (KKT 값과 대조), (2) 기본 인스턴스 벤치마크 (ALGL) 를 본다.

연결 방법: 기존 파일을 고치지 않고, nz_alpha_bnb_dr 가 부르는 프리솔브 nz_presolve_hU(nd, x̄) 에 더 구체적인 메서드 (x̄::Vector{Float64}) 를 이 파일에서 더한다.
그 메서드가 기존 프리솔브 (닫힌 점포) 를 부른 뒤 nz_response_presolve 를 적용한다. x̄ 마다 한 번만 계산 (같은 x̄ 의 재호출·이어 풀기는 저장한 값).
RB_RT = 0 이면 연결하지 않는다 (기준 실행).

  RB_MODE = check : S = RB_S (기본 3) 에서 모든 x 를 KKT 로 평가한 값과 α-B&B (프리솔브 연결) 의 [LB, UB] 를 대조
  RB_MODE = bench : run_loc_base_scaling.jl 의 ALGL 과 같은 인자로 Benders (허용오차 0.5%, oracle gap 1e-3, Ipopt 120 s, local 60 s)

환경변수: NZ_SEED = 1  RB_S = 20  RB_TL = 3600  RB_WORKERS = 12  RB_RT = 1  RB_MODE = bench
실행: julia -t 14,1 nonzero_sum/lab_rtight_bench.jl
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
seed = parse(Int, envf("NZ_SEED", "1")); TL = parse(Float64, envf("RB_TL", "3600")); nw = parse(Int, envf("RB_WORKERS", "12"))
RT = envf("RB_RT", "1") == "1"; mode = envf("RB_MODE", "bench")
S = parse(Int, envf("RB_S", mode == "check" ? "3" : "20"))

const _RT_CACHE = Dict{Any,Tuple{Vector{Float64},Int}}()
const _RT_STAT = Ref((calls=0, time=0.0, fixed=0))
if RT
    # nz_alpha_bnb_dr 의 프리솔브 호출을 가로챈다 (x̄ 가 Vector{Float64} 인 호출만. 기존 메서드는 그대로 남는다)
    function nz_presolve_hU(nd::NZData, x̄::Vector{Float64})
        key = (nd.name, nd.S, copy(nd.hU), copy(x̄))
        haskey(_RT_CACHE, key) && return _RT_CACHE[key]
        hU1, n1 = invoke(nz_presolve_hU, Tuple{NZData,Any}, nd, x̄)
        hU2, n2, t = nz_response_presolve(nz_with_hU(nd, hU1), x̄; env=_nz_envs(1)[1])
        _RT_STAT[] = (calls=_RT_STAT[].calls + 1, time=_RT_STAT[].time + t, fixed=_RT_STAT[].fixed + n2)
        @printf("    [반응 프리솔브] x=%s: 활성 %d 중 %d 개를 0 으로 고정 (%.1fs)\n", string(findall(x̄ .> 0.5)), count(hU1 .> 0), n2, t); flush(stdout)
        return (_RT_CACHE[key] = (hU2, n1 + n2))
    end
end
function mk(S)
    nd = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=seed,
                                              eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
    return nz_with_bounds(nd; thetaU=nz_theta_circuit_exact(nd)[1])
end
all_x(nd) = [Float64.(collect(bits)) for bits in Iterators.product(fill(0:1, nd.nx)...)
             if sum(bits) <= nd.gamma && all(nd.x_allowed[i] || bits[i] == 0 for i in 1:nd.nx)]
xstr(x) = "{" * join(findall(x .> 0.5), ",") * "}"
nd = mk(S)
@printf("lab_rtight_bench: SGB128 seed=%d S=%d mode=%s 반응 프리솔브 %s, worker %d, 제한 %.0fs\n", seed, S, mode, RT ? "켬" : "끔", nw, TL); flush(stdout)

if mode == "check"
    nbad = 0
    for x in all_x(nd)
        k = try nz_kkt_value_reduced(nd, x; optimizer=GRB, time_limit=120.0) catch e; nothing end
        (k === nothing || k[:status] != MOI.OPTIMAL) && (@printf("  x=%-9s KKT 미해결 → 건너뜀\n", xstr(x)); continue)
        r = nz_alpha_bnb_dr(nd, x; nworkers=nw, time_limit=120.0, rel_gap=1e-4, verbose=false, ipopt_time=30.0)
        tolv = 2e-4 * max(1.0, abs(k[:value]))
        ok = r[:LB] <= k[:value] + tolv && k[:value] <= r[:UB] + tolv
        ok || (global nbad += 1)
        @printf("  x=%-9s KKT %12.4f | α-B&B [%12.4f, %12.4f] %s 노드 %d (%.1fs) %s\n", xstr(x), k[:value], r[:LB], r[:UB], r[:is_exact] ? "Optimal" : "TimeLim",
                r[:nodes], r[:time], ok ? "" : "  ← 불일치"); flush(stdout)
    end
    @printf("RESULT check seed=%d S=%d 프리솔브=%s 불일치 %d 건 | 프리솔브 %d 회, 0 고정 합 %d, %.1fs\n", seed, S, RT ? "켬" : "끔", nbad, _RT_STAT[].calls, _RT_STAT[].fixed, _RT_STAT[].time)
else
    t0 = time()
    r = nz_belief_menu_benders(nd; oracle=:alpha_bnb, bnb=:dr, local_first=true, local_time=60.0, optimizer=GRB, tol=5e-3, oracle_gap=1e-3,
                               nworkers=nw, time_limit=TL, verbose=true, bnb_kw=(ipopt_time=120.0,), oracle_time=min(600.0, TL), boost_time=min(2400.0, TL))
    t = time() - t0
    gap = abs(r[:UB] - r[:LB]) / max(1.0, abs(r[:UB]))
    @printf("RESULT S=%d seed=%d method=ALGL 프리솔브=%s status=%s x=%s LB=%.6f UB=%.6f gap=%.2e iters=%d global=%d local=%s menu=%s time=%.1f | 프리솔브 %d 회, 0 고정 합 %d, %.1fs\n",
            S, seed, RT ? "켬" : "끔", r[:status], xstr(r[:x]), r[:LB], r[:UB], gap, r[:iters], r[:oracle_calls], string(get(r, :local_cuts, "-")),
            string(get(r, :menu_size, "-")), t, _RT_STAT[].calls, _RT_STAT[].fixed, _RT_STAT[].time)
end
