"""
run_loc_base_scaling.jl — 기본 인스턴스 (SGB128 Goyal 기본 사례 + 예약, quota 150) 에서 시나리오 수 S 를 늘리며 방법 비교.

방법 (원고 6.1 Algorithms)
  E    : 모든 x 를 KKT 독립 평가 (big-M·재정식화 없음). x 하나라도 시간 제한이면 미해결.
  SBG  : 표준 Benders, local (OptimalityTarget=1) cut 먼저, 위반 cut 이 없을 때만 Gurobi 전역 Ω (남은 시간 전체)
  SBA  : 같은 구조, 전역은 α-B&B (남은 시간 전체). MW 는 α-B&B 의 restricted cut 에만 (SB-G 전역 cut, local cut 은 MW 없음)
  ALG  : oracle 600 s, 재방문 시 2,400 s, 모두 남은 시간으로 자름
  ALGG : Algorithm 2 와 같고 전역 oracle 만 Gurobi Ω (목표값 조기 종료 없음)
  ALG  : Algorithm 2 (belief-menu + α-B&B, MW)
작은 S 부터. E 는 미해결이면 더 큰 S 를 건너뜀. Benders 계열은 LB_SKIP_BENDERS=1 일 때만 같은 규칙 (기본 0: 모든 S 를
시간 제한까지 돌려 최종 gap 과 t_LB (LB 가 최종 LB 에 처음 도달한 경과 시간) 를 기록. 기본 인스턴스에서는 모든 방법이 최적해는
금방 찾고 x 인증 (상한) 에서 막히므로, 건너뛰면 큰 S 의 정보가 사라짐).

환경변수
  LB_S = "3,20,50,200"   LB_METHODS = "E,SBG,SBA,ALG"   LB_EPS = 0.3   LB_BETA = 0.4   LB_DELTA = 0.1   LB_QUOTA = 150
  LB_TL = 3600 (방법당 전체)   LB_E_TL = 300 (E 의 x 하나당)   LB_E_FORM = reduced | full
  LB_TOL = 5e-3 (Benders 상대 허용오차, zero-sum 과 같음)   LB_OGAP = 1e-3 (전역 oracle 목표 gap)
  LB_THETA = circuit | res | 숫자 (θᵁ)   LB_ORACLE_TIME = 600   LB_BOOST = 2400   LB_LOCAL_TIME = 60   LB_WORKERS = 12   NZ_SEED = 1
실행: julia -t 14,1 nonzero_sum/run_loc_base_scaling.jl
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

Ss = sort(parse.(Int, strs("LB_S", "3,20,50,200")))
methods = strs("LB_METHODS", "E,SBG,SBA,ALG")
ε = parse(Float64, envf("LB_EPS", "0.3")); β = parse(Float64, envf("LB_BETA", "0.4")); δ = pd(envf("LB_DELTA", "0.1"))
quota = parse(Float64, envf("LB_QUOTA", "150")); seed = parse(Int, envf("NZ_SEED", "1"))
TL = parse(Float64, envf("LB_TL", "3600")); E_TL = parse(Float64, envf("LB_E_TL", "300"))
kkt = envf("LB_E_FORM", "reduced") == "full" ? nz_kkt_value : nz_kkt_value_reduced
nw = parse(Int, envf("LB_WORKERS", "12"))
Threads.nthreads() >= nw + 2 || error("α-B&B: Julia 스레드 $(Threads.nthreads()) < LB_WORKERS+2 = $(nw + 2) (julia -t $(nw + 2),1)")
# 허용오차: zero-sum 실험과 같은 0.5% (2026-10-10 결정). α-B&B 목표 gap 은 그보다 작게 (LB_OGAP).
tol_b = parse(Float64, envf("LB_TOL", "5e-3")); ogap = parse(Float64, envf("LB_OGAP", "1e-3"))
common = (optimizer=GRB, tol=tol_b, oracle_gap=ogap, nworkers=nw, time_limit=TL, verbose=true,
          oracle_time=min(parse(Float64, envf("LB_ORACLE_TIME", "600")), TL),
          boost_time=min(parse(Float64, envf("LB_BOOST", "2400")), TL))
local_time = parse(Float64, envf("LB_LOCAL_TIME", "60"))

# LB_THETA = circuit (기본: nz_theta_circuit_exact 의 정확한 circuit 값, 증명 가능) | res (기존 v / 거리 해상도) | 숫자
function mk(S)
    nd = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=quota, wres=300.0, S=S, seed=seed,
                                              eps_hat=ε, eps_tilde=ε, beta=β), δ)
    th = envf("LB_THETA", "circuit")
    th == "res" && return nd
    return nz_with_bounds(nd; thetaU=(th == "circuit" ? nz_theta_circuit_exact(nd)[1] : parse(Float64, th)))
end
all_x(nd) = [Float64.(collect(bits)) for bits in Iterators.product(fill(0:1, nd.nx)...)
             if sum(bits) <= nd.gamma && all(nd.x_allowed[i] || bits[i] == 0 for i in 1:nd.nx)]
xstr(x) = "{" * join(findall(x .> 0.5), ",") * "}"

println("="^100)
@printf("기본 인스턴스 S 스케일링: SGB128 pair quota=%g ε̂=ε̃=%.2f β=%.2f δ=%s seed=%d | TL=%.0fs E_TL/x=%.0fs E=%s workers=%d θᵁ=%.4f (%s) tol=%.0e ogap=%.0e\n",
        quota, ε, β, string(δ), seed, TL, E_TL, string(kkt), nw, mk(3).thetaU, envf("LB_THETA", "circuit"), tol_b, ogap)
println("="^100); flush(stdout)

given_up = Set{String}()
for S in Ss
    nd = mk(S)
    @printf("\n--- S=%d (n_h=%d, |X|=%d)\n", S, nz_nh(nd), length(all_x(nd))); flush(stdout)
    for meth in methods
        if meth in given_up
            @printf("  %s: 더 작은 S 에서 미해결 → 건너뜀\n", meth); continue
        end
        t0 = time()
        if meth == "E"
            best = (Inf, Float64[]); solved = true
            for x in all_x(nd)
                r = try kkt(nd, x; optimizer=GRB, time_limit=E_TL) catch e; nothing end
                if r === nothing || r[:status] != MOI.OPTIMAL
                    solved = false
                    @printf("  E: x=%s 에서 %s (%.0fs)\n", xstr(x), r === nothing ? "시간 제한 (incumbent 없음)" :
                            string(r[:status]), time() - t0)
                    break
                end
                v = dot(nd.qx, x) + r[:value]; v < best[1] && (best = (v, x))
                time() - t0 > TL && (solved = false; println("  E: 전체 TL 초과"); break)
            end
            solved || push!(given_up, meth)
            @printf("RESULT S=%d method=E status=%s x=%s value=%.6f time=%.1f\n", S, solved ? "Optimal" : "Unsolved",
                    solved ? xstr(best[2]) : "-", solved ? best[1] : NaN, time() - t0)
        else
            r = meth == "ALG"  ? nz_belief_menu_benders(nd; oracle=:alpha_bnb, common...) :
                meth == "ALGG" ? nz_belief_menu_benders(nd; oracle=:gurobi, common...) :
                nz_standard_benders(nd; oracle=(meth == "SBG" ? :gurobi : :alpha_bnb), local_first=true,
                                    local_time=local_time, oracle_remaining=true, common...)
            t = time() - t0
            envf("LB_SKIP_BENDERS", "0") == "1" && r[:status] != :Optimal && push!(given_up, meth)
            gap = abs(r[:UB] - r[:LB]) / max(1.0, abs(r[:UB]))
            iLB = findfirst(h -> h.LB >= r[:LB] - 1e-4 * max(1.0, abs(r[:LB])), r[:hist])
            tLB = iLB === nothing ? t : r[:hist][iLB].w        # 마지막 반복에서 처음 도달 (기록 전 종료) 이면 종료 시각
            @printf("RESULT S=%d method=%s status=%s x=%s LB=%.6f UB=%.6f gap=%.2e iters=%d global=%d local=%s menu=%s t_LB=%.1f time=%.1f\n",
                    S, meth, r[:status], xstr(r[:x]), r[:LB], r[:UB], gap, r[:iters], r[:oracle_calls],
                    string(get(r, :local_cuts, "-")), string(get(r, :menu_size, "-")), tLB, t)
        end
        flush(stdout)
    end
end
