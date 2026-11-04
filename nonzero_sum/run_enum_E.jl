"""
run_enum_E.jl — 벤치마크 E (열거, 2026-11-03): 𝒳 의 모든 x 에서 V*(x) 를 α-B&B 로 풀고 min_x qᵀx + V*(x) 를 고른다 (Benders 없음).

인스턴스·α-B&B 설정은 pilot_pooled_store.jl (본 실험 = Ours) 와 같다: 같은 PP_* 환경변수, 같은 반응 정의역 프리솔브 (PP_PRE),
같은 α-B&B 인자 (ipopt_time 120 + PP_BNBKW, 목표 gap 1e-3, worker PP_WORKERS), 허용오차 0.5%, 인스턴스당 시간 한도 PP_TL (전체).
가지치기 (E_PRUNE = 1, 기본): x 를 차례로 풀며 목표값 = (지금까지 최선 상한) − qᵀx 를 넘긴다. α-B&B 가
  incumbent ≥ 목표값 → x 는 최선보다 좋을 수 없음 → 그 x 는 멈춤 (하한만 기록)
  상한 ≤ 목표값 → x 가 최선을 바꿈 → 같은 탐색 상태에서 목표값 없이 이어 풀어 (resume) 값을 확정
E_PRUNE = 0 이면 모든 x 를 목표값 없이 끝까지 (독립 열거).
결과: 상한 = min_x qᵀx + UB(x), 하한 = min_x qᵀx + LB(x) (LB = incumbent 값, 시간 한도로 못 푼 x 는 그때의 LB), 간격 ≤ 0.5% 면 Optimal.
환경변수: pilot_pooled_store.jl 과 같음 (PP_NLOC, PP_NA, PP_NB, PP_NBMAX, PP_S, PP_SEEDS, PP_QB, PP_TL, PP_PRE, PP_BNBKW, ...) + E_PRUNE
실행: julia -t 14,1 nonzero_sum/run_enum_E.jl
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
quota = pf("PP_QUOTA", "300"); lam = pf("PP_LAMBDA", "2000"); fres = pf("PP_F", "0.1"); prem = pf("PP_P", "0.3")
TL = pf("PP_TL", "1800"); nw = pint("PP_WORKERS", "12"); PRE = envf("PP_PRE", "1") == "1"; ε = pf("PP_EPS", "0.3"); δ = pf("PP_DELTA", "0.1")
βL = pf("PP_BETA", "0.4"); PRUNE = envf("E_PRUNE", "1") == "1"
Threads.nthreads() >= nw + 2 || error("julia -t $(nw + 2),1 이상 필요")
m = nA + nB; capv = 150.0 * (nloc - m) / nA
qB = envf("PP_QB", "750") == "cap" ? capv : pf("PP_QB", "750")
# pilot_pooled_store.jl 의 mk 와 같음
function mk(S, seed)
    nd = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=nA, nB=nB, NB=NB, S=S, seed=seed,
        bA=capv, bB=capv, qB=qB, dlo=50.0, dhi=150.0, p=prem, f=fres, reservation=:pooled, pool_limit=:store, quota=quota,
        eps_hat=ε, eps_tilde=ε, beta=βL), δ)
    return nz_with_bounds(nd; thetaU=nz_theta_circuit_exact(nd)[1], lambdaU=lam)
end
const _RT_CACHE = Dict{Any,Tuple{Vector{Float64},Int}}()
if PRE          # pilot_pooled_store.jl 과 같은 반응 정의역 프리솔브 (x̄ 마다 한 번)
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
function _pp_kw(str)        # pilot_pooled_store.jl 과 같음
    kv = Pair{Symbol,Any}[]
    for t in split(str, ","; keepempty=false)
        a, b = strip.(split(t, "="))
        push!(kv, Symbol(a) => (b == "true" ? true : b == "false" ? false : occursin(r"^[0-9]+$", b) ? parse(Int, b) : parse(Float64, b)))
    end
    return (; kv...)
end
kw = merge((ipopt_time=120.0,), _pp_kw(envf("PP_BNBKW", "")))
all_x(nd) = [Float64.(collect(bits)) for bits in Iterators.product(fill(0:1, nd.nx)...)
             if sum(bits) <= nd.gamma && all(nd.x_allowed[i] || bits[i] == 0 for i in 1:nd.nx)]
tol = 5e-3; gap = 1e-3
relg(lb, ub) = (ub - lb) / max(1.0, abs(ub))
agap(r) = (r[:UB] - r[:LB]) / max(1.0, abs(r[:LB]))      # α-B&B 의 종료 기준과 같은 gap (|LB| 기준)

for seed in seeds, S in sort(Ss)
    nd = mk(S, seed); X = all_x(nd)
    @printf("\n--- E (열거%s) pooled(거점별 %.0f) d=%d A=%d B=%d (N_B≤%d) seed=%d S=%d: n_h=%d, |X|=%d, θᵁ=%.1f λᵁ=%.1f, TL=%.0fs, 프리솔브 %s\n",
            PRUNE ? ", 가지치기" : ", 독립", quota, nloc, nA, nB, NB, seed, S, nz_nh(nd), length(X), nd.thetaU, nd.lambdaU, TL, PRE ? "켬" : "끔")
    flush(stdout)
    t0 = time(); best_ub = Inf; best_x = X[1]; lbs = fill(-Inf, length(X)); calls = 0; npruned = 0; nunfin = 0
    for (i, x) in enumerate(X)
        rem = TL - (time() - t0)
        if rem <= 1.0
            nunfin += 1; @printf("  x=%-8s 시간 없음 (하한 −∞)\n", xstr(x)); continue
        end
        cx = dot(nd.qx, x); tgt = (PRUNE && isfinite(best_ub)) ? best_ub - cx - 0.99 * tol * max(1.0, abs(best_ub)) : nothing
        ti = time()
        r = nz_alpha_bnb_dr(nd, x; nworkers=nw, time_limit=rem, rel_gap=gap, verbose=false, target=tgt, kw...); calls += 1
        how = "끝"
        if tgt !== nothing && r[:LB] >= tgt && agap(r) > gap
            how = "가지침 (incumbent ≥ 목표)"; npruned += 1
        else
            # Benders 의 oracle 과 같은 gap 조임 (_oracle_gap_eff): 전체 목적값 |qᵀx + V| 가 |V| 보다 작으면 V 기준 gap 을 줄여
            # 전체 기준 0.5% 를 맞춘다. 목표값으로 멈췄거나 (상한 ≤ 목표) gap 이 모자라면 같은 탐색 상태에서 목표값 없이 이어 푼다.
            # ge 는 새 UB 로 다시 계산 (2026-11-04 수정: 이어 풀기 전 UB 로 고정해 두어 시간이 남았는데도 "시간 한도" 로 표시되던 것).
            ge = _oracle_gap_eff(gap, tol, cx + r[:UB], r[:UB])
            nres = 0
            while agap(r) > ge && TL - (time() - t0) > 1.0 && nres < 5
                r = nz_alpha_bnb_dr(nd, x; nworkers=nw, time_limit=TL - (time() - t0), rel_gap=ge, verbose=false, resume=r[:state], kw...)
                calls += 1; nres += 1; how = tgt !== nothing ? "최선 갱신 → 이어 풂" : "gap 조여 이어 풂"
                ge = _oracle_gap_eff(gap, tol, cx + r[:UB], r[:UB])
            end
            if agap(r) > ge
                nunfin += 1
                how *= TL - (time() - t0) <= 1.0 ? " (시간 한도)" : @sprintf(" (gap %.1e > %.1e)", agap(r), ge)
            end
        end
        lbs[i] = cx + r[:LB]
        if cx + r[:UB] < best_ub && !(how == "가지침 (incumbent ≥ 목표)")
            best_ub = cx + r[:UB]; best_x = x
        end
        @printf("  x=%-8s qx=%8.1f LB=%12.4f UB=%12.4f  %-26s %.1fs (전체 %.0fs)\n", xstr(x), cx, cx + r[:LB], cx + r[:UB], how,
                time() - ti, time() - t0)
        flush(stdout)
    end
    LBt = minimum(lbs); g = relg(LBt, best_ub)
    st = g <= tol ? "Optimal" : "TimeLimit"
    @printf("RESULT E pooled-store λ=%.0f qB=%.0f d=%d m=%d nA=%d nB=%d seed=%d S=%d nh=%d status=%s x=%s LB=%.6f UB=%.6f gap=%.2e calls=%d pruned=%d unfinished=%d nx=%d time=%.1f%s\n",
            lam, qB, nloc, m, nA, nB, seed, S, nz_nh(nd), st, xstr(best_x), LBt, best_ub, g, calls, npruned, nunfin, length(X), time() - t0,
            (PRUNE ? "" : " prune=0") * (PRE ? "" : " pre=0"))
    flush(stdout)
end
