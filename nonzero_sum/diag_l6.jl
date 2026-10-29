"""
diag_l6.jl — L6 (Goyal 9: d=25, A 5 / B 5, |X|=32) 의 고정 x̄ oracle 진단 (2026-10-27 밤, 사용자 요청 "L6 이 왜 안 되는지 다각도로"). 기존 파일 미수정.

인스턴스는 pilot_pooled_store.jl 의 mk 와 같다 (반응 정의역 프리솔브도 같은 방식으로 끼움).
DL_SOLVERS (쉼표): abbdr (기본 구성 α-B&B, verbose) | grb (Ω 를 Gurobi NonConvex 로) | info (인스턴스 정보만)
환경변수: DL_SEED = 1  DL_S = 3  DL_X = "2,3"  DL_TL = 600  DL_GAP = 1e-3  DL_WORKERS = 12  DL_PRE = 1  DL_LAMBDA = 2000  DL_THETA = (circuit 정확값)
          DL_ABBKW = "" (nz_alpha_bnb_dr 에 넘길 Bool/숫자 인자, 예 "dd=false,dual_range=false")
실행: julia -t 14,1 nonzero_sum/diag_l6.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
include(joinpath(root, "nonzero_sum", "nz_root_tighten.jl"))
envf(k, d) = get(ENV, k, d); pf(k, d) = parse(Float64, envf(k, d)); pint(k, d) = parse(Int, envf(k, d))
seed = pint("DL_SEED", "1"); S = pint("DL_S", "3"); xi = parse.(Int, split(envf("DL_X", "2,3"), ","; keepempty=false))
TL = pf("DL_TL", "600"); GAP = pf("DL_GAP", "1e-3"); nw = pint("DL_WORKERS", "12"); PRE = envf("DL_PRE", "1") == "1"
lam = pf("DL_LAMBDA", "2000")
solvers = split(envf("DL_SOLVERS", "info,abbdr"), ",")
nloc, nA, nB, NB = 25, 5, 5, 5; m = nA + nB; capv = 150.0 * (nloc - m) / nA
nd0 = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=nA, nB=nB, NB=NB, S=S, seed=seed,
    bA=capv, bB=capv, qB=750.0, dlo=50.0, dhi=150.0, p=0.3, f=0.1, reservation=:pooled, pool_limit=:store, quota=300.0,
    eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
θc = nz_theta_circuit_exact(nd0)[1]
θ = haskey(ENV, "DL_THETA") ? pf("DL_THETA", "0") : θc
nd = nz_with_bounds(nd0; thetaU=θ, lambdaU=lam)
const _RT_CACHE = Dict{Any,Tuple{Vector{Float64},Int}}()
if PRE
    function nz_presolve_hU(nd::NZData, x̄::Vector{Float64})
        key = (nd.name, nd.S, copy(nd.hU), copy(x̄))
        haskey(_RT_CACHE, key) && return _RT_CACHE[key]
        hU1, n1 = invoke(nz_presolve_hU, Tuple{NZData,Any}, nd, x̄)
        hU2, n2, t = nz_response_presolve(nz_with_hU(nd, hU1), x̄; env=_nz_envs(1)[1])
        @printf("    [반응 프리솔브] x=%s: 활성 %d 중 %d 개를 0 으로 고정 (%.1fs)\n", string(findall(x̄ .> 0.5)), count(hU1 .> 0), n2, t); flush(stdout)
        return (_RT_CACHE[key] = (hU2, n1 + n2))
    end
end
x̄ = zeros(nd.nx); x̄[xi] .= 1
@printf("diag_l6: seed=%d S=%d x̄=%s n_h=%d θ=%.3f (circuit %.3f) λ=%.0f TL=%.0f gap=%.0e workers=%d 프리솔브 %s\n",
        seed, S, string(xi), nz_nh(nd), θ, θc, lam, TL, GAP, nw, PRE ? "켬" : "끔"); flush(stdout)

if "info" in solvers
    @printf("  hU = %s\n  q_hat = %s\n", string(round.(nd.hU; digits=1)), string(round.(nd.q_hat; digits=3)))
    flush(stdout)
end
function parse_kw(s)
    kv = Pair{Symbol,Any}[]
    for t in split(s, ","; keepempty=false)
        a, b = split(t, "=")
        v = b == "true" ? true : b == "false" ? false : startswith(b, ":") ? Symbol(b[2:end]) : occursin(r"^-?[0-9]+$", b) ? parse(Int, b) : parse(Float64, b)
        push!(kv, Symbol(strip(a)) => v)
    end
    return (; kv...)
end
if "abbdr" in solvers
    kw = parse_kw(envf("DL_ABBKW", ""))
    t = @elapsed r = nz_alpha_bnb_dr(nd, x̄; nworkers=nw, time_limit=TL, rel_gap=GAP, verbose=true, log_every=60.0, ipopt_time=120.0, kw...)
    @printf("RESULT abbdr kw=%s status=%s LB=%.4f UB=%.4f gap=%.2e nodes=%s time=%.1f\n", string(kw), string(get(r, :status, "-")), r[:LB], r[:UB],
            abs(r[:UB] - r[:LB]) / max(1.0, abs(r[:UB])), string(get(r, :nodes, "-")), t)
    for k in sort(collect(keys(r)); by=string)
        k in (:α, :state) && continue
        v = r[k]; (v isa Number || v isa Symbol || v isa String) && @printf("    %s = %s\n", string(k), string(v))
    end
    flush(stdout)
end
if "grbr" in solvers
    # Gurobi Ω + 루트 상자 [0, hU] 의 꼭짓점 LP 쌍대 범위 (nz_alpha_bnb_dr 의 range_lp 와 같은 전제: recourse 값함수 submodular):
    #   σlo r_s ≤ ϖ_ks ≤ σhi r_s,  σlo d_s ≤ π̃_ks ≤ σhi d_s  (H 행, 노드 α-B&B 의 W·F 행과 같음)
    ndp = presolve_for_grbr = nz_with_hU(nd, nz_presolve_hU(nd, x̄)[1])
    mk1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
    CL = [_nz_dr_build_corner(ndp, x̄, s; optimizer=mk1) for s in 1:ndp.S]
    Hk = [k for k in findall(ndp.hrow .> 0) if ndp.hU[ndp.hrow[k]] > 0 && isfinite(ndp.varpiU[k])]
    function cext(s, αc, k, ismax)
        C = CL[s]; rr = nz_rhs(ndp, x̄, αc, s)
        for kk in eachindex(rr); ndp.hrow[kk] > 0 && set_normalized_coefficient(C.opt, C.σ[kk], rr[kk]); end
        set_normalized_rhs(C.opt, 1e20); @objective(C.model, Min, sum(rr[kk] * C.σ[kk] for kk in eachindex(rr) if rr[kk] != 0)); optimize!(C.model)
        z = objective_value(C.model); set_normalized_rhs(C.opt, z + 1e-7 * max(1.0, abs(z)))
        @objective(C.model, ismax ? MAX_SENSE : MIN_SENSE, C.σ[k]); optimize!(C.model)
        termination_status(C.model) == MOI.OPTIMAL || error("cext $(termination_status(C.model))")
        return objective_value(C.model)
    end
    O = build_nz_omega(ndp; optimizer=GRB)
    set_optimizer_attribute(O.model, "Threads", nw)
    nfx = 0; wsum = 0.0
    for k in Hk, s in 1:ndp.S
        lo_ = max(cext(s, ndp.hU, k, false) - 1e-6, 0.0); hi_ = min(cext(s, zeros(nz_nh(ndp)), k, true) + 1e-6, ndp.varpiU[k])
        global wsum += hi_ - lo_; hi_ - lo_ <= 1e-4 && (global nfx += 1)
        @constraint(O.model, O.v[:ϖ][k, s] <= hi_ * O.v[:r][s]); @constraint(O.model, O.v[:ϖ][k, s] >= lo_ * O.v[:r][s])
        @constraint(O.model, O.v[:π̃][k, s] <= hi_ * O.v[:d][s]); @constraint(O.model, O.v[:π̃][k, s] >= lo_ * O.v[:d][s])
    end
    @printf("  grbr: 루트 꼭짓점 범위 %d 개 (고정 %d, 평균 폭 %.3f)\n", length(Hk) * ndp.S, nfx, wsum / max(1, length(Hk) * ndp.S)); flush(stdout)
    t = @elapsed res = nz_solve!(O, ndp, x̄; time_limit=TL, gap=GAP)
    @printf("RESULT grbr status=%s LB=%.4f UB=%.4f gap=%.2e time=%.1f\n", string(termination_status(O.model)), res[:Fval], res[:bound],
            abs(res[:bound] - res[:Fval]) / max(1.0, abs(res[:bound])), t)
    flush(stdout)
end
if "grb" in solvers
    O = build_nz_omega(nd; optimizer=GRB)
    set_optimizer_attribute(O.model, "Threads", nw)
    t = @elapsed res = nz_solve!(O, nd, x̄; time_limit=TL, gap=GAP)
    @printf("RESULT grb status=%s LB=%.4f UB=%.4f gap=%.2e time=%.1f\n", string(termination_status(O.model)), res[:Fval], res[:bound],
            abs(res[:bound] - res[:Fval]) / max(1.0, abs(res[:bound])), t)
    flush(stdout)
end
