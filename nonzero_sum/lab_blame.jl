"""
lab_blame.jl — 정체한 상자의 노드 완화 해가 "어디서" 참 문제를 벗어나는지 (완화 오차의 출처) 진단.

저장된 전선 (lab_frontier.jl) 에서 노드 상한이 큰 상자 K 개를 골라, 수렴할 때까지 분리한 노드 LP 의 해 (α̂, d̂, r̂, …) 에서
  z      : 노드 LP 상한
  zE     : α̂ 고정 정확 LP 값 (belief 자유) — 참 G(α̂)
  zEd    : α̂ 와 follower belief d̂ 를 함께 고정한 정확 LP 값. 실행 불가능이면 "α̂ 가 d̂ 에서 follower 최적이 아님" → follower 쪽 (F) 완화 오차
           실행 가능이면 z − zEd 는 리더 쪽 (W, L) 완화 오차
  dist   : α̂ 를 follower 최적으로 만드는 belief 중 d̂ 에 가장 가까운 것까지의 L1 거리 (0 이면 d̂ 그대로 가능)
  곱 위반: Σ|ζL − α̂ r̂|, Σ|ζF − α̂ d̂|, Σ|ζW − α̂ ϖ̂|, Σ|ζΠ − α̂ π̃|
  시나리오별: 리더 블록의 ℓᵀŷ/r̂ 와 참 φ_s(α̂), follower 블록의 cᵀȳ/d̂ 와 참 Q_s(α̂) 의 차이
환경변수: LB_IN  LB_K = 8  LM_KW = "(fwd=true,)"  (_nz_dr_attach! 인자)  LM_TOL = 1e-7
실행: julia -t 2 nonzero_sum/lab_blame.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Serialization, Statistics
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d)
fr = deserialize(envf("LB_IN", joinpath(tempdir(), "nz_frontier_seed3_S20.jls")))
K = parse(Int, envf("LB_K", "8")); kw = eval(Meta.parse(envf("LM_KW", "(fwd=true,)"))); tolv = parse(Float64, envf("LM_TOL", "1e-7"))
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=fr.S, seed=fr.seed,
                                           eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
ndf = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
x̄ = zeros(ndf.nx); x̄[fr.x] .= 1
hUp, _ = nz_presolve_hU(ndf, x̄); nd = nz_with_hU(ndf, hUp)
nh, S, m, ny = nz_nh(nd), nd.S, nz_m(nd), nz_ny(nd)
LB = fr.LB; sc = max(1.0, abs(LB))
env = Gurobi.Env()
mk() = (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
P = _nz_build_sep(nd, x̄; optimizer=mk)
DR = _nz_dr_attach!(P, nd, x̄; optimizer=mk, kw...)
for a in ("FeasibilityTol", "OptimalityTol"); set_optimizer_attribute(P.O.model, a, tolv); end
set_optimizer_attribute(P.O.model, "NumericFocus", 0); P.O.model.ext[:lp_tol] = tolv
F = nz_build_fixed_alpha(nd, x̄; optimizer=mk)
lo0, hi0 = _nz_dr_global(nd, DR)
v = P.O.v; Hr = v[:Hr]
function solve_box!(l, u, lop, hip, phibp; rounds=30)
    _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
    rg = _nz_dr_ranges(DR, nd, l, u, lop, hip, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), phibp)
    rg === nothing && return -Inf, nothing
    lo, hi, _, phib = rg
    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib)
    z = _nz_sep_solve!(P; maxrounds=rounds)
    return z, (lo, hi)
end
solve_box!(zeros(nh), copy(nd.hU), lo0, hi0, fill(Inf, S))
# 수렴한 상한이 큰 순서로 K 개
cand = sort(fr.boxes; by=b -> -b.ub)[1:min(4K, end)]
zs = [solve_box!(b.l, b.u, b.lo, b.hi, b.phib)[1] for b in cand]
ordz = sortperm(zs; rev=true)[1:min(K, end)]
"시나리오 s 의 recourse 값 Q_s(α) 와 pessimistic 리더 평가 φ_s(α)"
function true_vals(α)
    Q = zeros(S); φ = zeros(S)
    for s in 1:S
        rr = nz_rhs(nd, x̄, α, s)
        mdl = Model(mk); @variable(mdl, y[1:ny] >= 0)
        @constraint(mdl, [k = 1:m], sum(nd.A[k, j] * y[j] for j in 1:ny if nd.A[k, j] != 0) <= rr[k])
        @objective(mdl, Max, dot(nd.c, y)); optimize!(mdl); Q[s] = objective_value(mdl)
        @constraint(mdl, dot(nd.c, y) >= Q[s] - 1e-7 * max(1, abs(Q[s])))
        @objective(mdl, Max, dot(nd.ell, y)); optimize!(mdl); φ[s] = objective_value(mdl)
    end
    return Q, φ
end
println("LB = ", LB, " | 변형 ", kw, " | 상자 ", length(ordz), " 개 (수렴 상한이 큰 순서)")
for n in ordz
    b = cand[n]
    z, rg = solve_box!(b.l, b.u, b.lo, b.hi, b.phib)
    lo, hi = rg
    α̂ = clamp.(value.(v[:α]), b.l, b.u); r̂ = value.(v[:r]); d̂ = value.(v[:d])
    vL = sum(abs(value(v[:ζL][i, s]) - α̂[i] * r̂[s]) for i in 1:nh, s in 1:S)
    vF = sum(abs(value(v[:ζF][i, s]) - α̂[i] * d̂[s]) for i in 1:nh, s in 1:S)
    vW = sum(abs(value(v[:ζW][jj, s]) - α̂[nd.hrow[k]] * value(v[:ϖ][k, s])) for (jj, k) in enumerate(Hr), s in 1:S)
    vP = isempty(DR.zP) ? NaN : sum(abs(value(DR.zP[a, s]) - α̂[nd.hrow[k]] * value(v[:π̃][k, s])) for (a, k) in enumerate(DR.Hk), s in 1:S)
    ℓŷ = [sum(nd.ell[j] * value(v[:ŷ][j, s]) for j in 1:ny) for s in 1:S]
    cŷ = [sum(nd.c[j] * value(v[:ŷ][j, s]) for j in 1:ny) for s in 1:S]
    cȳ = [sum(nd.c[j] * value(v[:ȳ][j, s]) for j in 1:ny) for s in 1:S]
    Q, φ = true_vals(α̂)
    zE = nz_eval_alpha!(F, nd, α̂)
    # d̂ 고정
    dv = F.v[:d]; lbs = lower_bound.(dv); ubs = upper_bound.(dv)
    for s in 1:S; set_lower_bound(dv[s], max(lbs[s], d̂[s] - 1e-9)); set_upper_bound(dv[s], min(ubs[s], d̂[s] + 1e-9)); end
    zEd = nz_eval_alpha!(F, nd, α̂)
    for s in 1:S; set_lower_bound(dv[s], lbs[s]); set_upper_bound(dv[s], ubs[s]); end
    # d̂ 에 가장 가까운 "α̂ 를 follower 최적으로 만드는" belief 까지의 L1 거리
    dist = NaN
    if zE > -Inf
        obj0 = objective_function(F.model)
        dev = @variable(F.model, [1:S], lower_bound = 0.0)
        c1 = @constraint(F.model, [s = 1:S], dev[s] >= dv[s] - d̂[s]); c2 = @constraint(F.model, [s = 1:S], dev[s] >= d̂[s] - dv[s])
        @objective(F.model, Min, sum(dev)); optimize!(F.model)
        dist = termination_status(F.model) == MOI.OPTIMAL ? objective_value(F.model) : NaN
        delete(F.model, c1); delete(F.model, c2); delete(F.model, dev)
        set_objective(F.model, MAX_SENSE, obj0)
    end
    act = [s for s in 1:S if r̂[s] > 1e-6]
    eL = sum(ℓŷ[s] - r̂[s] * φ[s] for s in 1:S)                 # 리더 블록이 참 φ(α̂) 보다 얼마나 더 받았나 (r̂ 가중 합)
    eQL = sum(cŷ[s] - r̂[s] * Q[s] for s in 1:S)                # 리더 블록 primal 의 follower 값이 참 Q(α̂) 와 얼마나 다른가
    eQF = sum(cȳ[s] - d̂[s] * Q[s] for s in 1:S)                # follower 블록 primal 값 − 참 값 (양수 = wait-and-see 이득)
    nun = count(hi .- lo .> 1e-4)
    @printf("상자: z−LB=%+.2f (gap %.2e) | G(α̂)−LB=%s | d̂ 고정: %s | d 거리 %.3f | 고정 안 된 성분 %d | 폭 %s\n",
            z - LB, (z - LB) / sc, isfinite(zE) ? @sprintf("%+.2f", zE - LB) : "실행 불가능",
            isfinite(zEd) ? @sprintf("%+.2f", zEd - LB) : "실행 불가능", dist, nun, string(round.((b.u .- b.l)[nd.hU .> 0]; digits=1)))
    @printf("      곱 위반 ζL %.3f ζF %.3f ζW %.1f ζΠ %.1f | 리더 블록 Σ(ℓŷ − r̂φ(α̂)) = %+.2f, Σ(cŷ − r̂Q(α̂)) = %+.2f | follower 블록 Σ(cȳ − d̂Q(α̂)) = %+.2f\n",
            vL, vF, vW, vP, eL, eQL, eQF)
    flush(stdout)
end
