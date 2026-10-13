"""
lab_inspect.jl — 상자 하나의 노드 LP 해를 시나리오별로 풀어 본다 (완화가 무엇을 "속이는지" 눈으로 확인).

시나리오마다: belief (d̂, r̂), 좌표별 시나리오 α (follower 블록 ζF/d̂, 리더 블록 ζL/r̂) 와 공통 α̂ 의 차이, recourse 쌍대해의 비율 (π̃/d̂, ϖ/r̂) 과 쌍대 범위,
follower 블록 primal 값 cȳ/d̂ − Q_s(α̂), 리더 블록 ℓŷ/r̂ − φ_s(α̂).
환경변수: LB_IN  LI_BOX = 51 (상한 내림차순 번호)  LM_KW = "(fwd=true,)"  LM_TOL = 1e-7  LI_COORDS = "3,4,6,7,8"
실행: julia -t 1 nonzero_sum/lab_inspect.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Serialization, Statistics
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d)
fr = deserialize(envf("LB_IN", joinpath(tempdir(), "nz_frontier_seed3_S20.jls")))
kw = eval(Meta.parse(envf("LM_KW", "(fwd=true,)"))); tolv = parse(Float64, envf("LM_TOL", "1e-7"))
coords = parse.(Int, split(envf("LI_COORDS", "3,4,6,7,8"), ","))
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=fr.S, seed=fr.seed,
                                           eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
ndf = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
x̄ = zeros(ndf.nx); x̄[fr.x] .= 1
hUp, _ = nz_presolve_hU(ndf, x̄); nd = nz_with_hU(ndf, hUp)
nh, S, m, ny = nz_nh(nd), nd.S, nz_m(nd), nz_ny(nd)
LB = fr.LB
env = Gurobi.Env()
mk() = (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
P = _nz_build_sep(nd, x̄; optimizer=mk); DR = _nz_dr_attach!(P, nd, x̄; optimizer=mk, kw...)
for a in ("FeasibilityTol", "OptimalityTol"); set_optimizer_attribute(P.O.model, a, tolv); end
set_optimizer_attribute(P.O.model, "NumericFocus", 0); P.O.model.ext[:lp_tol] = tolv
v = P.O.v; Hr = v[:Hr]
function solve!(l, u, lop, hip, phibp; rounds=parse(Int, envf("LI_ROUNDS", "30")))
    _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
    lo, hi, _, phib = _nz_dr_ranges(DR, nd, l, u, lop, hip, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), phibp)
    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib)
    return _nz_sep_solve!(P; maxrounds=rounds, stall_rounds=parse(Int, envf("LI_STALL", "2"))), lo, hi
end
lo0, hi0 = _nz_dr_global(nd, DR)
solve!(zeros(nh), copy(nd.hU), lo0, hi0, fill(Inf, S))
b = sort(fr.boxes; by=b -> -b.ub)[parse(Int, envf("LI_BOX", "51"))]
z, lo, hi = solve!(b.l, b.u, b.lo, b.hi, b.phib)
α̂ = value.(v[:α]); r̂ = value.(v[:r]); d̂ = value.(v[:d]); â = value.(v[:a])
ξ = nd.meta[:xi]
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
Q, φ = true_vals(α̂)
aof = Dict(nd.hrow[k] => a for (a, k) in enumerate(DR.Hk)); kof = Dict(nd.hrow[k] => k for k in DR.Hk)
@printf("상자 z−LB = %+.2f | κ = %s\n", z - LB, string(round.(value.(v[:κ]); digits=2)))
println("좌표 ", coords, ": l = ", round.(b.l[coords]; digits=1), " u = ", round.(b.u[coords]; digits=1), " α̂ = ", round.(α̂[coords]; digits=2))
println("수요 ξ (고객 1,2,3) 와 시나리오별 값. 좌표마다 [αF−α̂ | αL−α̂ | π̃/d (범위) | ϖ/r]")
ℓŷ = [sum(nd.ell[j] * value(v[:ŷ][j, s]) for j in 1:ny) for s in 1:S]
cȳ = [sum(nd.c[j] * value(v[:ȳ][j, s]) for j in 1:ny) for s in 1:S]
for s in sortperm(d̂; rev=true)
    @printf("s=%2d ξ=(%5.1f,%5.1f,%5.1f) d=%.3f a=%.3f r=%.3f | F: cȳ/d−Q=%+8.1f | L: ℓŷ/r−φ=%+7.2f (φ=%7.1f) |", s, ξ[1, s], ξ[2, s], ξ[3, s], d̂[s], â[s], r̂[s],
            d̂[s] > 1e-7 ? cȳ[s] / d̂[s] - Q[s] : NaN, r̂[s] > 1e-7 ? ℓŷ[s] / r̂[s] - φ[s] : NaN, φ[s])
    for i in coords
        a = aof[i]; k = kof[i]
        aF = d̂[s] > 1e-7 ? value(v[:ζF][i, s]) / d̂[s] - α̂[i] : NaN; aL = r̂[s] > 1e-7 ? value(v[:ζL][i, s]) / r̂[s] - α̂[i] : NaN
        sF = d̂[s] > 1e-7 ? value(v[:π̃][k, s]) / d̂[s] : NaN; sL = r̂[s] > 1e-7 ? value(v[:ϖ][k, s]) / r̂[s] : NaN
        fx = hi[a, s] - lo[a, s] <= 1e-4
        @printf(" [%+5.1f|%+5.1f|%3.0f %s|%3.0f]", aF, aL, sF, fx ? @sprintf("=%3.0f   ", lo[a, s]) : @sprintf("%3.0f~%3.0f", lo[a, s], hi[a, s]), sL)
    end
    println()
end
@printf("합: Σ d(cȳ/d − Q) = %+.2f, Σ r(ℓŷ/r − φ) = %+.2f, Σ r φ(α̂) = %.2f (LB %.2f)\n", sum(cȳ[s] - d̂[s] * Q[s] for s in 1:S),
        sum(ℓŷ[s] - r̂[s] * φ[s] for s in 1:S), dot(r̂, φ), LB)
