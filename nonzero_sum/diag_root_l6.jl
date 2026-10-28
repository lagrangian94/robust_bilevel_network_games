"""
diag_root_l6.jl — L6 고정 x̄ 에서 α-B&B 루트 완화가 왜 약한지 분해 (2026-10-28, 사용자 요청 "루트 완화 강화"). 기존 파일 미수정.

루트 노드 LP (nz_alpha_bnb_dr 와 같은 구성: 전역 쌍대 범위, FWD·KMC·상보성, dd=false) 를 풀고
  (1) 목적값 항목 분해: 리더 수익 ℓᵀŷ, θ 항 θ(cᵀŷ − 쌍대목적), π̂ᵁ 항, follower 벌점 (λ 항)
  (2) 루트 해의 α̂ 를 고정한 정확한 값 (nz_eval_alpha!)
  (3) 곱 한 종류씩 루트 해 값으로 고정해 다시 푼 값 (그 종류의 McCormick 이 상한을 얼마나 부풀리는지)
      d = d̂ 고정 (ζF = α d̂ 정확), r = r̂ 고정 (ζL = α r̂, ζW 의 r 쪽), 둘 다
  (4) 최적해 α* (α-B&B 60 s) 의 같은 분해
환경변수: DL_SEED = 7  DL_S = 3  DL_X = "4,5"  DL_LAMBDA = 2000  DL_RANGE = 0 (1 이면 루트 범위 MIP 를 풀어 범위 반영, 마감 120 s)
실행: julia -t 14,1 nonzero_sum/diag_root_l6.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d)
seed = parse(Int, envf("DL_SEED", "7")); S = parse(Int, envf("DL_S", "3")); xi = parse.(Int, split(envf("DL_X", "4,5"), ","))
lam = parse(Float64, envf("DL_LAMBDA", "2000")); RANGE = envf("DL_RANGE", "0") == "1"
nloc, nA, nB, NB = 25, 5, 5, 5; capv = 150.0 * (nloc - nA - nB) / nA
nd0 = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=nA, nB=nB, NB=NB, S=S, seed=seed,
    bA=capv, bB=capv, qB=750.0, dlo=50.0, dhi=150.0, p=0.3, f=0.1, reservation=:pooled, pool_limit=:store, quota=300.0,
    eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
nd1 = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1], lambdaU=lam)
x̄ = zeros(nd1.nx); x̄[xi] .= 1
hUp, _ = nz_presolve_hU(nd1, x̄); nd = nz_with_hU(nd1, hUp)
nh = nz_nh(nd); Hr = findall(nd.hrow .> 0)
mk1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
@printf("diag_root_l6: seed=%d S=%d x̄=%s n_h=%d (활성 %d) θ=%.1f λ=%.0f\n", seed, S, string(xi), nh, count(nd.hU .> 0), nd.thetaU, nd.lambdaU)
@printf("  hU = %s\n", string(round.(nd.hU; digits=1)))

# ---- 목적값 분해 (Ω 변수 v, 값 함수 val) ----
function decompose(v, val)
    θ, λU = nd.thetaU, nd.lambdaU; m, ny = nz_m(nd), nz_ny(nd); Xr = v[:Xr]
    ŷ = val.(v[:ϖ] isa Any ? v[:ŷ] : v[:ŷ])
    L = sum(nd.ell[j] * val(v[:ŷ][j, s]) for j in 1:ny, s in 1:S if nd.ell[j] != 0)
    cy = sum(nd.c[j] * val(v[:ŷ][j, s]) for j in 1:ny, s in 1:S if nd.c[j] != 0)
    du = sum(nd.u[k, s] * val(v[:ϖ][k, s]) for k in 1:m, s in 1:S if nd.u[k, s] != 0) +
         sum(nd.hcoef[k] * val(v[:ζW][jj, s]) for (jj, k) in enumerate(Hr), s in 1:S) +
         sum(nd.g[k, s] * x̄[nd.xrow[k]] * val(v[:ϖ][k, s]) for k in Xr, s in 1:S)
    piL = -sum(nd.piLU[k] * (x̄[nd.xrow[k]] * val(v[:ρ̂1][jj, s]) + (1 - x̄[nd.xrow[k]]) * val(v[:ρ̂3][jj, s])) for (jj, k) in enumerate(Xr), s in 1:S)
    pen = -λU * (sum(nd.piFU[k] * (x̄[nd.xrow[k]] * val(v[:ρ̃1][jj, s]) + (1 - x̄[nd.xrow[k]]) * val(v[:ρ̃3][jj, s])) for (jj, k) in enumerate(Xr), s in 1:S) +
                 sum(x̄[i] * val(v[:ρ01][i]) + (1 - x̄[i]) * val(v[:ρ03][i]) for i in 1:nd.nx))
    α = [val(v[:α][i]) for i in 1:nh]
    return (F=L + θ * (cy - du) + piL + pen, L=L, θterm=θ * (cy - du), piL=piL, pen=pen, α=α,
            r=[val(v[:r][s]) for s in 1:S], d=[val(v[:d][s]) for s in 1:S], a=[val(v[:a][s]) for s in 1:S])
end
show_dec(name, D) = @printf("  %-22s F=%10.2f | 리더 ℓᵀŷ=%10.2f θ항=%9.2f π̂ᵁ항=%8.2f λ벌점=%9.2f | α=%s r=%s d=%s\n", name, D.F, D.L, D.θterm, D.piL, D.pen,
                            string(round.(D.α; digits=1)), string(round.(D.r; digits=3)), string(round.(D.d; digits=3)))

# ---- 루트 노드 LP ----
P = _nz_build_sep(nd, x̄; optimizer=mk1)
DR = _nz_dr_attach!(P, nd, x̄; optimizer=mk1, vcut=true, cslift=true, fwd=true, kmc=true, dd=false)
for k in ("FeasibilityTol", "OptimalityTol"); set_optimizer_attribute(P.O.model, k, 1e-7); end
set_optimizer_attribute(P.O.model, "NumericFocus", 0)
l0, u0 = zeros(nh), copy(nd.hU)
lo, hi = _nz_dr_global(nd, DR); phib = fill(Inf, S)
if RANGE
    t = @elapsed (rg = _nz_dr_ranges(DR, nd, l0, u0, lo, hi, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), phib; deadline=time() + 120, mip_deadline=time() + 120))
    lo, hi, _, phib = rg
    @printf("  루트 범위 MIP %.1fs: 고정 비율 %.2f, φ̄ = %s\n", t, count(hi .- lo .<= 1e-4) / max(1, length(lo)), string(round.(phib; digits=1)))
end
_nz_sticky_set_box!(P, l0, u0); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
_nz_dr_set!(P, DR, nd, l0, u0, lo, hi, phib; comp=true, eq=falses(nh), dlo=zeros(0, S), dhi=zeros(0, S))
t = @elapsed (z = _nz_sticky_solve!(P; maxrounds=30, maxadd=5000))
v = P.O.v
Droot = decompose(v, value)
@printf("\n루트 LP %.4f (%.1fs, RLT 행 %d)\n", z, t, length(P.allrows))
show_dec("루트 완화", Droot)
F = nz_build_fixed_alpha(nd, x̄; optimizer=mk1)
zE = nz_eval_alpha!(F, nd, clamp.(Droot.α, l0, u0))
show_dec("α̂ 고정 (정확)", decompose(F.v, value))

# ---- 곱 한 종류씩 고정 ----
function fixed_solve(fixr, fixd)
    cs = ConstraintRef[]
    # belief 를 고정하고 그 곱 (ζL = α r̂, ζF = α d̂) 을 정확한 선형 등식으로 넣는다 (상자 McCormick 은 변수 범위를 쓰므로 등식만으론 안 조여짐)
    fixr && for s in 1:S; push!(cs, @constraint(P.O.model, v[:r][s] == Droot.r[s]))
        for i in 1:nh; push!(cs, @constraint(P.O.model, v[:ζL][i, s] == Droot.r[s] * v[:α][i])); end; end
    fixd && for s in 1:S; push!(cs, @constraint(P.O.model, v[:d][s] == Droot.d[s]))
        for i in 1:nh; push!(cs, @constraint(P.O.model, v[:ζF][i, s] == Droot.d[s] * v[:α][i])); end; end
    zz = _nz_sticky_solve!(P; maxrounds=30, maxadd=5000)
    D = decompose(v, value)
    delete(P.O.model, cs)
    return zz, D
end
for (nm, fr, fd) in (("r = r̂, ζL=αr̂", true, false), ("d = d̂, ζF=αd̂", false, true), ("r, d 둘 다", true, true))
    zz, D = fixed_solve(fr, fd)
    show_dec(nm, D)
end

# ---- 최적해 α* ----
r = nz_alpha_bnb_dr(nd, x̄; nworkers=12, time_limit=90.0, rel_gap=1e-3, verbose=false, ipopt_time=60.0, presolve=false, range_time=30.0, dd=false)
@printf("\nα-B&B 90 s: LB=%.4f UB=%.4f\n", r[:LB], r[:UB])
zs = nz_eval_alpha!(F, nd, r[:α])
show_dec("α* 고정 (최적 근처)", decompose(F.v, value))
# 루트 완화에 α = α* 를 고정하면 (곱은 모두 정확해짐)
# 상자를 한 점 {α*} 로 (bound 로 설정)
_nz_sticky_set_box!(P, r[:α], r[:α]); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
_nz_dr_set!(P, DR, nd, r[:α], r[:α], lo, hi, phib; comp=true, eq=falses(nh), dlo=zeros(0, S), dhi=zeros(0, S))
zz = _nz_sticky_solve!(P; maxrounds=30, maxadd=5000)
show_dec("완화 LP, 상자 = {α*}", decompose(v, value))

# =====================================================================
# (5) 리더 recourse 값함수 하한 cut (시험): Φ_s(α) = max{cᵀy : Ay ≤ nz_rhs(x̄, α, s)} 는 α 에 오목.
#     상자 꼭짓점 값으로 만든 affine 하한 g₀ + gᵀα ≤ Φ_s(α) (상자 위) → cᵀŷ_s ≥ g₀ r_s + gᵀ ζL[:, s]  (ζL = α r 이면 cᵀŷ_s ≥ r_s Φ_s(α) 의 완화)
#     전제: Ω 의 최적해 중 리더 ŷ 가 recourse 최적인 것이 있다 (θ 정확 벌점). 먼저 α* 에서 값이 그대로인지 확인.
# =====================================================================
m_, ny_ = nz_m(nd), nz_ny(nd)
act = findall(nd.hU .> 0)
function phi_model()
    M = Model(mk1); @variable(M, y[1:ny_] >= 0); @variable(M, α[1:nh])
    @constraint(M, con[k=1:m_], sum(nd.A[k, j] * y[j] for j in 1:ny_ if nd.A[k, j] != 0) -
                                 (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0) <= 0.0)
    @objective(M, Max, sum(nd.c[j] * y[j] for j in 1:ny_ if nd.c[j] != 0))
    return M, α, con
end
PM, Pα, Pcon = phi_model()
function Φ(s, αv)
    rr0 = nz_rhs(nd, x̄, zeros(nh), s)
    for k in 1:m_; set_normalized_rhs(Pcon[k], rr0[k]); end
    for i in 1:nh; fix(Pα[i], αv[i]; force=true); end
    optimize!(PM); termination_status(PM) == MOI.OPTIMAL || error("Φ LP: $(termination_status(PM))")
    return objective_value(PM)
end
# (5a) α* 에서 cᵀŷ_s ≥ r_s Φ_s(α*) 를 넣어도 값이 그대로인가
αs = r[:α]
cs5 = [@constraint(F.model, sum(nd.c[j] * F.v[:ŷ][j, s] for j in 1:ny_ if nd.c[j] != 0) >= Φ(s, αs) * F.v[:r][s]) for s in 1:S]
z5 = nz_eval_alpha!(F, nd, αs)
@printf("\n(5a) α* 고정 + 리더 recourse 최적 조건: %.4f (조건 없이 %.4f)\n", z5, zs)
show_dec("  α* + 조건", decompose(F.v, value))
delete(F.model, cs5)

# (5b) 루트 상자 [0, hU] 에서 꼭짓점 하한 cut 분리
_nz_sticky_set_box!(P, l0, u0); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
_nz_dr_set!(P, DR, nd, l0, u0, lo, hi, phib; comp=true, eq=falses(nh), dlo=zeros(0, S), dhi=zeros(0, S))
verts = [begin v_ = zeros(nh); for (n, i) in enumerate(act); v_[i] = ((b >> (n - 1)) & 1) == 1 ? u0[i] : l0[i]; end; v_ end for b in 0:(2^length(act) - 1)]
tΦ = @elapsed (Φv = [Φ(s, vv) for vv in verts, s in 1:S])
@printf("\n(5b) 꼭짓점 %d 개 × S=%d 의 Φ (%.1fs). Φ 범위 %s\n", length(verts), S, tΦ, string(round.(extrema(Φv); digits=1)))
function envelope(s, α̂)
    E = Model(mk1); @variable(E, g0); @variable(E, g[1:length(act)])
    @constraint(E, [n=1:length(verts)], g0 + sum(g[t] * verts[n][act[t]] for t in eachindex(act)) <= Φv[n, s])
    @objective(E, Max, g0 + sum(g[t] * α̂[act[t]] for t in eachindex(act))); optimize!(E)
    return objective_value(E), value(g0), value.(g)
end
nc = 0
for rnd in 1:30
    z_ = _nz_sticky_solve!(P; maxrounds=30, maxadd=5000)
    αh = value.(v[:α]); rh = value.(v[:r]); ŷh = value.(v[:ŷ]); ζh = value.(v[:ζL])    # cut 을 넣으면 해가 무효가 되므로 먼저 읽음
    added = 0
    for s in 1:S
        env, g0, g = envelope(s, αh)
        lhs = sum(nd.c[j] * ŷh[j, s] for j in 1:ny_ if nd.c[j] != 0)
        rhs = g0 * rh[s] + sum(g[t] * ζh[act[t], s] for t in eachindex(act))
        if lhs < rhs - 1e-6 * max(1.0, abs(rhs))
            @constraint(P.O.model, sum(nd.c[j] * v[:ŷ][j, s] for j in 1:ny_ if nd.c[j] != 0) >=
                                   g0 * v[:r][s] + sum(g[t] * v[:ζL][act[t], s] for t in eachindex(act)))
            added += 1; global nc += 1
        end
    end
    @printf("  round %2d: 루트 상한 %.4f, cut 추가 %d (누적 %d)\n", rnd, z_, added, nc)
    added == 0 && break
end
show_dec("루트 + 값함수 하한 cut", decompose(v, value))

# (5c) α* 주변 상자 (반폭 w·hU) 에서 cut 유무별 완화 상한 — B&B 가 시간을 쓰는 작은 상자에서의 효과
# 새 sep LP (cut 이 없는 상태) 를 상자마다 만들어 비교
function box_bound(lb, ub; cuts::Bool, rng::Bool=false)
    Q = _nz_build_sep(nd, x̄; optimizer=mk1)
    DQ = _nz_dr_attach!(Q, nd, x̄; optimizer=mk1, vcut=true, cslift=true, fwd=true, kmc=true, dd=false)
    for k in ("FeasibilityTol", "OptimalityTol"); set_optimizer_attribute(Q.O.model, k, 1e-7); end
    _nz_sticky_set_box!(Q, lb, ub); _nz_sep_set_wbox!(Q, Q.Lw, Q.Uw)
    lb_, hb_, pb_ = lo, hi, phib
    if rng
        tr = @elapsed (rg = _nz_dr_ranges(DQ, nd, lb, ub, lo, hi, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), phib; deadline=time() + 300, mip_deadline=time() + 300))
        lb_, hb_, _, pb_ = rg
        @printf("      [범위 MIP %.1fs: 고정 비율 %.2f, 평균 폭 %.3f (전역 %.3f)]
", tr, count(hb_ .- lb_ .<= 1e-4) / max(1, length(lb_)), sum(hb_ .- lb_) / length(lb_), sum(hi .- lo) / length(lo))
    end
    _nz_dr_set!(Q, DQ, nd, lb, ub, lb_, hb_, pb_; comp=true, eq=falses(nh), dlo=zeros(0, S), dhi=zeros(0, S))
    z0 = _nz_sticky_solve!(Q; maxrounds=30, maxadd=5000)
    cuts || return z0, 0
    vs = [begin v_ = zeros(nh); for (n, i) in enumerate(act); v_[i] = ((b >> (n - 1)) & 1) == 1 ? ub[i] : lb[i]; end; v_ end for b in 0:(2^length(act) - 1)]
    Φb = [Φ(s, vv) for vv in vs, s in 1:S]
    w = Q.O.v; n = 0; z_ = z0
    for rnd in 1:30
        αh = value.(w[:α]); rh = value.(w[:r]); ŷh = value.(w[:ŷ]); ζh = value.(w[:ζL]); added = 0
        for s in 1:S
            E = Model(mk1); @variable(E, g0); @variable(E, g[1:length(act)])
            @constraint(E, [q=1:length(vs)], g0 + sum(g[t] * vs[q][act[t]] for t in eachindex(act)) <= Φb[q, s])
            @objective(E, Max, g0 + sum(g[t] * αh[act[t]] for t in eachindex(act))); optimize!(E)
            g0v, gv = value(g0), value.(g)
            lhs = sum(nd.c[j] * ŷh[j, s] for j in 1:ny_ if nd.c[j] != 0)
            rhs = g0v * rh[s] + sum(gv[t] * ζh[act[t], s] for t in eachindex(act))
            if lhs < rhs - 1e-6 * max(1.0, abs(rhs))
                @constraint(Q.O.model, sum(nd.c[j] * w[:ŷ][j, s] for j in 1:ny_ if nd.c[j] != 0) >= g0v * w[:r][s] + sum(gv[t] * w[:ζL][act[t], s] for t in eachindex(act)))
                added += 1; n += 1
            end
        end
        added == 0 && break
        z_ = _nz_sticky_solve!(Q; maxrounds=30, maxadd=5000)
    end
    return z_, n
end
println("\n(5c) α* 주변 상자: 반폭 = w·hU (상자는 [0, hU] 로 자름), 최적값 ", round(zs; digits=2))
for wfrac in parse.(Float64, split(envf("DL_W", "0.5,0.25,0.1,0.05,0.02"), ","))
    lb = max.(αs .- wfrac .* nd.hU, 0.0); ub = min.(αs .+ wfrac .* nd.hU, nd.hU)
    if envf("DL_RANGEBOX", "0") == "1"
        t3 = @elapsed (zC, _) = box_bound(lb, ub; cuts=false, rng=true)
        t4 = @elapsed (zD, nD) = box_bound(lb, ub; cuts=true, rng=true)
        @printf("  w=%.2f 범위 MIP: cut 없음 %10.2f (gap %6.1f) | cut %2d 개 %10.2f (gap %6.1f)  [%.1fs / %.1fs]
", wfrac, zC, zC - zs, nD, zD, zD - zs, t3, t4)
    end
    t1 = @elapsed (zA, _) = box_bound(lb, ub; cuts=false)
    t2 = @elapsed (zB, nB) = box_bound(lb, ub; cuts=true)
    @printf("  w=%.2f: cut 없음 %10.2f (gap %6.1f) | cut %2d 개 %10.2f (gap %6.1f)  [%.1fs / %.1fs]\n", wfrac, zA, zA - zs, nB, zB, zB - zs, t1, t2)
end

# =====================================================================
# (6) 쌍대 범위를 상자 두 꼭짓점의 LP 로 (시험): σ_k 가 모든 α 좌표에 단조 감소 (값함수 submodular) 이면
#     max_{α∈B} σ_k = α = l 에서 최적 쌍대면 위의 max,  min_{α∈B} σ_k = α = u 에서 최적 쌍대면 위의 min.
#     범위 MIP (정확) 과 대조: 꼭짓점 범위가 MIP 범위를 포함하면 (같으면) 이 상자에서 성질이 맞음.
# =====================================================================
function face_extreme(s, αv, k, ismax)
    rr = nz_rhs(nd, x̄, αv, s)
    M = Model(mk1); @variable(M, y[1:ny_] >= 0)
    @constraint(M, [kk=1:m_], sum(nd.A[kk, j] * y[j] for j in 1:ny_ if nd.A[kk, j] != 0) <= rr[kk])
    @objective(M, Max, sum(nd.c[j] * y[j] for j in 1:ny_ if nd.c[j] != 0)); optimize!(M); z = objective_value(M)
    D = Model(mk1); @variable(D, σ[1:m_] >= 0)
    for kk in Hr; isfinite(nd.varpiU[kk]) && set_upper_bound(σ[kk], nd.varpiU[kk]); end
    @constraint(D, [j=1:ny_], sum(nd.A[kk, j] * σ[kk] for kk in 1:m_ if nd.A[kk, j] != 0) >= nd.c[j])
    @constraint(D, sum(rr[kk] * σ[kk] for kk in 1:m_ if rr[kk] != 0) <= z + 1e-7 * max(1.0, abs(z)))
    @objective(D, ismax ? MAX_SENSE : MIN_SENSE, σ[k]); optimize!(D)
    termination_status(D) == MOI.OPTIMAL || error("face LP $(termination_status(D))")
    return objective_value(D)
end
println("\n(6) 꼭짓점 LP 범위 vs 범위 MIP")
for wfrac in parse.(Float64, split(envf("DL_W6", "0.5,0.1,0.02"), ","))
    lb = max.(αs .- wfrac .* nd.hU, 0.0); ub = min.(αs .+ wfrac .* nd.hU, nd.hU)
    tm = @elapsed (rg = _nz_dr_ranges(DR, nd, lb, ub, lo, hi, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), phib; deadline=time() + 600, mip_deadline=time() + 600))
    mlo, mhi = rg[1], rg[2]
    tc = @elapsed begin
        clo = [face_extreme(s, ub, DR.Hk[a], false) for a in eachindex(DR.Hk), s in 1:S]
        chi = [face_extreme(s, lb, DR.Hk[a], true) for a in eachindex(DR.Hk), s in 1:S]
    end
    narrower = count((clo .> mlo .+ 1e-5) .| (chi .< mhi .- 1e-5))      # 꼭짓점 범위가 MIP 범위보다 좁은 성분 (단조성 위반 → 무효)
    wider = count((clo .< mlo .- 1e-5) .| (chi .> mhi .+ 1e-5))
    @printf("  w=%.2f: MIP %.1fs, 꼭짓점 LP %.2fs | 성분 %d 개 중 같음 %d, 꼭짓점이 더 좁음 (무효) %d, 더 넓음 %d | 평균 폭 MIP %.3f 꼭짓점 %.3f\n",
            wfrac, tm, tc, length(mlo), length(mlo) - narrower - wider, narrower, wider, sum(mhi .- mlo) / length(mlo), sum(max.(chi .- clo, 0)) / length(mlo))
    narrower > 0 && for a in eachindex(DR.Hk), s in 1:S
        (clo[a, s] > mlo[a, s] + 1e-5 || chi[a, s] < mhi[a, s] - 1e-5) &&
            @printf("      k=%d s=%d: MIP [%.4f, %.4f] 꼭짓점 [%.4f, %.4f]\n", DR.Hk[a], s, mlo[a, s], mhi[a, s], clo[a, s], chi[a, s])
    end
end
