"""
lab_kkt_start.jl — 분기 차원이 큰 인스턴스에서 KKT 재정식화 (nz_kkt_value_reduced 와 같은 모델: 곱은 d_s π_ks 와 r_s φ_s 뿐, α 와의 곱이 없다) 를
Gurobi 로 풀 때, (1) belief 표본으로 얻은 최적 반응을 시작해로 주고 (2) 쌍대·예약에 유효한 상계를 주면 상한이 닫히는지 본다.

기존 nz_kkt_value_reduced 는 쌍대 π 에 상계가 없어 d × π 의 완화가 약하고, 240 s 에 실행가능해도 못 찾았다 (문서 §10.15).
여기서 넣는 상계 (location 인스턴스, 모두 "그 범위의 최적 쌍대가 있다" 는 종류):
  예약 행 ϖᵁ = p,  수요 행 max_i (c_ij + p),  용량 행 π_Fᵁ (열린 B 점포) 또는 0 (A 점포: 총용량 > 총수요라 여유),  예약 상한 = min(할당량, 고객의 최대 수요)
시작해: TV 볼의 d 격자 (S = 3) 또는 무작위 표본에서 follower LP 의 최적 반응을 모아 α 고정 LP 로 값을 매기고 가장 좋은 (α, d) 를 준다.

환경변수: NZ_SEED = 1  KS_S = 3  KS_X = "1"  KS_NLOC = 15  KS_TL = 600  KS_THREADS = 12  KS_GAP = 1e-3  KS_START = 1  KS_BOUNDS = 1  KS_N = 5000
실행: julia -t 2 nonzero_sum/lab_kkt_start.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics
const GRB_ENV = Gurobi.Env()
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
envf(k, d) = get(ENV, k, d); pf(s) = parse(Float64, s)
seed = parse(Int, envf("NZ_SEED", "1")); S = parse(Int, envf("KS_S", "3")); nloc = parse(Int, envf("KS_NLOC", "15"))
xi = parse.(Int, split(envf("KS_X", "1"), ",")); TL = pf(envf("KS_TL", "600")); GAP = pf(envf("KS_GAP", "1e-3"))
useS = envf("KS_START", "1") == "1"; useB = envf("KS_BOUNDS", "1") == "1"
cap = 150.0 * (nloc - 5) / 2
nd0 = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=2, nB=3, NB=3, S=S, seed=seed,
    bA=cap, bB=cap, qB=cap, dlo=50.0, dhi=150.0, reservation=:pair, quota=300.0, wres=300.0, eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
x̄ = zeros(nd0.nx); x̄[xi] .= 1
hU1, _ = nz_presolve_hU(nd0, x̄)
mt = nd0.meta; nst, ncu, nA = mt[:nst], mt[:ncu], mt[:nA]; ξ = mt[:xi]
useB && (hU1 = [min(hU1[mt[:hidx](i, j)], maximum(ξ[j, :])) for i in 1:nst for j in 1:ncu])
nd = nz_with_hU(nd0, hU1)
A, c, ℓ = nd.A, nd.c, nd.ell; m, ny, nh = nz_m(nd), nz_ny(nd), nz_nh(nd); nW = size(nd.W, 1); q = nd.q_hat; ε̂, ε̃, β = nd.eps_hat, nd.eps_tilde, nd.beta
@printf("lab_kkt_start: 무작위 d=%d seed=%d S=%d x=%s n_h=%d (활성 %d) | 시작해 %s, 상계 %s, 제한 %.0fs\n", nloc, seed, S, string(xi), nh, count(nd.hU .> 0),
        useS ? "켬" : "끔", useB ? "켬" : "끔", TL); flush(stdout)

# ---------------- 시작해: belief 표본의 최적 반응 ----------------
best = (-Inf, zeros(nh), copy(q)); tS = 0.0
if useS
    global tS = @elapsed begin
        fm = Model(GRB1); @variable(fm, 0 <= αf[i = 1:nh] <= nd.hU[i]); @variable(fm, yf[1:ny, 1:S] >= 0); @constraint(fm, nd.W * αf .<= nd.wvec)
        for s in 1:S, k in 1:m
            @constraint(fm, sum(A[k, j] * yf[j, s] for j in 1:ny if A[k, j] != 0) <=
                            nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * αf[nd.hrow[k]] : 0.0) + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0))
        end
        ds = Vector{Float64}[]
        if S == 3
            for d1 in 0:0.01:1, d2 in 0:0.01:(1-d1)
                d = [d1, d2, 1 - d1 - d2]; sum(abs, d .- q) <= 2ε̃ + 1e-9 && push!(ds, d)
            end
        else
            rng = MersenneTwister(1)
            for _ in 1:parse(Int, envf("KS_N", "5000"))
                δ = randn(rng, S); δ .-= mean(δ); δ .*= (2ε̃ * rand(rng)^(1 / (S - 1))) / sum(abs, δ); d = q .+ δ
                any(<(0), d) && (d = max.(d, 0.0); d ./= sum(d)); sum(abs, d .- q) <= 2ε̃ + 1e-9 && push!(ds, d)
            end
        end
        seen = Dict{Vector{Float64},Tuple{Vector{Float64},Vector{Float64}}}()
        for d in ds
            @objective(fm, Max, dot(nd.c0, αf) + sum(d[s] * c[j] * yf[j, s] for j in 1:ny, s in 1:S if c[j] != 0)); optimize!(fm)
            a = value.(αf); get!(seen, round.(a; digits=3), (a, d))
        end
        F = nz_build_fixed_alpha(nd, x̄; optimizer=GRB1)
        for (a, d) in values(seen)
            z = nz_eval_alpha!(F, nd, a)
            isfinite(z) && z > best[1] && (global best = (z, a, value.(F.v[:d])))
        end
    end
    @printf("시작해: belief 표본 %d → 서로 다른 반응 %d → 최댓값 %.4f (%.1fs)\n", length(ds), length(seen), best[1], tS); flush(stdout)
end

# ---------------- KKT 모델 (nz_kkt_value_reduced 와 같은 구조 + 상계 + 시작해) ----------------
model = Model(() -> Gurobi.Optimizer(GRB_ENV))
set_optimizer_attribute(model, "NonConvex", 2); set_optimizer_attribute(model, "MIPGap", GAP); set_time_limit_sec(model, TL)
set_optimizer_attribute(model, "Threads", parse(Int, envf("KS_THREADS", "12")))
@variable(model, 0 <= a[1:S] <= 1); @variable(model, b[1:S] >= 0); @variable(model, 0 <= r[1:S] <= 1)
@constraint(model, sum(a) == 1); @constraint(model, sum(r) == 1)
@constraint(model, [s = 1:S], a[s] - b[s] <= q[s]); @constraint(model, [s = 1:S], a[s] + b[s] >= q[s])
@constraint(model, sum(b) <= 2ε̂); @constraint(model, [s = 1:S], (1 - β) * r[s] <= a[s])
@variable(model, max(0.0, q[s] - 2ε̃) <= d[s = 1:S] <= min(1.0, q[s] + 2ε̃)); @variable(model, e[1:S] >= 0); @constraint(model, sum(d) == 1)
@constraint(model, [s = 1:S], d[s] - e[s] <= q[s]); @constraint(model, [s = 1:S], d[s] + e[s] >= q[s]); @constraint(model, sum(e) <= 2ε̃)
if nz_coupled(nd)
    @variable(model, cpl[1:S] >= 0)
    @constraint(model, [s = 1:S], a[s] - d[s] <= cpl[s]); @constraint(model, [s = 1:S], d[s] - a[s] <= cpl[s]); @constraint(model, sum(cpl) <= 2nz_delta(nd))
end
@variable(model, 0 <= α[i = 1:nh] <= nd.hU[i]); @variable(model, y[1:ny, 1:S] >= 0); @variable(model, π[1:m, 1:S] >= 0); @variable(model, κ[1:nW] >= 0)
if useB
    # A 총용량 > 총수요라 여유 있는 A 점포가 항상 있다 → 수요 쌍대 ≤ max_{a∈A} c_aj + p, 점포 i 의 용량 쌍대 ≤ max_j (그 값 − c_ij)⁺ (= meta[:gapB], nz_data.jl)
    cAj = [maximum(mt[:dist][a, j] for a in 1:nA) + mt[:p] for j in 1:ncu]
    for s in 1:S
        for j in 1:ncu; set_upper_bound(π[mt[:rdem](j), s], cAj[j]); end
        for k in 1:m
            nd.hrow[k] > 0 && set_upper_bound(π[k, s], nd.varpiU[k])
        end
        for i in 1:nst; set_upper_bound(π[mt[:rcap](i), s], mt[:gapB][i]); end
    end
    for l in 1:nW; set_upper_bound(κ[l], maximum(nd.varpiU[k] for k in 1:m if nd.hrow[k] > 0)); end   # κ ≤ Σ_s d_s π ≤ ϖᵁ (c0 < 0)
    for j in 1:ny, s in 1:S; set_upper_bound(y[j, s], maximum(sum(ξ; dims=1))); end
end
rhs(s) = [nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0) + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0) for k in 1:m]
R = [rhs(s) for s in 1:S]
slackP = [@expression(model, R[s][k] - sum(A[k, j] * y[j, s] for j in 1:ny if A[k, j] != 0)) for k in 1:m, s in 1:S]
slackD = [@expression(model, sum(A[k, j] * π[k, s] for k in 1:m if A[k, j] != 0) - c[j]) for j in 1:ny, s in 1:S]
@constraint(model, slackP .>= 0); @constraint(model, slackD .>= 0)
slackW = [@expression(model, nd.wvec[l] - sum(nd.W[l, i] * α[i] for i in 1:nh)) for l in 1:nW]
@constraint(model, slackW .>= 0)
hk = [k for k in 1:m if nd.hrow[k] > 0 && nd.hU[nd.hrow[k]] > 0]
@variable(model, ρ[k = hk, s = 1:S] >= 0); @constraint(model, [k = hk, s = 1:S], ρ[k, s] == d[s] * π[k, s])
act = findall(nd.hU .> 0)
@variable(model, sA[act] >= 0)
@constraint(model, [i = act], sA[i] == sum(nd.W[l, i] * κ[l] for l in 1:nW) - nd.c0[i] - sum(nd.hcoef[k] * ρ[k, s] for k in hk, s in 1:S if nd.hrow[k] == i))
for k in 1:m, s in 1:S; _comp!(model, π[k, s], slackP[k, s]); end
for j in 1:ny, s in 1:S; _comp!(model, y[j, s], slackD[j, s]); end
for l in 1:nW; _comp!(model, κ[l], slackW[l]); end
for i in act; _comp!(model, α[i], sA[i]); end
@variable(model, ŷ[1:ny, 1:S] >= 0)
@constraint(model, [k = 1:m, s = 1:S], sum(A[k, j] * ŷ[j, s] for j in 1:ny if A[k, j] != 0) <= R[s][k])
@constraint(model, [s = 1:S], sum(c[j] * ŷ[j, s] for j in 1:ny) >= sum(c[j] * y[j, s] for j in 1:ny))
φlo, φhi = _phi_range(nd, x̄; optimizer=GRB1)
@variable(model, φlo[s] <= φ[s = 1:S] <= φhi[s]); @constraint(model, [s = 1:S], φ[s] == sum(ℓ[j] * ŷ[j, s] for j in 1:ny if ℓ[j] != 0))
@objective(model, Max, sum(r[s] * φ[s] for s in 1:S))
if useS && isfinite(best[1])
    for i in 1:nh; set_start_value(α[i], best[2][i]); end
    for s in 1:S; set_start_value(d[s], best[3][s]); end
end
# 진행 기록: 30 s 마다 (incumbent, bound)
t0 = time()
optimize!(model)
st = termination_status(model)
val = has_values(model) ? objective_value(model) : NaN; bd = objective_bound(model)
@printf("RESULT kkt_start S=%d 시작해=%s 상계=%s status=%s value=%.4f bound=%.4f gap=%.3e time=%.1f (시작해 계산 %.1fs) nodes=%d\n", S, useS ? "켬" : "끔", useB ? "켬" : "끔",
        string(st), val, bd, isnan(val) ? NaN : (bd - val) / max(1.0, abs(val)), time() - t0, tS, Int(MOI.get(model, MOI.NodeCount())))
