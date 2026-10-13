"""
lab_resp_cuts.jl — 고정 x̄ 의 oracle 을 **반응 생성 (value-function cut)** 으로 푸는 시제품. 분기 차원 n_h 와 무관하게 곱이 2S 개뿐인 정식화.

  V*(x̄) = sup { Σ_s r_s φ_s :  (a, r, d) ∈ 𝔅,  α ∈ Ψ(x̄, d),  φ_s = ℓᵀŷ_s,  ŷ_s 는 α 에서 recourse 최적 }

follower 의 2 단계 LP 는 실행가능 집합이 belief d 와 무관하고 d 는 목적에만 들어간다. 따라서 follower 의 실행가능해 (α^k, y^k) 하나마다
    c0ᵀα + Σ_s d_s Q_s(α)  ≥  c0ᵀα^k + Σ_s d_s cᵀy^k_s          (모든 d 에서 유효: follower 는 α 를 α^k 보다 못하게 고르지 않는다)
가 Ψ 의 유효 부등식이고, 이런 cut 을 다 모으면 α ∈ Ψ(x̄, d) 와 같다. 마스터에서 Q_s(α) = cᵀy_s (y_s 는 recourse 최적: KKT, indicator) 로 두면
cut 의 곱은 w_s = d_s · (cᵀy_s) 의 S 개, 목적의 곱은 r_s · φ_s 의 S 개뿐이다 (α 와의 곱, big-M, θ 벌점이 없다).

  반복:  마스터 (cut 일부만) 를 풀어 상한 UB 와 (d̂, α̂) → d̂ 에서 follower LP 를 풀어 실제 최적 반응 (α*, y*) →
         α* 를 α 고정 LP 로 평가해 하한 LB 갱신 → α̂ 가 d̂ 에서 이미 최적이면 끝, 아니면 (α*, y*) 의 cut 을 추가
  마스터의 값은 항상 유효한 상한이고 (완화), 반응의 수가 유한해 유한 번에 끝난다.
recourse 최적성 (리더 복사본 ŷ_s 가 recourse 최적이어야 함: d_s = 0 인 시나리오에서도) 은 시나리오별 KKT (indicator) 로 둔다 — 이 부분의 이진변수가 S (m + ny) 개.

환경변수: NZ_SEED = 1  RC_S = 3  RC_X = "1"  RC_NLOC = 15  RC_INST = random | sgb  RC_TL = 1800  RC_THREADS = 12  RC_GAP = 1e-3
          RC_KKT = 1 (기본: 시나리오별 recourse KKT) | 0 (KKT 없음: y_s 는 실행가능하기만 하면 됨. d_s > 0 인 시나리오는 cut 이 recourse 최적성을 함의하지만
                    d_s = 0 인 시나리오의 리더 복사본은 풀려 있어 상한이 느슨할 수 있다 — 얼마나 느슨한지 보는 용도)
          RC_DMIN = 0 (follower belief 의 하한 d_s ≥ RC_DMIN. 0 보다 크면 belief 집합을 줄인 다른 문제: KKT 없는 마스터가 정확해지는 조건을 보는 용도)
          RC_INIT = nominal (처음 cut: d = q̂ 의 반응 하나) | sample (belief 표본의 반응 전부, RC_N 개 표본)   RC_MASTER_TL = 600
실행: julia -t 2 nonzero_sum/lab_resp_cuts.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics
const GRB_ENV = Gurobi.Env()
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
envf(k, d) = get(ENV, k, d); pf(s) = parse(Float64, s)
seed = parse(Int, envf("NZ_SEED", "1")); S = parse(Int, envf("RC_S", "3")); nloc = parse(Int, envf("RC_NLOC", "15")); inst = envf("RC_INST", "random")
xi = parse.(Int, split(envf("RC_X", inst == "sgb" ? "1,2" : "1"), ",")); TL = pf(envf("RC_TL", "1800")); GAP = pf(envf("RC_GAP", "1e-3"))
if inst == "sgb"
    ndb = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=seed, eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
    nd0 = nz_with_bounds(ndb; thetaU=nz_theta_circuit_exact(ndb)[1])
else
    cap = 150.0 * (nloc - 5) / 2
    nd0 = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=2, nB=3, NB=3, S=S, seed=seed,
        bA=cap, bB=cap, qB=cap, dlo=50.0, dhi=150.0, reservation=:pair, quota=300.0, wres=300.0, eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
end
x̄ = zeros(nd0.nx); x̄[xi] .= 1
hU1, _ = nz_presolve_hU(nd0, x̄)
mt = nd0.meta; nst, ncu, nA = mt[:nst], mt[:ncu], mt[:nA]; ξ = mt[:xi]
hU1 = [min(hU1[mt[:hidx](i, j)], maximum(ξ[j, :])) for i in 1:nst for j in 1:ncu]           # 예약 상한 ≤ 고객의 최대 수요 (지배 논증, 문서 §10.15)
nd = nz_with_hU(nd0, hU1)
A, c, ℓ = nd.A, nd.c, nd.ell; m, ny, nh = nz_m(nd), nz_ny(nd), nz_nh(nd); nW = size(nd.W, 1); q = nd.q_hat; ε̂, ε̃, β = nd.eps_hat, nd.eps_tilde, nd.beta
@printf("lab_resp_cuts: %s seed=%d S=%d x=%s n_h=%d (활성 %d) | gap %.0e, 제한 %.0fs, 처음 cut %s\n", inst == "sgb" ? "SGB128" : "무작위 d=$nloc", seed, S, string(xi), nh,
        count(nd.hU .> 0), GAP, TL, envf("RC_INIT", "nominal")); flush(stdout)
t0 = time()

# ---------------- follower LP (반응 oracle) 와 α 고정 LP (하한) ----------------
fm = Model(GRB1); @variable(fm, 0 <= αf[i = 1:nh] <= nd.hU[i]); @variable(fm, yf[1:ny, 1:S] >= 0); @constraint(fm, nd.W * αf .<= nd.wvec)
for s in 1:S, k in 1:m
    @constraint(fm, sum(A[k, j] * yf[j, s] for j in 1:ny if A[k, j] != 0) <=
                    nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * αf[nd.hrow[k]] : 0.0) + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0))
end
"belief d 에서 follower 의 최적 반응 (α, y) 과 최적값. d_s = 0 인 시나리오의 y_s 도 recourse 최적이 되도록 작은 가중을 더해 다시 푼다 (cut 은 어떤 실행가능해로도 유효)"
function follower(d)
    @objective(fm, Max, dot(nd.c0, αf) + sum(d[s] * c[j] * yf[j, s] for j in 1:ny, s in 1:S if c[j] != 0)); optimize!(fm)
    termination_status(fm) == MOI.OPTIMAL || error("follower LP: $(termination_status(fm))")
    a = value.(αf); v = objective_value(fm)
    for i in 1:nh; fix(αf[i], a[i]; force=true); end                       # α 를 고정하고 모든 시나리오의 recourse 를 최적으로
    @objective(fm, Max, sum(c[j] * yf[j, s] for j in 1:ny, s in 1:S if c[j] != 0)); optimize!(fm)
    y = value.(yf)
    for i in 1:nh; unfix(αf[i]); set_lower_bound(αf[i], 0.0); set_upper_bound(αf[i], nd.hU[i]); end
    return a, y, v
end
F = nz_build_fixed_alpha(nd, x̄; optimizer=GRB1)
resp = Tuple{Vector{Float64},Vector{Float64}}[]            # (α^k, [cᵀy^k_s]_s)
LB = Ref(-Inf); bestα = Ref(zeros(nh))
function add_resp!(a, y)
    key = round.(a; digits=4)
    any(r -> round.(r[1]; digits=4) == key, resp) && return false
    push!(resp, (a, [sum(c[j] * y[j, s] for j in 1:ny) for s in 1:S]))
    z = nz_eval_alpha!(F, nd, a)
    isfinite(z) && z > LB[] && (LB[] = z; bestα[] = copy(a))
    return true
end
function init_resp!()
    if envf("RC_INIT", "nominal") == "sample"
        rng = MersenneTwister(1)
        for _ in 1:parse(Int, envf("RC_N", "2000"))
            δ = randn(rng, S); δ .-= mean(δ); δ .*= (2ε̃ * rand(rng)^(1 / (S - 1))) / sum(abs, δ); dd = q .+ δ
            any(<(0), dd) && (dd = max.(dd, 0.0); dd ./= sum(dd)); sum(abs, dd .- q) <= 2ε̃ + 1e-9 || continue
            a0, y0, _ = follower(dd); add_resp!(a0, y0)
        end
    else
        a0, y0, _ = follower(q); add_resp!(a0, y0)
    end
end
init_resp!()
@printf("처음 반응 %d 개, LB = %.4f (%.1fs)\n", length(resp), LB[], time() - t0); flush(stdout)

# ---------------- 마스터 ----------------
M = Model(() -> Gurobi.Optimizer(GRB_ENV)); set_silent(M)
set_optimizer_attribute(M, "NonConvex", 2); set_optimizer_attribute(M, "Threads", parse(Int, envf("RC_THREADS", "12")))
@variable(M, 0 <= a[1:S] <= 1); @variable(M, b[1:S] >= 0); @variable(M, 0 <= r[1:S] <= 1)
@constraint(M, sum(a) == 1); @constraint(M, sum(r) == 1)
@constraint(M, [s = 1:S], a[s] - b[s] <= q[s]); @constraint(M, [s = 1:S], a[s] + b[s] >= q[s])
@constraint(M, sum(b) <= 2ε̂); @constraint(M, [s = 1:S], (1 - β) * r[s] <= a[s])
@variable(M, max(0.0, q[s] - 2ε̃) <= d[s = 1:S] <= min(1.0, q[s] + 2ε̃)); @variable(M, e[1:S] >= 0); @constraint(M, sum(d) == 1)
@constraint(M, [s = 1:S], d[s] - e[s] <= q[s]); @constraint(M, [s = 1:S], d[s] + e[s] >= q[s]); @constraint(M, sum(e) <= 2ε̃)
if nz_coupled(nd)
    @variable(M, cpl[1:S] >= 0)
    @constraint(M, [s = 1:S], a[s] - d[s] <= cpl[s]); @constraint(M, [s = 1:S], d[s] - a[s] <= cpl[s]); @constraint(M, sum(cpl) <= 2nz_delta(nd))
end
totd = maximum(sum(ξ; dims=1))
@variable(M, 0 <= α[i = 1:nh] <= nd.hU[i]); @constraint(M, nd.W * α .<= nd.wvec)
@variable(M, 0 <= y[1:ny, 1:S] <= totd); @variable(M, π[1:m, 1:S] >= 0)
# 쌍대 상계 (location): 수요 행 max_{a∈A} c_aj + p, 예약 행 ϖᵁ, 용량 행 meta[:gapB] (A 총용량 > 총수요라 여유 있는 A 점포가 항상 있다)
cAj = [maximum(mt[:dist][a, j] for a in 1:nA) + mt[:p] for j in 1:ncu]
for s in 1:S
    for j in 1:ncu; set_upper_bound(π[mt[:rdem](j), s], cAj[j]); end
    for k in 1:m; nd.hrow[k] > 0 && set_upper_bound(π[k, s], nd.varpiU[k]); end
    for i in 1:nst; set_upper_bound(π[mt[:rcap](i), s], mt[:gapB][i]); end
end
rhsM(s) = [nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0) + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0) for k in 1:m]
R = [rhsM(s) for s in 1:S]
slackP = [@expression(M, R[s][k] - sum(A[k, j] * y[j, s] for j in 1:ny if A[k, j] != 0)) for k in 1:m, s in 1:S]
slackD = [@expression(M, sum(A[k, j] * π[k, s] for k in 1:m if A[k, j] != 0) - c[j]) for j in 1:ny, s in 1:S]
@constraint(M, slackP .>= 0); @constraint(M, slackD .>= 0)
useK = envf("RC_KKT", "1") == "1"; dminv = pf(envf("RC_DMIN", "0"))
if useK
    for k in 1:m, s in 1:S; _comp!(M, π[k, s], slackP[k, s]); end           # recourse KKT (시나리오별): y_s 가 α 에서 recourse 최적
    for j in 1:ny, s in 1:S; _comp!(M, y[j, s], slackD[j, s]); end
end
dminv > 0 && for s in 1:S; set_lower_bound(d[s], max(lower_bound(d[s]), dminv)); end
println("마스터: recourse KKT ", useK ? "켬" : "끔", dminv > 0 ? @sprintf(", d_s ≥ %.3g", dminv) : ""); flush(stdout)
@variable(M, ŷ[1:ny, 1:S] >= 0)                                          # 리더 복사본: recourse 최적해 중 리더에게 가장 나쁜 것
@constraint(M, [k = 1:m, s = 1:S], sum(A[k, j] * ŷ[j, s] for j in 1:ny if A[k, j] != 0) <= R[s][k])
qexp = [@expression(M, sum(c[j] * y[j, s] for j in 1:ny if c[j] != 0)) for s in 1:S]
@constraint(M, [s = 1:S], sum(c[j] * ŷ[j, s] for j in 1:ny if c[j] != 0) >= qexp[s])
φlo, φhi = _phi_range(nd, x̄; optimizer=GRB1)
@variable(M, φlo[s] <= φ[s = 1:S] <= φhi[s]); @constraint(M, [s = 1:S], φ[s] == sum(ℓ[j] * ŷ[j, s] for j in 1:ny if ℓ[j] != 0))
qlo = [-sum((maximum(-c) ) * ξ[j, s] for j in 1:ncu) for s in 1:S]
@variable(M, qlo[s] <= qv[s = 1:S] <= 0); @constraint(M, [s = 1:S], qv[s] == qexp[s])
@variable(M, w[1:S]); @constraint(M, [s = 1:S], w[s] == d[s] * qv[s])                 # w_s = d_s Q_s(α)
@objective(M, Max, sum(r[s] * φ[s] for s in 1:S))
ncut = Ref(0)
function addcut!(k)
    αk, Qk = resp[k]
    @constraint(M, dot(nd.c0, α) + sum(w) >= dot(nd.c0, αk) + sum(Qk[s] * d[s] for s in 1:S))
    ncut[] += 1
end
for k in eachindex(resp); addcut!(k); end

UB = Inf; it = 0; status = "TimeLimit"; tM = 0.0
while time() - t0 < TL
    global it += 1
    set_time_limit_sec(M, min(pf(envf("RC_MASTER_TL", "600")), max(TL - (time() - t0), 1.0)))
    set_optimizer_attribute(M, "MIPGap", GAP / 2)
    global tM += @elapsed optimize!(M)
    st = termination_status(M)
    has_values(M) || (@printf("  it %d: 마스터 %s (해 없음)\n", it, string(st)); break)
    global UB = min(UB, objective_bound(M))
    d̂ = clamp.(value.(d), 0.0, 1.0); d̂ ./= sum(d̂); α̂ = value.(α); zM = objective_value(M)
    aS, yS, vF = follower(d̂)
    gM = dot(nd.c0, α̂) + sum(d̂[s] * value(qv[s]) for s in 1:S)                       # 마스터 해의 follower 값
    if !useK                                                                          # KKT 없는 마스터: 해의 recourse 최적성 여유 Q_s(α̂) − q̂_s 를 따로 본다
        for i in 1:nh; fix(αf[i], clamp(α̂[i], 0.0, nd.hU[i]); force=true); end
        @objective(fm, Max, sum(c[j] * yf[j, s] for j in 1:ny, s in 1:S if c[j] != 0)); optimize!(fm)
        Qα̂ = [sum(c[j] * value(yf[j, s]) for j in 1:ny) for s in 1:S]
        for i in 1:nh; unfix(αf[i]); set_lower_bound(αf[i], 0.0); set_upper_bound(αf[i], nd.hU[i]); end
        @printf("         d̂ = %s r̂ = %s | recourse 여유 Q_s(α̂) − q̂_s = %s
", string(round.(d̂; digits=3)), string(round.(value.(r); digits=3)),
                string(round.(Qα̂ .- value.(qv); digits=2)))
    end
    new = add_resp!(aS, yS)
    zα̂ = nz_eval_alpha!(F, nd, α̂)                                                    # 마스터의 α̂ 가 어떤 belief 에서 최적 반응이면 그 값도 하한
    isfinite(zα̂) && zα̂ > LB[] && (LB[] = zα̂; bestα[] = copy(α̂))
    gap = (UB - LB[]) / max(1.0, abs(LB[]))
    @printf("  it %3d: 마스터 값 %.4f 상한 %.4f (%s, %.1fs) | follower 최적성 여유 %.4f | LB %.4f gap %.2e | cut %d %s\n", it, zM, UB, string(st), solve_time(M),
            vF - gM, LB[], gap, ncut[], new ? "" : "(새 반응 없음)"); flush(stdout)
    if gap <= GAP
        global status = "Optimal"; break
    end
    if !new
        # 같은 반응이 다시 나왔다: 마스터 해가 그 cut 을 허용오차 안에서 만족 → 마스터 해가 (거의) follower 최적. 값의 차이는 마스터의 MIPGap·동률 처리 몫
        vF - gM <= 1e-6 * max(1.0, abs(vF)) && (global status = "Stalled(follower-optimal)")
        global status = status == "TimeLimit" ? "Stalled" : status
        break
    end
    addcut!(length(resp))
end
@printf("RESULT resp_cuts %s S=%d n_h=%d status=%s LB=%.4f UB=%.4f gap=%.3e iters=%d cuts=%d time=%.1f (마스터 %.1fs)\n", inst, S, nh, status, LB[], UB,
        (UB - LB[]) / max(1.0, abs(LB[])), it, ncut[], time() - t0, tM)
