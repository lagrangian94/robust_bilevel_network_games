"""
lab_lambda_certify.jl — follower 벌점 λᵁ 의 사후 인증 (문서 §10.19 (6)). 기존 솔버 파일은 고치지 않는다.

원고 Lemma certificate (iii) 는 유한한 λ̄ 의 존재만 준다 (Hoffman 상수, 닫힌 형태 없음). 대신 x 마다 다음으로 인증한다.
  V*_λ(x) = sup_{h, b} [ν φ_x(h) − λ gap_x(p̃, h)] 는 λ 의 아핀 함수들의 상한이라 볼록·비증가이고, V*_λ ≥ V* (Lemma (ii)).
  α-B&B 는 UB ≥ V*_λ(x) 를 준다. λ 와 무관하게 정확히 평가한 실행가능 반응의 값 LB_exact ≤ V*(x) 를 구하면
      UB − LB_exact ≤ tol·max(1, |LB_exact|)   ⇒   V*_λ(x) − V*(x) ≤ tol·max(1, |LB_exact|)
  즉 그 x 에서 λ 가 허용오차 안에서 exact 다 (이후 Ω 의 점이 만드는 cut 은 그 x 에서 V* 를 허용오차 이상 넘지 않음).
  아니면 λ 를 두 배로 올린다. X 가 유한하므로 모든 x 를 인증하면 cut 의 유효성이 보장된다.

정확한 평가 nz_exact_value(α): α 를 고정하면 Q_s(α) (follower recourse 최적값) 와 φ_s(α) (그 최적해 중 리더 손실 최대) 가 상수라
  max_{(a,r,d) ∈ 𝔅_δ} Σ_s r_s φ_s(α)   s.t.  c0ᵀα + Σ_s d_s Q_s(α) ≥ Σ_s b_sᵀπ̃_s + wᵀκ − tolF,  Aᵀπ̃_s ≥ d_s c,  Σ_s Hᵀπ̃_s + c0 ≤ Wᵀκ
  가 LP 다 (follower 2 단계 LP 의 강쌍대: α 가 belief d 에서 최적). θ, λ, big-M 이 없다. 실행 불가능이면 −∞ (α 는 어떤 belief 에서도 최적 반응이 아님).
  tolF 는 LP 허용오차 몫 (1e-7 상대).
LB_exact 후보: B&B incumbent α°, α° 의 belief d° 에서 follower LP 최적 꼭짓점 (simplex), d° 의 최적면 위에서 α° 에 가장 가까운 점.

환경변수: LC_NLOC = 15  LC_NA = 2  LC_NB = 3  LC_S = 3  NZ_SEED = 1  LC_POOL = store  LC_QUOTA = 300  LC_QB = 750  LC_LAM0 = 1000  LC_LAMMAX = 64000
          LC_TOL = 1e-3 (인증 허용오차, Benders 0.5% 보다 엄격)  LC_BNB_TL = 300  LC_WORKERS = 6  LC_X = "" (비우면 모든 x, 예 "1,2;3")
실행: julia -t 8,1 nonzero_sum/lab_lambda_certify.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d); pf(k, d) = parse(Float64, envf(k, d)); pint(k, d) = parse(Int, envf(k, d))

"α 고정 recourse: Q_s(α) = max cᵀy, φ_s(α) = max ℓᵀy s.t. cᵀy ≥ Q_s (pessimistic)"
function nz_recourse_values(nd::NZData, x̄, α)
    S, m, ny = nd.S, nz_m(nd), nz_ny(nd)
    Q = zeros(S); φ = zeros(S)
    for s in 1:S
        mdl = Model(GRB1); @variable(mdl, y[1:ny] >= 0); rr = nz_rhs(nd, x̄, α, s)
        @constraint(mdl, [k = 1:m], sum(nd.A[k, j] * y[j] for j in 1:ny if nd.A[k, j] != 0) <= rr[k])
        @objective(mdl, Max, dot(nd.c, y)); optimize!(mdl)
        termination_status(mdl) == MOI.OPTIMAL || error("recourse LP s=$s: $(termination_status(mdl))")
        Q[s] = objective_value(mdl)
        @constraint(mdl, dot(nd.c, y) >= Q[s] - 1e-9 * max(1.0, abs(Q[s])))
        @objective(mdl, Max, dot(nd.ell, y)); optimize!(mdl)
        termination_status(mdl) == MOI.OPTIMAL || error("pessimistic LP s=$s: $(termination_status(mdl))")
        φ[s] = objective_value(mdl)
    end
    return Q, φ
end

"λ·θ 와 무관한 정확한 값: α 가 어떤 belief 에서 follower 최적 반응일 때 리더 평가의 최댓값, 아니면 −Inf"
function nz_exact_value(nd::NZData, x̄, α; tolF=1e-7)
    S, m, ny, nh = nd.S, nz_m(nd), nz_ny(nd), nz_nh(nd); nW = size(nd.W, 1)
    any(α .< -1e-9) && return -Inf
    any(nd.W * α .> nd.wvec .+ 1e-7) && return -Inf
    Q, φ = nz_recourse_values(nd, x̄, α)
    q, ε̂, ε̃, β = nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta
    M = Model(GRB1)
    @variable(M, 0 <= a[1:S] <= 1); @variable(M, b[1:S] >= 0); @variable(M, 0 <= r[1:S] <= 1)
    @constraint(M, sum(a) == 1); @constraint(M, sum(r) == 1)
    @constraint(M, [s = 1:S], a[s] - b[s] <= q[s]); @constraint(M, [s = 1:S], a[s] + b[s] >= q[s])
    @constraint(M, sum(b) <= 2ε̂); @constraint(M, [s = 1:S], (1 - β) * r[s] <= a[s])
    @variable(M, 0 <= d[1:S] <= 1); @variable(M, e[1:S] >= 0); @constraint(M, sum(d) == 1)
    @constraint(M, [s = 1:S], d[s] - e[s] <= q[s]); @constraint(M, [s = 1:S], d[s] + e[s] >= q[s]); @constraint(M, sum(e) <= 2ε̃)
    if nz_coupled(nd)
        @variable(M, cpl[1:S] >= 0)
        @constraint(M, [s = 1:S], a[s] - d[s] <= cpl[s]); @constraint(M, [s = 1:S], d[s] - a[s] <= cpl[s]); @constraint(M, sum(cpl) <= 2nz_delta(nd))
    end
    @variable(M, π̃[1:m, 1:S] >= 0); @variable(M, κ[1:nW] >= 0)
    @constraint(M, [j = 1:ny, s = 1:S], sum(nd.A[k, j] * π̃[k, s] for k in 1:m if nd.A[k, j] != 0) >= nd.c[j] * d[s])
    @constraint(M, [i = 1:nh], sum(nd.hcoef[k] * π̃[k, s] for k in 1:m, s in 1:S if nd.hrow[k] == i) + nd.c0[i] <= sum(nd.W[l, i] * κ[l] for l in 1:nW))
    B = [nz_rhs(nd, x̄, zeros(nh), s) for s in 1:S]
    lhs = dot(nd.c0, α)
    @constraint(M, lhs + sum(Q[s] * d[s] for s in 1:S) >= sum(B[s][k] * π̃[k, s] for k in 1:m, s in 1:S if B[s][k] != 0) + dot(nd.wvec, κ)
                                                          - tolF * (1 + abs(lhs) + maximum(abs, Q)))
    @objective(M, Max, sum(φ[s] * r[s] for s in 1:S)); optimize!(M)
    st = termination_status(M)
    st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED) && return -Inf
    st == MOI.OPTIMAL || error("exact value LP: $st")
    return objective_value(M), value.(d)
end
_ev(nd, x̄, α) = (z = nz_exact_value(nd, x̄, α); z isa Tuple ? z[1] : z)

"incumbent α° 와 그 둘레의 정확히 최적인 반응들의 정확한 값 중 최댓값 (LB_exact)"
function exact_lb(nd, x̄, α°)
    nh, ny, S = nz_nh(nd), nz_ny(nd), nd.S
    hU, _ = nz_presolve_hU(nd, x̄); nd0 = nz_with_hU(nd, hU); a0 = clamp.(α°, 0.0, nd0.hU)
    best = _ev(nd, x̄, a0); src = "incumbent"
    F = nz_build_fixed_alpha(nd0, x̄; optimizer=GRB1); nz_eval_alpha!(F, nd0, a0)
    d° = [value(F.v[:d][s]) for s in 1:S]
    fm = Model(GRB1); set_optimizer_attribute(fm, "Method", 0)
    @variable(fm, 0 <= αf[i = 1:nh] <= nd0.hU[i]); @variable(fm, yf[1:ny, 1:S] >= 0); @constraint(fm, nd0.W * αf .<= nd0.wvec)
    for s in 1:S, k in 1:nz_m(nd0)
        @constraint(fm, sum(nd0.A[k, j] * yf[j, s] for j in 1:ny if nd0.A[k, j] != 0) <= nz_rhs(nd0, x̄, zeros(nh), s)[k] + (nd0.hrow[k] > 0 ? nd0.hcoef[k] * αf[nd0.hrow[k]] : 0.0))
    end
    fobj = @expression(fm, dot(nd0.c0, αf) + sum(d°[s] * nd0.c[j] * yf[j, s] for j in 1:ny, s in 1:S if nd0.c[j] != 0))
    @objective(fm, Max, fobj); optimize!(fm); vF = objective_value(fm)
    z = _ev(nd, x̄, clamp.(value.(αf), 0.0, nd0.hU)); z > best && (best = z; src = "꼭짓점")
    @variable(fm, dev[1:nh] >= 0); @constraint(fm, [i = 1:nh], dev[i] >= αf[i] - a0[i]); @constraint(fm, [i = 1:nh], dev[i] >= a0[i] - αf[i])
    @constraint(fm, fobj >= vF - 1e-9 * max(1.0, abs(vF))); @objective(fm, Min, sum(dev)); optimize!(fm)
    z = _ev(nd, x̄, clamp.(value.(αf), 0.0, nd0.hU)); z > best && (best = z; src = "최적면 사영")
    return best, src
end

nloc = pint("LC_NLOC", "15"); nA = pint("LC_NA", "2"); nB = pint("LC_NB", "3"); S = pint("LC_S", "3"); seed = pint("NZ_SEED", "1")
pool = Symbol(envf("LC_POOL", "store")); quota = pf("LC_QUOTA", "300"); qB = pf("LC_QB", "750")
λ0 = pf("LC_LAM0", "1000"); λmax = pf("LC_LAMMAX", "64000"); tol = pf("LC_TOL", "1e-3"); TL = pf("LC_BNB_TL", "300"); nw = pint("LC_WORKERS", "6")
capv = 150.0 * (nloc - nA - nB) / nA
nd1 = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=nA, nB=nB, NB=nB, S=S, seed=seed, bA=capv, bB=capv, qB=qB,
    dlo=50.0, dhi=150.0, reservation=:pooled, wres=300.0, pool_limit=pool, quota=quota, eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
θ̄ = nz_theta_circuit_exact(nd1)[1]
xs = [Float64.(collect(bb)) for bb in Iterators.product(fill(0:1, nd1.nx)...) if sum(bb) <= nd1.gamma]
xsel = strip(envf("LC_X", "")); xsel = isempty(xsel) ? nothing : [sort(parse.(Int, split(t, ","))) for t in split(xsel, ";")]
@printf("lab_lambda_certify: pooled(%s) d=%d A=%d B=%d seed=%d S=%d n_h=%d θ̄=%.1f λ0=%.0f 허용오차 %.0e\n", pool, nloc, nA, nB, seed, S, nz_nh(nd1), θ̄, λ0, tol); flush(stdout)
λcert = λ0
for x̄ in xs
    xi = findall(x̄ .> 0.5); xsel === nothing || xi in xsel || continue
    λ = λ0
    while true
        nd = nz_with_bounds(nd1; thetaU=θ̄, lambdaU=λ)
        t = @elapsed r = nz_alpha_bnb_dr(nd, x̄; nworkers=nw, time_limit=TL, rel_gap=tol / 4, verbose=false, ipopt_time=30.0)
        lb, src = exact_lb(nd, x̄, r[:α])
        ok = r[:UB] - lb <= tol * max(1.0, abs(lb))
        @printf("  x=%-9s λ=%7.0f  B&B [%.4f, %.4f] %s %.0fs | LB_exact %.4f (%s) | UB − LB_exact = %.3e → %s\n", string(xi), λ, r[:LB], r[:UB],
                r[:is_exact] ? "Opt" : "TL", t, lb, src, (r[:UB] - lb) / max(1.0, abs(lb)), ok ? "인증" : (r[:is_exact] ? "λ 부족" : "미종료"))
        flush(stdout)
        ok && (global λcert = max(λcert, λ); break)
        r[:is_exact] || error("B&B 가 시간 안에 안 끝나 판정 불가 (x=$xi, λ=$λ) — LC_BNB_TL 을 늘릴 것")
        λ *= 2
        λ > λmax && error("λ 가 $λmax 를 넘어도 인증 안 됨 (x=$xi)")
    end
end
@printf("RESULT 인증된 λᵁ = %.0f (모든 x 에서 UB − LB_exact ≤ %.0e 상대)\n", λcert, tol)
