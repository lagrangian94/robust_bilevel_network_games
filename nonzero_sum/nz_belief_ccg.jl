"""
nz_belief_ccg.jl — 비제로섬 문제의 belief 블록 column-and-constraint generation (원고 6 절 Algorithm 2, 2026-11-21).

원고 Section "Column-and-Constraint Generation" 의 구현. 시제품은 docs/nz_cut_locality_scripts/dromu.jl, 설명은 docs/nz_belief_block_ccg.md.

  마스터 (CMP):  min q_xᵀx + t0  s.t.  t0 ≥ L_b(x) (b ∈ 𝒞, belief 블록),  안전장치 (no-good cut, restricted ISP cut)
  belief 블록 L_b(x) = min_{plan (h,y)} [ C_L + μ (C_F − η C_L⁺ − V_F^η(x)) ]   (원고 식 (block))
      C_F = c0 비용 + Σ_s d_s (follower 비용),  C_L = Σ_s r_s ℓᵀy_s (리더 손실),  C_L⁺ = d_s > 0 인 시나리오만의 C_L.
      plan 은 마스터 안의 사본 (x 는 우변), V_F^η(x) 는 쌍대 + McCormick (정수 x 에서 정확).
      모든 μ ≥ 0 에서 하한 (η ≤ η̄_b 일 때). d > 0 이고 μ 가 크면 정확. d_s = 0 인 시나리오에서는 유효하지만 느슨하다.
  x̄ 에서: 싼 belief 탐색 (후보 belief 의 follower 반응 → 그 반응을 최적으로 만드는 belief 위의 정확 평가, LP) → 마스터 값보다 크면 블록 추가.
          없으면 α-B&B (nz_alpha_bnb_dr) 로 V*(x̄) 의 상한과 최악 belief.
  x̄ 확보 (원고 Algorithm 2 의 10–11 줄): 새 블록이 x̄ 에서 평가값 v 에 못 미치면 μ 를 키우고 (블록 교체), 그래도 못 미치면 안전장치.

인자 (기본값, 뜻):
  | 인자          | 기본      | 뜻 |
  |---------------|-----------|----|
  | time_limit    | 3600.0    | 전체 시간 (s) |
  | tol           | 5e-3      | 상대 허용오차 (UB − LB ≤ tol·max(1,|UB|)) |
  | mu            | 10.0      | 블록 벌점 μ (리더 비용 / follower 비용 단위). loc_tkp 의 DRO belief 에서 1/η 수준이 필요 (문서 5b) |
  | eta           | 0.1       | 동률 섭동 η. 안전한 값 η̄ 는 계산하지 않음 (원고 Prop belief blocks, 문서 "비관 값의 수치 조건") |
  | mu_max        | 1e4       | x̄ 확보 때 μ 를 키우는 상한 |
  | safeguard     | :nogood   | :nogood | :restricted (Ω 의 α 고정 LP cut, λᵁ ≥ λ̄ 가정) | :both |
  | master_gap    | 1e-3      | 인증용 마스터의 MIPGap |
  | obj_cut       | false     | 마스터에 목적값 하한 q_xᵀx + t0 ≥ LB (지금까지의 유효 하한, 이전 마스터의 objective_bound). 마스터 값이 LB 에서 gap 안으로만
  |               |           | 움직이면 incumbent 하나로 나머지 나무를 쳐냄 (Tsang–Shehadeh–Curtis 의 inexact C&CG, 식 (4c)). 정확 모드에서도 유효.
  |               |           | 2026-11-21 후보지 7 곳: 마스터 1,089 s (끔 805~868 s) → 기본 끔 |
  | early_gap     | 0.0       | > 0 이면 부정확 마스터 (i-C&CG, Tsang–Shehadeh–Curtis): 마스터를 이 MIPGap 으로 풀고 그 incumbent 에서 탐색·oracle. |
  |               |           | incumbent 값이 UB 의 0.75·tol 안이거나 oracle 이 진전을 못 주면 gap 을 gap_shrink 배로 줄임 (master_gap 까지). obj_cut 과 함께 |
  | gap_shrink    | 0.2       | 위의 gap 축소 비율 |
  | wd            | false     | 블록 약쌍대: follower 섭동 비용 (사본) + 쌍대 목적 ≥ 0 (정수 x 에서 이미 성립). 분수 x 의 McCormick 횡재를 막음 |
  | pool          | 1         | > 1 이면 마스터 해 풀의 앞 pool 개 x 에서도 belief 탐색 |
  | nrand         | 40        | 싼 탐색의 무작위 후보 수 |
  | vub           | false     | 입지 인스턴스 (meta 에 :iyR) 의 호 단위 제약 y^R+y^S ≤ ξ Σ_k x (정수 x 에서 중복). 2026-11-21 후보지 7 곳: 켜면 노드는 같고
  |               |           | 블록이 25% 커져 마스터 한 번이 10~25% 느림 (전체 X 에서는 5~10%) → 기본 끔 |
  | threads       | nthreads  | 마스터 스레드, 탐색 평가기 수, α-B&B worker = threads − 1 |
실행: julia -t T,1 (T = threads). 결과 Dict 의 :impl = "nz_belief_ccg". 2026-10-22 의 KKT 기반 벤치마크 CCG (nz_ccg.jl, 함수 nz_ccg) 와는 다른 방법.
"""

using JuMP, Gurobi, Printf, LinearAlgebra, Random
import MathOptInterface as MOI
isdefined(Main, :nz_alpha_bnb) || include(joinpath(@__DIR__, "nz_alpha_bnb.jl"))
isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(@__DIR__, "nz_alpha_bnb_dr.jl"))

# ---------------------------------------------------------------------------------------------------------------
# 싼 평가기 (시제품 bgen_core.jl 의 BEv): g(x, d) = [belief d 에서 follower 최적 반응 α] 의 정확 평가 ≤ V*(x)
# ---------------------------------------------------------------------------------------------------------------
mutable struct BCCGEval
    R::Model; Rα; Ry; Rcon; curd::Vector{Float64}
    MQ::Model; MQcon; Mφ::Model; Mφcon; MφcQ
    E::Model; Er; Ed; Eπ; Eκ; EFO
    lastr::Vector{Float64}
    gx::Matrix{Float64}
    xrows::Vector{Int}
end

function BCCGEval(nd::NZData, env)
    mk = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
    m, ny, nh, S = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.S; nW = size(nd.W, 1); A = nd.A
    R = Model(mk); @variable(R, 0 <= α[i=1:nh] <= nd.hU[i]); @constraint(R, nd.W * α .<= nd.wvec); @variable(R, y[1:ny, 1:S] >= 0)
    @constraint(R, con[k=1:m, s=1:S], sum(A[k, j] * y[j, s] for j in 1:ny if A[k, j] != 0) - (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0) <= nd.u[k, s])
    @objective(R, Max, dot(nd.c0, α) + sum(0.0 * y[j, s] for j in 1:ny, s in 1:S))
    MQ = Model(mk); @variable(MQ, yq[1:ny] >= 0); @constraint(MQ, cq[k=1:m], sum(A[k, j] * yq[j] for j in 1:ny if A[k, j] != 0) <= 0.0); @objective(MQ, Max, dot(nd.c, yq))
    Mφ = Model(mk); @variable(Mφ, yp[1:ny] >= 0); @constraint(Mφ, cp[k=1:m], sum(A[k, j] * yp[j] for j in 1:ny if A[k, j] != 0) <= 0.0)
    cQ = @constraint(Mφ, dot(nd.c, yp) >= -1e20); @objective(Mφ, Max, dot(nd.ell, yp))
    q, ε̂, ε̃, β = nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta
    E = Model(mk)
    @variable(E, 0 <= a[1:S] <= 1); @variable(E, b[1:S] >= 0); @variable(E, 0 <= r[1:S] <= 1)
    @constraint(E, sum(a) == 1); @constraint(E, sum(r) == 1)
    @constraint(E, [s = 1:S], a[s] - b[s] <= q[s]); @constraint(E, [s = 1:S], a[s] + b[s] >= q[s])
    @constraint(E, sum(b) <= 2ε̂); @constraint(E, [s = 1:S], (1 - β) * r[s] <= a[s])
    @variable(E, 0 <= d[1:S] <= 1); @variable(E, e[1:S] >= 0); @constraint(E, sum(d) == 1)
    @constraint(E, [s = 1:S], d[s] - e[s] <= q[s]); @constraint(E, [s = 1:S], d[s] + e[s] >= q[s]); @constraint(E, sum(e) <= 2ε̃)
    nz_add_box!(E, a, d, nd)
    if nz_coupled(nd)
        @variable(E, cpl[1:S] >= 0)
        @constraint(E, [s = 1:S], a[s] - d[s] <= cpl[s]); @constraint(E, [s = 1:S], d[s] - a[s] <= cpl[s]); @constraint(E, sum(cpl) <= 2nz_delta(nd))
    end
    @variable(E, π̃[1:m, 1:S] >= 0); @variable(E, κ[1:nW] >= 0)
    @constraint(E, [j = 1:ny, s = 1:S], sum(A[k, j] * π̃[k, s] for k in 1:m if A[k, j] != 0) >= nd.c[j] * d[s])
    @constraint(E, [i = 1:nh], sum(nd.hcoef[k] * π̃[k, s] for k in 1:m, s in 1:S if nd.hrow[k] == i) + nd.c0[i] <= sum(nd.W[l, i] * κ[l] for l in 1:nW))
    FO = @constraint(E, sum(0.0 * d[s] for s in 1:S) - sum(nd.u[k, s] * π̃[k, s] for k in 1:m, s in 1:S) - dot(nd.wvec, κ) >= 0.0)
    @objective(E, Max, sum(0.0 * r[s] for s in 1:S))
    return BCCGEval(R, α, y, con, fill(NaN, S), MQ, cq, Mφ, cp, cQ, E, r, d, π̃, κ, FO, copy(nd.q_hat), zeros(m, S),
                   unique(e.k for e in nz_xentries(nd)))
end

function _bccg_setx!(ev::BCCGEval, nd::NZData, x̄)
    for k in ev.xrows, s in 1:nd.S
        gv = nz_gx(nd, k, s, x̄)
        if gv != ev.gx[k, s]
            ev.gx[k, s] = gv
            set_normalized_rhs(ev.Rcon[k, s], nd.u[k, s] + gv)
            set_normalized_coefficient(ev.EFO, ev.Eπ[k, s], -(nd.u[k, s] + gv))
        end
    end
end

function _bccg_resp!(ev::BCCGEval, nd::NZData, d)
    if d != ev.curd
        for j in 1:nz_ny(nd), s in 1:nd.S
            nd.c[j] != 0 && set_objective_coefficient(ev.R, ev.Ry[j, s], d[s] * nd.c[j])
        end
        ev.curd = copy(d)
    end
    optimize!(ev.R)
    termination_status(ev.R) == MOI.OPTIMAL || return nothing
    return [max(value(ev.Rα[i]), 0.0) for i in 1:nz_nh(nd)]
end

"""
follower 최적 행의 여유 κ (상대, 척도 1 + |c0ᵀα| + max|Q|). 작은 값부터 풀어 E 가 OPTIMAL 인 첫 값을 쓴다 (2026-11-25).
여유가 블록의 동률 깨기 η·ΔC_L 보다 크면 E 가 동률 반대편 belief (α 가 여유만큼 비최적인 belief) 를 돌려주고, 그 belief 의 블록은
정확해도 평가값 v 에 못 미쳐 확보 규칙이 no-good 을 넣는다. 예전 고정값 1e-7 은 절대 약 0.12 라 η = 1e-4 블록 (0.005) 이 못 덮었다
(η = 1e-4 판에서 belief 추가 56 번 중 35 번이 no-good, η = 0.1 판은 36 번 중 1 번). κ 를 1e-7 → 0 으로 줄여도 v 는 0.01 만 바뀜.
α 가 follower LP 의 해가 아닐 때 (oracle incumbent) 는 κ = 0 이 실행 불가능할 수 있어 단계적으로 늘린다. docs/nz_ccg_bnp_2026-11-23.md 11.10.
"""
const _BCCG_KAPPAS = (0.0, 1e-11, 1e-9, 1e-7)

"반응 α 의 정확 평가: (값, follower belief d). α 가 어떤 belief 에서도 최적이 아니면 (−Inf, nothing). 재가중 r 은 ev.lastr. 여유는 `_BCCG_KAPPAS` 순서"
function _bccg_value!(ev::BCCGEval, nd::NZData, α)
    S, m = nd.S, nz_m(nd); Q = zeros(S); φ = zeros(S)
    for s in 1:S
        for k in 1:m
            rr = nd.u[k, s] + ev.gx[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0)
            set_normalized_rhs(ev.MQcon[k], rr); set_normalized_rhs(ev.Mφcon[k], rr)
        end
        optimize!(ev.MQ); termination_status(ev.MQ) == MOI.OPTIMAL || return -Inf, nothing
        Q[s] = objective_value(ev.MQ)
        set_normalized_rhs(ev.MφcQ, Q[s] - 1e-9 * max(1.0, abs(Q[s])))
        optimize!(ev.Mφ); termination_status(ev.Mφ) == MOI.OPTIMAL || return -Inf, nothing
        φ[s] = objective_value(ev.Mφ)
    end
    lhs = dot(nd.c0, α)
    for s in 1:S
        set_normalized_coefficient(ev.EFO, ev.Ed[s], Q[s]); set_objective_coefficient(ev.E, ev.Er[s], φ[s])
    end
    sc = 1 + abs(lhs) + maximum(abs, Q)
    for κ in _BCCG_KAPPAS
        set_normalized_rhs(ev.EFO, -lhs - κ * sc); optimize!(ev.E)
        termination_status(ev.E) == MOI.OPTIMAL && break
    end
    termination_status(ev.E) == MOI.OPTIMAL || return -Inf, nothing
    ev.lastr = [value(ev.Er[s]) for s in 1:S]
    return objective_value(ev.E), [value(ev.Ed[s]) for s in 1:S]
end

function _bccg_g!(ev::BCCGEval, nd::NZData, d)
    α = _bccg_resp!(ev, nd, d); α === nothing && return -Inf, nothing, nothing, nothing
    v, dd = _bccg_value!(ev, nd, α)
    return v, dd, copy(ev.lastr), α
end

# ---------------------------------------------------------------------------------------------------------------
# 블록 (마스터 안) 과 고정 x̄ 에서의 블록 값
# ---------------------------------------------------------------------------------------------------------------
"섭동된 follower 목적의 계수 (최대화형): d_s c_j + η r_s ℓ_j (d_s > 0 인 시나리오만 섭동)"
_bccg_ceta(nd, d, r, η, j, s) = d[s] * nd.c[j] + (d[s] > 0 ? η * r[s] * nd.ell[j] : 0.0)

"belief (d, r) 의 블록을 마스터에 추가. 반환 = 지울 때 쓸 (변수, 제약) 목록"
function _bccg_add_block!(C, d, r, μ)
    nd, M, x, t0, η = C.nd, C.M, C.x, C.t0, C.eta
    m, ny, nh, S = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.S; nW = size(nd.W, 1)
    vars = VariableRef[]; cons = ConstraintRef[]
    h = @variable(M, [i=1:nh], lower_bound = 0.0, upper_bound = nd.hU[i]); append!(vars, h)
    append!(cons, @constraint(M, nd.W * h .<= nd.wvec))
    y = @variable(M, [1:ny, 1:S], lower_bound = 0.0); append!(vars, vec(y))
    for k in 1:m, s in 1:S
        push!(cons, @constraint(M, sum(a * y[j, s] for (j, a) in C.rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0)
                                   - sum(e.g[s] * x[e.i] for e in C.ents_of_row[k]; init=0.0) <= nd.u[k, s]))
    end
    if C.vub
        MMv = nd.meta
        for st in 1:(MMv[:nst] - MMv[:nA]), j in 1:MMv[:ncu], s in 1:S
            push!(cons, @constraint(M, y[MMv[:iyR](MMv[:nA] + st, j), s] + y[MMv[:iyS](MMv[:nA] + st, j), s] <=
                                        -nd.u[MMv[:rdem](j), s] * sum(x[(st - 1) * MMv[:K] + k] for k in 1:MMv[:K])))
        end
    end
    CF = -(dot(nd.c0, h) + sum(d[s] * nd.c[j] * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0 && d[s] != 0; init=AffExpr(0.0)))
    CL = sum(r[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0; init=AffExpr(0.0))
    CLp = sum(r[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0 && d[s] > 0; init=AffExpr(0.0))
    pd = @variable(M, [1:m, 1:S], lower_bound = 0.0); kp = @variable(M, [1:nW], lower_bound = 0.0); om = @variable(M, [1:nh], lower_bound = 0.0)
    append!(vars, vec(pd)); append!(vars, kp); append!(vars, om)
    for j in 1:ny, s in 1:S
        push!(cons, @constraint(M, sum(a * pd[k, s] for (k, a) in C.colA[j]; init=AffExpr(0.0)) >= _bccg_ceta(nd, d, r, η, j, s)))
    end
    for i in 1:nh
        push!(cons, @constraint(M, -sum(nd.hcoef[k] * pd[k, s] for k in 1:m, s in 1:S if nd.hrow[k] == i; init=AffExpr(0.0)) +
                                   sum(nd.W[l, i] * kp[l] for l in 1:nW; init=AffExpr(0.0)) + om[i] >= nd.c0[i]))
    end
    dobj = AffExpr(0.0)
    add_to_expression!(dobj, sum(nd.u[k, s] * pd[k, s] for k in 1:m, s in 1:S if nd.u[k, s] != 0; init=AffExpr(0.0)) + dot(nd.wvec, kp) + dot(nd.hU, om))
    maxℓ = maximum(abs, nd.ell)
    for e in C.XE, s in 1:S
        U = d[s] * get(C, :UF, nd.piFU)[e.k] + (d[s] > 0 ? η * r[s] * maxℓ : 0.0)
        set_upper_bound(pd[e.k, s], U)
        ps = @variable(M, lower_bound = 0.0); push!(vars, ps)
        push!(cons, @constraint(M, ps >= pd[e.k, s] - U * (1 - x[e.i])))
        add_to_expression!(dobj, e.g[s], ps)
    end
    # t0 ≥ C_L + μ (C_F − η C_L⁺) + μ·(쌍대 목적 ≥ −V_F^η)
    push!(cons, @constraint(M, t0 >= μ * (CF - η * CLp) + CL + μ * dobj))
    # 블록 약쌍대 (wd): follower 섭동 비용 (사본) + 쌍대 목적 ≥ 0. 정수 x 에서는 약쌍대로 이미 성립 (McCormick 이 정확) 하므로 유효하고,
    # 분수 x 에서 사본은 부분 용량을 쓰는데 쌍대는 그 용량을 쳐주지 않아 간격이 크게 음수가 되는 것 (횡재) 을 막는다.
    get(C, :wd, false) && push!(cons, @constraint(M, CF - η * CLp + dobj >= 0))
    return (vars = vars, cons = cons, d = copy(d), r = copy(r), μ = μ, h = h, y = y, pd = pd)
end

function _bccg_delete_block!(C, blk)
    for c in blk.cons; is_valid(C.M, c) && delete(C.M, c); end
    delete(C.M, filter(v -> is_valid(C.M, v), blk.vars))
end

"""고정 x̄ 에서 블록 값: (L_b(x̄) at μ, 극한값 L∞ = min{C_L : 섭동 최적 plan}, 필요한 μ*). LP 두 개."""
function _bccg_block_at(C, x̄, d, r, μ)
    nd, η = C.nd, C.eta
    m, ny, nh, S = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.S
    P = Model(C.lp_opt)
    @variable(P, 0 <= h[i=1:nh] <= nd.hU[i]); @constraint(P, nd.W * h .<= nd.wvec); @variable(P, y[1:ny, 1:S] >= 0)
    @constraint(P, [k=1:m, s=1:S], sum(a * y[j, s] for (j, a) in C.rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0) <= nd.u[k, s] + nz_gx(nd, k, s, x̄))
    fη = @expression(P, dot(nd.c0, h) + sum(_bccg_ceta(nd, d, r, η, j, s) * y[j, s] for j in 1:ny, s in 1:S))   # 최대화형 (= −(C_F − ηC_L⁺))
    CL = @expression(P, sum(r[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0; init=AffExpr(0.0)))
    @objective(P, Max, fη); optimize!(P)
    termination_status(P) == MOI.OPTIMAL || error("nz_belief_ccg: 블록 검사 LP (follower) $(termination_status(P))")
    zη = objective_value(P)
    @objective(P, Min, CL + μ * (zη - fη)); optimize!(P)
    termination_status(P) == MOI.OPTIMAL || error("nz_belief_ccg: 블록 검사 LP (μ) $(termination_status(P))")
    Lμ = objective_value(P)
    c = @constraint(P, fη >= zη - 1e-9 * max(1.0, abs(zη)))
    @objective(P, Min, CL); optimize!(P)
    termination_status(P) == MOI.OPTIMAL || error("nz_belief_ccg: 블록 검사 LP (극한) $(termination_status(P))")
    return Lμ, objective_value(P), abs(dual(c))
end

# ---------------------------------------------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------------------------------------------
function nz_belief_ccg(nd::NZData; time_limit=3600.0, tol=5e-3, mu=10.0, eta=0.1, mu_max=1e4, safeguard::Symbol=:nogood,
                master_gap=1e-3, early_gap=0.0, gap_shrink=0.2, obj_cut::Bool=false, wd::Bool=false, pool::Int=1, nrand::Int=40, vub::Bool=false,
                threads::Int=Threads.nthreads(), seed::Int=2026, verbose::Bool=true)
    safeguard in (:nogood, :restricted, :both) || error("nz_belief_ccg: safeguard = :nogood | :restricted | :both")
    wall = time()
    m, ny, nh, nx, S = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.nx, nd.S; A = nd.A; XE = nz_xentries(nd)
    envs = _nz_envs(threads + 1)
    lp_opt = () -> (o = Gurobi.Optimizer(envs[end]); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
    pool_ev = Channel{BCCGEval}(threads); for k in 1:threads; put!(pool_ev, BCCGEval(nd, envs[k])); end
    M = Model(() -> (o = Gurobi.Optimizer(envs[1]); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), threads); o))
    set_optimizer_attribute(M, "IntFeasTol", 1e-9); set_optimizer_attribute(M, "IntegralityFocus", 1)
    pool > 1 && (set_optimizer_attribute(M, "PoolSolutions", pool))
    @variable(M, x[1:nx], Bin); @variable(M, t0 >= -1e7)
    @constraint(M, sum(x) <= nd.gamma)
    for (idx, rhs) in get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[]); @constraint(M, sum(x[i] for i in idx) <= rhs); end
    for i in 1:nx; nd.x_allowed[i] || fix(x[i], 0.0; force=true); end
    @objective(M, Min, dot(nd.qx, x) + t0)
    C = (nd = nd, M = M, x = x, t0 = t0, eta = Float64(eta), vub = vub, wd = wd, XE = XE, lp_opt = lp_opt,
         rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m], colA = [[(k, A[k, j]) for k in 1:m if A[k, j] != 0] for j in 1:ny],
         ents_of_row = [[e for e in XE if e.k == k] for k in 1:m])
    verbose && @printf("  [설정] nz_belief_ccg: μ %.3g, η %.3g, μ 상한 %.3g, 안전장치 %s, 마스터 gap %.0e, 부정확 마스터 gap %s, 목적값 하한 %s, 블록 약쌍대 %s, 해 풀 %d, 무작위 후보 %d, vub %s, 스레드 %d\n",
                       mu, eta, mu_max, safeguard, master_gap, early_gap > 0 ? @sprintf("%.0e", early_gap) : "끔", obj_cut ? "켬" : "끔", wd ? "켬" : "끔", pool, nrand, vub, threads); flush(stdout)

    # belief 후보: 명목, 두 성분 사이 질량 이동, 무작위
    qh = nd.q_hat; bx = nd.meta[:box]
    inset(d) = all(d .>= bx.dl .- 1e-12) && all(d .<= bx.du .+ 1e-12) && sum(abs.(d .- qh)) <= 2nd.eps_tilde + 1e-12
    pairbels = Vector{Vector{Float64}}()
    for i in 1:S, j in 1:S
        i == j && continue
        mv = min(qh[i] - bx.dl[i], bx.du[j] - qh[j], nd.eps_tilde); mv <= 1e-9 && continue
        dd = copy(qh); dd[i] -= mv; dd[j] += mv; inset(dd) && push!(pairbels, dd)
    end
    rng = MersenneTwister(seed)
    function randbel()
        d = copy(qh)
        for t in 1:rand(rng, 1:4)
            i, j = randperm(rng, S)[1:2]; mv = rand(rng) * min(d[i] - bx.dl[i], bx.du[j] - d[j]); dd = copy(d); dd[i] -= mv; dd[j] += mv; inset(dd) && (d = dd)
        end
        return d
    end

    blocks = Any[]; Dd = [copy(qh)]
    push!(blocks, _bccg_add_block!(C, qh, qh, mu))
    # no-good 의 t 하한: 명목 블록만 둔 LP 완화의 min t0 (≤ min_x L_b0(x) ≤ min_x V*(x))
    undo = relax_integrality(M); @objective(M, Min, t0); optimize!(M)
    termination_status(M) == MOI.OPTIMAL || error("nz_belief_ccg: t 하한 LP $(termination_status(M))")
    tlow = objective_value(M) - 1e-6 * max(1.0, abs(objective_value(M))); undo(); @objective(M, Min, dot(nd.qx, x) + t0)
    objcon = @constraint(M, dot(nd.qx, x) + t0 >= -1e9)      # 목적값 하한 (obj_cut). 마스터 값은 반복마다 줄지 않으므로 지금까지의 LB 가 유효

    function search(xb, t0v)
        cand = vcat(Dd, pairbels, [randbel() for _ in 1:nrand]); vals = fill(-Inf, length(cand))
        Threads.@threads for t in eachindex(cand)
            local ev = take!(pool_ev)
            try
                _bccg_setx!(ev, nd, xb); local v1, _, _, _ = _bccg_g!(ev, nd, cand[t]); vals[t] = v1
            finally
                put!(pool_ev, ev)
            end
        end
        tb = argmax(vals); best = (v = -Inf, d = nothing, r = nothing, α = nothing)
        ev2 = take!(pool_ev)
        try
            _bccg_setx!(ev2, nd, xb); dcur = cand[tb]
            for it in 1:4
                v2, dd, r2, α2 = _bccg_g!(ev2, nd, dcur)
                dd === nothing && break
                v2 > best.v + 1e-9 || break
                best = (v = v2, d = dd, r = r2, α = α2)
                inset(dd) || break
                dcur = dd
            end
        finally
            put!(pool_ev, ev2)
        end
        return best
    end
    function exact_value(xb, α)
        ev3 = take!(pool_ev)
        try
            _bccg_setx!(ev3, nd, xb)
            a1 = [(v = clamp(α[i], 0.0, nd.hU[i]); v < 1e-6 * max(1.0, nd.hU[i]) ? 0.0 : v) for i in eachindex(α)]
            v3, dd = _bccg_value!(ev3, nd, a1)
            if dd === nothing
                a2 = nz_feasible_alpha(nd, xb, a1; lp_optimizer=lp_opt, tol=1e-3)
                a1 = a2; v3, dd = _bccg_value!(ev3, nd, a2)
            end
            return v3, dd, (dd === nothing ? nothing : copy(ev3.lastr)), a1
        finally
            put!(pool_ev, ev3)
        end
    end
    n_ng = 0; n_rc = 0; n_mu = 0
    function nogood!(xb, v)
        Δ = sum((1 - x[i]) for i in 1:nx if xb[i] > 0.5; init=AffExpr(0.0)) + sum(x[i] for i in 1:nx if xb[i] < 0.5 && nd.x_allowed[i]; init=AffExpr(0.0))
        @constraint(M, t0 >= v - (v - tlow) * Δ); n_ng += 1
    end
    function restricted!(xb, α)
        F = nz_build_fixed_alpha(nd, xb; optimizer=lp_opt)
        z = nz_eval_alpha!(F, nd, α)
        isfinite(z) || return false
        cut = nz_cut_from_solution(F, nd, xb)
        @constraint(M, t0 >= cut[:intercept] + dot(cut[:slope], x)); n_rc += 1
        return true
    end
    "새 belief 를 블록으로 넣고 x̄ 를 확보 (원고 Algorithm 2 의 10–11 줄)"
    function add_and_secure!(xb, v, d, r, α)
        push!(Dd, d); blk = _bccg_add_block!(C, d, r, mu); push!(blocks, blk)
        thr = 1e-4 * max(1.0, abs(v))
        Lμ, Linf, μstar = _bccg_block_at(C, xb, d, r, mu)
        Lμ >= v - thr && return :block
        if Linf >= v - thr && μstar <= mu_max
            μn = min(mu_max, max(2mu, 1.1μstar)); _bccg_delete_block!(C, blk); blocks[end] = _bccg_add_block!(C, d, r, μn); n_mu += 1
            Lμ2, _, _ = _bccg_block_at(C, xb, d, r, μn)
            Lμ2 >= v - thr && return :mu
        end
        safeguard in (:nogood, :both) && nogood!(xb, v)
        safeguard in (:restricted, :both) && (restricted!(xb, α) || safeguard == :both || nogood!(xb, v))
        return :safeguard
    end

    UB = Inf; LB = -Inf; xbest = nothing; status = :TimeLimit; iter = 0; t_ms = 0.0; t_or = 0.0; n_or = 0
    # 마스터 gap ε_MP (i-C&CG): early_gap 에서 시작해, incumbent 값이 UB 에 붙거나 oracle 이 진전을 못 주면 gap_shrink 배로 줄임 (master_gap 까지)
    eps = early_gap > 0 ? max(early_gap, master_gap) : master_gap
    seen = Dict{Vector{Float64},Int}()
    while time() - wall < time_limit
        iter += 1
        set_optimizer_attribute(M, "MIPGap", eps)
        (obj_cut && isfinite(LB)) && set_normalized_rhs(objcon, LB - 1e-6 * max(1.0, abs(LB)))
        set_time_limit_sec(M, max(10.0, time_limit - (time() - wall)))
        tm = @elapsed optimize!(M); t_ms += tm
        termination_status(M) in (MOI.OPTIMAL, MOI.TIME_LIMIT) || error("nz_belief_ccg: 마스터 $(termination_status(M))")
        has_values(M) || break
        LB = max(LB, objective_bound(M))
        xb = Float64.(value.(x) .> 0.5); t0v = objective_value(M) - dot(nd.qx, xb)
        if isfinite(UB) && UB - LB <= tol * max(1.0, abs(UB)); status = :Optimal; break; end
        if eps > master_gap && isfinite(UB) && UB - objective_value(M) <= 0.75 * tol * max(1.0, abs(UB))   # i-C&CG 의 exploitation
            eps = max(master_gap, gap_shrink * eps)
            verbose && (@printf("  [mstr] it %3d incumbent 가 UB 에 붙음 → 마스터 gap %.1e 로 다시 (마스터 %.1fs)\n", iter, eps, tm); flush(stdout))
            continue
        end
        nodes = try Int(MOI.get(M, MOI.NodeCount())) catch; -1 end
        if isfinite(UB) && UB - LB <= tol * max(1.0, abs(UB)); status = :Optimal; break; end
        # 탐색: incumbent (와 해 풀) 에서
        pts = [(xb, t0v)]
        if pool > 1
            for k in 2:min(pool, result_count(M))
                xk = Float64.(value.(x; result=k) .> 0.5); tk = value(t0; result=k)
                any(p -> p[1] == xk, pts) || push!(pts, (xk, tk))
            end
        end
        (verbose && pool > 1) && (@printf("  [dbg] 마스터 %.1fs, 해 풀 %d 개 (결과 %d)
", tm, length(pts), result_count(M)); flush(stdout))
        added = 0; how = :none
        for (k, (xk, tk)) in enumerate(pts)
            tsr = @elapsed (b = search(xk, tk))
            (verbose && pool > 1) && (@printf("  [dbg]   점 %d %s 탐색 %.1fs v=%.2f t=%.2f
", k, loc_tkp_xstr(nd, xk), tsr, b.v, tk); flush(stdout))
            (b.d !== nothing && b.v > tk + 1e-3 * max(1.0, abs(b.v)) && get(seen, xk, 0) < 3) || continue
            seen[xk] = get(seen, xk, 0) + 1
            if k == 1
                how = add_and_secure!(xk, b.v, b.d, b.r, b.α)
            else
                push!(Dd, b.d); push!(blocks, _bccg_add_block!(C, b.d, b.r, mu))
            end
            added += 1
        end
        if added > 0
            verbose && @printf("  [srch] it %3d LB=%10.2f UB=%10.2f x=%-24s t0 %.2f, belief +%d (%s), 마스터 %.1fs 노드 %d %s, beliefs=%d\n",
                               iter, LB, UB, loc_tkp_xstr(nd, xb), t0v, added, how, tm, nodes, eps > master_gap ? @sprintf("(gap %.1e)", eps) : "", length(blocks)); flush(stdout)
            continue
        end
        # 전역 oracle
        tau = t0v + 0.9 * tol * max(1.0, abs(LB))
        gap_o = min(1e-3, 0.4 * tol * max(1.0, abs(LB)) / max(1.0, abs(t0v)))
        lim = max(10.0, time_limit - (time() - wall))
        to = @elapsed (res = nz_alpha_bnb_dr(nd, xb; nworkers=max(1, threads - 1), time_limit=lim, rel_gap=gap_o, target=tau, verbose=false))
        t_or += to; n_or += 1
        if dot(nd.qx, xb) + res[:UB] < UB; UB = dot(nd.qx, xb) + res[:UB]; xbest = copy(xb); end
        how = :none
        if res[:α] !== nothing && res[:LB] > t0v + 1e-6 * max(1.0, abs(t0v))
            vv, dd, r2, a1 = exact_value(xb, res[:α])
            if dd !== nothing
                how = add_and_secure!(xb, vv, dd, r2, a1)
                if vv <= t0v + 1e-6 * max(1.0, abs(t0v)) && safeguard == :nogood
                    restricted!(xb, a1)          # 정확 평가가 진전을 못 주면 (λᵁ ≥ λ̄ 아래의) restricted ISP cut 로 진전 보장
                end
            end
        end
        # oracle 이 t0 보다 큰 하한을 증명했는데 belief 를 못 만든 경우 (α 의 정확 평가 실패, 2026-11-22 지리 8·13 에서 Stalled):
        # 증명된 하한 v = res[:LB] ≤ V*(x̄) 로 no-good cut (유효, x̄ 에서 진전 보장)
        if how == :none && res[:LB] > t0v + 1e-6 * max(1.0, abs(t0v))
            nogood!(xb, res[:LB]); how = :nogood_lb
        end
        verbose && @printf("  [orcl] it %3d LB=%10.2f UB=%10.2f x=%-24s V in [%.2f, %.2f], t0 %.2f (%.0fs, 마스터 %.1fs 노드 %d) %s, beliefs=%d\n",
                           iter, LB, UB, loc_tkp_xstr(nd, xb), res[:LB], res[:UB], t0v, to, tm, nodes, how, length(blocks)); flush(stdout)
        if isfinite(UB) && UB - LB <= tol * max(1.0, abs(UB)); status = :Optimal; break; end
        if how == :none
            time() - wall >= time_limit - 1.0 && break       # oracle 이 남은 시간이 없어 끝난 경우 = TimeLimit (Stalled 아님)
            if eps > master_gap                    # 부정확 마스터의 해라 진전이 없을 수 있음 → gap 을 줄여 다시
                eps = max(master_gap, gap_shrink * eps); continue
            end
            status = :Stalled; verbose && println("  [!] oracle 이 t0 보다 큰 하한도 새 belief 도 못 줌"); break
        end
    end
    return Dict(:impl => "nz_belief_ccg", :status => status, :x => xbest, :UB => UB, :LB => LB, :gap => abs(UB - LB) / max(1.0, abs(UB)),
                :iters => iter, :beliefs => length(blocks), :master_time => t_ms, :oracle_calls => n_or, :oracle_time => t_or,
                :nogood => n_ng, :restricted => n_rc, :mu_raises => n_mu, :time => time() - wall,
                :args => (mu = mu, eta = eta, mu_max = mu_max, safeguard = safeguard, master_gap = master_gap, early_gap = early_gap, gap_shrink = gap_shrink, obj_cut = obj_cut, wd = wd, pool = pool, nrand = nrand, vub = vub))
end
