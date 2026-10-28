"""
lab_vf_form.jl — 고정 x̄ 의 oracle 을 범용 전역 솔버 (Gurobi) 에 맡길 때, 정식화에 따라 상한이 얼마나 달라지는지 본다 (시제품, 기존 파일 미수정).

  VF_FORM = omega : 지금의 Ω (build_nz_omega mode = :global). 곱 = α×r, α×d (n_h·S 개씩), α×ϖ (H 행·S 개). θ 벌점 있음.
          = vf    : 값함수 정식화. α 와의 곱, ϖ, θ 벌점, big-M 이 없다. 곱은 d_s × t_s 와 r_s × φ_s 의 2S 개.

      max  Σ_s r_s φ_s
      s.t. (a, r, d) ∈ 𝔅_δ,  α ∈ H,  y_s ∈ Y_s(x̄, α),  t_s = cᵀy_s,  φ_s = ℓᵀy_s
           c0ᵀα + Σ_s d_s t_s  ≥  Σ_s b_sᵀπ̃_s + wᵀκ          (follower 2 단계 LP 의 강쌍대: 우변은 V_F(d) 의 쌍대, d 에 선형)
           Aᵀπ̃_s ≥ d_s c,   Σ_s Hᵀπ̃_s + c0 ≤ Wᵀκ,   π̃, κ ≥ 0

    근거: follower 의 2 단계 LP 는 실행가능 집합이 d 와 무관하고 d 는 목적에만 들어간다. 따라서 쌍대 LP 의 실행가능 집합이 (π̃, κ, d) 에 선형이고
    강쌍대 한 줄이 α ∈ Ψ(x̄, d) 와 같다. d_s > 0 이면 follower 최적인 (α, y) 의 y_s 는 recourse 최적이므로 리더 복사본을 따로 둘 필요가 없다
    (y_s 가 recourse 최적해 집합 전체를 움직이고 목적이 그 위에서 ℓᵀy 를 최대화 = pessimistic 선택).
    d_s = 0 이 가능하면 (q̂_s ≤ ε̃) 그 시나리오의 recourse 최적성이 빠져 상한이 느슨해진다 → VF_REC = lattice.

  VF_REC = none (기본) | lattice : 시나리오별 recourse 강쌍대 t_s ≥ b_sᵀσ_s + Σ_{k∈H} α_{i(k)} σ_ks 를 더한다. A 가 TU 이고 비용이 1/D 의 배수라
          recourse 쌍대 꼭짓점의 H 행 성분은 {0, 1/D, …, p} 의 격자점 → σ_ks 를 이진 전개하면 α×σ 가 (이진 × 연속) 으로 정확히 선형화된다.
  VF_BITS = 0 (기본: 곱을 Gurobi 의 bilinear 로) | p ≥ 1 : belief 쪽 변수 (d_s, r_s) 를 p 자리 이진 전개 (NMDT, Castro 2016). 남는 곱은 폭 2^-p 의 나머지 × t_s
          뿐이라 McCormick 오차가 2^-p 배로 준다. 공간 분기를 MIP 분기로 바꾸는 기법 — 분기 차원이 n_h 가 아니라 2S.

환경변수: NZ_SEED = 1  VF_S = 3  VF_X = "1"  VF_NLOC = 15  VF_INST = random | sgb  VF_TL = 300  VF_THREADS = 12  VF_GAP = 1e-3
          VF_RES = pair | pooled (예약 구조. pooled = 점포별 예약 풀. θᵁ 는 둘 다 정확한 circuit 값)
          VF_POOL = total | store (pooled 의 한도: 전체 1ᵀh ≤ 300, 또는 거점별 h_i ≤ VF_QUOTA = 300)
          VF_PRE = resp (기본: 닫힌 점포 + 예약 상한 ≤ 최대 수요 + 반응 정의역 프리솔브) | basic (앞의 둘만)   VF_START = 1 (belief 표본의 최적 반응을 시작해로)
          VF_RLT = 0 | 1 (vf 전용: 확률 합 Σ_s d_s = 1, Σ_s r_s = 1 에 t_s', φ_s' 를 곱한 RLT 등식 — pooling 의 pq 정식화와 같은 종류. 교차 곱 d_s t_s', r_s φ_s' 를
                  변수로 두므로 곱이 2S² 개로 는다)
          VF_GRB = "" (Gurobi 인자, 예 "OBBT=3,RLTCuts=2")
          VF_WRITE = "" (vf 전용: 모델을 파일로 내보낸다. 확장자가 형식을 정한다 — .nl (AMPL NL: BARON·SCIP·Couenne 가 읽음), .mof.json, .lp)
          VF_DMIN = 0 (vf 전용 진단: follower belief 에 하한 d_s ≥ VF_DMIN 을 둔 **다른 문제**. belief 집합을 줄였으므로 값은 V* 이하이고 VF_REC = none 이 정확해진다.
                    "d_s = 0 이 가능해서 생기는 어려움" 과 "S 가 커서 생기는 어려움" 을 가르는 용도. 시작해는 쓰지 않는다)
실행: julia -t 2 nonzero_sum/lab_vf_form.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics
const GRB_ENV = Gurobi.Env()
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl")); include(joinpath(root, "nonzero_sum", "nz_root_tighten.jl"))
envf(k, d) = get(ENV, k, d); pf(s) = parse(Float64, s)
seed = parse(Int, envf("NZ_SEED", "1")); S = parse(Int, envf("VF_S", "3")); nloc = parse(Int, envf("VF_NLOC", "15")); inst = envf("VF_INST", "random")
xi = parse.(Int, split(envf("VF_X", inst == "sgb" ? "1,2" : "1"), ",")); TL = pf(envf("VF_TL", "300")); GAP = pf(envf("VF_GAP", "1e-3"))
res = Symbol(envf("VF_RES", "pair")); res in (:pair, :pooled) || error("VF_RES = pair | pooled")
pool = Symbol(envf("VF_POOL", "total")); vquota = parse(Float64, envf("VF_QUOTA", "300"))
poolkw = res == :pooled ? (pool_limit=pool, quota=vquota) : (quota=300.0,)
form = envf("VF_FORM", "vf"); rec = envf("VF_REC", "none"); bits = parse(Int, envf("VF_BITS", "0")); pre = envf("VF_PRE", "resp")
form in ("omega", "vf") || error("VF_FORM = omega | vf"); rec in ("none", "lattice") || error("VF_REC = none | lattice"); pre in ("resp", "basic") || error("VF_PRE = resp | basic")
nthr = parse(Int, envf("VF_THREADS", "12")); rlt = parse(Int, envf("VF_RLT", "0")); dminv = pf(envf("VF_DMIN", "0"))
grbkw = [(String(strip(a)), parse(Float64, b)) for (a, b) in (split(kv, "=") for kv in split(envf("VF_GRB", ""), ","; keepempty=false))]
setgrb!(mdl) = for (a, b) in grbkw; set_optimizer_attribute(mdl, a, isinteger(b) ? Int(b) : b); end
if inst == "sgb"
    ndb = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=res, quota=150.0, wres=300.0, S=S, seed=seed, eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
    nd0 = nz_with_bounds(ndb; thetaU=nz_theta_circuit_exact(ndb)[1])
else
    # VF_NA = 2, VF_NB = 3, VF_NBMAX = VF_NB, VF_QB = cap (출점비; 숫자면 그 값) — 2026-10-28 L6 (A 5 / B 5, 출점비 750) 용, 기본값은 그대로
    vnA = parse(Int, envf("VF_NA", "2")); vnB = parse(Int, envf("VF_NB", "3")); vNB = parse(Int, envf("VF_NBMAX", string(vnB)))
    cap = 150.0 * (nloc - vnA - vnB) / vnA
    vqB = envf("VF_QB", "cap") == "cap" ? cap : pf(envf("VF_QB", "cap"))
    nd0 = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=vnA, nB=vnB, NB=vNB, S=S, seed=seed,
        bA=cap, bB=cap, qB=vqB, dlo=50.0, dhi=150.0, reservation=res, wres=300.0, eps_hat=0.3, eps_tilde=0.3, beta=0.4, poolkw...), 0.1)
    res == :pooled && (nd0 = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1]))
end
x̄ = zeros(nd0.nx); x̄[xi] .= 1
mt = nd0.meta; nst, ncu, nA = mt[:nst], mt[:ncu], mt[:nA]; ξ = mt[:xi]
# ---------------- 프리솔브 (모두 "최적 반응이 될 수 없는 예약" 을 지우는 지배 논증, 문서 §10.15~10.16) ----------------
hU1, _ = nz_presolve_hU(nd0, x̄)                                                            # 닫힌 점포
res == :pair && (hU1 = [min(hU1[mt[:hidx](i, j)], maximum(ξ[j, :])) for i in 1:nst for j in 1:ncu])   # 예약 상한 ≤ 고객의 최대 수요 (pair 전용)
nd = nz_with_hU(nd0, hU1); nact0 = count(nd.hU .> 0); tpre = 0.0
if pre == "resp"
    hU2, nz0, tpre = nz_response_presolve(nd, x̄; env=_nz_envs(1)[1])
    global nd = nz_with_hU(nd, hU2)
end
A, c, ℓ = nd.A, nd.c, nd.ell; m, ny, nh = nz_m(nd), nz_ny(nd), nz_nh(nd); nW = size(nd.W, 1); q = nd.q_hat; ε̂, ε̃, β = nd.eps_hat, nd.eps_tilde, nd.beta
act = findall(nd.hU .> 0)
@printf("lab_vf_form: %s seed=%d S=%d x=%s n_h=%d (활성 %d → 프리솔브 %s 뒤 %d, %.1fs) | form=%s rec=%s bits=%d | gap %.0e, 제한 %.0fs, 스레드 %d\n",
        inst == "sgb" ? "SGB128" : "무작위 d=$nloc", seed, S, string(xi), nh, nact0, pre, length(act), tpre, form, rec, bits, GAP, TL, nthr); flush(stdout)
minimum(q) > ε̃ || rec == "lattice" || form == "omega" || dminv > 0 ||
    println("주의: q̂_s ≤ ε̃ 라 d_s = 0 이 가능 — VF_REC = none 의 상한은 유효하지만 느슨할 수 있다 (정확하려면 VF_REC = lattice)")
bvec(s) = [nd.u[k, s] + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0) for k in 1:m]       # α 를 뺀 우변

# ---------------- 시작해·하한: belief 표본의 최적 반응을 α 고정 LP 로 평가 ----------------
best = (-Inf, zeros(nh), copy(q)); tS = 0.0
dminv > 0 && println("진단: d_s ≥ ", dminv, " 을 둔 다른 문제 (값 ≤ V*). 시작해·표본 하한 없음")
if envf("VF_START", "1") == "1" && dminv == 0
    global tS = @elapsed begin
        fm = Model(GRB1); @variable(fm, 0 <= αf[i = 1:nh] <= nd.hU[i]); @variable(fm, yf[1:ny, 1:S] >= 0); @constraint(fm, nd.W * αf .<= nd.wvec)
        for s in 1:S
            b = bvec(s)
            @constraint(fm, [k = 1:m], sum(A[k, j] * yf[j, s] for j in 1:ny if A[k, j] != 0) <= b[k] + (nd.hrow[k] > 0 ? nd.hcoef[k] * αf[nd.hrow[k]] : 0.0))
        end
        ds = Vector{Float64}[]
        if S == 3
            for d1 in 0:0.01:1, d2 in 0:0.01:(1-d1)
                dd = [d1, d2, 1 - d1 - d2]; sum(abs, dd .- q) <= 2ε̃ + 1e-9 && push!(ds, dd)
            end
        else
            rng = MersenneTwister(1)
            for _ in 1:parse(Int, envf("VF_N", "3000"))
                δ = randn(rng, S); δ .-= mean(δ); δ .*= (2ε̃ * rand(rng)^(1 / (S - 1))) / sum(abs, δ); dd = q .+ δ
                any(<(0), dd) && (dd = max.(dd, 0.0); dd ./= sum(dd)); sum(abs, dd .- q) <= 2ε̃ + 1e-9 && push!(ds, dd)
            end
        end
        seen = Dict{Vector{Float64},Tuple{Vector{Float64},Vector{Float64}}}()
        for dd in ds
            @objective(fm, Max, dot(nd.c0, αf) + sum(dd[s] * c[j] * yf[j, s] for j in 1:ny, s in 1:S if c[j] != 0)); optimize!(fm)
            av = value.(αf); get!(seen, round.(av; digits=3), (av, dd))
        end
        F = nz_build_fixed_alpha(nd, x̄; optimizer=GRB1)
        for (av, dd) in values(seen)
            z = nz_eval_alpha!(F, nd, av)
            isfinite(z) && z > best[1] && (global best = (z, av, dd))
        end
    end
    @printf("시작해: belief 표본 %d → 서로 다른 반응 %d → 최댓값 %.4f (%.1fs)\n", length(ds), length(seen), best[1], tS); flush(stdout)
end

function report(model, name)
    st = termination_status(model); val = has_values(model) ? objective_value(model) : NaN; bd = objective_bound(model)
    lb = max(isnan(val) ? -Inf : val, best[1])
    @printf("RESULT %s S=%d n_h=%d(활성 %d) status=%s 해=%.4f (표본 하한 %.4f) 상한=%.4f gap=%.3e time=%.1f nodes=%d | 이진 %d 곱 %d\n", name, S, nh, length(act),
            string(st), val, best[1], bd, (bd - lb) / max(1.0, abs(lb)), solve_time(model), Int(MOI.get(model, MOI.NodeCount())),
            count(is_binary, all_variables(model)), num_constraints(model, QuadExpr, MOI.EqualTo{Float64}))
end

B = [bvec(s) for s in 1:S]
# t_s = cᵀy_s 의 범위: recourse 최적인 y_s 는 Q_s(0) ≤ t_s ≤ max_α Q_s(α) (Q_s 는 α 에 단조증가). φ_s 범위는 nz_kkt_eval 의 것
function trange(s)
    mdl = Model(GRB1); @variable(mdl, 0 <= αv[i = 1:nh] <= nd.hU[i]); @variable(mdl, yv[1:ny] >= 0); @constraint(mdl, nd.W * αv .<= nd.wvec)
    @constraint(mdl, [k = 1:m], sum(A[k, j] * yv[j] for j in 1:ny if A[k, j] != 0) <= B[s][k] + (nd.hrow[k] > 0 ? nd.hcoef[k] * αv[nd.hrow[k]] : 0.0))
    @objective(mdl, Max, dot(c, yv)); optimize!(mdl); hi = objective_value(mdl)
    for i in 1:nh; fix(αv[i], 0.0; force=true); end
    optimize!(mdl); return objective_value(mdl) - 1e-6, hi + 1e-6
end
if form == "omega"
    O = build_nz_omega(nd; optimizer=() -> Gurobi.Optimizer(GRB_ENV), mode=:global, silent=false)
    set_optimizer_attribute(O.model, "Threads", nthr); setgrb!(O.model)
    if envf("VF_LINK", "0") == "1"
        # 연결 부등식: 리더 복사본 ŷ_s = r_s ŷᵘ (ŷᵘ 는 recourse 최적), follower 복사본 ȳ_s = d_s yᵘ (yᵘ ∈ Y_s(α)) 이므로 cᵀŷᵘ ≥ cᵀyᵘ 에 r_s d_s 를 곱해
        #   d_s · cᵀŷ_s ≥ r_s · cᵀȳ_s.   d_s = 0 또는 r_s = 0 이면 0 ≥ 0. 곱은 스칼라끼리의 2S 개 (α 와 무관)
        v = O.v; tr0 = [trange(s) for s in 1:S]
        @variable(O.model, T̂[s = 1:S] <= 0); @variable(O.model, T̄[s = 1:S] <= 0)
        @constraint(O.model, [s = 1:S], T̂[s] == sum(c[j] * v[:ŷ][j, s] for j in 1:ny if c[j] != 0))
        @constraint(O.model, [s = 1:S], T̄[s] == sum(c[j] * v[:ȳ][j, s] for j in 1:ny if c[j] != 0))
        for s in 1:S                                   # recourse 최적인 복사본의 값은 Q_s(0) r_s ≤ T̂_s ≤ max Q_s · r_s (follower 쪽도 같음)
            @constraint(O.model, T̂[s] >= tr0[s][1] * v[:r][s]); @constraint(O.model, T̂[s] <= tr0[s][2] * v[:r][s])
            @constraint(O.model, T̄[s] >= tr0[s][1] * v[:d][s]); @constraint(O.model, T̄[s] <= tr0[s][2] * v[:d][s])
            set_lower_bound(T̂[s], tr0[s][1] * upper_bound(v[:r][s])); set_lower_bound(T̄[s], tr0[s][1] * upper_bound(v[:d][s]))
            @constraint(O.model, v[:d][s] * T̂[s] >= v[:r][s] * T̄[s])
        end
        println("연결 부등식 d_s·cᵀŷ_s ≥ r_s·cᵀȳ_s 추가 (곱 ", 2S, " 개)")
    end
    if isfinite(best[1])
        for i in 1:nh; set_start_value(O.v[:α][i], best[2][i]); end
        for s in 1:S; set_start_value(O.v[:d][s], best[3][s]); end
    end
    nz_solve!(O, nd, x̄; time_limit=TL, gap=GAP)
    report(O.model, "omega" * (envf("VF_LINK", "0") == "1" ? "+link" : "") * (isempty(grbkw) ? "" : "+" * envf("VF_GRB", ""))); exit()
end

# ---------------- 값함수 정식화 ----------------
M = Model(() -> Gurobi.Optimizer(GRB_ENV))
set_optimizer_attribute(M, "NonConvex", 2); set_optimizer_attribute(M, "MIPGap", GAP); set_time_limit_sec(M, TL); set_optimizer_attribute(M, "Threads", nthr); setgrb!(M)
amin = [max(0.0, q[s] - ε̂) for s in 1:S]; amax = [min(1.0, q[s] + ε̂) for s in 1:S]            # TV 볼 + 합 1 이면 한 좌표의 이동은 ε 이하
dmin = [max(dminv, q[s] - ε̃) for s in 1:S]; dmax = [min(1.0, q[s] + ε̃) for s in 1:S]
@variable(M, amin[s] <= a[s = 1:S] <= amax[s]); @variable(M, b[1:S] >= 0); @variable(M, 0 <= r[s = 1:S] <= min(1.0, amax[s] / (1 - β)))
@constraint(M, sum(a) == 1); @constraint(M, sum(r) == 1)
@constraint(M, [s = 1:S], a[s] - b[s] <= q[s]); @constraint(M, [s = 1:S], a[s] + b[s] >= q[s])
@constraint(M, sum(b) <= 2ε̂); @constraint(M, [s = 1:S], (1 - β) * r[s] <= a[s])
@variable(M, dmin[s] <= d[s = 1:S] <= dmax[s]); @variable(M, e[1:S] >= 0); @constraint(M, sum(d) == 1)
@constraint(M, [s = 1:S], d[s] - e[s] <= q[s]); @constraint(M, [s = 1:S], d[s] + e[s] >= q[s]); @constraint(M, sum(e) <= 2ε̃)
if nz_coupled(nd)
    @variable(M, cpl[1:S] >= 0)
    @constraint(M, [s = 1:S], a[s] - d[s] <= cpl[s]); @constraint(M, [s = 1:S], d[s] - a[s] <= cpl[s]); @constraint(M, sum(cpl) <= 2nz_delta(nd))
end
@variable(M, 0 <= α[i = 1:nh] <= nd.hU[i]); @constraint(M, nd.W * α .<= nd.wvec)
@variable(M, y[1:ny, 1:S] >= 0)
@constraint(M, Yrow[k = 1:m, s = 1:S], sum(A[k, j] * y[j, s] for j in 1:ny if A[k, j] != 0) <= B[s][k] + (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0))
tr = [trange(s) for s in 1:S]; φlo, φhi = _phi_range(nd, x̄; optimizer=GRB1)
@variable(M, tr[s][1] <= t[s = 1:S] <= tr[s][2]); @constraint(M, [s = 1:S], t[s] == sum(c[j] * y[j, s] for j in 1:ny if c[j] != 0))
@variable(M, φlo[s] <= φ[s = 1:S] <= φhi[s]); @constraint(M, [s = 1:S], φ[s] == sum(ℓ[j] * y[j, s] for j in 1:ny if ℓ[j] != 0))
@printf("범위: t_s ∈ %s, φ_s ∈ %s\n", string([round.(collect(x); digits=1) for x in tr]), string([(round(φlo[s]; digits=1), round(φhi[s]; digits=1)) for s in 1:S]))

"w = x·v. bits = 0 이면 bilinear 등식, 아니면 x 를 bits 자리 이진 전개 (NMDT): x = xL + Δ(Σ_k 2^-k z_k + ρ), w = xL v + Δ(Σ_k 2^-k z_k v + ρ v)"
function prod!(M, x, v, bits)
    w = @variable(M)
    bits == 0 && (@constraint(M, w == x * v); return w)
    xL, xU = lower_bound(x), upper_bound(x); Δ = xU - xL; vL, vU = lower_bound(v), upper_bound(v)
    Δ <= 1e-12 && (@constraint(M, w == xL * v); return w)
    z = @variable(M, [1:bits], Bin); ρ = @variable(M, lower_bound = 0.0, upper_bound = 2.0^-bits)
    @constraint(M, x == xL + Δ * (sum(2.0^-k * z[k] for k in 1:bits) + ρ))
    zv = @variable(M, [1:bits])
    for k in 1:bits
        @constraint(M, zv[k] <= vU * z[k]); @constraint(M, zv[k] >= vL * z[k])
        @constraint(M, zv[k] <= v - vL * (1 - z[k])); @constraint(M, zv[k] >= v - vU * (1 - z[k]))
    end
    ρv = @variable(M, lower_bound = min(0.0, 2.0^-bits * vL), upper_bound = max(0.0, 2.0^-bits * vU)); @constraint(M, ρv == ρ * v)
    @constraint(M, w == xL * v + Δ * (sum(2.0^-k * zv[k] for k in 1:bits) + ρv))
    return w
end
wF = [prod!(M, d[s], t[s], bits) for s in 1:S]; wL = [prod!(M, r[s], φ[s], bits) for s in 1:S]
if rlt >= 1
    # (Σ_s d_s − 1) t_s' = 0, (Σ_s r_s − 1) φ_s' = 0: 교차 곱은 Gurobi 의 bilinear 로 두고 (McCormick + 분기) 합 등식으로 묶는다
    Wd = [s == s2 ? wF[s] : prod!(M, d[s], t[s2], 0) for s in 1:S, s2 in 1:S]      # Wd[s, s2] = d_s t_s2
    Wr = [s == s2 ? wL[s] : prod!(M, r[s], φ[s2], 0) for s in 1:S, s2 in 1:S]      # Wr[s, s2] = r_s φ_s2
    @constraint(M, [s2 = 1:S], sum(Wd[:, s2]) == t[s2]); @constraint(M, [s2 = 1:S], sum(Wr[:, s2]) == φ[s2])
    if rlt >= 2
        # belief 다면체의 나머지 행 × (t_s2 − tᴸ), (tᵁ − t_s2) 와 × (φ_s2 − φᴸ), (φᵁ − φ_s2). 한 변수 × 다면체의 볼록포를 주는 행 집합.
        # 새 곱: e_s t_s2 (TV 볼), a_s φ_s2, b_s φ_s2 (TV 볼·CVaR)
        We = [prod!(M, e[s], t[s2], 0) for s in 1:S, s2 in 1:S]; Wa = [prod!(M, a[s], φ[s2], 0) for s in 1:S, s2 in 1:S]; Wb = [prod!(M, b[s], φ[s2], 0) for s in 1:S, s2 in 1:S]
        set_upper_bound.(e, 2ε̃); set_upper_bound.(b, 2ε̂)
        @constraint(M, [s2 = 1:S], sum(Wa[:, s2]) == φ[s2])
        for s2 in 1:S
            # g(belief) ≥ 0 인 행마다 (g 와 g·v 의 선형화 G) : G − vᴸ g ≥ 0,  vᵁ g − G ≥ 0
            rowsF = [(q[s] - d[s] + e[s], q[s] * t[s2] - Wd[s, s2] + We[s, s2]) for s in 1:S]
            append!(rowsF, [(d[s] + e[s] - q[s], Wd[s, s2] + We[s, s2] - q[s] * t[s2]) for s in 1:S])
            push!(rowsF, (2ε̃ - sum(e), 2ε̃ * t[s2] - sum(We[:, s2])))
            for (g, G) in rowsF
                @constraint(M, G - tr[s2][1] * g >= 0); @constraint(M, tr[s2][2] * g - G >= 0)
            end
            rowsL = [(q[s] - a[s] + b[s], q[s] * φ[s2] - Wa[s, s2] + Wb[s, s2]) for s in 1:S]
            append!(rowsL, [(a[s] + b[s] - q[s], Wa[s, s2] + Wb[s, s2] - q[s] * φ[s2]) for s in 1:S])
            push!(rowsL, (2ε̂ - sum(b), 2ε̂ * φ[s2] - sum(Wb[:, s2])))
            append!(rowsL, [(a[s] - (1 - β) * r[s], Wa[s, s2] - (1 - β) * Wr[s, s2]) for s in 1:S])
            for (g, G) in rowsL
                @constraint(M, G - φlo[s2] * g >= 0); @constraint(M, φhi[s2] * g - G >= 0)
            end
        end
    end
end
# follower 2 단계 LP 의 쌍대 (d 로 스케일된 π̃_s) 와 강쌍대
@variable(M, π̃[1:m, 1:S] >= 0); @variable(M, κ[1:nW] >= 0)
@constraint(M, [j = 1:ny, s = 1:S], sum(A[k, j] * π̃[k, s] for k in 1:m if A[k, j] != 0) >= c[j] * d[s])
@constraint(M, [i = 1:nh], sum(nd.hcoef[k] * π̃[k, s] for k in 1:m, s in 1:S if nd.hrow[k] == i) + nd.c0[i] <= sum(nd.W[l, i] * κ[l] for l in 1:nW))
@constraint(M, dot(nd.c0, α) + sum(wF) >= sum(B[s][k] * π̃[k, s] for k in 1:m, s in 1:S if B[s][k] != 0) + dot(nd.wvec, κ))
if rec == "lattice"
    # 시나리오별 recourse 강쌍대 (d_s = 0 에서도 y_s 가 recourse 최적). σ_H ∈ {0, 1/D, …, p}: 최적 쌍대 꼭짓점 (μ, γ) 가 격자점 (TU) 이고
    # 예약 행은 σ_ij = (μ_j − γ_i − c_ij)⁺ ≤ p 로 고를 수 있다 (예약 행의 우변 h_ij ≥ 0 이라 줄여도 최적)
    get(mt, :kind, nothing) == :location || error("VF_REC = lattice 는 location 인스턴스 전용 (격자 논증)")
    Dg = mt[:D]; L = round(Int, mt[:p] * Dg); nb = ceil(Int, log2(L + 1))
    @variable(M, σ[1:m, 1:S] >= 0); @constraint(M, [j = 1:ny, s = 1:S], sum(A[k, j] * σ[k, s] for k in 1:m if A[k, j] != 0) >= c[j])
    hk = [k for k in 1:m if nd.hrow[k] > 0 && nd.hU[nd.hrow[k]] > 0]
    @variable(M, zσ[hk, 1:S, 1:nb], Bin); @variable(M, gσ[hk, 1:S, 1:nb] >= 0)
    for k in hk, s in 1:S
        i = nd.hrow[k]; set_upper_bound(σ[k, s], nd.varpiU[k])
        @constraint(M, σ[k, s] == sum(2.0^(bb - 1) * zσ[k, s, bb] for bb in 1:nb) / Dg)
        for bb in 1:nb
            @constraint(M, gσ[k, s, bb] <= nd.hU[i] * zσ[k, s, bb]); @constraint(M, gσ[k, s, bb] <= α[i]); @constraint(M, gσ[k, s, bb] >= α[i] - nd.hU[i] * (1 - zσ[k, s, bb]))
        end
    end
    @constraint(M, [s = 1:S], t[s] >= sum(B[s][k] * σ[k, s] for k in 1:m if B[s][k] != 0) +
                                      sum(nd.hcoef[k] * 2.0^(bb - 1) * gσ[k, s, bb] for k in hk, bb in 1:nb) / Dg)
    @printf("recourse 격자: 1/D = %.3g, 단계 %d, 자리 %d, 이진 %d 개\n", 1 / Dg, L, nb, length(hk) * S * nb)
end
@objective(M, Max, sum(wL))
if isfinite(best[1])
    for i in 1:nh; set_start_value(α[i], best[2][i]); end
    for s in 1:S; set_start_value(d[s], best[3][s]); end
end
if !isempty(envf("VF_WRITE", ""))
    write_to_file(M, envf("VF_WRITE", "")); println("모델을 ", envf("VF_WRITE", ""), " 에 썼다 (max 문제, 변수 ", num_variables(M), " 개)")
end
optimize!(M)
report(M, "vf" * (rec == "lattice" ? "+lattice" : "") * (bits > 0 ? "+nmdt$bits" : "") * (rlt > 0 ? "+rlt$rlt" : "") * (isempty(grbkw) ? "" : "+" * envf("VF_GRB", "")))
@printf("  해: d = %s r = %s t = %s φ = %s\n", string(round.(value.(d); digits=4)), string(round.(value.(r); digits=4)), string(round.(value.(t); digits=2)), string(round.(value.(φ); digits=2)))
