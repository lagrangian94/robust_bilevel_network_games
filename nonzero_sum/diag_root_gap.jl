"""
diag_root_gap.jl — 분기 차원이 큰 무작위 인스턴스 (scale_oracle_random.jl 과 같은 생성기) 에서 α-B&B 의 루트 상한이 왜 0 인지,
상한이 무엇을 줄여야 내려가는지 본다 (문서 §10.14 의 "루트 완화가 정보를 주지 않는다" 의 원인 진단).

  A. incumbent α° (α-B&B 를 DG_INC_TL 초 돌려 얻음, 임시 폴더에 저장) 와 그 belief (a°, r°, d°) = α° 고정 LP 의 해
  B. 루트 완화 해가 무엇을 하는가: 시나리오별 예약 ζF/d, ζL/r 이 공통 α̂ 에서 얼마나 벗어나는지, 리더 블록의 B 판매량, α̂ 의 follower 최적성 여유
  C. α 상자 폭 곡선: α° 를 중심으로 폭 f × hᵁ 인 상자의 상한 (RLT 만 / 쌍대 범위 / + 차이 범위)
  D. belief 를 고정하면: 루트 α 상자에서 d = d° 만, (a, r) = (a°, r°) 만, 둘 다
  E. belief 상자 폭 곡선: d 를 d° 둘레 폭 w 로 묶고, 그 d 상자에서 follower 의 최적 반응이 있을 수 있는 α 범위 (KKT-MIP) 로 α 상자를 줄인 뒤의 상한
     (이 인스턴스에서는 indicator 정식화의 MIP 60 개가 모두 20 s 제한에 걸려 범위를 못 줄였다 — 기본 실행에서는 뺌)
  F. 상수의 영향: hᵁ 를 고객의 최대 수요로 줄임, θᵁ 를 줄임 (θᵁ 를 줄인 값은 유효성 보장이 없는 민감도 확인용)
  G. d 상자 + 노드 LP 위의 α 범위 조임 (목적 ≥ LB 영역의 좌표별 min/max), d 상자 없이 같은 범위 조임 (S = 3 전용)

환경변수: NZ_SEED = 1  DG_S = 3  DG_X = "1"  DG_NLOC = 15  DG_QUOTA = 300  DG_PARTS = "B,C,D,F,G"  DG_INC_TL = 60  DG_THREADS = 12
          DG_FRACS = "1,0.5,0.25,0.125,0.06,0.03,0.015" (f = 2 면 루트 상자 전체, f = 1 은 α° ± hᵁ/2 라 루트보다 작다)  DG_WB = "0,0.01,0.02,0.05,0.1,0.2,0.4"
실행: julia -t 14,1 nonzero_sum/diag_root_gap.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Serialization, Statistics
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d); pf(s) = parse(Float64, s)
seed = parse(Int, envf("NZ_SEED", "1")); S = parse(Int, envf("DG_S", "3")); nloc = parse(Int, envf("DG_NLOC", "15"))
xi = parse.(Int, split(envf("DG_X", "1"), ",")); NT = parse(Int, envf("DG_THREADS", "12"))
parts = Set(String.(strip.(split(envf("DG_PARTS", "B,C,D,F,G"), ","))))
cap = 150.0 * (nloc - 5) / 2
nd0 = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=2, nB=3, NB=3, S=S, seed=seed,
    bA=cap, bB=cap, qB=cap, dlo=50.0, dhi=150.0, reservation=:pair, quota=pf(envf("DG_QUOTA", "300")), wres=300.0,
    eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
x̄ = zeros(nd0.nx); x̄[xi] .= 1
hUp, _ = nz_presolve_hU(nd0, x̄); nd = nz_with_hU(nd0, hUp)
nh, m, ny = nz_nh(nd), nz_m(nd), nz_ny(nd)
act = findall(nd.hU .> 0)
meta = nd.meta; nst, ncu, nA = meta[:nst], meta[:ncu], meta[:nA]; iyR, iyS, hidx = meta[:iyR], meta[:iyS], meta[:hidx]
ξ = meta[:xi]
envs = _nz_envs(NT + 1)
mk(env) = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
@printf("diag_root_gap: d=%d seed=%d S=%d x=%s n_h=%d (활성 %d) θᵁ=%.1f λᵁ=%.0f | hᵁ %.0f, 수요 %.0f~%.0f\n", nloc, seed, S, string(xi), nh,
        length(act), nd.thetaU, nd.lambdaU, maximum(nd.hU), minimum(ξ), maximum(ξ)); flush(stdout)

# ---------------- A. incumbent ----------------
incfile = joinpath(tempdir(), "nz_diag_root_gap_d$(nloc)_seed$(seed)_S$(S)_x$(join(xi, "-")).jls")
if isfile(incfile)
    inc = deserialize(incfile)
else
    r = nz_alpha_bnb_dr(nd0, x̄; nworkers=NT, time_limit=pf(envf("DG_INC_TL", "60")), rel_gap=1e-3, verbose=false, ipopt_time=120.0,
                        dual_range=false)
    inc = (LB=r[:LB], α=r[:α]); serialize(incfile, inc)
end
LB = inc.LB; α° = inc.α; sc = max(1.0, abs(LB))
F0 = nz_build_fixed_alpha(nd, x̄; optimizer=mk(envs[1]))
zfix = nz_eval_alpha!(F0, nd, α°)
a° = value.(F0.v[:a]); r° = value.(F0.v[:r]); d° = value.(F0.v[:d])
@printf("A. incumbent LB=%.4f (α° 고정 LP %.4f) | a°=%s r°=%s d°=%s | α° > 0 인 좌표 %d, 합 %.1f\n", LB, zfix, string(round.(a°; digits=3)),
        string(round.(r°; digits=3)), string(round.(d°; digits=3)), count(>(1e-6), α°), sum(α°)); flush(stdout)

# ---------------- 참값 계산용 LP ----------------
"시나리오별 Q_s(α) (follower recourse 값) 와 φ_s(α) (pessimistic 리더 값)"
function true_vals(ndx, α; env=envs[1])
    Q = zeros(S); φ = zeros(S)
    for s in 1:S
        rr = nz_rhs(ndx, x̄, α, s)
        mdl = Model(mk(env)); @variable(mdl, y[1:ny] >= 0)
        @constraint(mdl, [k = 1:m], sum(ndx.A[k, j] * y[j] for j in 1:ny if ndx.A[k, j] != 0) <= rr[k])
        @objective(mdl, Max, dot(ndx.c, y)); optimize!(mdl); Q[s] = objective_value(mdl)
        @constraint(mdl, dot(ndx.c, y) >= Q[s] - 1e-7 * max(1, abs(Q[s])))
        @objective(mdl, Max, dot(ndx.ell, y)); optimize!(mdl); φ[s] = objective_value(mdl)
    end
    return Q, φ
end
"belief d 에서 follower 의 here-and-now 최적값"
function follower_opt(ndx, d; env=envs[1])
    mdl = Model(mk(env)); @variable(mdl, 0 <= α[i = 1:nh] <= ndx.hU[i]); @variable(mdl, y[1:ny, 1:S] >= 0)
    @constraint(mdl, ndx.W * α .<= ndx.wvec)
    for s in 1:S, k in 1:m
        @constraint(mdl, sum(ndx.A[k, j] * y[j, s] for j in 1:ny if ndx.A[k, j] != 0) <=
                         ndx.u[k, s] + (ndx.hrow[k] > 0 ? ndx.hcoef[k] * α[ndx.hrow[k]] : 0.0) + (ndx.xrow[k] > 0 ? ndx.g[k, s] * x̄[ndx.xrow[k]] : 0.0))
    end
    @objective(mdl, Max, dot(ndx.c0, α) + sum(d[s] * ndx.c[j] * y[j, s] for j in 1:ny, s in 1:S)); optimize!(mdl)
    return objective_value(mdl), value.(α)
end

"""
d 상자 [dl, dh] (∩ TV 볼) 의 어떤 belief 에서 follower 의 최적 반응이 될 수 있는 α 의 좌표별 범위. follower 의 2 단계 LP 의 KKT 를 d 를 변수로 둔 MIP 로
(쌍대를 d_s 로 스케일하면 d 에 대해 선형, 상보성은 indicator). nz_kkt_eval.jl 의 follower 부분과 같다.
"""
function alpha_range(ndx, dl, dh; env=envs[1], tl=20.0)
    q, ε̃ = ndx.q_hat, ndx.eps_tilde; nW = size(ndx.W, 1); A, c = ndx.A, ndx.c
    mdl = Model(mk(env)); set_optimizer_attribute(mdl, "MIPGap", 0.0); set_optimizer_attribute(mdl, "MIPGapAbs", 0.0); set_time_limit_sec(mdl, tl)
    @variable(mdl, dl[s] <= d[s = 1:S] <= dh[s]); @variable(mdl, e[1:S] >= 0); @constraint(mdl, sum(d) == 1)
    @constraint(mdl, [s = 1:S], d[s] - e[s] <= q[s]); @constraint(mdl, [s = 1:S], d[s] + e[s] >= q[s]); @constraint(mdl, sum(e) <= 2ε̃)
    @variable(mdl, 0 <= α[i = 1:nh] <= ndx.hU[i]); @variable(mdl, y[1:ny, 1:S] >= 0); @variable(mdl, π[1:m, 1:S] >= 0); @variable(mdl, κ[1:nW] >= 0)
    sP = [@expression(mdl, ndx.u[k, s] + (ndx.hrow[k] > 0 ? ndx.hcoef[k] * α[ndx.hrow[k]] : 0.0) + (ndx.xrow[k] > 0 ? ndx.g[k, s] * x̄[ndx.xrow[k]] : 0.0)
                           - sum(A[k, j] * y[j, s] for j in 1:ny if A[k, j] != 0)) for k in 1:m, s in 1:S]
    sD = [@expression(mdl, sum(A[k, j] * π[k, s] for k in 1:m if A[k, j] != 0) - c[j] * d[s]) for j in 1:ny, s in 1:S]
    sW = [@expression(mdl, ndx.wvec[l] - sum(ndx.W[l, i] * α[i] for i in 1:nh)) for l in 1:nW]
    sA = [@expression(mdl, sum(ndx.W[l, i] * κ[l] for l in 1:nW) - ndx.c0[i] - sum(ndx.hcoef[k] * π[k, s] for k in 1:m, s in 1:S if ndx.hrow[k] == i)) for i in 1:nh]
    @constraint(mdl, sP .>= 0); @constraint(mdl, sD .>= 0); @constraint(mdl, sW .>= 0)
    comp(var, slack) = (z = @variable(mdl, binary = true); @constraint(mdl, z => {var == 0}); @constraint(mdl, !z => {slack == 0}))
    for k in 1:m, s in 1:S; comp(π[k, s], sP[k, s]); end
    for j in 1:ny, s in 1:S; comp(y[j, s], sD[j, s]); end
    for l in 1:nW; comp(κ[l], sW[l]); end
    for i in 1:nh
        ndx.hU[i] > 0 || continue                       # 프리솔브로 0 에 고정된 좌표는 정상성 조건에서 뺀다 (지배 논증, nz_presolve_hU)
        @constraint(mdl, sA[i] >= 0); comp(α[i], sA[i])
    end
    l = zeros(nh); u = copy(ndx.hU); nto = 0
    for i in 1:nh
        ndx.hU[i] > 0 || continue
        for ismax in (false, true)
            @objective(mdl, ismax ? MOI.MAX_SENSE : MOI.MIN_SENSE, α[i]); optimize!(mdl)
            st = termination_status(mdl)
            st == MOI.INFEASIBLE && return nothing
            st == MOI.OPTIMAL || (nto += 1)
            bd = objective_bound(mdl)
            ismax ? (u[i] = min(u[i], bd + 1e-6)) : (l[i] = max(l[i], bd - 1e-6))
        end
    end
    return l, u, nto
end

"""
상자 [l, u] 의 노드 완화 상한. cfg = :rlt (RLT 만) | :dr (쌍대 범위, 차이 범위 끔) | :full (기본 구성).
belief = Dict(:d => (lo, hi), :a => …, :r => …) 를 주면 그 변수의 범위를 줄이고 (범위 × α 상자) RLT 행을 후보에 넣는다 (lo = hi 면 곱이 정확해진다).
"""
function relax(ndx, l, u, cfg; belief=Dict{Symbol,Any}(), env=envs[1])
    P = _nz_build_sep(ndx, x̄; optimizer=mk(env))
    for a in ("FeasibilityTol", "OptimalityTol"); set_optimizer_attribute(P.O.model, a, 1e-7); end
    set_optimizer_attribute(P.O.model, "NumericFocus", 0); P.O.model.ext[:lp_tol] = 1e-7; P.O.model.ext[:retry_fresh] = true
    for (fam, (bl, bh)) in belief, s in 1:S
        y = P.Y[fam][s]
        set_lower_bound(y, max(lower_bound(y), bl[s])); set_upper_bound(y, min(upper_bound(y), bh[s]))
        push!(P.rows, _NZRow(-lower_bound(y), [(fam, s, 1.0)], 0)); push!(P.rowscen, s)
        push!(P.rows, _NZRow(upper_bound(y), [(fam, s, -1.0)], 0)); push!(P.rowscen, s)
    end
    tg = 0.0; fixfrac = NaN
    if cfg == :rlt
        _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
    else
        DR = _nz_dr_attach!(P, ndx, x̄; optimizer=mk(env), dd=(cfg == :full))
        lo0, hi0 = _nz_dr_global(ndx, DR)
        _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
        tg = @elapsed (rg = _nz_dr_ranges(DR, ndx, l, u, lo0, hi0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), fill(Inf, S)))
        rg === nothing && return (z=-Inf, tg=tg, tl=0.0, fix=NaN, P=P)
        lo, hi, wt, phib = rg
        dl, dh = _nz_dr_dglobal(ndx, DR)
        isempty(DR.pairs) || (tg += @elapsed ((dl, dh) = _nz_dr_dranges!(DR, ndx, l, u, lo, hi, dl, dh, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), wt)))
        _nz_dr_set!(P, DR, ndx, l, u, lo, hi, phib; dlo=dl, dhi=dh)
        fixfrac = count(hi .- lo .<= 1e-4) / length(lo)
    end
    tl = @elapsed (z = _nz_sep_solve!(P; maxrounds=30))
    z == -Inf && (z = _nz_dr_recheck_infeasible!(P.O.model))
    return (z=z, tg=tg, tl=tl, fix=fixfrac, P=P)
end
"여러 경우를 worker 에 나눠 실행. cases = [(이름, 함수 env -> 결과)]"
function runcases(cases)
    res = Vector{Any}(undef, length(cases)); nxt = Threads.Atomic{Int}(1)
    tasks = map(1:min(NT, length(cases))) do w
        Threads.@spawn while true
            n = Threads.atomic_add!(nxt, 1); n > length(cases) && break
            res[n] = try cases[n][2](envs[w]) catch err; (z=NaN, tg=0.0, tl=0.0, fix=NaN, err=sprint(showerror, err)) end
        end
    end
    foreach(wait, tasks)
    return res
end
gapof(z) = (z - LB) / sc
abox(ndx, f) = (max.(0.0, α° .- f .* ndx.hU ./ 2), min.(ndx.hU, α° .+ f .* ndx.hU ./ 2))
show1(name, r) = (haskey(r, :err) ? @printf("  %-34s 오류: %s\n", name, first(r.err, 200)) :
                  @printf("  %-34s 상한 %10.2f  gap %7.1f%%  | 고정 비율 %4.2f  범위 MIP %6.1fs  LP %6.1fs\n", name, r.z, 100 * gapof(r.z), r.fix, r.tg, r.tl); flush(stdout))

# ---------------- B. 루트 완화 해 ----------------
if "B" in parts
    println("\nB. 루트 완화 (쌍대 범위, 차이 범위 끔) 의 해")
    rB = relax(nd, zeros(nh), copy(nd.hU), :dr); v = rB.P.O.v
    α̂ = value.(v[:α]); d̂ = value.(v[:d]); r̂ = value.(v[:r]); â = value.(v[:a])
    Q, φ = true_vals(nd, α̂); vHN, αHN = follower_opt(nd, d̂)
    @printf("  상한 %.4f | d̂=%s r̂=%s â=%s\n", rB.z, string(round.(d̂; digits=3)), string(round.(r̂; digits=3)), string(round.(â; digits=3)))
    isB(j) = any(j == iy(i, c) for i in nA+1:nst for c in 1:ncu for iy in (iyR, iyS))
    for s in 1:S
        Bl = r̂[s] > 1e-7 ? sum(value(v[:ŷ][j, s]) for j in 1:ny if isB(j)) / r̂[s] : NaN
        Bf = d̂[s] > 1e-7 ? sum(value(v[:ȳ][j, s]) for j in 1:ny if isB(j)) / d̂[s] : NaN
        dF = d̂[s] > 1e-7 ? [value(v[:ζF][i, s]) / d̂[s] - α̂[i] for i in act] : fill(NaN, length(act))
        dL = r̂[s] > 1e-7 ? [value(v[:ζL][i, s]) / r̂[s] - α̂[i] for i in act] : fill(NaN, length(act))
        cF = d̂[s] > 1e-7 ? sum(nd.c[j] * value(v[:ȳ][j, s]) for j in 1:ny) / d̂[s] : NaN
        @printf("  s=%d 총수요 %6.1f | B 판매: 리더 블록 %6.1f, follower 블록 %6.1f, 참 (α̂ 에서 비관적) %6.1f | 시나리오 예약 − α̂: follower Σ|·| %7.1f (최대 %6.1f), 리더 Σ|·| %7.1f | cȳ/d − Q_s(α̂) = %+.2f\n",
                s, sum(ξ[:, s]), Bl, Bf, -φ[s] / meta[:v], sum(abs, dF), maximum(abs, dF), sum(abs, dL), cF - Q[s])
    end
    vα̂ = dot(nd.c0, α̂) + dot(d̂, Q)
    @printf("  α̂ 의 follower 값 (d̂) %.3f vs follower 최적값 %.3f → 최적성 여유 %.3f | Σα̂ %.1f (α° %.1f, follower 최적 반응 %.1f)\n", vα̂, vHN, vHN - vα̂, sum(α̂), sum(α°), sum(αHN))
    Q°, φ° = true_vals(nd, α°)
    @printf("  참고: α° 에서 B 판매 (비관적) = %s, 리더 값 Σ r°φ = %.2f\n", string(round.(-φ° ./ meta[:v]; digits=1)), dot(r°, φ°)); flush(stdout)
end

# ---------------- C. α 상자 폭 곡선 ----------------
fracs = pf.(split(envf("DG_FRACS", "1,0.5,0.25,0.125,0.06,0.03,0.015"), ","))
if "C" in parts
    println("\nC. α° 둘레 폭 f × hᵁ 인 상자의 상한 (gap = (상한 − LB) / |LB|)")
    cases = [(@sprintf("f=%-6.3f %s", f, cfg), env -> relax(nd, abox(nd, f)..., cfg; env=env)) for f in fracs for cfg in (:rlt, :dr, :full)]
    t = @elapsed (res = runcases(cases))
    for (c, r) in zip(cases, res); show1(c[1], r); end
    @printf("  (%.0fs)\n", t)
end

# ---------------- D. belief 고정 ----------------
if "D" in parts
    println("\nD. 루트 α 상자에서 belief 를 incumbent 의 값에 고정 (그 belief 로 제한한 문제의 상한. LB 는 이 belief 에서 달성됨)")
    bd = Dict{Symbol,Any}(:d => (d°, d°)); bl = Dict{Symbol,Any}(:a => (a°, a°), :r => (r°, r°)); ball = merge(bd, bl)
    cases = [(nm * " " * string(cfg), env -> relax(nd, zeros(nh), copy(nd.hU), cfg; belief=b, env=env))
             for (nm, b) in (("belief 자유", Dict{Symbol,Any}()), ("d 고정", bd), ("(a, r) 고정", bl), ("모두 고정", ball)) for cfg in (:rlt, :dr)]
    res = runcases(cases)
    for (c, r) in zip(cases, res); show1(c[1], r); end
end

# ---------------- E. belief 상자 폭 곡선 ----------------
if "E" in parts
    println("\nE. d 를 d° 둘레 폭 w 로 묶음 → 그 d 상자의 최적 반응 α 범위 (KKT-MIP) 로 α 상자를 줄임 → 상한. (a, r) 은 자유")
    wbs = pf.(split(envf("DG_WB", "0,0.01,0.02,0.05,0.1,0.2,0.4"), ","))
    function caseE(w, env)
        dl = max.(0.0, d° .- w / 2); dh = min.(1.0, d° .+ w / 2)
        tr = @elapsed (rg = alpha_range(nd, dl, dh; env=env))
        rg === nothing && return (z=-Inf, tg=tr, tl=0.0, fix=NaN, wsum=NaN, nfree=0, nto=0, out=Dict())
        l, u, nto = rg
        bel = Dict{Symbol,Any}(:d => (dl, dh))
        out = Dict(cfg => relax(nd, l, u, cfg; belief=bel, env=env) for cfg in (:rlt, :dr, :full))
        out[:nob] = relax(nd, l, u, :dr; env=env)                     # 같은 α 상자, belief 행 없이 (α 범위만의 효과)
        return (tr=tr, wsum=sum(u .- l) / sum(nd.hU), nfree=count(u .- l .> 1e-4), nto=nto, out=out, inbox=all(l .- 1e-5 .<= α° .<= u .+ 1e-5))
    end
    cases = [(w, env -> caseE(w, env)) for w in wbs]
    t = @elapsed (res = runcases(cases))
    for (c, r) in zip(cases, res)
        if haskey(r, :err); @printf("  w=%.3f 오류: %s\n", c[1], first(r.err, 300)); continue; end
        isempty(r.out) && (@printf("  w=%.3f  α 범위 MIP 가 실행 불가능\n", c[1]); continue)
        @printf("  w=%-5.3f α 범위: 폭 합 %5.1f%% (hᵁ 대비), 자유 좌표 %2d / %d, α° 포함 %s, MIP %5.1fs (시간 제한 %d) | gap: RLT %7.1f%%  쌍대 범위 %7.1f%%  기본 %7.1f%%  (belief 행 없이 쌍대 범위 %7.1f%%) | 고정 비율 %.2f\n",
                c[1], 100 * r.wsum, r.nfree, length(act), string(r.inbox), r.tr, r.nto, 100 * gapof(r.out[:rlt].z), 100 * gapof(r.out[:dr].z),
                100 * gapof(r.out[:full].z), 100 * gapof(r.out[:nob].z), r.out[:dr].fix)
    end
    @printf("  (%.0fs)\n", t); flush(stdout)
end

# ---------------- F. 상수 ----------------
if "F" in parts
    println("\nF. 상수의 영향 (쌍대 범위 구성, 루트와 α° 둘레 폭 0.25·0.06 의 상자)")
    dmax = [maximum(ξ[j, :]) for j in 1:ncu]
    hUd = [min(nd.hU[hidx(i, j)], dmax[j]) for i in 1:nst for j in 1:ncu]
    ndh = nz_with_hU(nd, hUd)
    @printf("  hᵁ 를 고객의 최대 수요로: Σhᵁ %.0f → %.0f\n", sum(nd.hU), sum(hUd))
    vars = Any[("기본", nd), ("hᵁ ≤ 최대 수요", ndh)]
    for θ in (20.0, 5.0); push!(vars, (@sprintf("θᵁ = %.0f (민감도)", θ), nz_with_bounds(nd; thetaU=θ))); end
    cases = [(@sprintf("%-18s f=%-5.2f", nm, f), env -> relax(ndx, abox(ndx, f)..., :dr; env=env)) for (nm, ndx) in vars for f in (2.0, 0.25, 0.06)]
    res = runcases(cases)
    for (c, r) in zip(cases, res); show1(c[1], r); end
end

# ---------------- G. d 상자 + LP 범위 조임 ----------------
# d 를 좁히면 follower 블록의 곱 ζF = α d 가 정확해져 완화 자체가 α 를 Ψ(d) 근처로 묶는다. 그 효과를 노드 LP 위의 범위 조임 (좌표마다 min/max α_i,
# 목적 ≥ LB 조건 포함) 으로 α 상자에 옮기고, 줄어든 α 상자에서 쌍대 범위를 다시 계산해 상한을 본다. d 상자의 중심은 belief 격자에서 찾은 셀의 무게중심.
if "G" in parts
    println("\nG. d 상자 (셀 무게중심 둘레 폭 w) + 노드 LP 위의 α 범위 조임 (LP 2 × 활성 좌표 개, 라운드 반복)")
    fm = Model(mk(envs[1])); @variable(fm, 0 <= αf[i = 1:nh] <= nd.hU[i]); @variable(fm, yf[1:ny, 1:S] >= 0); @constraint(fm, nd.W * αf .<= nd.wvec)
    for s in 1:S, k in 1:m
        @constraint(fm, sum(nd.A[k, j] * yf[j, s] for j in 1:ny if nd.A[k, j] != 0) <=
                        nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * αf[nd.hrow[k]] : 0.0) + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0))
    end
    resp(d) = (@objective(fm, Max, dot(nd.c0, αf) + sum(d[s] * nd.c[j] * yf[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0)); optimize!(fm); value.(αf))
    S == 3 || error("G 는 S = 3 전용 (격자)")
    qv, εt = nd.q_hat, nd.eps_tilde
    cells = Dict{Vector{Float64},Any}()
    for d1 in 0:0.01:1, d2 in 0:0.01:(1-d1)
        d = [d1, d2, 1 - d1 - d2]; sum(abs, d .- qv) <= 2εt + 1e-9 || continue
        a = resp(d); key = round.(a; digits=3); c = get!(cells, key, (α=a, ds=Vector{Float64}[])); push!(c.ds, d)
    end
    cl = [(α=c.α, n=length(c.ds), dc=mean(c.ds), G=nz_eval_alpha!(F0, nd, c.α)) for c in values(cells)]
    sort!(cl; by=c -> -c.G)
    @printf("  셀 %d 개. 값이 큰 순서로 1, 2 번째와 표본이 가장 많은 셀을 본다\n", length(cl))
    function caseG(c, w, env; rounds=4, l0=zeros(nh), u0=copy(nd.hU))
        dl = max.(0.0, c.dc .- w / 2); dh = min.(1.0, c.dc .+ w / 2)
        P = _nz_build_sep(nd, x̄; optimizer=mk(env)); mdl = P.O.model; αv = P.O.v[:α]
        for a in ("FeasibilityTol", "OptimalityTol"); set_optimizer_attribute(mdl, a, 1e-7); end
        set_optimizer_attribute(mdl, "NumericFocus", 0); mdl.ext[:lp_tol] = 1e-7; mdl.ext[:retry_fresh] = true
        for s in 1:S
            y = P.Y[:d][s]; set_lower_bound(y, max(lower_bound(y), dl[s])); set_upper_bound(y, min(upper_bound(y), dh[s]))
            push!(P.rows, _NZRow(-lower_bound(y), [(:d, s, 1.0)], 0)); push!(P.rowscen, s)
            push!(P.rows, _NZRow(upper_bound(y), [(:d, s, -1.0)], 0)); push!(P.rowscen, s)
        end
        DR = _nz_dr_attach!(P, nd, x̄; optimizer=mk(env), dd=false); lo0, hi0 = _nz_dr_global(nd, DR); dl0, dh0 = _nz_dr_dglobal(nd, DR)
        l = copy(l0); u = copy(u0); hist = String[]; z = Inf; tt = 0.0
        fobj = objective_function(mdl)
        for rnd in 1:rounds
            tt += @elapsed begin
                _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
                rg = _nz_dr_ranges(DR, nd, l, u, lo0, hi0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), fill(Inf, S))
                rg === nothing && (push!(hist, "범위 MIP: 실행 불가능"); z = -Inf; break)
                lo, hi, _, phib = rg
                _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; dlo=dl0, dhi=dh0)
                z = _nz_sep_solve!(P; maxrounds=30)
            end
            push!(hist, @sprintf("라운드 %d: 상한 %9.2f (gap %6.2f%%), α 폭 합 %6.1f%%, 자유 좌표 %2d, 쌍대 고정 %.2f", rnd, z, 100 * gapof(z), 100 * sum(u .- l) / sum(nd.hU),
                                 count(u .- l .> 1e-4), count(hi .- lo .<= 1e-4) / length(lo)))
            (z == -Inf || z <= LB + 1e-3 * sc || rnd == rounds) && break
            tt += @elapsed begin                        # 범위 조임: 목적 ≥ LB 인 영역에서 좌표별 min/max
                cut = @constraint(mdl, fobj >= LB - 1e-6 * sc); nl = copy(l); nu = copy(u); bad = false
                for i in act, ismax in (false, true)
                    @objective(mdl, ismax ? MOI.MAX_SENSE : MOI.MIN_SENSE, αv[i]); optimize!(mdl)
                    st = termination_status(mdl)
                    st == MOI.INFEASIBLE && (bad = true; break)
                    st == MOI.OPTIMAL || continue
                    ismax ? (nu[i] = min(u[i], objective_value(mdl) + 1e-6)) : (nl[i] = max(l[i], objective_value(mdl) - 1e-6))
                end
                delete(mdl, cut); @objective(mdl, Max, fobj)
            end
            bad && (push!(hist, "범위 조임: 목적 ≥ LB 인 점이 없음 → 가지치기"); z = -Inf; break)
            l = nl; u = max.(nu, nl)
        end
        return (hist=hist, z=z, t=tt)
    end
    pick = unique([1, 2, argmax([c.n for c in cl])])
    cases = [((k, w), env -> caseG(cl[k], w, env)) for k in pick for w in (0.2, 0.1, 0.05, 0.02)]
    res = runcases(cases)
    for (c, r) in zip(cases, res)
        k, w = c[1]
        @printf("  셀 %d (G = %.2f, 표본 %d, 무게중심 %s) w=%.2f  (%.0fs)\n", k, cl[k].G, cl[k].n, string(round.(cl[k].dc; digits=3)), w, haskey(r, :err) ? NaN : r.t)
        haskey(r, :err) ? println("      오류: ", first(r.err, 300)) : foreach(h -> println("      ", h), r.hist)
    end
    flush(stdout)
    # d 상자 없이 (w = 2 → TV 볼 전체): α 상자만 있는 노드에서 같은 범위 조임이 얼마나 줄이는가 (α-B&B 의 노드에 그대로 넣을 수 있는지)
    println("  d 상자 없이 (belief 전체): 루트와 α° 둘레 폭 f × hᵁ 의 상자에서 범위 조임")
    cases2 = [(f, env -> caseG(cl[1], 2.0, env; rounds=6, l0=(f >= 2 ? zeros(nh) : abox(nd, f)[1]), u0=(f >= 2 ? copy(nd.hU) : abox(nd, f)[2]))) for f in (2.0, 0.5, 0.25, 0.06)]
    res2 = runcases(cases2)
    for (c, r) in zip(cases2, res2)
        @printf("  %s  (%.0fs)\n", c[1] >= 2 ? "루트" : @sprintf("f=%.2f", c[1]), haskey(r, :err) ? NaN : r.t)
        haskey(r, :err) ? println("      오류: ", first(r.err, 300)) : foreach(h -> println("      ", h), r.hist)
    end
    flush(stdout)
end
