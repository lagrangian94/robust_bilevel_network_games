"""
nz_alpha_bnb.jl — 비제로섬 Ω (nz_omega.jl) 의 전역 해법: α 만 분기하는 spatial B&B.
true_dro/global_bilinear_solver.jl (zero-sum) 의 구조를 그대로 따른다.

비제로섬 Ω 의 bilinear 항은 전부 α × (belief 또는 인증서):
  ζL_is = α_i r_s,  ζF_is = α_i d_s  (기존과 같은 종류)
  ζW_js = α_{h(j)} ϖ_js              (신규: H 행 j 의 follower 인증서, html Step 4′ 의 α_k ϖ_k)
→ α 고정이면 LP (정확한 값, valid LB), α 상자 위 RLT 완화도 LP (valid UB), 상자가 점이면 exact.

노드 완화 (level-1 RLT): 선형 행 g(y) ≥ 0 × 상자 인수 (α_i − l_i), (u_i − α_i) 를 곱 변수로 선형화.
  belief 행: a, b, r, d, e 의 상·하한, TV 볼 (a−b≤q̂, a+b≥q̂, Σb≤2ε̂, d·e 동일), CVaR r ≤ a/(1−β)  → 모든 α_i 와 곱
  인증서 행: ϖ_js ≥ 0, ϖ_js ≤ ϖᵁ_j r_s (Ω 의 Wcap)                                  → 자기 α_{h(j)} 와만 곱
     (다른 α_i 와의 곱 α_i ϖ_js 는 Ω 에 없으므로 만들지 않음. Wcap RLT 가 ζW 와 ζL 을 묶어 McCormick 보다 강함)
  정적 RLT: Σ_s Z_is = α_i (Σa = Σr = Σd = 1), Σ_i W_li Z_is ≤ w_l y_s (Wα ≤ w).
RLT 행은 위반된 것만 분리해 추가. 부모 상자 행은 자식에서도 유효 → 지우지 않고 무효 행은 rhs=−∞ 로 비활성.

탐색: worker 병렬 (worker 마다 Gurobi env/LP), best-first + diving. 분기: 곱 위반이 가장 큰 α_i, 분할점 α̂_i.
LB: 노드 α̂ 고정 LP + Ipopt 로컬 해 (α 를 고정해 정확한 LP 로 재평가한 값만 LB).

사용:  r = nz_alpha_bnb(nd, x̄; nworkers=12, time_limit=300.0, rel_gap=1e-4)
Julia 는 `-t (nworkers + 2),1` 로 실행 (interactive 스레드 1 개: 메인 루프·타이머).
"""

using JuMP, Printf, LinearAlgebra
import Gurobi
using Ipopt
import MathOptInterface as MOI


# =====================================================================
# 선형 행  g(y) = c0 + Σ c·Y[fam][idx] ≥ 0,  ionly > 0 이면 그 α_i 와만 곱함
# =====================================================================
struct _NZRow
    c0::Float64
    terms::Vector{Tuple{Symbol,Int,Float64}}    # (family, idx, coef)
    ionly::Int
end

"""노드 LP: Ω (mode=:relax) + 곱 변수 + 정적 RLT + 분리된 RLT 행."""
mutable struct _NZSepLP
    O::NZOmega
    nh::Int
    S::Int
    Y::Dict{Symbol,Vector{VariableRef}}
    Z::Dict{Symbol,Matrix{VariableRef}}          # a, b, r, d, e: nh × S
    wpos::Vector{Tuple{Int,Int,Int}}             # :w 의 idx → (jj, s, i = h(Hr[jj]))
    rows::Vector{_NZRow}
    allrows::Vector{Tuple{Int,Bool,Float64,ConstraintRef,Float64,Int}}   # (i, islo, b, 행, 원래 rhs, j)
    active::Vector{Bool}
    l::Vector{Float64}
    u::Vector{Float64}
    mcW::Vector{NTuple{4,ConstraintRef}}         # ζW = α_i ϖ 의 노드별 McCormick 4 행 (wpos 순서), ϖ 분기용
    Lw::Vector{Float64}                          # 노드의 ϖ 상자 (wpos 순서)
    Uw::Vector{Float64}
end

function _nz_zvar(P::_NZSepLP, f, i, idx)
    f == :w || return P.Z[f][i, idx]
    jj, s, ih = P.wpos[idx]
    @assert ih == i
    return P.O.v[:ζW][jj, s]
end

function _nz_rows(nd::NZData, Y, wpos)
    S, q = nd.S, nd.q_hat
    rows = _NZRow[]
    fams = haskey(Y, :c) ? (:a, :b, :r, :d, :e, :c) : (:a, :b, :r, :d, :e)
    for fam in fams, s in 1:S
        lo, hi = lower_bound(Y[fam][s]), upper_bound(Y[fam][s])
        push!(rows, _NZRow(-lo, [(fam, s, 1.0)], 0))
        push!(rows, _NZRow(hi, [(fam, s, -1.0)], 0))
    end
    for (yf, ef, ε) in ((:a, :b, nd.eps_hat), (:d, :e, nd.eps_tilde))
        for s in 1:S
            push!(rows, _NZRow(q[s], [(ef, s, 1.0), (yf, s, -1.0)], 0))
            push!(rows, _NZRow(-q[s], [(yf, s, 1.0), (ef, s, 1.0)], 0))
        end
        push!(rows, _NZRow(2ε, [(ef, s, -1.0) for s in 1:S], 0))
    end
    for s in 1:S
        push!(rows, _NZRow(0.0, [(:a, s, 1.0 / (1.0 - nd.beta)), (:r, s, -1.0)], 0))
    end
    if haskey(Y, :c)                                  # belief 결합: cpl − a + d ≥ 0, cpl + a − d ≥ 0, 2δ − Σcpl ≥ 0
        for s in 1:S
            push!(rows, _NZRow(0.0, [(:c, s, 1.0), (:a, s, -1.0), (:d, s, 1.0)], 0))
            push!(rows, _NZRow(0.0, [(:c, s, 1.0), (:a, s, 1.0), (:d, s, -1.0)], 0))
        end
        push!(rows, _NZRow(2nz_delta(nd), [(:c, s, -1.0) for s in 1:S], 0))
    end
    Hr = findall(nd.hrow .> 0)
    for (idx, (jj, s, i)) in enumerate(wpos)
        k = Hr[jj]
        push!(rows, _NZRow(0.0, [(:w, idx, 1.0)], i))                                  # ϖ ≥ 0
        isfinite(nd.varpiU[k]) &&
            push!(rows, _NZRow(0.0, [(:r, s, nd.varpiU[k]), (:w, idx, -1.0)], i))       # ϖᵁ r − ϖ ≥ 0
    end
    return rows
end

function _nz_build_sep(nd::NZData, x̄; optimizer)
    O = build_nz_omega(nd; optimizer=optimizer, mode=:relax)
    m = O.model; v = O.v
    set_optimizer_attribute(m, "Method", 1)          # dual simplex (행 추가 후 warm start)
    set_optimizer_attribute(m, "Presolve", 0)
    nh, S = nz_nh(nd), nd.S
    Hr = v[:Hr]
    Y = Dict{Symbol,Vector{VariableRef}}(f => vec(v[f]) for f in (:a, :b, :r, :d, :e))
    wpos = [(jj, s, nd.hrow[Hr[jj]]) for jj in eachindex(Hr) for s in 1:S]
    Y[:w] = [v[:ϖ][Hr[jj], s] for (jj, s, _) in wpos]
    newZ(name) = @variable(m, [1:nh, 1:S], lower_bound = 0.0, base_name = name)
    Z = Dict{Symbol,Matrix{VariableRef}}(:r => v[:ζL], :d => v[:ζF], :a => newZ("Za"), :b => newZ("Zb"), :e => newZ("Ze"))
    if v[:cpl] !== nothing                           # belief 결합 변수 cpl 과 그 곱 α_i cpl_s
        Y[:c] = vec(v[:cpl]); Z[:c] = newZ("Zc")
    end
    α = v[:α]
    for fam in (:a, :r, :d)                          # Σ_s y_s = 1
        @constraint(m, [i = 1:nh], sum(Z[fam][i, s] for s in 1:S) == α[i])
    end
    for fam in keys(Z), l in 1:size(nd.W, 1)         # (w_l − W_l α) y_s ≥ 0
        cs = @constraint(m, [s = 1:S], sum(nd.W[l, i] * Z[fam][i, s] for i in 1:nh) <= nd.wvec[l] * Y[fam][s])
        fam == :c && for s in 1:S; set_name(cs[s], "rltc_static_$(l)_$(s)"); end   # 진단용 이름 (결합 RLT)
    end
    # ζW = α_i ϖ 의 McCormick (α 상자 × 노드 ϖ 상자). 계수·rhs 는 _nz_sep_set_wbox! 가 노드마다 채움.
    # α 만 분기하면 α·ϖ 오차가 (α 폭) × (ϖ 범위 0 ~ p r) 에 비례해 θ·비용 스케일로 증폭됨 → ϖ 도 분기 (diag_box_width.jl)
    mcW = NTuple{4,ConstraintRef}[]
    Lw = zeros(length(wpos)); Uw = zeros(length(wpos))
    for (idx, (jj, s, i)) in enumerate(wpos)
        z = v[:ζW][jj, s]; w = Y[:w][idx]; a = α[i]
        c1 = @constraint(m, z - 0.0 * w - 0.0 * a >= 0.0); c2 = @constraint(m, z - 0.0 * w - 0.0 * a >= 0.0)
        c3 = @constraint(m, z - 0.0 * w - 0.0 * a <= 0.0); c4 = @constraint(m, z - 0.0 * w - 0.0 * a <= 0.0)
        push!(mcW, (c1, c2, c3, c4))
        Uw[idx] = has_upper_bound(w) ? upper_bound(w) : Inf
    end
    nz_set_objective!(O, nd, x̄)
    P = _NZSepLP(O, nh, S, Y, Z, wpos, _nz_rows(nd, Y, wpos),
                 Tuple{Int,Bool,Float64,ConstraintRef,Float64,Int}[], Bool[], zeros(nh), copy(nd.hU), mcW, Lw, Uw)
    _nz_sep_set_box!(P, zeros(nh), copy(nd.hU))      # 자리표시 계수 (0) 를 기본 상자의 McCormick 으로 채움
    _nz_sep_set_wbox!(P, copy(Lw), copy(Uw))
    return P
end

"노드의 ϖ 상자와 ζW McCormick 갱신 (α 상자 P.l, P.u 를 먼저 _nz_sep_set_box! 로 설정한 뒤 호출)"
function _nz_sep_set_wbox!(P::_NZSepLP, Lw, Uw)
    for (idx, (_, _, i)) in enumerate(P.wpos)
        w = P.Y[:w][idx]; a = P.O.v[:α][i]
        L, U = Lw[idx], Uw[idx]; la, ua = P.l[i], P.u[i]
        set_lower_bound(w, L); isfinite(U) && set_upper_bound(w, U)
        c1, c2, c3, c4 = P.mcW[idx]
        isfinite(U) || (U = 1e6)                    # ϖ 상한이 없으면 (H 행 밖) 위쪽 두 행은 사실상 비활성
        for (c, cw, ca, rhs) in ((c1, la, L, -la * L), (c2, ua, U, -ua * U), (c3, ua, L, -ua * L), (c4, la, U, -la * U))
            set_normalized_coefficient(c, w, -cw); set_normalized_coefficient(c, a, -ca); set_normalized_rhs(c, rhs)
        end
    end
    P.Lw .= Lw; P.Uw .= Uw
end

"""RLT 행 (i, j, side, b) 추가.  lo: (α_i − b)·g_j ≥ 0,  hi: (b − α_i)·g_j ≥ 0."""
function _nz_sep_add!(P::_NZSepLP, i, j, islo, b)
    row = P.rows[j]
    α = P.O.v[:α]
    Gy = sum(c * P.Y[f][idx] for (f, idx, c) in row.terms)
    GZ = row.c0 * α[i] + sum(c * _nz_zvar(P, f, i, idx) for (f, idx, c) in row.terms)
    cref = islo ? @constraint(P.O.model, GZ - b * Gy >= b * row.c0) :
                  @constraint(P.O.model, b * Gy - GZ >= -b * row.c0)
    # 진단용 이름: 결합 (cpl) 이 들어간 RLT 행 → "rltc_...", 나머지 → 이름 없음
    any(t -> t[1] == :c, row.terms) && set_name(cref, "rltc_$(i)_$(j)_$(islo ? "lo" : "hi")_$(length(P.allrows) + 1)")
    push!(P.allrows, (i, islo, b, cref, normalized_rhs(cref), j))
    push!(P.active, true)
end

function _nz_sep_set_box!(P::_NZSepLP, l, u)
    α = P.O.v[:α]
    for i in 1:P.nh
        set_lower_bound(α[i], l[i]); set_upper_bound(α[i], u[i])
    end
    for (n, (i, islo, b, cref, rhs0, _)) in enumerate(P.allrows)
        valid = islo ? (l[i] >= b - 1e-12) : (u[i] <= b + 1e-12)
        if valid && !P.active[n]
            set_normalized_rhs(cref, rhs0); P.active[n] = true
        elseif !valid && P.active[n]
            set_normalized_rhs(cref, -1e30); P.active[n] = false
        end
    end
    P.l .= l; P.u .= u
end

function _nz_sep_violations(P::_NZSepLP; tol=1e-6)
    αv = value.(P.O.v[:α])
    yv = Dict(f => value.(y) for (f, y) in P.Y)
    zv = Dict(f => value.(z) for (f, z) in P.Z)
    ζWv = value.(P.O.v[:ζW])
    zget(f, i, idx) = f == :w ? ζWv[P.wpos[idx][1], P.wpos[idx][2]] : zv[f][i, idx]
    out = Tuple{Float64,Int,Int,Bool}[]
    for (j, row) in enumerate(P.rows)
        Gy = row.c0
        for (f, idx, c) in row.terms; Gy += c * yv[f][idx]; end
        for i in (row.ionly > 0 ? (row.ionly:row.ionly) : (1:P.nh))
            GZ = row.c0 * αv[i]
            for (f, idx, c) in row.terms; GZ += c * zget(f, i, idx); end
            vlo = GZ - P.l[i] * Gy
            vhi = P.u[i] * Gy - GZ
            vlo < -tol && push!(out, (vlo, i, j, true))
            vhi < -tol && push!(out, (vhi, i, j, false))
        end
    end
    return out
end

"노드 LP 수치 오류 횟수 (재시도로 해결 / 끝내 실패)"
const _NZ_NUMERR = Threads.Atomic{Int}(0)
const _NZ_NUMERR_FAIL = Threads.Atomic{Int}(0)

"""
반환: 최적값, −Inf (infeasible), NaN (시간 제한), Inf (수치 오류가 재시도로도 안 풀림 → 호출 측이 부모 상한으로 분기).
NUMERICAL_ERROR 는 결합 (δ) 행을 넣은 RLT 완화에서 처음 관찰됨 (SGB128 pair, x=∅, δ=0.1. δ=Inf 는 정상).
재시도: NumericFocus=3 → 그다음 barrier (Method=2). 끝나면 원래 설정 (dual simplex, NumericFocus 0) 으로 되돌림.
"""
const _NZ_LPCOUNT = Threads.Atomic{Int}(0)
const _NZ_RESTORE = IdDict{Any,Bool}()       # 재시도 설정을 다음 optimize 전에 되돌릴 모델
const _NZ_RESTORE_LOCK = ReentrantLock()

function _nz_solve_lp!(model)
    # 직전 호출에서 재시도 설정을 바꿨으면 여기서 되돌림. 해를 읽은 뒤에 되돌리면 JuMP 가 해를 무효로 봐서
    # (OptimizeNotCalled) 이어지는 value 읽기 (RLT 위반 계산, α 읽기) 가 깨진다.
    restore = lock(() -> pop!(_NZ_RESTORE, model, false), _NZ_RESTORE_LOCK)
    restore && (set_optimizer_attribute(model, "NumericFocus", 0); set_optimizer_attribute(model, "Method", 1))
    optimize!(model)
    st = termination_status(model)
    if st == MOI.NUMERICAL_ERROR || st == MOI.OTHER_ERROR
        k = Threads.atomic_add!(_NZ_NUMERR, 1) + 1
        # 진단: NZ_NUMERR_DUMP=<폴더> 면 처음 3 개의 실패 LP 를 Gurobi 내부 모델 (.mps) 과 basis (.bas) 로 저장
        if haskey(ENV, "NZ_NUMERR_DUMP") && k <= 3
            o = JuMP.unsafe_backend(model); base = joinpath(ENV["NZ_NUMERR_DUMP"], "numerr_$k")
            try
                Gurobi.GRBwrite(o, base * ".mps"); Gurobi.GRBwrite(o, base * ".bas")
            catch e
                @warn "numerr dump 실패" e
            end
        end
        lock(() -> (_NZ_RESTORE[model] = true), _NZ_RESTORE_LOCK)
        for (nf, meth) in ((3, 1), (3, 2))
            set_optimizer_attribute(model, "NumericFocus", nf); set_optimizer_attribute(model, "Method", meth)
            optimize!(model)
            st = termination_status(model)
            st in (MOI.NUMERICAL_ERROR, MOI.OTHER_ERROR) || break
        end
        st in (MOI.NUMERICAL_ERROR, MOI.OTHER_ERROR) && (Threads.atomic_add!(_NZ_NUMERR_FAIL, 1); return Inf)
    end
    if st == MOI.OPTIMAL && haskey(ENV, "NZ_LP_DUMP_EVERY")      # 진단: 정상 노드 LP 를 N 번째마다 저장 (조건수 비교용)
        n = Threads.atomic_add!(_NZ_LPCOUNT, 1) + 1
        if n % parse(Int, ENV["NZ_LP_DUMP_EVERY"]) == 0 && n ÷ parse(Int, ENV["NZ_LP_DUMP_EVERY"]) <= 6
            o = JuMP.unsafe_backend(model); base = joinpath(ENV["NZ_NUMERR_DUMP"], "ok_$n")
            try Gurobi.GRBwrite(o, base * ".mps"); Gurobi.GRBwrite(o, base * ".bas") catch end
        end
    end
    st == MOI.OPTIMAL && return objective_value(model)
    st == MOI.INFEASIBLE && return -Inf
    st == MOI.TIME_LIMIT && return NaN
    error("nz_alpha_bnb LP: $st")
end

function _nz_sep_solve!(P::_NZSepLP; maxrounds=30, maxadd=5000, tol=1e-6, stall_tol=1e-6, stall_rounds=2)
    zprev = Inf; nstall = 0
    for _ in 1:maxrounds
        z = _nz_solve_lp!(P.O.model)
        (isnan(z) || isinf(z)) && return z
        V = _nz_sep_violations(P; tol=tol)
        isempty(V) && return z
        nstall = (zprev - z <= stall_tol * max(1.0, abs(z))) ? nstall + 1 : 0
        zprev = z
        nstall >= stall_rounds && return z
        sort!(V; by=first)
        for (_, i, j, islo) in V[1:min(maxadd, length(V))]
            _nz_sep_add!(P, i, j, islo, islo ? P.l[i] : P.u[i])
        end
    end
    return _nz_solve_lp!(P.O.model)
end


# =====================================================================
# α 고정 LP (정확한 값)
# =====================================================================
function nz_build_fixed_alpha(nd::NZData, x̄; optimizer)
    F = build_nz_omega(nd; optimizer=optimizer, mode=:fixed_alpha)
    nz_set_objective!(F, nd, x̄)
    return F
end

"""α 고정 Ω 값 (실현 가능해의 정확한 목적값). 시간 초과 → NaN, infeasible → −Inf."""
function nz_eval_alpha!(F::NZOmega, nd::NZData, α̂)
    v = F.v; nh, S = nz_nh(nd), nd.S
    Hr = v[:Hr]
    for i in 1:nh
        fix(v[:α][i], α̂[i]; force=true)
        for s in 1:S
            set_normalized_coefficient(v[:cL][i, s], v[:r][s], -α̂[i])
            set_normalized_coefficient(v[:cF][i, s], v[:d][s], -α̂[i])
        end
    end
    for (jj, k) in enumerate(Hr), s in 1:S
        set_normalized_coefficient(v[:cW][jj, s], v[:ϖ][k, s], -α̂[nd.hrow[k]])
    end
    z = _nz_solve_lp!(F.model)
    return z == Inf ? NaN : z          # 수치 오류 (Inf) 는 "값 모름" (NaN) 으로: 실현 가능해 값으로 쓰면 안 됨
end

"H = {α ≥ 0 : Wα ≤ w} 로 사영 (W ≥ 0 가정, 비례 축소)"
function _nz_capH(a, nd::NZData)
    b = clamp.(a, 0.0, nd.hU)
    for l in 1:size(nd.W, 1)
        t = dot(nd.W[l, :], b)
        t > nd.wvec[l] && (b .*= nd.wvec[l] / t)
    end
    return b
end

"""Ω 를 Ipopt 로 로컬 최적화. 출발점 = α̂ 고정 LP 해. 반환 로컬 해의 α (없으면 nothing)."""
function nz_ipopt_local_alpha(nd::NZData, x̄, F::NZOmega, α̂; max_time=60.0, deadline=Inf)
    z = nz_eval_alpha!(F, nd, α̂)
    (isnan(z) || z == -Inf) && return nothing
    L = build_nz_omega(nd; optimizer=Ipopt.Optimizer, mode=:global, solver_params=false)
    nz_set_objective!(L, nd, x̄)
    set_optimizer_attribute(L.model, "max_wall_time", max(max_time, 1.0))
    set_optimizer_attribute(L.model, "print_level", 0)
    set_optimizer_attribute(L.model, "warm_start_init_point", "yes")
    set_optimizer_attribute(L.model, "mu_init", 1e-4)
    MOI.set(L.model, Ipopt.CallbackFunction(), (args...) -> time() < deadline)
    for (name, X) in L.v
        haskey(F.v, name) || continue
        X isa VariableRef && (set_start_value(X, value(F.v[name])); continue)
        X isa AbstractArray{VariableRef} || continue
        for I in eachindex(X); set_start_value(X[I], value(F.v[name][I])); end
    end
    # ŷ 등 F.v 에 등록되지 않은 변수도 이름으로 넘김
    for X in all_variables(L.model)
        has_start_value(X) && continue
        Y = variable_by_name(F.model, name(X))
        Y === nothing || set_start_value(X, value(Y))
    end
    optimize!(L.model)
    has_values(L.model) || return nothing
    return _nz_capH(value.(L.v[:α]), nd)
end


# =====================================================================
# Gurobi Env 풀: WLS 라이선스는 Env 하나가 세션 하나라, 호출마다 Env 를 새로 만들면 세션 한도를 넘는다
# (Error 10009). Env 를 프로세스 안에서 한 번만 만들고 재사용한다.
# =====================================================================
const _NZ_ENV_POOL = Gurobi.Env[]
const _NZ_ENV_LOCK = ReentrantLock()
function _nz_envs(n)
    lock(_NZ_ENV_LOCK) do
        while length(_NZ_ENV_POOL) < n
            push!(_NZ_ENV_POOL, Gurobi.Env())
        end
        return _NZ_ENV_POOL[1:n]
    end
end


# =====================================================================
# 주 함수
# =====================================================================
"""
    nz_alpha_bnb(nd, x̄; nworkers, time_limit, rel_gap, target, ...)

target: LB ≥ target 이거나 UB ≤ target 이면 즉시 종료 (Benders "t₀ 보다 큰 값이 있는가" 판정용).
반환 Dict: :LB (incumbent 값), :UB (dual bound), :α (incumbent α), :is_exact, :nodes, :time, :root_UB, :ipopt_calls
"""
function nz_alpha_bnb(nd::NZData, x̄; nworkers=Threads.nthreads() - 2, time_limit=300.0, rel_gap=1e-4,
                      dive_max=3, maxrounds=30, min_width=1e-7, ipopt_time=60.0,
                      verbose=true, log_every=30.0, target=nothing, envs=nothing, heuristic::Bool=true,
                      presolve::Bool=true, varpi_branch::Bool=false, varpi_alpha_width=3.0,
                      branch_score::Symbol=:viol, branch_point::Symbol=:alphahat)
    # branch_score: :viol (곱 위반 합, 기존) | :weighted (위반 × 목적 민감도: ζL·ζF 는 해당 흐름 행의 쌍대값 × hcoef,
    #               ζW 는 목적계수 θ hcoef). branch_point: :alphahat (완화 해 α̂, 끝 10% 면 중점, 기존) | :mid (항상 중점)
    branch_score in (:viol, :weighted) || error("branch_score = :viol | :weighted")
    branch_point in (:alphahat, :mid) || error("branch_point = :alphahat | :mid")   # ϖ 분기: 시험 결과 더 나빠 기본 끔 (experiments.md)
    # 프리솔브: x̄ 에서 최적 반응이 항상 0 인 예약 좌표를 고정 (nz_presolve_hU). 내부 모델은 줄인 상자로 만들고,
    # 반환 α 는 원래 정의역의 점이다 (호출자는 원래 nd 로 cut 을 만듦).
    nfix = 0
    if presolve
        hUp, nfix = nz_presolve_hU(nd, x̄)
        nfix > 0 && (nd = nz_with_hU(nd, hUp))
    end
    nworkers >= 1 || error("nz_alpha_bnb: nworkers ≥ 1 필요 (Julia 를 -t (nworkers+2),1 로 실행)")
    Threads.nthreads(:interactive) >= 1 ||
        error("nz_alpha_bnb: interactive 스레드가 필요합니다. julia -t $(nworkers + 2),1 로 실행하세요")
    t0 = time(); t_end = t0 + time_limit
    nh, S = nz_nh(nd), nd.S
    Hr = findall(nd.hrow .> 0)
    mk(env) = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true);
                     MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
    # envs: 호출자가 Env 를 넘길 수 있음 (worker 당 하나 + heuristic 용 하나, 동시 사용 금지).
    #   학술 WLS 라이선스 (동시 세션 2 개) 에서 로컬 검증할 때만 쓰는 용도. 기본은 풀에서 nworkers+1 개.
    envs = envs === nothing ? _nz_envs(nworkers + 1) : envs
    length(envs) >= nworkers + (heuristic ? 1 : 0) || error("nz_alpha_bnb: envs 가 부족함 (worker $nworkers + heuristic $(heuristic))")
    Ps = [_nz_build_sep(nd, x̄; optimizer=mk(envs[i])) for i in 1:nworkers]
    Fs = [nz_build_fixed_alpha(nd, x̄; optimizer=mk(envs[i])) for i in 1:nworkers]
    tol(v) = rel_gap * (isfinite(v) ? max(1.0, abs(v)) : 1.0)
    boxok(l) = all(nd.W * l .<= nd.wvec .+ 1e-9)

    lk = ReentrantLock()
    Lw0 = copy(Ps[1].Lw); Uw0 = copy(Ps[1].Uw)
    wmax = [isfinite(U) ? max(U, 1e-9) : 1.0 for U in Uw0]       # ϖ 정규화 폭
    open = Any[(zeros(nh), copy(nd.hU), Inf, Lw0, Uw0)]
    inflight = fill(-Inf, nworkers)
    LB = Ref(-Inf); best_α = Ref(zeros(nh))
    nodes = Ref(0); done = Ref(false); root_UB = Ref(Inf)
    root_rows = Ref{Any}(nothing)
    node_cands = Tuple{Float64,Vector{Float64}}[]
    ipopt_calls = Ref(0)
    pruned_ub = Ref(-Inf)          # 허용오차로 버린 노드의 상한 (보고 UB 에 포함해야 유효)

    globalUB() = max(LB[], isempty(open) ? -Inf : maximum(n[3] for n in open), maximum(inflight), pruned_ub[])
    offer!(z, a) = lock(lk) do
        z > LB[] && (LB[] = z; best_α[] = copy(a))
    end

    function worker(wid)
        P = Ps[wid]; F = Fs[wid]
        localnode = nothing; dive_left = 0; synced = false
        while true
            yield()
            node = nothing
            lock(lk)
            try
                ub_now = globalUB()
                if time() >= t_end || ub_now - LB[] <= tol(LB[]) ||
                   (target !== nothing && (LB[] >= target || ub_now <= target))
                    done[] = true
                end
                if done[]
                    localnode !== nothing && push!(open, localnode)
                    return
                end
                if localnode !== nothing && dive_left > 0 && localnode[3] > LB[] + tol(LB[])
                    node = localnode; dive_left -= 1
                else
                    if localnode !== nothing
                        localnode[3] > LB[] + tol(LB[]) ? push!(open, localnode) :
                                                          (pruned_ub[] = max(pruned_ub[], localnode[3]))
                    end
                    if !isempty(open)
                        node = popat!(open, argmax([n[3] for n in open])); dive_left = dive_max
                    end
                end
                localnode = nothing
                node !== nothing && (inflight[wid] = node[3])
            finally
                unlock(lk)
            end
            if node === nothing
                sleep(0.05); continue
            end
            l, u, ubp, Lw, Uw = node
            if !synced && root_rows[] !== nothing && isempty(P.allrows) && t_end - time() > 30.0
                for (i, j, islo, b) in root_rows[]; _nz_sep_add!(P, i, j, islo, b); end
                synced = true
            end
            z = -Inf; α̂ = nothing; children = nothing; zE = -Inf; unfinished = false
            pruned_val = -Inf
            boxok(l) && ubp <= LB[] + tol(LB[]) && (pruned_val = ubp)
            if ubp > LB[] + tol(LB[]) && boxok(l)
                _nz_sep_set_box!(P, l, u)
                _nz_sep_set_wbox!(P, Lw, Uw)
                set_time_limit_sec(P.O.model, max(t_end - time(), 0.1))
                z = _nz_sep_solve!(P; maxrounds=maxrounds)
                unfinished = isnan(z)
                numfail = z == Inf               # 수치 오류가 재시도로도 안 풀림: 부모 상한 유지, 가장 넓은 α_i 를 반으로
                if numfail
                    z = ubp
                    i = argmax(u .- l)
                    if u[i] - l[i] > min_width
                        t = 0.5 * (l[i] + u[i]); u1 = copy(u); u1[i] = t; l2 = copy(l); l2[i] = t
                        children = ((copy(l), u1, z, Lw, Uw), (l2, copy(u), z, Lw, Uw))
                    else
                        pruned_val = z           # 더 못 나누면 상한만 남김 (유효)
                    end
                end
                z = unfinished ? ubp : min(z, ubp)
                (!unfinished && !numfail && z <= LB[] + tol(LB[])) && (pruned_val = z)
                if !unfinished && !numfail && z > LB[] + tol(LB[])
                    v = P.O.v
                    α̂ = clamp.(value.(v[:α]), l, u)
                    rv = value.(v[:r]); dv = value.(v[:d]); ϖv = value.(v[:ϖ])
                    ζLv = value.(v[:ζL]); ζFv = value.(v[:ζF]); ζWv = value.(v[:ζW])
                    set_time_limit_sec(F.model, max(t_end - time(), 0.1))
                    zE = nz_eval_alpha!(F, nd, α̂)
                    isnan(zE) && (zE = -Inf)
                    viol = zeros(nh)
                    if branch_score == :weighted
                        mL = P.O.model[:Lflow]; mF = P.O.model[:Fflow]
                        θw = nd.thetaU
                    end
                    for i in 1:nh
                        u[i] - l[i] > min_width || continue
                        if branch_score == :viol
                            viol[i] = sum(abs(ζLv[i, s] - α̂[i] * rv[s]) + abs(ζFv[i, s] - α̂[i] * dv[s]) for s in 1:S)
                        end
                        for (jj, k) in enumerate(Hr)
                            nd.hrow[k] == i || continue
                            if branch_score == :viol
                                viol[i] += sum(abs(ζWv[jj, s] - α̂[i] * ϖv[k, s]) for s in 1:S)
                            else                  # 목적 민감도 가중: 흐름 행 k 의 쌍대값 × hcoef × 위반 + θ hcoef × ζW 위반
                                hc = abs(nd.hcoef[k])
                                viol[i] += sum(abs(dual(mL[k, s])) * hc * abs(ζLv[i, s] - α̂[i] * rv[s]) +
                                               abs(dual(mF[k, s])) * hc * abs(ζFv[i, s] - α̂[i] * dv[s]) +
                                               θw * hc * abs(ζWv[jj, s] - α̂[i] * ϖv[k, s]) for s in 1:S)
                            end
                        end
                    end
                    # ϖ 분기 후보: 위반이 가장 큰 ζW 곱에서 ϖ 의 정규화 폭이 α 의 정규화 폭보다 크면 ϖ 를 나눔
                    wbr = 0
                    if varpi_branch
                        best_w = 0.0
                        for (idx, (jj, s, i)) in enumerate(P.wpos)
                            Uw[idx] - Lw[idx] > 1e-6 * wmax[idx] || continue
                            vw = abs(ζWv[jj, s] - α̂[i] * ϖv[Hr[jj], s])
                            # α 상자가 이미 좁을 때만 ϖ 를 나눔 (넓은 α 에서 ϖ 를 나누면 트리만 커짐: diag_box_width.jl, 1차 시도)
                            u[i] - l[i] <= varpi_alpha_width || continue
                            vw > best_w && (best_w = vw; wbr = idx)
                        end
                    end
                    i = argmax(viol)
                    if wbr > 0
                        jj, s, _ = P.wpos[wbr]
                        t = ϖv[Hr[jj], s]; width = Uw[wbr] - Lw[wbr]
                        (t - Lw[wbr] < 0.1 * width || Uw[wbr] - t < 0.1 * width) && (t = 0.5 * (Lw[wbr] + Uw[wbr]))
                        U1 = copy(Uw); U1[wbr] = t; L2 = copy(Lw); L2[wbr] = t
                        children = ((copy(l), copy(u), z, copy(Lw), U1), (copy(l), copy(u), z, L2, copy(Uw)))
                    elseif viol[i] > 1e-9
                        width = u[i] - l[i]; t = α̂[i]
                        (branch_point == :mid || t - l[i] < 0.1 * width || u[i] - t < 0.1 * width) && (t = 0.5 * (l[i] + u[i]))
                        u1 = copy(u); u1[i] = t; l2 = copy(l); l2[i] = t
                        children = ((copy(l), u1, z, Lw, Uw), (l2, copy(u), z, Lw, Uw))
                    else
                        zE = max(zE, z)      # 완화가 exact
                    end
                end
            end
            lock(lk)
            try
                if unfinished
                    push!(open, node); inflight[wid] = -Inf; dive_left = 0
                    continue
                end
                nodes[] += 1
                pruned_ub[] = max(pruned_ub[], pruned_val)
                if ubp == Inf
                    root_UB[] = z
                    root_rows[] = [(r[1], r[6], r[2], r[3]) for r in P.allrows]
                end
                zE > LB[] && (LB[] = zE; best_α[] = copy(α̂))
                if α̂ !== nothing
                    push!(node_cands, (z, copy(α̂)))
                    length(node_cands) > 200 && popfirst!(node_cands)
                end
                if children !== nothing
                    push!(open, children[2]); localnode = children[1]
                else
                    dive_left = 0
                end
                inflight[wid] = -Inf
            finally
                unlock(lk)
            end
        end
    end

    function heuristic_task()
        Fh = nz_build_fixed_alpha(nd, x̄; optimizer=mk(envs[end]))
        tried = Set{Vector{Float64}}()
        a0 = _nz_capH(fill(minimum(nd.wvec) / nh, nh), nd)
        while !done[] && t_end - time() > 15.0
            push!(tried, round.(a0; digits=3))
            aloc = nz_ipopt_local_alpha(nd, x̄, Fh, a0; max_time=min(ipopt_time, t_end - time()), deadline=t_end)
            ipopt_calls[] += 1
            if aloc !== nothing
                z = nz_eval_alpha!(Fh, nd, aloc)
                isnan(z) || offer!(z, aloc)
            end
            a0 = nothing
            while a0 === nothing && !done[]
                lock(lk)
                try
                    best = -Inf
                    for (zc, a) in node_cands
                        (zc > best && !(round.(a; digits=3) in tried)) && (best = zc; a0 = a)
                    end
                finally
                    unlock(lk)
                end
                a0 === nothing && sleep(0.5)
            end
            a0 === nothing && break
            a0 = _nz_capH(a0, nd)
        end
    end

    tasks = [Threads.@spawn worker(i) for i in 1:nworkers]
    heuristic && push!(tasks, Threads.@spawn heuristic_task())
    t_log = time()
    while true
        sleep(0.2)
        fin = false; UBn = Inf
        lock(lk)
        try
            UBn = globalUB()
            idle = isempty(open) && all(inflight .== -Inf)
            fin = (time() >= t_end) || (UBn - LB[] <= tol(LB[])) || (idle && nodes[] > 0) ||
                  (target !== nothing && (LB[] >= target || UBn <= target))
            fin && (done[] = true)
        finally
            unlock(lk)
        end
        if verbose && time() - t_log > log_every
            @printf("    [nz-α-B&B] t=%6.1fs nodes=%5d open=%5d LB=%.6f UB=%.6f gap=%.3e\n",
                    time() - t0, nodes[], length(open), LB[], UBn, (UBn - LB[]) / max(1.0, abs(LB[])))
            flush(stdout); t_log = time()
        end
        fin && break
    end
    foreach(wait, tasks)
    UB = max(LB[], isempty(open) ? -Inf : maximum(n[3] for n in open), pruned_ub[])
    return Dict(:LB => LB[], :UB => UB, :α => best_α[], :is_exact => (UB - LB[] <= tol(LB[])),
                :nodes => nodes[], :time => time() - t0, :root_UB => root_UB[], :ipopt_calls => ipopt_calls[], :presolve_fixed => nfix,
                :numerr => _NZ_NUMERR[], :numerr_fail => _NZ_NUMERR_FAIL[])
end
