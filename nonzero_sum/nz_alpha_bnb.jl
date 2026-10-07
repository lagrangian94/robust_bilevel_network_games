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
    for fam in (:a, :b, :r, :d, :e), s in 1:S
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
    α = v[:α]
    for fam in (:a, :r, :d)                          # Σ_s y_s = 1
        @constraint(m, [i = 1:nh], sum(Z[fam][i, s] for s in 1:S) == α[i])
    end
    for fam in (:a, :b, :r, :d, :e), l in 1:size(nd.W, 1)   # (w_l − W_l α) y_s ≥ 0
        @constraint(m, [s = 1:S], sum(nd.W[l, i] * Z[fam][i, s] for i in 1:nh) <= nd.wvec[l] * Y[fam][s])
    end
    nz_set_objective!(O, nd, x̄)
    return _NZSepLP(O, nh, S, Y, Z, wpos, _nz_rows(nd, Y, wpos),
                    Tuple{Int,Bool,Float64,ConstraintRef,Float64,Int}[], Bool[], zeros(nh), copy(nd.hU))
end

"""RLT 행 (i, j, side, b) 추가.  lo: (α_i − b)·g_j ≥ 0,  hi: (b − α_i)·g_j ≥ 0."""
function _nz_sep_add!(P::_NZSepLP, i, j, islo, b)
    row = P.rows[j]
    α = P.O.v[:α]
    Gy = sum(c * P.Y[f][idx] for (f, idx, c) in row.terms)
    GZ = row.c0 * α[i] + sum(c * _nz_zvar(P, f, i, idx) for (f, idx, c) in row.terms)
    cref = islo ? @constraint(P.O.model, GZ - b * Gy >= b * row.c0) :
                  @constraint(P.O.model, b * Gy - GZ >= -b * row.c0)
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

function _nz_solve_lp!(model)
    optimize!(model)
    st = termination_status(model)
    st == MOI.OPTIMAL && return objective_value(model)
    st == MOI.INFEASIBLE && return -Inf
    st == MOI.TIME_LIMIT && return NaN
    error("nz_alpha_bnb LP: $st")
end

function _nz_sep_solve!(P::_NZSepLP; maxrounds=30, maxadd=5000, tol=1e-6, stall_tol=1e-6, stall_rounds=2)
    zprev = Inf; nstall = 0
    for _ in 1:maxrounds
        z = _nz_solve_lp!(P.O.model)
        (isnan(z) || z == -Inf) && return z
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
    return _nz_solve_lp!(F.model)
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
                      verbose=true, log_every=30.0, target=nothing, envs=nothing, heuristic::Bool=true)
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
    open = Any[(zeros(nh), copy(nd.hU), Inf)]
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
            l, u, ubp = node
            if !synced && root_rows[] !== nothing && isempty(P.allrows) && t_end - time() > 30.0
                for (i, j, islo, b) in root_rows[]; _nz_sep_add!(P, i, j, islo, b); end
                synced = true
            end
            z = -Inf; α̂ = nothing; children = nothing; zE = -Inf; unfinished = false
            pruned_val = -Inf
            boxok(l) && ubp <= LB[] + tol(LB[]) && (pruned_val = ubp)
            if ubp > LB[] + tol(LB[]) && boxok(l)
                _nz_sep_set_box!(P, l, u)
                set_time_limit_sec(P.O.model, max(t_end - time(), 0.1))
                z = _nz_sep_solve!(P; maxrounds=maxrounds)
                unfinished = isnan(z)
                z = unfinished ? ubp : min(z, ubp)
                (!unfinished && z <= LB[] + tol(LB[])) && (pruned_val = z)
                if !unfinished && z > LB[] + tol(LB[])
                    v = P.O.v
                    α̂ = clamp.(value.(v[:α]), l, u)
                    rv = value.(v[:r]); dv = value.(v[:d]); ϖv = value.(v[:ϖ])
                    ζLv = value.(v[:ζL]); ζFv = value.(v[:ζF]); ζWv = value.(v[:ζW])
                    set_time_limit_sec(F.model, max(t_end - time(), 0.1))
                    zE = nz_eval_alpha!(F, nd, α̂)
                    isnan(zE) && (zE = -Inf)
                    viol = zeros(nh)
                    for i in 1:nh
                        u[i] - l[i] > min_width || continue
                        viol[i] = sum(abs(ζLv[i, s] - α̂[i] * rv[s]) + abs(ζFv[i, s] - α̂[i] * dv[s]) for s in 1:S)
                        for (jj, k) in enumerate(Hr)
                            nd.hrow[k] == i || continue
                            viol[i] += sum(abs(ζWv[jj, s] - α̂[i] * ϖv[k, s]) for s in 1:S)
                        end
                    end
                    i = argmax(viol)
                    if viol[i] > 1e-9
                        width = u[i] - l[i]; t = α̂[i]
                        (t - l[i] < 0.1 * width || u[i] - t < 0.1 * width) && (t = 0.5 * (l[i] + u[i]))
                        u1 = copy(u); u1[i] = t; l2 = copy(l); l2[i] = t
                        children = ((copy(l), u1, z), (l2, copy(u), z))
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
                :nodes => nodes[], :time => time() - t0, :root_UB => root_UB[], :ipopt_calls => ipopt_calls[])
end
