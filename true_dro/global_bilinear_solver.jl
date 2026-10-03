"""
global_bilinear_solver.jl — Ω subproblem V*(x̄) 의 전역 해법: α-공간 branch-and-bound.

Ω (build_true_dro_subproblem) 의 bilinear 항은 전부 α_k × (r_s | a_s | d_s).
  - α 고정 → Ω 는 LP (정확한 값, 실현 가능해 → valid LB)
  - α 상자 [l,u] 위 RLT 완화 → LP (valid UB), 상자가 점으로 줄면 exact
→ α (K 차원, S 와 무관) 만 분기하는 spatial B&B.

노드 완화 (level-1 RLT):
  belief 다면체 제약 g_j(y) ≥ 0 (상·하한, TV 볼 a−b≤q̂, a+b≥q̂, Σb≤2ε̂, d·e 동일, CVaR r≤a/(1−β))
  × 상자 인수 (α_k − l_k) ≥ 0, (u_k − α_k) ≥ 0  을 곱 변수 Z = α·y 로 선형화.
  + 정적 RLT  Σ_s Z_ks = α_k (Σy=1),  Σ_k Z_ks ≤ w y_s (Σα≤w).
  TV 볼 RLT 가 "초과 질량 합 ≤ 2ε" 를 보존 → root gap 이 S 에 비례해 커지지 않음.
  RLT 행은 전부 넣지 않고 위반된 것만 분리 (root 에서 dual≠0 인 행은 ~2%).
  부모 상자에서 만든 행은 자식에서도 유효 → 행은 지우지 않고 (basis 유지), 무효 행은 rhs=−∞ 로 비활성.

탐색: worker 병렬 (worker 마다 독립 Gurobi env/LP), best-first + worker 별 diving.
분기: McCormick 위반 Σ_s |Z_ks − α̂_k y_s| 이 가장 큰 α_k, 분할점 α̂_k.
LB (primal heuristic): 노드 α̂ 고정 LP + Ipopt 로컬 최적화 (출발점 = 상한 높은 노드의 α̂).
  로컬 해의 α 를 고정해 정확한 LP 로 재평가한 값만 LB 로 사용 (NLP 허용오차와 무관).

사용:
    r = global_bilinear_solve(td, x̄; nworkers=12, time_limit=300.0, rel_gap=5e-3)
    r[:LB], r[:UB], r[:α]
Julia 는 `-t (nworkers + 2)` 이상으로 실행.
"""

using JuMP, Printf
import Gurobi
using Ipopt


# =====================================================================
# belief 다면체 행  g(y) = c0 + Σ c·Y[fam][s] ≥ 0
# =====================================================================
struct _BeliefRow
    c0::Float64
    terms::Vector{Tuple{Symbol,Int,Float64}}    # (family, s, coef)
end

function _belief_rows(td, Y)
    S, q = td.S, td.q_hat
    rows = _BeliefRow[]
    for fam in keys(Y), s in 1:S                                     # 상·하한 (McCormick 에 해당)
        lo, hi = lower_bound(Y[fam][s]), upper_bound(Y[fam][s])
        push!(rows, _BeliefRow(-lo, [(fam, s, 1.0)]))                 # y − lo ≥ 0
        push!(rows, _BeliefRow(hi, [(fam, s, -1.0)]))                 # hi − y ≥ 0
    end
    for (yf, ef, ε) in ((:a, :b, td.eps_hat), (:d, :e, td.eps_tilde))   # TV 볼
        for s in 1:S
            push!(rows, _BeliefRow(q[s], [(ef, s, 1.0), (yf, s, -1.0)]))    # q + e − y ≥ 0
            push!(rows, _BeliefRow(-q[s], [(yf, s, 1.0), (ef, s, 1.0)]))    # y + e − q ≥ 0
        end
        push!(rows, _BeliefRow(2ε, [(ef, s, -1.0) for s in 1:S]))           # 2ε − Σe ≥ 0
    end
    if haskey(Y, :r)                                                  # CVaR: a/(1−β) − r ≥ 0
        for s in 1:S
            push!(rows, _BeliefRow(0.0, [(:a, s, 1.0 / (1.0 - td.beta)), (:r, s, -1.0)]))
        end
    end
    return rows
end


# =====================================================================
# 노드 LP: Ω − ζ 정의 + 곱 변수 + 정적 RLT + 분리된 RLT 행
# =====================================================================
mutable struct _SepLP
    model::Model
    vars::Dict
    K::Int
    S::Int
    lead::Vector{VariableRef}          # r (CVaR) 또는 a
    fol::Vector{VariableRef}           # d
    Y::Dict{Symbol,Vector{VariableRef}}
    Z::Dict{Symbol,Matrix{VariableRef}}
    rows::Vector{_BeliefRow}
    allrows::Vector{Tuple{Int,Bool,Float64,ConstraintRef,Float64,Int}}   # (k, islo, b, 행, 원래 rhs, j)
    active::Vector{Bool}
    l::Vector{Float64}
    u::Vector{Float64}
end

function _build_sep_lp(td, x̄; optimizer)
    model, vars = build_true_dro_subproblem(td, x̄; optimizer=optimizer, silent=true)
    for name in (:ζL_def, :ζF_def)
        delete.(model, model[name]); unregister(model, name)
    end
    set_optimizer_attribute(model, "Method", 1)       # dual simplex (행 추가 후 warm start)
    set_optimizer_attribute(model, "Presolve", 0)     # 재풀이 warm start 유지
    K, S, w = td.num_arcs, td.S, td.w
    α, ζL, ζF = vars[:α], vars[:ζL], vars[:ζF]
    cvar = vars[:_use_cvar]
    lead = cvar ? vec(vars[:r]) : vec(vars[:a])
    fol = vec(vars[:d])
    Y = Dict{Symbol,Vector{VariableRef}}(:a => vec(vars[:a]), :b => vec(vars[:b]), :d => fol, :e => vec(vars[:e]))
    Z = Dict{Symbol,Matrix{VariableRef}}(:d => ζF)
    newZ(name) = @variable(model, [1:K, 1:S], lower_bound = 0.0, base_name = name)
    Z[:b] = newZ("Zb"); Z[:e] = newZ("Ze")
    if cvar
        Y[:r] = lead; Z[:r] = ζL; Z[:a] = newZ("Za")
    else
        Z[:a] = ζL
    end
    for fam in (:a, :r, :d)                            # 정적 RLT
        haskey(Y, fam) || continue
        @constraint(model, [k = 1:K], sum(Z[fam][k, s] for s in 1:S) == α[k])
        @constraint(model, [s = 1:S], sum(Z[fam][k, s] for k in 1:K) <= w * Y[fam][s])
    end
    return _SepLP(model, vars, K, S, lead, fol, Y, Z, _belief_rows(td, Y),
                  Tuple{Int,Bool,Float64,ConstraintRef,Float64,Int}[], Bool[], zeros(K), fill(w, K))
end

"""RLT 행 (k, j, side, b) 추가.  lo: (α_k − b)·g_j ≥ 0,  hi: (b − α_k)·g_j ≥ 0."""
function _sep_add!(P::_SepLP, k, j, islo, b)
    row = P.rows[j]
    α = P.vars[:α]
    Gy = sum(c * P.Y[f][s] for (f, s, c) in row.terms)
    GZ = row.c0 * α[k] + sum(c * P.Z[f][k, s] for (f, s, c) in row.terms)
    cref = islo ? @constraint(P.model, GZ - b * Gy >= b * row.c0) :
                  @constraint(P.model, b * Gy - GZ >= -b * row.c0)
    push!(P.allrows, (k, islo, b, cref, normalized_rhs(cref), j))
    push!(P.active, true)
end

"""상자 (l,u) 로 이동. 무효 행은 삭제 대신 rhs=−∞ (basis 유지), 다시 유효해지면 복원."""
function _sep_set_box!(P::_SepLP, l, u)
    α = P.vars[:α]
    for k in 1:P.K
        set_lower_bound(α[k], l[k]); set_upper_bound(α[k], u[k])
    end
    for (i, (k, islo, b, cref, rhs0, _)) in enumerate(P.allrows)
        valid = islo ? (l[k] >= b - 1e-12) : (u[k] <= b + 1e-12)
        if valid && !P.active[i]
            set_normalized_rhs(cref, rhs0); P.active[i] = true
        elseif !valid && P.active[i]
            set_normalized_rhs(cref, -1e30); P.active[i] = false
        end
    end
    P.l .= l; P.u .= u
end

"""현재 LP 해에서 노드 상자 기준 RLT 위반 [(viol, k, j, islo)]."""
function _sep_violations(P::_SepLP; tol=1e-6)
    K = P.K
    αv = value.(P.vars[:α])
    yv = Dict(f => value.(y) for (f, y) in P.Y)
    zv = Dict(f => value.(z) for (f, z) in P.Z)
    out = Tuple{Float64,Int,Int,Bool}[]
    GZ = zeros(K)
    for (j, row) in enumerate(P.rows)
        Gy = row.c0
        for (f, s, c) in row.terms; Gy += c * yv[f][s]; end
        GZ .= row.c0 .* αv
        for (f, s, c) in row.terms, k in 1:K
            GZ[k] += c * zv[f][k, s]
        end
        for k in 1:K
            vlo = GZ[k] - P.l[k] * Gy
            vhi = P.u[k] * Gy - GZ[k]
            vlo < -tol && push!(out, (vlo, k, j, true))
            vhi < -tol && push!(out, (vhi, k, j, false))
        end
    end
    return out
end

"""LP 값. infeasible → −Inf, 시간 초과 → NaN (호출 측에서 미처리로 다룸)."""
function _solve_lp!(model)
    optimize!(model)
    st = termination_status(model)
    st == MOI.OPTIMAL && return objective_value(model)
    st == MOI.INFEASIBLE && return -Inf
    st == MOI.TIME_LIMIT && return NaN
    error("global_bilinear_solver LP: $st")
end

"""분리 루프. 반환 LP 값 (어느 시점에 멈춰도 valid UB). 정체 시 (상대 개선 < stall_tol, stall_rounds 연속) 중단."""
function _sep_solve!(P::_SepLP; maxrounds=30, maxadd=5000, tol=1e-6, stall_tol=1e-6, stall_rounds=2)
    zprev = Inf; nstall = 0
    for _ in 1:maxrounds
        z = _solve_lp!(P.model)
        (isnan(z) || z == -Inf) && return z
        V = _sep_violations(P; tol=tol)
        isempty(V) && return z
        nstall = (zprev - z <= stall_tol * max(1.0, abs(z))) ? nstall + 1 : 0
        zprev = z
        nstall >= stall_rounds && return z
        sort!(V; by=first)
        for (_, k, j, islo) in V[1:min(maxadd, length(V))]
            _sep_add!(P, k, j, islo, islo ? P.l[k] : P.u[k])
        end
    end
    return _solve_lp!(P.model)
end


# =====================================================================
# α 고정 LP (정확한 값): ζL_ks = α_k·lead_s, ζF_ks = α_k·d_s 를 선형 등식으로
# =====================================================================
mutable struct _FixedAlphaLP
    model::Model
    vars::Dict
    cL::Matrix{ConstraintRef}
    cF::Matrix{ConstraintRef}
end

function _build_fixed_alpha_lp(td, x̄; optimizer)
    model, vars = build_true_dro_subproblem(td, x̄; optimizer=optimizer, silent=true)
    for name in (:ζL_def, :ζF_def)
        delete.(model, model[name]); unregister(model, name)
    end
    K, S = td.num_arcs, td.S
    lead = vars[:_use_cvar] ? vec(vars[:r]) : vec(vars[:a])
    fol = vec(vars[:d])
    cL = Matrix{ConstraintRef}(undef, K, S); cF = Matrix{ConstraintRef}(undef, K, S)
    for k in 1:K, s in 1:S
        cL[k, s] = @constraint(model, vars[:ζL][k, s] - 0.0 * lead[s] == 0)
        cF[k, s] = @constraint(model, vars[:ζF][k, s] - 0.0 * fol[s] == 0)
    end
    return _FixedAlphaLP(model, vars, cL, cF)
end

"""α 고정 Ω 값 (실현 가능해의 정확한 목적값). 시간 초과 → NaN."""
function _eval_alpha!(F::_FixedAlphaLP, α̂)
    K, S = size(F.cL)
    lead = F.vars[:_use_cvar] ? vec(F.vars[:r]) : vec(F.vars[:a])
    fol = vec(F.vars[:d])
    α = F.vars[:α]
    for k in 1:K
        fix(α[k], α̂[k]; force=true)
        for s in 1:S
            set_normalized_coefficient(F.cL[k, s], lead[s], -α̂[k])
            set_normalized_coefficient(F.cF[k, s], fol[s], -α̂[k])
        end
    end
    return _solve_lp!(F.model)
end


# =====================================================================
# Ipopt 로컬 최적화 (primal heuristic)
# =====================================================================
_capw(a, w) = (b = max.(a, 0.0); s = sum(b); s > w ? b .* (w / s) : b)

"""Ω 를 Ipopt 로 로컬 최적화. 출발점 = α̂ 고정 LP 해. 반환 로컬 해의 α (없으면 nothing)."""
function _ipopt_local_alpha(td, x̄, F::_FixedAlphaLP, α̂; max_time=120.0, deadline=Inf)
    isnan(_eval_alpha!(F, α̂)) && return nothing
    m, v = build_true_dro_subproblem(td, x̄; optimizer=Ipopt.Optimizer, silent=true)
    delete!(unsafe_backend(m).options, "DualReductions")      # 빌더가 넣는 Gurobi 전용 옵션 제거
    set_optimizer_attribute(m, "max_wall_time", max(max_time, 1.0))
    set_optimizer_attribute(m, "print_level", 0)
    set_optimizer_attribute(m, "warm_start_init_point", "yes")
    set_optimizer_attribute(m, "mu_init", 1e-4)
    MOI.set(m, Ipopt.CallbackFunction(), (args...) -> time() < deadline)   # 반복마다 마감 확인
    for (name, X) in v
        haskey(F.vars, name) || continue
        X isa VariableRef && (set_start_value(X, value(F.vars[name])); continue)
        X isa AbstractArray{VariableRef} || continue
        for I in eachindex(X); set_start_value(X[I], value(F.vars[name][I])); end
    end
    optimize!(m)
    has_values(m) || return nothing
    return _capw(value.(v[:α]), td.w)
end


# =====================================================================
# 주 함수
# =====================================================================
"""
    global_bilinear_solve(td, x̄; nworkers, time_limit, rel_gap, ...)

반환 Dict: :LB (incumbent 값), :UB (dual bound), :α (incumbent α), :is_exact (gap ≤ rel_gap),
           :nodes, :time, :root_UB, :ipopt_calls
"""
function global_bilinear_solve(td, x̄; nworkers=Threads.nthreads() - 2, time_limit=300.0, rel_gap=5e-3,
                               dive_max=3, maxrounds=30, min_width=1e-7, ipopt_time=120.0,
                               verbose=true, log_every=30.0)
    nworkers >= 1 || error("global_bilinear_solve: nworkers ≥ 1 필요 (Julia 를 -t (nworkers+2) 로 실행)")
    t0 = time()
    t_end = t0 + time_limit
    K, S, w = td.num_arcs, td.S, td.w
    mk(env) = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true);
                     MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
    envs = [Gurobi.Env() for _ in 1:nworkers]
    Ps = [_build_sep_lp(td, x̄; optimizer=mk(envs[i])) for i in 1:nworkers]
    Fs = [_build_fixed_alpha_lp(td, x̄; optimizer=mk(envs[i])) for i in 1:nworkers]
    tol(v) = rel_gap * (isfinite(v) ? max(1.0, abs(v)) : 1.0)

    lk = ReentrantLock()
    open = Any[(zeros(K), fill(w, K), Inf)]          # (l, u, 부모 상한)
    inflight = fill(-Inf, nworkers)
    LB = Ref(-Inf); best_α = Ref(zeros(K))
    nodes = Ref(0); done = Ref(false); root_UB = Ref(Inf)
    root_rows = Ref{Any}(nothing)                    # root 상자 [0,w] 에서 만든 행 — 모든 노드에서 유효
    node_cands = Tuple{Float64,Vector{Float64}}[]    # (상한, α̂) — Ipopt 출발점 후보
    ipopt_calls = Ref(0)

    globalUB() = max(LB[], isempty(open) ? -Inf : maximum(n[3] for n in open), maximum(inflight))
    function offer!(z, a)
        lock(lk)
        try
            z > LB[] && (LB[] = z; best_α[] = copy(a))
        finally
            unlock(lk)
        end
    end

    function worker(wid)
        P = Ps[wid]; F = Fs[wid]
        localnode = nothing; dive_left = 0; synced = false
        while true
            node = nothing
            lock(lk)
            try
                if done[]
                    localnode !== nothing && push!(open, localnode)
                    return
                end
                if localnode !== nothing && dive_left > 0 && localnode[3] > LB[] + tol(LB[])
                    node = localnode; dive_left -= 1
                else
                    localnode !== nothing && localnode[3] > LB[] + tol(LB[]) && push!(open, localnode)
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
            if !synced && root_rows[] !== nothing && isempty(P.allrows)
                for (k, j, islo, b) in root_rows[]; _sep_add!(P, k, j, islo, b); end
                synced = true
            end
            z = -Inf; α̂ = nothing; children = nothing; zE = -Inf; unfinished = false
            if ubp > LB[] + tol(LB[]) && sum(l) <= w + 1e-9
                _sep_set_box!(P, l, u)
                set_time_limit_sec(P.model, max(t_end - time(), 0.1))
                z = _sep_solve!(P; maxrounds=maxrounds)
                unfinished = isnan(z)
                z = unfinished ? ubp : min(z, ubp)
                if !unfinished && z > LB[] + tol(LB[])
                    α̂ = clamp.(value.(P.vars[:α]), l, u)
                    rv = value.(P.lead); dv = value.(P.fol)
                    ζLv = value.(P.vars[:ζL]); ζFv = value.(P.vars[:ζF])
                    set_time_limit_sec(F.model, max(t_end - time(), 0.1))
                    zE = _eval_alpha!(F, α̂)
                    isnan(zE) && (zE = -Inf)
                    viol = [(u[k] - l[k] > min_width) ?
                            sum(abs(ζLv[k, s] - α̂[k] * rv[s]) + abs(ζFv[k, s] - α̂[k] * dv[s]) for s in 1:S) : 0.0
                            for k in 1:K]
                    k = argmax(viol)
                    if viol[k] > 1e-9
                        width = u[k] - l[k]; t = α̂[k]
                        (t - l[k] < 0.1 * width || u[k] - t < 0.1 * width) && (t = 0.5 * (l[k] + u[k]))
                        u1 = copy(u); u1[k] = t; l2 = copy(l); l2[k] = t
                        children = ((copy(l), u1, z), (l2, copy(u), z))
                    else
                        zE = max(zE, z)          # 완화가 exact
                    end
                end
            end
            lock(lk)
            try
                if unfinished                    # 시간 초과로 못 푼 노드는 부모 상한 그대로 되돌림
                    push!(open, node); inflight[wid] = -Inf; dive_left = 0
                    continue
                end
                nodes[] += 1
                if ubp == Inf
                    root_UB[] = z
                    root_rows[] = [(r[1], r[6], r[2], r[3]) for r in P.allrows]
                end
                if zE > LB[]
                    LB[] = zE; best_α[] = copy(α̂)
                end
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

    # primal heuristic: Ipopt (균등 α 에서 1회 → 상한 높은 노드의 α̂ 에서 반복)
    function heuristic()
        henv = Gurobi.Env()
        Fh = _build_fixed_alpha_lp(td, x̄; optimizer=mk(henv))
        tried = Set{Vector{Float64}}()
        a0 = fill(w / K, K)
        while !done[] && t_end - time() > 30.0       # 남은 시간이 짧으면 새 호출 안 함 (모델 생성 오버헤드)
            push!(tried, round.(a0; digits=3))
            aloc = _ipopt_local_alpha(td, x̄, Fh, a0; max_time=min(ipopt_time, t_end - time()), deadline=t_end)
            ipopt_calls[] += 1
            if aloc !== nothing
                z = _eval_alpha!(Fh, aloc)
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
            a0 = _capw(a0, w)
        end
    end

    tasks = [Threads.@spawn worker(i) for i in 1:nworkers]
    push!(tasks, Threads.@spawn heuristic())
    t_log = time()
    while true
        sleep(0.2)
        fin = false; UBn = Inf
        lock(lk)
        try
            UBn = globalUB()
            idle = isempty(open) && all(inflight .== -Inf)
            fin = (time() >= t_end) || (UBn - LB[] <= tol(LB[])) || (idle && nodes[] > 0)
            fin && (done[] = true)
        finally
            unlock(lk)
        end
        if verbose && time() - t_log > log_every
            @printf("    [global-bilinear] t=%6.1fs nodes=%5d open=%5d LB=%.6f UB=%.6f gap=%.3e\n",
                    time() - t0, nodes[], length(open), LB[], UBn, (UBn - LB[]) / max(1.0, abs(LB[])))
            flush(stdout); t_log = time()
        end
        fin && break
    end
    foreach(wait, tasks)
    UB = max(LB[], isempty(open) ? -Inf : maximum(n[3] for n in open))
    return Dict(:LB => LB[], :UB => UB, :α => best_α[], :is_exact => (UB - LB[] <= tol(LB[])),
                :nodes => nodes[], :time => time() - t0, :root_UB => root_UB[], :ipopt_calls => ipopt_calls[])
end
