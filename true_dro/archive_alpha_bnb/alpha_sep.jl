"""
alpha_sep.jl — α-B&B (full RLT) 에서 RLT 행을 전부 넣지 않고 cut 분리(separation)로 추가.

RLT 행 (k, j, lo, L):  (α_k − L)·g_j(y) ≥ 0  →  c0 α_k + Σ c Z_k − L(c0 + Σ c y) ≥ 0
RLT 행 (k, j, hi, U):  (U − α_k)·g_j(y) ≥ 0  →  U(c0 + Σ c y) − c0 α_k − Σ c Z_k ≥ 0
유효 조건: 상자가 l_k ≥ L (lo) / u_k ≤ U (hi) 이면 유효. 자식 상자는 부모 상자 안 → 부모 행 그대로 유효.
같은 (k,j,side)의 더 조인 행은 이전 행을 지배 (g_j(y) ≥ 0 이 Ω 에 있으므로) → 교체.
분리 도중 어느 시점에 멈춰도 LP 는 full RLT LP 의 완화 → valid UB.
"""

using JuMP, Printf

mutable struct _SepLP
    model::Model
    vars::Dict
    K::Int
    S::Int
    lead::Vector{VariableRef}
    fol::Vector{VariableRef}
    Y::Dict{Symbol,Vector{VariableRef}}
    Z::Dict{Symbol,Matrix{VariableRef}}
    rows::Vector{_BeliefRow}
    pool::Dict{Tuple{Int,Int,Bool},Tuple{ConstraintRef,Float64}}   # (k, j, islo) → 가장 조인 활성 행 (행, L 또는 U)
    allrows::Vector{Tuple{Int,Bool,Float64,ConstraintRef,Float64,Int}}  # (k, islo, b, 행, 원래 rhs, j) — 삭제하지 않음
    active::Vector{Bool}
    l::Vector{Float64}
    u::Vector{Float64}
    allvars::Vector{VariableRef}                                      # basis 저장용 (변수는 삭제 안 함)
    basecons::Vector{ConstraintRef}                                   # Ω + 정적 RLT 행
    w::Float64
    simplex_rlt::Bool          # (w − Σ_k α_k)·g_j(y) ≥ 0 행도 분리 (k = 0 으로 표기, 항상 유효)
end


function _build_sep_lp(td, x̄; optimizer, simplex_rlt=false)
    model, vars = build_true_dro_subproblem(td, x̄; optimizer=optimizer, silent=true)
    for name in (:ζL_def, :ζF_def)
        delete.(model, model[name]); unregister(model, name)
    end
    try set_optimizer_attribute(model, "Method", 1) catch; end
    try set_optimizer_attribute(model, "Presolve", 0) catch; end   # 재풀이 warm start 유지 (측정: S=50 15k→4k 반복)
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
    for fam in (:a, :r, :d)
        haskey(Y, fam) || continue
        @constraint(model, [k = 1:K], sum(Z[fam][k, s] for s in 1:S) == α[k])
        @constraint(model, [s = 1:S], sum(Z[fam][k, s] for k in 1:K) <= w * Y[fam][s])
    end
    rows = _belief_rows(td, Y; full=true)
    basecons = ConstraintRef[]
    for (F, Sset) in list_of_constraint_types(model)
        F == VariableRef && continue
        append!(basecons, all_constraints(model, F, Sset))
    end
    return _SepLP(model, vars, K, S, lead, fol, Y, Z, rows,
                  Dict{Tuple{Int,Int,Bool},Tuple{ConstraintRef,Float64}}(),
                  Tuple{Int,Bool,Float64,ConstraintRef,Float64,Int}[], Bool[], zeros(K), fill(w, K),
                  all_variables(model), basecons, w, simplex_rlt)
end


function _sep_add!(P::_SepLP, k, j, islo, b)
    row = P.rows[j]
    α = P.vars[:α]
    Gy = sum(c * P.Y[f][s] for (f, s, c) in row.terms)
    if k == 0
        # (w − Σ_k α_k)·g_j ≥ 0 :  w·Σ c y − c0 Σ_k α_k − Σ_k Σ c Z_k ≥ −w c0
        GZs = row.c0 * sum(α) + sum(c * P.Z[f][kk, s] for (f, s, c) in row.terms for kk in 1:P.K)
        cref = @constraint(P.model, P.w * Gy - GZs >= -P.w * row.c0)
    else
        GZ = row.c0 * α[k] + sum(c * P.Z[f][k, s] for (f, s, c) in row.terms)
        cref = islo ? @constraint(P.model, GZ - b * Gy >= b * row.c0) :
                      @constraint(P.model, b * Gy - GZ >= -b * row.c0)
    end
    # 이전 행은 지우지 않음 (지배되지만 유효; 삭제하면 basis 손실 → cold restart)
    P.pool[(k, j, islo)] = (cref, b)
    push!(P.allrows, (k, islo, b, cref, normalized_rhs(cref), j))
    push!(P.active, true)
end


"""상자 (l,u) 로 이동: α bound 설정, 무효가 된 풀 행 삭제."""
function _sep_set_box!(P::_SepLP, l, u)
    α = P.vars[:α]
    for k in 1:P.K
        set_lower_bound(α[k], l[k]); set_upper_bound(α[k], u[k])
    end
    # 무효 행은 삭제 대신 우변을 −∞ 로 풀어 비활성화, 다시 유효해지면 복원 (basis 유지)
    for (i, (k, islo, b, cref, rhs0, _)) in enumerate(P.allrows)
        k == 0 && continue                       # 단체 RLT 행은 항상 유효
        valid = islo ? (l[k] >= b - 1e-12) : (u[k] <= b + 1e-12)
        if valid && !P.active[i]
            set_normalized_rhs(cref, rhs0); P.active[i] = true
        elseif !valid && P.active[i]
            set_normalized_rhs(cref, -1e30); P.active[i] = false
        end
    end
    P.l .= l; P.u .= u
end


"""현재 LP 해에서 노드 상자 기준 RLT 위반 계산. 반환 [(viol, k, j, islo)] (viol < −tol 인 것)."""
function _sep_violations(P::_SepLP; tol=1e-7)
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
            # 이미 같은 bound로 들어 있는 행은 위반될 수 없음 (수치 오차 제외)
            vlo < -tol && push!(out, (vlo, k, j, true))
            vhi < -tol && push!(out, (vhi, k, j, false))
        end
        if P.simplex_rlt
            vs = P.w * Gy - sum(GZ)
            vs < -tol && push!(out, (vs, 0, j, true))
        end
    end
    return out
end


"""분리 루프. 반환 (LP 값 = valid UB, 라운드 수, 수렴 여부). LP infeasible → -Inf."""
function _sep_solve!(P::_SepLP; maxrounds=30, maxadd=5000, tol=1e-6, time_budget=Inf,
                     stall_tol=1e-6, stall_rounds=2)
    t0 = time()
    z = Inf
    zprev = Inf
    nstall = 0
    for round in 1:maxrounds
        z = _solve_alpha_lp!(P)
        isnan(z) && return (NaN, round, false)
        z == -Inf && return (-Inf, round, true)
        V = _sep_violations(P; tol=tol)
        isempty(V) && return (z, round, true)
        time() - t0 > time_budget && return (z, round, false)
        # 정체: 값 개선이 상대 stall_tol 미만이 stall_rounds 번 연속 → 중단 (LP 값은 여전히 valid UB)
        nstall = (zprev - z <= stall_tol * max(1.0, abs(z))) ? nstall + 1 : 0
        zprev = z
        nstall >= stall_rounds && return (z, round, false)
        sort!(V; by=first)
        for (v, k, j, islo) in V[1:min(maxadd, length(V))]
            _sep_add!(P, k, j, islo, k == 0 ? P.w : (islo ? P.l[k] : P.u[k]))
        end
    end
    z = _solve_alpha_lp!(P)        # 마지막으로 추가한 행까지 반영 (값 조회 가능 상태로)
    return (z, maxrounds, false)
end

const _GRB_VBASIS = Gurobi.VariableAttribute("VBasis")
const _GRB_CBASIS = Gurobi.ConstraintAttribute("CBasis")

"""현재 LP basis 저장 (변수, 기본 행, RLT 행 — 인덱스 고정)."""
function _get_basis(P::_SepLP)
    vb = Int8[MOI.get(P.model, _GRB_VBASIS, v) for v in P.allvars]
    cb = Int8[MOI.get(P.model, _GRB_CBASIS, c) for c in P.basecons]
    cr = Int8[MOI.get(P.model, _GRB_CBASIS, r[4]) for r in P.allrows]
    return (vb, cb, cr)
end

"""저장한 basis 복원. 그 뒤 추가된 행과 비활성 행은 basic."""
function _set_basis!(P::_SepLP, B)
    vb, cb, cr = B
    for (v, b) in zip(P.allvars, vb); MOI.set(P.model, _GRB_VBASIS, v, Int(b)); end
    for (c, b) in zip(P.basecons, cb); MOI.set(P.model, _GRB_CBASIS, c, Int(b)); end
    for (i, r) in enumerate(P.allrows)
        b = (i <= length(cr) && P.active[i]) ? Int(cr[i]) : 0
        MOI.set(P.model, _GRB_CBASIS, r[4], b)
    end
end

_solve_alpha_lp!(P::_SepLP) = (optimize!(P.model);
    st = termination_status(P.model);
    st == MOI.OPTIMAL ? objective_value(P.model) :
    st == MOI.INFEASIBLE ? -Inf :
    st == MOI.TIME_LIMIT ? NaN : error("α-sep LP: $st"))     # NaN = 시간 초과로 미처리


"""
    alpha_bnb_sep(td, x̄; lp_optimizer, time_limit, rel_gap, maxadd, verbose)

α-B&B (best-first), 노드 bound = RLT 분리 LP. 하한 = α̂ 고정 LP (basic).
"""
function alpha_bnb_sep(td, x̄; lp_optimizer, time_limit=300.0, rel_gap=1e-4, maxadd=5000,
                       verbose=true, log_every=30.0, min_width=1e-7, node_sep_budget=Inf,
                       heur=false, heur_root_N=20, heur_root_climb=3, heur_node_N=3, heur_node_climb=1,
                       warm_basis=true, dive_max=3, maxrounds=30,
                       obbt=false, LB0=-Inf, obbt_budget=120.0, obbt_passes=2, simplex_rlt=false)
    t0 = time()
    K, S, w = td.num_arcs, td.S, td.w
    P = _build_sep_lp(td, x̄; optimizer=lp_optimizer, simplex_rlt=simplex_rlt)
    E = _build_alpha_lp(td, x̄; optimizer=lp_optimizer, rlt=:basic)
    B = heur ? _build_belief_fix_lp(td, x̄; optimizer=lp_optimizer) : nothing
    t_build = time() - t0
    root_LB = -Inf
    t_lp, t_eval = 0.0, 0.0
    tol(v) = rel_gap * (isfinite(v) ? max(1.0, abs(v)) : 1.0)

    LB, best_α = LB0, zeros(K)
    root_UB, root_rounds, root_time = Inf, 0, 0.0
    l0, u0 = zeros(K), fill(w, K)
    obbt_info = Dict{Symbol,Any}()
    if obbt
        # root 완화 + LB (외부 LB0 또는 heuristic) → OBBT 로 root 상자 축소
        _sep_set_box!(P, l0, u0)
        tr = @elapsed (zr, _, _) = _sep_solve!(P)
        obbt_info[:UB_before] = zr
        if heur
            t_eval += @elapsed (zH, αH, _) = rlt_primal_heuristic!(P, E, B, w; N=heur_root_N, n_climb=heur_root_climb)
            zH > LB && (LB = zH; best_α = copy(αH))
        end
        obbt_info[:LB_used] = LB
        verbose && @printf("    [OBBT] root UB=%.6f (%.1fs), LB=%.6f
", zr, tr, LB)
        to = @elapsed (l0, u0, ost) = _sep_obbt!(P, l0, u0, LB; time_budget=obbt_budget, passes=obbt_passes, verbose=verbose)
        obbt_info[:time] = to; obbt_info[:status] = ost
        obbt_info[:nfixed] = count(u0 .- l0 .< 1e-6); obbt_info[:width] = sum(u0 .- l0) / K
        ost == :infeasible && (l0 .= 0; u0 .= -1)    # LB 가 최적 → 열린 노드 없음
    end
    open = Any[]
    all(u0 .>= l0) && push!(open, (l0, u0, Inf, nothing))   # (l, u, 부모 상한, 부모 basis)
    nodes = 0
    dive_left = 0                                          # >0 이면 다음 노드 = 마지막으로 넣은 자식
    t_basis = 0.0
    t_log = time()
    open_UB() = isempty(open) ? LB : max(LB, maximum(n[3] for n in open))

    while !isempty(open)
        UB = open_UB()
        UB - LB <= tol(LB) && break
        time() - t0 > time_limit && break
        diving = dive_left > 0
        i = diving ? length(open) : argmax([n[3] for n in open])
        dive_left = diving ? dive_left - 1 : dive_max
        l, u, ubp, Bpar = popat!(open, i)
        ubp <= LB + tol(LB) && (dive_left = 0; continue)
        sum(l) > w + 1e-9 && (dive_left = 0; continue)

        _sep_set_box!(P, l, u)
        # 다이빙(직전 노드의 자식)이 아니면 부모 basis 복원 → 형제/먼 노드로 옮겨도 warm start
        if warm_basis && Bpar !== nothing && !diving
            t_basis += @elapsed _set_basis!(P, Bpar)
        end
        tl = @elapsed (z, rounds, conv) = _sep_solve!(P; maxadd=maxadd, maxrounds=maxrounds,
                          time_budget=min(node_sep_budget, max(time_limit - (time() - t0), 0.0)))
        t_lp += tl
        z = min(z, ubp)
        nodes += 1
        if nodes == 1
            root_UB, root_rounds, root_time = z, rounds, tl
            verbose && @printf("    [α-sep] root: UB=%.6f rounds=%d pool=%d conv=%s %.1fs\n",
                               z, rounds, count(P.active), conv, tl)
        end
        z <= LB + tol(LB) && (dive_left = 0; continue)

        α̂ = clamp.(value.(P.vars[:α]), l, u)
        Bnode = nothing
        warm_basis && (t_basis += @elapsed Bnode = _get_basis(P))
        rv = value.(P.lead); dv = value.(P.fol)
        ζLv = value.(P.vars[:ζL]); ζFv = value.(P.vars[:ζF])

        t_eval += @elapsed begin
            if heur
                zE, αE, _ = rlt_primal_heuristic!(P, E, B, w;
                                N=(nodes == 1 ? heur_root_N : heur_node_N),
                                n_climb=(nodes == 1 ? heur_root_climb : heur_node_climb))
            else
                _set_box!(E, α̂, α̂)
                zE, αE = _solve_alpha_lp!(E), α̂
            end
        end
        zE > LB && (LB = zE; best_α = copy(αE))
        if nodes == 1
            root_LB = LB
            verbose && @printf("    [α-sep] root LB=%.6f (heur=%s, eval %.1fs)
", LB, heur, t_eval)
        end

        viol = [(u[k] - l[k] > min_width) ?
                sum(abs(ζLv[k, s] - α̂[k] * rv[s]) + abs(ζFv[k, s] - α̂[k] * dv[s]) for s in 1:S) : 0.0
                for k in 1:K]
        k = argmax(viol)
        if viol[k] <= 1e-9
            z > LB && (LB = z; best_α = copy(α̂))
            dive_left = 0
            continue
        end
        width = u[k] - l[k]
        t = α̂[k]
        (t - l[k] < 0.1 * width || u[k] - t < 0.1 * width) && (t = 0.5 * (l[k] + u[k]))
        u1 = copy(u); u1[k] = t
        l2 = copy(l); l2[k] = t
        push!(open, (l2, copy(u), z, Bnode))
        push!(open, (copy(l), u1, z, Bnode))        # 다이빙 시 이 자식이 다음 노드

        if verbose && time() - t_log > log_every
            UBn = open_UB()
            @printf("    [α-sep] t=%6.1fs nodes=%5d open=%5d pool=%6d LB=%.6f UB=%.6f gap=%.3e (lp %.0fs, eval %.0fs)\n",
                    time() - t0, nodes, length(open), count(P.active), LB, UBn,
                    (UBn - LB) / max(1.0, abs(LB)), t_lp, t_eval)
            flush(stdout); t_log = time()
        end
    end
    UB = open_UB()
    return Dict(:LB => LB, :UB => UB, :α => best_α, :nodes => nodes, :time => time() - t0,
                :t_build => t_build, :root_UB => root_UB, :root_LB => root_LB, :root_rounds => root_rounds, :root_time => root_time,
                :t_lp => t_lp, :t_eval => t_eval, :t_basis => t_basis, :pool => count(P.active), :allrows => length(P.allrows),
                :is_exact => (UB - LB <= tol(LB)), :obbt => obbt_info)
end


"""
    _sep_obbt!(P, l, u, LB; time_budget, passes, tol)

OBBT (optimization-based bound tightening). 현재 완화 LP 에 목적 컷 obj ≥ LB 를 걸고
α_k 마다 max α_k, min α_k 를 풀어 상자를 줄임. "LB 보다 좋은 해가 있다면 α_k 는 이 범위 안" → 최적해 손실 없음.
패스 사이에 새 상자로 RLT 재분리 (원래 목적). 반환 (l, u, status) — status=:infeasible 이면 LB 가 최적 (UB ≤ LB).
"""
function _sep_obbt!(P::_SepLP, l, u, LB; time_budget=Inf, passes=2, tol=1e-7, verbose=true)
    t0 = time()
    α = P.vars[:α]
    K = P.K
    l = copy(l); u = copy(u)
    obj0 = objective_function(P.model)
    cut = @constraint(P.model, obj0 >= LB)
    status = :ok
    for pass in 1:passes
        set_optimizer_attribute(P.model, "Method", 0)     # 목적만 바뀜 → primal simplex warm
        nchg = 0
        for k in 1:K, sense in (:max, :min)
            time() - t0 > time_budget && break
            u[k] - l[k] <= tol && continue
            @objective(P.model, sense == :max ? MAX_SENSE : MIN_SENSE, α[k])
            optimize!(P.model)
            st = termination_status(P.model)
            if st == MOI.INFEASIBLE
                status = :infeasible; break
            end
            st == MOI.OPTIMAL || error("OBBT LP: $st")
            v = objective_value(P.model)
            if sense == :max && v + tol < u[k]
                u[k] = max(v + tol, l[k]); nchg += 1
            elseif sense == :min && v - tol > l[k]
                l[k] = min(v - tol, u[k]); nchg += 1
            end
        end
        set_objective(P.model, MAX_SENSE, obj0)
        set_optimizer_attribute(P.model, "Method", 1)
        status == :infeasible && break
        # 새 상자로 이동 + 재분리 (목적 컷은 유지: LB 보다 나쁜 영역 배제)
        _sep_set_box!(P, l, u)
        _sep_solve!(P)
        verbose && @printf("    [OBBT] pass %d: 변경 %d, 고정(폭<1e-6) %d/%d, 평균 폭 %.4f, UB=%.6f (%.1fs)\n",
                           pass, nchg, count(u .- l .< 1e-6), K, sum(u .- l) / K,
                           objective_value(P.model), time() - t0)
        flush(stdout)
        (nchg == 0 || time() - t0 > time_budget) && break
    end
    set_normalized_rhs(cut, -1e30)              # 목적 컷 해제 (B&B 에서는 prune 으로 처리)
    set_objective(P.model, MAX_SENSE, obj0)
    set_optimizer_attribute(P.model, "Method", 1)
    return l, u, status
end
