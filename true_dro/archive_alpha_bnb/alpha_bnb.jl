"""
alpha_bnb.jl — α-공간 branch-and-bound으로 Ω subproblem V*(x̄)의 dual bound 계산.

근거: Ω(build_true_dro_subproblem)의 bilinear 항은 전부 α_k·r_s (또는 α_k·a_s) 와 α_k·d_s.
  - α 고정 → Ω 전체가 LP
  - α 상자 [l,u] 위 RLT 완화 → LP, 상자가 점으로 줄면 exact
따라서 α (K차원, S와 무관)만 분기하면 수렴. belief(r, a, b, d, e)는 분기하지 않음.

완화 (rlt 옵션):
  :basic — belief 변수 상·하한 × α 상자 (= McCormick) + Σ=1 등식 RLT
  :full  — belief 다면체의 모든 제약 (TV 볼: a−b≤q̂, a+b≥q̂, Σb≤2ε, CVaR: r≤a/(1−β))
           × α 상자 인수 (α_k−l_k), (u_k−α_k)  (level-1 RLT).
           TV 볼의 "초과 질량 합 ≤ 2ε" 가 완화에 들어가 오차가 S에 비례해 쌓이지 않음.

노드 = α 상자. 상한 = RLT LP 값, 하한 = 상자 안 α̂ 고정 LP 값.
"""

using JuMP, Printf

# belief 제약 한 줄: g(y) = c0 + Σ c·Y[fam][s] ≥ 0
struct _BeliefRow
    c0::Float64
    terms::Vector{Tuple{Symbol,Int,Float64}}    # (family, s, coef)
end

mutable struct _AlphaLP
    model::Model
    vars::Dict
    K::Int
    S::Int
    lead::Vector{VariableRef}                   # r (CVaR) 또는 a  (분기 위반 계산용)
    fol::Vector{VariableRef}                    # d
    Y::Dict{Symbol,Vector{VariableRef}}         # belief 변수 family
    Z::Dict{Symbol,Matrix{VariableRef}}         # Z[fam][k,s] = α_k · Y[fam][s]
    rows::Vector{_BeliefRow}
    con_lo::Matrix{ConstraintRef}               # K × nrows : (α_k − l_k)·g ≥ 0
    con_hi::Matrix{ConstraintRef}               # K × nrows : (u_k − α_k)·g ≥ 0
    l::Vector{Float64}                          # 현재 모델에 반영된 α 상자
    u::Vector{Float64}
end


function _belief_rows(td, Y; full::Bool)
    S = td.S
    q = td.q_hat
    rows = _BeliefRow[]
    bnd(fam, s) = (lower_bound(Y[fam][s]), upper_bound(Y[fam][s]))
    # 상·하한 (McCormick에 해당)
    for fam in keys(Y), s in 1:S
        (full || fam in (:lead, :d)) || continue
        lo, hi = bnd(fam, s)
        push!(rows, _BeliefRow(-lo, [(fam, s, 1.0)]))      # y − lo ≥ 0
        push!(rows, _BeliefRow(hi, [(fam, s, -1.0)]))      # hi − y ≥ 0
    end
    full || return rows
    # TV 볼 (leader: a, b / follower: d, e)
    for (yf, ef, ε) in ((:a, :b, td.eps_hat), (:d, :e, td.eps_tilde))
        haskey(Y, ef) || continue
        for s in 1:S
            push!(rows, _BeliefRow(q[s], [(ef, s, 1.0), (yf, s, -1.0)]))    # q + e − y ≥ 0
            push!(rows, _BeliefRow(-q[s], [(yf, s, 1.0), (ef, s, 1.0)]))    # y + e − q ≥ 0
        end
        push!(rows, _BeliefRow(2ε, [(ef, s, -1.0) for s in 1:S]))           # 2ε − Σe ≥ 0
    end
    # CVaR: a/(1−β) − r ≥ 0
    if haskey(Y, :r)
        β = td.beta
        for s in 1:S
            push!(rows, _BeliefRow(0.0, [(:a, s, 1.0 / (1.0 - β)), (:r, s, -1.0)]))
        end
    end
    return rows
end


function _add_rlt_row!(model, α, Y, Z, k, row::_BeliefRow, l, u)
    # (α_k − l)·g ≥ 0 :  c0·α_k + Σ c·Z − l·Σ c·Y ≥ l·c0
    lo = @constraint(model, row.c0 * α[k] + sum(c * Z[f][k, s] for (f, s, c) in row.terms)
                            - l * sum(c * Y[f][s] for (f, s, c) in row.terms) >= l * row.c0)
    # (u − α_k)·g ≥ 0 :  u·Σ c·Y − c0·α_k − Σ c·Z ≥ −u·c0
    hi = @constraint(model, u * sum(c * Y[f][s] for (f, s, c) in row.terms)
                            - row.c0 * α[k] - sum(c * Z[f][k, s] for (f, s, c) in row.terms) >= -u * row.c0)
    return lo, hi
end


"""Ω에서 ζ 정의(quadratic equality)를 지우고 α 상자 RLT로 대체한 LP."""
function _build_alpha_lp(td, x̄; optimizer, rlt=:full)
    rlt in (:basic, :full) || error("rlt must be :basic or :full (got $rlt)")
    full = (rlt == :full)
    model, vars = build_true_dro_subproblem(td, x̄; optimizer=optimizer, silent=true)
    for name in (:ζL_def, :ζF_def)
        delete.(model, model[name])
        unregister(model, name)
    end
    try set_optimizer_attribute(model, "Method", 1) catch; end   # dual simplex (warm start)
    K, S, w = td.num_arcs, td.S, td.w
    α, ζL, ζF = vars[:α], vars[:ζL], vars[:ζF]
    cvar = vars[:_use_cvar]
    lead = cvar ? vec(vars[:r]) : vec(vars[:a])
    fol = vec(vars[:d])

    # belief family 와 곱 변수
    Y = Dict{Symbol,Vector{VariableRef}}()
    Z = Dict{Symbol,Matrix{VariableRef}}()
    newZ(name) = @variable(model, [1:K, 1:S], lower_bound = 0.0, base_name = name)
    if full
        Y[:a] = vec(vars[:a]); Y[:b] = vec(vars[:b]); Y[:d] = fol; Y[:e] = vec(vars[:e])
        Z[:d] = ζF; Z[:b] = newZ("Zb"); Z[:e] = newZ("Ze")
        if cvar
            Y[:r] = lead; Z[:r] = ζL; Z[:a] = newZ("Za")
        else
            Z[:a] = ζL
        end
    else
        Y[:lead] = lead; Z[:lead] = ζL; Y[:d] = fol; Z[:d] = ζF
    end

    rows = _belief_rows(td, Y; full=full)
    l, u = zeros(K), fill(w, K)
    con_lo = Matrix{ConstraintRef}(undef, K, length(rows))
    con_hi = Matrix{ConstraintRef}(undef, K, length(rows))
    for k in 1:K, (j, row) in enumerate(rows)
        con_lo[k, j], con_hi[k, j] = _add_rlt_row!(model, α, Y, Z, k, row, l[k], u[k])
    end

    # 정적 RLT: (Σ_s y_s = 1)·α_k,  (w − Σ_k α_k)·(y_s ≥ 0)
    for fam in keys(Y)
        fam in (:lead, :a, :r, :d) || continue
        @constraint(model, [k = 1:K], sum(Z[fam][k, s] for s in 1:S) == α[k])
        @constraint(model, [s = 1:S], sum(Z[fam][k, s] for k in 1:K) <= w * Y[fam][s])
    end
    return _AlphaLP(model, vars, K, S, lead, fol, Y, Z, rows, con_lo, con_hi, l, u)
end


function _set_box!(A::_AlphaLP, l, u)
    α = A.vars[:α]
    for k in 1:A.K
        (A.l[k] == l[k] && A.u[k] == u[k]) && continue
        set_lower_bound(α[k], l[k])
        set_upper_bound(α[k], u[k])
        for (j, row) in enumerate(A.rows)
            clo, chi = A.con_lo[k, j], A.con_hi[k, j]
            for (f, s, c) in row.terms
                set_normalized_coefficient(clo, A.Y[f][s], -l[k] * c)
                set_normalized_coefficient(chi, A.Y[f][s], u[k] * c)
            end
            set_normalized_rhs(clo, l[k] * row.c0)
            set_normalized_rhs(chi, -u[k] * row.c0)
        end
        A.l[k] = l[k]
        A.u[k] = u[k]
    end
end


"""LP 값. 상자가 infeasible이면 -Inf. 그 외 비정상 종료는 에러."""
function _solve_alpha_lp!(A::_AlphaLP)
    optimize!(A.model)
    st = termination_status(A.model)
    st == MOI.OPTIMAL && return objective_value(A.model)
    st == MOI.INFEASIBLE && return -Inf
    st == MOI.TIME_LIMIT && return NaN          # 호출 측에서 "미처리" 로 다룸
    error("α-B&B LP: $st")
end




"""
    alpha_bnb(td, x̄; optimizer, time_limit, rel_gap, verbose, log_every)

α-공간 B&B. best-first (최대 상한 노드 우선), 분기 변수 = McCormick 위반이 가장 큰 α_k.
반환 Dict: :LB, :UB, :α (incumbent), :nodes, :time, :root_UB, :is_exact, :history
"""
function alpha_bnb(td, x̄; optimizer, time_limit=300.0, rel_gap=1e-4, verbose=true, rlt=:full,
                   log_every=30.0, min_width=1e-7)
    t0 = time()
    K, S, w = td.num_arcs, td.S, td.w
    A = _build_alpha_lp(td, x̄; optimizer=optimizer, rlt=rlt)   # 완화용
    E = _build_alpha_lp(td, x̄; optimizer=optimizer, rlt=:basic)   # α 고정 평가용 (α 고정이면 McCormick도 exact)
    t_build = time() - t0

    LB, best_α = -Inf, zeros(K)
    root_UB = Inf
    open = [(zeros(K), fill(w, K), Inf)]           # (l, u, 부모 상한)
    nodes = 0
    history = Tuple{Float64,Float64,Float64,Int}[]  # (t, LB, UB, nodes)
    t_log = time()
    tol(LBv) = rel_gap * (isfinite(LBv) ? max(1.0, abs(LBv)) : 1.0)

    open_UB() = isempty(open) ? LB : max(LB, maximum(n[3] for n in open))

    while !isempty(open)
        UB = open_UB()
        UB - LB <= tol(LB) && break
        time() - t0 > time_limit && break

        i = argmax([n[3] for n in open])
        l, u, ub_parent = popat!(open, i)
        ub_parent <= LB + tol(LB) && continue
        sum(l) > w + 1e-9 && continue               # Σα ≤ w 와 상자가 만나지 않음

        _set_box!(A, l, u)
        z = _solve_alpha_lp!(A)
        nodes += 1
        nodes == 1 && (root_UB = z)
        z <= LB + tol(LB) && continue

        α̂ = clamp.(value.(A.vars[:α]), l, u)
        rv = value.(A.lead); dv = value.(A.fol)
        ζLv = value.(A.vars[:ζL]); ζFv = value.(A.vars[:ζF])

        # 하한: α̂ 고정 LP
        _set_box!(E, α̂, α̂)
        zE = _solve_alpha_lp!(E)
        if zE > LB
            LB, best_α = zE, copy(α̂)
        end

        # 분기 변수: McCormick 위반 합이 가장 큰 α_k (폭이 남은 것만)
        viol = [ (u[k] - l[k] > min_width) ?
                 sum(abs(ζLv[k, s] - α̂[k] * rv[s]) + abs(ζFv[k, s] - α̂[k] * dv[s]) for s in 1:S) : 0.0
                 for k in 1:K ]
        k = argmax(viol)
        if viol[k] <= 1e-9
            # 완화가 이 상자에서 exact → z 는 실현 가능한 값
            z > LB && (LB = z; best_α = copy(α̂))
            continue
        end
        width = u[k] - l[k]
        t = α̂[k]
        (t - l[k] < 0.1 * width || u[k] - t < 0.1 * width) && (t = 0.5 * (l[k] + u[k]))
        u1 = copy(u); u1[k] = t
        l2 = copy(l); l2[k] = t
        push!(open, (copy(l), u1, z))
        push!(open, (l2, copy(u), z))

        if verbose && time() - t_log > log_every
            UBn = open_UB()
            @printf("    [α-B&B] t=%6.1fs nodes=%6d open=%6d LB=%.6f UB=%.6f gap=%.3e\n",
                    time() - t0, nodes, length(open), LB, UBn, (UBn - LB) / max(1.0, abs(LB)))
            flush(stdout)
            t_log = time()
        end
        push!(history, (time() - t0, LB, open_UB(), nodes))
    end

    UB = open_UB()
    return Dict(:LB => LB, :UB => UB, :α => best_α, :nodes => nodes, :time => time() - t0,
                :t_build => t_build, :root_UB => root_UB,
                :is_exact => (UB - LB <= tol(LB)), :history => history)
end
