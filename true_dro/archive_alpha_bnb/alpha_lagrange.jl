"""
alpha_lagrange.jl — α-B&B 노드 bound를 시나리오별 라그랑지 분해로 계산.

노드 LP (alpha_bnb.jl의 full RLT 완화)를 다음으로 분해:
  - 전역 변수: α, δ, ρ⁰₁₂₃ (+ 전역 행 Σα ≤ w)
  - 시나리오 s 블록: s 인덱스 변수 전부 + α 복사본 α^s (상자 [l,u])
  - 연결 행: 둘 이상의 시나리오 또는 δ/ρ⁰ 를 포함하는 행 (Σa=1, Σb≤2ε, Σ_s β ≤ δ, Σ_s Z = α, ...)
  - 복사 등식: α^s_k = α_k

라그랑지:  L(ν, μ) = max  f(x) + Σ_i ν_i (b_i − a_iᵀx) + Σ_{k,s} μ_ks (α_k − α^s_k)
  ν_i = ∂f/∂b_i 부호 규약 (≤ 행 ν ≥ 0, ≥ 행 ν ≤ 0, = 행 자유) 이면 모든 (ν, μ)에서 L ≥ 노드 LP 값 (valid UB).
  JuMP max 문제에서 dual(c) = −∂f/∂b  →  ν = −dual.
root: 통째 LP의 dual로 (ν, μ)를 잡으면 L = LP 값 (검증용).
자식: 부모 승수 상속 + projected subgradient 몇 회.
"""

using JuMP, Printf
import Gurobi

const _SCEN2D = (:u_hat, :ρ_hat_1, :ρ_hat_2, :ρ_hat_3, :ζL, :ζF, :u_tilde, :ω, :β,
                 :ρ_tilde_1, :ρ_tilde_2, :ρ_tilde_3)
const _SCEN1D = (:σ_hat, :a, :b, :d, :e, :σ_tilde, :r)
const _GLOBAL = (:α, :δ, :ρ_psi0_1, :ρ_psi0_2, :ρ_psi0_3)

_sense(::MOI.LessThan) = :le
_sense(::MOI.GreaterThan) = :ge
_sense(::MOI.EqualTo) = :eq

struct _LinkRow
    cref::ConstraintRef
    sense::Symbol
    rhs::Float64
    gterms::Vector{Tuple{VariableRef,Float64}}        # 전역 변수 (G 모델 변수)
    sterms::Vector{Tuple{Int,VariableRef,Float64}}    # (s, 시나리오 모델 변수, coef)
end

mutable struct _LagDecomp
    A::_AlphaLP
    K::Int
    S::Int
    vscen::Dict{VariableRef,Int}              # A 변수 → 시나리오 (0 = 전역)
    αidx::Dict{VariableRef,Int}               # A의 α_k → k
    G::Model
    gmap::Dict{VariableRef,VariableRef}       # A 전역 변수 → G 변수
    M::Vector{Model}                          # 시나리오 모델
    smap::Vector{Dict{VariableRef,VariableRef}}  # A 변수 → M[s] 변수
    αs::Matrix{VariableRef}                   # K × S 복사본
    rowmap::Dict{ConstraintRef,Tuple{Int,ConstraintRef}}  # A 시나리오 RLT 행 → (s, M[s] 행)
    linkrefs::Vector{ConstraintRef}           # A 연결 행
    links::Vector{_LinkRow}
    fobj_g::Vector{Tuple{VariableRef,Float64}}             # 목적함수 전역 부분 (G 변수)
    fobj_s::Vector{Vector{Tuple{VariableRef,Float64}}}     # 목적함수 시나리오 부분 (M[s] 변수)
    chunks::Vector{UnitRange{Int}}
    l::Vector{Float64}
    u::Vector{Float64}
end


function _var_scenarios(A::_AlphaLP, S)
    vs = Dict{VariableRef,Int}()
    αidx = Dict{VariableRef,Int}()
    for name in _GLOBAL
        haskey(A.vars, name) || continue
        x = A.vars[name]
        for (i, v) in enumerate(x isa VariableRef ? [x] : vec(x))
            vs[v] = 0
            name == :α && (αidx[v] = i)
        end
    end
    for name in _SCEN2D
        haskey(A.vars, name) || continue
        X = A.vars[name]
        for s in 1:S, i in axes(X, 1)
            vs[X[i, s]] = s
        end
    end
    for name in _SCEN1D
        haskey(A.vars, name) || continue
        X = A.vars[name]
        for s in 1:S
            vs[X[s]] = s
        end
    end
    for (fam, Z) in A.Z
        for s in 1:S, k in axes(Z, 1)
            vs[Z[k, s]] = s
        end
    end
    for v in all_variables(A.model)
        haskey(vs, v) || error("α-Lagrange: 분류되지 않은 변수 $(name(v))")
    end
    return vs, αidx
end


function _copy_var!(m, v)
    nv = @variable(m)
    has_lower_bound(v) && set_lower_bound(nv, lower_bound(v))
    has_upper_bound(v) && set_upper_bound(nv, upper_bound(v))
    return nv
end


function _classify(func, vs)
    scen = 0
    multi = false
    nonα_global = false
    for (v, _) in func.terms
        s = vs[v]
        if s > 0
            if scen == 0
                scen = s
            elseif scen != s
                multi = true
            end
        end
    end
    return scen, multi
end


"""A (full RLT 노드 LP)에서 분해 구조 생성. nthreads개 chunk, chunk마다 Gurobi Env."""
function _build_lag_decomp(A::_AlphaLP, td; nchunks=Threads.nthreads())
    K, S = A.K, A.S
    vs, αidx = _var_scenarios(A, S)

    G = Model(Gurobi.Optimizer); set_silent(G)
    set_optimizer_attribute(G, "DualReductions", 0)
    gmap = Dict{VariableRef,VariableRef}()
    for (v, s) in vs
        s == 0 && (gmap[v] = _copy_var!(G, v))
    end

    nchunks = max(1, min(nchunks, S))
    chunks = [round(Int, (c - 1) * S / nchunks) + 1 : round(Int, c * S / nchunks) for c in 1:nchunks]
    envs = [Gurobi.Env() for _ in 1:nchunks]
    M = Vector{Model}(undef, S)
    smap = [Dict{VariableRef,VariableRef}() for _ in 1:S]
    αs = Matrix{VariableRef}(undef, K, S)
    for (c, rg) in enumerate(chunks), s in rg
        env = envs[c]
        m = Model(() -> Gurobi.Optimizer(env)); set_silent(m)
        set_optimizer_attribute(m, "Threads", 1)
        set_optimizer_attribute(m, "Method", 1)
        set_optimizer_attribute(m, "DualReductions", 0)
        M[s] = m
        for k in 1:K
            αs[k, s] = @variable(m, lower_bound = A.l[k], upper_bound = A.u[k])
        end
    end
    for (v, s) in vs
        s > 0 && (smap[s][v] = _copy_var!(M[s], v))
    end
    αA = A.vars[:α]
    mapv(s, v) = haskey(αidx, v) ? αs[αidx[v], s] : smap[s][v]

    rowmap = Dict{ConstraintRef,Tuple{Int,ConstraintRef}}()
    linkrefs = ConstraintRef[]
    for (F, Sset) in list_of_constraint_types(A.model)
        F == VariableRef && continue
        F == AffExpr || error("α-Lagrange: 지원하지 않는 제약 형식 $F")
        for c in all_constraints(A.model, F, Sset)
            obj = constraint_object(c)
            scen, multi = _classify(obj.func, vs)
            hasglob_nonα = any(vs[v] == 0 && !haskey(αidx, v) for (v, _) in obj.func.terms)
            if scen == 0 && !multi
                aff = AffExpr(0.0)
                for (v, coef) in obj.func.terms
                    add_to_expression!(aff, coef, gmap[v])
                end
                @constraint(G, aff in obj.set)
            elseif !multi && !hasglob_nonα
                aff = AffExpr(0.0)
                for (v, coef) in obj.func.terms
                    add_to_expression!(aff, coef, mapv(scen, v))
                end
                rowmap[c] = (scen, @constraint(M[scen], aff in obj.set))
            else
                push!(linkrefs, c)
            end
        end
    end

    fobj = objective_function(A.model)
    fobj_g = Tuple{VariableRef,Float64}[]
    fobj_s = [Tuple{VariableRef,Float64}[] for _ in 1:S]
    for (v, coef) in fobj.terms
        s = vs[v]
        s == 0 ? push!(fobj_g, (gmap[v], coef)) : push!(fobj_s[s], (mapv(s, v), coef))
    end
    abs(fobj.constant) < 1e-12 || error("α-Lagrange: 목적함수 상수항 미지원")

    D = _LagDecomp(A, K, S, vs, αidx, G, gmap, M, smap, αs, rowmap, linkrefs, _LinkRow[],
                   fobj_g, fobj_s, chunks, copy(A.l), copy(A.u))
    _refresh_links!(D)
    return D
end


function _refresh_links!(D::_LagDecomp)
    links = _LinkRow[]
    for c in D.linkrefs
        obj = constraint_object(c)
        gt = Tuple{VariableRef,Float64}[]
        st = Tuple{Int,VariableRef,Float64}[]
        for (v, coef) in obj.func.terms
            s = D.vscen[v]
            if s == 0
                push!(gt, (D.gmap[v], coef))
            else
                push!(st, (s, haskey(D.αidx, v) ? D.αs[D.αidx[v], s] : D.smap[s][v], coef))
            end
        end
        push!(links, _LinkRow(c, _sense(obj.set), MOI.constant(obj.set), gt, st))
    end
    D.links = links
end


"""A의 상자를 (l,u)로 바꾸고 분해 모델에 반영."""
function _lag_set_box!(D::_LagDecomp, l, u)
    A = D.A
    changed = [k for k in 1:D.K if D.l[k] != l[k] || D.u[k] != u[k]]
    isempty(changed) && return
    _set_box!(A, l, u)
    αG = [D.gmap[A.vars[:α][k]] for k in 1:D.K]
    for k in changed
        set_lower_bound(αG[k], l[k]); set_upper_bound(αG[k], u[k])
        for s in 1:D.S
            set_lower_bound(D.αs[k, s], l[k]); set_upper_bound(D.αs[k, s], u[k])
        end
        for (j, row) in enumerate(A.rows), c in (A.con_lo[k, j], A.con_hi[k, j])
            haskey(D.rowmap, c) || continue          # 연결 행은 _refresh_links! 에서
            s, cs = D.rowmap[c]
            for (f, ss, _) in row.terms
                y = A.Y[f][ss]
                set_normalized_coefficient(cs, D.smap[s][y], normalized_coefficient(c, y))
            end
            set_normalized_rhs(cs, normalized_rhs(c))
        end
        D.l[k] = l[k]; D.u[k] = u[k]
    end
    _refresh_links!(D)
end


"""root: A를 통째로 풀어 (ν, μ) 초기화. 반환 (LP 값, ν, μ)."""
function _lag_init_multipliers(D::_LagDecomp)
    A = D.A
    z = _solve_alpha_lp!(A)
    ν = [-dual(L.cref) for L in D.links]
    μ = zeros(D.K, D.S)
    # μ_ks = −Σ_{r ∈ 시나리오 s 행} ν_r · a_{r,α_k}   (α^s 정상성 조건)
    for (c, (s, _)) in D.rowmap
        νr = -dual(c)
        νr == 0.0 && continue
        for (v, coef) in constraint_object(c).func.terms
            k = get(D.αidx, v, 0)
            k > 0 && (μ[k, s] -= νr * coef)
        end
    end
    return z, ν, μ
end


"""L(ν, μ) 계산. 반환 (L, 부분기울기 gν, gμ, α^s 해, 시나리오 해 접근용 status)."""
function _lag_eval!(D::_LagDecomp, ν, μ)
    K, S = D.K, D.S
    # --- 전역 ---
    objG = AffExpr(0.0)
    for (v, c) in D.fobj_g; add_to_expression!(objG, c, v); end
    constG = 0.0
    for (i, L) in enumerate(D.links)
        ν[i] == 0.0 && continue
        constG += ν[i] * L.rhs
        for (v, c) in L.gterms; add_to_expression!(objG, -ν[i] * c, v); end
    end
    αG = [D.gmap[D.A.vars[:α][k]] for k in 1:K]
    for k in 1:K
        add_to_expression!(objG, sum(μ[k, s] for s in 1:S), αG[k])
    end
    @objective(D.G, Max, objG)
    # --- 시나리오 목적함수 ---
    objS = [AffExpr(0.0) for _ in 1:S]
    for s in 1:S
        for (v, c) in D.fobj_s[s]; add_to_expression!(objS[s], c, v); end
        for k in 1:K; add_to_expression!(objS[s], -μ[k, s], D.αs[k, s]); end
    end
    for (i, L) in enumerate(D.links)
        ν[i] == 0.0 && continue
        for (s, v, c) in L.sterms; add_to_expression!(objS[s], -ν[i] * c, v); end
    end
    for s in 1:S
        @objective(D.M[s], Max, objS[s])
    end
    # --- 풀기 (chunk 병렬) ---
    zs = zeros(S)
    stat = fill(:ok, S)
    tasks = map(D.chunks) do rg
        Threads.@spawn begin
            for s in rg
                optimize!(D.M[s])
                st = termination_status(D.M[s])
                if st == MOI.OPTIMAL
                    zs[s] = objective_value(D.M[s])
                elseif st == MOI.INFEASIBLE
                    stat[s] = :infeasible
                elseif st == MOI.DUAL_INFEASIBLE
                    stat[s] = :unbounded
                else
                    error("α-Lagrange 시나리오 $s: $st")
                end
            end
        end
    end
    foreach(fetch, tasks)
    optimize!(D.G)
    stG = termination_status(D.G)
    any(==(:infeasible), stat) && return (-Inf, nothing, nothing, nothing)
    stG == MOI.INFEASIBLE && return (-Inf, nothing, nothing, nothing)
    (any(==(:unbounded), stat) || stG == MOI.DUAL_INFEASIBLE) && return (Inf, nothing, nothing, nothing)
    stG == MOI.OPTIMAL || error("α-Lagrange 전역: $stG")
    Lval = sum(zs) + objective_value(D.G) + constG

    # --- 부분기울기: ∂L/∂ν_i = b_i − a_iᵀx̂,  ∂L/∂μ_ks = α̂_k − α̂^s_k ---
    gν = zeros(length(D.links))
    for (i, L) in enumerate(D.links)
        ax = 0.0
        for (v, c) in L.gterms; ax += c * value(v); end
        for (s, v, c) in L.sterms; ax += c * value(v); end
        gν[i] = L.rhs - ax
    end
    αhat = [value(αG[k]) for k in 1:K]
    αsv = [value(D.αs[k, s]) for k in 1:K, s in 1:S]
    gμ = [αhat[k] - αsv[k, s] for k in 1:K, s in 1:S]
    return (Lval, gν, gμ, αsv)
end


function _project_ν!(D::_LagDecomp, ν)
    for (i, L) in enumerate(D.links)
        L.sense == :le && (ν[i] = max(ν[i], 0.0))
        L.sense == :ge && (ν[i] = min(ν[i], 0.0))
    end
end


"""
노드 bound: 부모 승수에서 출발, projected subgradient (Polyak) n_sg회. 최선 L과 그 승수 반환.
"""
function _lag_node_bound!(D::_LagDecomp, ν0, μ0; LB, n_sg=5, θ=1.0)
    ν, μ = copy(ν0), copy(μ0)
    best = (Inf, ν0, μ0, nothing)
    for it in 0:n_sg
        Lval, gν, gμ, αsv = _lag_eval!(D, ν, μ)
        Lval == -Inf && return (-Inf, ν, μ, nothing)
        if Lval < best[1]
            best = (Lval, copy(ν), copy(μ), αsv)
        end
        (it == n_sg || !isfinite(Lval)) && break
        nrm = sum(abs2, gν) + sum(abs2, gμ)
        nrm < 1e-16 && break
        target = isfinite(LB) ? LB : Lval - 0.01 * max(1.0, abs(Lval))
        t = θ * max(Lval - target, 1e-6 * max(1.0, abs(Lval))) / nrm
        ν .-= t .* gν
        μ .-= t .* gμ
        _project_ν!(D, ν)
    end
    return best
end


"""
    alpha_bnb_lagrange(td, x̄; lp_optimizer, time_limit, rel_gap, n_sg, verbose)

α-B&B (full RLT) 에서 노드 bound를 시나리오 라그랑지 분해로 계산.
root만 통째 LP (Method=2, crossover 없음)로 풀어 승수 초기화.
"""
function alpha_bnb_lagrange(td, x̄; lp_optimizer, time_limit=300.0, rel_gap=1e-4, n_sg=5,
                            verbose=true, log_every=30.0, min_width=1e-7, root_method=2)
    t0 = time()
    K, S, w = td.num_arcs, td.S, td.w
    A = _build_alpha_lp(td, x̄; optimizer=lp_optimizer, rlt=:full)
    E = _build_alpha_lp(td, x̄; optimizer=lp_optimizer, rlt=:basic)
    D = _build_lag_decomp(A, td)
    t_build = time() - t0

    set_optimizer_attribute(A.model, "Method", root_method)
    root_method == 2 && set_optimizer_attribute(A.model, "Crossover", 0)
    t_root = @elapsed (zroot, ν0, μ0) = _lag_init_multipliers(D)
    Lroot, _, _, αsv0 = _lag_eval!(D, ν0, μ0)
    verbose && @printf("    [α-Lag] build=%.1fs rootLP=%.6f (%.1fs)  L(root duals)=%.6f  links=%d\n",
                       t_build, zroot, t_root, Lroot, length(D.links))
    flush(stdout)

    tol(v) = rel_gap * (isfinite(v) ? max(1.0, abs(v)) : 1.0)
    LB, best_α = -Inf, zeros(K)
    open = Any[(zeros(K), fill(w, K), zroot, ν0, μ0, αsv0)]
    nodes = 0
    t_log = time()
    open_UB() = isempty(open) ? LB : max(LB, maximum(n[3] for n in open))
    history = Tuple{Float64,Float64,Float64,Int}[]

    while !isempty(open)
        UB = open_UB()
        UB - LB <= tol(LB) && break
        time() - t0 > time_limit && break
        i = argmax([n[3] for n in open])
        l, u, ubp, ν, μ, αsv = popat!(open, i)
        ubp <= LB + tol(LB) && continue
        sum(l) > w + 1e-9 && continue

        if nodes == 0
            z = ubp       # root는 이미 계산
        else
            _lag_set_box!(D, l, u)
            z, ν, μ, αsv = _lag_node_bound!(D, ν, μ; LB=LB, n_sg=n_sg)
            z = min(z, ubp)
        end
        nodes += 1
        (z <= LB + tol(LB) || αsv === nothing) && continue

        # 하한: 복사본 평균 α̂ (상자 내, Σ ≤ w) 고정 LP
        α̂ = clamp.(vec(sum(αsv; dims=2)) ./ S, l, u)
        sum(α̂) > w && (α̂ .*= w / sum(α̂); α̂ = clamp.(α̂, l, u))
        _set_box!(E, α̂, α̂)
        zE = _solve_alpha_lp!(E)
        zE > LB && (LB = zE; best_α = copy(α̂))

        # 분기: 시나리오 복사본의 퍼짐(max−min)이 크고 상자 폭이 남은 α_k
        spread = [(u[k] - l[k] > min_width) ? (maximum(αsv[k, :]) - minimum(αsv[k, :]) + 1e-3 * (u[k] - l[k])) : 0.0
                  for k in 1:K]
        k = argmax(spread)
        spread[k] <= 0 && continue
        t = α̂[k]
        width = u[k] - l[k]
        (t - l[k] < 0.1 * width || u[k] - t < 0.1 * width) && (t = 0.5 * (l[k] + u[k]))
        u1 = copy(u); u1[k] = t
        l2 = copy(l); l2[k] = t
        push!(open, (copy(l), u1, z, ν, μ, αsv))
        push!(open, (l2, copy(u), z, ν, μ, αsv))

        if verbose && time() - t_log > log_every
            UBn = open_UB()
            @printf("    [α-Lag] t=%6.1fs nodes=%5d open=%5d LB=%.6f UB=%.6f gap=%.3e\n",
                    time() - t0, nodes, length(open), LB, UBn, (UBn - LB) / max(1.0, abs(LB)))
            flush(stdout); t_log = time()
        end
        push!(history, (time() - t0, LB, open_UB(), nodes))
    end
    UB = open_UB()
    return Dict(:LB => LB, :UB => UB, :α => best_α, :nodes => nodes, :time => time() - t0,
                :t_build => t_build, :root_UB => zroot, :L_root => Lroot,
                :is_exact => (UB - LB <= tol(LB)), :history => history)
end
