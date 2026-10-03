"""
alpha_dw.jl — α-B&B 노드 LP (full RLT) 를 Dantzig–Wolfe column generation 으로 시나리오별 분해.

alpha_lagrange.jl 의 _LagDecomp (α 복사 α^s, 시나리오 모델 M[s], 전역 모델 G, 연결 행) 를 재사용.
  master : 전역 변수 (α, δ, ρ⁰) + 블록별 열 λ (극점) / 극방향 열
           연결 행 (Σa=1, Σb≤2ε, Σ_s β ≤ δ, ...), 복사 행 Σ λ α^{s,p} − α = 0, 볼록결합 Σ_p λ_sp = 1
           초기 실현 가능성을 위해 연결/복사 행에 큰 벌점의 인공 변수.
  pricing: master dual (ν = −dual, max 규약) 로 _lag_eval! 과 같은 시나리오 LP S개 (병렬)
  bound  : 매 반복 L(ν, μ) = Σ_s pricing 값 + 전역 LP 값 + 상수  — 어떤 승수에서도 valid UB.
"""

using JuMP, Printf

mutable struct _DWMaster
    model::Model
    xg::Dict{VariableRef,VariableRef}        # G 변수 → master 변수
    link::Vector{ConstraintRef}
    copy::Matrix{ConstraintRef}              # K × S
    conv::Vector{ConstraintRef}              # S
    ncols::Vector{Int}
    nrays::Vector{Int}
end


function _build_dw_master(D::_LagDecomp, optimizer; Mpen=1e4)
    K, S = D.K, D.S
    m = Model(optimizer); set_silent(m)
    xg = Dict{VariableRef,VariableRef}()
    for v in all_variables(D.G)
        nv = @variable(m)
        has_lower_bound(v) && set_lower_bound(nv, lower_bound(v))
        has_upper_bound(v) && set_upper_bound(nv, upper_bound(v))
        xg[v] = nv
    end
    for (F, Sset) in list_of_constraint_types(D.G)
        F == VariableRef && continue
        for c in all_constraints(D.G, F, Sset)
            o = constraint_object(c)
            aff = AffExpr(0.0)
            for (v, coef) in o.func.terms; add_to_expression!(aff, coef, xg[v]); end
            @constraint(m, aff in o.set)
        end
    end
    obj = AffExpr(0.0)
    for (v, c) in D.fobj_g; add_to_expression!(obj, c, xg[v]); end
    link = ConstraintRef[]
    for L in D.links
        aff = AffExpr(0.0)
        for (v, c) in L.gterms; add_to_expression!(aff, c, xg[v]); end
        if L.sense == :eq
            ap = @variable(m, lower_bound = 0.0); an = @variable(m, lower_bound = 0.0)
            add_to_expression!(aff, 1.0, ap); add_to_expression!(aff, -1.0, an)
            add_to_expression!(obj, -Mpen, ap); add_to_expression!(obj, -Mpen, an)
            push!(link, @constraint(m, aff == L.rhs))
        elseif L.sense == :le
            a = @variable(m, lower_bound = 0.0); add_to_expression!(aff, -1.0, a); add_to_expression!(obj, -Mpen, a)
            push!(link, @constraint(m, aff <= L.rhs))
        else
            a = @variable(m, lower_bound = 0.0); add_to_expression!(aff, 1.0, a); add_to_expression!(obj, -Mpen, a)
            push!(link, @constraint(m, aff >= L.rhs))
        end
    end
    αA = D.A.vars[:α]
    copy = Matrix{ConstraintRef}(undef, K, S)
    for k in 1:K, s in 1:S
        ap = @variable(m, lower_bound = 0.0); an = @variable(m, lower_bound = 0.0)
        add_to_expression!(obj, -Mpen, ap); add_to_expression!(obj, -Mpen, an)
        copy[k, s] = @constraint(m, -xg[D.gmap[αA[k]]] + ap - an == 0)
    end
    # 볼록결합 Σ_p λ_sp = 1 — 열이 없을 때를 위해 인공 변수로 시작
    conv = Vector{ConstraintRef}(undef, S)
    for s in 1:S
        a = @variable(m, lower_bound = 0.0)
        add_to_expression!(obj, -Mpen, a)
        conv[s] = @constraint(m, a == 1)
    end
    @objective(m, Max, obj)
    try set_optimizer_attribute(m, "Method", 0) catch; end     # 열 추가 → primal simplex warm
    return _DWMaster(m, xg, link, copy, conv, zeros(Int, S), zeros(Int, S))
end


"""블록 s 의 해(또는 극방향) 값에서 master 열 추가. isray 면 볼록결합 계수 0."""
function _dw_add_column!(Mst::_DWMaster, D::_LagDecomp, s, linkterms_s, val, isray)
    m = Mst.model
    cobj = 0.0
    for (v, c) in D.fobj_s[s]; cobj += c * val(v); end
    λ = @variable(m, lower_bound = 0.0)
    set_objective_coefficient(m, λ, cobj)
    acc = Dict{Int,Float64}()
    for (i, v, c) in linkterms_s
        acc[i] = get(acc, i, 0.0) + c * val(v)
    end
    for (i, a) in acc
        abs(a) > 1e-12 && set_normalized_coefficient(Mst.link[i], λ, a)
    end
    for k in 1:D.K
        a = val(D.αs[k, s])
        abs(a) > 1e-12 && set_normalized_coefficient(Mst.copy[k, s], λ, a)
    end
    isray || set_normalized_coefficient(Mst.conv[s], λ, 1.0)
    isray ? (Mst.nrays[s] += 1) : (Mst.ncols[s] += 1)
end


"""
DW 로 노드 LP 풀기 (D 의 현재 상자 기준). 반환 Dict(:UB 최선 Lagrangian bound, :master 값, :iters, :time, :trace)
"""
function _dw_linkterms(D::_LagDecomp)
    linkterms = [Tuple{Int,VariableRef,Float64}[] for _ in 1:D.S]
    for (i, L) in enumerate(D.links), (s, v, c) in L.sterms
        push!(linkterms[s], (i, v, c))
    end
    return linkterms
end


"""
α̂ 고정 Ω 해 (E: rlt=:basic, 상자 [α̂,α̂]) 를 시나리오별로 쪼개 초기 열로 추가.
α 가 고정되면 RLT 행이 정확히 성립 (Z = α̂·y) → 노드 LP 의 실현 가능해 → master 가 처음부터 실현 가능.
α̂ 는 노드 상자 안에 있어야 함.
"""
function _dw_seed!(Mst::_DWMaster, D::_LagDecomp, E::_AlphaLP, α̂)
    A = D.A
    _set_box!(E, α̂, α̂)
    _solve_alpha_lp!(E) == -Inf && return false
    valA = Dict{VariableRef,Float64}()
    for (name, X) in A.vars
        haskey(E.vars, name) || continue
        X isa VariableRef && (valA[X] = value(E.vars[name]); continue)
        X isa AbstractArray{VariableRef} || continue
        EX = E.vars[name]
        for I in eachindex(X)
            valA[X[I]] = value(EX[I])
        end
    end
    for (f, Z) in A.Z
        y = A.Y[f]
        for k in axes(Z, 1), s in axes(Z, 2)
            haskey(valA, Z[k, s]) || (valA[Z[k, s]] = α̂[k] * valA[y[s]])
        end
    end
    linkterms = _dw_linkterms(D)
    for s in 1:D.S
        inv = Dict{VariableRef,Float64}()
        for (vA, vM) in D.smap[s]; inv[vM] = valA[vA]; end
        for k in 1:D.K; inv[D.αs[k, s]] = α̂[k]; end
        _dw_add_column!(Mst, D, s, linkterms[s], v -> inv[v], false)
    end
    return true
end


function dw_solve_node!(D::_LagDecomp, Mst::_DWMaster; max_iter=500, rtol=1e-5, time_budget=300.0,
                        verbose=true, log_every=10)
    t0 = time()
    K, S = D.K, D.S
    for s in 1:S
        set_optimizer_attribute(D.M[s], "InfUnbdInfo", 1)
    end
    linkterms = _dw_linkterms(D)
    bestUB = Inf
    trace = Tuple{Int,Float64,Float64,Float64}[]
    zmaster = -Inf
    ν = zeros(length(D.links)); μ = zeros(K, S); η = zeros(S)
    for it in 1:max_iter
        if it > 1 || sum(Mst.ncols) > 0
            optimize!(Mst.model)
            termination_status(Mst.model) == MOI.OPTIMAL || error("DW master: $(termination_status(Mst.model))")
            zmaster = objective_value(Mst.model)
            ν = [-dual(c) for c in Mst.link]
            μ = [-dual(Mst.copy[k, s]) for k in 1:K, s in 1:S]
            η = [-dual(Mst.conv[s]) for s in 1:S]
        end
        # --- pricing: 시나리오 LP (목적 = f_s − ν·a_s − μ·α^s), 병렬 ---
        Lval, nadd, nray = _dw_pricing!(D, Mst, ν, μ, η, linkterms)
        bestUB = min(bestUB, Lval)
        push!(trace, (it, zmaster, Lval, time() - t0))
        if verbose && (it % log_every == 1 || nadd == 0)
            @printf("    [DW] it=%4d master=%.6f L=%.6f bestUB=%.6f cols+%d rays+%d (%.1fs)\n",
                    it, zmaster, Lval, bestUB, nadd, nray, time() - t0); flush(stdout)
        end
        nadd + nray == 0 && break
        isfinite(bestUB) && isfinite(zmaster) && bestUB - zmaster <= rtol * max(1.0, abs(bestUB)) && break
        time() - t0 > time_budget && break
    end
    return Dict(:UB => bestUB, :master => zmaster, :iters => length(trace), :time => time() - t0, :trace => trace)
end


function _dw_pricing!(D::_LagDecomp, Mst::_DWMaster, ν, μ, η, linkterms; rc_tol=1e-7)
    K, S = D.K, D.S
    objS = [AffExpr(0.0) for _ in 1:S]
    for s in 1:S
        for (v, c) in D.fobj_s[s]; add_to_expression!(objS[s], c, v); end
        for k in 1:K; add_to_expression!(objS[s], -μ[k, s], D.αs[k, s]); end
        for (i, v, c) in linkterms[s]
            ν[i] == 0.0 || add_to_expression!(objS[s], -ν[i] * c, v)
        end
        @objective(D.M[s], Max, objS[s])
    end
    zs = zeros(S); stat = fill(:ok, S)
    tasks = map(D.chunks) do rg
        Threads.@spawn for s in rg
            optimize!(D.M[s])
            st = termination_status(D.M[s])
            if st == MOI.OPTIMAL
                zs[s] = objective_value(D.M[s])
            elseif st == MOI.DUAL_INFEASIBLE
                stat[s] = :ray
            elseif st == MOI.INFEASIBLE
                stat[s] = :infeasible
            else
                error("DW pricing $s: $st")
            end
        end
    end
    foreach(fetch, tasks)
    any(==(:infeasible), stat) && error("DW: 블록 infeasible (노드 infeasible)")
    nadd = 0; nray = 0
    for s in 1:S
        if stat[s] == :ray
            primal_status(D.M[s]) == MOI.INFEASIBILITY_CERTIFICATE || error("DW: 극방향 없음 (블록 $s)")
            _dw_add_column!(Mst, D, s, linkterms[s], v -> value(v), true); nray += 1
        elseif zs[s] - η[s] > rc_tol * max(1.0, abs(zs[s]))
            _dw_add_column!(Mst, D, s, linkterms[s], v -> value(v), false); nadd += 1
        end
    end
    # Lagrangian bound = Σ_s z_s + max_{전역}[...] + Σ ν_i b_i   (ray 가 있으면 +∞)
    nray > 0 && return (Inf, nadd, nray)
    objG = AffExpr(0.0)
    for (v, c) in D.fobj_g; add_to_expression!(objG, c, v); end
    constG = 0.0
    for (i, L) in enumerate(D.links)
        ν[i] == 0.0 && continue
        constG += ν[i] * L.rhs
        for (v, c) in L.gterms; add_to_expression!(objG, -ν[i] * c, v); end
    end
    for k in 1:K
        add_to_expression!(objG, sum(μ[k, s] for s in 1:S), D.gmap[D.A.vars[:α][k]])
    end
    @objective(D.G, Max, objG)
    optimize!(D.G)
    stG = termination_status(D.G)
    stG == MOI.DUAL_INFEASIBLE && return (Inf, nadd, nray)
    stG == MOI.OPTIMAL || error("DW 전역 LP: $stG")
    return (sum(zs) + objective_value(D.G) + constG, nadd, nray)
end
