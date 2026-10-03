"""
global_rlt_cb.jl — Gurobi global (Ω, NonConvex) 에 root RLT 를 callback user cut 으로 분리 추가.

모델 = Ω + 곱 변수 (Zb = α·b, Ze = α·e, CVaR 이면 Za = α·a; bilinear 등식) + 정적 RLT (Σ_s Z = α, Σ_k Z ≤ w y).
root RLT 행 (상자 [0, w] 기준, 전역 유효) 은 모델에 넣지 않고, MIPNODE 에서 위반된 것만 user cut 으로.
global_rlt.jl 의 build_omega_rlt (전체 주입) 과 비교용.
"""

using JuMP, Printf
import Gurobi

function build_omega_rlt_cb(td, x̄; optimizer, maxcuts=500, tol=1e-6, silent=true,
                            root_rounds=typemax(Int), maxcuts_node=maxcuts)
    model, vars = build_true_dro_subproblem(td, x̄; optimizer=optimizer, silent=silent)
    K, S, w = td.num_arcs, td.S, td.w
    α = vars[:α]
    cvar = vars[:_use_cvar]
    Y = Dict{Symbol,Vector{VariableRef}}(:a => vec(vars[:a]), :b => vec(vars[:b]),
                                         :d => vec(vars[:d]), :e => vec(vars[:e]))
    Z = Dict{Symbol,Matrix{VariableRef}}(:d => vars[:ζF])
    function prodvar(name, y)
        Zv = @variable(model, [k = 1:K, s = 1:S], lower_bound = 0.0,
                       upper_bound = w * upper_bound(y[s]), base_name = name)
        @constraint(model, [k = 1:K, s = 1:S], Zv[k, s] == α[k] * y[s])
        return Zv
    end
    Z[:b] = prodvar("Zb", Y[:b]); Z[:e] = prodvar("Ze", Y[:e])
    if cvar
        Y[:r] = vec(vars[:r]); Z[:r] = vars[:ζL]; Z[:a] = prodvar("Za", Y[:a])
    else
        Z[:a] = vars[:ζL]
    end
    for fam in (:a, :d, :r)
        haskey(Y, fam) || continue
        @constraint(model, [k = 1:K], sum(Z[fam][k, s] for s in 1:S) == α[k])
        @constraint(model, [s = 1:S], sum(Z[fam][k, s] for k in 1:K) <= w * Y[fam][s])
    end
    rows = _belief_rows(td, Y; full=true)

    added = Set{Tuple{Int,Int,Bool}}()
    stats = Dict(:calls => 0, :cuts => 0, :root_calls => 0)
    function cb(cb_data)
        stats[:calls] += 1
        # 노드 번호: root (0) 에서는 root_rounds 번까지만 cut 추가 → 분기로 빨리 넘어가게
        nodP = Ref{Cdouble}()
        Gurobi.GRBcbget(cb_data, cb_data.cb_where, Gurobi.GRB_CB_MIPNODE_NODCNT, nodP)
        atroot = nodP[] < 0.5
        if atroot
            stats[:root_calls] += 1
            stats[:root_calls] > root_rounds && return
        end
        cap = atroot ? maxcuts : maxcuts_node
        αv = [callback_value(cb_data, α[k]) for k in 1:K]
        yv = Dict(f => [callback_value(cb_data, y[s]) for s in 1:S] for (f, y) in Y)
        zv = Dict(f => [callback_value(cb_data, Z[f][k, s]) for k in 1:K, s in 1:S] for f in keys(Z))
        V = Tuple{Float64,Int,Int,Bool}[]
        GZ = zeros(K)
        for (j, row) in enumerate(rows)
            Gy = row.c0
            for (f, s, c) in row.terms; Gy += c * yv[f][s]; end
            GZ .= row.c0 .* αv
            for (f, s, c) in row.terms, k in 1:K
                GZ[k] += c * zv[f][k, s]
            end
            for k in 1:K
                GZ[k] < -tol && push!(V, (GZ[k], k, j, true))              # (α_k − 0)·g ≥ 0
                (w * Gy - GZ[k]) < -tol && push!(V, (w * Gy - GZ[k], k, j, false))   # (w − α_k)·g ≥ 0
            end
        end
        sort!(V; by=first)
        n = 0
        for (v, k, j, islo) in V
            (k, j, islo) in added && continue
            row = rows[j]
            GZe = row.c0 * α[k] + sum(c * Z[f][k, s] for (f, s, c) in row.terms)
            Gye = row.c0 + sum(c * Y[f][s] for (f, s, c) in row.terms)
            con = islo ? @build_constraint(GZe >= 0) : @build_constraint(w * Gye - GZe >= 0)
            MOI.submit(model, MOI.UserCut(cb_data), con)
            push!(added, (k, j, islo)); n += 1
            n >= cap && break
        end
        stats[:cuts] += n
    end
    MOI.set(model, MOI.UserCutCallback(), cb)
    set_optimizer_attribute(model, "PreCrush", 1)
    vars[:_rltZ] = Z; vars[:_rltY] = Y
    return model, vars, stats
end


"""
MIP start: α̂ 고정 Ω 해 (E: rlt=:basic, 상자 [α̂,α̂]) 를 모델의 모든 변수 시작값으로.
곱 변수 Z 는 α̂·y. 반환 시작해의 목적값.
"""
function rlt_cb_mip_start!(model, vars, E::_AlphaLP, α̂)
    _set_box!(E, α̂, α̂)
    z = _solve_alpha_lp!(E)
    for (name, X) in vars
        haskey(E.vars, name) || continue
        X isa VariableRef && (set_start_value(X, value(E.vars[name])); continue)
        X isa AbstractArray{VariableRef} || continue
        EX = E.vars[name]
        for I in eachindex(X)
            set_start_value(X[I], value(EX[I]))
        end
    end
    Z, Y = vars[:_rltZ], vars[:_rltY]
    for (f, Zf) in Z, k in axes(Zf, 1), s in axes(Zf, 2)
        f in (:b, :e) || (f == :a && Zf !== vars[:ζL]) || continue
        set_start_value(Zf[k, s], α̂[k] * value(E.vars[f == :a ? :a : f][s]))
    end
    return z
end
