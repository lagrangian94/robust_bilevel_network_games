"""
alpha_heur.jl — RLT 완화 해에서 좋은 해(LB) 만들기.
Zhen et al. (IJOC 2022) "binding scenario + mountain climbing" 을 Ω 에 맞게 옮김.

  후보:  RLT 해의 곱 변수 Z_ks ≈ α_k·y_s 에서  α^(s) = Z_{·s} / y_s   (y = lead(r 또는 a), d)
         α ≥ 0, Σα ≤ w 로 사영 → 어떤 α 든 α 고정 LP 값이 실현 가능 값 (valid LB)
  mountain climbing:
         α 고정 LP  (belief 와 나머지 최적화)  ↔  belief 고정 LP (α 와 나머지 최적화)
         번갈아 풀면 목적값 단조 증가.
"""

using JuMP, Printf

mutable struct _BeliefFixLP
    model::Model
    vars::Dict
    cL::Matrix{ConstraintRef}      # ζL_ks − lead_s·α_k = 0
    cF::Matrix{ConstraintRef}      # ζF_ks − d_s·α_k = 0
    fam::Vector{Symbol}            # 고정할 belief 변수 이름
end


function _build_belief_fix_lp(td, x̄; optimizer)
    model, vars = build_true_dro_subproblem(td, x̄; optimizer=optimizer, silent=true)
    for name in (:ζL_def, :ζF_def)
        delete.(model, model[name]); unregister(model, name)
    end
    K, S = td.num_arcs, td.S
    α, ζL, ζF = vars[:α], vars[:ζL], vars[:ζF]
    cvar = vars[:_use_cvar]
    cL = Matrix{ConstraintRef}(undef, K, S)
    cF = Matrix{ConstraintRef}(undef, K, S)
    for k in 1:K, s in 1:S
        cL[k, s] = @constraint(model, ζL[k, s] - 0.0 * α[k] == 0)
        cF[k, s] = @constraint(model, ζF[k, s] - 0.0 * α[k] == 0)
    end
    fam = cvar ? [:a, :b, :r, :d, :e] : [:a, :b, :d, :e]
    return _BeliefFixLP(model, vars, cL, cF, fam)
end


_belief_values(vars, fam) = Dict(f => value.(vec(vars[f])) for f in fam)


"""belief 고정 LP. 반환 (값, α)."""
function _belief_step!(B::_BeliefFixLP, yv)
    K, S = size(B.cL)
    α = B.vars[:α]
    lead = haskey(yv, :r) ? yv[:r] : yv[:a]
    for k in 1:K, s in 1:S
        set_normalized_coefficient(B.cL[k, s], α[k], -lead[s])
        set_normalized_coefficient(B.cF[k, s], α[k], -yv[:d][s])
    end
    for f in B.fam
        X = vec(B.vars[f])
        for s in 1:S
            fix(X[s], yv[f][s]; force=true)
        end
    end
    optimize!(B.model)
    termination_status(B.model) == MOI.OPTIMAL || error("belief 고정 LP: $(termination_status(B.model))")
    return objective_value(B.model), value.(α)
end


"""α 고정 LP (E: rlt=:basic, 상자 [α,α] 이면 exact). 반환 (값, belief 값)."""
function _alpha_step!(E::_AlphaLP, α, fam)
    _set_box!(E, α, α)
    z = _solve_alpha_lp!(E)
    return z, _belief_values(E.vars, fam)
end


function _project_simplex_cap(α, w)
    a = max.(α, 0.0)
    s = sum(a)
    s > w && (a .*= w / s)
    return a
end


"""RLT 해 (P: _SepLP 또는 _AlphaLP with Y/Z) 에서 후보 α 목록. y 질량이 큰 순, 중복 제거."""
function _alpha_candidates(P, w; N=10, ymin=1e-6, digits=4)
    αv = value.(P.vars[:α])
    cands = [(Inf, _project_simplex_cap(αv, w))]
    for f in (:r, :a, :d)
        haskey(P.Y, f) || continue
        yv = value.(P.Y[f]); Zv = value.(P.Z[f])
        for s in eachindex(yv)
            yv[s] > ymin || continue
            push!(cands, (yv[s], _project_simplex_cap(Zv[:, s] ./ yv[s], w)))
        end
    end
    sort!(cands; by=c -> -c[1])
    out = Vector{Vector{Float64}}()
    seen = Set{Vector{Float64}}()
    for (_, a) in cands
        key = round.(a; digits=digits)
        key in seen && continue
        push!(seen, key); push!(out, a)
        length(out) >= N && break
    end
    return out
end


"""mountain climbing. 반환 (최선 값, 최선 α, 반복 수)."""
function _mountain_climb!(E::_AlphaLP, B::_BeliefFixLP, α0; maxit=10, tol=1e-7)
    z, yv = _alpha_step!(E, α0, B.fam)
    best, bestα = z, copy(α0)
    for it in 1:maxit
        zb, α = _belief_step!(B, yv)
        α = max.(α, 0.0)                     # 수치 보정 (belief 고정 LP 해는 이미 Σα ≤ w)
        z, yv = _alpha_step!(E, α, B.fam)
        if z > best + tol
            best, bestα = z, copy(α)
        else
            return best, bestα, it
        end
    end
    return best, bestα, maxit
end


"""
후보 평가 + 상위 n_climb 개 mountain climbing. 반환 (최선 값, 최선 α, 평가 수).
"""
function rlt_primal_heuristic!(P, E::_AlphaLP, B::_BeliefFixLP, w; N=10, n_climb=2, maxit=10)
    cands = _alpha_candidates(P, w; N=N)
    vals = Float64[]
    for a in cands
        z, _ = _alpha_step!(E, a, B.fam)
        push!(vals, z)
    end
    order = sortperm(vals; rev=true)
    best, bestα = vals[order[1]], cands[order[1]]
    for i in order[1:min(n_climb, length(order))]
        z, α, _ = _mountain_climb!(E, B, cands[i]; maxit=maxit)
        z > best && (best, bestα = z, α)
    end
    return best, bestα, length(cands)
end


using Ipopt

_capw(a, w) = (b = max.(a, 0.0); s = sum(b); s > w ? b .* (w / s) : b)

"""
Ω 를 Ipopt 로 로컬 최적화. 시작점 = α̂ 고정 LP 해 (E). 반환 로컬 해의 α (Σα ≤ w 로 사영).
값은 호출 측에서 α 고정 LP 로 다시 평가 (NLP 허용오차와 무관하게 valid LB).
"""
function ipopt_local_alpha(td, x̄, E::_AlphaLP, α̂; max_time=120.0, deadline=Inf)
    _set_box!(E, α̂, α̂)
    _solve_alpha_lp!(E)
    m, v = build_true_dro_subproblem(td, x̄; optimizer=Ipopt.Optimizer, silent=true)
    delete!(unsafe_backend(m).options, "DualReductions")      # 빌더가 넣는 Gurobi 전용 옵션 제거
    set_optimizer_attribute(m, "max_wall_time", max(max_time, 1.0))   # 병렬 실행 중 CPU≠wall
    set_optimizer_attribute(m, "print_level", 0)
    set_optimizer_attribute(m, "warm_start_init_point", "yes")
    set_optimizer_attribute(m, "mu_init", 1e-4)
    # 반복마다 마감 시각 확인 → 넘으면 즉시 중단 (반복 하나가 길어 wall_time 만으로는 초과 가능)
    MOI.set(m, Ipopt.CallbackFunction(), (args...) -> time() < deadline)
    for (name, X) in v
        haskey(E.vars, name) || continue
        X isa VariableRef && (set_start_value(X, value(E.vars[name])); continue)
        X isa AbstractArray{VariableRef} || continue
        for I in eachindex(X); set_start_value(X[I], value(E.vars[name][I])); end
    end
    optimize!(m)
    has_values(m) || return nothing
    return _capw(value.(v[:α]), td.w)
end
