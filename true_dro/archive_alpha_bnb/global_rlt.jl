"""
global_rlt.jl — Gurobi global(Ω, NonConvex=2)에 α-B&B root의 full RLT 행을 영구 제약으로 추가.

α 상자가 전체 범위 [0, w]일 때의 RLT 행은 문제 전체에서 유효:
  인수 α_k ≥ 0, w − α_k ≥ 0  ×  belief 다면체 제약 (TV 볼, CVaR, 상·하한)
새 곱 변수 Zb = α·b, Ze = α·e, (CVaR) Za = α·a 는 bilinear 등식으로 정의 → Gurobi가 곱으로 인식.
Ω 원본(build_true_dro_subproblem)은 수정하지 않음.
"""

using JuMP

function build_omega_rlt(td, x̄; optimizer, silent=true)
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
    Z[:b] = prodvar("Zb", Y[:b])
    Z[:e] = prodvar("Ze", Y[:e])
    if cvar
        Y[:r] = vec(vars[:r]); Z[:r] = vars[:ζL]; Z[:a] = prodvar("Za", Y[:a])
    else
        Z[:a] = vars[:ζL]
    end
    rows = _belief_rows(td, Y; full=true)
    for k in 1:K, row in rows
        _add_rlt_row!(model, α, Y, Z, k, row, 0.0, w)
    end
    for fam in (:a, :d, :r)
        haskey(Y, fam) || continue
        @constraint(model, [k = 1:K], sum(Z[fam][k, s] for s in 1:S) == α[k])
        @constraint(model, [s = 1:S], sum(Z[fam][k, s] for k in 1:K) <= w * Y[fam][s])
    end
    return model, vars
end


"""
    build_omega_pwl(td, x̄; optimizer, bps)

Ω + root RLT 에 더해, bps[k] (0 = t₀ < t₁ < … < t_P = w) 로 분할한 α_k 마다
구간 binary z_kp 와 구간별 RLT 를 추가 (원래 bilinear 등식 유지 → exact 모델).
  α_k = Σ_p α_kp,  t_{p−1} z_kp ≤ α_kp ≤ t_p z_kp,  Σ_p z_kp = 1
  Z_ks = Σ_p Z_kps  (Z_kps ≈ α_kp·y_s),  W_kps ≈ z_kp·y_s
  belief 행 g(y) ≥ 0 마다:  (α_kp − t_{p−1}z_kp)·g ≥ 0,  (t_p z_kp − α_kp)·g ≥ 0,
                            z_kp·g ≥ 0,  (1 − z_kp)·g ≥ 0      (선형화)
z 가 정수면 W = z·y 가 exact, 활성 구간 p 에서 노드 RLT 와 같은 완화가 됨.
"""
function build_omega_pwl(td, x̄; optimizer, bps::Dict{Int,Vector{Float64}}, silent=true)
    model, vars = build_omega_rlt(td, x̄; optimizer=optimizer, silent=silent)
    K, S = td.num_arcs, td.S
    α = vars[:α]
    cvar = vars[:_use_cvar]
    # build_omega_rlt 와 같은 Y, Z 재구성
    Y = Dict{Symbol,Vector{VariableRef}}(:a => vec(vars[:a]), :b => vec(vars[:b]),
                                         :d => vec(vars[:d]), :e => vec(vars[:e]))
    cvar && (Y[:r] = vec(vars[:r]))
    Zall = Dict{Symbol,Matrix{VariableRef}}()
    Zall[:d] = vars[:ζF]
    allv = all_variables(model)
    getZ(name) = reshape([v for v in allv if startswith(JuMP.name(v), name * "[")], K, S)
    Zall[:b] = getZ("Zb"); Zall[:e] = getZ("Ze")
    if cvar
        Zall[:r] = vars[:ζL]; Zall[:a] = getZ("Za")
    else
        Zall[:a] = vars[:ζL]
    end
    rows = _belief_rows(td, Y; full=true)
    for (k, t) in bps
        P = length(t) - 1
        z = @variable(model, [1:P], Bin, base_name = "zpwl_$k")
        αp = @variable(model, [1:P], lower_bound = 0.0)
        @constraint(model, sum(z) == 1)
        @constraint(model, α[k] == sum(αp))
        for p in 1:P
            @constraint(model, t[p] * z[p] <= αp[p])
            @constraint(model, αp[p] <= t[p+1] * z[p])
        end
        Zp = Dict(f => @variable(model, [1:P, 1:S], lower_bound = 0.0) for f in keys(Y))
        Wp = Dict(f => @variable(model, [1:P, 1:S], lower_bound = 0.0) for f in keys(Y))
        for f in keys(Y)
            @constraint(model, [s = 1:S], Zall[f][k, s] == sum(Zp[f][p, s] for p in 1:P))
            if f in (:a, :d, :r)
                @constraint(model, [p = 1:P], sum(Zp[f][p, s] for s in 1:S) == αp[p])
                @constraint(model, [p = 1:P], sum(Wp[f][p, s] for s in 1:S) == z[p])
            end
        end
        for row in rows, p in 1:P
            GW = row.c0 * z[p] + sum(c * Wp[f][p, s] for (f, s, c) in row.terms)
            GZ = row.c0 * αp[p] + sum(c * Zp[f][p, s] for (f, s, c) in row.terms)
            Gy = row.c0 + sum(c * Y[f][s] for (f, s, c) in row.terms)
            @constraint(model, GZ - t[p] * GW >= 0)
            @constraint(model, t[p+1] * GW - GZ >= 0)
            @constraint(model, GW >= 0)
            @constraint(model, Gy - GW >= 0)
        end
    end
    return model, vars
end


"""root full-RLT LP 에서 완화 위반이 큰 α_k 를 골라 α̂_k 에서 2분할한 breakpoint."""
function pwl_breakpoints(td, x̄; lp_optimizer, m=td.num_arcs, minfrac=0.1)
    A = _build_alpha_lp(td, x̄; optimizer=lp_optimizer, rlt=:full)
    try set_optimizer_attribute(A.model, "Method", 2); set_optimizer_attribute(A.model, "Crossover", 0) catch; end
    _solve_alpha_lp!(A)
    K, S, w = A.K, A.S, td.w
    α̂ = clamp.(value.(A.vars[:α]), 0.0, w)
    rv = value.(A.lead); dv = value.(A.fol)
    ζLv = value.(A.vars[:ζL]); ζFv = value.(A.vars[:ζF])
    viol = [sum(abs(ζLv[k, s] - α̂[k] * rv[s]) + abs(ζFv[k, s] - α̂[k] * dv[s]) for s in 1:S) for k in 1:K]
    ks = sortperm(viol; rev=true)[1:min(m, K)]
    bps = Dict{Int,Vector{Float64}}()
    for k in ks
        viol[k] <= 1e-9 && continue
        t = α̂[k]
        (t < minfrac * w || t > (1 - minfrac) * w) && (t = 0.5 * w)
        bps[k] = [0.0, t, w]
    end
    return bps, viol
end
