"""
true_dro_structured_eval.jl — ISP 정확 평가의 구조 활용 MILP 경로 (paper/joc/ccg.tex).

Benders의 exact cut 단계 (M1, global bilinear solve)를 대체한다.
OMP / cut 공식 / 수렴 논리는 그대로:
  1. x̄에서 V*(x̄) = max{ V̄ᴸ(x̄,α) : α ∈ Ψ(x̄, 𝒟̃) } 를 MILP로 정확히 평가 → α*
  2. α* 고정 ISP-L + ISP-F LP (기존 mini-Benders LP) → cut (M3 형태지만 α* 최적이므로 x̄에서 tight)

세 평가 경로 (ccg.tex §subsec:eval, §subsec:disj, §subsec:oneshot):
  :monolithic  — p̃ 변수, 𝒦(p̃,z) + 평가 copy + leader KKT, MILP 1개 (식 oneshot)
  :decomposed  — belief cover 생성 (Alg.1) 후 belief별 MILP, V* = max_j W_j (식 evalMILP)
  :disjunctive — cover + 선택 binary σ_j로 MILP 1개 (식 disjMILP)

Complementarity 인코딩: encoding = :indicator (Gurobi indicator) | :bigM.

## NIG 인스턴스화 (ccg.tex 일반형과의 차이)
x̄ 고정 시 c_ks := ξ_ks (1 - v_ks x̄_k). Follower extensive form (belief p̃):
  max Σ_s p̃_s f_s
  s.t. Σ_k h_k ≤ w                          (ρ ≥ 0)
       N_y y^s + N_ts f_s = 0        ∀s     (μ^s free)   ← 등식: binary 없음
       y^s_k - h_k ≤ c_ks            ∀k,s   (λ^s_k ≥ 0)
       h, y ≥ 0,  f_s free                               ← free: dual 등식, binary 없음
Dual feasibility:
  λ^s_k + (N_yᵀ μ^s)_k ≥ 0   (y^s_k),   N_tsᵀ μ^s = p̃_s   (f_s),   ρ - Σ_s λ^s_k ≥ 0   (h_k)
Pattern binary 수 n = 1 + K + 2KS.
Follower center q̃ = q̂ (기존 true_dro 코드와 동일 규약, ISP-F의 d bound 참조).

Leader envelope 𝒬 (코드 규약: a = p̂ (TV ball), r = CVaR weight):
  CVaR (β≠nothing): Σr=1, r ≤ a/(1-β), Σa=1, |a-q̂| ≤ b, Σb ≤ 2ε̂
  Expectation (β=nothing): r ≡ a
V̄ᴸ(x̄,α) = max_{𝒬} Σ_s r_s Q_s(α).

## Big-M 근거 (encoding=:bigM)
- Primal: Σh ≤ w ⇒ slack ≤ w, h_k ≤ w, y^s_k ≤ c_ks + w, cap slack ≤ c_ks + w.
- Follower dual (M_dual_F, 기본 2): node potential μ^s를 [source, sink] potential 구간으로
  clipping하면 arc별 potential 차이의 양수부가 줄어들므로 λ ≤ p̃_s ≤ 1, ρ ≤ 1인 optimal dual
  존재. Reduced cost ≤ λ + |potential 차| ≤ 2.
- Leader dual (M_dual_L): 엄밀한 a-priori bound 없음 (ccg.tex Remark bigM). 기본값은
  4·T/(1-β)+4 (T = max_s Q_s 상한). :indicator 결과와 비교 검증 필요.
"""

using JuMP
using LinearAlgebra
using Printf


# ============================================================================
# 공통 데이터
# ============================================================================

"""
    StructEvalData

x̄ 고정 시 recourse 데이터. `make_struct_eval_data`로 생성.
"""
struct StructEvalData
    td::TrueDROData
    x_bar::Vector{Float64}
    c::Matrix{Float64}                              # K×S, c_ks = ξ_ks(1 - v_ks x̄_k)
    Tmax::Vector{Float64}                           # S, Q_s(α) 상한 = maxflow_s(c) + w
    Fabs::Vector{Float64}                           # S, |f_s| 상한 = Σ_k c_ks + w (cycle 포함)
    q_tilde::Vector{Float64}                        # follower center
    Ny_rows::Vector{Vector{Tuple{Int,Float64}}}     # row j → [(k, Ny[j,k]) nonzero]
    Ny_cols::Vector{Vector{Tuple{Int,Float64}}}     # col k → [(j, Ny[j,k]) nonzero]
    Nts_nz::Vector{Tuple{Int,Float64}}              # [(j, Nts[j]) nonzero]
end


"""
    make_struct_eval_data(td, x_bar; lp_optimizer)

c_ks와 max-flow 상한 Tmax_s 계산. Q_s(α) ≤ maxflow_s(c) + Σα ≤ maxflow_s(c) + w
(capacity를 총 w만큼 늘리면 max-flow는 최대 w 증가).
"""
function make_struct_eval_data(td::TrueDROData, x_bar::Vector{Float64}; lp_optimizer)
    K, S, m = td.num_arcs, td.S, td.nv1
    c = [td.xi_bar[k, s] * (1.0 - td.v[k, s] * x_bar[k]) for k in 1:K, s in 1:S]
    @assert all(c .>= -1e-12) "capacity c_ks < 0: v·x̄ > 1?"
    c = max.(c, 0.0)

    Ny_rows = [[(k, td.Ny[j, k]) for k in 1:K if td.Ny[j, k] != 0] for j in 1:m]
    Ny_cols = [[(j, td.Ny[j, k]) for j in 1:m if td.Ny[j, k] != 0] for k in 1:K]
    Nts_nz = [(j, td.Nts[j]) for j in 1:m if td.Nts[j] != 0]

    Tmax = zeros(S)
    for s in 1:S
        mf = Model(lp_optimizer)
        set_silent(mf)
        @variable(mf, 0 <= y[k=1:K] <= c[k, s])
        @variable(mf, f)
        @constraint(mf, [j=1:m], sum(a * y[k] for (k, a) in Ny_rows[j]; init=0.0) +
                                 td.Nts[j] * f == 0)
        @objective(mf, Max, f)
        optimize!(mf)
        termination_status(mf) == MOI.OPTIMAL ||
            error("max-flow LP (s=$s): $(termination_status(mf))")
        Tmax[s] = objective_value(mf) + td.w
    end
    Fabs = [sum(c[:, s]) + td.w for s in 1:S]

    return StructEvalData(td, copy(x_bar), c, Tmax, Fabs, copy(td.q_hat),
                          Ny_rows, Ny_cols, Nts_nz)
end


_check_encoding(enc) = enc in (:indicator, :bigM) ||
    error("encoding must be :indicator or :bigM, got $enc")


"""
    _complement!(model, z, A, B, MA, MB, encoding)

Nonneg 쌍 (A, B)의 complementarity: z=0 ⇒ A=0, z=1 ⇒ B=0.
(행: A=dual, B=slack. 변수: A=variable, B=reduced cost.)
"""
function _complement!(model, z, A, B, MA, MB, encoding)
    if encoding == :indicator
        @constraint(model, !z --> {A <= 0})
        @constraint(model, z --> {B <= 0})
    else
        @constraint(model, A <= MA * z)
        @constraint(model, B <= MB * (1 - z))
    end
    return nothing
end


# ============================================================================
# Building blocks
# ============================================================================

"""
    _add_follower_primal!(model, sd)

Follower extensive form의 primal feasibility (h=α, y', f'). Returns NamedTuple.
"""
function _add_follower_primal!(model, sd::StructEvalData)
    td = sd.td
    K, S, m, w = td.num_arcs, td.S, td.nv1, td.w
    c = sd.c

    α = @variable(model, [1:K], lower_bound = 0.0, upper_bound = w, base_name = "α")
    yp = @variable(model, [1:K, 1:S], lower_bound = 0.0, base_name = "yp")
    fp = @variable(model, [1:S], base_name = "fp")

    budget_slack = @expression(model, w - sum(α))
    @constraint(model, budget_slack >= 0)
    @constraint(model, [j=1:m, s=1:S],
        sum(a * yp[k, s] for (k, a) in sd.Ny_rows[j]; init=AffExpr(0.0)) +
        td.Nts[j] * fp[s] == 0)
    cap_slack = @expression(model, [k=1:K, s=1:S], c[k, s] + α[k] - yp[k, s])
    @constraint(model, [k=1:K, s=1:S], cap_slack[k, s] >= 0)

    return (α=α, yp=yp, fp=fp, budget_slack=budget_slack, cap_slack=cap_slack)
end


"""
    _add_follower_dual!(model, sd)

Follower extensive form의 dual feasibility (belief 무관 부분).
Belief 행 N_tsᵀμ^s = p̃_s 는 `ntsμ`로 반환하여 호출자가 추가.
"""
function _add_follower_dual!(model, sd::StructEvalData)
    td = sd.td
    K, S, m = td.num_arcs, td.S, td.nv1

    ρ = @variable(model, lower_bound = 0.0, base_name = "ρ")
    λ = @variable(model, [1:K, 1:S], lower_bound = 0.0, base_name = "λ")
    μ = @variable(model, [1:m, 1:S], base_name = "μ")

    rc_y = @expression(model, [k=1:K, s=1:S],
        λ[k, s] + sum(a * μ[j, s] for (j, a) in sd.Ny_cols[k]; init=AffExpr(0.0)))
    @constraint(model, [k=1:K, s=1:S], rc_y[k, s] >= 0)
    rc_h = @expression(model, [k=1:K], ρ - sum(λ[k, s] for s in 1:S))
    @constraint(model, [k=1:K], rc_h[k] >= 0)
    ntsμ = @expression(model, [s=1:S],
        sum(a * μ[j, s] for (j, a) in sd.Nts_nz; init=AffExpr(0.0)))

    return (ρ=ρ, λ=λ, μ=μ, rc_y=rc_y, rc_h=rc_h, ntsμ=ntsμ)
end


"""
    _add_pattern_certificate!(model, sd, P, D; encoding, M_dual_F)

ccg.tex 식 (cs): 𝒦(p̃,z)의 complementarity. Binary z = (zρ, zλ, zh, zy).
"""
function _add_pattern_certificate!(model, sd::StructEvalData, P, D;
                                   encoding::Symbol, M_dual_F::Float64)
    td = sd.td
    K, S, w = td.num_arcs, td.S, td.w
    c = sd.c

    zρ = @variable(model, binary = true, base_name = "zρ")
    zλ = @variable(model, [1:K, 1:S], binary = true, base_name = "zλ")
    zh = @variable(model, [1:K], binary = true, base_name = "zh")
    zy = @variable(model, [1:K, 1:S], binary = true, base_name = "zy")

    _complement!(model, zρ, D.ρ, P.budget_slack, M_dual_F, w, encoding)
    for k in 1:K, s in 1:S
        _complement!(model, zλ[k, s], D.λ[k, s], P.cap_slack[k, s],
                     M_dual_F, c[k, s] + w, encoding)
        _complement!(model, zy[k, s], P.yp[k, s], D.rc_y[k, s],
                     c[k, s] + w, M_dual_F, encoding)
    end
    for k in 1:K
        _complement!(model, zh[k], P.α[k], D.rc_h[k], w, M_dual_F, encoding)
    end

    z_all = vcat([zρ], vec(zλ), zh, vec(zy))
    return (zρ=zρ, zλ=zλ, zh=zh, zy=zy, z_all=z_all)
end


"""
    _add_eval_block!(model, sd, α)

평가 copy y^s (certificate copy y'와 분리, ccg.tex Prop. eval 뒤 설명):
  N_y y^s + N_ts t_s = 0,  0 ≤ y^s_k ≤ c_ks + α_k,  0 ≤ t_s ≤ Tmax_s.
t_s ≥ 0: 최적점 t_s = Q_s(α) ≥ 0을 자르지 않음.
"""
function _add_eval_block!(model, sd::StructEvalData, α)
    td = sd.td
    K, S, m = td.num_arcs, td.S, td.nv1

    y = @variable(model, [1:K, 1:S], lower_bound = 0.0, base_name = "y")
    t = @variable(model, [s=1:S], lower_bound = 0.0, upper_bound = sd.Tmax[s], base_name = "t")
    @constraint(model, [j=1:m, s=1:S],
        sum(a * y[k, s] for (k, a) in sd.Ny_rows[j]; init=AffExpr(0.0)) +
        td.Nts[j] * t[s] == 0)
    @constraint(model, [k=1:K, s=1:S], y[k, s] <= sd.c[k, s] + α[k])
    return (y=y, t=t)
end


"""
    _default_M_dual_L(sd)

Leader KKT dual big-M 기본값 (a-priori 보장 없음, 모듈 docstring 참조).
"""
function _default_M_dual_L(sd::StructEvalData)
    β = sd.td.beta === nothing ? 0.0 : sd.td.beta
    return 4.0 * maximum(sd.Tmax) / (1.0 - β) + 4.0
end


"""
    _add_leader!(model, sd, t; leader=:kkt, encoding, M_dual_L)

leader 블록 선택:
- `:kkt`      — ccg.tex 식 (evalMILP): 𝒬의 KKT를 binary로 (MILP)
- `:bilinear` — 목적 Σ r_s t_s 를 그대로 두고 Gurobi NonConvex=2 (disjoint bilinear,
                bilinear 항 S개뿐; 원래 ISP의 α_k a_s K·S개 대비 훨씬 작음)
"""
function _add_leader!(model, sd::StructEvalData, t; leader::Symbol=:kkt,
                      encoding::Symbol, M_dual_L::Float64)
    if leader == :kkt
        return _add_leader_kkt!(model, sd, t; encoding, M_dual_L)
    elseif leader == :bilinear
        return _add_leader_bilinear!(model, sd, t)
    else
        error("leader must be :kkt or :bilinear, got $leader")
    end
end

function _add_leader_bilinear!(model, sd::StructEvalData, t)
    td = sd.td
    S, q̂, ε̂ = td.S, td.q_hat, td.eps_hat
    use_cvar = (td.beta !== nothing)
    β = use_cvar ? td.beta : 0.0
    a = @variable(model, [1:S], lower_bound = 0.0, upper_bound = 1.0, base_name = "a")
    b = @variable(model, [1:S], lower_bound = 0.0, base_name = "b")
    @constraint(model, sum(a) == 1)
    @constraint(model, [s=1:S], a[s] - b[s] <= q̂[s])
    @constraint(model, [s=1:S], a[s] + b[s] >= q̂[s])
    @constraint(model, sum(b) <= 2ε̂)
    if use_cvar
        r = @variable(model, [1:S], lower_bound = 0.0, upper_bound = 1.0, base_name = "r")
        @constraint(model, sum(r) == 1)
        @constraint(model, [s=1:S], r[s] <= a[s] / (1.0 - β))
    else
        r = a
    end
    obj = @expression(model, sum(r[s] * t[s] for s in 1:S))
    set_optimizer_attribute(model, "NonConvex", 2)
    return (obj=obj, r=r, a=a, b=b, zL=VariableRef[], zkeys=Tuple{Symbol,Int}[])
end


"""
    _add_leader_kkt!(model, sd, t; encoding, M_dual_L)

ccg.tex 식 (evalMILP)의 leader-side 블록: max_{(r,a,b)∈𝒬} Σ r_s t_s 의 KKT.
Returns NamedTuple with `obj` (= fᵀu, complementarity 하에서 = Σ r_s t_s).
"""
function _add_leader_kkt!(model, sd::StructEvalData, t;
                          encoding::Symbol, M_dual_L::Float64)
    td = sd.td
    S = td.S
    q̂ = td.q_hat
    ε̂ = td.eps_hat
    use_cvar = (td.beta !== nothing)
    β = use_cvar ? td.beta : 0.0

    # ---- primal (q, v) = (r, a, b) ----
    a = @variable(model, [1:S], lower_bound = 0.0, base_name = "a")
    b = @variable(model, [1:S], lower_bound = 0.0, base_name = "b")
    @constraint(model, sum(a) == 1)                                   # u3 free
    s4 = @expression(model, [s=1:S], q̂[s] - a[s] + b[s])              # u4: a - b ≤ q̂
    s5 = @expression(model, [s=1:S], a[s] + b[s] - q̂[s])              # u5: -a - b ≤ -q̂
    s6 = @expression(model, 2ε̂ - sum(b))                              # u6: Σb ≤ 2ε̂
    @constraint(model, [s=1:S], s4[s] >= 0)
    @constraint(model, [s=1:S], s5[s] >= 0)
    @constraint(model, s6 >= 0)

    # ---- dual u ----
    u3 = @variable(model, base_name = "u3")
    u4 = @variable(model, [1:S], lower_bound = 0.0, base_name = "u4")
    u5 = @variable(model, [1:S], lower_bound = 0.0, base_name = "u5")
    u6 = @variable(model, lower_bound = 0.0, base_name = "u6")

    rc_b = @expression(model, [s=1:S], u6 - u4[s] - u5[s])
    @constraint(model, [s=1:S], rc_b[s] >= 0)

    # primal big-M (implied bounds): a ≤ 1, b ≤ 2ε̂, slack4/5 ≤ 1+2ε̂, slack6 ≤ 2ε̂
    Mb = 2ε̂
    M45 = 1.0 + 2ε̂
    M6 = 2ε̂

    if use_cvar
        r = @variable(model, [1:S], lower_bound = 0.0, base_name = "r")
        @constraint(model, sum(r) == 1)                               # u2 free
        s1 = @expression(model, [s=1:S], a[s] / (1.0 - β) - r[s])     # u1: r - a/(1-β) ≤ 0
        @constraint(model, [s=1:S], s1[s] >= 0)
        u1 = @variable(model, [1:S], lower_bound = 0.0, base_name = "u1")
        u2 = @variable(model, base_name = "u2")
        rc_r = @expression(model, [s=1:S], u1[s] + u2 - t[s])
        rc_a = @expression(model, [s=1:S], -u1[s] / (1.0 - β) + u3 + u4[s] - u5[s])
        @constraint(model, [s=1:S], rc_r[s] >= 0)
        @constraint(model, [s=1:S], rc_a[s] >= 0)
        obj = @expression(model, u2 + u3 + sum(q̂[s] * (u4[s] - u5[s]) for s in 1:S) + 2ε̂ * u6)
    else
        r = a
        rc_a = @expression(model, [s=1:S], u3 + u4[s] - u5[s] - t[s])
        @constraint(model, [s=1:S], rc_a[s] >= 0)
        obj = @expression(model, u3 + sum(q̂[s] * (u4[s] - u5[s]) for s in 1:S) + 2ε̂ * u6)
    end

    # ---- complementarity (binary 6S+1 or 4S+1) ----
    zL = VariableRef[]
    zkeys = Tuple{Symbol,Int}[]        # MIP start용: 각 ζ가 어느 상보쌍인지
    function _cz(key)
        z = @variable(model, binary = true, base_name = "ζ")
        push!(zL, z); push!(zkeys, key)
        return z
    end
    for s in 1:S
        _complement!(model, _cz((:u4, s)), u4[s], s4[s], M_dual_L, M45, encoding)
        _complement!(model, _cz((:u5, s)), u5[s], s5[s], M_dual_L, M45, encoding)
        _complement!(model, _cz((:a, s)), a[s], rc_a[s], 1.0, M_dual_L, encoding)
        _complement!(model, _cz((:b, s)), b[s], rc_b[s], Mb, M_dual_L, encoding)
        if use_cvar
            _complement!(model, _cz((:u1, s)), u1[s], s1[s], M_dual_L, 1.0 / (1.0 - β), encoding)
            _complement!(model, _cz((:r, s)), r[s], rc_r[s], 1.0, M_dual_L, encoding)
        end
    end
    _complement!(model, _cz((:u6, 0)), u6, s6, M_dual_L, M6, encoding)

    # ---- Valid inequality: 목적 상한 (relaxation 유계화) ----
    # 가능해에서 fᵀu = Σ r_s t_s (complementarity). r_s ∈ [0,1], t_s ∈ [0,Tmax_s]의
    # McCormick 상한: r_s t_s ≤ t_s, r_s t_s ≤ Tmax_s r_s. 해를 자르지 않음.
    # 없으면 u 무상한(indicator 인코딩)으로 root relaxation이 unbounded.
    ω = @variable(model, [1:S], base_name = "ωrt")
    @constraint(model, [s=1:S], ω[s] <= t[s])
    @constraint(model, [s=1:S], ω[s] <= sd.Tmax[s] * r[s])
    @constraint(model, obj <= sum(ω))

    return (obj=obj, r=r, a=a, b=b, zL=zL, zkeys=zkeys)
end


"""
    _make_mip(optimizer; silent, time_limit, mip_gap)
"""
function _make_mip(optimizer; silent::Bool=true, time_limit=nothing, mip_gap=1e-6)
    model = Model(optimizer)
    silent && set_silent(model)
    try set_optimizer_attribute(model, "MIPGap", mip_gap) catch; end
    if time_limit !== nothing
        set_time_limit_sec(model, Float64(time_limit))
    end
    return model
end


function _mip_result(model, α, extra::Dict)
    st = termination_status(model)
    has_sol = has_values(model) &&
              (st == MOI.OPTIMAL || st == MOI.TIME_LIMIT || st == MOI.NODE_LIMIT)
    has_sol || error("structured eval MILP: $st (no feasible solution)")
    res = Dict{Symbol,Any}(
        :V => objective_value(model),
        :V_bound => (st == MOI.OPTIMAL ? objective_bound(model) :
                     try objective_bound(model) catch; Inf end),
        :α => max.(value.(α), 0.0),
        :is_exact => (st == MOI.OPTIMAL),
        :status => st,
        :n_bin => count(is_binary, all_variables(model)),
    )
    merge!(res, extra)
    return res
end


# ============================================================================
# MIP start: belief 하나에서 follower/leader LP의 primal-dual 쌍 → complementarity binary
# ============================================================================

"""
    _follower_lp_pd(sd, p; lp_optimizer)

belief p에서 extensive-form LP의 (h, y, ρ, λ) — 단체법 primal-dual 쌍 (상보성 성립).
"""
function _follower_lp_pd(sd::StructEvalData, p::Vector{Float64}; lp_optimizer)
    td = sd.td
    K, S, m, w = td.num_arcs, td.S, td.nv1, td.w
    md = Model(lp_optimizer); set_silent(md)
    h = @variable(md, [1:K], lower_bound = 0.0)
    y = @variable(md, [1:K, 1:S], lower_bound = 0.0)
    f = @variable(md, [1:S])
    cb = @constraint(md, sum(h) <= w)
    @constraint(md, [j=1:m, s=1:S],
        sum(a * y[k, s] for (k, a) in sd.Ny_rows[j]; init=AffExpr(0.0)) + td.Nts[j] * f[s] == 0)
    cc = @constraint(md, [k=1:K, s=1:S], y[k, s] - h[k] <= sd.c[k, s])
    @objective(md, Max, sum(p[s] * f[s] for s in 1:S))
    optimize!(md)
    termination_status(md) == MOI.OPTIMAL || return nothing
    return (h=value.(h), y=value.(y), ρ=shadow_price(cb), λ=shadow_price.(cc))
end


"""
    _leader_lp_pd(sd, t; lp_optimizer)

max_{(r,a,b)∈𝒬} Σ r_s t_s 의 primal-dual. Returns (val, vals::Dict{Symbol,...}).
"""
function _leader_lp_pd(sd::StructEvalData, t::Vector{Float64}; lp_optimizer)
    td = sd.td
    S, q̂, ε̂ = td.S, td.q_hat, td.eps_hat
    use_cvar = (td.beta !== nothing)
    β = use_cvar ? td.beta : 0.0
    md = Model(lp_optimizer); set_silent(md)
    a = @variable(md, [1:S], lower_bound = 0.0)
    b = @variable(md, [1:S], lower_bound = 0.0)
    @constraint(md, sum(a) == 1)
    c4 = @constraint(md, [s=1:S], a[s] - b[s] <= q̂[s])
    c5 = @constraint(md, [s=1:S], -a[s] - b[s] <= -q̂[s])
    c6 = @constraint(md, sum(b) <= 2ε̂)
    if use_cvar
        r = @variable(md, [1:S], lower_bound = 0.0)
        @constraint(md, sum(r) == 1)
        c1 = @constraint(md, [s=1:S], r[s] - a[s] / (1.0 - β) <= 0)
    else
        r = a
    end
    @objective(md, Max, sum(r[s] * t[s] for s in 1:S))
    optimize!(md)
    termination_status(md) == MOI.OPTIMAL || return nothing
    vals = Dict{Symbol,Any}(:a => value.(a), :b => value.(b), :r => value.(r),
                            :u4 => shadow_price.(c4), :u5 => shadow_price.(c5),
                            :u6 => shadow_price(c6))
    use_cvar && (vals[:u1] = shadow_price.(c1))
    return objective_value(md), vals
end


"""
    _set_kkt_start!(sd, Z, L, p; lp_optimizer, tol=1e-7) → leader value or nothing

belief p에서 LP 해로 follower 패턴 Z와 leader ζ의 시작값 지정 (연속변수는 Gurobi가 완성).
규칙: 상보쌍 (A,B)에서 A > tol 이면 z=1 (B=0이어야 함), 아니면 z=0.
"""
function _kkt_start_values(sd::StructEvalData, p::Vector{Float64}; lp_optimizer, tol=1e-7)
    td = sd.td
    K, S = td.num_arcs, td.S
    fp = _follower_lp_pd(sd, p; lp_optimizer)
    fp === nothing && return nothing
    # t_s = Q_s(h)
    oracle = _build_follower_oracle(sd; lp_optimizer)
    t = zeros(S)
    for s in 1:S
        for k in 1:K
            set_upper_bound(oracle.mf_y[s][k], sd.c[k, s] + fp.h[k])
        end
        optimize!(oracle.mf[s])
        t[s] = objective_value(oracle.mf[s])
    end
    lp = _leader_lp_pd(sd, t; lp_optimizer)
    lp === nothing && return nothing
    val, lv = lp
    zf = vcat([fp.ρ > tol], vec(fp.λ .> tol), fp.h .> tol, vec(fp.y .> tol))
    return (val=val, zf=zf, lv=lv)
end

function _apply_kkt_start!(Z, L, st; tol=1e-7)
    for (z, v) in zip(Z.z_all, st.zf)
        set_start_value(z, v ? 1.0 : 0.0)
    end
    for (z, (key, s)) in zip(L.zL, L.zkeys)
        A = s == 0 ? st.lv[key] : st.lv[key][s]
        set_start_value(z, A > tol ? 1.0 : 0.0)
    end
end


# ============================================================================
# Route 3: Monolithic (ccg.tex 식 oneshot)
# ============================================================================

"""
    evaluate_monolithic(sd; optimizer, encoding=:indicator, M_dual_F=2.0, M_dual_L=nothing,
                        time_limit=nothing, mip_gap=1e-6, silent=true)

p̃를 변수로 두고 𝒦(p̃,z) + 평가 copy + leader KKT를 MILP 하나로 푼다.
Returns Dict(:V, :V_bound, :α, :p_tilde, :is_exact, :status, :n_bin, :time).
"""
function evaluate_monolithic(sd::StructEvalData; optimizer, encoding::Symbol=:indicator,
                             M_dual_F::Float64=2.0, M_dual_L=nothing,
                             time_limit=nothing, mip_gap=1e-6, silent::Bool=true, leader::Symbol=:kkt,
                             mip_start::Bool=true, n_start::Int=50, lp_optimizer=nothing)
    _check_encoding(encoding)
    td = sd.td
    S = td.S
    ML = M_dual_L === nothing ? _default_M_dual_L(sd) : Float64(M_dual_L)

    t0 = time()
    model = _make_mip(optimizer; silent, time_limit, mip_gap)

    p̃ = @variable(model, [1:S], lower_bound = 0.0, base_name = "p̃")
    η = @variable(model, [1:S], lower_bound = 0.0, base_name = "η")
    @constraint(model, sum(p̃) == 1)
    @constraint(model, [s=1:S], η[s] >= p̃[s] - sd.q_tilde[s])
    @constraint(model, [s=1:S], η[s] >= sd.q_tilde[s] - p̃[s])
    @constraint(model, 0.5 * sum(η) <= td.eps_tilde)

    P = _add_follower_primal!(model, sd)
    D = _add_follower_dual!(model, sd)
    @constraint(model, [s=1:S], D.ntsμ[s] == p̃[s])
    Z = _add_pattern_certificate!(model, sd, P, D; encoding, M_dual_F)
    E = _add_eval_block!(model, sd, P.α)
    L = _add_leader!(model, sd, E.t; leader, encoding, M_dual_L=ML)

    @objective(model, Max, L.obj)

    # MIP start: q̃ + 볼 꼭짓점 일부에서 LP로 만든 KKT 패턴 중 leader 값 최대인 것
    start_val = NaN
    if mip_start
        lpo = lp_optimizer === nothing ? optimizer : lp_optimizer
        cands = [copy(sd.q_tilde)]
        if td.eps_tilde > 0
            V = _tv_ball_vertices(sd.q_tilde, td.eps_tilde)
            step = max(1, cld(length(V), n_start))
            append!(cands, V[1:step:end])
        end
        best = nothing
        for pc in cands
            st = _kkt_start_values(sd, pc; lp_optimizer=lpo)
            st === nothing && continue
            if best === nothing || st.val > best.val
                best = st
            end
        end
        if best !== nothing
            _apply_kkt_start!(Z, L, best)
            start_val = best.val
        end
    end
    optimize!(model)

    res = _mip_result(model, P.α, Dict(:route => :monolithic, :encoding => encoding))
    res[:p_tilde] = value.(p̃)
    res[:start_val] = start_val
    res[:time] = time() - t0
    return res
end


# ============================================================================
# Belief cover generation (ccg.tex Alg. 1, 식 enumMILP)
# ============================================================================

"""
    generate_belief_cover(sd; optimizer, encoding=:indicator, M_dual_F=2.0,
                          max_patterns=2000, threshold_mode=false, time_limit=nothing,
                          belief_tol=1e-6, silent=true, verbose=false)

min ½Ση s.t. p̃∈Δ, (½Ση ≤ ε̃), 𝒦(p̃,z), no-good cuts 를 반복.
- `threshold_mode=true`: ball 제약 없이 δ_k > ε̃에서 정지 (Remark thresholds).
- `max_patterns`에 도달하면 `:complete=false` (cover 불완전 → 평가값은 하한, cut은 valid).
Returns Dict(:patterns => [(p̃, z, δ)], :beliefs => 중복 제거된 p̃ 목록,
             :complete, :n_milps, :time).
"""
function generate_belief_cover(sd::StructEvalData; optimizer, encoding::Symbol=:indicator,
                               M_dual_F::Float64=2.0, max_patterns::Int=2000,
                               threshold_mode::Bool=false, time_limit=nothing,
                               belief_tol::Float64=1e-6, silent::Bool=true,
                               verbose::Bool=false)
    _check_encoding(encoding)
    td = sd.td
    S = td.S
    t0 = time()

    model = _make_mip(optimizer; silent, mip_gap=1e-9)
    p̃ = @variable(model, [1:S], lower_bound = 0.0, base_name = "p̃")
    η = @variable(model, [1:S], lower_bound = 0.0, base_name = "η")
    @constraint(model, sum(p̃) == 1)
    @constraint(model, [s=1:S], η[s] >= p̃[s] - sd.q_tilde[s])
    @constraint(model, [s=1:S], η[s] >= sd.q_tilde[s] - p̃[s])
    if !threshold_mode
        @constraint(model, 0.5 * sum(η) <= td.eps_tilde)
    end

    P = _add_follower_primal!(model, sd)
    D = _add_follower_dual!(model, sd)
    @constraint(model, [s=1:S], D.ntsμ[s] == p̃[s])
    Z = _add_pattern_certificate!(model, sd, P, D; encoding, M_dual_F)
    @objective(model, Min, 0.5 * sum(η))

    patterns = Tuple{Vector{Float64},BitVector,Float64}[]
    complete = false
    n_milps = 0
    while true
        if time_limit !== nothing
            remaining = time_limit - (time() - t0)
            remaining <= 0 && break
            set_time_limit_sec(model, remaining)
        end
        optimize!(model)
        n_milps += 1
        st = termination_status(model)
        if st == MOI.INFEASIBLE || st == MOI.INFEASIBLE_OR_UNBOUNDED
            complete = true
            break
        end
        st == MOI.OPTIMAL || break   # time limit 등 → 불완전
        δ = objective_value(model)
        if threshold_mode && δ > td.eps_tilde + 1e-9
            complete = true
            break
        end
        z̄ = BitVector(value.(Z.z_all) .> 0.5)
        push!(patterns, (value.(p̃), z̄, δ))
        verbose && @printf("  cover[%d]: δ=%.6f\n", length(patterns), δ)
        length(patterns) >= max_patterns && break

        # no-good cut: z ≠ z̄
        @constraint(model,
            sum((1 - Z.z_all[i]) for i in eachindex(z̄) if z̄[i]; init=AffExpr(0.0)) +
            sum(Z.z_all[i] for i in eachindex(z̄) if !z̄[i]; init=AffExpr(0.0)) >= 1)
    end

    # 대표 belief 중복 제거: decomposed/disjunctive 평가는 p̃에만 의존 (z 무관)
    beliefs = Vector{Float64}[]
    for (pv, _, _) in patterns
        if all(maximum(abs.(pv .- b)) > belief_tol for b in beliefs)
            push!(beliefs, pv)
        end
    end

    return Dict{Symbol,Any}(:patterns => patterns, :beliefs => beliefs,
                            :complete => complete, :n_milps => n_milps,
                            :n_bin => length(Z.z_all), :time => time() - t0)
end


# ============================================================================
# Belief vertex cover (degeneracy-free; docs/ccg_degeneracy.md)
# ============================================================================
#
# φ(p̃) := max_{h∈H} Σ_s p̃_s Q_s(h)  (follower 가치함수, p̃에 대해 convex PWL)
# Ψ(x̄,p̃) = h-projection of the optimal face F(p̃) of P_x̄.
# Cell C_F = {p̃ : F(p̃) = F}. 최적성은 닫힌 조건 → v ∈ cl(C_F) ⇒ F ⊆ F(v) ⇒ Ψ(p̃) ⊆ Ψ(v).
# 따라서 Ψ(x̄,𝒟̃) = ∪_{v ∈ vert(𝒞_x̄ ∩ 𝒟̃)} Ψ(x̄,v)   (𝒞_x̄: φ의 linearity complex).
#
# 계산: 폴리토프 Π_T = {(p̃,τ): p̃∈Δ, TV cuts, τ ≥ p̃ᵀθⁱ (i∈T), τmin ≤ τ ≤ τmax}의 꼭짓점을
# double description(DD)으로 유지하며 cutting plane 추가:
#   (a) 꼭짓점 p̃ ∉ 𝒟̃  → TV facet  Σ_{s∈A}(p̃_s - q̃_s) ≤ ε̃,  A = {s: p̃_s > q̃_s}
#   (b) φ(p̃) > τ       → envelope cut τ ≥ p̃ᵀθ,  θ_s = Q_s(h*), h* ∈ Ψ(x̄,p̃) (LP oracle)
# 모든 하단 꼭짓점이 (a),(b)를 만족하면 φ_T = φ on 𝒟̃ (convexity), 하단 꼭짓점 = cover.
# 오라클은 θ = (Q_s(h*))_s 값만 사용 → recourse y, 쌍대 (λ,μ)의 비유일성(degeneracy) 무관.

mutable struct _DDRay
    r::Vector{Float64}        # (p̃_1..p̃_S, τ), Σp̃ = 1 정규화
    tight::BitSet
    verified::Bool
end


"""
    _FollowerOracle

φ(p̃) 계산용 extensive-form LP (목적계수만 갱신) + 시나리오별 max-flow LP (θ 재계산).
"""
struct _FollowerOracle
    ext::Model
    h::Vector{VariableRef}
    f::Vector{VariableRef}
    mf::Vector{Model}
    mf_y::Vector{Vector{VariableRef}}
    mf_f::Vector{VariableRef}
end

function _build_follower_oracle(sd::StructEvalData; lp_optimizer)
    td = sd.td
    K, S, m, w = td.num_arcs, td.S, td.nv1, td.w
    ext = Model(lp_optimizer); set_silent(ext)
    h = @variable(ext, [1:K], lower_bound = 0.0)
    y = @variable(ext, [1:K, 1:S], lower_bound = 0.0)
    f = @variable(ext, [1:S])
    @constraint(ext, sum(h) <= w)
    @constraint(ext, [j=1:m, s=1:S],
        sum(a * y[k, s] for (k, a) in sd.Ny_rows[j]; init=AffExpr(0.0)) + td.Nts[j] * f[s] == 0)
    @constraint(ext, [k=1:K, s=1:S], y[k, s] - h[k] <= sd.c[k, s])

    mf, mf_y, mf_f = Model[], Vector{VariableRef}[], VariableRef[]
    for s in 1:S
        mm = Model(lp_optimizer); set_silent(mm)
        yy = @variable(mm, [1:K], lower_bound = 0.0)
        ff = @variable(mm)
        @constraint(mm, [j=1:m],
            sum(a * yy[k] for (k, a) in sd.Ny_rows[j]; init=AffExpr(0.0)) + td.Nts[j] * ff == 0)
        @objective(mm, Max, ff)
        push!(mf, mm); push!(mf_y, yy); push!(mf_f, ff)
    end
    return _FollowerOracle(ext, h, f, mf, mf_y, mf_f)
end

"""φ(p̃)만 (extensive-form LP 1개)."""
function _phi!(o::_FollowerOracle, sd::StructEvalData, p::Vector{Float64})
    S = sd.td.S
    @objective(o.ext, Max, sum(p[s] * o.f[s] for s in 1:S))
    optimize!(o.ext)
    termination_status(o.ext) == MOI.OPTIMAL ||
        error("follower oracle LP: $(termination_status(o.ext))")
    return objective_value(o.ext)
end

"""φ(p̃), h* ∈ Ψ(x̄,p̃), θ = (Q_s(h*))_s."""
function _oracle!(o::_FollowerOracle, sd::StructEvalData, p::Vector{Float64})
    td = sd.td
    K, S = td.num_arcs, td.S
    @objective(o.ext, Max, sum(p[s] * o.f[s] for s in 1:S))
    optimize!(o.ext)
    termination_status(o.ext) == MOI.OPTIMAL ||
        error("follower oracle LP: $(termination_status(o.ext))")
    φ = objective_value(o.ext)
    h = max.(value.(o.h), 0.0)
    θ = zeros(S)
    for s in 1:S
        for k in 1:K
            set_upper_bound(o.mf_y[s][k], sd.c[k, s] + h[k])
        end
        optimize!(o.mf[s])
        termination_status(o.mf[s]) == MOI.OPTIMAL ||
            error("max-flow LP (s=$s): $(termination_status(o.mf[s]))")
        θ[s] = objective_value(o.mf[s])
    end
    # θ_s = Q_s(h*) ≥ f_s  ⇒  p̃ᵀθ ≥ φ(p̃);  h* 최적이므로 p̃ᵀθ ≤ φ(p̃)
    abs(dot(p, θ) - φ) <= 1e-6 * max(1.0, abs(φ)) ||
        error(@sprintf("oracle inconsistency: p̃ᵀθ=%.8f φ=%.8f", dot(p, θ), φ))
    return φ, h, θ
end


"""
    _dd_add!(rays, cons, a, d; tol)

DD step: 제약 aᵀr ≥ 0 추가. Combinatorial adjacency test.
"""
function _dd_add!(rays::Vector{_DDRay}, cons::Vector{Vector{Float64}}, a::Vector{Float64},
                  d::Int; tol::Float64=1e-9)
    push!(cons, a)
    ci = length(cons)
    scale = max(1.0, maximum(abs, a))
    vals = [dot(a, ray.r) / scale for ray in rays]
    Rp = findall(>(tol), vals)
    Rm = findall(<(-tol), vals)
    R0 = findall(v -> abs(v) <= tol, vals)
    for i in R0
        push!(rays[i].tight, ci)
    end
    isempty(Rm) && return 0

    # 역인덱스: 제약 c → c가 tight한 ray 목록. 인접성 검사 시 common에서 가장 드문
    # 제약의 목록만 훑음 (combinatorial test는 동일, 비용만 감소)
    occ = Dict{Int,Vector{Int}}()
    for (l, ray) in enumerate(rays), c in ray.tight
        push!(get!(occ, c, Int[]), l)
    end
    new_rays = _DDRay[]
    for i in Rp, j in Rm
        common = intersect(rays[i].tight, rays[j].tight)
        length(common) >= d - 2 || continue
        cmin, nmin = 0, typemax(Int)
        for c in common
            n = length(occ[c])
            n < nmin && ((cmin, nmin) = (c, n))
        end
        adjacent = true
        for l in occ[cmin]
            (l == i || l == j) && continue
            if issubset(common, rays[l].tight)
                adjacent = false
                break
            end
        end
        adjacent || continue
        r = vals[i] .* rays[j].r .- vals[j] .* rays[i].r
        r ./= sum(@view r[1:end-1])
        push!(new_rays, _DDRay(r, union(common, BitSet(ci)), false))
    end
    deleteat!(rays, Rm)
    append!(rays, new_rays)
    return length(new_rays)
end


"""
    _tv_ball_vertices(q, ε)

𝒟̃ = {p ∈ Δ : Σ_s (p_s - q_s)^+ ≤ ε} 의 꼭짓점 (닫힌 형태).
극점 p는 (i) TV(p) < ε 이면 Δ의 꼭짓점 e_i, (ii) TV(p) = ε 이면 초과 좌표가 하나(i)이고
부족 좌표는 0까지 깎인 집합 J와 분수 좌표 최대 1개(k)로 이루어진다
(초과 좌표 2개 또는 분수 부족 좌표 2개면 양방향 섭동 가능 → 극점 아님;
위 형태는 모든 섭동이 한쪽 방향으로만 가능 → 극점).
열거 비용 O(S·2^{S-1}).
"""
function _tv_ball_vertices(q::Vector{Float64}, ε::Float64; tol::Float64=1e-12)
    S = length(q)
    cands = Vector{Float64}[]
    for i in 1:S
        exc = min(ε, 1.0 - q[i])          # i의 초과량 = 부족 총량
        exc <= tol && continue
        others = [j for j in 1:S if j != i]
        n = length(others)
        for mask in 0:(2^n - 1)
            J = [others[b] for b in 1:n if (mask >> (b - 1)) & 1 == 1]
            sJ = sum(q[J]; init=0.0)
            sJ > exc + tol && continue
            rem = exc - sJ
            base = copy(q); base[i] += exc; base[J] .= 0.0
            if rem <= tol
                push!(cands, base)
            else
                for k in others
                    (k in J || rem >= q[k] - tol) && continue
                    p = copy(base); p[k] -= rem
                    push!(cands, p)
                end
            end
        end
    end
    V = Vector{Float64}[]
    for p in cands
        any(v -> maximum(abs.(v .- p)) <= 1e-10, V) || push!(V, p)
    end
    return V
end


"""
    _tv_tight_sets(V, q, ε)

꼭짓점 집합 V에서 tight한 TV facet 첨자집합 A (Σ_{s∈A}(p_s - q_s) = ε)들의 합집합.
p에서 tight ⇔ A ⊇ P⁺(p) 이고 A ⊆ P⁺(p) ∪ Z(p)  (Z: p_s = q_s 좌표).
"""
function _tv_tight_sets(V::Vector{Vector{Float64}}, q::Vector{Float64}, ε::Float64;
                        tol::Float64=1e-10)
    S = length(q)
    sets = Set{Vector{Int}}()
    for p in V
        tv = sum(max(p[s] - q[s], 0.0) for s in 1:S)
        abs(tv - ε) <= tol || continue
        Pp = [s for s in 1:S if p[s] > q[s] + tol]
        Z = [s for s in 1:S if abs(p[s] - q[s]) <= tol]
        for mask in 0:(2^length(Z) - 1)
            A = sort(vcat(Pp, [Z[b] for b in eachindex(Z) if (mask >> (b - 1)) & 1 == 1]))
            length(A) < S && push!(sets, A)
        end
    end
    return collect(sets)
end


"""
    generate_belief_vertex_cover(sd; lp_optimizer, max_rays=200_000, time_limit=nothing,
                                 verbose=false)

Degeneracy-free belief cover: vert(𝒞_x̄ ∩ 𝒟̃). Returns Dict(:beliefs, :complete,
:n_oracle, :n_cuts_tv, :n_cuts_env, :thetas, :hs, :time).
"""
function generate_belief_vertex_cover(sd::StructEvalData; lp_optimizer,
                                      max_rays::Int=200_000, time_limit=nothing,
                                      warm_hs::Vector{Vector{Float64}}=Vector{Float64}[],
                                      eps_override::Union{Nothing,Float64}=nothing,
                                      verbose::Bool=false)
    td = sd.td
    S = td.S
    q̃ = sd.q_tilde
    ε̃ = eps_override === nothing ? td.eps_tilde : eps_override
    t0 = time()

    if ε̃ == 0.0
        return Dict{Symbol,Any}(:beliefs => [copy(q̃)], :complete => true, :n_oracle => 0,
                                :n_cuts_tv => 0, :n_cuts_env => 0, :thetas => Vector{Float64}[],
                                :hs => Vector{Float64}[], :n_rays => 1, :time => time() - t0)
    end

    oracle = _build_follower_oracle(sd; lp_optimizer)
    d = S + 1
    τmin = -1.0
    τmax = maximum(sd.Tmax) + 1.0

    # 초기 prism: 𝒟̃ × [τmin, τmax]  (동차화: 1 ↔ Σp̃).
    # 𝒟̃의 V-rep은 닫힌 형태(_tv_ball_vertices), H-rep은 p̃ ≥ 0 과 어떤 꼭짓점에서 tight한
    # TV facet Σ_{s∈A}(p̃_s - q̃_s) ≤ ε̃ 들. (lazy TV cut은 중간 폴리토프 꼭짓점 폭증 → 사용 안 함)
    ball_V = _tv_ball_vertices(q̃, ε̃)
    cons = Vector{Float64}[]
    for s in 1:S
        a = zeros(d); a[s] = 1.0; push!(cons, a)                  # p̃_s ≥ 0
    end
    push!(cons, vcat(fill(-τmin, S), 1.0))                         # τ - τmin Σp̃ ≥ 0
    push!(cons, vcat(fill(τmax, S), -1.0))                         # τmax Σp̃ - τ ≥ 0
    i_lo, i_hi = S + 1, S + 2
    for A in _tv_tight_sets(ball_V, q̃, ε̃)
        a = zeros(d)
        for s in 1:S
            a[s] = ε̃ + sum(q̃[A]) - (s in A ? 1.0 : 0.0)         # ε̃Σp̃ - Σ_A(p̃_s - q̃_sΣp̃) ≥ 0
        end
        push!(cons, a)
    end
    rays = _DDRay[]
    for v in ball_V, τv in (τmin, τmax)
        r = vcat(v, τv)
        tight = BitSet(i for (i, a) in enumerate(cons)
                       if abs(dot(a, r)) <= 1e-9 * max(1.0, maximum(abs, a)))
        push!(rays, _DDRay(r, tight, false))
    end

    thetas = Vector{Float64}[]
    hs = Vector{Float64}[]
    n_oracle = n_tv = n_env = 0

    function add_env_cut!(θ)
        n_env += 1
        push!(thetas, θ)
        tdd = @elapsed nnew = _dd_add!(rays, cons, vcat(-θ, 1.0), d)   # τ - p̃ᵀθ ≥ 0
        verbose && @printf("  vertex-cover: env cut %d → +%d rays (total %d) DD %.2fs, elapsed %.1fs\n",
                           n_env, nnew, length(rays), tdd, time() - t0); verbose && flush(stdout)
    end
    verbose && @printf("  vertex-cover: ball |V|=%d, cons=%d, init rays=%d (%.2fs)\n",
                       length(ball_V), length(cons), length(rays), time() - t0); verbose && flush(stdout)

    # warm start: 이전 x̄의 h들 → 현재 x̄에서 θ(h) = Q(x̄,h) (유효한 envelope 조각)
    for h in warm_hs
        θ = zeros(S)
        for s in 1:S
            for k in 1:td.num_arcs
                set_upper_bound(oracle.mf_y[s][k], sd.c[k, s] + h[k])
            end
            optimize!(oracle.mf[s])
            θ[s] = objective_value(oracle.mf[s])
        end
        push!(hs, h)
        add_env_cut!(θ)
    end

    complete = false
    while true
        if time_limit !== nothing && time() - t0 > time_limit
            break
        end
        length(rays) > max_rays && break
        idx = findfirst(r -> !r.verified, rays)
        if idx === nothing
            complete = true
            break
        end
        ray = rays[idx]
        if i_hi in ray.tight                     # 상단 꼭짓점: 무관
            ray.verified = true
            continue
        end
        p = ray.r[1:S]
        τ = ray.r[end]
        # (a) TV 멤버십
        tv = sum(max(p[s] - q̃[s], 0.0) for s in 1:S)
        if tv > ε̃ + 1e-9
            A = [s for s in 1:S if p[s] > q̃[s]]
            # ε̃ Σp̃ - Σ_{s∈A}(p̃_s - q̃_s Σp̃) ≥ 0
            a = zeros(d)
            for s in 1:S
                a[s] = ε̃ + sum(q̃[A]) - (s in A ? 1.0 : 0.0)
            end
            n_tv += 1
            _dd_add!(rays, cons, a, d)
            continue
        end
        # (b) envelope
        n_oracle += 1
        φ = _phi!(oracle, sd, p)               # 검증엔 φ 값만 필요 (LP 1개)
        if φ > τ + 1e-7 * max(1.0, abs(φ))
            φ, h, θ = _oracle!(oracle, sd, p)  # 위반 시에만 θ = Q(h*) 계산
            push!(hs, h)
            add_env_cut!(θ)
        else
            ray.verified = true
        end
        verbose && n_oracle % 50 == 0 &&
            @printf("  vertex-cover: rays=%d oracle=%d env=%d tv=%d\n",
                    length(rays), n_oracle, n_env, n_tv); verbose && flush(stdout)
    end

    beliefs = Vector{Float64}[]
    for ray in rays
        i_hi in ray.tight && continue
        p = max.(ray.r[1:S], 0.0)
        p ./= sum(p)
        if all(maximum(abs.(p .- b)) > 1e-9 for b in beliefs)
            push!(beliefs, p)
        end
    end

    return Dict{Symbol,Any}(:beliefs => beliefs, :complete => complete,
                            :n_oracle => n_oracle, :n_cuts_tv => n_tv, :n_cuts_env => n_env,
                            :thetas => thetas, :hs => hs, :n_rays => length(rays),
                            :time => time() - t0)
end


# ============================================================================
# Minimal belief cover (docs/ccg_degeneracy.md §8)
# ============================================================================
#
# Θ := proj_θ P_x̄ (down-closed), σ_Θ(p) = φ(p),  F(p) = argmax_Θ pᵀθ = ∂σ_Θ(p),
# Ψ(x̄,p) = {h ∈ H : Q_{S(p)}(h) ∈ F_{S(p)}(p)}   (S(p) = supp p).
# Lemma (국소 결정): φ_T = φ 를 확대 볼 𝒟̃⁺ ⊋ 𝒟̃ 에서 보장하면, 모든 p ∈ 𝒟̃ 에서
#   F_{S(p)}(p) = conv{θⁱ_{S(p)} : i ∈ A_T(p)},  A_T(p) = argmax_{i∈T} pᵀθⁱ.
#   (subdifferential은 p 근방의 σ 값만으로 결정; 근방은 𝒟̃⁺ 안)
# Theorem (minimal cover): 패턴 π(p) := (Z(p), A_T(p)), Z(p) = {s: p_s = 0}.
#   π ≤ π' (성분별 ⊆) ⇒ Ψ(p) ⊆ Ψ(p'). 따라서 극대 패턴마다 belief 하나면 cover.
# 계산:
#   (1) envelope 분리: max_{p∈𝒟̃⁺, (h,y,f) feasible} Σ p_s f_s − max_i pᵀθⁱ
#       (bilinear 항 S개, NonConvex=2). 양수면 θ = Q(h*) 추가, 0이면 φ_T = φ on 𝒟̃⁺.
#   (2) 극대 패턴 열거: binary δ_i (조각 i 활성), ζ_s (p_s = 0), max Σδ+Σζ,
#       찾은 패턴의 부분집합 배제 cut ("not-subset") → 극대 패턴을 정확히 열거.

"""
    _envelope_separation(sd, thetas, ε⁺; optimizer, time_limit, tol)

max_{p ∈ 𝒟̃⁺} φ(p) − φ_T(p) 를 bilinear 프로그램으로 (φ(p) = max_{(h,y,f)} Σ p_s f_s).
Returns (viol, p*, certified::Bool).
"""
function _envelope_separation(sd::StructEvalData, thetas::Vector{Vector{Float64}}, ε⁺::Float64;
                              optimizer, time_limit=nothing, tol::Float64=1e-6)
    td = sd.td
    K, S, m, w = td.num_arcs, td.S, td.nv1, td.w
    q̃ = sd.q_tilde
    md = Model(optimizer); set_silent(md)
    set_optimizer_attribute(md, "NonConvex", 2)
    set_optimizer_attribute(md, "MIPGap", 0.0)
    set_optimizer_attribute(md, "MIPGapAbs", tol / 10)
    time_limit === nothing || set_time_limit_sec(md, Float64(time_limit))

    p = @variable(md, [1:S], lower_bound = 0.0, upper_bound = 1.0)
    η = @variable(md, [1:S], lower_bound = 0.0)
    @constraint(md, sum(p) == 1)
    @constraint(md, [s=1:S], η[s] >= p[s] - q̃[s])
    @constraint(md, [s=1:S], η[s] >= q̃[s] - p[s])
    @constraint(md, 0.5 * sum(η) <= ε⁺)
    h = @variable(md, [1:K], lower_bound = 0.0)
    y = @variable(md, [1:K, 1:S], lower_bound = 0.0)
    f = @variable(md, [s=1:S], lower_bound = -sd.Fabs[s], upper_bound = sd.Tmax[s])
    @constraint(md, sum(h) <= w)
    @constraint(md, [j=1:m, s=1:S],
        sum(a * y[k, s] for (k, a) in sd.Ny_rows[j]; init=AffExpr(0.0)) + td.Nts[j] * f[s] == 0)
    @constraint(md, [k=1:K, s=1:S], y[k, s] - h[k] <= sd.c[k, s])
    τ = @variable(md)
    for θ in thetas
        @constraint(md, τ >= sum(p[s] * θ[s] for s in 1:S))
    end
    @objective(md, Max, sum(p[s] * f[s] for s in 1:S) - τ)
    optimize!(md)
    st = termination_status(md)
    has_values(md) || error("envelope separation: $st (no solution)")
    viol = objective_value(md)
    bound = try objective_bound(md) catch; Inf end
    certified = (st == MOI.OPTIMAL) && bound <= tol
    return viol, max.(value.(p), 0.0), certified, bound
end


"""
    _envelope_separation_milp(sd, thetas, ε⁺; optimizer, encoding=:indicator,
                              time_limit=nothing, tol=1e-6, stop_viol=1e-4)

max_{p∈𝒟̃⁺} φ(p) − φ_T(p) 를 **MILP**로: follower KKT 패턴 𝒦(p,z) (monolithic의 follower 절반)
하에서 φ(p) = wρ + Σ c λ (쌍대 목적, p에 선형).
  max  wρ + Σ_{k,s} c_ks λ_ks − τ
  s.t. p ∈ 𝒟̃⁺,  𝒦(p,z),  N_tsᵀμ^s = p_s,  τ ≥ pᵀθ ∀θ∈T,
       wρ + Σcλ ≤ Σ_s p_s Tmax_s        (valid: 상보성 하에서 = φ(p) ≤ Σ p_s Tmax_s)
조기 종료: incumbent ≥ stop_viol (위반점 발견) 또는 bound ≤ tol (인증).
Returns (viol, p*, certified, bound).
"""
function _envelope_separation_milp(sd::StructEvalData, thetas::Vector{Vector{Float64}}, ε⁺::Float64;
                                   optimizer, lp_optimizer=nothing, encoding::Symbol=:indicator,
                                   time_limit=nothing, tol::Float64=1e-6, stop_viol::Float64=1e-4,
                                   silent::Bool=true)
    _check_encoding(encoding)
    td = sd.td
    K, S, w = td.num_arcs, td.S, td.w
    q̃ = sd.q_tilde
    md = Model(optimizer)
    silent && set_silent(md)
    set_optimizer_attribute(md, "MIPGap", 0.0)
    set_optimizer_attribute(md, "MIPGapAbs", tol / 10)
    set_optimizer_attribute(md, "BestBdStop", tol)
    set_optimizer_attribute(md, "BestObjStop", stop_viol)
    time_limit === nothing || set_time_limit_sec(md, Float64(time_limit))

    p = @variable(md, [1:S], lower_bound = 0.0, upper_bound = 1.0)
    η = @variable(md, [1:S], lower_bound = 0.0)
    @constraint(md, sum(p) == 1)
    @constraint(md, [s=1:S], η[s] >= p[s] - q̃[s])
    @constraint(md, [s=1:S], η[s] >= q̃[s] - p[s])
    @constraint(md, 0.5 * sum(η) <= ε⁺)

    P = _add_follower_primal!(md, sd)
    D = _add_follower_dual!(md, sd)
    @constraint(md, [s=1:S], D.ntsμ[s] == p[s])
    Z = _add_pattern_certificate!(md, sd, P, D; encoding, M_dual_F=2.0)
    dual_obj = @expression(md, w * D.ρ + sum(sd.c[k, s] * D.λ[k, s] for k in 1:K, s in 1:S))
    @constraint(md, dual_obj <= sum(p[s] * sd.Tmax[s] for s in 1:S))
    τ = @variable(md)
    for θ in thetas
        @constraint(md, τ >= sum(p[s] * θ[s] for s in 1:S))
    end
    @objective(md, Max, dual_obj - τ)

    # MIP start: q̃에서 follower LP primal-dual 패턴
    lpo = lp_optimizer === nothing ? optimizer : lp_optimizer
    fp = _follower_lp_pd(sd, copy(q̃); lp_optimizer=lpo)
    if fp !== nothing
        zf = vcat([fp.ρ > 1e-7], vec(fp.λ .> 1e-7), fp.h .> 1e-7, vec(fp.y .> 1e-7))
        for (z, v) in zip(Z.z_all, zf)
            set_start_value(z, v ? 1.0 : 0.0)
        end
    end

    optimize!(md)
    st = termination_status(md)
    bound = try objective_bound(md) catch; Inf end
    if !has_values(md)
        return -Inf, copy(q̃), (bound <= tol), bound
    end
    viol = objective_value(md)
    certified = bound <= tol
    return viol, max.(value.(p), 0.0), certified, bound
end


"""
    _pattern_of(p, thetas; tol_tie, tol_zero) → (Z::BitVector, A::BitVector)
"""
function _pattern_of(p::Vector{Float64}, thetas::Vector{Vector{Float64}};
                     tol_tie::Float64=1e-7, tol_zero::Float64=1e-9)
    vals = [dot(p, θ) for θ in thetas]
    φT = maximum(vals)
    A = BitVector([φT - v <= tol_tie * max(1.0, abs(φT)) for v in vals])
    Z = BitVector([p[s] <= tol_zero for s in eachindex(p)])
    return Z, A
end


"""
    _maximal_patterns(sd, thetas; optimizer, max_patterns, time_limit)

𝒟̃ 위 패턴 (Z, A_T)의 극대 원소를 순차 MILP로 열거. 각 극대 패턴의 대표 belief 반환.
찾은 p*는 패턴 고정 LP로 polish (동률을 등식으로 정확히 만족).
"""
function _maximal_patterns(sd::StructEvalData, thetas::Vector{Vector{Float64}};
                           optimizer, lp_optimizer, max_patterns::Int=10_000, time_limit=nothing)
    td = sd.td
    S = td.S
    q̃ = sd.q_tilde
    ε̃ = td.eps_tilde
    n = length(thetas)
    M = maximum(maximum.(thetas)) - minimum(minimum.(thetas)) + 1.0
    t0 = time()

    md = Model(optimizer); set_silent(md)
    p = @variable(md, [1:S], lower_bound = 0.0, upper_bound = 1.0)
    η = @variable(md, [1:S], lower_bound = 0.0)
    @constraint(md, sum(p) == 1)
    @constraint(md, [s=1:S], η[s] >= p[s] - q̃[s])
    @constraint(md, [s=1:S], η[s] >= q̃[s] - p[s])
    @constraint(md, 0.5 * sum(η) <= ε̃)
    τ = @variable(md)
    δ = @variable(md, [1:n], binary = true)
    ζ = @variable(md, [1:S], binary = true)
    @constraint(md, [i=1:n], τ >= sum(p[s] * thetas[i][s] for s in 1:S))
    @constraint(md, [i=1:n], τ - sum(p[s] * thetas[i][s] for s in 1:S) <= M * (1 - δ[i]))
    @constraint(md, [s=1:S], p[s] <= 1 - ζ[s])
    @objective(md, Max, sum(δ) + sum(ζ))

    reps = Vector{Float64}[]
    pats = Tuple{BitVector,BitVector}[]
    complete = false
    while length(reps) < max_patterns
        if time_limit !== nothing
            rem = time_limit - (time() - t0)
            rem <= 0 && break
            set_time_limit_sec(md, rem)
        end
        optimize!(md)
        st = termination_status(md)
        if st == MOI.INFEASIBLE || st == MOI.INFEASIBLE_OR_UNBOUNDED
            complete = true
            break
        end
        st == MOI.OPTIMAL || break
        Zf = BitVector(value.(ζ) .> 0.5)
        Af = BitVector(value.(δ) .> 0.5)
        # polish: 패턴 (Zf, Af)를 등식으로 강제한 LP → 동률 정확
        pp = _polish_pattern(sd, thetas, Zf, Af; lp_optimizer)
        pp === nothing && (pp = max.(value.(p), 0.0); pp ./= sum(pp))
        Z, A = _pattern_of(pp, thetas)
        Z .|= Zf; A .|= Af
        push!(reps, pp); push!(pats, (Z, A))
        # not-subset cut: 다음 패턴은 (Z, A)의 부분집합이 아니어야 함
        @constraint(md, sum(ζ[s] for s in 1:S if !Z[s]; init=AffExpr(0.0)) +
                        sum(δ[i] for i in 1:n if !A[i]; init=AffExpr(0.0)) >= 1)
    end
    return reps, pats, complete, time() - t0
end


function _polish_pattern(sd::StructEvalData, thetas, Z::BitVector, A::BitVector; lp_optimizer)
    S = sd.td.S
    q̃ = sd.q_tilde
    md = Model(lp_optimizer); set_silent(md)
    p = @variable(md, [1:S], lower_bound = 0.0)
    η = @variable(md, [1:S], lower_bound = 0.0)
    τ = @variable(md)
    @constraint(md, sum(p) == 1)
    @constraint(md, [s=1:S], η[s] >= p[s] - q̃[s])
    @constraint(md, [s=1:S], η[s] >= q̃[s] - p[s])
    @constraint(md, 0.5 * sum(η) <= sd.td.eps_tilde)
    for (i, θ) in enumerate(thetas)
        if A[i]
            @constraint(md, τ == sum(p[s] * θ[s] for s in 1:S))
        else
            @constraint(md, τ >= sum(p[s] * θ[s] for s in 1:S))
        end
    end
    for s in 1:S
        Z[s] && fix(p[s], 0.0; force=true)
    end
    @objective(md, Min, sum(η))     # 대표점: 패턴 내에서 q̃에 가장 가까운 점
    optimize!(md)
    termination_status(md) == MOI.OPTIMAL || return nothing
    pv = max.(value.(p), 0.0)
    return pv ./ sum(pv)
end


"""
    generate_belief_minimal_cover(sd; optimizer, lp_optimizer, δ_ball=0.01,
                                  sep_time_limit=nothing, time_limit=nothing,
                                  warm_hs=[], verbose=false)

Minimal cover (모듈 주석의 Lemma/Theorem). Returns Dict(:beliefs, :complete,
:n_pieces, :n_sep, :hs, :thetas, :patterns, :time, :sep_time, :pat_time).
"""
function generate_belief_minimal_cover(sd::StructEvalData; optimizer, lp_optimizer,
                                       δ_ball::Float64=0.01, sep_time_limit=nothing,
                                       time_limit=nothing, max_pieces::Int=500,
                                       warm_hs::Vector{Vector{Float64}}=Vector{Float64}[],
                                       envelope_method::Symbol=:dd, milp_encoding::Symbol=:indicator,
                                       verbose::Bool=false)
    td = sd.td
    S = td.S
    t0 = time()
    if td.eps_tilde == 0.0
        return Dict{Symbol,Any}(:beliefs => [copy(sd.q_tilde)], :complete => true, :n_pieces => 0,
                                :n_sep => 0, :hs => Vector{Float64}[], :thetas => Vector{Float64}[],
                                :patterns => [], :time => 0.0, :sep_time => 0.0, :pat_time => 0.0)
    end
    ε⁺ = min(td.eps_tilde + δ_ball, 1.0)
    oracle = _build_follower_oracle(sd; lp_optimizer)
    thetas = Vector{Float64}[]
    hs = Vector{Float64}[]
    function add_piece!(h, θ)
        any(t -> maximum(abs.(t .- θ)) <= 1e-9, thetas) && return false
        push!(thetas, θ); push!(hs, h); return true
    end
    # seed: q̃ + warm start h들
    _, h0, θ0 = _oracle!(oracle, sd, copy(sd.q_tilde))
    add_piece!(h0, θ0)
    for h in warm_hs
        θ = zeros(S)
        for s in 1:S
            for k in 1:td.num_arcs
                set_upper_bound(oracle.mf_y[s][k], sd.c[k, s] + h[k])
            end
            optimize!(oracle.mf[s]); θ[s] = objective_value(oracle.mf[s])
        end
        add_piece!(h, θ)
    end

    # (1) envelope 인증: :dd (𝒟̃⁺ 위 lower-envelope 꼭짓점 검증) | :separation (bilinear 분리)
    n_sep = 0
    env_ok = false
    t_sep0 = time()
    if envelope_method == :dd
        vc = generate_belief_vertex_cover(sd; lp_optimizer, time_limit=sep_time_limit,
                                          warm_hs=copy(hs), eps_override=ε⁺)
        for (h, θ) in zip(vc[:hs], vc[:thetas])
            add_piece!(h, θ)
        end
        env_ok = vc[:complete]
        n_sep = vc[:n_oracle]
    elseif !(envelope_method in (:separation, :milp))
        error("envelope_method must be :dd, :separation or :milp, got $envelope_method")
    end
    while envelope_method in (:separation, :milp) && length(thetas) < max_pieces
        # 누적 시간 제한 (sep_time_limit: 인증 단계 전체, time_limit: cover 전체)
        rem = min(sep_time_limit === nothing ? Inf : sep_time_limit - (time() - t_sep0),
                  time_limit === nothing ? Inf : time_limit - (time() - t0))
        rem <= 0 && break
        rem = isfinite(rem) ? rem : nothing
        viol, ps, cert, bnd = envelope_method == :milp ?
            _envelope_separation_milp(sd, thetas, ε⁺; optimizer, lp_optimizer,
                                      encoding=milp_encoding, time_limit=rem) :
            _envelope_separation(sd, thetas, ε⁺; optimizer, time_limit=rem)
        n_sep += 1
        verbose && (@printf("  min-cover sep %d: pieces=%d viol=%.3e bound=%.3e cert=%s\n",
                            n_sep, length(thetas), viol, bnd, cert); flush(stdout))
        if cert
            env_ok = true
            break
        end
        viol > 1e-7 || break                # 증명 실패(시간) & 위반점 없음 → 불완전
        φ, h, θ = _oracle!(oracle, sd, ps)
        add_piece!(h, θ) || break
    end
    sep_time = time() - t_sep0

    # (2) 극대 패턴 열거
    rem2 = time_limit === nothing ? nothing : time_limit - (time() - t0)
    reps, pats, pat_ok, pat_time = _maximal_patterns(sd, thetas; optimizer, lp_optimizer,
                                                     time_limit=rem2)
    verbose && (@printf("  min-cover: pieces=%d patterns=%d env_ok=%s pat_ok=%s (sep %.1fs, pat %.1fs)\n",
                        length(thetas), length(reps), env_ok, pat_ok, sep_time, pat_time); flush(stdout))
    return Dict{Symbol,Any}(:beliefs => reps, :complete => env_ok && pat_ok,
                            :n_pieces => length(thetas), :n_sep => n_sep, :hs => hs,
                            :thetas => thetas, :patterns => pats, :time => time() - t0,
                            :sep_time => sep_time, :pat_time => pat_time)
end


# ============================================================================
# Route 4: Belief-bilinear (belief 공간 분기한정을 Gurobi spatial B&B에 위임)
# ============================================================================
#
# ccg.tex 식 (Fp)의 strong-duality 행은 belief p가 변수일 때 bilinear이지만,
# 그 bilinear 항은 Σ_s p_s f'_s — 스칼라 곱 **S개뿐** (f'_s = 시나리오 s 흐름량).
# leader 쪽도 Σ_s r_s t_s 로 S개. 따라서
#   V*(x̄) = max  Σ_s r_s t_s
#           s.t. p ∈ 𝒟̃,  primal/dual feasibility (α=h, y', f', ρ, λ, μ),  N_tsᵀμ^s = p_s,
#                Σ_s p_s f'_s = wρ + Σ c λ,         ← bilinear S개
#                평가 copy (y, t),  (r,a,b) ∈ 𝒬      ← bilinear S개 (leader=:bilinear)
# p 고정 시 ℱ(p) 그대로 (Lemma linmem) → 정확. bilinear 항 2S개 (Ω 정식화는 2KS개).
# spatial B&B는 p, r에서만 분기 → belief 공간 B&B + leader 값 가지치기.

"""
    evaluate_belief_bilinear(sd; optimizer, leader=:bilinear, encoding=:indicator,
                             time_limit=nothing, mip_gap=1e-6, silent=true)
"""
function evaluate_belief_bilinear(sd::StructEvalData; optimizer, leader::Symbol=:bilinear,
                                  encoding::Symbol=:indicator, M_dual_L=nothing,
                                  time_limit=nothing, mip_gap=1e-6, silent::Bool=true)
    td = sd.td
    K, S, w = td.num_arcs, td.S, td.w
    q̃ = sd.q_tilde
    ε̃ = td.eps_tilde
    ML = M_dual_L === nothing ? _default_M_dual_L(sd) : Float64(M_dual_L)
    t0 = time()
    model = _make_mip(optimizer; silent, time_limit, mip_gap)
    set_optimizer_attribute(model, "NonConvex", 2)

    # belief: TV 볼에서 유도되는 좌표별 범위 (McCormick 간격 축소)
    p = @variable(model, [s=1:S], lower_bound = max(0.0, q̃[s] - 2ε̃),
                  upper_bound = min(1.0, q̃[s] + 2ε̃), base_name = "p")
    η = @variable(model, [1:S], lower_bound = 0.0, base_name = "η")
    @constraint(model, sum(p) == 1)
    @constraint(model, [s=1:S], η[s] >= p[s] - q̃[s])
    @constraint(model, [s=1:S], η[s] >= q̃[s] - p[s])
    @constraint(model, 0.5 * sum(η) <= ε̃)

    P = _add_follower_primal!(model, sd)
    D = _add_follower_dual!(model, sd)
    for s in 1:S                                   # 유효 범위 (f'_s ≤ Q_s(α) ≤ Tmax_s)
        set_lower_bound(P.fp[s], -sd.Fabs[s])
        set_upper_bound(P.fp[s], sd.Tmax[s])
    end
    @constraint(model, [s=1:S], D.ntsμ[s] == p[s])
    @constraint(model, sum(p[s] * P.fp[s] for s in 1:S) ==
                       w * D.ρ + sum(sd.c[k, s] * D.λ[k, s] for k in 1:K, s in 1:S))

    E = _add_eval_block!(model, sd, P.α)
    L = _add_leader!(model, sd, E.t; leader, encoding, M_dual_L=ML)
    @objective(model, Max, L.obj)
    optimize!(model)

    res = _mip_result(model, P.α, Dict(:route => :belief_bilinear, :leader => leader))
    res[:p_tilde] = value.(p)
    res[:time] = time() - t0
    return res
end


# ============================================================================
# Route 5: Belief-space branch-and-bound (Ω relaxation + belief 분기 + 정확한 잎 평가)
# ============================================================================
#
# Ω 정식화의 bilinear은 ζL = α·b (b = r (CVaR) 또는 a: leader belief 가중치) 와
# ζF = α·d (d: follower belief). 두 곱 모두 한쪽 인자가 belief 좌표 → belief 좌표 상자
# [ℓ, u]만 분기해도 상자가 점으로 줄면 McCormick이 정확 → 수렴.
#   노드 상한: Ω에서 ζ = α·b 를 McCormick 4부등식으로 대체한 LP (계수만 갱신, warm start)
#   노드 하한: LP 해의 follower belief d̂ 에서 W(d̂) = max{V̄ᴸ(α) : α ∈ Ψ(x̄, d̂)} (정확)
#   분기: Σ_k |ζ_ks − α̂_k b̂_s| 가 가장 큰 (그룹, s) 좌표를 b̂_s (또는 중점) 에서 이분
#   선택: best-bound

mutable struct _BnBNode
    ub::Float64
    lo::Dict{Symbol,Vector{Float64}}     # :L, :F → belief 좌표 하한
    hi::Dict{Symbol,Vector{Float64}}
    depth::Int
end


"""
    _WEvaluator — 고정 belief p에서 W(p) 를 반복 평가 (모델 1회 빌드, belief 행만 갱신)
"""
struct _WEvaluator
    model::Model
    α::Vector{VariableRef}
    belief_rows::Vector{ConstraintRef}
    sd_row::ConstraintRef
    fp::Vector{VariableRef}
end

function _build_W_evaluator(sd::StructEvalData; optimizer, leader::Symbol=:bilinear,
                            encoding::Symbol=:indicator, mip_gap=1e-6)
    S = sd.td.S
    model = _make_mip(optimizer; silent=true, mip_gap)
    P = _add_follower_primal!(model, sd)
    D = _add_follower_dual!(model, sd)
    br, sr = _add_sd_membership!(model, sd, P, D, copy(sd.q_tilde))
    E = _add_eval_block!(model, sd, P.α)
    L = _add_leader!(model, sd, E.t; leader, encoding, M_dual_L=_default_M_dual_L(sd))
    @objective(model, Max, L.obj)
    return _WEvaluator(model, P.α, collect(br), sr, P.fp)
end

"""W(p): (value, bound, α, exact). time_limit 초과 시 incumbent (없으면 -Inf)."""
function _eval_W!(we::_WEvaluator, p::Vector{Float64}; time_limit=nothing, cutoff=nothing)
    for s in eachindex(p)
        set_normalized_rhs(we.belief_rows[s], p[s])
        set_normalized_coefficient(we.sd_row, we.fp[s], p[s])
    end
    time_limit === nothing || set_time_limit_sec(we.model, max(time_limit, 1e-3))
    if cutoff !== nothing
        set_optimizer_attribute(we.model, "Cutoff", cutoff)
    end
    optimize!(we.model)
    st = termination_status(we.model)
    if !has_values(we.model)
        return -Inf, (try objective_bound(we.model) catch; Inf end), nothing, false
    end
    return objective_value(we.model), objective_bound(we.model),
           max.(value.(we.α), 0.0), st == MOI.OPTIMAL
end


"""
    evaluate_belief_bnb(sd; lp_optimizer, mip_optimizer, w_leader=:bilinear,
                        time_limit=300, tol=1e-4, max_nodes=100_000, verbose=false)

Belief 공간 분기한정. Returns Dict(:V, :V_bound, :α, :p_tilde, :is_exact, :nodes, :n_W, :time,
:root_ub).
"""
function evaluate_belief_bnb(sd::StructEvalData; lp_optimizer, mip_optimizer,
                             w_leader::Symbol=:bilinear, time_limit=300.0, tol::Float64=1e-4,
                             max_nodes::Int=100_000, verbose::Bool=false)
    td = sd.td
    K, S, w = td.num_arcs, td.S, td.w
    t0 = time()

    # ---- Ω LP relaxation (bilinear 등식 삭제 → McCormick) ----
    lp, v = build_true_dro_subproblem(td, sd.x_bar; optimizer=lp_optimizer, silent=true)
    use_cvar = v[:_use_cvar]
    bvar = Dict(:L => (use_cvar ? v[:r] : v[:a]), :F => v[:d])
    ζ = Dict(:L => v[:ζL], :F => v[:ζF])
    α = v[:α]
    for name in (:ζL_def, :ζF_def)
        delete.(lp, lp[name]); unregister(lp, name)
    end
    root_lo = Dict(g => [lower_bound(bvar[g][s]) for s in 1:S] for g in (:L, :F))
    root_hi = Dict(g => [upper_bound(bvar[g][s]) for s in 1:S] for g in (:L, :F))
    αU = w
    # mc[g][j][k,s]: j=1..4  (ζ ≥ ℓα, ζ ≥ uα + αU b − αU u, ζ ≤ uα, ζ ≤ ℓα + αU b − αU ℓ)
    mc = Dict{Symbol,Vector{Matrix{ConstraintRef}}}()
    for g in (:L, :F)
        z, b = ζ[g], bvar[g]
        c1 = @constraint(lp, [k=1:K, s=1:S], z[k, s] - root_lo[g][s] * α[k] >= 0)
        c2 = @constraint(lp, [k=1:K, s=1:S], z[k, s] - root_hi[g][s] * α[k] - αU * b[s] >= -αU * root_hi[g][s])
        c3 = @constraint(lp, [k=1:K, s=1:S], z[k, s] - root_hi[g][s] * α[k] <= 0)
        c4 = @constraint(lp, [k=1:K, s=1:S], z[k, s] - root_lo[g][s] * α[k] - αU * b[s] <= -αU * root_lo[g][s])
        mc[g] = [Matrix(c1), Matrix(c2), Matrix(c3), Matrix(c4)]
    end

    function set_box!(lo, hi)
        for g in (:L, :F), s in 1:S
            ℓ, u = lo[g][s], hi[g][s]
            set_lower_bound(bvar[g][s], ℓ); set_upper_bound(bvar[g][s], u)
            for k in 1:K
                set_normalized_coefficient(mc[g][1][k, s], α[k], -ℓ)
                set_normalized_coefficient(mc[g][2][k, s], α[k], -u)
                set_normalized_rhs(mc[g][2][k, s], -αU * u)
                set_normalized_coefficient(mc[g][3][k, s], α[k], -u)
                set_normalized_coefficient(mc[g][4][k, s], α[k], -ℓ)
                set_normalized_rhs(mc[g][4][k, s], -αU * ℓ)
            end
        end
    end
    function solve_lp!(lo, hi)
        set_box!(lo, hi)
        optimize!(lp)
        termination_status(lp) == MOI.OPTIMAL || return -Inf, nothing
        sol = (α=value.(α), b=Dict(g => value.(bvar[g]) for g in (:L, :F)),
               ζ=Dict(g => value.(ζ[g]) for g in (:L, :F)))
        return objective_value(lp), sol
    end

    we = _build_W_evaluator(sd; optimizer=mip_optimizer, leader=w_leader)
    LB, best_α, best_p = -Inf, zeros(K), copy(sd.q_tilde)
    evaluated = Vector{Float64}[]
    n_W = 0
    function try_incumbent!(p)
        pp = max.(p, 0.0); pp ./= sum(pp)
        any(e -> maximum(abs.(e .- pp)) <= 1e-7, evaluated) && return
        push!(evaluated, pp)
        rem = time_limit - (time() - t0)
        rem <= 0 && return
        Wv, _, αv, _ = _eval_W!(we, pp; time_limit=rem)
        n_W += 1
        if αv !== nothing && Wv > LB
            LB, best_α, best_p = Wv, αv, pp
        end
    end

    # ---- root ----
    try_incumbent!(copy(sd.q_tilde))
    ub0, sol0 = solve_lp!(root_lo, root_hi)
    sol0 === nothing && error("belief B&B: root LP infeasible")
    try_incumbent!(sol0.b[:F])
    open = _BnBNode[_BnBNode(ub0, deepcopy(root_lo), deepcopy(root_hi), 0)]
    nodes = 1
    global_ub = ub0
    status = :TimeLimit
    while true
        isempty(open) && (global_ub = LB; status = :Optimal; break)
        idx = argmax(n.ub for n in open)
        node = open[idx]
        global_ub = node.ub
        if global_ub - LB <= tol * max(1.0, abs(LB))
            status = :Optimal; break
        end
        (time() - t0 > time_limit || nodes >= max_nodes) && break
        deleteat!(open, idx)

        # 이 노드의 LP 재풀이 (자식 생성용 해 필요)
        ub, sol = solve_lp!(node.lo, node.hi)
        sol === nothing && continue
        ub = min(ub, node.ub)
        ub <= LB + tol * max(1.0, abs(LB)) && continue
        try_incumbent!(sol.b[:F])

        # 분기 좌표: McCormick 위반 최대
        best_g, best_s, best_v = :F, 1, -1.0
        for g in (:L, :F), s in 1:S
            (node.hi[g][s] - node.lo[g][s]) <= 1e-9 && continue
            viol = sum(abs(sol.ζ[g][k, s] - sol.α[k] * sol.b[g][s]) for k in 1:K)
            viol > best_v && ((best_g, best_s, best_v) = (g, s, viol))
        end
        if best_v <= 1e-9          # relaxation 정확 → LP 해가 Ω 가능해: 그 값이 이 노드의 최적
            if ub > LB
                LB, best_α, best_p = ub, max.(sol.α, 0.0), sol.b[:F] ./ sum(sol.b[:F])
            end
            continue
        end
        ℓ, u = node.lo[best_g][best_s], node.hi[best_g][best_s]
        m = sol.b[best_g][best_s]
        (m - ℓ < 0.1 * (u - ℓ) || u - m < 0.1 * (u - ℓ)) && (m = (ℓ + u) / 2)
        for (newlo, newhi) in ((ℓ, m), (m, u))
            lo2 = deepcopy(node.lo); hi2 = deepcopy(node.hi)
            lo2[best_g][best_s] = newlo; hi2[best_g][best_s] = newhi
            cub, csol = solve_lp!(lo2, hi2)
            nodes += 1
            csol === nothing && continue
            cub = min(cub, ub)
            cub > LB + tol * max(1.0, abs(LB)) &&
                push!(open, _BnBNode(cub, lo2, hi2, node.depth + 1))
        end
        if verbose && nodes % 50 < 2
            @printf("  bnb: nodes=%d open=%d LB=%.6f UB=%.6f nW=%d t=%.1fs\n",
                    nodes, length(open), LB, global_ub, n_W, time() - t0); flush(stdout)
        end
    end
    global_ub = max(global_ub, LB)
    return Dict{Symbol,Any}(:V => LB, :V_bound => global_ub, :α => best_α, :p_tilde => best_p,
                            :is_exact => status == :Optimal, :nodes => nodes, :n_W => n_W,
                            :root_ub => ub0, :time => time() - t0, :route => :belief_bnb)
end


# ============================================================================
# Route 6: Follower-response CCG (Zeng형, subproblem 내부)
# ============================================================================
#
# V*(x̄) = max_{α, p̃∈𝒟̃} min_{h̄∈H} { V̄ᴸ(α) : p̃ᵀQ(α) ≥ p̃ᵀQ(h̄) }.
# 열 = follower 대안 반응 h̄_j (조각 θʲ = Q(h̄_j)). 일부만 넣은 master는 relaxation → 상한.
#   master:  max V̄ᴸ(α)  s.t.  p̃ ∈ 𝒟̃,  p̃ᵀf' ≥ p̃ᵀθʲ ∀j,  f' ≤ Q(α) (흐름 copy)
#   pricing: (α̂, p̂)에서 follower LP: φ(p̂) > p̂ᵀQ(α̂) 이면 θ* = Q(h*) 추가 (LP)
#   하한:    h*는 p̂에서의 실제 최적 반응 → V̄ᴸ(h*) 는 가능해 값; α̂가 Ψ(p̂)에 있으면 master 값 달성
# master 두 버전:
#   :bilinear — p̃ 변수, F ≤ Σ p̃_s f'_s (비볼록 곱 S개), F ≥ p̃ᵀθʲ (선형)
#   :milp     — T-complex ∩ 𝒟̃ 의 꼭짓점 v 위의 disjunction:  σ_v=1 ⇒ vᵀf' ≥ φ_T(v)
#               (셀 R_a 안에서 조건이 p̃에 선형 → 존재 ⇔ 셀 꼭짓점에서 성립. bigM = φ_T(v), f' ≥ 0)

"""TV 볼 × [τmin, τmax] DD 초기 상태 (generate_belief_vertex_cover와 동일 구성)."""
function _dd_init_ball(sd::StructEvalData, ε::Float64)
    S = sd.td.S
    q̃ = sd.q_tilde
    d = S + 1
    τmin, τmax = -1.0, maximum(sd.Tmax) + 1.0
    ball_V = _tv_ball_vertices(q̃, ε)
    cons = Vector{Float64}[]
    for s in 1:S
        a = zeros(d); a[s] = 1.0; push!(cons, a)
    end
    push!(cons, vcat(fill(-τmin, S), 1.0))
    push!(cons, vcat(fill(τmax, S), -1.0))
    i_hi = S + 2
    for A in _tv_tight_sets(ball_V, q̃, ε)
        a = zeros(d)
        for s in 1:S
            a[s] = ε + sum(q̃[A]) - (s in A ? 1.0 : 0.0)
        end
        push!(cons, a)
    end
    rays = _DDRay[]
    for v in ball_V, τv in (τmin, τmax)
        r = vcat(v, τv)
        tight = BitSet(i for (i, a) in enumerate(cons)
                       if abs(dot(a, r)) <= 1e-9 * max(1.0, maximum(abs, a)))
        push!(rays, _DDRay(r, tight, false))
    end
    return rays, cons, d, i_hi
end

"""Q(α) = (Q_s(α))_s via 시나리오별 max-flow LP."""
function _Q_of!(oracle::_FollowerOracle, sd::StructEvalData, α::Vector{Float64})
    S, K = sd.td.S, sd.td.num_arcs
    θ = zeros(S)
    for s in 1:S
        for k in 1:K
            set_upper_bound(oracle.mf_y[s][k], sd.c[k, s] + max(α[k], 0.0))
        end
        optimize!(oracle.mf[s])
        θ[s] = objective_value(oracle.mf[s])
    end
    return θ
end

"""master 공통 블록: α, 흐름 copy (f' ≥ 0), 평가 copy, leader."""
function _ccg_master_common!(model, sd::StructEvalData; leader::Symbol, encoding::Symbol=:indicator)
    td = sd.td
    K, S, m, w = td.num_arcs, td.S, td.nv1, td.w
    α = @variable(model, [1:K], lower_bound = 0.0, upper_bound = w, base_name = "α")
    @constraint(model, sum(α) <= w)
    yp = @variable(model, [1:K, 1:S], lower_bound = 0.0, base_name = "yp")
    fp = @variable(model, [s=1:S], lower_bound = 0.0, upper_bound = sd.Tmax[s], base_name = "fp")
    @constraint(model, [j=1:m, s=1:S],
        sum(a * yp[k, s] for (k, a) in sd.Ny_rows[j]; init=AffExpr(0.0)) + td.Nts[j] * fp[s] == 0)
    @constraint(model, [k=1:K, s=1:S], yp[k, s] <= sd.c[k, s] + α[k])
    E = _add_eval_block!(model, sd, α)
    L = _add_leader!(model, sd, E.t; leader, encoding, M_dual_L=_default_M_dual_L(sd))
    @objective(model, Max, L.obj)
    return α, fp
end


"""
    evaluate_response_ccg(sd; master=:bilinear, mip_optimizer, lp_optimizer, leader=nothing,
                          time_limit=300, tol=1e-4, max_iter=500, verbose=false)

Follower-response CCG. leader 기본값: :bilinear master → :bilinear, :milp master → :kkt (MILP 유지).
Returns Dict(:V (하한), :V_bound (상한), :α, :p_tilde, :is_exact, :iters, :n_pieces, :time,
:master_time, :n_vertices (milp)).
"""
function evaluate_response_ccg(sd::StructEvalData; master::Symbol=:bilinear, mip_optimizer,
                               lp_optimizer, leader=nothing, time_limit=300.0, tol::Float64=1e-4,
                               max_iter::Int=500, verbose::Bool=false)
    td = sd.td
    K, S = td.num_arcs, td.S
    q̃ = sd.q_tilde
    ε̃ = td.eps_tilde
    ldr = leader === nothing ? (master == :milp ? :kkt : :bilinear) : leader
    master in (:bilinear, :milp) || error("master must be :bilinear or :milp, got $master")
    t0 = time()
    oracle = _build_follower_oracle(sd; lp_optimizer)

    thetas = Vector{Float64}[]
    LB, best_α, best_p = -Inf, zeros(K), copy(q̃)
    UB = Inf
    function add_piece_at!(p)
        φ, h, θ = _oracle!(oracle, sd, p)
        # h는 p에서의 실제 follower 최적 반응 → 가능해: V̄ᴸ(h) 하한
        lv = _leader_lp_pd(sd, θ; lp_optimizer)
        if lv !== nothing && lv[1] > LB
            LB, best_α, best_p = lv[1], h, copy(p)
        end
        isnew = !any(t -> maximum(abs.(t .- θ)) <= 1e-9, thetas)
        isnew && push!(thetas, θ)
        return isnew
    end
    add_piece_at!(copy(q̃))

    # ---- master 모델 ----
    local mdl, α, fp, pvar, Fvar
    dd = nothing
    if master == :bilinear
        mdl = _make_mip(mip_optimizer; silent=true, mip_gap=1e-6)
        set_optimizer_attribute(mdl, "NonConvex", 2)
        α, fp = _ccg_master_common!(mdl, sd; leader=ldr)
        pvar = @variable(mdl, [s=1:S], lower_bound = max(0.0, q̃[s] - 2ε̃),
                         upper_bound = min(1.0, q̃[s] + 2ε̃), base_name = "p")
        η = @variable(mdl, [1:S], lower_bound = 0.0)
        @constraint(mdl, sum(pvar) == 1)
        @constraint(mdl, [s=1:S], η[s] >= pvar[s] - q̃[s])
        @constraint(mdl, [s=1:S], η[s] >= q̃[s] - pvar[s])
        @constraint(mdl, 0.5 * sum(η) <= ε̃)
        Fvar = @variable(mdl, base_name = "F")
        @constraint(mdl, Fvar <= sum(pvar[s] * fp[s] for s in 1:S))
        for θ in thetas
            @constraint(mdl, Fvar >= sum(pvar[s] * θ[s] for s in 1:S))
        end
    else
        dd = ε̃ > 0 ? _dd_init_ball(sd, ε̃) : nothing
        if dd !== nothing
            for θ in thetas
                _dd_add!(dd[1], dd[2], vcat(-θ, 1.0), dd[3])
            end
        end
    end
    n_added = length(thetas)

    function milp_vertices()
        dd === nothing && return [copy(q̃)]
        rays, _, _, i_hi = dd
        V = Vector{Float64}[]
        for r in rays
            i_hi in r.tight && continue
            p = max.(r.r[1:S], 0.0); p ./= sum(p)
            any(v -> maximum(abs.(v .- p)) <= 1e-10, V) || push!(V, p)
        end
        return V
    end

    iters = 0
    master_time = 0.0
    n_vert = 0
    status = :TimeLimit
    while iters < max_iter
        iters += 1
        rem = time_limit - (time() - t0)
        rem <= 0 && break
        local α̂, p̂, mval, mbound
        if master == :bilinear
            for θ in thetas[n_added+1:end]
                @constraint(mdl, Fvar >= sum(pvar[s] * θ[s] for s in 1:S))
            end
            n_added = length(thetas)
            set_time_limit_sec(mdl, rem)
            master_time += @elapsed optimize!(mdl)
            has_values(mdl) || break
            mval, mbound = objective_value(mdl), objective_bound(mdl)
            α̂ = max.(value.(α), 0.0)
            p̂ = max.(value.(pvar), 0.0); p̂ ./= sum(p̂)
            termination_status(mdl) == MOI.OPTIMAL || (UB = min(UB, mbound))
        else
            if dd !== nothing
                for θ in thetas[n_added+1:end]
                    _dd_add!(dd[1], dd[2], vcat(-θ, 1.0), dd[3])
                end
            end
            n_added = length(thetas)
            V = milp_vertices()
            n_vert = length(V)
            φT = [maximum(dot(v, θ) for θ in thetas) for v in V]
            mdl = _make_mip(mip_optimizer; silent=true, mip_gap=1e-6)
            α, fp = _ccg_master_common!(mdl, sd; leader=ldr)
            σ = @variable(mdl, [1:n_vert], binary = true)
            @constraint(mdl, sum(σ) == 1)
            # σ_v = 1 ⇒ vᵀf' ≥ φ_T(v);  f' ≥ 0 이므로 bigM = φ_T(v) 로 충분
            @constraint(mdl, [i=1:n_vert],
                sum(V[i][s] * fp[s] for s in 1:S) >= φT[i] * σ[i])
            set_time_limit_sec(mdl, rem)
            master_time += @elapsed optimize!(mdl)
            has_values(mdl) || break
            mval, mbound = objective_value(mdl), objective_bound(mdl)
            α̂ = max.(value.(α), 0.0)
            p̂ = V[argmax(value.(σ))]
        end
        UB = min(UB, mbound)

        # ---- pricing: α̂ 가 p̂ 에서 follower 최적인가? ----
        Qα = _Q_of!(oracle, sd, α̂)
        φp = _phi!(oracle, sd, p̂)
        if dot(p̂, Qα) >= φp - 1e-7 * max(1.0, abs(φp))
            lv = _leader_lp_pd(sd, Qα; lp_optimizer)        # α̂ 가능 → V̄ᴸ(α̂) 하한
            if lv !== nothing && lv[1] > LB
                LB, best_α, best_p = lv[1], α̂, p̂
            end
        else
            if !add_piece_at!(p̂)
                # master가 이미 이 조각을 만족 → 위반은 허용오차 수준: 더 진행 불가
                verbose && println("  ccg: pricing piece already in T (수치 허용오차) → 중단")
                break
            end
        end
        verbose && (@printf("  ccg[%s] it=%d pieces=%d LB=%.6f UB=%.6f master=%.2fs%s\n",
                            master, iters, length(thetas), LB, UB, master_time,
                            master == :milp ? @sprintf(" |V|=%d", n_vert) : ""); flush(stdout))
        if UB - LB <= tol * max(1.0, abs(LB))
            status = :Optimal
            break
        end
    end
    return Dict{Symbol,Any}(:V => LB, :V_bound => max(UB, LB), :α => best_α, :p_tilde => best_p,
                            :is_exact => status == :Optimal, :iters => iters,
                            :n_pieces => length(thetas), :time => time() - t0,
                            :master_time => master_time, :n_vertices => n_vert,
                            :route => Symbol("ccg_", master))
end


# ============================================================================
# Route 1: Decomposed (ccg.tex 식 evalMILP, belief별)
# ============================================================================

"""
    _add_sd_membership!(model, sd, P, D, p̃)

ℱ(p̃) (ccg.tex 식 Fp)의 belief 의존 행: N_tsᵀμ^s = p̃_s, strong duality.
행 ref를 반환 → belief 교체 시 계수/rhs만 갱신.
"""
function _add_sd_membership!(model, sd::StructEvalData, P, D, p̃::Vector{Float64})
    td = sd.td
    K, S, w = td.num_arcs, td.S, td.w
    belief_rows = @constraint(model, [s=1:S], D.ntsμ[s] == p̃[s])
    sd_row = @constraint(model,
        sum(p̃[s] * P.fp[s] for s in 1:S) - w * D.ρ -
        sum(sd.c[k, s] * D.λ[k, s] for k in 1:K, s in 1:S) == 0)
    return belief_rows, sd_row
end


"""
    evaluate_decomposed(sd, beliefs; optimizer, encoding=:indicator, M_dual_L=nothing,
                        time_limit=nothing, mip_gap=1e-6, silent=true)

각 대표 belief p̃ʲ에 대해 W_j = max{V̄ᴸ(x̄,α) : α∈Ψ(x̄,p̃ʲ)} (follower 쪽 binary 없음).
모델 1회 빌드 후 belief 행만 갱신하여 순차 solve. V* = max_j W_j.
"""
function evaluate_decomposed(sd::StructEvalData, beliefs::Vector{Vector{Float64}};
                             optimizer, encoding::Symbol=:indicator, M_dual_L=nothing,
                             time_limit=nothing, mip_gap=1e-6, silent::Bool=true, leader::Symbol=:kkt,
                             prune::Bool=true)
    _check_encoding(encoding)
    isempty(beliefs) && error("evaluate_decomposed: empty belief cover")
    td = sd.td
    S = td.S
    ML = M_dual_L === nothing ? _default_M_dual_L(sd) : Float64(M_dual_L)
    t0 = time()

    model = _make_mip(optimizer; silent, mip_gap)
    P = _add_follower_primal!(model, sd)
    D = _add_follower_dual!(model, sd)
    belief_rows, sd_row = _add_sd_membership!(model, sd, P, D, beliefs[1])
    E = _add_eval_block!(model, sd, P.α)
    L = _add_leader!(model, sd, E.t; leader, encoding, M_dual_L=ML)
    @objective(model, Max, L.obj)

    W = fill(-Inf, length(beliefs))
    W_bound = fill(Inf, length(beliefs))
    α_list = Vector{Vector{Float64}}(undef, length(beliefs))
    all_exact = true
    n_pruned = 0
    best = -Inf
    for (j, pj) in enumerate(beliefs)
        for s in 1:S
            set_normalized_rhs(belief_rows[s], pj[s])
            set_normalized_coefficient(sd_row, P.fp[s], pj[s])
        end
        if time_limit !== nothing
            remaining = time_limit - (time() - t0)
            if remaining <= 0          # 시간 소진: 미방문 belief는 bound=∞ (하한만 반환)
                all_exact = false
                break
            end
            set_time_limit_sec(model, remaining)
        end
        # prune: W_j ≤ incumbent 이면 Gurobi CUTOFF로 조기 종료 (W_j 값 불필요)
        cutoff = isfinite(best) ? best + 1e-6 * max(1.0, abs(best)) : nothing
        if prune && cutoff !== nothing
            set_optimizer_attribute(model, "Cutoff", cutoff)
        end
        optimize!(model)
        st = termination_status(model)
        if prune && cutoff !== nothing && (st == MOI.OBJECTIVE_LIMIT ||
                                           (st == MOI.INFEASIBLE && !has_values(model)))
            W_bound[j] = cutoff
            n_pruned += 1
            continue
        end
        if !has_values(model)          # TIME_LIMIT 등, 해 없음: bound만 기록하고 종료
            W_bound[j] = max(something(cutoff, -Inf), try objective_bound(model) catch; Inf end)
            all_exact = false
            break
        end
        r = _mip_result(model, P.α, Dict())
        W[j] = r[:V]
        W_bound[j] = r[:V_bound]
        α_list[j] = r[:α]
        all_exact &= r[:is_exact]
        best = max(best, W[j])
    end

    any(isfinite, W) || error("evaluate_decomposed: no belief evaluated within time limit")
    jstar = argmax(W)
    return Dict{Symbol,Any}(
        :V => W[jstar], :V_bound => maximum(W_bound), :α => α_list[jstar],
        :p_tilde => beliefs[jstar], :W => W, :is_exact => all_exact,
        :route => :decomposed, :encoding => encoding,
        :n_bin => count(is_binary, all_variables(model)),
        :n_beliefs => length(beliefs), :n_pruned => n_pruned, :time => time() - t0)
end


# ============================================================================
# Route 2: Disjunctive (ccg.tex 식 disjMILP)
# ============================================================================

"""
    evaluate_disjunctive(sd, beliefs; optimizer, encoding=:indicator, M_dual_L=nothing,
                         time_limit=nothing, mip_gap=1e-6, silent=true)

선택 binary σ ∈ {0,1}^J, Σσ=1. belief 의존 행 (N_tsᵀμ^s = p̃ʲ_s, strong duality)만
σ_j로 활성화. bigM 값 (활성 belief j* 대비 비활성 j'의 행 잔차):
  |N_tsᵀμ^s - p̃ʲ'_s| = |p̃ʲ*_s - p̃ʲ'_s| ≤ 1
  |Σ p̃ʲ'_s f'_s - wρ - Σcλ| = |Σ (p̃ʲ'_s - p̃ʲ*_s) f'_s| ≤ 2 max_s Fabs_s
"""
function evaluate_disjunctive(sd::StructEvalData, beliefs::Vector{Vector{Float64}};
                              optimizer, encoding::Symbol=:indicator, M_dual_L=nothing,
                              time_limit=nothing, mip_gap=1e-6, silent::Bool=true, leader::Symbol=:kkt)
    _check_encoding(encoding)
    isempty(beliefs) && error("evaluate_disjunctive: empty belief cover")
    td = sd.td
    K, S, w = td.num_arcs, td.S, td.w
    J = length(beliefs)
    ML = M_dual_L === nothing ? _default_M_dual_L(sd) : Float64(M_dual_L)
    t0 = time()

    model = _make_mip(optimizer; silent, time_limit, mip_gap)
    P = _add_follower_primal!(model, sd)
    D = _add_follower_dual!(model, sd)
    σ = @variable(model, [1:J], binary = true, base_name = "σ")
    @constraint(model, sum(σ) == 1)

    M_nts = 1.0
    M_sd = 2.0 * maximum(sd.Fabs)
    dual_obj = @expression(model, w * D.ρ + sum(sd.c[k, s] * D.λ[k, s] for k in 1:K, s in 1:S))
    for (j, pj) in enumerate(beliefs)
        sd_expr = @expression(model, sum(pj[s] * P.fp[s] for s in 1:S) - dual_obj)
        if encoding == :indicator
            for s in 1:S
                @constraint(model, σ[j] --> {D.ntsμ[s] == pj[s]})
            end
            @constraint(model, σ[j] --> {sd_expr == 0})
        else
            for s in 1:S
                @constraint(model, D.ntsμ[s] - pj[s] <= M_nts * (1 - σ[j]))
                @constraint(model, pj[s] - D.ntsμ[s] <= M_nts * (1 - σ[j]))
            end
            @constraint(model, sd_expr <= M_sd * (1 - σ[j]))
            @constraint(model, -sd_expr <= M_sd * (1 - σ[j]))
        end
    end

    E = _add_eval_block!(model, sd, P.α)
    L = _add_leader!(model, sd, E.t; leader, encoding, M_dual_L=ML)
    @objective(model, Max, L.obj)
    optimize!(model)

    res = _mip_result(model, P.α, Dict(:route => :disjunctive, :encoding => encoding))
    jstar = argmax(value.(σ))
    res[:p_tilde] = beliefs[jstar]
    res[:n_beliefs] = J
    res[:time] = time() - t0
    return res
end


# ============================================================================
# 통합 인터페이스 + cut
# ============================================================================

"""
    evaluate_structured(td, x_bar; method, encoding=:indicator, optimizer, lp_optimizer,
                        cover_kwargs=(;), kwargs...)

method ∈ (:monolithic, :decomposed, :disjunctive). decomposed/disjunctive는 x̄에서
belief cover를 먼저 생성 (Alg.1). cover가 불완전하면 :is_exact=false.
"""
function evaluate_structured(td::TrueDROData, x_bar::Vector{Float64};
                             method::Symbol, encoding::Symbol=:indicator,
                             cover_method::Symbol=:vertex,
                             optimizer, lp_optimizer, cover_kwargs=(;), kwargs...)
    sd = make_struct_eval_data(td, x_bar; lp_optimizer)
    if method == :monolithic
        return evaluate_monolithic(sd; optimizer, encoding, kwargs...)
    elseif method in (:decomposed, :disjunctive)
        cover = if cover_method == :minimal
            generate_belief_minimal_cover(sd; optimizer, lp_optimizer, cover_kwargs...)
        elseif cover_method == :vertex
            generate_belief_vertex_cover(sd; lp_optimizer, cover_kwargs...)
        elseif cover_method == :pattern
            generate_belief_cover(sd; optimizer, encoding, cover_kwargs...)
        else
            error("cover_method must be :minimal, :vertex or :pattern, got $cover_method")
        end
        evalf = method == :decomposed ? evaluate_decomposed : evaluate_disjunctive
        res = evalf(sd, cover[:beliefs]; optimizer, encoding, kwargs...)
        res[:cover_time] = cover[:time]
        res[:cover] = cover
        res[:n_patterns] = haskey(cover, :patterns) ? length(cover[:patterns]) : cover[:n_oracle]
        res[:cover_complete] = cover[:complete]
        res[:time] += cover[:time]
        if !cover[:complete]
            res[:is_exact] = false
            res[:V_bound] = Inf   # 불완전 cover → 상한 정보 없음
        end
        return res
    else
        error("method must be :monolithic, :decomposed or :disjunctive, got $method")
    end
end


"""
    structured_outer_cut(td, x_bar, α_star; lp_optimizer)

α* 고정 ISP-L + ISP-F LP (기존 mini-Benders LP)로 cut 생성.
Returns (cut, Z0_lp, V̄ᴸ, V̄ᶠ). Z0_lp = V̄ᴸ(x̄,α*) + V̄ᶠ(x̄,α*) (α*∈Ψ면 V̄ᶠ=0).
"""
function structured_outer_cut(td::TrueDROData, x_bar::Vector{Float64},
                              α_star::Vector{Float64}; lp_optimizer)
    l_model, l_vars = build_true_dro_isp_leader(td, x_bar, α_star; optimizer=lp_optimizer)
    f_model, f_vars = build_true_dro_isp_follower(td, x_bar, α_star; optimizer=lp_optimizer)
    l_info = solve_isp_leader!(l_model, l_vars, td)
    f_info = solve_isp_follower!(f_model, f_vars, td)
    sub_info = Dict(
        :Z0_val => l_info[:obj_val] + f_info[:obj_val],
        :rho_hat_1_val => l_info[:rho_hat_1_val], :rho_hat_3_val => l_info[:rho_hat_3_val],
        :rho_tilde_1_val => f_info[:rho_tilde_1_val], :rho_tilde_3_val => f_info[:rho_tilde_3_val],
        :rho_psi0_1_val => f_info[:rho_psi0_1_val], :rho_psi0_3_val => f_info[:rho_psi0_3_val],
    )
    cut = compute_true_dro_outer_cut(td, sub_info, x_bar)
    return cut, sub_info[:Z0_val], l_info[:obj_val], f_info[:obj_val]
end


# ============================================================================
# Benders driver (exact cut만 사용하는 단순 루프; 검증용)
# ============================================================================

"""
    structured_benders_optimize!(td; method=:monolithic, encoding=:indicator,
        mip_optimizer, lp_optimizer, max_iter=200, tol=1e-4, verbose=true,
        consistency_tol=1e-4, eval_kwargs=(;), cover_kwargs=(;))

OMP ↔ structured exact 평가. 매 iteration:
  x̄ ← OMP,  V*(x̄), α* ← MILP,  cut ← α* 고정 LP,  UB ← min(UB, V_bound)
MILP 값 V와 LP 값 Z0_lp가 consistency_tol 이상 다르면 error (구현 오류 신호).
"""
function structured_benders_optimize!(td::TrueDROData; method::Symbol=:monolithic,
                                      encoding::Symbol=:indicator,
                                      cover_method::Symbol=:vertex, warm_start::Bool=true,
                                      mip_optimizer, lp_optimizer,
                                      max_iter::Int=200, tol::Float64=1e-4,
                                      verbose::Bool=true, consistency_tol::Float64=1e-4,
                                      eval_kwargs=(;), cover_kwargs=(;))
    K = td.num_arcs
    t_start = time()
    omp_model, omp_vars = build_true_dro_omp(td; optimizer=mip_optimizer)

    lb, ub = -Inf, Inf
    best_x, best_α = zeros(K), zeros(K)
    history = Dict{Symbol,Vector{Any}}(:lb => [], :ub => [], :V => [], :eval_time => [],
                                       :n_beliefs => [], :n_patterns => [])
    status = :MaxIter
    iter = 0
    hs_pool = Vector{Float64}[]     # vertex cover warm start: 이전 x̄들의 h* (H의 점, x 무관)
    for it in 1:max_iter
        iter = it
        optimize!(omp_model)
        termination_status(omp_model) == MOI.OPTIMAL ||
            error("OMP: $(termination_status(omp_model))")
        x̄ = round.(value.(omp_vars[:x]))
        lb = objective_value(omp_model)

        ckw = (method != :monolithic && cover_method == :vertex && warm_start) ?
              merge(cover_kwargs, (; warm_hs=hs_pool)) : cover_kwargs
        ev = evaluate_structured(td, x̄; method, encoding, cover_method,
                                 optimizer=mip_optimizer, lp_optimizer,
                                 cover_kwargs=ckw, eval_kwargs...)
        if haskey(ev, :cover) && haskey(ev[:cover], :hs)
            for h in ev[:cover][:hs]
                any(hp -> maximum(abs.(hp .- h)) <= 1e-9, hs_pool) || push!(hs_pool, h)
            end
        end
        cut, Z0_lp, _, zf = structured_outer_cut(td, x̄, ev[:α]; lp_optimizer)

        if ev[:is_exact] && abs(Z0_lp - ev[:V]) > consistency_tol * max(1.0, abs(ev[:V]))
            error(@sprintf("MILP V=%.8f ≠ α*-fixed LP Z0=%.8f (V̄F=%.2e) at iter %d",
                           ev[:V], Z0_lp, zf, it))
        end
        if ev[:V_bound] < ub
            ub = ev[:V_bound]
            best_x, best_α = copy(x̄), copy(ev[:α])
        end
        add_true_dro_optimality_cut!(omp_model, omp_vars, cut, it)

        push!(history[:lb], lb); push!(history[:ub], ub); push!(history[:V], ev[:V])
        push!(history[:eval_time], ev[:time])
        push!(history[:n_beliefs], get(ev, :n_beliefs, missing))
        push!(history[:n_patterns], get(ev, :n_patterns, missing))
        if verbose
            @printf("[%s/%s] it %3d  LB=%.6f  UB=%.6f  V(x̄)=%.6f  eval=%.2fs%s\n",
                    method, encoding, it, lb, ub, ev[:V], ev[:time],
                    haskey(ev, :n_beliefs) ? @sprintf("  J=%d (pat %d)", ev[:n_beliefs], ev[:n_patterns]) : "")
        end
        if (ub - lb) / max(abs(ub), 1.0) <= tol
            status = :Optimal
            break
        end
    end

    return Dict{Symbol,Any}(:status => status, :Z0 => ub, :x => best_x, :α => best_α,
                            :lower_bound => lb, :upper_bound => ub, :iters => iter,
                            :wall_time => time() - t_start, :history => history)
end
