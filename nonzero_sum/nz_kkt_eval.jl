"""
nz_kkt_eval.jl — V*(x) 의 독립 평가 (Ω 검증용). big-M (θᵁ, λᵁ, πᵁ) 를 쓰지 않는다.

  V*(x) = max  Σ_s r_s · ℓᵀŷˢ                                  (bilinear r × φ, Gurobi NonConvex)
          s.t. (a, r): TV 볼 ε̂ + CVaR_β 재가중                  (sup_P CVaR)
               d: TV 볼 ε̃                                       (follower belief, 변수 → Ψ(x, D̃) = ∪ Ψ(x, P̃))
               α ∈ H, (α, yˢ) 가 follower 2단계 LP (belief d) 의 최적해   ← KKT, 상보성은 indicator
               ŷˢ 가 시나리오 s recourse LP (rhs r_s(x, α)) 의 최적해     ← KKT, 상보성은 indicator
x 를 고정하므로 x·π 곱이 없고, pessimistic 선택은 max 로 그대로 들어간다.
follower dual 은 d_s 로 스케일된 πˢ (Aᵀπˢ ≥ d_s c) 를 써서 선형으로 둔다.
"""

using JuMP, LinearAlgebra
import MathOptInterface as MOI

"v ≥ 0, slack ≥ 0, v·slack = 0 (indicator)"
function _comp!(model, var, slack)
    z = @variable(model, binary = true)
    @constraint(model, z => {var == 0})
    @constraint(model, !z => {slack == 0})
    return z
end

function nz_kkt_value(nd::NZData, x̄; optimizer, time_limit=600.0, gap=1e-6, silent=true)
    A, c, ℓ = nd.A, nd.c, nd.ell
    m, ny, nh, S = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.S
    nW = size(nd.W, 1)
    q, ε̂, ε̃, β = nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta
    model = Model(optimizer)
    silent && set_silent(model)
    set_optimizer_attribute(model, "NonConvex", 2)
    set_optimizer_attribute(model, "MIPGap", gap)
    set_time_limit_sec(model, time_limit)

    # 리더 측정
    @variable(model, 0 <= a[1:S] <= 1); @variable(model, b[1:S] >= 0); @variable(model, r[1:S] >= 0)
    @constraint(model, sum(a) == 1); @constraint(model, sum(r) == 1)
    @constraint(model, [s=1:S], a[s] - b[s] <= q[s]); @constraint(model, [s=1:S], a[s] + b[s] >= q[s])
    @constraint(model, sum(b) <= 2ε̂); @constraint(model, [s=1:S], (1 - β) * r[s] <= a[s])
    set_upper_bound.(r, 1.0)
    # follower belief
    @variable(model, 0 <= d[1:S] <= 1); @variable(model, e[1:S] >= 0)
    @constraint(model, sum(d) == 1)
    @constraint(model, [s=1:S], d[s] - e[s] <= q[s]); @constraint(model, [s=1:S], d[s] + e[s] >= q[s])
    @constraint(model, sum(e) <= 2ε̃)
    # belief 결합 (종속 ambiguity set): d_TV(a, d) ≤ δ
    if nz_coupled(nd)
        @variable(model, cpl[1:S] >= 0)
        @constraint(model, [s=1:S], a[s] - d[s] <= cpl[s]); @constraint(model, [s=1:S], d[s] - a[s] <= cpl[s])
        @constraint(model, sum(cpl) <= 2nz_delta(nd))
    end

    # follower 2단계 LP: max c0ᵀα + Σ_s d_s cᵀyˢ  s.t. A yˢ ≤ r_s(x̄, α), Wα ≤ w, α, y ≥ 0
    @variable(model, 0 <= α[i=1:nh] <= nd.hU[i])
    @variable(model, y[1:ny, 1:S] >= 0)
    @variable(model, π[1:m, 1:S] >= 0)       # d_s 스케일 dual
    @variable(model, κ[1:nW] >= 0)
    rhs(s) = [nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0) +
              (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0) for k in 1:m]
    R = [rhs(s) for s in 1:S]
    slackP = [@expression(model, R[s][k] - sum(A[k, j] * y[j, s] for j in 1:ny if A[k, j] != 0)) for k in 1:m, s in 1:S]
    slackD = [@expression(model, sum(A[k, j] * π[k, s] for k in 1:m if A[k, j] != 0) - c[j] * d[s]) for j in 1:ny, s in 1:S]
    @constraint(model, [k=1:m, s=1:S], slackP[k, s] >= 0)
    @constraint(model, [j=1:ny, s=1:S], slackD[j, s] >= 0)
    slackW = [@expression(model, nd.wvec[l] - sum(nd.W[l, i] * α[i] for i in 1:nh)) for l in 1:nW]
    @constraint(model, [l=1:nW], slackW[l] >= 0)
    slackA = [@expression(model, sum(nd.W[l, i] * κ[l] for l in 1:nW) - nd.c0[i]
                                 - sum(nd.hcoef[k] * π[k, s] for k in 1:m, s in 1:S if nd.hrow[k] == i)) for i in 1:nh]
    @constraint(model, [i=1:nh], slackA[i] >= 0)
    for k in 1:m, s in 1:S; _comp!(model, π[k, s], slackP[k, s]); end
    for j in 1:ny, s in 1:S; _comp!(model, y[j, s], slackD[j, s]); end
    for l in 1:nW; _comp!(model, κ[l], slackW[l]); end
    for i in 1:nh; _comp!(model, α[i], slackA[i]); end

    # 리더 평가 copy: ŷˢ ∈ argmax{cᵀy : Ay ≤ r_s(x̄, α)}, 그 위에서 ℓᵀŷ 최대 (pessimistic)
    @variable(model, ŷ[1:ny, 1:S] >= 0)
    @variable(model, σ[1:m, 1:S] >= 0)
    sP = [@expression(model, R[s][k] - sum(A[k, j] * ŷ[j, s] for j in 1:ny if A[k, j] != 0)) for k in 1:m, s in 1:S]
    sD = [@expression(model, sum(A[k, j] * σ[k, s] for k in 1:m if A[k, j] != 0) - c[j]) for j in 1:ny, s in 1:S]
    @constraint(model, [k=1:m, s=1:S], sP[k, s] >= 0)
    @constraint(model, [j=1:ny, s=1:S], sD[j, s] >= 0)
    for k in 1:m, s in 1:S; _comp!(model, σ[k, s], sP[k, s]); end
    for j in 1:ny, s in 1:S; _comp!(model, ŷ[j, s], sD[j, s]); end

    # φ_s 범위 (spatial B&B 용): α 를 풀어 둔 LP 로 상하한
    φlo, φhi = _phi_range(nd, x̄; optimizer=optimizer)
    @variable(model, φlo[s] <= φ[s=1:S] <= φhi[s])
    @constraint(model, [s=1:S], φ[s] == sum(ℓ[j] * ŷ[j, s] for j in 1:ny if ℓ[j] != 0))
    @objective(model, Max, sum(r[s] * φ[s] for s in 1:S))
    optimize!(model)
    st = termination_status(model)
    (st == MOI.OPTIMAL || (st == MOI.TIME_LIMIT && has_values(model))) || error("KKT eval: $st")
    return Dict(:value => objective_value(model), :bound => objective_bound(model), :status => st,
                :α => value.(α), :a => value.(a), :d => value.(d), :r => value.(r), :φ => value.(φ))
end

"""
nz_kkt_value_reduced — nz_kkt_value 와 같은 V*(x) 를 이진변수 약 절반으로 (big-M 없음, 결과는 같아야 함).

  1. recourse 최적성은 시나리오마다 한 번만: (yˢ, πˢ) 의 KKT (πˢ 는 d 로 스케일하지 않은 dual).
     follower 1단계 최적성은 기대 subgradient Σ_s d_s Bᵀπˢ 로 → d_s·πˢ bilinear (NonConvex 가 처리).
     d_s = 0 인 시나리오도 yˢ 가 recourse 최적이라 아래 2 가 정확하다.
  2. 리더 평가 복사본 ŷˢ 는 KKT 없이 "Aŷ ≤ r_s(x̄, α), cᵀŷ ≥ cᵀyˢ" (= recourse 최적해 집합) 만 요구하고,
     목적의 max 가 그 위에서 ℓᵀŷ 를 최대화 (pessimistic).
원래 정식화 대비 시나리오당 (m + ny) 개 상보성 이진변수가 빠진다.
"""
function nz_kkt_value_reduced(nd::NZData, x̄; optimizer, time_limit=600.0, gap=1e-6, silent=true)
    A, c, ℓ = nd.A, nd.c, nd.ell
    m, ny, nh, S = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.S
    nW = size(nd.W, 1)
    q, ε̂, ε̃, β = nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta
    model = Model(optimizer)
    silent && set_silent(model)
    set_optimizer_attribute(model, "NonConvex", 2)
    set_optimizer_attribute(model, "MIPGap", gap)
    set_time_limit_sec(model, time_limit)

    # 리더 측정, follower belief, 결합 (nz_kkt_value 와 같음)
    @variable(model, 0 <= a[1:S] <= 1); @variable(model, b[1:S] >= 0); @variable(model, 0 <= r[1:S] <= 1)
    @constraint(model, sum(a) == 1); @constraint(model, sum(r) == 1)
    @constraint(model, [s=1:S], a[s] - b[s] <= q[s]); @constraint(model, [s=1:S], a[s] + b[s] >= q[s])
    @constraint(model, sum(b) <= 2ε̂); @constraint(model, [s=1:S], (1 - β) * r[s] <= a[s])
    @variable(model, 0 <= d[1:S] <= 1); @variable(model, e[1:S] >= 0)
    @constraint(model, sum(d) == 1)
    @constraint(model, [s=1:S], d[s] - e[s] <= q[s]); @constraint(model, [s=1:S], d[s] + e[s] >= q[s])
    @constraint(model, sum(e) <= 2ε̃)
    if nz_coupled(nd)
        @variable(model, cpl[1:S] >= 0)
        @constraint(model, [s=1:S], a[s] - d[s] <= cpl[s]); @constraint(model, [s=1:S], d[s] - a[s] <= cpl[s])
        @constraint(model, sum(cpl) <= 2nz_delta(nd))
    end

    # 1. recourse 최적성 (시나리오별, 스케일 없는 dual) + follower 1단계 최적성
    @variable(model, 0 <= α[i=1:nh] <= nd.hU[i])
    @variable(model, y[1:ny, 1:S] >= 0)
    @variable(model, π[1:m, 1:S] >= 0)
    @variable(model, κ[1:nW] >= 0)
    rhs(s) = [nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0) +
              (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0) for k in 1:m]
    R = [rhs(s) for s in 1:S]
    slackP = [@expression(model, R[s][k] - sum(A[k, j] * y[j, s] for j in 1:ny if A[k, j] != 0)) for k in 1:m, s in 1:S]
    slackD = [@expression(model, sum(A[k, j] * π[k, s] for k in 1:m if A[k, j] != 0) - c[j]) for j in 1:ny, s in 1:S]
    @constraint(model, [k=1:m, s=1:S], slackP[k, s] >= 0)
    @constraint(model, [j=1:ny, s=1:S], slackD[j, s] >= 0)
    slackW = [@expression(model, nd.wvec[l] - sum(nd.W[l, i] * α[i] for i in 1:nh)) for l in 1:nW]
    @constraint(model, [l=1:nW], slackW[l] >= 0)
    # 기대 subgradient: g_i = Σ_s d_s Σ_{k: hrow=i} hcoef_k π_ks  (bilinear d × π 를 보조변수로)
    hk = findall(nd.hrow .> 0)
    @variable(model, ρ[k=hk, s=1:S] >= 0)                      # ρ = d_s π_ks
    @constraint(model, [k=hk, s=1:S], ρ[k, s] == d[s] * π[k, s])
    @variable(model, sA[1:nh] >= 0)
    @constraint(model, [i=1:nh], sA[i] == sum(nd.W[l, i] * κ[l] for l in 1:nW) - nd.c0[i]
                                         - sum(nd.hcoef[k] * ρ[k, s] for k in hk, s in 1:S if nd.hrow[k] == i))
    for k in 1:m, s in 1:S; _comp!(model, π[k, s], slackP[k, s]); end
    for j in 1:ny, s in 1:S; _comp!(model, y[j, s], slackD[j, s]); end
    for l in 1:nW; _comp!(model, κ[l], slackW[l]); end
    for i in 1:nh; _comp!(model, α[i], sA[i]); end

    # 2. pessimistic 평가: ŷˢ ∈ recourse 최적해 집합 (cᵀŷ ≥ cᵀyˢ), 목적에서 ℓᵀŷ 최대
    @variable(model, ŷ[1:ny, 1:S] >= 0)
    @constraint(model, [k=1:m, s=1:S], sum(A[k, j] * ŷ[j, s] for j in 1:ny if A[k, j] != 0) <= R[s][k])
    @constraint(model, [s=1:S], sum(c[j] * ŷ[j, s] for j in 1:ny) >= sum(c[j] * y[j, s] for j in 1:ny))
    φlo, φhi = _phi_range(nd, x̄; optimizer=optimizer)
    @variable(model, φlo[s] <= φ[s=1:S] <= φhi[s])
    @constraint(model, [s=1:S], φ[s] == sum(ℓ[j] * ŷ[j, s] for j in 1:ny if ℓ[j] != 0))
    @objective(model, Max, sum(r[s] * φ[s] for s in 1:S))
    optimize!(model)
    st = termination_status(model)
    (st == MOI.OPTIMAL || (st == MOI.TIME_LIMIT && has_values(model))) || error("KKT eval: $st")
    return Dict(:value => objective_value(model), :bound => objective_bound(model), :status => st,
                :α => value.(α), :a => value.(a), :d => value.(d), :r => value.(r), :φ => value.(φ))
end

function _phi_range(nd::NZData, x̄; optimizer)
    S, m, ny, nh = nd.S, nz_m(nd), nz_ny(nd), nz_nh(nd)
    lo = zeros(S); hi = zeros(S)
    for s in 1:S, sense in (MIN_SENSE, MAX_SENSE)
        mdl = Model(optimizer); set_silent(mdl)
        @variable(mdl, 0 <= α[i=1:nh] <= nd.hU[i]); @variable(mdl, yv[1:ny] >= 0)
        @constraint(mdl, [l=1:size(nd.W, 1)], sum(nd.W[l, i] * α[i] for i in 1:nh) <= nd.wvec[l])
        @constraint(mdl, [k=1:m], sum(nd.A[k, j] * yv[j] for j in 1:ny if nd.A[k, j] != 0) <=
            nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0) +
            (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0))
        @objective(mdl, sense, dot(nd.ell, yv))
        optimize!(mdl)
        val = objective_value(mdl)
        sense == MIN_SENSE ? (lo[s] = val - 1e-6) : (hi[s] = val + 1e-6)
    end
    return lo, hi
end


"""
θ 진단: (x, α, s) 에서 φ 를 정확히 (2단계 LP) 구하고, 필요한 최소 θ (최적성 제약 cᵀy ≥ z* 의 dual) 와
θᵁ 벌점 값 L(θᵁ) − φ (> 0 이면 θᵁ 부족 → Ω 과대평가) 를 돌려준다.
"""
function nz_theta_diag(nd::NZData, x̄, α̂, s; lp_optimizer)
    m, ny = nz_m(nd), nz_ny(nd)
    rr = nz_rhs(nd, x̄, α̂, s)
    function lp(obj; zF=nothing)
        mdl = Model(lp_optimizer); set_silent(mdl)
        @variable(mdl, y[1:ny] >= 0)
        @constraint(mdl, con[k=1:m], sum(nd.A[k, j] * y[j] for j in 1:ny if nd.A[k, j] != 0) <= rr[k])
        opt = zF === nothing ? nothing : @constraint(mdl, dot(nd.c, y) >= zF)
        @objective(mdl, Max, dot(obj, y))
        optimize!(mdl)
        return objective_value(mdl), (opt === nothing ? 0.0 : abs(shadow_price(opt))),
               [shadow_price(con[k]) for k in 1:m]
    end
    zF, _, _ = lp(nd.c)
    φ, θneed, _ = lp(nd.ell; zF=zF)
    Lθ, _, πpen = lp(nd.ell .+ nd.thetaU .* nd.c)
    Lθ -= nd.thetaU * zF
    Xr = findall(nd.xrow .> 0)
    πratio = maximum([nd.piLU[k] > 0 ? πpen[k] / nd.piLU[k] : 0.0 for k in Xr]; init=0.0)
    return Dict(:phi => φ, :theta_need => θneed, :penalty_excess => Lθ - φ, :pi_ratio => πratio, :zF => zF)
end
