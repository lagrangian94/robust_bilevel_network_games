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

# best_obj_stop / best_bd_stop: Gurobi BestObjStop / BestBdStop (목표값 조기 종료, BD-KKT 벤치마크가 Algorithm 2 의 목표값과 같게 쓴다).
#   incumbent ≥ best_obj_stop 이거나 상한 ≤ best_bd_stop 이면 멈추고 :status = OBJECTIVE_LIMIT. 기본 nothing = 끝까지 증명.
function nz_kkt_value(nd::NZData, x̄; optimizer, time_limit=600.0, gap=1e-6, silent=true,
                      best_obj_stop=nothing, best_bd_stop=nothing, allow_no_incumbent::Bool=false,
                      start_beliefs=nothing, lp_optimizer=optimizer, start_report::Bool=false)
    A, c, ℓ = nd.A, nd.c, nd.ell
    m, ny, nh, S = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.S
    nW = size(nd.W, 1)
    q, ε̂, ε̃, β = nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta
    model = Model(optimizer)
    silent && set_silent(model)
    set_optimizer_attribute(model, "NonConvex", 2)
    set_optimizer_attribute(model, "MIPGap", gap)
    set_time_limit_sec(model, time_limit)
    best_obj_stop === nothing || set_optimizer_attribute(model, "BestObjStop", best_obj_stop)
    best_bd_stop === nothing || set_optimizer_attribute(model, "BestBdStop", best_bd_stop)

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
    comps = Tuple{VariableRef,Any,VariableRef}[]     # (변수, slack 식, 그 상보성 이진) — 시작해용
    for k in 1:m, s in 1:S; push!(comps, (π[k, s], slackP[k, s], _comp!(model, π[k, s], slackP[k, s]))); end
    for j in 1:ny, s in 1:S; push!(comps, (y[j, s], slackD[j, s], _comp!(model, y[j, s], slackD[j, s]))); end
    for l in 1:nW; push!(comps, (κ[l], slackW[l], _comp!(model, κ[l], slackW[l]))); end
    for i in 1:nh; push!(comps, (α[i], slackA[i], _comp!(model, α[i], slackA[i]))); end

    # 리더 평가 copy: ŷˢ ∈ argmax{cᵀy : Ay ≤ r_s(x̄, α)}, 그 위에서 ℓᵀŷ 최대 (pessimistic)
    @variable(model, ŷ[1:ny, 1:S] >= 0)
    @variable(model, σ[1:m, 1:S] >= 0)
    sP = [@expression(model, R[s][k] - sum(A[k, j] * ŷ[j, s] for j in 1:ny if A[k, j] != 0)) for k in 1:m, s in 1:S]
    sD = [@expression(model, sum(A[k, j] * σ[k, s] for k in 1:m if A[k, j] != 0) - c[j]) for j in 1:ny, s in 1:S]
    @constraint(model, [k=1:m, s=1:S], sP[k, s] >= 0)
    @constraint(model, [j=1:ny, s=1:S], sD[j, s] >= 0)
    for k in 1:m, s in 1:S; push!(comps, (σ[k, s], sP[k, s], _comp!(model, σ[k, s], sP[k, s]))); end
    for j in 1:ny, s in 1:S; push!(comps, (ŷ[j, s], sD[j, s], _comp!(model, ŷ[j, s], sD[j, s]))); end

    # φ_s 범위 (spatial B&B 용): α 를 풀어 둔 LP 로 상하한
    φlo, φhi = _phi_range(nd, x̄; optimizer=optimizer)
    @variable(model, φlo[s] <= φ[s=1:S] <= φhi[s])
    @constraint(model, [s=1:S], φ[s] == sum(ℓ[j] * ŷ[j, s] for j in 1:ny if ℓ[j] != 0))
    @objective(model, Max, sum(r[s] * φ[s] for s in 1:S))
    # 시작해 (BD-KKT 벤치마크): 후보 belief 마다 LP 로 KKT 를 만족하는 점을 만들고 리더 값이 가장 큰 것을 넣는다
    start_val = -Inf
    if start_beliefs !== nothing
        best = nothing
        for d̂ in start_beliefs
            stv = _nz_kkt_start(nd, x̄, d̂; optimizer=lp_optimizer)
            stv === nothing && continue
            (best === nothing || stv.val > best.val) && (best = stv)
        end
        if best !== nothing
            start_val = best.val
            for (vv, vals) in ((a, best.a), (b, best.b), (r, best.r), (d, best.d), (e, best.e), (α, best.α), (κ, best.κ),
                               (y, best.y), (π, best.π), (ŷ, best.ŷ), (σ, best.σ), (φ, best.φ))
                set_start_value.(vv, vals)
            end
            nz_coupled(nd) && set_start_value.(cpl, abs.(best.a .- best.d))
            # 상보성 쌍마다 0 에 더 가까운 쪽을 0 으로: 변수 ≤ slack 이면 z = 1 (변수 = 0, 시작값도 0 으로 정리), 아니면 z = 0 (slack = 0).
            # (LP 해의 10⁻⁷ 수준 잔여 양수를 그대로 두면 z = 0 이 slack = 0 을 강제해 Gurobi 가 시작해를 버린다, 2026-10-22)
            for (vv, sl, z) in comps
                sv = value(start_value, sl)
                if start_value(vv) <= max(sv, 0.0)
                    set_start_value(z, 1.0); set_start_value(vv, 0.0)
                else
                    set_start_value(z, 0.0)
                end
            end
            if start_report                            # 진단: 시작해가 어기는 제약 (상위 10 개). indicator 는 comps 로 직접 확인
                pv(v) = something(start_value(v), 0.0)
                viols = Tuple{Float64,String}[]
                for (F, Sset) in list_of_constraint_types(model)
                    Sset <: MOI.Indicator && continue
                    for con in all_constraints(model, F, Sset)
                        co = constraint_object(con)
                        val = co.func isa VariableRef ? pv(co.func) : value(pv, co.func)
                        dd = MOI.Utilities.distance_to_set(val, co.set)
                        dd > 1e-6 && push!(viols, (dd, string(con)[1:min(end, 120)]))
                    end
                end
                for (vv, sl, z) in comps
                    if start_value(z) == 1.0
                        abs(pv(vv)) > 1e-6 && push!(viols, (abs(pv(vv)), "ind z=1 var≠0: " * string(vv)))
                    else
                        sv = value(pv, sl); abs(sv) > 1e-6 && push!(viols, (abs(sv), "ind z=0 slack≠0: " * string(vv)))
                    end
                end
                println("  [start_report] 위반 ", length(viols), " 개 (시작값 ", best.val, ")")
                for (dd, nm) in first(sort(viols; by=first, rev=true), 10); println("    ", dd, "  ", nm); end
            end
        end
    end
    optimize!(model)
    st = termination_status(model)
    # incumbent 없이 멈춤: 상한 ≤ best_bd_stop (OBJECTIVE_LIMIT), 또는 allow_no_incumbent 일 때 시간 제한 (벤치마크가 미해결로 기록)
    if !has_values(model) && (st == MOI.OBJECTIVE_LIMIT || (allow_no_incumbent && st == MOI.TIME_LIMIT))
        return Dict(:value => -Inf, :bound => objective_bound(model), :status => st, :α => nothing, :start_val => start_val)
    end
    (st == MOI.OPTIMAL || ((st == MOI.TIME_LIMIT || st == MOI.OBJECTIVE_LIMIT) && has_values(model))) || error("KKT eval: $st")
    return Dict(:value => objective_value(model), :bound => objective_bound(model), :status => st,
                :α => value.(α), :a => value.(a), :d => value.(d), :r => value.(r), :φ => value.(φ), :start_val => start_val)
end

"""
_nz_kkt_start(nd, x̄, d̂; optimizer) — nz_kkt_value 의 시작해 (BD-KKT 벤치마크, 2026-10-22). belief d̂ 에서
  1. follower 2 단계 LP (max c0ᵀα + Σ_s d̂_s cᵀyˢ) 를 simplex 로 → (α, y) 와 행 쌍대 (π, κ) (= d_s 스케일 dual, 상보성 성립)
  2. 시나리오마다 recourse LP max cᵀŷ (쌍대 σ) → 그 최적면에서 max ℓᵀŷ (pessimistic) → φ_s
  3. 리더 LP: max Σ r_s φ_s over (a, b, r) (+ 결합 |a − d̂| ≤ cpl)
d̂ 가 follower 볼 밖이거나 LP 가 실패하면 nothing. 반환 NamedTuple (val = 리더 값, 각 변수 값).
"""
function _nz_kkt_start(nd::NZData, x̄, d̂; optimizer)
    A, c, ℓ = nd.A, nd.c, nd.ell
    m, ny, nh, S = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.S
    nW = size(nd.W, 1)
    q, ε̂, ε̃, β = nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta
    d̂ = max.(d̂, 0.0); d̂ = d̂ ./ sum(d̂)
    ê = abs.(d̂ .- q)
    sum(ê) <= 2ε̃ + 1e-9 || return nothing
    # 1. follower 2 단계 LP
    F = Model(optimizer); set_silent(F)
    # α ≤ hU 는 변수 상한으로 두지 않는다: KKT 모형의 정류 조건 (slackA) 에는 W 행의 쌍대 κ 만 있으므로, 같은 한도가 변수 상한으로도
    # 있으면 simplex 가 쌍대값을 변수 상한 쪽에 줘서 slackA < 0 이 될 수 있다 (거점별 한도는 hU = W 행). 풀고 나서 hU 를 확인한다.
    @variable(F, αv[i=1:nh] >= 0); @variable(F, yv[1:ny, 1:S] >= 0)
    @constraint(F, cw[l=1:nW], sum(nd.W[l, i] * αv[i] for i in 1:nh) <= nd.wvec[l])
    @constraint(F, ca[k=1:m, s=1:S], sum(A[k, j] * yv[j, s] for j in 1:ny if A[k, j] != 0) <=
        nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * αv[nd.hrow[k]] : 0.0) + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0))
    @objective(F, Max, dot(nd.c0, αv) + sum(d̂[s] * dot(c, yv[:, s]) for s in 1:S))
    optimize!(F)
    termination_status(F) == MOI.OPTIMAL || return nothing
    α0 = max.(value.(αv), 0.0); y0 = max.(value.(yv), 0.0)
    all(α0 .<= nd.hU .+ 1e-9) || return nothing
    π0 = [max(shadow_price(ca[k, s]), 0.0) for k in 1:m, s in 1:S]; κ0 = [max(shadow_price(cw[l]), 0.0) for l in 1:nW]
    # 2. 리더 평가 copy (pessimistic recourse)
    ŷ0 = zeros(ny, S); σ0 = zeros(m, S); φ0 = zeros(S)
    for s in 1:S
        rr = nz_rhs(nd, x̄, α0, s)
        G = Model(optimizer); set_silent(G)
        @variable(G, yy[1:ny] >= 0)
        @constraint(G, cg[k=1:m], sum(A[k, j] * yy[j] for j in 1:ny if A[k, j] != 0) <= rr[k])
        @objective(G, Max, dot(c, yy)); optimize!(G)
        termination_status(G) == MOI.OPTIMAL || return nothing
        v = objective_value(G); σ0[:, s] = [max(shadow_price(cg[k]), 0.0) for k in 1:m]
        # 최적면을 σ 와의 상보성으로 정확히: reduced cost > 0 인 열은 0, σ > 0 인 행은 등호 (cᵀy ≥ v − tol 로 두면
        # 10⁻⁷ 수준 잔여가 생겨 indicator 상보성이 깨지고 Gurobi 가 시작해를 버린다, 2026-10-22)
        rc = A' * σ0[:, s] .- c
        for j in 1:ny; rc[j] > 1e-9 && fix(yy[j], 0.0; force=true); end
        for k in 1:m; σ0[k, s] > 1e-9 && set_normalized_rhs(cg[k], rr[k]); end
        @constraint(G, [k=1:m; σ0[k, s] > 1e-9], sum(A[k, j] * yy[j] for j in 1:ny if A[k, j] != 0) >= rr[k])
        @objective(G, Max, dot(ℓ, yy)); optimize!(G)
        termination_status(G) == MOI.OPTIMAL || return nothing
        ŷ0[:, s] = max.(value.(yy), 0.0); φ0[s] = dot(ℓ, ŷ0[:, s])
    end
    # 3. 리더 LP
    Lm = Model(optimizer); set_silent(Lm)
    @variable(Lm, 0 <= av[1:S] <= 1); @variable(Lm, bv[1:S] >= 0); @variable(Lm, 0 <= rv[1:S] <= 1)
    @constraint(Lm, sum(av) == 1); @constraint(Lm, sum(rv) == 1)
    @constraint(Lm, [s=1:S], av[s] - bv[s] <= q[s]); @constraint(Lm, [s=1:S], av[s] + bv[s] >= q[s])
    @constraint(Lm, sum(bv) <= 2ε̂); @constraint(Lm, [s=1:S], (1 - β) * rv[s] <= av[s])
    if nz_coupled(nd)
        @variable(Lm, cv[1:S] >= 0)
        @constraint(Lm, [s=1:S], av[s] - d̂[s] <= cv[s]); @constraint(Lm, [s=1:S], d̂[s] - av[s] <= cv[s])
        @constraint(Lm, sum(cv) <= 2nz_delta(nd))
    end
    @objective(Lm, Max, dot(rv, φ0)); optimize!(Lm)
    termination_status(Lm) == MOI.OPTIMAL || return nothing
    return (val=objective_value(Lm), a=value.(av), b=value.(bv), r=value.(rv), d=d̂, e=ê, α=α0, κ=κ0,
            y=y0, π=π0, ŷ=ŷ0, σ=σ0, φ=φ0)
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
    (st == MOI.OPTIMAL || ((st == MOI.TIME_LIMIT || st == MOI.OBJECTIVE_LIMIT) && has_values(model))) || error("KKT eval: $st")
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
