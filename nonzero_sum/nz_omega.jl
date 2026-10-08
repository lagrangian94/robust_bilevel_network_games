"""
nz_omega.jl — 비제로섬 Ω (html Step 4′ 파라미터 버전, θ = θ^U 고정) 와 belief 고정 LP.

  V*(x) = sup_{(α,ζ)∈Ω} F(x, α, ζ),   Ω 는 x 와 무관, F 는 x 에 affine.

리더 블록 (CVaR_β, TV 볼 ε̂; a: TV 분포, r: CVaR 재가중, 원고 코드와 같은 형태)
  F^L = Σ_s (ℓ + θc)ᵀŷˢ − θ Σ_s [ uˢᵀϖˢ + Σ_{k∈H행} hcoef_k ζW_ks + Σ_{k∈X행} g_kˢ x_{ι(k)} ϖ_ks ]
        − Σ_{k∈X행,s} π̂ᵁ_k [ x ρ̂¹ + (1−x) ρ̂³ ]
  (Aŷˢ)_k + [k∈X행](ρ̂² − ρ̂³) ≤ u_kˢ r_s + hcoef_k ζL_{h(k),s}       (ζL = α r)
  ρ̂¹ + ρ̂² − ρ̂³ ≥ −g_kˢ r_s                                         (k ∈ X행)
  Aᵀϖˢ ≥ r_s c                                                      (N1, ζW = α ϖ)
follower 블록 (원고 그대로 + 1단계 목적 c0ᵀh)
  F^F = − Σ λᵁ π_Fᵁ_k [x ρ̃¹ + (1−x) ρ̃³] − λᵁ Σ_i [x_i ρ⁰¹ + (1−x_i) ρ⁰³]
  (Aȳˢ)_k + [X행](ρ̃² − ρ̃³) ≤ u_kˢ d_s + hcoef_k ζF_{h(k),s}       (ζF = α d)
  ρ̃¹ + ρ̃² − ρ̃³ ≥ −g_kˢ d_s
  Aᵀπ̃ˢ ≥ d_s c
  Σ_s (Bᵀπ̃ˢ)_i + c0_i ≤ (Wᵀκ)_i
  Σ_s uˢᵀπ̃ˢ + wᵀκ + Σ_i(ρ⁰² − ρ⁰³) − Σ_s cᵀȳˢ ≤ c0ᵀα
  Σ_s Σ_{k: ι(k)=i} g_kˢ π̃_ks − ρ⁰¹ − ρ⁰² + ρ⁰³ ≤ 0
θ = 0, ℓ = c 이면 원고 coro:algorithmic-TV 의 Ω 와 같다.

belief 고정 LP (html "확장 belief")
  belief (a, b, r, d, e) 와 follower 인증서 σˢ (ϖˢ = r_s σˢ) 를 고정하면 ζL, ζF, ζW 가 α 에 선형 → LP.
  cert_rows = :hrows (기본) 는 bilinear 에 들어가는 H 행의 ϖ 만 고정 (나머지는 LP 변수, 더 강한 restriction),
  :all 은 html 그대로 ϖˢ 전체를 고정.
"""

using JuMP, LinearAlgebra, Printf
import MathOptInterface as MOI

struct NZOmega
    model::Model
    v::Dict{Symbol,Any}
    belief_lp::Bool
end

function build_nz_omega(nd::NZData; optimizer, belief_lp::Bool=false, cert_rows::Symbol=:hrows,
                        silent::Bool=true, mode::Symbol=(belief_lp ? :belief_lp : :global),
                        solver_params::Bool=true)
    mode in (:global, :belief_lp, :fixed_alpha, :relax) || error("mode = :global | :belief_lp | :fixed_alpha | :relax")
    A, c, ℓ, u, g = nd.A, nd.c, nd.ell, nd.u, nd.g
    m, ny, nh, nx, S = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.nx, nd.S
    nW = size(nd.W, 1)
    q, ε̂, ε̃, β = nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta
    θ = nd.thetaU
    Xr = findall(nd.xrow .> 0); Hr = findall(nd.hrow .> 0)
    nXr, nHr = length(Xr), length(Hr)

    model = Model(optimizer)
    silent && set_silent(model)
    # F 는 θ(cᵀŷ − rᵀϖ) 처럼 큰 두 항의 상쇄를 포함 → 허용오차를 조임 (Gurobi 전용, Ipopt 등에서는 solver_params=false)
    if solver_params
        for (k, val) in (("FeasibilityTol", 1e-9), ("OptimalityTol", 1e-9), ("NumericFocus", 2))
            try set_optimizer_attribute(model, k, val) catch end
        end
    end

    amin = [max(0.0, q[s] - 2ε̂) for s in 1:S]; amax = [min(1.0, q[s] + 2ε̂) for s in 1:S]
    dmin = [max(0.0, q[s] - 2ε̃) for s in 1:S]; dmax = [min(1.0, q[s] + 2ε̃) for s in 1:S]
    rmax = amax ./ (1 - β)

    @variable(model, 0 <= α[i=1:nh] <= nd.hU[i])
    @constraint(model, Hcon[l=1:nW], sum(nd.W[l, i] * α[i] for i in 1:nh) <= nd.wvec[l])

    # ---------------- 리더 블록 ----------------
    @variable(model, amin[s] <= a[s=1:S] <= amax[s])
    @variable(model, 0 <= b[1:S] <= 2ε̂)
    @variable(model, 0 <= r[s=1:S] <= rmax[s])
    @constraint(model, sum(a) == 1)
    @constraint(model, [s=1:S], a[s] - b[s] <= q[s])
    @constraint(model, [s=1:S], a[s] + b[s] >= q[s])
    @constraint(model, sum(b) <= 2ε̂)
    @constraint(model, [s=1:S], (1 - β) * r[s] <= a[s])
    @constraint(model, sum(r) == 1)

    @variable(model, ŷ[1:ny, 1:S] >= 0)
    @variable(model, ϖ[1:m, 1:S] >= 0)
    for (jj, k) in enumerate(Hr), s in 1:S
        isfinite(nd.varpiU[k]) && set_upper_bound(ϖ[k, s], nd.varpiU[k] * rmax[s])
    end
    @variable(model, ρ̂1[1:nXr, 1:S] >= 0)
    @variable(model, ρ̂2[1:nXr, 1:S] >= 0)
    @variable(model, ρ̂3[1:nXr, 1:S] >= 0)
    @variable(model, 0 <= ζL[i=1:nh, s=1:S] <= nd.hU[i] * rmax[s])
    @variable(model, 0 <= ζW[jj=1:nHr, s=1:S] <= nd.hU[nd.hrow[Hr[jj]]] * nd.varpiU[Hr[jj]] * rmax[s])
    xpos = zeros(Int, m); for (jj, k) in enumerate(Xr); xpos[k] = jj; end
    hpos = zeros(Int, m); for (jj, k) in enumerate(Hr); hpos[k] = jj; end

    @constraint(model, Lflow[k=1:m, s=1:S],
        sum(A[k, j] * ŷ[j, s] for j in 1:ny if A[k, j] != 0)
        + (xpos[k] > 0 ? ρ̂2[xpos[k], s] - ρ̂3[xpos[k], s] : 0.0)
        <= u[k, s] * r[s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * ζL[nd.hrow[k], s] : 0.0))
    @constraint(model, LMC[jj=1:nXr, s=1:S], ρ̂1[jj, s] + ρ̂2[jj, s] - ρ̂3[jj, s] >= -g[Xr[jj], s] * r[s])
    @constraint(model, N1[j=1:ny, s=1:S], sum(A[k, j] * ϖ[k, s] for k in 1:m if A[k, j] != 0) >= c[j] * r[s])
    # ϖ_ks ≤ ϖᵁ_k r_s (H 행): ϖˢ/r_s 는 follower 최적 dual 이고 그중 H 행 성분이 ϖᵁ 이하인 것이 존재 (nz_data.jl)
    # → 최적값을 바꾸지 않는 강화. α-B&B 의 α·ϖ RLT 가 이 행을 쓴다.
    @constraint(model, Wcap[jj=1:nHr, s=1:S; isfinite(nd.varpiU[Hr[jj]])], ϖ[Hr[jj], s] <= nd.varpiU[Hr[jj]] * r[s])

    # ---------------- follower 블록 ----------------
    @variable(model, dmin[s] <= d[s=1:S] <= dmax[s])
    @variable(model, 0 <= e[1:S] <= 2ε̃)
    @constraint(model, sum(d) == 1)
    @constraint(model, [s=1:S], d[s] - e[s] <= q[s])
    @constraint(model, [s=1:S], d[s] + e[s] >= q[s])
    @constraint(model, sum(e) <= 2ε̃)
    # ---------------- belief 결합 (종속 ambiguity set) ----------------
    # 𝔅_δ = 𝔅 ∩ {|a_s − d_s| ≤ cpl_s, Σ cpl_s ≤ 2δ}  (a = 리더 TV 분포 p, d = follower belief p̃)
    # belief_lp 는 belief 를 고정하므로 넣지 않는다 (고정 belief 가 결합을 만족하도록 nz_clean_belief 가 보장).
    δc = nz_delta(nd)
    cpl = nothing
    if isfinite(δc) && mode != :belief_lp
        @variable(model, 0 <= cpl[s=1:S] <= min(2δc, max(amax[s], dmax[s])))
        @constraint(model, cpl_ad[s=1:S], a[s] - d[s] <= cpl[s])
        @constraint(model, cpl_da[s=1:S], d[s] - a[s] <= cpl[s])
        @constraint(model, cpl_sum, sum(cpl) <= 2δc)
    end
    @variable(model, ȳ[1:ny, 1:S] >= 0)
    @variable(model, π̃[1:m, 1:S] >= 0)
    @variable(model, κ[1:nW] >= 0)
    @variable(model, ρ̃1[1:nXr, 1:S] >= 0)
    @variable(model, ρ̃2[1:nXr, 1:S] >= 0)
    @variable(model, ρ̃3[1:nXr, 1:S] >= 0)
    @variable(model, ρ01[1:nx] >= 0)
    @variable(model, ρ02[1:nx] >= 0)
    @variable(model, ρ03[1:nx] >= 0)
    @variable(model, 0 <= ζF[i=1:nh, s=1:S] <= nd.hU[i] * dmax[s])

    @constraint(model, Fflow[k=1:m, s=1:S],
        sum(A[k, j] * ȳ[j, s] for j in 1:ny if A[k, j] != 0)
        + (xpos[k] > 0 ? ρ̃2[xpos[k], s] - ρ̃3[xpos[k], s] : 0.0)
        <= u[k, s] * d[s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * ζF[nd.hrow[k], s] : 0.0))
    @constraint(model, FMC[jj=1:nXr, s=1:S], ρ̃1[jj, s] + ρ̃2[jj, s] - ρ̃3[jj, s] >= -g[Xr[jj], s] * d[s])
    @constraint(model, Ffeas[j=1:ny, s=1:S], sum(A[k, j] * π̃[k, s] for k in 1:m if A[k, j] != 0) >= c[j] * d[s])
    hrows_of = [[k for k in Hr if nd.hrow[k] == i] for i in 1:nh]
    xrows_of = [[k for k in Xr if nd.xrow[k] == i] for i in 1:nx]
    @constraint(model, Fh[i=1:nh],
        sum(nd.hcoef[k] * π̃[k, s] for s in 1:S, k in hrows_of[i]) + nd.c0[i]
        <= sum(nd.W[l, i] * κ[l] for l in 1:nW))
    @constraint(model, Flam,
        sum(u[k, s] * π̃[k, s] for k in 1:m, s in 1:S if u[k, s] != 0) + sum(nd.wvec[l] * κ[l] for l in 1:nW)
        + sum(ρ02[i] - ρ03[i] for i in 1:nx) - sum(c[j] * ȳ[j, s] for j in 1:ny, s in 1:S if c[j] != 0)
        <= sum(nd.c0[i] * α[i] for i in 1:nh))
    @constraint(model, Fpsi[i=1:nx],
        sum(g[k, s] * π̃[k, s] for s in 1:S, k in xrows_of[i])
        - ρ01[i] - ρ02[i] + ρ03[i] <= 0)

    # ---------------- bilinear 정의 ----------------
    v = Dict{Symbol,Any}(:α => α, :a => a, :b => b, :r => r, :ŷ => ŷ, :ϖ => ϖ,
        :ρ̂1 => ρ̂1, :ρ̂3 => ρ̂3, :ζL => ζL, :ζW => ζW,
        :d => d, :e => e, :ȳ => ȳ, :π̃ => π̃, :κ => κ, :ρ̃1 => ρ̃1, :ρ̃3 => ρ̃3,
        :ρ01 => ρ01, :ρ03 => ρ03, :ζF => ζF, :Xr => Xr, :Hr => Hr, :cert_rows => cert_rows, :mode => mode,
        :cpl => cpl)
    if mode == :belief_lp
        # belief (r, d) 와 인증서 ϖ 고정 → α 의 계수. 계수는 nz_set_belief! 에서 채움
        v[:cL] = @constraint(model, [i=1:nh, s=1:S], ζL[i, s] - 0.0 * α[i] == 0)
        v[:cF] = @constraint(model, [i=1:nh, s=1:S], ζF[i, s] - 0.0 * α[i] == 0)
        v[:cW] = @constraint(model, [jj=1:nHr, s=1:S], ζW[jj, s] - 0.0 * α[nd.hrow[Hr[jj]]] == 0)
    elseif mode == :fixed_alpha
        # α 고정 → (r, d, ϖ) 의 계수. 계수는 nz_fix_alpha! 에서 채움 (α-B&B 의 정확한 평가)
        v[:cL] = @constraint(model, [i=1:nh, s=1:S], ζL[i, s] - 0.0 * r[s] == 0)
        v[:cF] = @constraint(model, [i=1:nh, s=1:S], ζF[i, s] - 0.0 * d[s] == 0)
        v[:cW] = @constraint(model, [jj=1:nHr, s=1:S], ζW[jj, s] - 0.0 * ϖ[Hr[jj], s] == 0)
    elseif mode == :global
        @constraint(model, [i=1:nh, s=1:S], ζL[i, s] == α[i] * r[s])
        @constraint(model, [i=1:nh, s=1:S], ζF[i, s] == α[i] * d[s])
        @constraint(model, [jj=1:nHr, s=1:S], ζW[jj, s] == α[nd.hrow[Hr[jj]]] * ϖ[Hr[jj], s])
        solver_params && set_optimizer_attribute(model, "NonConvex", 2)
    end
    # mode == :relax: 곱 정의 없음 (α-B&B 노드 완화가 RLT 행을 붙임)
    O = NZOmega(model, v, mode != :global)
    nz_set_objective!(O, nd, zeros(nx))
    return O
end


"F(x̄, ·) 를 목적함수로 설정 (Ω 는 x̄ 와 무관)"
function nz_set_objective!(O::NZOmega, nd::NZData, x̄)
    v = O.v
    S, ny, m = nd.S, nz_ny(nd), nz_m(nd)
    Xr, Hr = v[:Xr], v[:Hr]
    θ, λU = nd.thetaU, nd.lambdaU
    ŷ, ϖ, ζW = v[:ŷ], v[:ϖ], v[:ζW]
    FL = AffExpr(0.0)
    for j in 1:ny, s in 1:S
        cj = nd.ell[j] + θ * nd.c[j]
        cj != 0 && add_to_expression!(FL, cj, ŷ[j, s])
    end
    if θ != 0
        for k in 1:m, s in 1:S
            nd.u[k, s] != 0 && add_to_expression!(FL, -θ * nd.u[k, s], ϖ[k, s])
        end
        for (jj, k) in enumerate(Hr), s in 1:S
            add_to_expression!(FL, -θ * nd.hcoef[k], ζW[jj, s])
        end
        for k in Xr, s in 1:S
            add_to_expression!(FL, -θ * nd.g[k, s] * x̄[nd.xrow[k]], ϖ[k, s])
        end
    end
    for (jj, k) in enumerate(Xr), s in 1:S
        xi = x̄[nd.xrow[k]]
        add_to_expression!(FL, -nd.piLU[k] * xi, v[:ρ̂1][jj, s])
        add_to_expression!(FL, -nd.piLU[k] * (1 - xi), v[:ρ̂3][jj, s])
        add_to_expression!(FL, -λU * nd.piFU[k] * xi, v[:ρ̃1][jj, s])
        add_to_expression!(FL, -λU * nd.piFU[k] * (1 - xi), v[:ρ̃3][jj, s])
    end
    for i in 1:nd.nx
        add_to_expression!(FL, -λU * x̄[i], v[:ρ01][i])
        add_to_expression!(FL, -λU * (1 - x̄[i]), v[:ρ03][i])
    end
    @objective(O.model, Max, FL)
    return nothing
end


"현재 해에서 cut 계수와 F 값 (x 에 affine: F(x) = intercept + slopeᵀx)"
function nz_cut_from_solution(O::NZOmega, nd::NZData, x̄)
    v = O.v
    S = nd.S; Xr = v[:Xr]
    θ, λU = nd.thetaU, nd.lambdaU
    slope = zeros(nd.nx)
    for (jj, k) in enumerate(Xr), s in 1:S
        i = nd.xrow[k]
        slope[i] += -θ * nd.g[k, s] * value(v[:ϖ][k, s]) -
                    nd.piLU[k] * (value(v[:ρ̂1][jj, s]) - value(v[:ρ̂3][jj, s])) -
                    λU * nd.piFU[k] * (value(v[:ρ̃1][jj, s]) - value(v[:ρ̃3][jj, s]))
    end
    for i in 1:nd.nx
        slope[i] += -λU * (value(v[:ρ01][i]) - value(v[:ρ03][i]))
    end
    Fval = objective_value(O.model)
    return Dict(:Fval => Fval, :slope => slope, :intercept => Fval - dot(slope, x̄))
end


"belief + 인증서 읽기 (Ω 해에서)"
function nz_read_belief(O::NZOmega, nd::NZData)
    v = O.v; S, m = nd.S, nz_m(nd)
    b = Dict{Symbol,Any}(f => [value(v[f][s]) for s in 1:S] for f in (:a, :b, :r, :d, :e))
    b[:ϖ] = [value(v[:ϖ][k, s]) for k in 1:m, s in 1:S]
    b[:α] = [max(value(v[:α][i]), 0.0) for i in 1:nz_nh(nd)]
    return b
end


"""
follower 인증서 σˢ (html (b)): (x̄, α̂) 에서 시나리오별 follower recourse LP 를 simplex 로 풀어
최적 dual 꼭짓점을 얻는다. shadow price (rhs 1 증가 시 목적 증가) ≥ 0.
"""
function nz_follower_duals(nd::NZData, x̄, α̂; lp_optimizer)
    S, m, ny = nd.S, nz_m(nd), nz_ny(nd)
    σ = zeros(m, S); z = zeros(S)
    for s in 1:S
        mdl = Model(lp_optimizer); set_silent(mdl)
        @variable(mdl, y[1:ny] >= 0)
        rr = nz_rhs(nd, x̄, α̂, s)
        @constraint(mdl, con[k=1:m], sum(nd.A[k, j] * y[j] for j in 1:ny if nd.A[k, j] != 0) <= rr[k])
        @objective(mdl, Max, dot(nd.c, y))
        optimize!(mdl)
        termination_status(mdl) == MOI.OPTIMAL || error("follower LP s=$s: $(termination_status(mdl))")
        σ[:, s] = max.([shadow_price(con[k]) for k in 1:m], 0.0)
        z[s] = objective_value(mdl)
    end
    return nz_repair_cert(nd, σ; lp_optimizer=lp_optimizer), z
end


"""
인증서를 정확히 dual feasible (Aᵀσ ≥ c, σ ≥ 0) 한 점으로 보정: min ‖σ′ − σ‖₁.
solver dual 허용오차 (~1e-7) 나 ϖ̂/r̂ 의 잔차가 남으면 ϖ 고정 시 N1 이 깨져 belief LP 가 infeasible 이 된다.
보정 후 c 를 정확히 넘도록 1e-9 여유를 둔다 (cut 은 restriction 이므로 여유는 유효성에 영향 없음).
"""
function nz_repair_cert(nd::NZData, σ; lp_optimizer)
    m, ny, S = nz_m(nd), nz_ny(nd), size(σ, 2)
    out = similar(σ)
    for s in 1:S
        mdl = Model(lp_optimizer); set_silent(mdl)
        @variable(mdl, σp[1:m] >= 0); @variable(mdl, dev[1:m] >= 0)
        for k in 1:m                                   # Ω 의 Wcap (ϖ ≤ ϖᵁ r) 과 맞춤
            nd.hrow[k] > 0 && isfinite(nd.varpiU[k]) && set_upper_bound(σp[k], nd.varpiU[k])
        end
        @constraint(mdl, [k=1:m], dev[k] >= σp[k] - σ[k, s]); @constraint(mdl, [k=1:m], dev[k] >= σ[k, s] - σp[k])
        @constraint(mdl, [j=1:ny], sum(nd.A[k, j] * σp[k] for k in 1:m if nd.A[k, j] != 0) >= nd.c[j])
        @objective(mdl, Min, sum(dev))
        optimize!(mdl)
        termination_status(mdl) == MOI.OPTIMAL || error("cert repair s=$s: $(termination_status(mdl))")
        out[:, s] = max.(value.(σp), 0.0)
        # LP 해도 solver 허용오차 (~1e-8) 만큼 어긋날 수 있음 → 계수가 모두 ≥ 0 인 행의 dual 을 올려 정확히 맞춤
        # (그런 행을 올리면 다른 열의 (Aᵀσ)_j 는 줄지 않으므로 새 위반이 생기지 않음)
        nonneg = [all(nd.A[k, :] .>= 0) for k in 1:m]
        for j in 1:ny
            slack = dot(nd.A[:, j], out[:, s]) - nd.c[j]
            slack >= 0 && continue
            cand = [k for k in 1:m if nonneg[k] && nd.A[k, j] > 0 && !(nd.hrow[k] > 0 && isfinite(nd.varpiU[k]))]
            isempty(cand) && error("cert repair: 열 $j 를 보정할 비음 행이 없음 (상한 없는 행 중)")
            k = cand[argmax(nd.A[cand, j])]
            out[k, s] += (-slack) / nd.A[k, j] + 1e-12
        end
    end
    return out
end


"""
TV 볼·CVaR 다면체 안으로 정리 (solver 허용오차 제거). belief_menu_benders.jl 의 _clean_belief 와 같은 규칙.
ϖ 는 그대로 둔다 (인증서는 r 과 별도로 σ 로 저장).
결합 (δ 유한): (a, d) 를 함께 q̂ 쪽으로 같은 비율 t 만큼 줄인다. ‖a−q̂‖, ‖d−q̂‖, ‖a−d‖ 가 모두 (1−t) 배가
되므로 세 제약 (ε̂, ε̃, δ) 을 동시에 만족시키는 가장 작은 축소다 ((q̂, q̂) 는 모든 δ ≥ 0 에서 허용).
"""
function nz_clean_belief(bel, nd::NZData)
    q = nd.q_hat
    simp(y) = (y = max.(y, 0.0); y ./ sum(y))
    a = simp(bel[:a]); d = simp(bel[:d])
    ratio(dev, ε) = dev > 2ε ? 2ε / dev : 1.0
    ta = ratio(sum(abs.(a .- q)), nd.eps_hat); td = ratio(sum(abs.(d .- q)), nd.eps_tilde)
    if nz_coupled(nd)
        t = min(ta, td, ratio(sum(abs.(a .- d)), nz_delta(nd)))
        ta = td = t
    end
    out = Dict{Symbol,Any}()
    out[:a] = q .+ (a .- q) .* ta; out[:b] = abs.(out[:a] .- q)
    out[:d] = q .+ (d .- q) .* td; out[:e] = abs.(out[:d] .- q)
    cap = out[:a] ./ (1 - nd.beta)
    r = clamp.(bel[:r], 0.0, cap); sr = sum(r)
    if sr > 1
        r ./= sr
    elseif sr < 1
        slack = cap .- r; r .+= slack .* ((1 - sr) / sum(slack))
    end
    out[:r] = r
    return out
end


"belief LP 에 확장 belief (a,b,r,d,e,σ) 고정. ϖˢ = r_s σˢ"
function nz_set_belief!(O::NZOmega, nd::NZData, bel, σ)
    @assert O.belief_lp
    v = O.v; S, nh, m = nd.S, nz_nh(nd), nz_m(nd)
    α = v[:α]; Hr = v[:Hr]
    for f in (:a, :b, :r, :d, :e), s in 1:S
        fix(v[f][s], bel[f][s]; force=true)
    end
    for i in 1:nh, s in 1:S
        set_normalized_coefficient(v[:cL][i, s], α[i], -bel[:r][s])
        set_normalized_coefficient(v[:cF][i, s], α[i], -bel[:d][s])
    end
    ϖfix = σ .* reshape(bel[:r], 1, S)
    for (jj, k) in enumerate(Hr), s in 1:S
        set_normalized_coefficient(v[:cW][jj, s], α[nd.hrow[k]], -ϖfix[k, s])
    end
    rows = v[:cert_rows] == :all ? (1:m) : Hr
    for k in 1:m, s in 1:S
        if k in rows
            # 상한을 지우고 고정 (fix 는 bound 가 있으면 force 필요)
            fix(v[:ϖ][k, s], ϖfix[k, s]; force=true)
        elseif is_fixed(v[:ϖ][k, s])
            unfix(v[:ϖ][k, s]); set_lower_bound(v[:ϖ][k, s], 0.0)
        end
    end
    return nothing
end


"Ω (또는 belief LP) 를 x̄ 에서 풀기"
function nz_solve!(O::NZOmega, nd::NZData, x̄; time_limit=nothing, gap=1e-5)
    nz_set_objective!(O, nd, x̄)
    time_limit === nothing ? set_time_limit_sec(O.model, nothing) : set_time_limit_sec(O.model, time_limit)
    O.belief_lp || set_optimizer_attribute(O.model, "MIPGap", gap)
    optimize!(O.model)
    st = termination_status(O.model)
    ok = st == MOI.OPTIMAL || (st == MOI.TIME_LIMIT && has_values(O.model))
    ok || error("nz Ω: $st")
    bd = st == MOI.OPTIMAL ? objective_value(O.model) : objective_bound(O.model)
    return Dict(:status => st, :Fval => objective_value(O.model), :bound => bd)
end
