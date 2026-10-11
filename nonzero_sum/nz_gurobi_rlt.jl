"""
nz_gurobi_rlt.jl — Gurobi 전역 (Ω, NonConvex) 에 RLT 행을 정적으로 넣은 모형. α-B&B 의 대안 후보 ("Gurobi + RLT").

Gurobi 는 노드에서 곱 변수의 McCormick 만 갱신하고 사용자 행은 고칠 수 없으므로, 기본 상자 (l = 0, u = hU) 의
bound-factor 곱만 넣는다:  α_i · g(y) ≥ 0,  (hU_i − α_i) · g(y) ≥ 0   (g: belief 다면체·Wcap 의 선형 행).

level
  0 : 그대로 (build_nz_omega mode = :global)
  1 : 새 곱 변수 없이 쓸 수 있는 행만 — Σ_s ζL_is = α_i, Σ_s ζF_is = α_i, (w − Wα) r_s ≥ 0, (w − Wα) d_s ≥ 0,
      Wcap × α (ζW 와 ζL 을 묶음)
  2 : 1 + 곱 변수 Za = α a, Zb = α b, Ze = α e (결합이면 Zc = α cpl) 를 bilinear 등식으로 정의하고
      TV 볼·CVaR·결합 행 × α_i, × (hU_i − α_i)  (α-B&B 의 루트 RLT 와 같은 행 집합)
  3 : 2 + 여유 변수 곱 — 비상자 행 g(y) ≥ 0 마다 여유 t = g(y) 와 곱 T_i = α_i t 를 bilinear 로 정의.
      Gurobi 가 노드 상자에서 (α_i, t) 의 McCormick 을 갱신하므로 노드 상자의 bound-factor RLT 를 Gurobi 가 스스로 얻는다.
      (T_i 는 Z 의 선형식과 같다는 등식으로 연결)

유효성: 모든 행은 정확한 곱 (ζ = α y) 에서 성립하는 부등식이라 Ω 의 최적값을 바꾸지 않는다.
"""

using JuMP

function build_nz_omega_rlt(nd::NZData; optimizer, level::Int=2, silent::Bool=true)
    O = build_nz_omega(nd; optimizer=optimizer, mode=:global, silent=silent)
    level == 0 && return O
    m = O.model; v = O.v
    nh, S = nz_nh(nd), nd.S
    α = v[:α]; Hr = v[:Hr]
    act = [i for i in 1:nh if nd.hU[i] > 0]                 # 상한이 0 인 좌표 (프리솔브로 고정) 는 곱이 0
    nW = size(nd.W, 1)
    # ---- level 1 ----
    @constraint(m, [i in act], sum(v[:ζL][i, s] for s in 1:S) == α[i])
    @constraint(m, [i in act], sum(v[:ζF][i, s] for s in 1:S) == α[i])
    @constraint(m, [l=1:nW, s=1:S], sum(nd.W[l, i] * v[:ζL][i, s] for i in 1:nh) <= nd.wvec[l] * v[:r][s])
    @constraint(m, [l=1:nW, s=1:S], sum(nd.W[l, i] * v[:ζF][i, s] for i in 1:nh) <= nd.wvec[l] * v[:d][s])
    for (jj, k) in enumerate(Hr), s in 1:S
        i = nd.hrow[k]
        (isfinite(nd.varpiU[k]) && nd.hU[i] > 0) || continue
        # (ϖᵁ r − ϖ) α ≥ 0,  (ϖᵁ r − ϖ)(hU − α) ≥ 0
        @constraint(m, nd.varpiU[k] * v[:ζL][i, s] - v[:ζW][jj, s] >= 0)
        @constraint(m, nd.hU[i] * (nd.varpiU[k] * v[:r][s] - v[:ϖ][k, s]) - nd.varpiU[k] * v[:ζL][i, s] + v[:ζW][jj, s] >= 0)
    end
    level == 1 && return O
    # ---- level 2: belief 다면체 × α ----
    Y = Dict{Symbol,Any}(:a => v[:a], :b => v[:b], :r => v[:r], :d => v[:d], :e => v[:e])
    Z = Dict{Symbol,Any}(:r => v[:ζL], :d => v[:ζF])
    fams = [:a, :b, :e]
    v[:cpl] !== nothing && (Y[:c] = v[:cpl]; push!(fams, :c))
    for f in fams
        Z[f] = @variable(m, [i=1:nh, s=1:S], lower_bound = 0.0, upper_bound = nd.hU[i] * upper_bound(Y[f][s]), base_name = "Z$(f)")
        @constraint(m, [i in act, s=1:S], Z[f][i, s] == α[i] * Y[f][s])
        for i in 1:nh, s in 1:S; nd.hU[i] > 0 || fix(Z[f][i, s], 0.0; force=true); end
    end
    @constraint(m, [i in act], sum(Z[:a][i, s] for s in 1:S) == α[i])
    for f in fams
        @constraint(m, [l=1:nW, s=1:S], sum(nd.W[l, i] * Z[f][i, s] for i in 1:nh) <= nd.wvec[l] * Y[f][s])
    end
    q = nd.q_hat
    # 비상자 행 g(y) = c0 + Σ coef·y ≥ 0  (상·하한 행은 Gurobi 의 McCormick 과 같으므로 넣지 않음)
    rows = Tuple{Float64,Vector{Tuple{Symbol,Int,Float64}}}[]
    for (yf, ef, ε) in ((:a, :b, nd.eps_hat), (:d, :e, nd.eps_tilde))
        for s in 1:S
            push!(rows, (q[s], [(ef, s, 1.0), (yf, s, -1.0)]))
            push!(rows, (-q[s], [(yf, s, 1.0), (ef, s, 1.0)]))
        end
        push!(rows, (2ε, [(ef, s, -1.0) for s in 1:S]))
    end
    for s in 1:S
        push!(rows, (0.0, [(:a, s, 1.0 / (1.0 - nd.beta)), (:r, s, -1.0)]))
    end
    if v[:cpl] !== nothing
        for s in 1:S
            push!(rows, (0.0, [(:c, s, 1.0), (:a, s, -1.0), (:d, s, 1.0)]))
            push!(rows, (0.0, [(:c, s, 1.0), (:a, s, 1.0), (:d, s, -1.0)]))
        end
        push!(rows, (2nz_delta(nd), [(:c, s, -1.0) for s in 1:S]))
    end
    nslack = 0
    for (c0, terms) in rows
        Gy = c0 + sum(c * Y[f][idx] for (f, idx, c) in terms)
        t = nothing; tU = 0.0
        if level >= 3
            # 여유 t = g(y) ∈ [0, tU], 곱 T_i = α_i t. Gurobi 가 노드 상자에서 (α_i, t) McCormick 을 갱신
            tU = max(c0 + sum(c > 0 ? c * upper_bound(Y[f][idx]) : c * lower_bound(Y[f][idx]) for (f, idx, c) in terms), 0.0)
            t = @variable(m, lower_bound = 0.0, upper_bound = tU)
            @constraint(m, t == Gy)
        end
        for i in act
            GZ = c0 * α[i] + sum(c * Z[f][i, idx] for (f, idx, c) in terms)
            @constraint(m, GZ >= 0)                              # α_i g ≥ 0
            @constraint(m, nd.hU[i] * Gy - GZ >= 0)              # (hU_i − α_i) g ≥ 0
            if level >= 3
                T = @variable(m, lower_bound = 0.0, upper_bound = nd.hU[i] * tU)
                @constraint(m, T == α[i] * t)
                @constraint(m, T == GZ)
                nslack += 1
            end
        end
    end
    v[:rlt_level] = level; v[:rlt_slack] = nslack
    return O
end
