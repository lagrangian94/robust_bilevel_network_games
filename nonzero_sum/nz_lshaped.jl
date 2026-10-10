"""
nz_lshaped.jl — α-B&B 노드 완화의 시나리오 분해 (L-shaped). 설계: docs/dependent_ambiguity/lshaped_node_relaxation_design.md

노드 완화 LP (nz_alpha_bnb.jl 의 _NZSepLP) 를 master 로 바꾼다:
  - 리더 블록 변수 ŷ_s, ϖ_{N,s} (예약 행이 아닌 인증서 성분), ρ̂1·ρ̂2·ρ̂3 와 행 Lflow, LMC, N1, WD 를 삭제하고
  - 시나리오마다 η_s (블록 값의 상한 근사) 를 추가, 목적의 리더 블록 부분을 Σ η_s 로 교체.
예약 행 인증서 ϖ_H 는 master 에 남는다 → 상자 의존 RLT 행이 master 변수만 포함 → 블록 LP 가 상자 K 와 무관 → cut 이 트리 전체에서 유효.

블록 s (master 값 m_s = (r_s, ζL_s, ζW_s, ϖ_{H,s}) 고정):
  Q_s(m_s) = max (ℓ+θc)ᵀŷ − θ Σ_{k∈N}(u_ks + [k∈X] g_ks x̄) ϖ_ks − Σ_{k∈X} π̂ᵁ_k [x̄ ρ̂1 + (1−x̄) ρ̂3] − M σ
     s.t. Lflow: Aŷ + [X](ρ̂2 − ρ̂3) ≤ u_s r_s + hcoef ζL_s
          LMC:   ρ̂1 + ρ̂2 − ρ̂3 ≥ −g_s r_s
          N1:    Σ_{k∈N} A_kj ϖ_k ≥ c_j r_s − Σ_{k∈H} A_kj ϖ_{H,k}
          WD:    cᵀŷ − Σ_{k∈N}(u+[X] g x̄) ϖ_k − σ ≤ Σ_{k∈H} u ϖ_H + Σ hcoef ζW_s     (σ ≥ 0, 벌점 M: complete recourse)
  모든 우변이 m_s 에 선형·상수항 없음 → Q_s 는 1 차 동차 concave → cut  η_s ≤ Σ_rows (∂Q/∂rhs_row) · rhs_row(m)  (절편 없음).
  (∂Q/∂rhs 는 ≤ 행에서 shadow_price, ≥ 행에서 −shadow_price: JuMP 의 shadow_price 는 제약 완화 방향의 변화)
  WD 를 벌점으로 완화한 것은 유효 부등식의 완화라 master 는 여전히 원래 완화 R(K) 의 완화 (유효 상한).
초기 유계화: η_s ≤ U_s (상수), U_s = master 변수의 상·하한 안에서 Q_s 의 최댓값 (블록 + m 변수의 LP 한 번).
안정화: in-out (Ben-Ameur & Neto 2007) — 분리점 m̃ = λ m* + (1 − λ) m̄, m̄ 는 직전 master 해. m̃ 에서 위반이 없으면 m* 에서 다시 (유한 수렴).
"""

using JuMP, LinearAlgebra, Printf
import MathOptInterface as MOI

mutable struct _NZLBlock
    model::Model
    ŷ::Vector{VariableRef}
    ϖN::Vector{VariableRef}
    ρ1::Vector{VariableRef}; ρ2::Vector{VariableRef}; ρ3::Vector{VariableRef}
    cons::Vector{ConstraintRef}                  # Lflow (m), LMC (nXr), N1 (ny), WD (1) 순서
    rhs::Vector{AffExpr}                         # 각 행의 우변 (master 변수의 선형식)
    ξ::VariableRef                               # MW 용 자유 열 (평소 계수 0, 목적 0)
    mv::Vector{VariableRef}                      # 이 블록의 우변에 들어가는 master 변수 (in-out·MW 보간용)
end

mutable struct _NZLMaster
    η::Vector{VariableRef}
    blocks::Vector{_NZLBlock}
    mbar::Vector{Vector{Float64}}                # 안정화 중심 (시나리오별 m 값, rhs 행 순서의 값으로 저장하지 않고 master 변수 값으로)
    mcore::Vector{Vector{Float64}}               # MW core point (Papadakos: m° ← (m° + m*)/2)
    ncuts::Int
    pen::Float64
    t_master::Float64; t_blocks::Float64; t_rlt::Float64; rounds::Int     # 누적 계측
end

_mvars(P, s) = begin                              # 블록 s 의 우변에 들어가는 master 변수 (in-out 보간용)
    v = P.O.v; Hr = v[:Hr]
    vcat(v[:r][s], vec(v[:ζL][:, s]), vec(v[:ζW][:, s]), [v[:ϖ][k, s] for k in Hr])
end

"""
    nz_lshaped_master!(P, nd, x̄; block_optimizer, pen=1e4) -> _NZLMaster
_nz_build_sep 로 만든 노드 LP P 를 master 로 바꾸고 시나리오 블록 LP 를 만든다 (x̄ 고정).
"""
function nz_lshaped_master!(P, nd::NZData, x̄; block_optimizer, pen=1e4, follower::Bool=false, pen_f=1e6)
    m = P.O.model; v = P.O.v
    S, ny, mm, nh = nd.S, nz_ny(nd), nz_m(nd), nz_nh(nd)
    A, c, ℓ, u, g = nd.A, nd.c, nd.ell, nd.u, nd.g
    Hr, Xr = v[:Hr], v[:Xr]; nXr = length(Xr)
    θ, λU = nd.thetaU, nd.lambdaU
    isH = falses(mm); for k in Hr; isH[k] = true; end
    Nr = findall(.!isH)
    xpos = zeros(Int, mm); for (jj, k) in enumerate(Xr); xpos[k] = jj; end
    hpos = zeros(Int, mm); for (jj, k) in enumerate(Hr); hpos[k] = jj; end
    r, ζL, ζW, ϖ = v[:r], v[:ζL], v[:ζW], v[:ϖ]

    # ---- 리더 블록 행·변수 삭제 ----
    for cr in vec(m[:Lflow]); delete(m, cr); end
    for cr in vec(m[:LMC]); delete(m, cr); end
    for cr in vec(m[:N1]); delete(m, cr); end
    haskey(v, :wd) && for cr in v[:wd]; delete(m, cr); end
    delete(m, vec(v[:ŷ])); delete(m, vec(m[:ρ̂1])); delete(m, vec(m[:ρ̂2])); delete(m, vec(m[:ρ̂3]))
    delete(m, [ϖ[k, s] for k in Nr for s in 1:S])

    η = @variable(m, [1:S], base_name="η")
    # ---- 블록 LP ----
    blocks = _NZLBlock[]
    for s in 1:S
        bm = Model(block_optimizer); set_silent(bm)
        @variable(bm, ŷb[1:ny] >= 0); @variable(bm, ϖb[1:length(Nr)] >= 0)
        @variable(bm, ρ1[1:nXr] >= 0); @variable(bm, ρ2[1:nXr] >= 0); @variable(bm, ρ3[1:nXr] >= 0)
        @variable(bm, σ >= 0)
        @variable(bm, ξ)                                               # MW 용 (평소엔 계수 0)
        npos = zeros(Int, mm); for (t, k) in enumerate(Nr); npos[k] = t; end
        cons = ConstraintRef[]; rhs = AffExpr[]
        for k in 1:mm                                                   # Lflow
            e = @expression(bm, sum(A[k, j] * ŷb[j] for j in 1:ny if A[k, j] != 0) +
                                (xpos[k] > 0 ? ρ2[xpos[k]] - ρ3[xpos[k]] : 0.0))
            push!(cons, @constraint(bm, e <= 0.0))
            push!(rhs, u[k, s] * r[s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * ζL[nd.hrow[k], s] : 0.0))
        end
        for (jj, k) in enumerate(Xr)                                    # LMC
            push!(cons, @constraint(bm, ρ1[jj] + ρ2[jj] - ρ3[jj] >= 0.0))
            push!(rhs, -g[k, s] * r[s])
        end
        for j in 1:ny                                                   # N1
            e = sum((A[k, j] * ϖb[npos[k]] for k in Nr if A[k, j] != 0); init=AffExpr(0.0))
            push!(cons, @constraint(bm, e >= 0.0))
            push!(rhs, c[j] * r[s] - sum((A[k, j] * ϖ[k, s] for k in Hr if A[k, j] != 0); init=AffExpr(0.0)))
        end
        wdc = [nd.u[k, s] + (nd.xrow[k] > 0 ? g[k, s] * x̄[nd.xrow[k]] : 0.0) for k in Nr]
        push!(cons, @constraint(bm, sum(c[j] * ŷb[j] for j in 1:ny if c[j] != 0) -               # WD (벌점 완화)
                                    sum(wdc[t] * ϖb[t] for t in eachindex(Nr) if wdc[t] != 0) - σ <= 0.0))
        push!(rhs, sum((u[k, s] * ϖ[k, s] for k in Hr if u[k, s] != 0); init=AffExpr(0.0)) +
                   sum(nd.hcoef[k] * ζW[jj, s] for (jj, k) in enumerate(Hr)))
        bo = AffExpr(0.0)
        for j in 1:ny; (ℓ[j] + θ * c[j]) != 0 && add_to_expression!(bo, ℓ[j] + θ * c[j], ŷb[j]); end
        for (t, k) in enumerate(Nr)
            coef = u[k, s] + (nd.xrow[k] > 0 ? g[k, s] * x̄[nd.xrow[k]] : 0.0)
            coef != 0 && add_to_expression!(bo, -θ * coef, ϖb[t])
        end
        for (jj, k) in enumerate(Xr)
            xi = x̄[nd.xrow[k]]
            add_to_expression!(bo, -nd.piLU[k] * xi, ρ1[jj]); add_to_expression!(bo, -nd.piLU[k] * (1 - xi), ρ3[jj])
        end
        add_to_expression!(bo, -pen, σ)
        @objective(bm, Max, bo)
        push!(blocks, _NZLBlock(bm, ŷb, ϖb, ρ1, ρ2, ρ3, cons, rhs, ξ, _mvars(P, s)))
    end

    # ---- 초기 유계화: η_s ≤ U_s,  U_s = max { Q_s(m) : m 의 각 성분이 master 변수 상·하한 안 } ----
    # (상한을 1 차 동차성으로 U r 꼴로 두면 완화에서 ζ/r 가 단일 시나리오 최대를 넘을 수 있어 무효 → 상수 상한)
    for s in 1:S
        B = blocks[s]
        mv = B.mv
        bm = B.model
        mvar = Dict{VariableRef,VariableRef}()
        for x in mv
            lb = has_lower_bound(x) ? lower_bound(x) : 0.0
            ub = has_upper_bound(x) ? upper_bound(x) : 1e6
            mvar[x] = @variable(bm, lower_bound=lb, upper_bound=ub)
        end
        tmp = ConstraintRef[]
        for (cr, e) in zip(B.cons, B.rhs)           # 행을 "lhs − rhs(m) 부호" 로 임시 교체: 계수 추가 후 rhs 0
            for (x, coef) in e.terms; set_normalized_coefficient(cr, mvar[x], -coef); end
            set_normalized_rhs(cr, e.constant)
        end
        optimize!(bm)
        st = termination_status(bm)
        Us = st == MOI.OPTIMAL ? objective_value(bm) : 1e9
        for (cr, e) in zip(B.cons, B.rhs); for (x, _) in e.terms; set_normalized_coefficient(cr, mvar[x], 0.0); end; end
        delete(bm, collect(values(mvar)))
        @constraint(m, η[s] <= Us + 1e-6 * max(1.0, abs(Us)))
    end

    # ---- follower 블록 분해 (follower = true) ----
    # master 에 남김: d, ζF, π̃_H (예약 행), π̃_X (x 행), κ, ρ⁰, 새 집계 W_s.  블록: ȳ, π̃_N (나머지 행), ρ̃.
    # Flam 은 Σ_s W_s + wᵀκ + Σ(ρ02 − ρ03) ≤ c0ᵀα 로 다시 쓰고, 블록에 "Σ_N u π̃_N − cᵀȳ − σ ≤ W_s − Σ_{H∪X} u π̃" (σ 벌점 pen_f).
    # 블록 값 ≤ 0 (목적이 −λᵁπ_Fᵁ ρ̃ − pen σ) → 초기 상한 η^F ≤ 0. Fh·Fpsi 는 π̃_H·π̃_X 만 포함해 master 에 그대로.
    ηF = VariableRef[]
    if follower
        π̃, d, ζF = v[:π̃], v[:d], v[:ζF]
        isHX = isH .| (xpos .> 0); NF = findall(.!isHX); HX = findall(isHX)
        for cr in vec(m[:Fflow]); delete(m, cr); end
        for cr in vec(m[:FMC]); delete(m, cr); end
        for cr in vec(m[:Ffeas]); delete(m, cr); end
        delete(m, m[:Flam])
        delete(m, vec(v[:ȳ])); delete(m, vec(m[:ρ̃1])); delete(m, vec(m[:ρ̃2])); delete(m, vec(m[:ρ̃3]))
        delete(m, [π̃[k, s] for k in NF for s in 1:S])
        W = @variable(m, [1:S], base_name="Wf")
        @constraint(m, sum(W) + sum(nd.wvec[l] * v[:κ][l] for l in 1:size(nd.W, 1)) +
                       sum(m[:ρ02][i] - v[:ρ03][i] for i in 1:nd.nx) <= sum(nd.c0[i] * v[:α][i] for i in 1:nh))
        ηF = @variable(m, [1:S], upper_bound=0.0, base_name="ηF")
        nfpos = zeros(Int, mm); for (t, k) in enumerate(NF); nfpos[k] = t; end
        for s in 1:S
            bm = Model(block_optimizer); set_silent(bm)
            @variable(bm, yb[1:ny] >= 0); @variable(bm, πb[1:length(NF)] >= 0)
            @variable(bm, ρ1[1:nXr] >= 0); @variable(bm, ρ2[1:nXr] >= 0); @variable(bm, ρ3[1:nXr] >= 0)
            @variable(bm, σ >= 0); @variable(bm, ξ)
            cons = ConstraintRef[]; rhs = AffExpr[]
            for k in 1:mm                                               # Fflow
                e = @expression(bm, sum(A[k, j] * yb[j] for j in 1:ny if A[k, j] != 0) +
                                    (xpos[k] > 0 ? ρ2[xpos[k]] - ρ3[xpos[k]] : 0.0))
                push!(cons, @constraint(bm, e <= 0.0))
                push!(rhs, u[k, s] * d[s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * ζF[nd.hrow[k], s] : 0.0))
            end
            for (jj, k) in enumerate(Xr)                                # FMC
                push!(cons, @constraint(bm, ρ1[jj] + ρ2[jj] - ρ3[jj] >= 0.0))
                push!(rhs, -g[k, s] * d[s])
            end
            for j in 1:ny                                               # Ffeas
                e = sum((A[k, j] * πb[nfpos[k]] for k in NF if A[k, j] != 0); init=AffExpr(0.0))
                push!(cons, @constraint(bm, e >= 0.0))
                push!(rhs, c[j] * d[s] - sum((A[k, j] * π̃[k, s] for k in HX if A[k, j] != 0); init=AffExpr(0.0)))
            end
            push!(cons, @constraint(bm, sum((u[k, s] * πb[t] for (t, k) in enumerate(NF) if u[k, s] != 0); init=AffExpr(0.0)) -
                                        sum(c[j] * yb[j] for j in 1:ny if c[j] != 0) - σ <= 0.0))     # W 행
            push!(rhs, W[s] - sum((u[k, s] * π̃[k, s] for k in HX if u[k, s] != 0); init=AffExpr(0.0)))
            bo = AffExpr(0.0)
            for (jj, k) in enumerate(Xr)
                xi = x̄[nd.xrow[k]]
                add_to_expression!(bo, -λU * nd.piFU[k] * xi, ρ1[jj]); add_to_expression!(bo, -λU * nd.piFU[k] * (1 - xi), ρ3[jj])
            end
            add_to_expression!(bo, -pen_f, σ)
            @objective(bm, Max, bo)
            mvF = vcat(d[s], vec(ζF[:, s]), [π̃[k, s] for k in HX], W[s])
            push!(blocks, _NZLBlock(bm, yb, πb, ρ1, ρ2, ρ3, cons, rhs, ξ, mvF))
        end
    end

    # ---- master 목적: Σ η (리더) + Σ η^F (follower 분해 시) + master 에 남은 항 ----
    obj = AffExpr(0.0)
    for s in 1:S; add_to_expression!(obj, 1.0, η[s]); end
    for x in ηF; add_to_expression!(obj, 1.0, x); end
    for k in Hr, s in 1:S; u[k, s] != 0 && add_to_expression!(obj, -θ * u[k, s], ϖ[k, s]); end
    for (jj, k) in enumerate(Hr), s in 1:S; add_to_expression!(obj, -θ * nd.hcoef[k], ζW[jj, s]); end
    if !follower
        for (jj, k) in enumerate(Xr), s in 1:S
            xi = x̄[nd.xrow[k]]
            add_to_expression!(obj, -λU * nd.piFU[k] * xi, v[:ρ̃1][jj, s])
            add_to_expression!(obj, -λU * nd.piFU[k] * (1 - xi), v[:ρ̃3][jj, s])
        end
    end
    for i in 1:nd.nx
        add_to_expression!(obj, -λU * x̄[i], v[:ρ01][i]); add_to_expression!(obj, -λU * (1 - x̄[i]), v[:ρ03][i])
    end
    @objective(m, Max, obj)
    nb = length(blocks)
    return _NZLMaster(vcat(η, ηF), blocks, [Float64[] for _ in 1:nb], [Float64[] for _ in 1:nb], 0, pen, 0.0, 0.0, 0.0, 0)
end

"블록 LP 를 우변 값으로 풀고 (값, shadow price) 를 돌려줌. rhsval: AffExpr → 값"
function _nz_lblock_solve!(B::_NZLBlock, rhsval)
    for (cr, e) in zip(B.cons, B.rhs); set_normalized_rhs(cr, rhsval(e)); end
    optimize!(B.model)
    st = termination_status(B.model)
    st == MOI.OPTIMAL || error("L-shaped 블록 LP: $st")
    # ∂Q/∂rhs: JuMP 의 shadow_price 는 "제약 완화" 방향의 변화라 ≥ 행에서는 우변 감소 방향 → 부호를 뒤집음
    grad = [(constraint_object(cr).set isa MOI.GreaterThan ? -1.0 : 1.0) * shadow_price(cr) for cr in B.cons]
    return objective_value(B.model), grad
end

"""
Magnanti–Wong (Pareto-optimal) cut: m* 에서의 최적 쌍대해들 중 core point m° 에서 값이 가장 작은 것.
MW LP  min{πᵀh(m°) : π 쌍대 실행가능, πᵀh(m*) = q*} 의 쌍대 = 블록 LP 에 자유열 ξ 추가:
  각 행  lhs + ξ·h_row(m*)  (부등호)  h_row(m°),  목적 + q*·ξ.  그 shadow price 가 MW 쌍대해.
실패 (비최적) 하면 nothing → 호출 측이 일반 cut 을 씀.
"""
function _nz_lblock_mw!(B::_NZLBlock, hstar::Vector{Float64}, hcore::Vector{Float64}, qstar::Float64)
    for (t, cr) in enumerate(B.cons)
        set_normalized_coefficient(cr, B.ξ, hstar[t]); set_normalized_rhs(cr, hcore[t])
    end
    set_objective_coefficient(B.model, B.ξ, qstar)
    optimize!(B.model)
    ok = termination_status(B.model) == MOI.OPTIMAL
    grad = ok ? [(constraint_object(cr).set isa MOI.GreaterThan ? -1.0 : 1.0) * shadow_price(cr) for cr in B.cons] : nothing
    for cr in B.cons; set_normalized_coefficient(cr, B.ξ, 0.0); end
    set_objective_coefficient(B.model, B.ξ, 0.0)
    return grad
end

"""
    nz_lshaped_solve!(P, M; maxrounds=30, rlt=true, inout=0.5, tol=1e-6, stall_rounds=3) -> 상한
master 와 블록 cut, RLT 분리를 번갈아. 어느 시점에 멈춰도 반환값은 유효한 상한 (master 가 R(K) 의 완화).
inout ∈ (0, 1]: 분리점 = inout·m* + (1−inout)·m̄ (1 이면 안정화 없음).
"""
function nz_lshaped_solve!(P, M::_NZLMaster; maxrounds=30, rlt=true, inout=0.5, tol=1e-6, stall_rounds=3,
                           sep_maxadd=5000, mw::Bool=false, pool=nothing, plock=nothing, wid=0, imported=nothing)
    # pool: worker 간 공유 cut [(만든 worker, 블록 번호, ∂Q/∂rhs 벡터)]. 블록 LP 가 상자와 무관해 어느 노드·worker 에서나 유효.
    m = P.O.model
    S = length(M.blocks)
    if pool !== nothing                                  # 다른 worker 가 만든 cut 가져오기
        newp = lock(plock) do
            k0 = imported[]; imported[] = length(pool); pool[k0+1:end]
        end
        for (w, s, gr) in newp
            w == wid && continue
            B = M.blocks[s]
            @constraint(m, M.η[s] <= sum(gr[t] * B.rhs[t] for t in eachindex(gr)))
        end
    end
    zprev = Inf; nstall = 0; z = NaN
    for round in 1:maxrounds
        M.rounds += 1
        M.t_master += @elapsed (z = _nz_solve_lp!(m))
        (isnan(z) || isinf(z)) && return z
        # 해 값을 먼저 모두 읽고 (모델을 바꾸면 JuMP 가 해를 무효로 봄), cut·RLT 행은 마지막에 한꺼번에 추가
        mstar = [value.(B.mv) for B in M.blocks]
        ηv = value.(M.η)
        M.t_rlt += @elapsed (V = rlt ? _nz_sep_violations(P; tol=tol) : Tuple{Float64,Int,Int,Bool}[])
        newcuts = Tuple{Int,Vector{Float64}}[]
        tb0 = time()
        for s in 1:S
            B = M.blocks[s]
            vals_star = Dict(zip(B.mv, mstar[s]))
            λ = (isempty(M.mbar[s]) || inout >= 1) ? 1.0 : inout
            for attempt in 1:2
                mt = λ < 1 ? λ .* mstar[s] .+ (1 - λ) .* M.mbar[s] : mstar[s]
                vals = Dict(zip(B.mv, mt))
                q, sp = _nz_lblock_solve!(B, e -> value(x -> get(vals, x, 0.0), e))
                if mw                                    # Pareto-optimal 로 교체 (core point 는 지금까지 master 해의 반복 평균)
                    isempty(M.mcore[s]) && (M.mcore[s] = copy(mt))
                    cvals = Dict(zip(B.mv, M.mcore[s]))
                    hstar = [value(x -> get(vals, x, 0.0), e) for e in B.rhs]
                    hcore = [value(x -> get(cvals, x, 0.0), e) for e in B.rhs]
                    g = _nz_lblock_mw!(B, hstar, hcore, q - 1e-9 * max(1.0, abs(q)))
                    g === nothing || (sp = g)
                end
                cutval = sum(sp[t] * value(x -> get(vals_star, x, 0.0), B.rhs[t]) for t in eachindex(sp))
                if ηv[s] > cutval + tol * max(1.0, abs(ηv[s]))
                    push!(newcuts, (s, sp)); break
                end
                λ == 1.0 && break
                λ = 1.0                                  # 안정화 점에서 위반 없음 → m* 에서 다시
            end
            M.mbar[s] = mstar[s]
            mw && (M.mcore[s] = isempty(M.mcore[s]) ? copy(mstar[s]) : 0.5 .* (M.mcore[s] .+ mstar[s]))
        end
        M.t_blocks += time() - tb0
        for (s, sp) in newcuts
            B = M.blocks[s]
            @constraint(m, M.η[s] <= sum(sp[t] * B.rhs[t] for t in eachindex(sp)))
        end
        if pool !== nothing && !isempty(newcuts)
            lock(plock) do; for (s, sp) in newcuts; push!(pool, (wid, s, sp)); end; end
        end
        M.ncuts += length(newcuts)
        added = length(newcuts)
        if !isempty(V)
            sort!(V; by=first)
            for (_, i, j, islo) in V[1:min(sep_maxadd, length(V))]
                _nz_sep_add!(P, i, j, islo, islo ? P.l[i] : P.u[i]); added += 1
            end
        end
        added == 0 && return z
        # 정체 판정은 블록 cut 이 없는 (RLT 만 추가된) 라운드만 셈: cut 은 여러 라운드 상한을 못 움직여도 수렴에 필요
        nstall = (isempty(newcuts) && zprev - z <= tol * max(1.0, abs(z))) ? nstall + 1 : (isempty(newcuts) ? nstall : 0)
        zprev = z
        nstall >= stall_rounds && return _nz_solve_lp!(m)     # 행·cut 을 추가한 뒤라 다시 풀어 해를 유효하게
    end
    return _nz_solve_lp!(m)
end
