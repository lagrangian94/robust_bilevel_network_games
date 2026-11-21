# belief 생성 시제품의 싼 평가기: g(x, d) = [d 에서 follower 최적 반응 α] 의 정확 평가 (nz_exact_value 와 같은 LP) ≤ V*(x).
# 모형은 평가기마다 한 번 만들고, x 는 우변 (용량 행), d 는 resp 목적계수, (Q, φ) 는 정확 LP 의 계수만 바꾼다.
mutable struct BEv
    R::Model; Rα; Ry; Rcon; curd::Vector{Float64}
    MQ::Model; MQcon; Mφ::Model; Mφcon; MφcQ
    E::Model; Er; Ed; Eπ; Eκ; EFO
    lastr::Vector{Float64}             # 마지막 정확 LP 의 CVaR 재가중 r
    gx::Matrix{Float64}                # 지금 x 의 용량 행 우변 증분 (m × S)
end
function BEv(nd::NZData, env)
    mk = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
    m, ny, nh, S = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.S; nW = size(nd.W, 1); A = nd.A
    R = Model(mk); @variable(R, 0 <= α[i=1:nh] <= nd.hU[i]); @constraint(R, nd.W * α .<= nd.wvec); @variable(R, y[1:ny, 1:S] >= 0)
    @constraint(R, con[k=1:m, s=1:S], sum(A[k, j] * y[j, s] for j in 1:ny if A[k, j] != 0) - (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0) <= nd.u[k, s])
    @objective(R, Max, dot(nd.c0, α) + sum(0.0 * y[j, s] for j in 1:ny, s in 1:S))
    MQ = Model(mk); @variable(MQ, yq[1:ny] >= 0); @constraint(MQ, cq[k=1:m], sum(A[k, j] * yq[j] for j in 1:ny if A[k, j] != 0) <= 0.0); @objective(MQ, Max, dot(nd.c, yq))
    Mφ = Model(mk); @variable(Mφ, yp[1:ny] >= 0); @constraint(Mφ, cp[k=1:m], sum(A[k, j] * yp[j] for j in 1:ny if A[k, j] != 0) <= 0.0)
    cQ = @constraint(Mφ, dot(nd.c, yp) >= -1e20); @objective(Mφ, Max, dot(nd.ell, yp))
    q, ε̂, ε̃, β = nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta
    E = Model(mk)
    @variable(E, 0 <= a[1:S] <= 1); @variable(E, b[1:S] >= 0); @variable(E, 0 <= r[1:S] <= 1)
    @constraint(E, sum(a) == 1); @constraint(E, sum(r) == 1)
    @constraint(E, [s = 1:S], a[s] - b[s] <= q[s]); @constraint(E, [s = 1:S], a[s] + b[s] >= q[s])
    @constraint(E, sum(b) <= 2ε̂); @constraint(E, [s = 1:S], (1 - β) * r[s] <= a[s])
    @variable(E, 0 <= d[1:S] <= 1); @variable(E, e[1:S] >= 0); @constraint(E, sum(d) == 1)
    @constraint(E, [s = 1:S], d[s] - e[s] <= q[s]); @constraint(E, [s = 1:S], d[s] + e[s] >= q[s]); @constraint(E, sum(e) <= 2ε̃)
    nz_add_box!(E, a, d, nd)
    if nz_coupled(nd)
        @variable(E, cpl[1:S] >= 0)
        @constraint(E, [s = 1:S], a[s] - d[s] <= cpl[s]); @constraint(E, [s = 1:S], d[s] - a[s] <= cpl[s]); @constraint(E, sum(cpl) <= 2nz_delta(nd))
    end
    @variable(E, π̃[1:m, 1:S] >= 0); @variable(E, κ[1:nW] >= 0)
    @constraint(E, [j = 1:ny, s = 1:S], sum(A[k, j] * π̃[k, s] for k in 1:m if A[k, j] != 0) >= nd.c[j] * d[s])
    @constraint(E, [i = 1:nh], sum(nd.hcoef[k] * π̃[k, s] for k in 1:m, s in 1:S if nd.hrow[k] == i) + nd.c0[i] <= sum(nd.W[l, i] * κ[l] for l in 1:nW))
    FO = @constraint(E, sum(0.0 * d[s] for s in 1:S) - sum(nd.u[k, s] * π̃[k, s] for k in 1:m, s in 1:S) - dot(nd.wvec, κ) >= 0.0)
    @objective(E, Max, sum(0.0 * r[s] for s in 1:S))
    return BEv(R, α, y, con, fill(NaN, S), MQ, cq, Mφ, cp, cQ, E, r, d, π̃, κ, FO, copy(nd.q_hat), zeros(m, S))
end
const _BXROWS = Ref{Vector{Int}}()
function bev_setx!(ev::BEv, nd::NZData, x̄)
    S = nd.S
    for k in _BXROWS[], s in 1:S
        gv = nz_gx(nd, k, s, x̄)
        if gv != ev.gx[k, s]
            ev.gx[k, s] = gv
            set_normalized_rhs(ev.Rcon[k, s], nd.u[k, s] + gv)
            set_normalized_coefficient(ev.EFO, ev.Eπ[k, s], -(nd.u[k, s] + gv))
        end
    end
end
function bev_resp!(ev::BEv, nd::NZData, d)
    if d != ev.curd
        for j in 1:nz_ny(nd), s in 1:nd.S
            nd.c[j] != 0 && set_objective_coefficient(ev.R, ev.Ry[j, s], d[s] * nd.c[j])
        end
        ev.curd = copy(d)
    end
    optimize!(ev.R)
    termination_status(ev.R) == MOI.OPTIMAL || return nothing
    return [max(value(ev.Rα[i]), 0.0) for i in 1:nz_nh(nd)]
end
"α 의 정확 평가: (값, 그 값을 주는 follower belief). recourse 가 실행불가거나 α 가 어떤 belief 에서도 최적이 아니면 (−Inf, nothing)"
function bev_value!(ev::BEv, nd::NZData, α)
    S, m = nd.S, nz_m(nd); Q = zeros(S); φ = zeros(S)
    for s in 1:S
        for k in 1:m
            rr = nd.u[k, s] + ev.gx[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0)
            set_normalized_rhs(ev.MQcon[k], rr); set_normalized_rhs(ev.Mφcon[k], rr)
        end
        optimize!(ev.MQ); termination_status(ev.MQ) == MOI.OPTIMAL || return -Inf, nothing
        Q[s] = objective_value(ev.MQ)
        set_normalized_rhs(ev.MφcQ, Q[s] - 1e-9 * max(1.0, abs(Q[s])))
        optimize!(ev.Mφ); termination_status(ev.Mφ) == MOI.OPTIMAL || return -Inf, nothing
        φ[s] = objective_value(ev.Mφ)
    end
    lhs = dot(nd.c0, α)
    for s in 1:S
        set_normalized_coefficient(ev.EFO, ev.Ed[s], Q[s]); set_objective_coefficient(ev.E, ev.Er[s], φ[s])
    end
    set_normalized_rhs(ev.EFO, -lhs - 1e-7 * (1 + abs(lhs) + maximum(abs, Q)))
    optimize!(ev.E)
    termination_status(ev.E) == MOI.OPTIMAL || return -Inf, nothing
    ev.lastr = [value(ev.Er[s]) for s in 1:S]
    return objective_value(ev.E), [value(ev.Ed[s]) for s in 1:S]
end
"g(x, d): x 가 이미 bev_setx! 로 들어가 있어야 함"
function bev_g!(ev::BEv, nd::NZData, d)
    α = bev_resp!(ev, nd, d); α === nothing && return -Inf, nothing, nothing
    v, dd = bev_value!(ev, nd, α)
    return v, dd, α
end
