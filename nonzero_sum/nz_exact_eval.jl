# nz_exact_eval.jl — λ·θ 와 무관한 정확한 평가 (witness 값). lab_lambda_certify.jl 의 같은 이름 함수를 옮긴 것 (2026-10-17).
#   nz_recourse_values(nd, x̄, α) : α 고정 recourse 값 Q_s, pessimistic φ_s
#   nz_exact_value(nd, x̄, α)     : α 가 어떤 belief 에서 follower 최적일 때 리더 평가의 최댓값 (LP, 아니면 −Inf) ≤ V*(x̄)
#   exact_lb(nd, x̄, α°)          : incumbent α° 와 그 둘레의 정확히 최적인 반응들 (꼭짓점, 최적면 사영) 중 최댓값 ≤ V*(x̄)
# 주의: lab_lambda_certify.jl 은 아직 자기 안의 복사본을 쓴다 (돌고 있는 큐가 그 파일을 쓰므로 큐가 끝난 뒤 이 파일을 include 하도록 정리).
# GRB1() (스레드 1, 조용히) 를 부르는 쪽에서 정의해야 한다.
# (2026-10-17: 머리말을 docstring 으로 썼다가 바로 뒤 함수 docstring 과 겹쳐 "cannot document" 오류 → 주석으로 바꿈)
"α 고정 recourse: Q_s(α) = max cᵀy, φ_s(α) = max ℓᵀy s.t. cᵀy ≥ Q_s (pessimistic)"
function nz_recourse_values(nd::NZData, x̄, α)
    S, m, ny = nd.S, nz_m(nd), nz_ny(nd)
    Q = zeros(S); φ = zeros(S)
    for s in 1:S
        mdl = Model(GRB1); @variable(mdl, y[1:ny] >= 0); rr = nz_rhs(nd, x̄, α, s)
        @constraint(mdl, [k = 1:m], sum(nd.A[k, j] * y[j] for j in 1:ny if nd.A[k, j] != 0) <= rr[k])
        @objective(mdl, Max, dot(nd.c, y)); optimize!(mdl)
        termination_status(mdl) == MOI.OPTIMAL || error("recourse LP s=$s: $(termination_status(mdl))")
        Q[s] = objective_value(mdl)
        @constraint(mdl, dot(nd.c, y) >= Q[s] - 1e-9 * max(1.0, abs(Q[s])))
        @objective(mdl, Max, dot(nd.ell, y)); optimize!(mdl)
        termination_status(mdl) == MOI.OPTIMAL || error("pessimistic LP s=$s: $(termination_status(mdl))")
        φ[s] = objective_value(mdl)
    end
    return Q, φ
end

"λ·θ 와 무관한 정확한 값: α 가 어떤 belief 에서 follower 최적 반응일 때 리더 평가의 최댓값, 아니면 −Inf"
function nz_exact_value(nd::NZData, x̄, α; tolF=1e-7)
    S, m, ny, nh = nd.S, nz_m(nd), nz_ny(nd), nz_nh(nd); nW = size(nd.W, 1)
    any(α .< -1e-9) && return -Inf
    any(nd.W * α .> nd.wvec .+ 1e-7) && return -Inf
    Q, φ = nz_recourse_values(nd, x̄, α)
    q, ε̂, ε̃, β = nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta
    M = Model(GRB1)
    @variable(M, 0 <= a[1:S] <= 1); @variable(M, b[1:S] >= 0); @variable(M, 0 <= r[1:S] <= 1)
    @constraint(M, sum(a) == 1); @constraint(M, sum(r) == 1)
    @constraint(M, [s = 1:S], a[s] - b[s] <= q[s]); @constraint(M, [s = 1:S], a[s] + b[s] >= q[s])
    @constraint(M, sum(b) <= 2ε̂); @constraint(M, [s = 1:S], (1 - β) * r[s] <= a[s])
    @variable(M, 0 <= d[1:S] <= 1); @variable(M, e[1:S] >= 0); @constraint(M, sum(d) == 1)
    @constraint(M, [s = 1:S], d[s] - e[s] <= q[s]); @constraint(M, [s = 1:S], d[s] + e[s] >= q[s]); @constraint(M, sum(e) <= 2ε̃)
    if nz_coupled(nd)
        @variable(M, cpl[1:S] >= 0)
        @constraint(M, [s = 1:S], a[s] - d[s] <= cpl[s]); @constraint(M, [s = 1:S], d[s] - a[s] <= cpl[s]); @constraint(M, sum(cpl) <= 2nz_delta(nd))
    end
    @variable(M, π̃[1:m, 1:S] >= 0); @variable(M, κ[1:nW] >= 0)
    @constraint(M, [j = 1:ny, s = 1:S], sum(nd.A[k, j] * π̃[k, s] for k in 1:m if nd.A[k, j] != 0) >= nd.c[j] * d[s])
    @constraint(M, [i = 1:nh], sum(nd.hcoef[k] * π̃[k, s] for k in 1:m, s in 1:S if nd.hrow[k] == i) + nd.c0[i] <= sum(nd.W[l, i] * κ[l] for l in 1:nW))
    B = [nz_rhs(nd, x̄, zeros(nh), s) for s in 1:S]
    lhs = dot(nd.c0, α)
    @constraint(M, lhs + sum(Q[s] * d[s] for s in 1:S) >= sum(B[s][k] * π̃[k, s] for k in 1:m, s in 1:S if B[s][k] != 0) + dot(nd.wvec, κ)
                                                          - tolF * (1 + abs(lhs) + maximum(abs, Q)))
    @objective(M, Max, sum(φ[s] * r[s] for s in 1:S)); optimize!(M)
    st = termination_status(M)
    st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED) && return -Inf
    st == MOI.OPTIMAL || error("exact value LP: $st")
    return objective_value(M), value.(d)
end
_ev(nd, x̄, α) = (z = nz_exact_value(nd, x̄, α); z isa Tuple ? z[1] : z)

"incumbent α° 와 그 둘레의 정확히 최적인 반응들의 정확한 값 중 최댓값 (LB_exact)"
function exact_lb(nd, x̄, α°)
    nh, ny, S = nz_nh(nd), nz_ny(nd), nd.S
    hU, _ = nz_presolve_hU(nd, x̄); nd0 = nz_with_hU(nd, hU); a0 = clamp.(α°, 0.0, nd0.hU)
    best = _ev(nd, x̄, a0); src = "incumbent"
    F = nz_build_fixed_alpha(nd0, x̄; optimizer=GRB1); nz_eval_alpha!(F, nd0, a0)
    d° = [value(F.v[:d][s]) for s in 1:S]
    fm = Model(GRB1); set_optimizer_attribute(fm, "Method", 0)
    @variable(fm, 0 <= αf[i = 1:nh] <= nd0.hU[i]); @variable(fm, yf[1:ny, 1:S] >= 0); @constraint(fm, nd0.W * αf .<= nd0.wvec)
    for s in 1:S, k in 1:nz_m(nd0)
        @constraint(fm, sum(nd0.A[k, j] * yf[j, s] for j in 1:ny if nd0.A[k, j] != 0) <= nz_rhs(nd0, x̄, zeros(nh), s)[k] + (nd0.hrow[k] > 0 ? nd0.hcoef[k] * αf[nd0.hrow[k]] : 0.0))
    end
    fobj = @expression(fm, dot(nd0.c0, αf) + sum(d°[s] * nd0.c[j] * yf[j, s] for j in 1:ny, s in 1:S if nd0.c[j] != 0))
    @objective(fm, Max, fobj); optimize!(fm); vF = objective_value(fm)
    z = _ev(nd, x̄, clamp.(value.(αf), 0.0, nd0.hU)); z > best && (best = z; src = "꼭짓점")
    @variable(fm, dev[1:nh] >= 0); @constraint(fm, [i = 1:nh], dev[i] >= αf[i] - a0[i]); @constraint(fm, [i = 1:nh], dev[i] >= a0[i] - αf[i])
    @constraint(fm, fobj >= vF - 1e-9 * max(1.0, abs(vF))); @objective(fm, Min, sum(dev)); optimize!(fm)
    z = _ev(nd, x̄, clamp.(value.(αf), 0.0, nd0.hU)); z > best && (best = z; src = "최적면 사영")
    return best, src
end
