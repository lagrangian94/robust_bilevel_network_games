"""
nz_resp_heuristic.jl — α-B&B 의 primal 휴리스틱: 노드 완화가 낸 belief 에서 follower 의 **실제 최적 반응** 을 계산해 평가한다.

노드 LP 의 α̂ 는 대개 follower 최적이 아니다 (완화가 최적성을 느슨하게 보므로) — α̂ 를 고정한 정확 LP 는 벌점이 붙은 값을 낸다.
그래서 지금까지 incumbent 는 Ipopt 휴리스틱과 "우연히 정확해진 노드" 에서만 나왔고, S = 30 의 어려운 시드에서는 최적값을 찾는 데
인증보다 오래 걸렸다 (seed 2: 전역 호출 5 번 중 4 번이 더 좋은 해를 찾느라 쓴 시간).

여기서는 bilevel 의 표준 휴리스틱을 쓴다: 완화 해의 belief d̂ 를 받아 follower 의 2 단계 LP
    max_{α ∈ H, y}  c0ᵀα + Σ_s d̂_s cᵀyˢ   s.t.  A yˢ ≤ uˢ + gˢ x̄ + Bα
를 풀고, 최적 반응이 여럿이면 그중 α̂ 에 (L1 으로) 가장 가까운 것을 고른다 (완화가 가리킨, 리더에게 불리한 방향).
그 α 를 고정한 정확 LP (nz_eval_alpha!) 의 값이 실제 실행가능해의 값이다. 작은 LP 두 개 + α 고정 LP 한 개.
"""

mutable struct _NZResp
    model::Model
    α::Vector{VariableRef}
    y::Matrix{VariableRef}
    t::Vector{VariableRef}
    cdev::Vector{Tuple{ConstraintRef,ConstraintRef}}    # t_i ≥ α_i − a_i,  t_i ≥ a_i − α_i
    copt::ConstraintRef                                  # c0ᵀα + Σ_s d_s cᵀyˢ ≥ (1 단계 최적값)
end

function _nz_build_resp(nd::NZData, x̄; optimizer)
    m, ny, nh, S = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.S
    mdl = Model(optimizer); set_silent(mdl)
    @variable(mdl, 0 <= α[i=1:nh] <= nd.hU[i])
    @constraint(mdl, [l=1:size(nd.W, 1)], sum(nd.W[l, i] * α[i] for i in 1:nh) <= nd.wvec[l])
    @variable(mdl, y[1:ny, 1:S] >= 0)
    @constraint(mdl, [k=1:m, s=1:S], sum(nd.A[k, j] * y[j, s] for j in 1:ny if nd.A[k, j] != 0)
                                     - (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0)
                                     <= nd.u[k, s] + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0))
    @variable(mdl, t[1:nh] >= 0)
    cdev = [(@constraint(mdl, t[i] - α[i] >= 0.0), @constraint(mdl, t[i] + α[i] >= 0.0)) for i in 1:nh]
    copt = @constraint(mdl, sum(nd.c0[i] * α[i] for i in 1:nh) + sum(0.0 * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0) >= -1e30)
    return _NZResp(mdl, α, Matrix{VariableRef}(y), t, cdev, copt)
end

"belief d 에서 follower 의 최적 반응 중 a 에 가장 가까운 α. LP 가 안 풀리면 nothing"
function _nz_resp!(R::_NZResp, nd::NZData, d, a)
    ny, nh, S = nz_ny(nd), nz_nh(nd), nd.S
    sd = sum(d); sd > 1e-9 || return nothing
    d = d ./ sd
    set_normalized_rhs(R.copt, -1e30)
    @objective(R.model, Max, sum(nd.c0[i] * R.α[i] for i in 1:nh) + sum(d[s] * nd.c[j] * R.y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0))
    optimize!(R.model)
    termination_status(R.model) == MOI.OPTIMAL || return nothing
    v = objective_value(R.model); α1 = clamp.(value.(R.α), 0.0, nd.hU)
    for s in 1:S, j in 1:ny
        nd.c[j] != 0 && set_normalized_coefficient(R.copt, R.y[j, s], d[s] * nd.c[j])
    end
    set_normalized_rhs(R.copt, v - 1e-7 * max(1.0, abs(v)))
    for i in 1:nh; set_normalized_rhs(R.cdev[i][1], -a[i]); set_normalized_rhs(R.cdev[i][2], a[i]); end
    @objective(R.model, Min, sum(R.t))
    optimize!(R.model)
    termination_status(R.model) == MOI.OPTIMAL || return α1
    return clamp.(value.(R.α), 0.0, nd.hU)
end
