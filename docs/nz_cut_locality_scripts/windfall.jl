include(joinpath(@__DIR__, "lovasz_defs.jl"))
MM = nd.meta; nA, ncu = MM[:nA], MM[:ncu]
# 리더 비용식을 다시: 현장 판매를 약정가로 (횡재 W1 의 상한 효과)
tcon = [c for c in all_constraints(M, AffExpr, MOI.GreaterThan{Float64}) if normalized_coefficient(c, t0) != 0.0]
ℓ5 = copy(nd.ell); for st in 1:nB, j in 1:ncu; ℓ5[MM[:iyS](nA + st, j)] = nd.ell[MM[:iyR](nA + st, j)]; end
@constraint(M, t5, t0 >= sum(r[s] * ℓ5[j] * y[j, s] for j in 1:ny, s in 1:S if ℓ5[j] != 0))
optimize!(M); @printf("W1 현장을 약정가로 계산한 리더 비용 추가: 루트 %.2f\n", objective_value(M))
# 한 곳 값의 합 부등식 (superadd.jl 과 같은 값) 도 함께
function value_at(xv)
    for i in 1:nx; fix(x[i], xv[i]; force=true); end
    lev = zeros(Int, nB); for i in 1:nB, k in 1:K; xv[(i - 1) * K + k] > 0.5 && (lev[i] = k); end
    delete(M, t5)
    c = @constraint(M, Fexpr >= VF(lev)); optimize!(M); v = objective_value(M) - dot(nd.qx, xv)
    delete(M, c); for i in 1:nx; unfix(x[i]); set_lower_bound(x[i], 0.0); set_upper_bound(x[i], 1.0); end
    global t5 = @constraint(M, t0 >= sum(r[s] * ℓ5[j] * y[j, s] for j in 1:ny, s in 1:S if ℓ5[j] != 0))
    return v
end
hs = zeros(nx); for i in 1:nB, k in 1:K; xv = zeros(nx); xv[(i - 1) * K + k] = 1; hs[(i - 1) * K + k] = value_at(xv); end
@constraint(M, t0 >= dot(hs, x)); optimize!(M)
@printf("W1 + 한 곳 합: 루트 %.2f   (높은점 완화 −5150, 한 곳 합만 −3432, 최적 −2095.27)\n", objective_value(M))
x̄ = value.(x); @printf("  x̄: %s\n", string(sort([(round(x̄[i]; digits=3), i) for i in 1:nx if x̄[i] > 1e-6]; rev=true)[1:min(8, end)]))
