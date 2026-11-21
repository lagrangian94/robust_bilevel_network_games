include(joinpath(@__DIR__, "lovasz_defs.jl"))
# 한 곳 값: 같은 belief 블록 (낙관적, 높은점 완화 + follower 최적성) 을 x 고정으로 푼 값 = 그 x 에서의 정확한 값
#   (follower 최적성은 고정 x 에서 F ≥ V_F(x) 로 정확)
function value_at(xv)
    for i in 1:nx; fix(x[i], xv[i]; force=true); end
    c = @constraint(M, Fexpr >= VFx(xv)); optimize!(M); v = objective_value(M) - dot(nd.qx, xv)
    delete(M, c); for i in 1:nx; unfix(x[i]); set_lower_bound(x[i], 0.0); set_upper_bound(x[i], 1.0); end
    return v
end
VFx(xv) = (lev = zeros(Int, nB); for i in 1:nB, k in 1:K; xv[(i - 1) * K + k] > 0.5 && (lev[i] = k); end; VF(lev))
hsing = zeros(nx); t1 = @elapsed for i in 1:nB, k in 1:K
    xv = zeros(nx); xv[(i - 1) * K + k] = 1; hsing[(i - 1) * K + k] = value_at(xv)
end
@printf("한 곳 값 (리더 비용, 39 개, %.0fs): 범위 [%.1f, %.1f]\n", t1, minimum(hsing), maximum(hsing))
optimize!(M); @printf("루트 (높은점 완화): %.2f\n", objective_value(M))
@constraint(M, t0 >= dot(hsing, x)); optimize!(M)
x̄ = value.(x)
@printf("SUPERADD 루트 + [t0 ≥ Σ 한 곳 값 · x]: %.2f   (belief 하나의 최적 −2095.27)  x̄ 상위: %s\n", objective_value(M),
        string(sort([(round(x̄[i]; digits=3), i) for i in 1:nx if x̄[i] > 1e-6]; rev=true)[1:min(8, end)]))
