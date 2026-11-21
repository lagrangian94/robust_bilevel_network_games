include(joinpath(@__DIR__, "lovasz_defs.jl"))
MM = nd.meta; nA, ncu = MM[:nA], MM[:ncu]; dist = MM[:dist]; p = MM[:p]
ℓ5 = copy(nd.ell); for st in 1:nB, j in 1:ncu; ℓ5[MM[:iyS](nA + st, j)] = nd.ell[MM[:iyR](nA + st, j)]; end
W1 = get(ENV, "W1", "1") == "1"
W1 && @constraint(M, t0 >= sum(r[s] * ℓ5[j] * y[j, s] for j in 1:ny, s in 1:S if ℓ5[j] != 0))
ξ = [-nd.u[MM[:rdem](j), s] for j in 1:ncu, s in 1:S]
# 호 단위 제약 (VUB): 거점 st 에서 고객 j 로 ≤ ξ_js · (거점 개방 정도)
if get(ENV, "VUB", "1") == "1"
    for st in 1:nB, j in 1:ncu, s in 1:S
        @constraint(M, y[MM[:iyR](nA + st, j), s] + y[MM[:iyS](nA + st, j), s] <= ξ[j, s] * sum(x[(st - 1) * K + k] for k in 1:K))
    end
end
# 호 지배
if get(ENV, "DOM", "1") == "1"
    Dbar = [maximum(dist[a, j] for a in 1:nA) for j in 1:ncu]
    for st in 1:nB, j in 1:ncu
        ib = nA + st
        dist[ib, j] >= Dbar[j] && for s in 1:S; fix(y[MM[:iyS](ib, j), s], 0.0; force=true); end
        dist[ib, j] >= Dbar[j] + p && for s in 1:S; fix(y[MM[:iyR](ib, j), s], 0.0; force=true); end
    end
end
optimize!(M); x̄ = value.(x)
@printf("W1=%s VUB=%s DOM=%s: 루트 %.2f   (높은점 −5150, W1 만 −3150, 최적 −2095.27) | 분수 성분 %d | x̄ %s\n", get(ENV, "W1", "1"), get(ENV, "VUB", "1"), get(ENV, "DOM", "1"),
        objective_value(M), count(v -> 1e-6 < v < 1 - 1e-6, x̄), string(sort([(round(x̄[i]; digits=2), i) for i in 1:nx if x̄[i] > 1e-6]; rev=true)[1:min(6, end)]))
