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

# 약정 상권: 고객 j 에 대해 리더 거점 i 의 약정 (f + dist) 이 경쟁 점포 현장 구매 (Dbar_j + p) 보다 싼 경우만 j 를 위한 약정이 합리적
f_ = -nd.c0[1]; Dbar = [maximum(dist[a, j] for a in 1:nA) for j in 1:ncu]
RC = [[j for j in 1:ncu if dist[nA + st, j] < Dbar[j] + p - f_] for st in 1:nB]
ξmax = [maximum(ξ[j, s] for s in 1:S) for j in 1:ncu]
hB(st) = h[MM[:hidx](nA + st, 1)]
if get(ENV, "RCs", "1") == "1"
    for st in 1:nB
        cap = maximum(sum(ξ[j, s] for j in RC[st]; init=0.0) for s in 1:S)
        @constraint(M, hB(st) <= cap * sum(x[(st - 1) * K + k] for k in 1:K))       # 거점별 (열린 정도에 비례)
    end
end
if get(ENV, "RCu", "1") == "1"
    @variable(M, 0 <= cov[1:ncu] <= 1)
    for j in 1:ncu
        @constraint(M, cov[j] <= sum(x[(st - 1) * K + k] for st in 1:nB if j in RC[st] for k in 1:K; init=0.0))
    end
    @constraint(M, sum(hB(st) for st in 1:nB) <= sum(ξmax[j] * cov[j] for j in 1:ncu))   # 열린 거점들의 약정 합 ≤ 덮인 상권 수요
end
optimize!(M); x̄ = value.(x)
hs = [value(hB(st)) for st in 1:nB]
@printf("RC W1=%s VUB=%s DOM=%s RCs=%s RCu=%s: 루트 %.2f  (높은점 −5150, 최적 −2095.27) | 분수 %d | x̄ %s | Σh %.0f\n", get(ENV, "W1", "0"), get(ENV, "VUB", "1"), get(ENV, "DOM", "1"), get(ENV, "RCs", "1"), get(ENV, "RCu", "1"),
        objective_value(M), count(v -> 1e-6 < v < 1 - 1e-6, x̄), string(sort([(round(x̄[i]; digits=2), i) for i in 1:nx if x̄[i] > 1e-6]; rev=true)[1:min(6, end)]), sum(hs))
println("   상권 크기 (고객 수): ", [length(RC[st]) for st in 1:nB])
