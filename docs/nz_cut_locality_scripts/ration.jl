include(joinpath(@__DIR__, "lovasz_defs.jl"))
# follower 합리성 부등식 (x 와 무관, 쌍대·벌점 없음)
MM = nd.meta; nA, ncu, nst = MM[:nA], MM[:ncu], MM[:nst]; dist = MM[:dist]; p = MM[:p]; f = -nd.c0[1]
@printf("follower 가격: 약정비 f = %.0f, 현장 할증 p = %.0f, 경쟁 점포 %d 곳\n", f, p, nA)
Dbar = [maximum(dist[a, j] for a in 1:nA) for j in 1:ncu]          # 경쟁 점포 현장 구매는 항상 가능 (A 총용량 > 총수요): 비용 ≤ Dbar_j + p
npr_S = 0; npr_R = 0
for st in 1:nB, j in 1:ncu
    ib = nA + st
    if dist[ib, j] >= Dbar[j]                                       # 리더 현장 호: dist + p ≥ Dbar + p
        for s in 1:S; fix(y[MM[:iyS](ib, j), s], 0.0; force=true); end; global npr_S += 1
    end
    if dist[ib, j] >= Dbar[j] + p                                   # 리더 약정 호: (약정비는 이미 냄) dist ≥ Dbar + p
        for s in 1:S; fix(y[MM[:iyR](ib, j), s], 0.0; force=true); end; global npr_R += 1
    end
end
@printf("호 지배로 막은 호: 현장 %d, 약정 %d (리더 거점 %d × 고객 %d)\n", npr_S, npr_R, nB, ncu)
# 약정 합리성: 거점 i 의 약정 ≤ 어떤 시나리오에서도 그 거점이 이길 수 있는 고객 수요의 최대
ξ = [-nd.u[MM[:rdem](j), s] for j in 1:ncu, s in 1:S]
for st in 1:nB
    ib = nA + st; catch_ = [j for j in 1:ncu if dist[ib, j] < Dbar[j] + p]
    cap = maximum(sum(ξ[j, s] for j in catch_; init=0.0) for s in 1:S)
    MODE = get(ENV, "RES", "1")
    MODE == "1" && cap < nd.hU[MM[:hidx](ib, 1)] && set_upper_bound(h[MM[:hidx](ib, 1)], cap)
end
optimize!(M); @printf("RATION 루트 (높은점 완화 + 호 지배 + 약정 합리성): %.2f   (전: −5150, 최적 −2095.27)\n", objective_value(M))
x̄ = value.(x); @printf("  x̄ 상위: %s\n", string(sort([(round(x̄[i]; digits=3), i) for i in 1:nx if x̄[i] > 1e-6]; rev=true)[1:min(8, end)]))
