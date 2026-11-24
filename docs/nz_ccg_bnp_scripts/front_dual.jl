using Serialization, Printf, Statistics
for f in ARGS
    d = deserialize(f); B = d.boxes; S = size(B[1].lo, 2); H = size(B[1].lo, 1)
    unf = [count(>(1e-6), b.hi .- b.lo) for b in B]                   # 안 고정된 (행, 시나리오) 수
    uns = [count(s -> any(b.hi[:, s] .- b.lo[:, s] .> 1e-6), 1:S) for b in B]   # 안 고정된 성분이 있는 시나리오 수
    gap = [b.ub - d.LB for b in B]
    @printf("%s: S=%d H행 %d, 열린 상자 %d | 안 고정 (행,시나리오): 중앙 %.0f/%d, 안 고정 시나리오: 중앙 %.0f/%d (최소 %d 최대 %d)\n",
            f, S, H, length(B), median(unf), H * S, median(uns), S, minimum(uns), maximum(uns))
    # 안 고정 시나리오 수 ↔ 상한 여유
    for k in sort(unique(uns))
        sel = uns .== k; @printf("   안 고정 시나리오 %2d: 상자 %4d, 상한−incumbent 중앙 %.1f\n", k, count(sel), median(gap[sel]))
    end
    # 어느 행이 안 고정인가 (행별 빈도)
    rowf = [mean(b -> count(>(1e-6), b.hi[r, :] .- b.lo[r, :]), B) for r in 1:H]
    println("   행별 안 고정 시나리오 수 평균: ", join([@sprintf("%.1f", x) for x in rowf], " "))
end
