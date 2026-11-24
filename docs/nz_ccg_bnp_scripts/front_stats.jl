using Serialization, Printf, Statistics
f = ARGS[1]; d = deserialize(f)
B = d.boxes; n = length(B); nα = length(B[1].l)
@printf("%s: x=%s LB %.2f UB %.2f 루트 UB %.2f, 열린 노드 %d, α 차원 %d\n", f, d.xs, d.LB, d.UB, d.root_UB, n, nα)
W = reduce(hcat, [b.u .- b.l for b in B])   # nα × n
full = maximum(W, dims=2)[:]                  # 차원별 최대 폭
@printf("incumbent α: %s\n", join([@sprintf("%.3f", a) for a in d.α], " "))
println("차원별 열린 상자 폭 (중앙값 / 최대), α̂:")
for i in 1:nα
    @printf("  %2d  med %.4f  max %.4f  α̂ %.4f\n", i, median(W[i, :]), full[i], d.α[i])
end
nw = [count(>(1e-3), W[:, j]) for j in 1:n]
@printf("상자당 폭 > 1e-3 차원 수: 중앙값 %.1f, 최소 %d, 최대 %d\n", median(nw), minimum(nw), maximum(nw))
ubs = [b.ub for b in B]; q = sort(ubs, rev=true)
@printf("상한 상위 10: %s\n", join([@sprintf("%.2f", u) for u in q[1:min(10, end)]], " "))
gap = (q .- d.LB)
for t in (1.0, 5.0, 10.0, 50.0, 100.0)
    @printf("  상한 > incumbent + %.0f: %d 개\n", t, count(>(t), gap))
end
