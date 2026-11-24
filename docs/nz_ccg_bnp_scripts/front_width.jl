using Serialization, Printf, Statistics
for f in ARGS
    d = deserialize(f); B = d.boxes
    B = filter(b -> b.ub > -1e5, B)
    act = [i for i in 1:length(B[1].l) if maximum(b.u[i] - b.l[i] for b in B) > 1e-3]
    @printf("%s: 상자 %d (벌점 상자 뺌), 활성 차원 %s\n", f, length(B), string(act))
    mw = [maximum(b.u[act] .- b.l[act]) for b in B]; sl = [b.ub - d.LB for b in B]
    for (lo, hi) in ((0, 2), (2, 4), (4, 8), (8, 16), (16, 40), (40, 400))
        ix = findall(w -> lo < w <= hi, mw); isempty(ix) && continue
        @printf("  최대 폭 (%3d, %3d]: %5d 개, 상한−incumbent 중앙값 %.1f, 최소 %.1f\n", lo, hi, length(ix), median(sl[ix]), minimum(sl[ix]))
    end
end
