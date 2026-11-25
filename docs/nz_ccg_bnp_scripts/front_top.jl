# front_top.jl — 저장된 oracle 전선의 상한 상위 상자 폭과 상한 구간별 넓은 상자 수 (2026-11-26, md 11.14 (8)). 사용: julia front_top.jl 전선.jls ... (지리 5 활성 좌표·목표 박힘)
using Serialization, Printf, Statistics
for f in ARGS
    fr = deserialize(f); act = [1, 2, 4, 7, 12]; tgt = -3151.60
    bs = sort(fr.boxes; by=b -> -b.ub)
    mw = [maximum(b.u[act] .- b.l[act]) for b in bs]
    @printf("%s: 상위 20 상자 (상한 − 목표, 최대 폭)\n", basename(f))
    println("   ", join([@sprintf("%.0f/%.1f", bs[k].ub - tgt, mw[k]) for k in 1:min(20, length(bs))], " "))
    for (lo, hi) in ((60, Inf), (40, 60), (20, 40), (0, 20))
        sel = [k for k in eachindex(bs) if lo <= bs[k].ub - tgt < hi]
        @printf("   상한−목표 [%g, %g): %d 개, 그중 폭 > 25 %d, 폭 > 12.5 %d\n", lo, hi, length(sel), count(mw[sel] .> 25), count(mw[sel] .> 12.5))
    end
end
