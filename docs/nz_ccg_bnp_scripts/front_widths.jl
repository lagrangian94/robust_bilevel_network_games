# front_widths.jl — 저장된 oracle 전선 (orep_front_*.jls) 의 열린 상자 폭 분포 (2026-11-26). DD 문턱 (활성 좌표 최대 폭) 을 받는 상자가 몇인지 보려고.
#   사용: julia front_widths.jl 전선1.jls 전선2.jls ...  (지리 5 x={4:200;7:200;12:300} 의 활성 좌표·목표가 박혀 있음)
using Serialization, Printf, Statistics
for f in ARGS
    fr = deserialize(f)
    act = [1, 2, 4, 7, 12]
    tgt = -3151.60
    W = [b.u[act] .- b.l[act] for b in fr.boxes]
    mw = [maximum(w) for w in W]; mw7 = [maximum(w[[1, 2, 3, 5]]) for w in W]
    ub = [b.ub for b in fr.boxes]
    @printf("%s: 열린 %d, 상한 > 목표 %d\n", basename(f), length(ub), count(ub .> tgt))
    for th in (6.25, 12.5, 25.0)
        @printf("   최대 폭 ≤ %.2f: %d (좌표 7 빼면 %d), 그중 상한 > 목표 %d (좌표 7 빼면 %d)\n", th, count(mw .<= th), count(mw7 .<= th),
                count((mw .<= th) .& (ub .> tgt)), count((mw7 .<= th) .& (ub .> tgt)))
    end
    @printf("   좌표별 폭 중앙값 %s, 좌표 7 폭 분포 %s\n", string(round.([median(getindex.(W, k)) for k in 1:5]; digits=2)),
            string(sort(collect(Set(round.(getindex.(W, 4); digits=2))))))
    @printf("   상한 − 목표 중앙값 %.1f, 90%% %.1f\n", median(ub .- tgt), quantile(ub .- tgt, 0.9))
end
