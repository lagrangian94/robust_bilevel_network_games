# A2 vs standard: 5개 네트워크 × S=50, 200 중 아직 없는 조합 (Abilene S=50·200, Polska·grid S=50 은 run_improve_batch.jl)
# S=50: wall 2h, boost 1h (기존 배치와 동일). S=200: 두 방식 모두 wall 4h, boost 3h.
include(joinpath(@__DIR__, "run_improve_common.jl"))
for net in ("abilene", "polska", "grid5x5", "nobel_us", "sioux_falls")
    netsize(net)
end
for net in ("nobel_us", "sioux_falls")
    safe(runA, net, 50)
    safe(runB, net, 50)
end
for net in ("polska", "grid5x5", "nobel_us", "sioux_falls")
    safe(runA, net, 200; wall=14400.0, boost=10800.0)
    safe(runB, net, 200; wall=14400.0, boost=10800.0)
end
