# A2 + menu MW: 다른 네트워크 확인 (A2 기준선은 run_improve_batch.jl / run_improve_batch2.jl, 같은 조건)
include(joinpath(@__DIR__, "run_improve_common.jl"))
for net in ("abilene", "polska", "grid5x5", "nobel_us", "sioux_falls")
    safe(runA, net, 10; MW=true, tag="A2+MW")
end
for net in ("abilene", "polska", "nobel_us", "grid5x5")
    safe(runA, net, 50; MW=true, tag="A2+MW")
end
