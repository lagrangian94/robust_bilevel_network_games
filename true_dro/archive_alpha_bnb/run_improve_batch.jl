include(raw"C:\Users\user\AppData\Local\Temp\claude_improve_common.jl")
# 1) S=10 설계 A2 (개선 3+1), 5개 네트워크 (기준선 A / standard 는 claude_final.log)
for net in ("abilene", "polska", "grid5x5", "nobel_us", "sioux_falls")
    safe(runA, net, 10)
end
# 2) S=50: A2, standard, A (개선 전) — 같은 조건 (-t 14,1) 에서 재측정
for net in ("abilene", "polska", "grid5x5")
    safe(runA, net, 50)
    safe(runB, net, 50)
    safe(runA, net, 50; RB=false, LF=false, tag="A")
end
# 3) S=200 Abilene
safe(runA, "abilene", 200)
safe(runB, "abilene", 200)
