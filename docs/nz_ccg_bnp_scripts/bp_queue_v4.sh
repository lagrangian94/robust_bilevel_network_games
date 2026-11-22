#!/usr/bin/env bash
# B&P CCG v4 (oracle 상한 900 s + 깊게 다시, 닫힌 정수 노드 하한 유지, 걸러낸 탐색, 가지치기 기준 가격 멈춤) — 2026-11-23 아침
cd "C:/Users/user/AppData/Local/Temp/claude/b55b069927d98669/5ed04639-c8f0-4837-9037-f61ff4814c8e/scratchpad"
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic TL=7200 MU=1000 SEED_ALL=0 NS=10
for g in 12 9 11 1 10; do
  GEO=$g julia -t 12,1 --project="C:/Users/user/robust_bilevel_network_games" ccgbp_v4.jl > bp4_g$g.log 2>&1
  echo "g$g EXIT $?"; grep -E "RESULT|시간:" bp4_g$g.log
done
echo QUEUE_DONE
