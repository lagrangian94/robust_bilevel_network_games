#!/usr/bin/env bash
# B&P CCG (v3 설정: μ 1000, SEED_ALL 0, NS 10) 를 지리 여러 개에 차례로 (2026-11-23 밤)
cd "C:/Users/user/AppData/Local/Temp/claude/b55b069927d98669/5ed04639-c8f0-4837-9037-f61ff4814c8e/scratchpad"
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic TL=7200 MU=1000 SEED_ALL=0 NS=10
for g in 3 9 11 4 10 12; do
  GEO=$g julia -t 12,1 --project="C:/Users/user/robust_bilevel_network_games" ccgbp.jl > bpq_g$g.log 2>&1
  echo "g$g EXIT $?"; grep -E "RESULT|시간:" bpq_g$g.log
done
echo QUEUE_DONE
