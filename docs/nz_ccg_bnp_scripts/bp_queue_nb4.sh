#!/usr/bin/env bash
# N_B = 4 (|X| = 66,379) 규모 실험, v5 설정 + 고친 oracle (2026-11-23). v5fix 큐가 끝난 뒤 시작
cd "C:/Users/user/AppData/Local/Temp/claude/b55b069927d98669/5ed04639-c8f0-4837-9037-f61ff4814c8e/scratchpad"
until grep -q QUEUE_DONE bp_queue_v5fix.out 2>/dev/null; do sleep 60; done
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic TL=10800 ETA=1e-4 MU=1e4 MU_MAX=1e4 LAMU=1e4 SEED_ALL=0 NS=10 NB=4
for g in 1 11 10; do
  GEO=$g julia -t 12,1 --project="C:/Users/user/robust_bilevel_network_games" ccgbp_v5nb.jl > bpnb4_g$g.log 2>&1
  echo "g$g EXIT $?"; grep -E "RESULT|시간:" bpnb4_g$g.log; grep -m2 "nested task error\|Exception:" bpnb4_g$g.log
done
echo QUEUE_DONE
