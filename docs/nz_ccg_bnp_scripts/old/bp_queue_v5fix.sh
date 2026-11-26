#!/usr/bin/env bash
# oracle 수정 (재시도마다 기저 버림, 정확 평가용 α̂ 정리) 뒤 v5 를 지리 9·12 에 다시 (2026-11-23)
cd "C:/Users/user/AppData/Local/Temp/claude/b55b069927d98669/5ed04639-c8f0-4837-9037-f61ff4814c8e/scratchpad"
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic TL=7200 ETA=1e-4 MU=1e4 MU_MAX=1e4 LAMU=1e4 SEED_ALL=0 NS=10
for g in 12 9; do
  GEO=$g julia -t 12,1 --project="C:/Users/user/robust_bilevel_network_games" ccgbp_v5.jl > bp5fix_g$g.log 2>&1
  echo "g$g EXIT $?"; grep -E "RESULT|시간:" bp5fix_g$g.log; grep -m2 "nested task error\|Exception:" bp5fix_g$g.log
done
echo QUEUE_DONE
