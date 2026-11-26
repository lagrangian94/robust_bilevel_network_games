#!/usr/bin/env bash
# 기본 판 ccgbp.jl, S = 10, B&P 를 안 돌린 지리 8 개 (2026-11-23 밤). S=20 지리 1 (julia 21908) 이 끝난 뒤 시작
cd "C:/Users/user/AppData/Local/Temp/claude/b55b069927d98669/5ed04639-c8f0-4837-9037-f61ff4814c8e/scratchpad"
while tasklist //FI "PID eq 21908" 2>/dev/null | grep -qi julia; do sleep 30; done
echo "S20_G1_DONE"; grep -E "RESULT|시간:|열:" bps20_g1.log
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic TL=7200 S=10
for g in 2 6 14 15 7 8 13 5; do
  GEO=$g julia -t 12,1 --project="C:/Users/user/robust_bilevel_network_games" "C:/Users/user/robust_bilevel_network_games/docs/nz_ccg_bnp_scripts/ccgbp.jl" > bpr8_g$g.log 2>&1
  echo "g$g EXIT $?"; grep -E "RESULT|시간:|열:" bpr8_g$g.log; grep -m2 "nested task error\|Exception:\|ERROR" bpr8_g$g.log
done
echo QUEUE_DONE
