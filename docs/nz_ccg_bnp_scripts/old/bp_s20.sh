#!/usr/bin/env bash
# 기본 판 ccgbp.jl 을 S = 20 으로 (2026-11-23 밤), 지리 1 → 11
cd "C:/Users/user/AppData/Local/Temp/claude/b55b069927d98669/5ed04639-c8f0-4837-9037-f61ff4814c8e/scratchpad"
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic TL=7200 S=20
for g in 1 11; do
  GEO=$g julia -t 12,1 --project="C:/Users/user/robust_bilevel_network_games" "C:/Users/user/robust_bilevel_network_games/docs/nz_ccg_bnp_scripts/ccgbp.jl" > bps20_g$g.log 2>&1
  echo "g$g EXIT $?"; grep -E "RESULT|시간:|열:" bps20_g$g.log; grep -m2 "nested task error\|Exception:\|ERROR" bps20_g$g.log
done
echo QUEUE_DONE
