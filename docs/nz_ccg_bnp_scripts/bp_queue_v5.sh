#!/usr/bin/env bash
# B&P CCG v5 + 인증용 설정 (η, μ, λᵁ) = (1e-4, 1e4, 1e4), μ 올림 없음 — v4 큐가 끝난 뒤 시작 (2026-11-23)
cd "C:/Users/user/AppData/Local/Temp/claude/b55b069927d98669/5ed04639-c8f0-4837-9037-f61ff4814c8e/scratchpad"
until grep -q QUEUE_DONE bp_queue_v4.out 2>/dev/null; do sleep 60; done
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic TL=7200 ETA=1e-4 MU=1e4 MU_MAX=1e4 LAMU=1e4 SEED_ALL=0 NS=10
for g in 1 3 12 10 4 11 9; do
  GEO=$g julia -t 12,1 --project="C:/Users/user/robust_bilevel_network_games" ccgbp_v5.jl > bp5_g$g.log 2>&1
  echo "g$g EXIT $?"; grep -E "RESULT|시간:" bp5_g$g.log
done
echo QUEUE_DONE
