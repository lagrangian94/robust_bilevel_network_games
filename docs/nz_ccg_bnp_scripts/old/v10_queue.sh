#!/usr/bin/env bash
# 2026-11-25 v10 (v9 + 루트 정확 CG 하한): λ 큐 (bj1uu559e) 뒤 지리 5 S=10, 지리 1 S=20 (TimeLimit 에서도 LB 보고되는지)
cd "C:/Users/user/AppData/Local/Temp/claude/b55b069927d98669/5ed04639-c8f0-4837-9037-f61ff4814c8e/scratchpad"
until grep -q "QUEUE_DONE" "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/5ed04639-c8f0-4837-9037-f61ff4814c8e/tasks/bj1uu559e.output" 2>/dev/null; do sleep 60; done
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
GEO=5 S=10 TL=7200 julia -t 12,1 --project="$P" "$D/ccgbp_v10.jl" > bp10_s10_g5.log 2>&1; echo "v10 g5 EXIT $?"; grep -E "루트 정확|RESULT|시간:" bp10_s10_g5.log
GEO=1 S=20 TL=7200 julia -t 12,1 --project="$P" "$D/ccgbp_v10.jl" > bp10_s20_g1.log 2>&1; echo "v10 S20 g1 EXIT $?"; grep -E "루트 정확|RESULT|시간:" bp10_s20_g1.log
echo QUEUE_DONE
