#!/usr/bin/env bash
# 2026-11-25 v11 (v10 루트 정확 CG 하한 + oracle HUCAP): λ 큐 (bj1uu559e) 뒤. v10 큐를 대체 (두 변경은 독립이라 한 번에)
#  지리 1 S=10 (점검: 같은 답이 나오는지) → 지리 5 S=10 → 지리 1 S=20
cd "C:/Users/user/AppData/Local/Temp/claude/b55b069927d98669/5ed04639-c8f0-4837-9037-f61ff4814c8e/scratchpad"
until grep -q "QUEUE_DONE" "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/5ed04639-c8f0-4837-9037-f61ff4814c8e/tasks/bj1uu559e.output" 2>/dev/null; do sleep 60; done
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { GEO=$1 S=$2 TL=$3 julia -t 12,1 --project="$P" "$D/ccgbp_v11.jl" > "bp11_s$2_g$1.log" 2>&1; echo "v11 S$2 g$1 EXIT $?"; grep -E "루트 정확|RESULT|시간:" "bp11_s$2_g$1.log"; grep -m3 "ERROR\|Exception" "bp11_s$2_g$1.log"; }
run 1 10 3600
run 5 10 7200
run 1 20 7200
echo QUEUE_DONE
