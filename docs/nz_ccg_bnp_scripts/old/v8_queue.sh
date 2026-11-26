#!/usr/bin/env bash
# v8 (oracle 이어 풀기 + resp) 시험 (2026-11-24 오후): S=20 지리 1, S=10 지리 5·14. 분석 큐 after8_queue_b 가 끝난 뒤
cd "C:/Users/user/AppData/Local/Temp/claude/b55b069927d98669/5ed04639-c8f0-4837-9037-f61ff4814c8e/scratchpad"
until grep -q QUEUE_DONE "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/5ed04639-c8f0-4837-9037-f61ff4814c8e/tasks/bw9ucapy8.output" 2>/dev/null; do sleep 60; done
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic TL=7200
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { S=$2 GEO=$1 julia -t 12,1 --project="$P" "$D/ccgbp_v8.jl" > bp8_s$2_g$1.log 2>&1; echo "v8 S$2 g$1 EXIT $?"; grep -E "RESULT|시간:" bp8_s$2_g$1.log; grep -m2 "ERROR\|Exception" bp8_s$2_g$1.log; }
run 1 20
run 5 10
run 14 10
echo QUEUE_DONE
