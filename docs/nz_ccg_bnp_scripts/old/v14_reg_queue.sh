#!/usr/bin/env bash
# 2026-11-25 밤: v14 (ccgbp_v14.jl) 의 S=10 회귀 — 지리 15 개 (13 은 v13 점검 519 s 와 1 단계가 같아 뺌), TL 3,600 s. md 11.14.
#  기본 판 변경은 사용자 결정 (memory). 이 큐는 판단 자료: v8 (기본) 이 Optimal 인 14 개를 v14 도 푸는지, 시간은 어떤지.
#  순서 = 이전 판 시간이 짧은 순 (결과를 빨리 보려고). dd_bench_queue2.sh 가 끝나기를 기다린 뒤 시작.
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; mkdir -p "$SPD"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
for i in $(seq 1 120); do grep -qE "^전선:|^ERROR" orep_g5_dd12.log 2>/dev/null && break; sleep 60; done; sleep 30
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/ccgbp_v14.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "^RESULT" "$tag.log" | cut -c1-260; }
for g in 7 1 3 4 9 10 11 12 2 8 15 14 6 5; do
    run bp14r_s10_g$g GEO=$g S=10 TL=3600
done
echo "QUEUE DONE"
