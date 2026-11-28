#!/usr/bin/env bash
# 2026-11-28 (사용자 승인 "ㅇㅇㅇㅇ", md 11.33): 저장소로 옮긴 선별 DD (+ 강쌍대 cut) 를 고정 x̄ oracle 로 확인 (서버 cykim, 저장소 사본 worktree ~/rbng_runs/wt_dd).
#   점 = 13 지리 회귀 (DD 끔 판, logs/reg13/srv13nodd_s10_g*.log) 의 x* 에서 B&P 가 마지막으로 부른 oracle 의 목표 + 지리 5 막힌 점.
#   두 구성을 동시에 8 스레드씩 (zero-sum 작업 16 스레드와 합쳐 32): A = DD 끔 (resp = true), B = DD 켬 (resp = true, dd = true → 저장소 기본 dd_* = 강쌍대 포함).
#   300 s 상한, HUCAP 1, λᵁ 1e4, gap 1e-3 (orcl_replay 기본). 사용: bash srv_dd14_queue.sh A | B
set -u
C=$1; W=$HOME/rbng_runs/wt_dd; R=$HOME/rbng_runs; OUT=$R/dd14; mkdir -p "$OUT"; cd "$OUT" || exit 1
export GRB_LICENSE_FILE=$HOME/gurobi.lic
case $C in A) KW="(resp=true,)";; B) KW="(resp=true, dd=true)";; *) echo "구성 A | B"; exit 1;; esac
PTS=(
"9 {5:300;12:200} -2329.76"
"10 {4:200;13:300} -2390.37"
"4 {10:200;15:400} -2852.48"
"11 {5:300;9:300} -2886.69"
"7 {4:200;5:300;9:200} -3201.80"
"12 {8:200;14:400} -3137.10"
"3 {8:300;13:200;15:300} -3456.02"
"1 {3:300;10:200;15:300} -3673.31"
"2 {4:300;9:200;13:200} -3059.59"
"8 {7:300;10:200;13:200} -3206.93"
"15 {8:200;10:300;14:200} -3166.91"
"14 {8:200;9:200;13:200} -2835.52"
"6 {6:300;10:200;15:300} -3618.10"
"5 {4:200;7:200;12:300} -3151.60"
)
echo "시작 $C $(date +%H:%M) KW=$KW"
for p in "${PTS[@]}"; do
    read -r g x tau <<< "$p"
    tag=dd14${C}_g$g
    env HUCAP=1 GEO=$g S=10 XSTR="$x" TAU=$tau TLIM=300 LOG=60 KW="$KW" TAG=_$tag $HOME/.juliaup/bin/julia -t 8,1 $R/orcl_replay_wt.jl > $tag.log 2>&1
    echo "== $tag EXIT $? $(date +%H:%M) $(grep -E '^RESULT' $tag.log | cut -c1-200)"
done
echo "QUEUE DONE $C $(date +%H:%M)"
