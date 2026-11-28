#!/usr/bin/env bash
# 2026-11-28 (사용자 승인 "ㅇㅇㅇㅇ", md 11.33): 저장소로 옮긴 선별 DD (+ 강쌍대 cut) 를 고정 x̄ oracle 로 확인 (서버 cykim, 저장소 사본 worktree ~/rbng_runs/wt_dd).
#   점 = 13 지리 회귀 (DD 끔 판, logs/reg13/srv13nodd_s10_g*.log) 의 x* 에서 B&P 가 마지막으로 부른 oracle 의 목표 + 지리 5 막힌 점.
#   두 구성을 동시에 8 스레드씩 (zero-sum 작업 16 스레드와 합쳐 32): A = DD 끔 (resp = true), B = DD 켬 (resp = true, dd = true → 저장소 기본 dd_* = 강쌍대 포함).
#   300 s 상한, HUCAP 1, λᵁ 1e4, gap 1e-3 (orcl_replay 기본). 사용: bash srv_dd14_queue.sh A | B
#   2026-11-28 밤 (사용자 "어려운점을 더 재바", md 11.34): 환경 변수 GEOS (돌릴 지리, 공백 구분, 그 순서로; 없으면 14 점 전부) 와 TLIM (상한, 기본 300).
#     300 이 아니면 로그 이름에 _t$TLIM 을 붙인다. 예: GEOS="1 8 15 6" TLIM=1800 bash srv_dd14_queue.sh A
set -u
C=$1; W=$HOME/rbng_runs/wt_dd; R=$HOME/rbng_runs; OUT=$R/dd14; mkdir -p "$OUT"; cd "$OUT" || exit 1
TL=${TLIM:-300}; SFX=""; [ "$TL" = "300" ] || SFX=_t$TL
export GRB_LICENSE_FILE=$HOME/gurobi.lic
#   구성 G (2026-11-28 밤, 사용자 "구로비 솔바로 푼 거랑도 비교"): 같은 점·정의역의 Ω 를 Gurobi NonConvex 로 (orcl_replay 의 SOLVER=grb, Threads = 8)
SOLV=abb
case $C in A) KW="(resp=true, dd=false)";;   # 2026-11-29: 함수 기본이 DD 켬으로 바뀌어 A 는 dd=false 를 명시 (돌린 때는 기본 끔)
 B) KW="(resp=true, dd=true)";; G) KW="NamedTuple()"; SOLV=grb;; *) echo "구성 A | B | G"; exit 1;; esac
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
if [ -n "${GEOS:-}" ]; then
    SEL=()
    for g in $GEOS; do
        hit=""; for p in "${PTS[@]}"; do [ "${p%% *}" = "$g" ] && hit=$p; done
        [ -n "$hit" ] || { echo "지리 $g 는 점 목록에 없음"; exit 1; }
        SEL+=("$hit")
    done
    PTS=("${SEL[@]}")
fi
echo "시작 $C $(date +%H:%M) KW=$KW TLIM=$TL 지리 ${GEOS:-전부}"
for p in "${PTS[@]}"; do
    read -r g x tau <<< "$p"
    tag=dd14${C}_g$g$SFX
    env SOLVER=$SOLV HUCAP=1 GEO=$g S=10 XSTR="$x" TAU=$tau TLIM=$TL LOG=60 KW="$KW" TAG=_$tag $HOME/.juliaup/bin/julia -t 8,1 $R/orcl_replay_wt.jl > $tag.log 2>&1
    echo "== $tag EXIT $? $(date +%H:%M) $(grep -E '^RESULT' $tag.log | cut -c1-200)"
done
echo "QUEUE DONE $C $(date +%H:%M)"
