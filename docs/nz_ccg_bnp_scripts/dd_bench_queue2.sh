#!/usr/bin/env bash
# 2026-11-25 밤: dd_bench_queue.sh 의 기본 판 (orep_g5_base) 이 끝나기를 기다린 뒤 선별 DD 를 폭 문턱 12.5 로 3,600 s. md 11.13.
#  dd_maxw = 25 판은 1,800 s 에 DD 노드 502 개 중 비운 상자 0 (상한 −2,885). 실험실에서 쳐낸 전선 상자는 최대 폭 12.5 안팎.
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; mkdir -p "$SPD"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
for i in $(seq 1 60); do grep -qE "^전선:|^ERROR" orep_g5_base.log 2>/dev/null && break; sleep 60; done; sleep 20
grep -E "^RESULT|^전선:" orep_g5_base.log
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/orcl_replay.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "^RESULT|DD 선별|^전선:" "$tag.log"; }
run orep_g5_dd12 ABB_PATCH="$D/abb_dd_patch.jl" HUCAP=1 GEO=5 S=10 XSTR="{4:200;7:200;12:300}" TAU=-3151.60 TLIM=3600 LOG=60 KW="(resp=true, dd=true, dd_maxw=12.5)" TAG=_dd12
echo "QUEUE DONE"
