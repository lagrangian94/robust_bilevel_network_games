#!/usr/bin/env bash
# 2026-11-26: 선별 DD 의 KKT-MIP 에 꼭짓점 범위 경계 (dd_sigbd, 기본 켬) 를 넣은 패치로 지리 5 막힌 점 재현. md 11.14.
#   v14 S=10 회귀 큐가 끝나기를 (지리 5 로그가 생겨 skip_g5.sh 가 멈출 때까지) 기다린 뒤 시작. 기본 판 비교는 logs/orep_g5_base.log (같은 점·목표·1,800 s).
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
until [ -f bp14r_s10_g5.log ]; do sleep 60; done; sleep 90
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/orcl_replay.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "^RESULT|DD 선별|^전선:" "$tag.log" | cut -c1-300; }
run orep_g5_ddv1_12 ABB_PATCH="$D/abb_dd_patch.jl" HUCAP=1 GEO=5 S=10 XSTR="{4:200;7:200;12:300}" TAU=-3151.60 TLIM=1800 LOG=60 KW="(resp=true, dd=true, dd_maxw=12.5)" TAG=_ddv1_12
run orep_g5_ddv1_25 ABB_PATCH="$D/abb_dd_patch.jl" HUCAP=1 GEO=5 S=10 XSTR="{4:200;7:200;12:300}" TAU=-3151.60 TLIM=1800 LOG=60 KW="(resp=true, dd=true, dd_maxw=25.0)" TAG=_ddv1_25
echo "QUEUE DONE"
