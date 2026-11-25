#!/usr/bin/env bash
# 2026-11-26: DD-V1 (dd_sigbd, 문턱 25) 를 지리 6 S=10 의 막힌 점 {6:300;10:200;15:300} 에 (md 11.14 (8)).
#   v14 회귀 (logs/bp14r_s10_g6.log) 에서 이 점의 oracle 누적 상한: 300 s −3,556.57, 1,201 s −3,612.13, 2,101 s −3,617.13, 2,266 s −3,617.68,
#   목표 −3,618.10 (0.42 못 미쳐 TimeLimit 0.502%). 목표에 닿으면 α-B&B 가 멈춘다. 기본 판 재현은 돌리지 않음 (위 누적값이 같은 호출 인자).
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/orcl_replay.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "^RESULT|DD 선별|^전선:" "$tag.log" | cut -c1-300; }
run orep_g6_ddv1_25 ABB_PATCH="$D/abb_dd_patch.jl" HUCAP=1 GEO=6 S=10 XSTR="{6:300;10:200;15:300}" TAU=-3618.10 TLIM=1800 LOG=60 KW="(resp=true, dd=true, dd_maxw=25.0)" TAG=_g6_ddv1_25
echo "QUEUE DONE"
