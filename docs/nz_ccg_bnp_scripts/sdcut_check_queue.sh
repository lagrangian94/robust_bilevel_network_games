#!/usr/bin/env bash
# 2026-11-28 (사용자 "넣고 비교해"): dd_sdcut (KKT-MIP 강쌍대 McCormick, md 11.31) 나무 안 검증 판.
#   지리 5 막힌 점 300 s, 허용오차 1e-9 (ABB_SD_TIGHT), DD MIP 5 개마다 강쌍대 행 없이 먼저 풀어 그 실행가능 값과 행 넣은 판의 한계를 대조 (ABB_SD_CHECK).
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/orcl_replay.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "^RESULT|DD 선별|^전선:|dd_sdcut" "$tag.log" | cut -c1-600; }
KD="(resp=true, dd=true, dd_maxw=25.0, dd_inc=:viol1, dd_objstop=true, dd_fcache=true, dd_sdcut=true)"
run orep_g5_sdchk300 ABB_PATCH="$D/abb_dd_patch.jl" ABB_SD_CHECK=5 ABB_SD_TIGHT=1 HUCAP=1 GEO=5 S=10 XSTR="{4:200;7:200;12:300}" TAU=-3151.60 TLIM=300 LOG=60 KW="$KD" TAG=_sdchk300
echo "QUEUE DONE"
