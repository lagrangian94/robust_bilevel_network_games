#!/usr/bin/env bash
# 2026-11-27: 나눠 인증하는 DD (dd_inc = :viol1, dd_objstop) 를 막힌 점 두 곳에서 900 s (md 11.24). 기본 판 900 s 는 dd_recmp_queue.sh 의 결과와 비교.
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/orcl_replay.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "^RESULT|DD 선별|^전선:" "$tag.log" | cut -c1-460; }
KD="(resp=true, dd=true, dd_maxw=25.0, dd_inc=:viol1, dd_objstop=true)"
run orep_g5_rc900_ddinc ABB_PATCH="$D/abb_dd_patch.jl" HUCAP=1 GEO=5 S=10 XSTR="{4:200;7:200;12:300}" TAU=-3151.60 TLIM=900 LOG=60 KW="$KD" TAG=_rc900_ddinc
run orep_g6_rc900_ddinc ABB_PATCH="$D/abb_dd_patch.jl" HUCAP=1 GEO=6 S=10 XSTR="{6:300;10:200;15:300}" TAU=-3618.10 TLIM=900 LOG=60 KW="$KD" TAG=_rc900_ddinc
echo "QUEUE DONE"
