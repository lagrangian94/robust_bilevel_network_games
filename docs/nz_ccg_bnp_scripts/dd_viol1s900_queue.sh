#!/usr/bin/env bash
# 2026-11-27: 시나리오별 선별 DD (dd_inc = :viol1s, dd_objstop) 900 s, 지리 6 → 지리 5 막힌 점 (md 11.24 (7)). 비교 = dd_inc900_queue.sh 의 :viol1 판.
#   근거: 지리 6 은 :viol1 판 비용의 73% 가 선별 면 LP (DD 노드당 669 개). :viol1s 는 ζF 위반 큰 시나리오부터 선별하고 가지치기되면 남은 시나리오를 선별하지 않음.
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/orcl_replay.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "^RESULT|DD 선별|^전선:" "$tag.log" | cut -c1-460; }
KD="(resp=true, dd=true, dd_maxw=25.0, dd_inc=:viol1s, dd_objstop=true)"
run orep_g6_rc900_ddv1s ABB_PATCH="$D/abb_dd_patch.jl" HUCAP=1 GEO=6 S=10 XSTR="{6:300;10:200;15:300}" TAU=-3618.10 TLIM=900 LOG=60 KW="$KD" TAG=_rc900_ddv1s
run orep_g5_rc900_ddv1s ABB_PATCH="$D/abb_dd_patch.jl" HUCAP=1 GEO=5 S=10 XSTR="{4:200;7:200;12:300}" TAU=-3151.60 TLIM=900 LOG=60 KW="$KD" TAG=_rc900_ddv1s
echo "QUEUE DONE"
