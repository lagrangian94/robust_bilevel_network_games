#!/usr/bin/env bash
# 2026-11-27: 지금 기본 판 (꼭짓점 재사용·평가 적응·ScaleFlag 2) 과 적응 DD (문턱 25) 다시 비교 (사용자 "네, 비교해", md 11.23 (6)).
#   막힌 점 두 곳 × 두 판, 각 900 s, 12 스레드 혼자. 예전 비교 (md 11.14 (8)~(10)) 는 처리량 개선 전 기본 판 기준이었음.
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/orcl_replay.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "^RESULT|DD 선별|^전선:" "$tag.log" | cut -c1-300; }
G5='HUCAP=1 GEO=5 S=10 XSTR={4:200;7:200;12:300} TAU=-3151.60 TLIM=900 LOG=60'
G6='HUCAP=1 GEO=6 S=10 XSTR={6:300;10:200;15:300} TAU=-3618.10 TLIM=900 LOG=60'
run orep_g5_rc900_base $G5 KW="(resp=true,)" TAG=_rc900_base
run orep_g5_rc900_dd $G5 ABB_PATCH="$D/abb_dd_patch.jl" KW="(resp=true, dd=true, dd_maxw=25.0)" TAG=_rc900_dd
run orep_g6_rc900_base $G6 KW="(resp=true,)" TAG=_rc900_base
run orep_g6_rc900_dd $G6 ABB_PATCH="$D/abb_dd_patch.jl" KW="(resp=true, dd=true, dd_maxw=25.0)" TAG=_rc900_dd
echo "QUEUE DONE"
