#!/usr/bin/env bash
# 2026-11-25 밤: α-B&B 선별 DD 패치 (abb_dd_patch.jl, dd_mode = :screen) 의 oracle 재현 벤치마크. md 11.13.
#  지리 5 S=10 의 막힌 점 {4:200;7:200;12:300} (v13 에서 4,500 s 에 상한이 목표 59 위), 목표 −3,151.60 (v13 의 τ⁺), HUCAP, resp.
#  1) DD 판 1,800 s  2) 기본 판 1,800 s (같은 기계 상태에서 비교)  3) S=20 지리 1 의 대등한 점 {10:300;11:200;15:200} (v13 cinc, 인증 느린 점) DD 판 1,800 s
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; mkdir -p "$SPD"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/orcl_replay.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "^RESULT|DD 선별|^전선:" "$tag.log"; }
PATCH="$D/abb_dd_patch.jl"
run orep_g5_dd   ABB_PATCH=$PATCH HUCAP=1 GEO=5 S=10 XSTR="{4:200;7:200;12:300}" TAU=-3151.60 TLIM=1800 LOG=60 KW="(resp=true, dd=true)" TAG=_dd
run orep_g5_base                  HUCAP=1 GEO=5 S=10 XSTR="{4:200;7:200;12:300}" TAU=-3151.60 TLIM=1800 LOG=60 KW="(resp=true,)" TAG=_base
run orep_s20_tied_dd ABB_PATCH=$PATCH HUCAP=1 GEO=1 S=20 XSTR="{10:300;11:200;15:200}" TAU=-3397.58 TLIM=1800 LOG=60 KW="(resp=true, dd=true)" TAG=_s20dd
echo "QUEUE DONE"
