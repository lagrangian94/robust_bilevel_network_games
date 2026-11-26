#!/usr/bin/env bash
# 2026-11-26: 선별 DD 의 성공률 조절 (dd_adapt, 기본 warm 200·minrate 0.25·probe 10) 확인 (md 11.14 (10)).
#   지리 6 막힌 점 1,800 s: DD-V1 25 (적응 없음) 는 성공률 7% 로 기본 판보다 느렸음 → 기본 판 수준으로 돌아오는지.
#   지리 5 막힌 점 900 s: 성공률 77% 라 적응이 거의 안 걸려야 함 → DD-V1 25 궤적 (730 s −2,861.5, 850 s −2,893.1) 과 같은지.
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/orcl_replay.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "^RESULT|DD 선별|^전선:" "$tag.log" | cut -c1-300; }
run orep_g6_ddad_25 ABB_PATCH="$D/abb_dd_patch.jl" HUCAP=1 GEO=6 S=10 XSTR="{6:300;10:200;15:300}" TAU=-3618.10 TLIM=1800 LOG=60 KW="(resp=true, dd=true, dd_maxw=25.0)" TAG=_g6_ddad_25
run orep_g5_ddad_25 ABB_PATCH="$D/abb_dd_patch.jl" HUCAP=1 GEO=5 S=10 XSTR="{4:200;7:200;12:300}" TAU=-3151.60 TLIM=900 LOG=60 KW="(resp=true, dd=true, dd_maxw=25.0)" TAG=_g5_ddad_25
echo "QUEUE DONE"
