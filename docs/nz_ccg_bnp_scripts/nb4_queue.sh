#!/usr/bin/env bash
# 2026-11-26 밤: N_B = 4 (|X| = 66,379, N_B = 3 의 7.8 배) 에서 v17 (tabu 가격)·v16 비교. md 11.19.
#   사용자 "작은 X 를 훑는 게 자꾸 문제가 되면 N_B 를 늘려 tabu 를 돌려 보자, PADM 도 궁금". 지리 12 (11.18 v17 이 크게 이긴 곳) 와 지리 11 (5.4 절 v6 N_B 4 1,734 s).
#   본 판은 TL 3,600 s, DD 끔 (저장소 α-B&B).
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local v=$1 tag=$2; shift 2; env "$@" julia -t 12,1 --project="$P" "$D/ccgbp_$v.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "^RESULT|v1[78]:" "$tag.log" | cut -c1-300; grep -E "^\s+시간:" "$tag.log" | cut -c1-300; }
# (v18 점검은 끝남: bp18chk_s10_g13, bp18chk2_s10_g13)
# v18 점검 두 판 (bp18chk_s10_g13, bp18chk2_s10_g13) 에서 PADM 이 음의 감축 비용 열을 하나도 못 내서 N_B = 4 본 판에서는 v18 을 뺌 (md 11.19).
for g in 12 11; do
  run v17 bp17_nb4_g$g ABB_PATCH="" GEO=$g S=10 NB=4 TL=3600
  run v16 bp16_nb4_g$g ABB_PATCH="" GEO=$g S=10 NB=4 TL=3600
done
echo "QUEUE DONE"
