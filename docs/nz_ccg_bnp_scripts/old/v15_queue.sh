#!/usr/bin/env bash
# 2026-11-26: v15 (적응 선별 DD oracle + 1 단계 결론 없는 oracle 이어 풀기) 를 v14 회귀에서 TimeLimit 인 지리 15·6 (S=10, TL 3,600 s) 에. md 11.16.
#   v14: 지리 15 TimeLimit 0.877% (1 단계 belief 3 개), 지리 6 TimeLimit 0.502% (막힌 점 인증 0.42 부족). 지리 5 는 결과 보고 결정.
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/ccgbp_v15.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "^RESULT" "$tag.log" | cut -c1-300; }
for g in 15 6; do run bp15_s10_g$g GEO=$g S=10 TL=3600; done
echo "QUEUE DONE"
