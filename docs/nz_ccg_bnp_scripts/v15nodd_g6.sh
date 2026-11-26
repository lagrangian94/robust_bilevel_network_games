#!/usr/bin/env bash
# 2026-11-26: v15 에서 DD 를 끈 판 (ABB_PATCH="", 저장소 α-B&B, 1 단계 이어 풀기만) 을 지리 6 S=10 에 (TL 3,600 s). md 11.16.
#   v15 (적응 DD) 지리 6: 막힌 점 누적 2,376 s 에 상한 −3,616.30 → TimeLimit 0.532%. 기본 oracle 은 v14 에서 누적 2,266 s 에 −3,617.68 (목표 −3,618.10).
#   v15 의 1 단계 이어 풀기는 막힌 점에 시간을 더 주므로, 기본 oracle 이면 닫히는지 본다. 지리 15 를 푼 것도 이어 풀기 (DD 무관).
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
env ABB_PATCH="" GEO=6 S=10 TL=3600 julia -t 12,1 --project="$P" "$D/ccgbp_v15.jl" > bp15n_s10_g6.log 2>&1; echo "== bp15n_s10_g6 EXIT $?"; grep -E "^RESULT" bp15n_s10_g6.log | cut -c1-300
