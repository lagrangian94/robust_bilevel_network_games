#!/usr/bin/env bash
# 2026-11-25 v13 (v12 + oracle α 재사용 + 1 단계 반복 금지) 큐. md 11.12. v12 큐 (v12_queue.sh) 는 S=20 지리 1 만 돌리고 멈춤.
#  0) 점검: 지리 13 S=10 (v12 515 s, 0.500%) — RESULT 가 없으면 큐 중단
#  1) S=20 지리 1 (v11 0.526%, v12 결과는 md) 2) S=10 지리 5 (v11 20.2%) 3) S=10 지리 1 (같은 답 확인)
#  4) v11 + κ 수정만 지리 5 S=10 (κ 몫과 일정 규칙 몫을 나누기 위한 것, 맨 뒤)
cd "C:/Users/user/AppData/Local/Temp/claude/37ee3805408afe3f/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/$SCRIPT" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "RESULT|시간:|v12:|v13:" "$tag.log"; }
SCRIPT=ccgbp_v13.jl
run bp13_s10_g13 GEO=13 S=10 TL=1800
grep -q RESULT bp13_s10_g13.log || { echo "점검 실패 — 큐 중단"; exit 1; }
run bp13_s20_g1 GEO=1 S=20 TL=7200
run bp13_s10_g5 GEO=5 S=10 TL=7200
run bp13_s10_g1 GEO=1 S=10 TL=3600
SCRIPT=ccgbp_v11.jl
run bp11k_s10_g5 GEO=5 S=10 TL=7200
echo "QUEUE DONE"
