#!/usr/bin/env bash
# 2026-11-25 v12 (예산 분할 + 저장소 평가기 κ 단계) 큐. md 11.11.
#  1) S=20 지리 1 (v11 2 시간 0.526%, 4 시간 판 0.952%) 2) S=10 지리 5 (v11 20.2%) 3) S=10 지리 1 (같은 답 확인)
#  4) v11 + κ 수정만 지리 5 S=10 (효과를 κ 와 예산 분할로 나누기 위한 것, 맨 뒤)
cd "C:/Users/user/AppData/Local/Temp/claude/37ee3805408afe3f/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/$SCRIPT" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "RESULT|시간:|v12:" "$tag.log"; }
SCRIPT=ccgbp_v12.jl
run bp12_s20_g1 GEO=1 S=20 TL=7200
run bp12_s10_g5 GEO=5 S=10 TL=7200
run bp12_s10_g1 GEO=1 S=10 TL=3600
SCRIPT=ccgbp_v11.jl
run bp11k_s10_g5 GEO=5 S=10 TL=7200
echo "QUEUE DONE"
