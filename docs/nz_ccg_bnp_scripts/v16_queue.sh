#!/usr/bin/env bash
# 2026-11-26: v16 (v15 + 1 단계 총량 몫 PH1_SHARE 0.3·TL = 1,080 s) 를 지리 15·6 S=10 (TL 3,600 s) 에. md 11.17.
#   각 판은 v15 의 같은 oracle 과 짝: 지리 15 = 적응 DD (v15 Optimal 1,781 s, 1 단계 1,442 s), 지리 6 = DD 끔 (v15 1 단계 2,539 s → 5.661%).
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/ccgbp_v16.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "^RESULT" "$tag.log" | cut -c1-300; }
run bp16_s10_g15 GEO=15 S=10 TL=3600
run bp16n_s10_g6 ABB_PATCH="" GEO=6 S=10 TL=3600
echo "QUEUE DONE"
