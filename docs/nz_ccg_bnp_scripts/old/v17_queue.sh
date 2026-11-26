#!/usr/bin/env bash
# 2026-11-26: v17 (v16 + tabu 가격 생성기, DLH 2008 §4.3–4.4) 점검과 지리 12 S=10 (TL 3,600 s, DD 끔). md 11.18.
#   지리 13 TL 600 s = 오류 없이 도는지 점검. 지리 12 = v16 회귀 (bp16r_s10_g12, 2,740 s, 가격 MILP 116 번 2,258 s) 와 짝.
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/ccgbp_v17.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "^RESULT|v17:" "$tag.log" | cut -c1-300; }
run bp17chk_s10_g13 ABB_PATCH="" GEO=13 S=10 TL=600
run bp17_s10_g12 ABB_PATCH="" GEO=12 S=10 TL=3600
echo "QUEUE DONE"
