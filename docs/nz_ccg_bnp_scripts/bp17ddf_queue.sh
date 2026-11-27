#!/usr/bin/env bash
# 2026-11-27: B&P 기본 판 (ccgbp.jl = v17) + DD oracle 회귀, 지금 패치 기본값 (dd_inc = :viol1, dd_objstop, dd_fcache = 켬) 으로 S = 10 지리 5·6, TL 3,600 s (사용자 "둘다").
#   비교: 같은 설정에서 dd_fcache 만 없던 판 logs/bp17dd_s10_g{5,6}.log (지리 5 Optimal 1,853 s, 지리 6 TimeLimit 0.518%).
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/ccgbp.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $? $(date +%H:%M)"; grep -E "^RESULT" "$tag.log" | cut -c1-320; }
echo "시작 $(date +%H:%M)"
for g in 5 6; do run bp17ddf_s10_g$g ABB_PATCH="$D/abb_dd_patch.jl" ORCL_KW="(dd=true, dd_maxw=25.0)" GEO=$g S=10 TL=3600; done
echo "QUEUE DONE $(date +%H:%M)"
