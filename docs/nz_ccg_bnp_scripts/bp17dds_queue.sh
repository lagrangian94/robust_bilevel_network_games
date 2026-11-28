#!/usr/bin/env bash
# 2026-11-28 (사용자 "돌려"): B&P 기본 판 (ccgbp.jl = v17) + DD oracle 에 강쌍대 cut (dd_sdcut, md 11.31) 을 켜고 S = 10 지리 5·6, TL 3,600 s.
#   비교: dd_sdcut 만 없던 판 logs/bp17ddf_s10_g{5,6}.log (bp17ddf_queue.sh, 지리 5 Optimal 1,852 s, 지리 6 Optimal 1,951 s).
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/ccgbp.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $? $(date +%H:%M)"; grep -E "^RESULT" "$tag.log" | cut -c1-320; }
echo "시작 $(date +%H:%M)"
for g in 5 6; do run bp17dds_s10_g$g ABB_PATCH="$D/abb_dd_patch.jl" ORCL_KW="(dd=true, dd_maxw=25.0, dd_sdcut=true)" GEO=$g S=10 TL=3600; done
echo "QUEUE DONE $(date +%H:%M)"
