#!/usr/bin/env bash
# 2026-11-27: B&P 기본 판 (ccgbp.jl = v17) 에 나눠 인증하는 DD oracle (abb_dd_patch.jl, dd_inc = :viol1, dd_objstop) 을 켜고 S = 10 지리 5·6, TL 3,600 s (사용자 "비앤피로 확인", md 11.24 (6)).
#   비교: 지리 6 은 v16 DD 끔 (logs/bp16n_s10_g6.log, Optimal 0.469%, 3,198 s), 지리 5 는 한 번도 안 풀림 (v13 TimeLimit 5.144%, logs/bp13_s10_g5.log).
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/ccgbp.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "^RESULT" "$tag.log" | cut -c1-320; }
for g in 5 6; do run bp17dd_s10_g$g ABB_PATCH="$D/abb_dd_patch.jl" ORCL_KW="(dd=true, dd_maxw=25.0)" GEO=$g S=10 TL=3600; done
echo "QUEUE DONE"
