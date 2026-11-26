#!/usr/bin/env bash
# 2026-11-26: v16 (DD 끔 = 저장소 α-B&B) 의 S=10 회귀 중 경로가 v14 와 달라질 수 있는 지리만 (TL 3,600 s). md 11.17.
#   v14 회귀 (logs/bp14r_s10_g*.log) 에서 1 단계 마지막 oracle 이 인증으로 끝난 지리 1·2·3·4·9·10·11 은 v16 의 이어 풀기가 걸리지 않아 v14 와 같은 경로.
#   결론 없이 끝난 지리 7·8·12·14 만 v16 에서 이어 풂 (v14: 854 · 1,204 · 1,109 · 2,438 s, 모두 Optimal). 지리 6·15 는 v16_queue.sh.
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/ccgbp_v16.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "^RESULT" "$tag.log" | cut -c1-300; }
for g in 7 12 8 14; do run bp16r_s10_g$g ABB_PATCH="" GEO=$g S=10 TL=3600; done
echo "QUEUE DONE"
