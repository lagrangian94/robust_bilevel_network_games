#!/usr/bin/env bash
# repo_fix_runs.sh (2026-11-28, 사용자 "다 고쳐"): 저장소 세 곳 수정 뒤 Polska s4 (S = 10) 를 저장소 판으로 한 번씩.
#   B&P: gen_zs.py 로 만든 ccgbp_zs.jl (블록 McCormick 은 이제 저장소 판). 지난 서버 판 {7;18} [21.2856, 21.3926], 61 s.
#   Benders: zs_bd_ablation.jl (VIFIX 0 = 저장소 VI, menu_mw 켬). 사본 VI 판 {7;18} 21.3786, 171 반복, 62 s.
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"; L="$D/logs/zs/repofix"; mkdir -p "$L"
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
red() { sed -E 's/registered to .*/registered to [가림]/' | grep -vE "WLS|LicenseID"; }
PYTHONIOENCODING=utf-8 python "$D/gen_zs.py" "$SPD/ccgbp_zs.jl" || exit 1
cd "$SPD" || exit 1
env NET=polska ZSEED=4 S=10 TL=1800 julia -t 12,1 --project="$P" "$SPD/ccgbp_zs.jl" 2>&1 | red > "$L/bp_polska_s4.log"; echo "== B&P EXIT ${PIPESTATUS[0]} $(date +%H:%M)"
grep -E "^RESULT" "$L/bp_polska_s4.log" | cut -c1-300
cd "$D" || exit 1
env NET=polska ZSEED=4 CONF=mincut:1 VIFIX=0 WALL=900 julia -t 12,1 zs_bd_ablation.jl 2>&1 | red > "$L/bd_polska_s4.log"; echo "== Benders EXIT ${PIPESTATUS[0]} $(date +%H:%M)"
tail -8 "$L/bd_polska_s4.log"
echo "DONE $(date +%H:%M)"
