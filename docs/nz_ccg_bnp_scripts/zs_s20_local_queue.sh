#!/usr/bin/env bash
# 2026-11-28 새벽: zero-sum S = 20 시드 42 다섯 네트워크, B&P (ccgbp_zs, TL 1800) 다음 Benders (menu, TOL 5e-3, TL 1800) 를 이 컴퓨터에서 차례로 (같은 기계 비교).
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
PYTHONIOENCODING=utf-8 python "$D/gen_zs.py" "$SPD/ccgbp_zs.jl" || exit 1
echo "시작 $(date +%H:%M)"
for net in grid5x5 polska abilene nobel_us sioux_falls; do
  tag=zs20_${net}_s42; NET=$net ZSEED=42 S=20 TL=1800 julia -t 12,1 --project="$P" "$SPD/ccgbp_zs.jl" > $tag.log 2>&1
  echo "== $tag EXIT $? $(date +%H:%M)"; grep -E "^RESULT" $tag.log | cut -c1-300
done
cd "$P/true_dro/factor_5" || exit 1
NS_S=20 NS_EPS=0.3 NS_BETA=0.4 NS_DELTA=0.1 NS_TOL=5e-3 NS_WALL=1800 NS_NWORKERS=10 NS_VARIANTS_LARGE=menu NS_SEEDS=42 \
  NS_LOGROOT="$SPD/zs20_bd" julia -t 12,1 run_nig_coupled_scaling.jl > "$SPD/zs20_bd.stdout" 2>&1
echo "== bd EXIT $? $(date +%H:%M)"; grep -h "^RESULT" "$SPD"/zs20_bd/S_20/*.log | cut -c1-260
echo "QUEUE DONE $(date +%H:%M)"
