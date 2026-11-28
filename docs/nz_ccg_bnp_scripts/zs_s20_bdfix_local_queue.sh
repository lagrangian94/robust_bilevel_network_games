#!/usr/bin/env bash
# 2026-11-28 아침: zero-sum S = 20 시드 42 의 Benders (menu, TOL 5e-3, WALL 1800) 를 min-cut VI sink 수정판으로 다시 (이 컴퓨터, zs_s20_local_queue.sh 와 같은 인자).
#   sink 버그가 걸리는 세 망 (Polska·Nobel-US·Abilene) 만. grid5x5·Sioux Falls 는 t 가 마지막 행이라 옛 결과 그대로.
#   실행 사본은 gen_bd_vifix.py 가 만든다 (저장소 파일은 안 고침).
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
PYTHONIOENCODING=utf-8 python "$D/gen_bd_vifix.py" "$P" "$D/zs_vi_sinkfix.jl" "$SPD/run_nig_vifix.jl" || exit 1
echo "시작 $(date +%H:%M)"
cd "$P/true_dro/factor_5" || exit 1
NS_NETS=polska,nobel_us,abilene NS_S=20 NS_EPS=0.3 NS_BETA=0.4 NS_DELTA=0.1 NS_TOL=5e-3 NS_WALL=1800 NS_NWORKERS=10 NS_VARIANTS_LARGE=menu NS_SEEDS=42 \
  NS_LOGROOT="$SPD/zs20_bdfix" julia -t 12,1 "$SPD/run_nig_vifix.jl" > "$SPD/zs20_bdfix.stdout" 2>&1
echo "== bdfix EXIT $? $(date +%H:%M)"; grep -h "^RESULT" "$SPD"/zs20_bdfix/S_20/*.log | cut -c1-260
echo "QUEUE DONE $(date +%H:%M)"
