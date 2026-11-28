#!/usr/bin/env bash
# 2026-11-28 아침: cykim 에서 zero-sum Benders (menu) 를 min-cut VI sink 수정판으로 다시 (Polska·Abilene·Nobel-US × 시드 42·1~4, 저장소 판 VI 가 틀린 세 네트워크).
#   실행 사본 ~/rbng_runs/run_nig_vifix_srv.jl = gen_bd_vifix.py 출력 (저장소 run_nig_coupled_scaling.jl + zs_vi_sinkfix.jl). 조건은 srv_zs_queue.sh bd 와 같음.
#   두 줄을 동시에 16 스레드씩: 줄 a = polska,nobel_us / 줄 b = abilene.   사용: ./srv_bdfix_queue.sh a | b
lane=$1; R=~/rbng_runs; export GRB_LICENSE_FILE=$HOME/gurobi.lic; J=$HOME/.juliaup/bin/julia
case $lane in a) nets=polska,nobel_us;; b) nets=abilene;; *) echo "lane = a | b"; exit 1;; esac
echo "시작 $lane $(date +%H:%M)"
cd ~/robust_bilevel_network_games/true_dro/factor_5 || exit 1
NS_NETS=$nets NS_S=10 NS_EPS=0.3 NS_BETA=0.4 NS_DELTA=0.1 NS_TOL=5e-3 NS_WALL=3600 NS_NWORKERS=14 NS_VARIANTS_S10=menu NS_SEEDS=42,1,2,3,4 \
  NS_LOGROOT=$R/zs_bdfix $J -t 16,1 $R/run_nig_vifix_srv.jl > $R/zs_bdfix_$lane.stdout 2>&1
echo "== $lane EXIT $? $(date +%H:%M)"
echo "QUEUE DONE $(date +%H:%M)"
