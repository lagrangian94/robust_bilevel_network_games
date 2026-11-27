#!/usr/bin/env bash
# 2026-11-28 새벽: cykim 에서 zero-sum 네트워크 차단, Benders (menu) 대 B&P (ccgbp_zs) 를 같은 조건으로 (사용자 "제로섬도 ccg bnp로 풀어보고").
#   두 큐를 동시에 16 스레드씩. S = 10, ε̂ = ε̃ = 0.3, β = 0.4, δ = 0.1, TOL 5e-3, TL 3,600 s, 네트워크 5 × 시드 42·1~4.
#   Benders = true_dro/factor_5/run_nig_coupled_scaling.jl (NS_VARIANTS_S10=menu, λᵁ 10 고정, α-B&B worker 14).
#   B&P = gen_zs.py 로 만든 ccgbp_zs.jl 의 root 만 서버 경로로 바꾼 ~/rbng_runs/ccgbp_zs_srv.jl (λᵁ 1e4, oracle = nz_alpha_bnb (ZORCL plain), DD 끔). 인스턴스 지문 zs_fp.jl 로 이 컴퓨터와 같음 확인.
#   사용: ./srv_zs_queue.sh bd | bp
mode=$1; R=~/rbng_runs; export GRB_LICENSE_FILE=$HOME/gurobi.lic; J=$HOME/.juliaup/bin/julia
echo "시작 $mode $(date +%H:%M)"
if [ "$mode" = bd ]; then
  cd ~/robust_bilevel_network_games/true_dro/factor_5 || exit 1
  NS_S=10 NS_EPS=0.3 NS_BETA=0.4 NS_DELTA=0.1 NS_TOL=5e-3 NS_WALL=3600 NS_NWORKERS=14 NS_VARIANTS_S10=menu NS_SEEDS=42,1,2,3,4 \
    NS_LOGROOT=$R/zs_bd $J -t 16,1 run_nig_coupled_scaling.jl > $R/zs_bd.stdout 2>&1
  echo "== bd EXIT $? $(date +%H:%M)"; grep -h "^RESULT" $R/zs_bd/S_10/*.log | cut -c1-260
elif [ "$mode" = bp ]; then
  W=$R/zs_bp; mkdir -p $W; cd $W || exit 1
  for sd in 42 1 2 3 4; do for net in grid5x5 polska abilene nobel_us sioux_falls; do
    tag=zs_${net}_s${sd}; NET=$net ZSEED=$sd S=10 TL=3600 $J -t 16,1 $R/ccgbp_zs_srv.jl > $tag.log 2>&1
    echo "== $tag EXIT $? $(date +%H:%M)"; grep -E "^RESULT" $tag.log | cut -c1-300
  done; done
else echo "mode = bd | bp"; exit 1; fi
echo "QUEUE DONE $(date +%H:%M)"
