#!/usr/bin/env bash
# 2026-11-28 (사용자 "감마=3으로 해서 서버에서 돌려볼까" → "시범부터 돌려"): cykim 에서 zero-sum γ = 3 시범, Benders (menu) 대 B&P (ccgbp_zs).
#   조건은 srv_zs_queue.sh (γ = 2) 와 같고 γ 만 3 (w = 0.5 γ median 도 1.5 배): S = 10, ε̂ = ε̃ = 0.3, β = 0.4, δ = 0.1, TOL 5e-3, TL 3,600 s,
#   네트워크 5 × 시드 42. 두 줄을 동시에 16 스레드씩.
#   Benders = ~/rbng_runs/run_nig_gamma_srv.jl (gen_bd_gamma.py 출력, NS_GAMMA = 3, 저장소 VI = sink 수정판).
#   B&P = ~/rbng_runs/ccgbp_zs_g_srv.jl (gen_zs.py 출력의 root 만 서버 경로, ZGAMMA = 3, oracle nz_alpha_bnb, DD 끔).
#   사용: ./srv_zs_g3_queue.sh bd | bp
mode=$1; R=~/rbng_runs; export GRB_LICENSE_FILE=$HOME/gurobi.lic; J=$HOME/.juliaup/bin/julia
echo "시작 $mode $(date +%H:%M)"
if [ "$mode" = bd ]; then
  cd ~/robust_bilevel_network_games/true_dro/factor_5 || exit 1
  NS_GAMMA=3 NS_S=10 NS_EPS=0.3 NS_BETA=0.4 NS_DELTA=0.1 NS_TOL=5e-3 NS_WALL=3600 NS_NWORKERS=14 NS_VARIANTS_S10=menu NS_SEEDS=42 \
    NS_LOGROOT=$R/zs_bd_g3 $J -t 16,1 $R/run_nig_gamma_srv.jl > $R/zs_bd_g3.stdout 2>&1
  echo "== bd EXIT $? $(date +%H:%M)"; grep -h "^RESULT" $R/zs_bd_g3/S_10/*.log | cut -c1-260
elif [ "$mode" = bp ]; then
  W=$R/zs_bp_g3; mkdir -p $W; cd $W || exit 1
  for net in grid5x5 polska abilene nobel_us sioux_falls; do
    tag=zs_${net}_s42_g3; ZGAMMA=3 NET=$net ZSEED=42 S=10 TL=3600 $J -t 16,1 $R/ccgbp_zs_g_srv.jl > $tag.log 2>&1
    echo "== $tag EXIT $? $(date +%H:%M)"; grep -E "^RESULT" $tag.log | cut -c1-300
  done
else echo "mode = bd | bp"; exit 1; fi
echo "QUEUE DONE $(date +%H:%M)"
