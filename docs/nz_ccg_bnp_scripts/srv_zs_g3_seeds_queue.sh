#!/usr/bin/env bash
# 2026-11-28 (사용자 "dd kkt-mip 강쌍대컷 추가해서 시드넓혀보자"): zero-sum γ = 3 을 시드 1~4 로 넓힘 (시범 시드 42 와 합쳐 25 개). 조건은 srv_zs_g3_queue.sh 와 같음.
#   S = 10, ε̂ = ε̃ = 0.3, β = 0.4, δ = 0.1, TOL 5e-3, TL 3,600 s, 두 줄을 동시에 16 스레드씩.
#   B&P oracle 은 시범과 같은 nz_alpha_bnb (plain). 강쌍대 cut (abb_dd_patch.jl dd_sdcut) 은 범위 MIP 판 oracle 에만 붙는데, Abilene s42 γ = 2 {5;11} 300 s 에서
#   plain 0.543% < 범위 MIP + 강쌍대 0.618% < 범위 MIP 0.667% (zs_sdcut_check.jl, logs/zs/zs_sdcut/) 라 쓰지 않음.
#   Benders 반복 상한은 NS_MAXITER = 10⁶ (시범 때 띄울 때 준 값을 여기 적음).
#   사용: ./srv_zs_g3_seeds_queue.sh bd | bp
mode=$1; R=~/rbng_runs; export GRB_LICENSE_FILE=$HOME/gurobi.lic; J=$HOME/.juliaup/bin/julia
echo "시작 $mode $(date +%H:%M)"
if [ "$mode" = bd ]; then
  cd ~/robust_bilevel_network_games/true_dro/factor_5 || exit 1
  NS_GAMMA=3 NS_MAXITER=1000000 NS_S=10 NS_EPS=0.3 NS_BETA=0.4 NS_DELTA=0.1 NS_TOL=5e-3 NS_WALL=3600 NS_NWORKERS=14 NS_VARIANTS_S10=menu NS_SEEDS=1,2,3,4 \
    NS_LOGROOT=$R/zs_bd_g3s $J -t 16,1 $R/run_nig_gamma_srv.jl > $R/zs_bd_g3s.stdout 2>&1
  echo "== bd EXIT $? $(date +%H:%M)"; grep -h "^RESULT" $R/zs_bd_g3s/S_10/*.log | cut -c1-260
elif [ "$mode" = bp ]; then
  W=$R/zs_bp_g3s; mkdir -p $W; cd $W || exit 1
  for sd in 1 2 3 4; do
    for net in grid5x5 polska abilene nobel_us sioux_falls; do
      tag=zs_${net}_s${sd}_g3; ZGAMMA=3 NET=$net ZSEED=$sd S=10 TL=3600 $J -t 16,1 $R/ccgbp_zs_g_srv.jl > $tag.log 2>&1
      echo "== $tag EXIT $? $(date +%H:%M)"; grep -E "^RESULT" $tag.log | cut -c1-300
    done
  done
else echo "mode = bd | bp"; exit 1; fi
echo "QUEUE DONE $(date +%H:%M)"
