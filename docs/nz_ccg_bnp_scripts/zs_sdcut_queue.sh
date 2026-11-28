#!/bin/bash
# zs_sdcut_queue.sh (2026-11-28, 서버): zero-sum 고정 x̄ 에서 강쌍대 cut (dd_sdcut) 시험. zs_sdcut_check.jl 참고.
#   (1) Abilene s42 γ = 2 {5;11}, 모드 4 개 × 300 s   (2) 같은 점 mip_sd 를 행 없는 판과 대조 (ABB_SD_CHECK 5, 허용오차 1e-9) 120 s
R=$HOME/rbng_runs; J=$HOME/.juliaup/bin/julia; RT=$HOME/robust_bilevel_network_games
export GRB_LICENSE_FILE=$HOME/gurobi.lic MOSEK_NUM_THREADS=1
mkdir -p $R/zs_sdcut
ROOT=$RT NET=abilene ZSEED=42 ZGAMMA=2 XS=5,11 TLIM=300 MODES=plain,mip,mip_sd,mipdd_sd $J -t 16,1 $R/zs_sdcut_check.jl > $R/zs_sdcut/abl_s42_g2_x5_11.log 2>&1
ROOT=$RT NET=abilene ZSEED=42 ZGAMMA=2 XS=5,11 TLIM=120 MODES=mip_sd ABB_SD_CHECK=5 ABB_SD_TIGHT=1 $J -t 16,1 $R/zs_sdcut_check.jl > $R/zs_sdcut/abl_s42_g2_x5_11_chk.log 2>&1
echo DONE > $R/zs_sdcut/DONE
