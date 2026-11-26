#!/usr/bin/env bash
# 2026-11-25 v11 S=20 지리 1 을 4 시간 (7,200 s 에 간격 0.526%, 허용 0.5% 직전에서 끝남)
cd "C:/Users/user/AppData/Local/Temp/claude/b55b069927d98669/5ed04639-c8f0-4837-9037-f61ff4814c8e/scratchpad"
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
GEO=1 S=20 TL=14400 julia -t 12,1 --project="$P" "$D/ccgbp_v11.jl" > bp11_s20_g1_long.log 2>&1; echo "EXIT $?"; grep -E "루트 정확|RESULT|시간:" bp11_s20_g1_long.log
