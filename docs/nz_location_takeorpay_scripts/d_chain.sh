#!/bin/bash
# delta sweep (ledger 4.34): waits for the radius 0.5 run (4.33), then Coupled at radius 0.3 with delta 0, 0.3, 0.6 -> merge -> evaluation (rho 1, 0.5, 0, 0.25, 0.75). 150 runs each, 11 at a time. Log d_run.log
cd "/c/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
G="/c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo"
until [ -f r5_run.log ] && grep -q "R5_ALL_DONE\|R5_ABORT" r5_run.log; do sleep 120; done
echo "START $(date +%H:%M)" > d_run.log
echo "== Cd0.0 $(date +%H:%M)" >> d_run.log; sed 's/^/Cd0.0 0.3 0.0 /' onWP_list.txt | xargs -P 11 -L 1 bash coupWD_job.sh >> d_run.log 2>&1
echo "== Cd0.3 $(date +%H:%M)" >> d_run.log; sed 's/^/Cd0.3 0.3 0.3 /' onWP_list.txt | xargs -P 11 -L 1 bash coupWD_job.sh >> d_run.log 2>&1
echo "== Cd0.6 $(date +%H:%M)" >> d_run.log; sed 's/^/Cd0.6 0.3 0.6 /' onWP_list.txt | xargs -P 11 -L 1 bash coupWD_job.sh >> d_run.log 2>&1
echo "== merge $(date +%H:%M)" >> d_run.log; PYTHONIOENCODING=utf-8 python mk_seld.py >> d_run.log 2>&1
n=$(ls $G/dw | grep -c "^sel_dwps3d_g[0-9]*_s[0-9]*\.csv$"); if [ "$n" != "150" ]; then echo "D_ABORT: sel_dwps3d files $n" >> d_run.log; exit 1; fi
echo "== eval $(date +%H:%M)" >> d_run.log; export ZS_OUT=zss3d ZS_ENVCFG=dwps3
sed 's/$/ ZS_CFG=dwps3d ZS_MODELS=N,C0.3,Cd0.0,Cd0.3,Cd0.6 ZS_GAMMA=0.1,0.3,0.5,1,3,10 ZS_SEEDBASE=9595 ZS_M=100 ZS_RHO=1,0.5,0,0.25,0.75/' onWP_list.txt | xargs -P 11 -L 1 bash zsW_job.sh >> d_run.log 2>&1
echo "D_ALL_DONE $(date +%H:%M)" >> d_run.log
