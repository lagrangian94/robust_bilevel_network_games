#!/bin/bash
# radius 0.5 (ledger 4.33): waits for the rho run (4.32), then Data-DR 0.5 (full enumeration) -> Belief-only 0.5 -> Coupled 0.5 -> merge -> evaluation -> decomposition. 150 runs each, 11 at a time. Log r5_run.log
cd "/c/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
G="/c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo"
until [ -f rho_run.log ] && grep -q "RHO_ALL_DONE\|RHO_ABORT" rho_run.log; do sleep 120; done
echo "START $(date +%H:%M)" > r5_run.log
echo "== D0.5 $(date +%H:%M)" >> r5_run.log; cat onWP_list.txt | xargs -P 11 -L 1 bash dwD5_job.sh >> r5_run.log 2>&1
n=$(ls $G/dw | grep -c "^candx_dwps3_0.5_g[0-9]*_s[0-9]*\.csv$"); if [ "$n" != "150" ]; then echo "R5_ABORT: candx files $n" >> r5_run.log; exit 1; fi
echo "== B0.5 $(date +%H:%M)" >> r5_run.log; cat onWP_list.txt | xargs -P 11 -L 1 bash dwB5_job.sh >> r5_run.log 2>&1
echo "== C0.5 $(date +%H:%M)" >> r5_run.log; sed 's/^/C0.5 0.5 /' onWP_list.txt | xargs -P 11 -L 1 bash coupW5_job.sh >> r5_run.log 2>&1
echo "== merge $(date +%H:%M)" >> r5_run.log; PYTHONIOENCODING=utf-8 python mk_sely.py >> r5_run.log 2>&1
n=$(ls $G/dw | grep -c "^sel_dwps3y_g[0-9]*_s[0-9]*\.csv$"); if [ "$n" != "150" ]; then echo "R5_ABORT: sel_dwps3y files $n" >> r5_run.log; exit 1; fi
echo "== eval $(date +%H:%M)" >> r5_run.log; export ZS_OUT=zss3y ZS_ENVCFG=dwps3
sed 's/$/ ZS_CFG=dwps3y ZS_MODELS=N,B0.5,D0.5,C0.5 ZS_GAMMA=0.1,0.3,0.5,1,3,10 ZS_SEEDBASE=9595 ZS_M=100/' onWP_list.txt | xargs -P 11 -L 1 bash zsW_job.sh >> r5_run.log 2>&1
echo "== mech $(date +%H:%M)" >> r5_run.log; cat onWP_list.txt | xargs -P 11 -L 1 bash mechy_job.sh >> r5_run.log 2>&1
echo "R5_ALL_DONE $(date +%H:%M)" >> r5_run.log
