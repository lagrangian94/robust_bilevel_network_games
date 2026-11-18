#!/bin/bash
# radius sweep (ledger 4.31): Belief-only 0.3 -> Coupled 0.2 -> Coupled 0.3 -> merge -> evaluation -> decomposition. 150 runs each, 11 at a time. Log sweep_run.log
cd "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
echo "START $(date +%H:%M)" > sweep_run.log
echo "== B0.3 $(date +%H:%M)" >> sweep_run.log; cat onWP_list.txt | xargs -P 11 -L 1 bash dwB3_job.sh >> sweep_run.log 2>&1
echo "== C0.2 $(date +%H:%M)" >> sweep_run.log; sed 's/^/C0.2 0.2 /' onWP_list.txt | xargs -P 11 -L 1 bash coupWE_job.sh >> sweep_run.log 2>&1
echo "== C0.3 $(date +%H:%M)" >> sweep_run.log; sed 's/^/C0.3 0.3 /' onWP_list.txt | xargs -P 11 -L 1 bash coupWE_job.sh >> sweep_run.log 2>&1
echo "== merge $(date +%H:%M)" >> sweep_run.log; PYTHONIOENCODING=utf-8 python mk_selx.py >> sweep_run.log 2>&1
echo "== eval $(date +%H:%M)" >> sweep_run.log; export ZS_OUT=zss3x ZS_ENVCFG=dwps3
sed 's/$/ ZS_CFG=dwps3x ZS_MODELS=N,B0.3,D0.3,C0.2,C0.3 ZS_GAMMA=0.1,0.3,0.5,1,3,10 ZS_SEEDBASE=9595 ZS_M=100/' onWP_list.txt | xargs -P 11 -L 1 bash zsW_job.sh >> sweep_run.log 2>&1
echo "== mech $(date +%H:%M)" >> sweep_run.log; cat onWP_list.txt | xargs -P 11 -L 1 bash mechx_job.sh >> sweep_run.log 2>&1
echo "SWEEP_ALL_DONE $(date +%H:%M)" >> sweep_run.log
