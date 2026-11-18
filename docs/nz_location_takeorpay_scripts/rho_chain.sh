#!/bin/bash
# finer rho (ledger 4.32, H79): wait for the radius sweep (sweep_chain.sh) to finish, then evaluate all ten models at rho = 0.25, 0.75 into geo/zss3r.
cd "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
until grep -q "SWEEP_ALL_DONE" sweep_run.log; do sleep 120; done
G="C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo"
n=$(ls $G/dw | grep -c "^sel_dwps3x_g.*csv"); echo "== rho start $(date +%H:%M), sel_dwps3x files $n" > rho_run.log
if [ "$n" != "150" ]; then echo "RHO_ABORT: sel_dwps3x files $n" >> rho_run.log; exit 1; fi
export ZS_OUT=zss3r ZS_ENVCFG=dwps3
sed 's/$/ ZS_CFG=dwps3x ZS_MODELS=N,B0.1,B0.2,B0.3,D0.1,D0.2,D0.3,C,C0.2,C0.3 ZS_GAMMA=0.1,0.3,0.5,1,3,10 ZS_SEEDBASE=9595 ZS_M=100 ZS_RHO=0.25,0.75/' onWP_list.txt | xargs -P 11 -L 1 bash zsW_job.sh >> rho_run.log 2>&1
echo "RHO_ALL_DONE $(date +%H:%M)" >> rho_run.log
