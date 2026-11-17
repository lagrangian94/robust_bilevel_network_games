#!/bin/bash
# one-run test of the whole "unserved demand" pipeline on the test decisions (config dwps3r14n2, g1 s1): Coupled -> evaluation (tiny grid) -> decomposition -> shipments
cd "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
CP_CFG=dwps3r14n2 bash coupWR_job.sh 1 1
ZS_OUT=zs_rtest ZS_ENVCFG=dwps3r14n2 bash zsWR_job.sh 1 1 ZS_CFG=dwps3r14n2c ZS_MODELS=N,B0.1,B0.2,D0.1,D0.2,C ZS_GAMMA=1,10 ZS_ZETA=0 ZS_SEEDBASE=9595 ZS_M=5
bash r_job.sh mech dwps3r14n2 1 1; bash r_job.sh flow dwps3r14n2 1 1
O=/c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo
wc -l $O/zs_rtest/eval_g1_s1.csv $O/mech_dwps3r14n2/mech_g1_s1.csv $O/mech_dwps3r14n2/flow_g1_s1.csv; grep ";U;" $O/mech_dwps3r14n2/flow_g1_s1.csv | head -3; echo RTEST_END
