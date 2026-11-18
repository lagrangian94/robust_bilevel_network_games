#!/bin/bash
# bash coupWD_job.sh <lab> <eps> <delta> <g> <s>: Coupled at radius eps with coupling radius delta (ledger 4.34) on the candidates of dwps3 (coupWE.jl). Output geo/dw/{cpx,selx}_dwps3_<lab>_g_s.csv
lab=$1; eps=$2; dl=$3; g=$4; s=$5; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
envs=$(grep "^dwps3 " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; lg=geo/dw/cpx_dwps3_${lab}_g${g}_s$s.log
grep -q CP_DONE $lg 2>/dev/null || env $envs CP_CFG=dwps3 CP_LAB=$lab CP_EH=$eps CP_ET=$eps CP_DL=$dl FB_G=$g C0H_SEED=$s C0H_TAG=cpx_${lab}_g${g}_s$s julia -t 1 "$SP2/coupWE.jl" > $lg 2>&1
if grep -q CP_DONE $lg; then echo "$(grep CP_DONE $lg) | $(grep "^COUPLED" $lg | sed 's/.*eps_hat/eps_hat/' | cut -c1-60) | $(grep "SELC scen 2" $lg | cut -c1-120)"; else echo "CPD $lab g$g s$s FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-200)"; fi
