#!/bin/bash
# bash coupW5_job.sh <lab> <eps> <g> <s>: Coupled at radius eps (delta = eps/3) on the extended candidates candx_dwps3_<eps> (coupW5.jl), ledger 4.33. Output geo/dw/{cpx,selx}_dwps3_<lab>_g_s.csv
lab=$1; eps=$2; g=$3; s=$4; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
envs=$(grep "^dwps3 " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; lg=geo/dw/cpx_dwps3_${lab}_g${g}_s$s${D5_SUF}.log
grep -q CP_DONE $lg 2>/dev/null || env $envs CP_CFG=dwps3 CP_LAB=$lab CP_EH=$eps CP_ET=$eps CP_CX=$eps FB_G=$g C0H_SEED=$s C0H_TAG=cp5_${lab}_g${g}_s$s julia -t 1 "$SP2/coupW5.jl" > $lg 2>&1
if grep -q CP_DONE $lg; then echo "$(grep CP_DONE $lg) | $(grep "^COUPLED" $lg | sed 's/.*eps_hat/eps_hat/' | cut -c1-60) | $(grep "SELC scen 2" $lg | cut -c1-120)"; else echo "CP5 $lab g$g s$s FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-200)"; fi
