#!/bin/bash
# bash dwD5_job.sh <g> <s>: Data-DR at radius 0.5 by full enumeration (dwD5W.jl), ledger 4.33. Output geo/dw/{vnx,selx}_dwps3_D0.5_g_s.csv and candx_dwps3_0.5_g_s.csv
g=$1; s=$2; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
envs=$(grep "^dwps3 " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; lg=geo/dw/vnx_dwps3_D0.5_g${g}_s$s${D5_SUF}.log
grep -q D5_DONE $lg 2>/dev/null || env $envs FB_G=$g C0H_SEED=$s C0H_TAG=d5_dwps3_g${g}_s$s julia -t 1 "$SP2/dwD5W.jl" > $lg 2>&1
if grep -q D5_DONE $lg; then echo "$(grep D5_DONE $lg) | $(grep "SELD scen 2" $lg | cut -c1-150)"; else echo "D5 g$g s$s FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-200)"; fi
