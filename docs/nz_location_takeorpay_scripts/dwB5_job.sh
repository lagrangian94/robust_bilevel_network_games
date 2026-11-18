#!/bin/bash
# bash dwB5_job.sh <g> <s>: Belief-only at radius 0.5 on the extended candidates (dwB5W.jl), ledger 4.33. Output geo/dw/{dwx,selx}_dwps3_B0.5_g_s.csv
g=$1; s=$2; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
envs=$(grep "^dwps3 " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; lg=geo/dw/dwx_dwps3_B0.5_g${g}_s$s${D5_SUF}.log
grep -q B5_DONE $lg 2>/dev/null || env $envs FB_G=$g C0H_SEED=$s C0H_TAG=b5_dwps3_g${g}_s$s julia -t 1 "$SP2/dwB5W.jl" > $lg 2>&1
if grep -q B5_DONE $lg; then echo "$(grep B5_DONE $lg) | $(grep "SELB scen 2" $lg | cut -c1-150)"; else echo "B5 g$g s$s FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-200)"; fi
