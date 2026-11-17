#!/bin/bash
# bash nb2_job.sh <flow|mech> <g> <s>: shipments (flowW.jl) or per-scenario revenue (mechW.jl) of the N_B <= 2 decisions (sel_nb2_g_s.csv), sets of config dwps3. Output geo/mechn2/
k=$1; g=$2; s=$3; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
envs=$(grep "^dwps3 " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-); K=$(echo $k | tr a-z A-Z)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; mkdir -p geo/mechn2; lg=geo/mechn2/${k}_g${g}_s$s.log
grep -q ${K}_DONE $lg 2>/dev/null || env $envs ZS_OUT=mechn2 ZS_CFG=nb2 ZS_MODELS=N,D0.1,D0.2 FB_G=$g ON_SEED=$s C0H_SEED=$s C0H_TAG=${k}n2_g${g}_s$s julia -t 1 "$SP2/${k}W.jl" > $lg 2>&1
if grep -q ${K}_DONE $lg; then echo "$(grep ${K}_DONE $lg)"; else echo "$K g$g s$s FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-200)"; fi
