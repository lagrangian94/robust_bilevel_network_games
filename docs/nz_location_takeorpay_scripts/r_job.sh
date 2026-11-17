#!/bin/bash
# bash r_job.sh <flow|mech> <cfg> <g> <s>: shipments (flowWR.jl) or per-scenario revenue (mechWR.jl) of the decisions sel_<cfg>c_g_s.csv with the settings of config <cfg>. Output geo/mech_<cfg>/
k=$1; cfg=$2; g=$3; s=$4; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
envs=$(grep "^$cfg " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-); K=$(echo $k | tr a-z A-Z)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; mkdir -p geo/mech_$cfg; lg=geo/mech_$cfg/${k}_g${g}_s$s.log
grep -q ${K}_DONE $lg 2>/dev/null || env $envs ZS_OUT=mech_$cfg ZS_CFG=${cfg}c ZS_MODELS=N,B0.1,B0.2,D0.1,D0.2,C FB_G=$g ON_SEED=$s C0H_SEED=$s C0H_TAG=${k}_${cfg}_g${g}_s$s julia -t 1 "$SP2/${k}WR.jl" > $lg 2>&1
if grep -q ${K}_DONE $lg; then echo "$(grep ${K}_DONE $lg)"; else echo "$K $cfg g$g s$s FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-200)"; fi
