#!/bin/bash
# bash mechc_job.sh <g> <s>: per-scenario revenue (mechW.jl) of the decisions of config dwps3c (N, B0.1, B0.2, D0.1, D0.2, C; box r/sqrt(S)). Output geo/mechc/
g=$1; s=$2; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
envs=$(grep "^dwps3 " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; mkdir -p geo/mechc; lg=geo/mechc/mech_g${g}_s$s.log
grep -q MECH_DONE $lg 2>/dev/null || env $envs ZS_OUT=mechc ZS_CFG=dwps3c ZS_MODELS=N,B0.1,B0.2,D0.1,D0.2,C FB_G=$g ON_SEED=$s C0H_SEED=$s C0H_TAG=mechc_g${g}_s$s julia -t 1 "$SP2/mechW.jl" > $lg 2>&1
if grep -q MECH_DONE $lg; then echo "$(grep MECH_DONE $lg)"; else echo "MECH g$g s$s FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-200)"; fi
