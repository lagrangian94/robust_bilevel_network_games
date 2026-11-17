#!/bin/bash
# bash mech_job.sh <g> <s>: per-scenario revenue of the stored price-based decisions (config dwpl75) at belief q̂ and at the 90 pair-move beliefs
g=$1; s=$2; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
envs=$(grep "^${ZS_ENVCFG:-dwpl75} " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; ZDIR=${ZS_OUT:-mech}; export ZS_OUT=$ZDIR; mkdir -p geo/$ZDIR; lg=geo/$ZDIR/flow_g${g}_s$s.log
grep -q FLOW_DONE $lg 2>/dev/null || env $envs ZS_CFG=${ZS_ENVCFG:-dwpl75} ZS_MODELS=${MC_MODELS:-N,B0.1,D0.1,D0.3} FB_G=$g ON_SEED=$s C0H_SEED=$s C0H_TAG=flow_g${g}_s$s julia -t 1 "$SP2/flowW.jl" > $lg 2>&1
if grep -q FLOW_DONE $lg; then echo "$(grep FLOW_DONE $lg)"; else echo "FLOW g$g s$s FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-200)"; fi
