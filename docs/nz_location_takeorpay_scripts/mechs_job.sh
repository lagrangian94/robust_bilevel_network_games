#!/bin/bash
# bash mechs_job.sh <g> <s>: commitment per store (mechSW.jl) of the decisions N, B0.1, D0.1, C of config dwps3c. Output geo/mechs/
g=$1; s=$2; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
envs=$(grep "^dwps3 " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; mkdir -p geo/mechs; lg=geo/mechs/mechs_g${g}_s$s.log
grep -q MECHS_DONE $lg 2>/dev/null || env $envs ZS_OUT=mechs ZS_CFG=dwps3c ZS_MODELS=N,B0.1,D0.1,C FB_G=$g ON_SEED=$s C0H_SEED=$s C0H_TAG=mechs_g${g}_s$s julia -t 1 "$SP2/mechSW.jl" > $lg 2>&1
if grep -q MECHS_DONE $lg; then echo "$(grep MECHS_DONE $lg)"; else echo "MECHS g$g s$s FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-200)"; fi
