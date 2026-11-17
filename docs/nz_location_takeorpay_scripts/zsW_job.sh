#!/bin/bash
# bash zsW_job.sh <g> <s> [extra env ...]: evaluation on the zero-sum gamma-zeta grid (support = 10 scenarios) of the stored price-based decisions (config dwpl75)
g=$1; s=$2; shift; shift; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
envs=$(grep "^${ZS_ENVCFG:-dwpl75} " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; ZDIR=${ZS_OUT:-zs}; export ZS_OUT=$ZDIR; mkdir -p geo/$ZDIR; lg=geo/$ZDIR/zs_g${g}_s$s.log
grep -q ZS_DONE $lg 2>/dev/null || env $envs ZS_CFG=${ZS_ENVCFG:-dwpl75} "$@" FB_G=$g ON_SEED=$s C0H_SEED=$s C0H_TAG=zs_g${g}_s$s julia -t 1 "$SP2/zsW.jl" > $lg 2>&1
if grep -q ZS_DONE $lg; then echo "$(grep ZS_DONE $lg)"; else echo "ZS g$g s$s FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-200)"; fi
