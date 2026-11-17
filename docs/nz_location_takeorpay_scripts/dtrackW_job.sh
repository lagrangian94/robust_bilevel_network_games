#!/bin/bash
# bash dtrackW_job.sh <g> <s> [extra env ...]: Data-DR vs "belief follows the truth" on the set of config dwps3 (box r/sqrt(S)), price-based
g=$1; s=$2; shift; shift; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
envs=$(grep "^${TR_ENVCFG:-dwps3} " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; OD=${TR_OUT:-dtr}; export TR_OUT=$OD; mkdir -p geo/$OD; tag=g${g}_s$s; lg=geo/$OD/dtr_$tag.log
grep -q DTR_DONE $lg 2>/dev/null || env $envs "$@" FB_G=$g C0H_SEED=$s C0H_TAG=$tag julia -t 1 "$SP2/dtrackW.jl" > $lg 2>&1
if grep -q DTR_DONE $lg; then echo "$(grep DTR_DONE $lg)"; else echo "DTR g$g s$s FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-200)"; fi
