#!/bin/bash
# bash coupW_job.sh <g> <s> [extra env ...]: Coupled decision (price-based, sets of config dwps3) on the candidates of the decisions run
g=$1; s=$2; shift; shift; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
cfg=${CP_CFG:-dwps3}; export CP_CFG=$cfg; envs=$(grep "^$cfg " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; tag=${cfg}_g${g}_s$s; lg=geo/dw/cp_$tag.log
[ -f geo/dw/sel_$tag.csv ] || { echo "CP g$g s$s FAIL: no decisions file sel_$tag.csv"; exit 0; }
grep -q CP_DONE $lg 2>/dev/null || env $envs "$@" FB_G=$g C0H_SEED=$s C0H_TAG=cp_$tag julia -t 1 "$SP2/coupWR.jl" > $lg 2>&1
if grep -q CP_DONE $lg; then echo "$(grep CP_DONE $lg) | $(grep SELC $lg | tr '\n' ' ' | cut -c1-160)"; else echo "CP g$g s$s FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-200)"; fi
