#!/bin/bash
# bash truthW_job.sh <g> [extra env ...]: N*, B* and the true-cost evaluation for geography g (main configuration dw75 / dwl75)
g=$1; shift; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
envs=$(grep "^dw75 " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; mkdir -p geo/tw; lg=geo/tw/tw_g$g.log
grep -q TW_DONE $lg 2>/dev/null || env $envs "$@" FB_G=$g C0H_SEED=1 C0H_TAG=tw_g$g julia -t 1 "$SP2/truthW.jl" > $lg 2>&1
if grep -q TW_DONE $lg; then echo "$(grep TW_DONE $lg)"; else echo "TW g$g FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-200)"; fi
