#!/bin/bash
# bash splitW_job.sh <g> [extra env ...]: split of the loss N - N* into demand-estimate and order-prediction parts for geography g (main configuration dw75 / dwl75)
g=$1; shift; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
envs=$(grep "^dw75 " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; mkdir -p geo/tw; lg=${SP_LOG:-geo/tw/sp_g$g.log}
grep -q SP_DONE $lg 2>/dev/null || env $envs "$@" FB_G=$g C0H_SEED=1 C0H_TAG=sp_g$g julia -t 1 "$SP2/splitW.jl" > $lg 2>&1
if grep -q SP_DONE $lg; then echo "$(grep SP_DONE $lg) | notes $(grep -c NOTE $lg)"; else echo "SP g$g FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-200)"; fi
