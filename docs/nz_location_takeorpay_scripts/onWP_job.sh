#!/bin/bash
# bash onW_job.sh <g> <s> [extra env ...]: truth-unknown world on the support (10 scenarios, only probabilities change) for geography g, training seed s (main configuration dw75 / dwl75)
g=$1; s=$2; shift; shift; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
envs=$(grep "^${ON_ENVCFG:-dwpl75} " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; ONDIR=${ON_OUT:-onp}; export ON_OUT=$ONDIR; mkdir -p geo/$ONDIR; lg=${ON_LOG:-geo/$ONDIR/on_g${g}_s$s.log}
grep -q ON_DONE $lg 2>/dev/null || env $envs ON_CFG13=${ON_ENVCFG:-dwpl75} ON_CFG410=${ON_ENVCFG:-dwpl75} ON_D=0.0,0.1,0.2,0.3 "$@" FB_G=$g ON_SEED=$s C0H_SEED=$s C0H_TAG=on_g${g}_s$s julia -t 1 "$SP2/onWP.jl" > $lg 2>&1
if grep -q ON_DONE $lg; then echo "$(grep ON_DONE $lg) | notes $(grep -c NOTE $lg)"; else echo "ON g$g s$s FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-200)"; fi
