#!/bin/bash
# bash fixbias_job.sh <g>: fixed-decision bias decomposition for geography g (configuration expo15)
g=$1; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
envs=$(grep "^expo15 " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"
env $envs C0H_SEED=1 C0H_TAG=fixbias_g$g FB_G=$g julia -t 1 "$SP2/fixbias.jl" > geo/expo/fixbias_g$g.log 2>&1
if grep -q FIXBIAS_DONE geo/expo/fixbias_g$g.log; then echo "OK g$g"; else echo "FAIL g$g: $(grep -m1 -i "error\|assert" geo/expo/fixbias_g$g.log | cut -c1-200)"; fi
