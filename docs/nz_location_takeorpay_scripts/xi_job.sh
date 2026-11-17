#!/bin/bash
g=$1; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
envs=$(grep "^dwpl75 " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"
env $envs FB_G=$g C0H_SEED=1 C0H_TAG=xi_g$g julia -t 1 "$SP2/xiW.jl" 2>&1 | grep -i "XI_DONE\|error" | head -2
