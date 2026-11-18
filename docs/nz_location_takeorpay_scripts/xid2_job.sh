#!/bin/bash
# bash xid2_job.sh <g> <s>: cities, coordinates, regions, mean demand and the 10 demand scenarios of geography g, seed s (xidW.jl). Output geo/mech/xid_g<g>_s<s>.csv. Existing files are kept.
g=$1; s=$2; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
envs=$(grep "^dwps3 " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
[ -s geo/mech/xid_g${g}_s$s.csv ] && { echo "XID g$g s$s exists"; exit 0; }
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"
r=$(env $envs FB_G=$g ON_SEED=$s C0H_SEED=$s C0H_TAG=xid_g${g}_s$s julia -t 1 "$SP2/xidW.jl" 2>&1 | grep -i "XID_DONE\|error" | head -1); echo "XID g$g s$s $r"
