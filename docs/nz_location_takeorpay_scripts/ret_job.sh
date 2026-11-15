#!/bin/bash
# retailer-alone diagnosis: bash ret_job.sh <g> <s> <fr> <x> <tag>
g=$1; s=$2; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; mkdir -p geo/ret
C0H_SEED=$s SC_FR=$3 RD_X="$4" RD_TAG=$5 julia -t 1 "$SP2/ret_diag.jl" > geo/ret/ret_$5.log 2>&1
grep -q RD_DONE geo/ret/ret_$5.log && echo "RET $5 ok" || echo "RET $5 FAIL: $(grep -m1 -i "error" geo/ret/ret_$5.log | cut -c1-200)"
