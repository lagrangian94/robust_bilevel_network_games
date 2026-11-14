#!/bin/bash
# per-draw dump at N_F = 20 with the top-30 candidates: bash tkD20_job.sh <cfg> <g> <s>. output geo/tkd/tkd_<cfg>nf20_g_s.csv
cfg=$1; g=$2; s=$3; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
grep -q "^$cfg " "$SP2/scr_cfg.txt" || { echo "TKD20 $cfg g$g s$s FAIL: no config"; exit 0; }
envs=$(grep "^$cfg " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; mkdir -p geo/tkd
tag=${cfg}nf20_g${g}_s${s}; lg=geo/tkd/tkd_${tag}.log
grep -q TKD_DONE $lg 2>/dev/null || env $envs TK_NF=20 TKD_NC=30 C0H_SEED=$s C0H_TAG=$tag julia -t 1 "$SP2/tkD30.jl" > $lg 2>&1
if grep -q TKD_DONE $lg; then echo "TKD20 $tag $(grep "^TKD " $lg)"; else echo "TKD20 $tag FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-140)"; fi
