#!/bin/bash
# per-draw dump of one TK run: bash tkD_job.sh <cfg> <g> <s>. output geo/tkh/tkh_<cfg>_g_s.csv
cfg=$1; g=$2; s=$3; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
grep -q "^$cfg " "$SP2/scr_cfg.txt" || { echo "TKH $cfg g$g s$s FAIL: no config"; exit 0; }
envs=$(grep "^$cfg " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
case $cfg in topk*) scr=tkHK.jl;; *) scr=tkH.jl;; esac
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; mkdir -p geo/tkh
tag=${cfg}_g${g}_s${s}; lg=geo/tkh/tkh_${tag}.log
grep -q TKH_DONE $lg 2>/dev/null || env $envs TK_NF=5 C0H_SEED=$s C0H_TAG=$tag julia -t 1 "$SP2/$scr" > $lg 2>&1
if grep -q TKH_DONE $lg; then echo "TKH $tag $(grep "^TKH " $lg)"; else echo "TKH $tag FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-140)"; fi
