#!/bin/bash
# bash expoT_job.sh <cfg> <g> <s>: s = 0 computes the fixed-belief commitments h_B(x) for geography g (stage 1), s >= 1 evaluates training seed s (stage 2)
cfg=$1; g=$2; s=$3; SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
grep -q "^$cfg " "$SP2/scr_cfg.txt" || { echo "EXPO $cfg g$g s$s FAIL: no config"; exit 0; }
envs=$(grep "^$cfg " "$SP2/scr_cfg.txt" | head -1 | cut -d' ' -f2-)
cd /c/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad
eval "export $(python -I -X utf8 geo/pick_cities.py $g)"; mkdir -p geo/expo
if [ "$s" = "0" ]; then
  lg=geo/expo/hB_${cfg}_g${g}.log
  { grep -q HB_DONE $lg 2>/dev/null && [ -s geo/expo/hB_${cfg}_g${g}.bin ]; } || env $envs ET_STAGE=1 ET_CFG=$cfg FB_G=$g C0H_SEED=1 C0H_TAG=hB_${cfg}_g${g} julia -t 1 "$SP2/expoT.jl" > $lg 2>&1
  if grep -q HB_DONE $lg; then echo "$(grep HB_DONE $lg)"; else echo "HB $cfg g$g FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-160)"; fi
else
  tag=${cfg}_g${g}_s${s}; lg=geo/expo/expo_${tag}.log
  { grep -q EXPO_DONE $lg 2>/dev/null && [ -s geo/expo/expoS_${tag}.bin ]; } || env $envs ET_CFG=$cfg FB_G=$g C0H_SEED=$s C0H_TAG=$tag julia -t 1 "$SP2/expoT.jl" > $lg 2>&1
  if grep -q EXPO_DONE $lg; then echo "EXPO $(grep EXPO_DONE $lg | cut -c11-)"; else echo "EXPO $tag FAIL: $(grep -m1 -i "error\|assert" $lg | cut -c1-160)"; fi
fi
