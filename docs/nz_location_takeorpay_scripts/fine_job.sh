#!/bin/bash
# 촘촘한 격자 본 평가 한 건: bash fine_job.sh <cfg> <g> <s>. tkvf* 는 tkV_job.sh, dwf* 는 dwTier_job.sh 로 넘김
SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
case $1 in tkvf*) bash "$SP2/tkV_job.sh" $1 $2 $3;; dwf*) bash "$SP2/dwTier_job.sh" $1 $2 $3;; *) echo "FINE $1 FAIL: 알 수 없는 설정";; esac
