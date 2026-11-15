#!/bin/bash
# 본 구성 (세 단계 메뉴) 한 건: bash main_job.sh <cfg> <g> <s>. tkv* 는 tkV_job.sh, dw* 는 dwTier_job.sh 로 넘김 (끝난 건은 건너뜀)
SP2="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad"
case $1 in tkv*) bash "$SP2/tkV_job.sh" $1 $2 $3;; dw*) bash "$SP2/dwTier_job.sh" $1 $2 $3;; *) echo "MAIN $1 FAIL: 알 수 없는 설정";; esac
