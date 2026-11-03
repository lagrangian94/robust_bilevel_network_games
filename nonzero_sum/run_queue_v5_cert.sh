#!/bin/bash
# run_queue_v5_cert.sh — 큐 v5 의 δ·β 민감도 실행 (logs/sens/sens_*.log) 의 λ 무관 인증 (2026-11-03 사용자 허락).
#   v5 (run_queue_v5.sh) 가 돌고 있어 그 파일은 고치지 않고, QUEUE_V5_DONE 뒤에 이어 돈다. julia 는 한 번에 하나.
#   cert_lambda_free.jl 이 RESULT 의 beta=·delta= 를 읽어 같은 인스턴스를 만든다. 결과: logs/sens/cert_lambda_free.log
# 실행: bash nonzero_sum/run_queue_v5_cert.sh >> nonzero_sum/logs/queue_v5.out 2>&1
cd "$(dirname "$0")/.." || exit 1
L=nonzero_sum/logs
stamp() { date '+%m-%d %H:%M'; }
nojulia() { ! powershell -NoProfile -Command "if (Get-Process julia -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; }
until grep -q '^=== QUEUE_V5_DONE' $L/queue_v5.out 2>/dev/null; do sleep 120; done
until nojulia; do sleep 30; done
echo "=== 시작 $(stamp): δ·β 인증 $L/sens/cert_lambda_free.log"
CF_LOGDIR=sens CF_LOGS="sens_L*_S20_*.log" julia -t 14,1 nonzero_sum/cert_lambda_free.jl > $L/sens/cert_stdout.log 2>&1 \
  || { echo "=== 멈춤 $(stamp): δ·β 인증 (exit $?)"; exit 1; }
echo "=== 끝 $(stamp): δ·β 인증 certified=yes $(grep -c 'certified=yes' $L/sens/cert_lambda_free.log)/$(grep -c '^CERT' $L/sens/cert_lambda_free.log)"
echo "=== QUEUE_V5_CERT_DONE $(stamp)"
