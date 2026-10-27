#!/usr/bin/env bash
# run_bd_kkt_queue.sh — 벤치마크 BD-KKT 큐 (2026-10-21~22 사용자 결정): S = 10 만, KKT 에 시작해 제공 (다), 조기 판정 (나).
#   1. 비제로섬 입지 L1~L5 → nonzero_sum/logs/bd_kkt/bdkkt_<L>_S10.log
#      (나): 시드 1~3 을 먼저 돌려 셋 다 Optimal 이 아니면 나머지 시드는 건너뛰고 "SKIP_REST" 줄을 남긴다 (미해결 3/3 으로 보고).
#            하나라도 Optimal 이면 시드 4~10 을 같은 로그에 이어 붙인다.
#   2. zero-sum 네트워크 차단 5 개 (결합 δ = 0.1) → true_dro/factor_5/logs/nig_coupled/S_10/<net>_bdkkt.log (끝난 로그는 건너뜀)
# 끊었다가 다시 실행하면 이어서 한다. 요약은 표준출력 (=== 줄).
# 실행: bash nonzero_sum/run_bd_kkt_queue.sh >> nonzero_sum/logs/bd_kkt/queue.out 2>&1
cd /c/Users/user/robust_bilevel_network_games || exit 1
L=nonzero_sum/logs/bd_kkt; mkdir -p "$L"
nres() { if [ -f "$1" ]; then grep -c '^RESULT' "$1"; else echo 0; fi; }
nopt() { if [ -f "$1" ]; then grep '^RESULT' "$1" | grep -c 'status=Optimal'; else echo 0; fi; }
runpp() {   # d A B NB seeds log (>> 로 이어 붙임)
  PP_NLOC=$1 PP_NA=$2 PP_NB=$3 PP_NBMAX=$4 PP_QB=750 PP_LAMBDA=2000 PP_S=10 PP_SEEDS="$5" PP_TL=3600 PP_WORKERS=12 \
    julia -t 2 nonzero_sum/run_bd_kkt_pooled.jl >> "$6" 2>&1
}

# 이름 d A B NB (qB = 750, λ = 2000, 본 실험 목록과 같음)
LIST=("L1 15 2 3 3" "L2 15 3 2 2" "L3 20 2 3 3" "L4 20 3 2 2" "L5 25 2 3 3")
for item in "${LIST[@]}"; do
  set -- $item
  log="$L/bdkkt_$1_S10.log"
  if grep -q '^SKIP_REST' "$log" 2>/dev/null || [ "$(nres "$log")" -ge 10 ]; then echo "=== 건너뜀 (이미 끝남): $log"; continue; fi
  if [ "$(nres "$log")" -lt 3 ]; then
    [ -f "$log" ] && mv "$log" "$log.partial_$(date +%s)"
    echo "=== 시작 $(date '+%m-%d %H:%M'): $log 시드 1~3"
    runpp $2 $3 $4 $5 "1,2,3" "$log"
  fi
  if [ "$(nopt "$log")" -eq 0 ]; then
    echo "SKIP_REST 시드 1~3 모두 미해결 → 시드 4~10 건너뜀 (사용자 결정 (나))" >> "$log"
    echo "=== 조기 판정 $(date '+%m-%d %H:%M'): $log  Optimal 0/3 → 나머지 건너뜀"
    continue
  fi
  echo "=== 이어서 $(date '+%m-%d %H:%M'): $log 시드 4~10 (시드 1~3 Optimal $(nopt "$log")/3)"
  runpp $2 $3 $4 $5 "4,5,6,7,8,9,10" "$log"
  echo "=== 끝남 $(date '+%m-%d %H:%M'): $log  Optimal $(nopt "$log")/$(nres "$log")"
done

echo "=== 시작 $(date '+%m-%d %H:%M'): zero-sum BD-KKT S=10 (끝난 네트워크는 건너뜀)"
(cd true_dro/factor_5 && NS_S=10 julia -t 2 run_nig_coupled_bdkkt.jl)
echo "=== 끝남 $(date '+%m-%d %H:%M'): zero-sum BD-KKT S=10"
echo "=== BD-KKT 큐 끝"
