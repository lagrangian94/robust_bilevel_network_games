#!/bin/bash
# run_pooled_store_queue_v2.sh — 본문 목록 L1~L6 재실험 (2026-10-30, 사용자 결정: 논문용은 새 α-B&B 기본값으로 L1~L6 전부 다시, 메모리 nz_pooled_retailer·nz_target_instances)
#   α-B&B 기본값 (2026-10-30 변경, 문서 §10.29): range_lp, dd=false, farkas, infeas_pre(+retry), parent_basis, node_lp_cap 120 s. PP_BNBKW 없음.
#   목록·공통 설정·규칙은 run_pooled_store_queue.sh 와 같음 (바꾸지 않음): 1 시간, 허용오차 0.5%, λᵁ=2000, 시드 1~10,
#   S=10 → 20 → 50, 어떤 S 에서 Optimal < 5/10 이면 그 설정의 더 큰 S 는 안 돌림.
#   반응 프리솔브: L1~L5 는 예전처럼 켬 (PP_PRE=1), L6 는 끔 (PP_PRE=0: L6 에서 0 개 고정인데 x 하나에 최대 606 s, 이전 L6 실행도 모두 끔).
#   로그는 새 이름 (main2_list_*) — 예전 결과 (main_list_*, main_pooled_store_A2_d15_*) 는 덮어쓰지 않음.
#   끊었다가 다시 실행하면 RESULT 가 10 줄인 로그는 건너뜀. 순서: S 마다 L1→L6.
# 실행: bash nonzero_sum/run_pooled_store_queue_v2.sh >> nonzero_sum/logs/run_pooled_store_queue_v2.out 2>&1
cd "$(dirname "$0")/.." || exit 1
L=nonzero_sum/logs
SEEDS="1,2,3,4,5,6,7,8,9,10"
nres() { if [ -f "$1" ]; then grep -c '^RESULT' "$1"; else echo 0; fi; }
nopt() { if [ -f "$1" ]; then grep '^RESULT' "$1" | grep -c 'status=Optimal'; else echo 0; fi; }
pilot() {  # d A B NB S pre log
  local d=$1 A=$2 B=$3 NB=$4 S=$5 pre=$6 log=$7
  if [ "$(nres "$L/$log")" -ge 10 ]; then echo "건너뜀 (이미 끝남): $log"; return; fi
  echo "=== 시작 $(date '+%m-%d %H:%M'): $log"
  PP_NLOC=$d PP_NA=$A PP_NB=$B PP_NBMAX=$NB PP_QB=750 PP_LAMBDA=2000 PP_S=$S PP_SEEDS="$SEEDS" PP_TL=3600 PP_WORKERS=12 PP_PRE=$pre \
    julia -t 14,1 nonzero_sum/pilot_pooled_store.jl > "$L/$log" 2>&1
  echo "=== 끝 $(date '+%m-%d %H:%M'): $log  Optimal $(nopt "$L/$log")/$(nres "$L/$log")"
}
LIST=("L1 15 2 3 3 1" "L2 15 3 2 2 1" "L3 20 2 3 3 1" "L4 20 3 2 2 1" "L5 25 2 3 3 1" "L6 25 5 5 5 0")
declare -A ACTIVE
for e in "${LIST[@]}"; do set -- $e; ACTIVE[$1]=1; done
for S in 10 20 50; do
  for e in "${LIST[@]}"; do set -- $e
    [ "${ACTIVE[$1]}" = "1" ] || continue
    log="main2_list_$1_d$2_A$3B$4_S$S.log"
    pilot $2 $3 $4 $5 $S $6 "$log"
    [ "$(nopt "$L/$log")" -ge 5 ] || { ACTIVE[$1]=0; echo "멈춤 규칙: $1 은 S=$S 에서 Optimal $(nopt "$L/$log")/10 → 더 큰 S 안 돌림"; }
  done
done
echo "=== QUEUE_V2_DONE $(date '+%m-%d %H:%M')"
