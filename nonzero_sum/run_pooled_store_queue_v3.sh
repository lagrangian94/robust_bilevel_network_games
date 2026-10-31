#!/bin/bash
# run_pooled_store_queue_v3.sh — 보충자료 |A|=3·4 재실험 + λ 무관 인증 (2026-11-01, 사용자 요청 "보충자료 |A|=3·4 랑 λ 무관 인증도 새 기본값으로")
#   α-B&B 는 새 기본값 (2026-10-30, 문서 §10.29). PP_BNBKW 없음, cert_lambda_free.jl 의 CF_BNBKW 없음.
#   1. 보충 (run_pooled_store_queue.sh 의 B·D·E 와 같은 설정: 시장 10 고정, 출점비 = 거점 용량 (PP_QB=cap), B3 N_B3, 반응 프리솔브 켬)
#        |A|=3 d=16, |A|=4 d=17. S=10 → 20 → 50, 어떤 S 에서 Optimal < 5/10 이면 그 설정의 더 큰 S 는 안 돌림. 1 시간, 시드 1~10.
#      로그 main2_pooled_store_A{3,4}_d{16,17}_S*.log (예전 main_pooled_store_* 는 덮어쓰지 않음).
#   2. λ 무관 인증 (cert_lambda_free.jl, 그대로): 재실험 v2 본문 (main2_list_*) 와 1 의 보충 로그.
#      Optimal 인 RESULT 줄만 인증한다 — TimeLimit 은 λ 문제의 gap 이 이미 0.5% 보다 커서 인증 대상이 아니고,
#      x 전부 (L6 은 32 개) 를 10/120/600 s 씩 돌려 시간만 쓴다. 그래서 Optimal 줄만 모은 사본을 logs/cert2/ 에 만들어 CF_LOGDIR 로 넘긴다
#      (사본의 파일 이름은 원본과 같음). 결과 logs/cert2_lambda_free.log, CERT 줄이 있으면 이어서 (끊었다 다시 실행해도 됨).
# 실행: bash nonzero_sum/run_pooled_store_queue_v3.sh >> nonzero_sum/logs/run_pooled_store_queue_v3.out 2>&1
cd "$(dirname "$0")/.." || exit 1
L=nonzero_sum/logs
SEEDS="1,2,3,4,5,6,7,8,9,10"
nres() { if [ -f "$1" ]; then grep -c '^RESULT' "$1"; else echo 0; fi; }
nopt() { if [ -f "$1" ]; then grep '^RESULT' "$1" | grep -c 'status=Optimal'; else echo 0; fi; }
pilot() {  # d A B NB S log
  local d=$1 A=$2 B=$3 NB=$4 S=$5 log=$6
  if [ "$(nres "$L/$log")" -ge 10 ]; then echo "건너뜀 (이미 끝남): $log"; return; fi
  echo "=== 시작 $(date '+%m-%d %H:%M'): $log"
  PP_NLOC=$d PP_NA=$A PP_NB=$B PP_NBMAX=$NB PP_QB=cap PP_LAMBDA=2000 PP_S=$S PP_SEEDS="$SEEDS" PP_TL=3600 PP_WORKERS=12 PP_PRE=1 \
    julia -t 14,1 nonzero_sum/pilot_pooled_store.jl > "$L/$log" 2>&1
  echo "=== 끝 $(date '+%m-%d %H:%M'): $log  Optimal $(nopt "$L/$log")/$(nres "$L/$log")"
}

# ---- 1. 보충 ----
LIST=("A3 16 3 3 3" "A4 17 4 3 3")
declare -A ACTIVE
for e in "${LIST[@]}"; do set -- $e; ACTIVE[$1]=1; done
for S in 10 20 50; do
  for e in "${LIST[@]}"; do set -- $e
    [ "${ACTIVE[$1]}" = "1" ] || continue
    log="main2_pooled_store_$1_d$2_S$S.log"
    pilot $2 $3 $4 $5 $S "$log"
    [ "$(nopt "$L/$log")" -ge 5 ] || { ACTIVE[$1]=0; echo "멈춤 규칙: $1 은 S=$S 에서 Optimal $(nopt "$L/$log")/10 → 더 큰 S 안 돌림"; }
  done
done
echo "=== 보충 끝 $(date '+%m-%d %H:%M')"

# ---- 2. λ 무관 인증 (Optimal 줄만) ----
mkdir -p "$L/cert2"
for f in "$L"/main2_list_L*_S*.log "$L"/main2_pooled_store_A*_S*.log; do
  [ -f "$f" ] || continue
  grep '^RESULT pooled-store' "$f" | grep 'status=Optimal' > "$L/cert2/$(basename "$f")"
done
echo "=== 인증 시작 $(date '+%m-%d %H:%M'): Optimal 줄 $(cat "$L"/cert2/*.log | wc -l) 개"
CF_LOGDIR=cert2 CF_LOGS="main2_list_L*_S*.log,main2_pooled_store_A*_S*.log" CF_OUT="$L/cert2_lambda_free.log" CF_WORKERS=12 \
  julia -t 14,1 nonzero_sum/cert_lambda_free.jl >> "$L/cert2_lambda_free.stdout" 2>&1
echo "=== 인증 끝 $(date '+%m-%d %H:%M'): CERT $(grep -c '^CERT' "$L/cert2_lambda_free.log" 2>/dev/null) 개, certified=yes $(grep -c 'certified=yes' "$L/cert2_lambda_free.log" 2>/dev/null)"
echo "=== QUEUE_V3_DONE $(date '+%m-%d %H:%M')"
