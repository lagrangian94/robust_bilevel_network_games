#!/bin/bash
# run_pooled_store_queue.sh — retailer + pooled (거점별 한도) 본 실험 큐 (2026-10-16 사용자와 정한 계획, 문서 §10.24, 메모리 nz_pooled_retailer)
#
# 본문 목록 (Goyal·Yanıkoğlu 와 같은 패턴, 출점비 750 고정):
#   L2: d=15 A3 B2 N_B2   L3: d=20 A2 B3 N_B3   L4: d=20 A3 B2 N_B2   L5: d=25 A2 B3 N_B3   L6: d=25 A5 B5 N_B5
#   (L1 = d=15 A2 B3 N_B3 은 main_pooled_store_A2_d15_S*.log 로 이미 진행)
# 보충자료 (시장 10 곳 고정, |A| 증가, 출점비 = 거점 용량): |A|=3 d=16, |A|=4 d=17 (B3 N_B3)
# 규칙: S=10 → 20 → 50 순서로 올리고, 어떤 S 에서 10 개 중 Optimal 이 5 개 미만이면 그 설정은 더 큰 S 를 돌리지 않는다.
# λᵁ: 본문 목록은 설정마다 seed 1, S=10 에서 사후 인증 (lab_lambda_certify.jl, 시작 2000) 한 값을 쓴다. 인증이 안 끝나면 그 설정의 본 실험은 건너뛴다.
#
# 단계 (중간에 끊었다면 끝난 로그는 건너뛰도록 각 단계가 RESULT 줄 수를 확인한다):
#   A. L2~L6 스크리닝 S=3 (시드 10, 600 s)
#   B. 보충: |A|=4 S=10 마무리 (seed 3 재실행 + seed 6~10)
#   C. L2~L6 λ 인증 → S=10 본 실험 → 통과한 설정만 S=20 → S=50
#   D. 보충: |A|=3 S=50
#   E. 보충: |A|=4 S=20·50 (S=10 의 10 개 중 Optimal ≥ 5 일 때만)
# 실행: bash nonzero_sum/run_pooled_store_queue.sh   (로그: nonzero_sum/logs/, 요약 줄은 stdout)
cd "$(dirname "$0")/.." || exit 1
L=nonzero_sum/logs
SEEDS="1,2,3,4,5,6,7,8,9,10"

nres() { if [ -f "$1" ]; then grep -c '^RESULT' "$1"; else echo 0; fi; }
nopt() { if [ -f "$1" ]; then grep '^RESULT' "$1" | grep -c 'status=Optimal'; else echo 0; fi; }

# pilot_pooled_store.jl 한 번: d A B NB qB(숫자|cap) λ S seeds TL log
pilot() {
  local d=$1 A=$2 B=$3 NB=$4 qB=$5 lam=$6 S=$7 seeds=$8 TL=$9 log=${10}
  local want; want=$(echo "$seeds" | tr ',' '\n' | wc -l)
  if [ "$(nres "$L/$log")" -ge "$want" ]; then echo "건너뜀 (이미 끝남): $log"; return; fi
  PP_NLOC=$d PP_NA=$A PP_NB=$B PP_NBMAX=$NB PP_QB=$qB PP_LAMBDA=$lam PP_S=$S PP_SEEDS="$seeds" PP_TL=$TL PP_WORKERS=12 \
    julia -t 14,1 nonzero_sum/pilot_pooled_store.jl > "$L/$log" 2>&1
  echo "끝남: $log  Optimal $(nopt "$L/$log")/$(nres "$L/$log")"
}

# 본문 목록: 이름 d A B NB
LIST=("L2 15 3 2 2" "L3 20 2 3 3" "L4 20 3 2 2" "L5 25 2 3 3" "L6 25 5 5 5")

# ---- A. 스크리닝 ----
for e in "${LIST[@]}"; do set -- $e
  pilot $2 $3 $4 $5 750 2000 3 "$SEEDS" 600 "screen_list_$1_d$2_A$3B$4_S3.log"
done

# ---- B. 보충: |A|=4 S=10 마무리 ----
pilot 17 4 3 3 cap 2000 10 "3" 3600 main_pooled_store_A4_d17_S10_seed3_gapfix.log
pilot 17 4 3 3 cap 2000 10 "6,7,8,9,10" 3600 main_pooled_store_A4_d17_S10_seeds6-10.log

# ---- C. 본문 목록: λ 인증 → S=10 → 20 → 50 ----
declare -A LAM ACTIVE
for e in "${LIST[@]}"; do set -- $e
  cert="$L/lab_lambda_certify_list_$1_seed1_S10.log"
  if ! grep -q '^RESULT' "$cert" 2>/dev/null; then
    LC_NLOC=$2 LC_NA=$3 LC_NB=$4 LC_NBMAX=$5 LC_QB=750 LC_S=10 NZ_SEED=1 LC_LAM0=2000 LC_BNB_TL=1200 LC_WORKERS=12 \
      julia -t 14,1 nonzero_sum/lab_lambda_certify.jl 2>&1 | grep --line-buffered -E "^lab_lambda|^  x=|^RESULT|ERROR|rror" > "$cert"
  fi
  lam=$(grep '^RESULT' "$cert" | sed -E 's/.*λᵁ = ([0-9]+).*/\1/')
  if [ -n "$lam" ]; then LAM[$1]=$lam; ACTIVE[$1]=1; echo "인증: $1 λᵁ=$lam"; else ACTIVE[$1]=0; echo "인증 실패 → 본 실험 건너뜀: $1"; fi
done
for S in 10 20 50; do
  for e in "${LIST[@]}"; do set -- $e
    [ "${ACTIVE[$1]}" = "1" ] || continue
    log="main_list_$1_d$2_A$3B$4_S$S.log"
    pilot $2 $3 $4 $5 750 ${LAM[$1]} $S "$SEEDS" 3600 "$log"
    [ "$(nopt "$L/$log")" -ge 5 ] || { ACTIVE[$1]=0; echo "멈춤 규칙: $1 은 S=$S 에서 Optimal $(nopt "$L/$log")/10 → 더 큰 S 안 돌림"; }
  done
done

# ---- D. 보충: |A|=3 S=50 ----
pilot 16 3 3 3 cap 2000 50 "$SEEDS" 3600 main_pooled_store_A3_d16_S50.log

# ---- E. 보충: |A|=4 S=20·50 (조건부) ----
n4=$(( $(grep '^RESULT' $L/main_pooled_store_A4_d17_S10.log | grep -v 'seed=3 ' | grep -c 'status=Optimal') \
     + $(nopt $L/main_pooled_store_A4_d17_S10_seed3_gapfix.log) + $(nopt $L/main_pooled_store_A4_d17_S10_seeds6-10.log) ))
if [ "$n4" -ge 5 ]; then
  pilot 17 4 3 3 cap 2000 20 "$SEEDS" 3600 main_pooled_store_A4_d17_S20.log
  [ "$(nopt $L/main_pooled_store_A4_d17_S20.log)" -ge 5 ] && pilot 17 4 3 3 cap 2000 50 "$SEEDS" 3600 main_pooled_store_A4_d17_S50.log
else
  echo "멈춤 규칙: |A|=4 는 S=10 에서 Optimal $n4/10 → S=20·50 안 돌림"
fi
echo QUEUE_DONE
