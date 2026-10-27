#!/usr/bin/env bash
# run_bdg_s50_queue.sh — BD-G 를 큰 S·어려운 설정으로 확장 + λ 무관 인증 (2026-10-27 사용자 결정, JOC 리뷰어 관점 검토 반영).
#   (이미 끝남: 비제로섬 L1·L2·L3 S=50. L4 S=50 은 앞 큐에서 돌던 julia 가 이어서 씀. L5 S=50 은 정보 가치가 낮아 뺌)
#   1. zero-sum 네트워크 차단 5 개 BD-G, S = 20 → 50 → true_dro/factor_5/logs/nig_coupled/S_<S>/<net>_bdg.log
#   2. 비제로섬 BD-G, 우리 방법이 못 풀었거나 활성 차원이 큰 곳:
#      L6 S=3 시드 1~10 (스크리닝과 같은 600 s), |A|=3 S=50 시드 1~3, |A|=4 S=20 시드 1~3 (출점비 = 용량)
#   3. λ 무관 인증 (cert_lambda_free.jl) → nonzero_sum/logs/cert_lambda_free.log
# 설정은 run_bdg_queue.sh 와 같음 (Gurobi 스레드 = 12 = 우리 B&B worker 수). 끝난 로그는 건너뜀.
# 실행: bash nonzero_sum/run_bdg_s50_queue.sh >> nonzero_sum/logs/bd_g/queue.out 2>&1
cd /c/Users/user/robust_bilevel_network_games || exit 1
L=nonzero_sum/logs/bd_g; mkdir -p "$L"
nres() { if [ -f "$1" ]; then grep -c '^RESULT' "$1"; else echo 0; fi; }
nopt() { if [ -f "$1" ]; then grep '^RESULT' "$1" | grep -c 'status=Optimal'; else echo 0; fi; }
until ! powershell -NoProfile -Command "if (Get-Process julia -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M'): $L/bdg_L4_S50.log  Optimal $(nopt "$L/bdg_L4_S50.log")/$(nres "$L/bdg_L4_S50.log") (앞 큐에서 이어 돈 것)"

for S in 20 50; do
  echo "=== 시작 $(date '+%m-%d %H:%M'): zero-sum BD-G S=$S"
  (cd true_dro/factor_5 && NS_METHOD=omega NS_S=$S julia -t 2 run_nig_coupled_bdkkt.jl)
  echo "=== 끝남 $(date '+%m-%d %H:%M'): zero-sum BD-G S=$S"
done

# 이름 d A B NB qB S seeds TL 시드수
LIST=("L6 25 5 5 5 750 3 1,2,3,4,5,6,7,8,9,10 600 10" "A3_d16 16 3 3 3 cap 50 1,2,3 3600 3" "A4_d17 17 4 3 3 cap 20 1,2,3 3600 3")
for item in "${LIST[@]}"; do
  set -- $item
  log="$L/bdg_$1_S$7.log"
  if [ "$(nres "$log")" -ge "${10}" ]; then echo "=== 건너뜀 (이미 끝남): $log"; continue; fi
  echo "=== 시작 $(date '+%m-%d %H:%M'): $log 시드 $8"
  PP_METHOD=omega PP_NLOC=$2 PP_NA=$3 PP_NB=$4 PP_NBMAX=$5 PP_QB=$6 PP_LAMBDA=2000 PP_S=$7 PP_SEEDS="$8" PP_TL=$9 PP_WORKERS=12 \
    julia -t 2 nonzero_sum/run_bd_kkt_pooled.jl > "$log" 2>&1
  echo "=== 끝남 $(date '+%m-%d %H:%M'): $log  Optimal $(nopt "$log")/$(nres "$log")"
done

echo "=== 시작 $(date '+%m-%d %H:%M'): λ 무관 인증"
julia -t 14,1 nonzero_sum/cert_lambda_free.jl >> nonzero_sum/logs/cert_lambda_free.stdout 2>&1
echo "=== 끝남 $(date '+%m-%d %H:%M'): λ 무관 인증  CERT $(grep -c '^CERT' nonzero_sum/logs/cert_lambda_free.log 2>/dev/null)"
echo "=== BD-G S50 큐 끝"
