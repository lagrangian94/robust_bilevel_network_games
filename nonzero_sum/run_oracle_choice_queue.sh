#!/usr/bin/env bash
# run_oracle_choice_queue.sh — oracle 결정 (2026-10-27 사용자): 어려운 인스턴스에서 우리 알고리즘 (belief cut + local) 의 전역 oracle 을
#   α-B&B (기존 본 실험 로그) 와 Gurobi Ω (PP_ORACLE=omega_grb) 중 더 잘 푸는 쪽으로 고른다. 이긴 쪽으로 큰 인스턴스를 돌리고,
#   벤치마크는 belief cut 만 뺀 같은 oracle (α-B&B → SB-A, Gurobi → BD-G).
#   L6 S=3 (스크리닝과 같은 600 s, 시드 1~10), L4 S=50, |A|=3 S=50, |A|=4 S=20 (시드 1~3, 1 시간) → nonzero_sum/logs/abl/grb_<이름>_S<S>.log
# 실행: bash nonzero_sum/run_oracle_choice_queue.sh >> nonzero_sum/logs/abl/queue.out 2>&1
cd /c/Users/user/robust_bilevel_network_games || exit 1
L=nonzero_sum/logs/abl; mkdir -p "$L"
nres() { if [ -f "$1" ]; then grep -c '^RESULT' "$1"; else echo 0; fi; }
nopt() { if [ -f "$1" ]; then grep '^RESULT' "$1" | grep -c 'status=Optimal'; else echo 0; fi; }
until ! powershell -NoProfile -Command "if (Get-Process julia -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 60; done

# 이름 d A B NB qB S seeds TL 시드수
LIST=("L6 25 5 5 5 750 3 1,2,3,4,5,6,7,8,9,10 600 10" "L4 20 3 2 2 750 50 1,2,3 3600 3" "A3_d16 16 3 3 3 cap 50 1,2,3 3600 3" "A4_d17 17 4 3 3 cap 20 1,2,3 3600 3")
for item in "${LIST[@]}"; do
  set -- $item
  log="$L/grb_$1_S$7.log"
  if [ "$(nres "$log")" -ge "${10}" ]; then echo "=== 건너뜀 (이미 끝남): $log"; continue; fi
  echo "=== 시작 $(date '+%m-%d %H:%M'): $log 시드 $8"
  PP_ORACLE=omega_grb PP_NLOC=$2 PP_NA=$3 PP_NB=$4 PP_NBMAX=$5 PP_QB=$6 PP_LAMBDA=2000 PP_S=$7 PP_SEEDS="$8" PP_TL=$9 PP_WORKERS=12 \
    julia -t 14,1 nonzero_sum/pilot_pooled_store.jl > "$log" 2>&1
  echo "=== 끝남 $(date '+%m-%d %H:%M'): $log  Optimal $(nopt "$log")/$(nres "$log")"
done
echo "=== oracle 결정 큐 끝"
