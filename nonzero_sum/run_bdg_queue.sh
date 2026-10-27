#!/usr/bin/env bash
# run_bdg_queue.sh — 벤치마크 BD-G (Ω 를 Gurobi 로 푸는 Benders, belief cut 없음) 큐 (2026-10-22 사용자 결정, 토요일 발표용 경향 분석).
#   0. (앞에 다른 julia 가 끝날 때까지 기다림) zero-sum 상계 비교 진단 (Polska, Abilene; 방법당 300 s)
#   1. 비제로섬 입지 L1~L5, S = 10, 시드 1~3 → nonzero_sum/logs/bd_g/bdg_<L>_S10.log
#   2. zero-sum 네트워크 차단 5 개, S = 10 → true_dro/factor_5/logs/nig_coupled/S_10/<net>_bdg.log
# 끝난 로그 (RESULT 줄 수 ≥ 시드 수) 는 건너뜀. 요약은 표준출력 (=== 줄).
# 실행: bash nonzero_sum/run_bdg_queue.sh >> nonzero_sum/logs/bd_g/queue.out 2>&1
cd /c/Users/user/robust_bilevel_network_games || exit 1
L=nonzero_sum/logs/bd_g; mkdir -p "$L"
nres() { if [ -f "$1" ]; then grep -c '^RESULT' "$1"; else echo 0; fi; }
nopt() { if [ -f "$1" ]; then grep '^RESULT' "$1" | grep -c 'status=Optimal'; else echo 0; fi; }
until ! powershell -NoProfile -Command "if (Get-Process julia -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 60; done

if [ ! -f "$L/diag_bound_compare_zs.log" ] || ! grep -q "KKT   :.*" "$L/diag_bound_compare_zs.log" 2>/dev/null; then
  echo "=== 시작 $(date '+%m-%d %H:%M'): zero-sum 상계 비교 진단"
  (cd true_dro/factor_5 && DZ_T=300 DZ_THREADS=2 DZ_NETS=polska,abilene julia -t 4,1 diag_bound_compare_zs.jl) > "$L/diag_bound_compare_zs.log" 2>&1
  echo "=== 끝남 $(date '+%m-%d %H:%M'): zero-sum 상계 비교 진단"
fi

LIST=("L1 15 2 3 3" "L2 15 3 2 2" "L3 20 2 3 3" "L4 20 3 2 2" "L5 25 2 3 3")
for item in "${LIST[@]}"; do
  set -- $item
  log="$L/bdg_$1_S10.log"
  if [ "$(nres "$log")" -ge 3 ]; then echo "=== 건너뜀 (이미 끝남): $log"; continue; fi
  echo "=== 시작 $(date '+%m-%d %H:%M'): $log 시드 1~3"
  PP_METHOD=omega PP_NLOC=$2 PP_NA=$3 PP_NB=$4 PP_NBMAX=$5 PP_QB=750 PP_LAMBDA=2000 PP_S=10 PP_SEEDS="1,2,3" PP_TL=3600 PP_WORKERS=12 \
    julia -t 2 nonzero_sum/run_bd_kkt_pooled.jl > "$log" 2>&1
  echo "=== 끝남 $(date '+%m-%d %H:%M'): $log  Optimal $(nopt "$log")/$(nres "$log")"
done

echo "=== 시작 $(date '+%m-%d %H:%M'): zero-sum BD-G S=10"
(cd true_dro/factor_5 && NS_METHOD=omega NS_S=10 julia -t 2 run_nig_coupled_bdkkt.jl)
echo "=== 끝남 $(date '+%m-%d %H:%M'): zero-sum BD-G S=10"
echo "=== BD-G 큐 끝"
