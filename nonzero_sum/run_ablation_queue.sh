#!/usr/bin/env bash
# run_ablation_queue.sh — 2×2 ablation 의 빈 칸 "Algorithm 2 (belief cut + local) + Gurobi Ω oracle" (PP_ORACLE=omega_grb, 2026-10-27 사용자 결정).
#   나머지 세 칸: 우리 방법 (main_*), BD-G (bd_g/bdg_*), local 만 + α-B&B 는 원고의 SB-A.
#   BD-G 와는 belief cut 유무만, 우리 방법과는 전역 oracle 만 다르다. 설정은 본 실험과 같음 (1 시간, 허용오차 0.5%, Gurobi 스레드 12).
#   run_bdg_s50_queue.sh 가 끝난 뒤 (queue.out 의 끝 줄) 시작. L1~L5 × S = 10, 20 × 시드 1~3 → nonzero_sum/logs/abl/abl_omegagrb_<L>_S<S>.log
# 실행: bash nonzero_sum/run_ablation_queue.sh >> nonzero_sum/logs/abl/queue.out 2>&1
cd /c/Users/user/robust_bilevel_network_games || exit 1
L=nonzero_sum/logs/abl; mkdir -p "$L"
nres() { if [ -f "$1" ]; then grep -c '^RESULT' "$1"; else echo 0; fi; }
nopt() { if [ -f "$1" ]; then grep '^RESULT' "$1" | grep -c 'status=Optimal'; else echo 0; fi; }
until grep -q "=== BD-G S50 큐 끝" nonzero_sum/logs/bd_g/queue.out; do sleep 120; done
until ! powershell -NoProfile -Command "if (Get-Process julia -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 60; done

LIST=("L1 15 2 3 3" "L2 15 3 2 2" "L3 20 2 3 3" "L4 20 3 2 2" "L5 25 2 3 3")
for S in 10 20; do
  for item in "${LIST[@]}"; do
    set -- $item
    log="$L/abl_omegagrb_$1_S$S.log"
    if [ "$(nres "$log")" -ge 3 ]; then echo "=== 건너뜀 (이미 끝남): $log"; continue; fi
    echo "=== 시작 $(date '+%m-%d %H:%M'): $log 시드 1~3"
    PP_ORACLE=omega_grb PP_NLOC=$2 PP_NA=$3 PP_NB=$4 PP_NBMAX=$5 PP_QB=750 PP_LAMBDA=2000 PP_S=$S PP_SEEDS="1,2,3" PP_TL=3600 PP_WORKERS=12 \
      julia -t 14,1 nonzero_sum/pilot_pooled_store.jl > "$log" 2>&1
    echo "=== 끝남 $(date '+%m-%d %H:%M'): $log  Optimal $(nopt "$log")/$(nres "$log")"
  done
done
echo "=== ablation 큐 끝"
