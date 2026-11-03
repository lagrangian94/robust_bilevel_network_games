#!/bin/bash
# run_queue_v5.sh — 최종 실험 목록 (2026-11-03 사용자 확정, 메모리 nz_target_instances / bd_kkt_benchmark)
#   0. 큐 v4 축소: nob L6 S10 이 끝나면 v4 (cert_then_v4.sh, run_ablation_queue_v4.sh) 와 그 뒤에 시작된 julia (grb L6 S10) 를 멈춤.
#      grb L6 S10 과 L6 S20 의 세 칸은 돌리지 않음 (2×2 결론이 세 인스턴스에서 같음, BD-G L6 S10 < 5/10 이면 멈춤 규칙).
#   1. V1 기본 인스턴스 (SGB128 거점별 예약 w=150, S=3): loc_insample.jl (모든 x, 4 모형, δ sweep, w=0 대조, KKT 전수)
#   2. V2 기본 인스턴스 OOS: loc_oos.jl (Close / Moderate / Misspecified)
#   3. 벤치마크 E (열거 + α-B&B, 가지치기): L1~L5 S10 시드 1~10
#   4. zero-sum S10: 시드 42,1,2,3,4 × 5 네트워크 × {menu (Ours), nob (belief cut 끔)} (seed 42 menu 는 이미 있음 → 건너뜀)
#   5. δ·β 민감도 (Ours): L2·L4 S20 시드 1~10, δ ∈ {0, 0.3, 0.6 (= ε̂+ε̃, product)} (β 0.4), β ∈ {0, 0.8} (δ 0.1)
#   6. BD-G 나머지 칸을 시드 1~10 으로: L1·L3·L5 S10/20/50, L2 S10/20, L4 S10 (시드 1~3 은 bd_g/bdg_*.log 에 있음 → 4~10, L5 S50 은 1~10)
#   (δ·β 실행의 λ 무관 인증은 cert_lambda_free.jl 이 β·δ 를 읽도록 고쳐야 해서 사용자 허락 뒤 따로)
#   julia 는 한 번에 하나. 단계마다 julia 가 0 이 아닌 코드로 끝나면 큐를 멈춤 (오류를 고친 뒤 다시 실행하면 끝난 로그는 건너뜀).
# 실행: bash nonzero_sum/run_queue_v5.sh >> nonzero_sum/logs/queue_v5.out 2>&1
cd "$(dirname "$0")/.." || exit 1
R=$(pwd); L=nonzero_sum/logs; mkdir -p "$L/enumE" "$L/sens" "$L/bd_g"
nres() { if [ -f "$1" ]; then grep -c '^RESULT' "$1"; else echo 0; fi; }
nopt() { if [ -f "$1" ]; then grep '^RESULT' "$1" | grep -c 'status=Optimal'; else echo 0; fi; }
ps1() { powershell -NoProfile -Command "$1"; }
nojulia() { ! ps1 "if (Get-Process julia -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; }
stamp() { date '+%m-%d %H:%M'; }
fail() { echo "=== 멈춤 $(stamp): $1 (exit $2)"; exit 1; }

# ---- 0. 큐 v4 축소 ----
V4=$L/abl/queue_v4.out
if ! grep -q 'QUEUE_V4_DONE\|v4 멈춤 (v5)' "$V4" 2>/dev/null; then
  echo "=== v5 대기 $(stamp): nob L6 S10 끝 (queue_v4.out) 을 기다림"
  until grep -q '=== 끝 .*nob nonzero_sum/logs/abl/nob_L6_S10.log' "$V4" 2>/dev/null; do sleep 60; done
  # v4 bash 를 먼저 멈춘 뒤 (다음 칸을 띄우지 못하게) 그 뒤에 시작된 julia (grb L6 S10) 를 멈춤
  ps1 "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -match 'run_ablation_queue_v4|cert_then_v4' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
  sleep 5
  ps1 "Get-Process julia -ErrorAction SilentlyContinue | Stop-Process -Force"
  sleep 5
  [ -f "$L/abl/grb2_L6_S10.log" ] && mv "$L/abl/grb2_L6_S10.log" "$L/abl/grb2_L6_S10.aborted_v5.log"
  echo "=== v4 멈춤 (v5) $(stamp): grb L6 S10, L6 S20 세 칸은 돌리지 않음" | tee -a "$V4"
fi
until nojulia; do sleep 30; done
echo "=== v5 시작 $(stamp)"

# ---- 1. V1 기본 인스턴스 ----
V1=$L/loc_insample_sgb128_pooled_store.log
if ! grep -q '^CSV:' "$V1" 2>/dev/null; then
  echo "=== 시작 $(stamp): V1 $V1"
  LI_RES=pooled_store LI_QUOTAS=0,150 LI_EPS=0.3 LI_BETA=0.4 LI_TAB_DELTAS=0.1,0 LI_FIG_EPST=0.3 LI_FIG_QUOTA=150 LI_TL=300 NZ_S=3 NZ_SEED=1 \
    julia nonzero_sum/loc_insample.jl > "$V1" 2>&1 || fail "V1" $?
  echo "=== 끝 $(stamp): V1 (시간 제한 x: $(grep -c ',TL,' $L/loc_insample_sgb128_pooled_store_S3_seed1_eps0.30_beta0.40.csv))"
fi

# ---- 2. V2 OOS ----
V2=$L/loc_oos_sgb128_pooled_store.log
if ! grep -q '^LOC_OOS_DONE' "$V2" 2>/dev/null; then
  echo "=== 시작 $(stamp): V2 $V2"
  NZ_S=3 NZ_SEED=1 LO_QUOTA=150 LO_EPS=0.3 LO_DELTA0=0.1 LO_BETA=0.4 julia nonzero_sum/loc_oos.jl > "$V2" 2>&1 || fail "V2" $?
  echo "=== 끝 $(stamp): V2"
fi

ALL="1,2,3,4,5,6,7,8,9,10"
# 이름 d A B NB
LIST=("L1 15 2 3 3" "L2 15 3 2 2" "L3 20 2 3 3" "L4 20 3 2 2" "L5 25 2 3 3")
penv() { echo "PP_NLOC=$2 PP_NA=$3 PP_NB=$4 PP_NBMAX=$5 PP_QB=750 PP_LAMBDA=2000 PP_TL=3600 PP_WORKERS=12 PP_PRE=1"; }

# ---- 3. E (열거) ----
for e in "${LIST[@]}"; do set -- $e
  log=$L/enumE/E_$1_S10.log
  [ "$(nres "$log")" -ge 10 ] && { echo "건너뜀 (이미 끝남): $log"; continue; }
  echo "=== 시작 $(stamp): E $log"
  env $(penv "$@") PP_S=10 PP_SEEDS=$ALL julia -t 14,1 nonzero_sum/run_enum_E.jl > "$log" 2>&1 || fail "E $1" $?
  echo "=== 끝 $(stamp): E $log  Optimal $(nopt "$log")/$(nres "$log")"
done

# ---- 4. zero-sum S10 시드 (Ours, belief cut 끔) ----
ZS=true_dro/factor_5/logs/nig_coupled_seeds.stdout
if ! grep -q 'NIG coupled scaling done' "$ZS" 2>/dev/null; then
  echo "=== 시작 $(stamp): zero-sum S10 시드 42,1,2,3,4 × menu,nob ($ZS)"
  (cd true_dro/factor_5 && NS_S=10 NS_VARIANTS_S10=menu,nob NS_SEEDS=42,1,2,3,4 julia -t 14,1 run_nig_coupled_scaling.jl >> logs/nig_coupled_seeds.stdout 2>&1) || fail "zero-sum" $?
  echo "=== 끝 $(stamp): zero-sum ($(grep -h '^RESULT' true_dro/factor_5/logs/nig_coupled/S_10/*_menu*.log true_dro/factor_5/logs/nig_coupled/S_10/*_nob*.log | grep -c 'status=Optimal') Optimal)"
fi

# ---- 5. δ·β 민감도 (Ours) ----
for e in "${LIST[@]}"; do set -- $e
  [ "$1" = L2 ] || [ "$1" = L4 ] || continue
  for v in "d0 0 0.4" "d0.3 0.3 0.4" "d0.6 0.6 0.4" "b0 0.1 0" "b0.8 0.1 0.8"; do set -- $e $v
    log=$L/sens/sens_$1_S20_$6.log
    [ "$(nres "$log")" -ge 10 ] && { echo "건너뜀 (이미 끝남): $log"; continue; }
    echo "=== 시작 $(stamp): sens $log (δ=$7 β=$8)"
    env $(penv "$1" "$2" "$3" "$4" "$5") PP_S=20 PP_SEEDS=$ALL PP_DELTA=$7 PP_BETA=$8 julia -t 14,1 nonzero_sum/pilot_pooled_store.jl > "$log" 2>&1 || fail "sens $1 $6" $?
    echo "=== 끝 $(stamp): sens $log  Optimal $(nopt "$log")/$(nres "$log")"
  done
done

# ---- 6. BD-G 나머지 칸 ----
# 이름 d A B NB S 시드 로그
BL=("L1 15 2 3 3 10 4-10" "L1 15 2 3 3 20 4-10" "L2 15 3 2 2 10 4-10" "L2 15 3 2 2 20 4-10" "L3 20 2 3 3 10 4-10" "L3 20 2 3 3 20 4-10"
    "L4 20 3 2 2 10 4-10" "L5 25 2 3 3 10 4-10" "L5 25 2 3 3 20 4-10" "L1 15 2 3 3 50 4-10" "L3 20 2 3 3 50 4-10" "L5 25 2 3 3 50 1-10")
for e in "${BL[@]}"; do set -- $e
  if [ "$7" = 1-10 ]; then sd=$ALL; log=$L/bd_g/bdg_$1_S$6.log; want=10; else sd="4,5,6,7,8,9,10"; log=$L/bd_g/bdg_$1_S$6_s4-10.log; want=7; fi
  [ "$(nres "$log")" -ge $want ] && { echo "건너뜀 (이미 끝남): $log"; continue; }
  echo "=== 시작 $(stamp): BD-G $log"
  env PP_NLOC=$2 PP_NA=$3 PP_NB=$4 PP_NBMAX=$5 PP_QB=750 PP_LAMBDA=2000 PP_S=$6 PP_SEEDS=$sd PP_TL=3600 PP_WORKERS=12 PP_METHOD=omega \
    julia -t 2 nonzero_sum/run_bd_kkt_pooled.jl > "$log" 2>&1 || fail "BD-G $1 S$6" $?
  echo "=== 끝 $(stamp): BD-G $log  Optimal $(nopt "$log")/$(nres "$log")"
done
echo "=== QUEUE_V5_DONE $(stamp)"
