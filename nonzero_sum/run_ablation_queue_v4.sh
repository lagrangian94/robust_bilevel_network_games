#!/bin/bash
# run_ablation_queue_v4.sh — 어려운 인스턴스에서 벤치마크 BD-G 확장 + 2×2 ablation (2026-11-01 사용자 결정, 새 α-B&B 기본값)
#   2×2:            | Gurobi Ω (전역)                      | α-B&B (전역)
#     belief cut 켬 | PP_ORACLE=omega_grb  → abl/grb2_*     | 우리 방법 (재실험 v2, main2_list_*)
#     belief cut 끔 | BD-G (run_bd_kkt_pooled.jl) → bd_g/* | PP_BELIEF=0 → abl/nob_*
#   인스턴스 (우리 방법과 BD-G 의 차이가 꼬리에서 난 곳): L4 S20, L2 S50, L4 S50, L6 S10, L6 S20. 시드 1~10, 1 시간, 허용오차 0.5%, λᵁ=2000.
#   BD-G 시드 1~3 은 이미 있음 (bd_g/bdg_{L4_S20,L2_S50,L4_S50}.log, α-B&B 를 안 쓰므로 기본값 변경과 무관) → 시드 4~10 만. L6 는 시드 1~10.
#   반응 프리솔브 PP_PRE: 우리 방법과 같게 (L2·L4 = 1, L6 = 0). 멈춤 규칙 없음 (비교 대상이라 모든 칸을 같은 인스턴스로 채움).
#   순서: 인스턴스마다 BD-G → belief 끔 → Gurobi oracle (한 인스턴스의 2×2 를 먼저 채움). L6 S20 이 마지막.
#   0 단계: run_pooled_store_queue_v3.sh 가 끝날 때까지 (QUEUE_V3_DONE) 기다린 뒤, PP_BELIEF=0 점검 (L1 S10 seed 1: x* 가 v2 와 같고 값 차 ≤ 0.5% 아니면 멈춤).
# 실행: bash nonzero_sum/run_ablation_queue_v4.sh >> nonzero_sum/logs/abl/queue_v4.out 2>&1
cd "$(dirname "$0")/.." || exit 1
L=nonzero_sum/logs; A=$L/abl; G=$L/bd_g; mkdir -p "$A" "$G"
nres() { if [ -f "$1" ]; then grep -c '^RESULT' "$1"; else echo 0; fi; }
nopt() { if [ -f "$1" ]; then grep '^RESULT' "$1" | grep -c 'status=Optimal'; else echo 0; fi; }
nojulia() { ! powershell -NoProfile -Command "if (Get-Process julia -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; }
until grep -q QUEUE_V3_DONE "$L/run_pooled_store_queue_v3.out" 2>/dev/null && nojulia; do sleep 60; done
echo "=== v4 시작 $(date '+%m-%d %H:%M')"

# ---- 0. PP_BELIEF=0 점검 ----
chk=$A/nob_check_L1_S10_s1.log
if [ "$(nres "$chk")" -lt 1 ]; then
  PP_BELIEF=0 PP_NLOC=15 PP_NA=2 PP_NB=3 PP_NBMAX=3 PP_QB=750 PP_LAMBDA=2000 PP_S=10 PP_SEEDS=1 PP_TL=3600 PP_WORKERS=12 PP_PRE=1 \
    julia -t 14,1 nonzero_sum/pilot_pooled_store.jl > "$chk" 2>&1
fi
fx() { grep '^RESULT' "$1" | grep " seed=$2 " | sed -E 's/.* x=(\{[^}]*\}) .*/\1/'; }
fub() { grep '^RESULT' "$1" | grep " seed=$2 " | sed -E 's/.* UB=([^ ]+) .*/\1/'; }
ref=$L/main2_list_L1_d15_A2B3_S10.log
x0=$(fx "$ref" 1); x1=$(fx "$chk" 1); u0=$(fub "$ref" 1); u1=$(fub "$chk" 1)
grep -q 'belief cut 끔' "$chk" || { echo "점검 실패: 로그에 'belief cut 끔' 설정 줄 없음"; exit 1; }
ok=$(awk -v a="$u0" -v b="$u1" 'BEGIN{d=a-b; if(d<0)d=-d; m=a<0?-a:a; if(m<1)m=1; print (d/m<=5e-3)?1:0}')
if [ -z "$x1" ] || [ "$x0" != "$x1" ] || [ "$ok" != "1" ]; then echo "점검 실패: v2 x=$x0 UB=$u0, belief=0 x=$x1 UB=$u1 → 멈춤"; exit 1; fi
echo "점검 통과 $(date '+%m-%d %H:%M'): belief=0 L1 S10 seed 1 x=$x1 UB=$u1 (v2 UB=$u0), $(grep '^RESULT' "$chk" | sed -E 's/.*(iters=[0-9]+ global=[0-9]+ local=[0-9]+ menu=[0-9]+ time=[0-9.]+).*/\1/')"

run() {  # 칸 이름 d A B NB S seeds pre log
  local cell=$1 d=$2 a=$3 b=$4 nb=$5 S=$6 seeds=$7 pre=$8 log=$9
  local want; want=$(echo "$seeds" | tr ',' '\n' | wc -l)
  if [ "$(nres "$log")" -ge "$want" ]; then echo "건너뜀 (이미 끝남): $log"; return; fi
  echo "=== 시작 $(date '+%m-%d %H:%M'): $cell $log"
  local env="PP_NLOC=$d PP_NA=$a PP_NB=$b PP_NBMAX=$nb PP_QB=750 PP_LAMBDA=2000 PP_S=$S PP_SEEDS=$seeds PP_TL=3600 PP_WORKERS=12"
  case $cell in
    BD-G) env $env PP_METHOD=omega julia -t 2 nonzero_sum/run_bd_kkt_pooled.jl > "$log" 2>&1 ;;
    nob)  env $env PP_PRE=$pre PP_BELIEF=0 julia -t 14,1 nonzero_sum/pilot_pooled_store.jl > "$log" 2>&1 ;;
    grb)  env $env PP_PRE=$pre PP_ORACLE=omega_grb julia -t 14,1 nonzero_sum/pilot_pooled_store.jl > "$log" 2>&1 ;;
  esac
  echo "=== 끝 $(date '+%m-%d %H:%M'): $cell $log  Optimal $(nopt "$log")/$(nres "$log")"
}
ALL="1,2,3,4,5,6,7,8,9,10"; NEW="4,5,6,7,8,9,10"
# 이름 d A B NB S pre BD-G시드 BD-G로그
LIST=("L4 20 3 2 2 20 1 $NEW bdg_L4_S20_s4-10" "L2 15 3 2 2 50 1 $NEW bdg_L2_S50_s4-10" "L4 20 3 2 2 50 1 $NEW bdg_L4_S50_s4-10"
      "L6 25 5 5 5 10 0 $ALL bdg_L6_S10" "L6 25 5 5 5 20 0 $ALL bdg_L6_S20")
for e in "${LIST[@]}"; do set -- $e
  run BD-G $2 $3 $4 $5 $6 "$8" $7 "$G/$9.log"
  run nob  $2 $3 $4 $5 $6 "$ALL" $7 "$A/nob_$1_S$6.log"
  run grb  $2 $3 $4 $5 $6 "$ALL" $7 "$A/grb2_$1_S$6.log"
done
echo "=== QUEUE_V4_DONE $(date '+%m-%d %H:%M')"
