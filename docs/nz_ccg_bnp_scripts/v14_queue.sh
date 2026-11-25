#!/usr/bin/env bash
# 2026-11-25 v14 (v13 + 1 단계 oracle 상한 300 s + 하한이 생기면 마감 루트 CG 몫을 oracle 에) 큐. md 11.12.
#  v13 큐 (v13_queue.sh) 는 지리 5 S=10 까지 돌리고 멈춤 (그 julia 가 끝나기를 기다린 뒤 시작).
#  0) 점검: 지리 13 S=10 TL=400 (마감 경로를 짧게 지나가게) — RESULT 가 없으면 큐 중단
#  1) S=20 지리 1 (v11 0.526%, v12 1.965%, v13 1.795%) 2) S=10 지리 1 (같은 답 확인, v11 0.500% 3,341 s)
#  3) v11 + κ 수정만 지리 5 S=10 (κ 몫과 일정 규칙 몫을 나누기 위한 것, 맨 뒤)
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; mkdir -p "$SPD"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/$SCRIPT" > "$tag.log" 2>&1; echo "== $tag EXIT $?"; grep -E "RESULT|시간:|v12:|v13:|v14:" "$tag.log"; }
# v13 지리 5 가 끝날 때까지 (RESULT 나 오류 줄, 최대 2.5 시간)
for i in $(seq 1 150); do grep -qE "^RESULT|^ERROR" bp13_s10_g5.log 2>/dev/null && break; sleep 60; done
sleep 30; echo "== bp13_s10_g5 끝 (대기 ${i} 분)"; grep -E "RESULT|시간:|v12:|v13:" bp13_s10_g5.log
SCRIPT=ccgbp_v14.jl
run bp14_s10_g13_tl400 GEO=13 S=10 TL=400
grep -q RESULT bp14_s10_g13_tl400.log || { echo "점검 실패 — 큐 중단"; exit 1; }
run bp14_s20_g1 GEO=1 S=20 TL=7200
run bp14_s10_g1 GEO=1 S=10 TL=3600
SCRIPT=ccgbp_v11.jl
run bp11k_s10_g5 GEO=5 S=10 TL=7200
echo "QUEUE DONE"
