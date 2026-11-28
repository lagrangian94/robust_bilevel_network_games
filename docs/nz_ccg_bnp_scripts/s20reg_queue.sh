#!/usr/bin/env bash
# 2026-11-29 (사용자 "전부. DD cut 을 디폴트로 ... 일단 이 컴에서 돌려볼까"): B&P 기본 판 (ccgbp.jl = v17) S = 20 15 지리 회귀, 이 컴퓨터 12 스레드, 한 번에 하나.
#   oracle = 저장소 nz_alpha_bnb_dr (70fab87) + ccgbp 기본 ORCL_KW = (dd=true,) → 선별 DD + 강쌍대 cut (dd_sdcut 기본 켬). ABB_PATCH 없음.
#   TL 7,200 s (11.25·11.26 의 S = 20 지리 1 과 같음), 나머지는 ccgbp 기본값. 지리 순서 = S = 10 서버 회귀 (11.27, DD 켬) 시간이 짧은 것부터, 지리 5·6 은 끝.
#   작업 폴더 = scratchpad/s20reg (orcl_calls_*.jls 가 다른 판과 안 섞이게). 사용: bash s20reg_queue.sh [지리 ...] (없으면 15 지리 전부)
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; mkdir -p "$SPD/s20reg"; cd "$SPD/s20reg" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
GEOS=${*:-"3 11 4 10 13 12 7 1 8 2 9 15 14 5 6"}
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$D/ccgbp.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $? $(date +%m-%d_%H:%M)"; grep -E "^RESULT" "$tag.log" | cut -c1-320; }
echo "시작 $(date +%m-%d_%H:%M) 지리 $GEOS, 저장소 $(git -C "$P" log --oneline -1 | cut -c1-7)"
for g in $GEOS; do run bp17_s20_g$g GEO=$g S=20 TL=7200; done
echo "QUEUE DONE $(date +%m-%d_%H:%M)"
