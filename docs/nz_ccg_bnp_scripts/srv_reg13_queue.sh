#!/usr/bin/env bash
# 2026-11-27 밤: cykim 에서 B&P 기본 판 (ccgbp.jl = v17) S = 10 13 지리 회귀, DD 켬 (abb_dd_patch.jl 기본값 = :viol1·objstop·fcache) 대 DD 끔 (사용자 "서버에서 돌려").
#   두 큐를 동시에 16 스레드씩, 각자 폴더 (orcl_calls_*.jls 가 겹치지 않게). 지리 순서는 v14 회귀 (11.15) 와 같음. 인스턴스 지문은 inst_fp.jl 로 13 지리 모두 이 컴퓨터와 같음 확인.
#   사용: ./srv_reg13_queue.sh dd | nodd
mode=$1; [ "$mode" = dd ] || [ "$mode" = nodd ] || { echo "mode = dd | nodd"; exit 1; }
R=~/rbng_runs; W=$R/reg13_$mode; mkdir -p $W; cd $W || exit 1
export GRB_LICENSE_FILE=$HOME/gurobi.lic
J=$HOME/.juliaup/bin/julia; D=$HOME/robust_bilevel_network_games/docs/nz_ccg_bnp_scripts
if [ "$mode" = dd ]; then KW=(ABB_PATCH="$D/abb_dd_patch.jl" ORCL_KW="(dd=true, dd_maxw=25.0)"); else KW=(ABB_PATCH=""); fi
run() { local tag=$1; shift; env "$@" $J -t 16,1 $R/ccgbp_srv.jl > "$tag.log" 2>&1; echo "== $tag EXIT $? $(date +%H:%M)"; grep -E "^RESULT" "$tag.log" | cut -c1-320; }
echo "시작 $mode $(date +%H:%M)"
for g in 7 1 3 4 9 10 11 12 2 8 15 14 6; do run srv13${mode}_s10_g$g "${KW[@]}" GEO=$g S=10 TL=3600; done
echo "QUEUE DONE $(date +%H:%M)"
