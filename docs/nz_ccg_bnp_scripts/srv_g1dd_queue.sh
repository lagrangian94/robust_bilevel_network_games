#!/usr/bin/env bash
# 2026-11-27: cykim (32 스레드) 에서 B&P 기본 판 (ccgbp.jl = v17) + DD oracle (abb_dd_patch.jl 기본값 = :viol1·objstop·fcache) 지리 1 S=20, TL 7200 (사용자 "둘다").
#   비교: 같은 서버·같은 인스턴스 DD 끔 판 logs/srv_bp17_s20_g1.log (Optimal 0.299%, 3,945.8 s, oracle 8 번 2,751 s).
cd ~/rbng_runs || exit 1
export GRB_LICENSE_FILE=$HOME/gurobi.lic
J=$HOME/.juliaup/bin/julia; D=$HOME/robust_bilevel_network_games/docs/nz_ccg_bnp_scripts
run() { local tag=$1; shift; env "$@" $J -t 32,1 ccgbp_srv.jl > "$tag.log" 2>&1; echo "== $tag EXIT $? $(date +%H:%M)"; grep -E "^RESULT" "$tag.log" | cut -c1-320; }
echo "시작 $(date +%H:%M)"
run srv_bp17dd_s20_g1 ABB_PATCH="$D/abb_dd_patch.jl" ORCL_KW="(dd=true, dd_maxw=25.0)" GEO=1 S=20 TL=7200
echo "QUEUE DONE $(date +%H:%M)"
