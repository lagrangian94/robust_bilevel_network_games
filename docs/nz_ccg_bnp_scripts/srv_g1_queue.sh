#!/usr/bin/env bash
# 2026-11-27: cykim (Ryzen 9 9950X, 32 스레드) 에서 B&P 기본 판 (ccgbp.jl = v17, DD 끔, 인스턴스는 Julia 1.11 시드 hash 로 고정한 srv_src/loc_tkp_srv.jl) 지리 1 S=10 (TL 3600) → S=20 (TL 7200). 사용자 요청 "좋은 사양에선 얼마나 풀리는지"
cd ~/rbng_runs || exit 1
export GRB_LICENSE_FILE=$HOME/gurobi.lic
J=$HOME/.juliaup/bin/julia
run() { local tag=$1; shift; env "$@" $J -t 32,1 ccgbp_srv.jl > "$tag.log" 2>&1; echo "== $tag EXIT $? $(date +%H:%M)"; grep -E "^RESULT" "$tag.log" | cut -c1-320; }
echo "시작 $(date +%H:%M)"
run srv_bp17_s10_g1 GEO=1 S=10 TL=3600
run srv_bp17_s20_g1 GEO=1 S=20 TL=7200
echo "QUEUE DONE $(date +%H:%M)"
