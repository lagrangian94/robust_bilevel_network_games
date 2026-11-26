#!/usr/bin/env bash
# 2026-11-24 지리 5·인증 분석: 기준 재현 (b2oe1m7ef) 뒤 → 지리 5 incumbent 정확값 확인 → v9 (정확값 위반 판정) 지리 5 → HUCAP·:alphahat 재현 비교
cd "C:/Users/user/AppData/Local/Temp/claude/b55b069927d98669/5ed04639-c8f0-4837-9037-f61ff4814c8e/scratchpad"
until grep -q "EXIT" "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/5ed04639-c8f0-4837-9037-f61ff4814c8e/tasks/b2oe1m7ef.output" 2>/dev/null; do sleep 60; done
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
julia -t 1 --project="$P" exact_check.jl orep_front_g5_S10_g5a.jls 2>&1 | grep -v "WLS\|LicenseID\|Academic"
GEO=5 S=10 TL=7200 julia -t 12,1 --project="$P" "$D/ccgbp_v9.jl" > bp9_s10_g5.log 2>&1; echo "v9 g5 EXIT $?"; grep -E "RESULT|시간:" bp9_s10_g5.log
export LAMU=1e4 GAP=1e-3
run() { local tag="$1"; shift; env "$@" TAG="$tag" julia -t 12,1 --project="$P" "$D/orcl_replay.jl" > "orep$tag.log" 2>&1; echo "$tag EXIT $?"; grep -E "RESULT|전선:|HUCAP" "orep$tag.log"; grep -m2 "ERROR\|Exception" "orep$tag.log"; }
G1S10="GEO=1 S=10 XSTR={9:200;10:300;15:300} TAU=-3673.58"
G5="GEO=5 S=10 XSTR={4:200;7:200;12:300} TAU=-3150.62"
G1S20="GEO=1 S=20 XSTR={9:200;10:300;15:200} TAU=-3437.59"
run _s10_g1_hucap $G1S10 TLIM=900 HUCAP=1
run _s10_g5_hucap $G5 TLIM=1800 HUCAP=1 "KW=(resp=true,)"
run _s20_g1_hucap $G1S20 TLIM=1800 HUCAP=1 "KW=(resp=true,)"
run _s20_g1_ahat $G1S20 TLIM=1800 "KW=(resp=true, branch_point=:alphahat)"
echo QUEUE_DONE
