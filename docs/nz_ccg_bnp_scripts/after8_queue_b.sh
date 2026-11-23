#!/usr/bin/env bash
# 지리 8 개 큐 (bp_rest8) 뒤 분석 큐 (2026-11-24 새벽, 지리 14 재실행 추가): v7 (oracle 이어 풀기) 지리 6 → oracle 재현 S=10·S=20 → v7 S=20 지리 1 → S=20 ablation
cd "C:/Users/user/AppData/Local/Temp/claude/b55b069927d98669/5ed04639-c8f0-4837-9037-f61ff4814c8e/scratchpad"
until grep -q QUEUE_DONE "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/5ed04639-c8f0-4837-9037-f61ff4814c8e/tasks/b3dgp7dcc.output" 2>/dev/null; do sleep 60; done
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
jl() { julia -t 12,1 --project="$P" "$@"; }
# 0. 지리 14 재실행 (기본 판, closed_lb KeyError 수정 뒤)
GEO=14 S=10 TL=7200 jl "$D/ccgbp.jl" > bpr8_g14b.log 2>&1; echo "g14b EXIT $?"; grep -E "RESULT|시간:" bpr8_g14b.log
# 1. v7 지리 6 (v6 는 TimeLimit 0.507%, 같은 점 oracle 을 처음부터 세 번)
GEO=6 S=10 TL=7200 jl "$D/ccgbp_v7.jl" > bp7_g6.log 2>&1; echo "v7 g6 EXIT $?"; grep -E "RESULT|시간:" bp7_g6.log
# 2. oracle 재현
export LAMU=1e4 GEO=1 GAP=1e-3
S=10 XSTR="{9:200;10:300;15:300}" TAU=-3673.58 TLIM=900 jl "$D/orcl_replay.jl" > orep_s10_g1.log 2>&1; echo "rep S10 EXIT $?"; grep -E "RESULT|전선:" orep_s10_g1.log
S=20 XSTR="{9:200;10:300;15:200}" TAU=-3437.59 TLIM=3600 jl "$D/orcl_replay.jl" > orep_s20_g1.log 2>&1; echo "rep S20 EXIT $?"; grep -E "RESULT|전선:" orep_s20_g1.log
unset LAMU GAP
# 3. v7 S=20 지리 1
GEO=1 S=20 TL=7200 jl "$D/ccgbp_v7.jl" > bp7s20_g1.log 2>&1; echo "v7 S20 g1 EXIT $?"; grep -E "RESULT|시간:" bp7s20_g1.log
# 4. S=20 ablation (기존 옵션만)
export LAMU=1e4 GEO=1 GAP=1e-3 S=20 XSTR="{9:200;10:300;15:200}" TAU=-3437.59 TLIM=1800
run() { TAG="$1" KW="$2" jl "$D/orcl_replay.jl" > "orep_s20_g1$1.log" 2>&1; echo "$1 EXIT $?"; grep -E "RESULT|전선:" "orep_s20_g1$1.log"; grep -m2 "ERROR\|Exception" "orep_s20_g1$1.log"; }
run _resp "(resp=true,)"
run _bar "(lp_method=2, lp_crossover=0)"
run _cr3 "(child_rounds=3,)"
run _rmip "(range_lp=false,)"
echo QUEUE_DONE
