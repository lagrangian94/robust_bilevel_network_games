#!/usr/bin/env bash
# 2026-11-25 λᵁ 확인: v9·HUCAP 큐 (b9tl2rwr6) 뒤 지리 5·S=20 지리 1 oracle 재현을 λᵁ 10³·2·10³ 으로 (목표 τ 는 λᵁ=10⁴ 마스터 값 그대로, oracle UB 는 어떤 λ 에서도 유효)
cd "C:/Users/user/AppData/Local/Temp/claude/b55b069927d98669/5ed04639-c8f0-4837-9037-f61ff4814c8e/scratchpad"
until grep -q "QUEUE_DONE" "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/5ed04639-c8f0-4837-9037-f61ff4814c8e/tasks/b9tl2rwr6.output" 2>/dev/null; do sleep 60; done
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
export GAP=1e-3
run() { local tag="$1"; shift; env "$@" TAG="$tag" julia -t 12,1 --project="$P" "$D/orcl_replay.jl" > "orep$tag.log" 2>&1; echo "$tag EXIT $?"; grep -E "RESULT|전선:" "orep$tag.log"; grep -m2 "ERROR\|Exception" "orep$tag.log"; }
G5="GEO=5 S=10 XSTR={4:200;7:200;12:300} TAU=-3150.62"
G1S20="GEO=1 S=20 XSTR={9:200;10:300;15:200} TAU=-3437.59"
run _s10_g5_lam1e3 $G5 TLIM=1800 LAMU=1e3 "KW=(resp=true,)"
run _s10_g5_lam2e3 $G5 TLIM=1800 LAMU=2e3 "KW=(resp=true,)"
run _s20_g1_lam1e3 $G1S20 TLIM=1800 LAMU=1e3 "KW=(resp=true,)"
run _s20_g1_lam2e3 $G1S20 TLIM=1800 LAMU=2e3 "KW=(resp=true,)"
echo QUEUE_DONE
