#!/usr/bin/env bash
# 2026-11-28 (사용자 승인 "ㅇㅇㅇㅇ"): 선별 DD 를 저장소 nz_alpha_bnb_dr 로 옮긴 판 점검.
#   (1) test_nz_alpha_bnb_dr.jl: 작은 인스턴스 (S = 3) 에서 기본 구성과 dd = true 의 [LB, UB] 가 독립 KKT 평가기 V* 를 담는지, NZ_SD_CHECK = 1 (모든 강쌍대 MIP 대조)
#   (2) orcl_replay.jl: 지리 5 막힌 점 300 s, 저장소 판 (ABB_PATCH 없음) + NZ_SD_CHECK 5·NZ_SD_TIGHT 1 — 패치 판 logs/orep_g5_sdchk300.log 와 같은 설정
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
echo "시작 $(date +%H:%M)"
env TD_S=3 TD_X="1,2;1;2" TD_TL=60 TD_MODES="on,kw" TD_KW="(dd=true,)" NZ_SD_CHECK=1 TD_WORKERS=4 julia -t 6,1 --project="$P" "$P/nonzero_sum/test_nz_alpha_bnb_dr.jl" > port_test_s3.log 2>&1
echo "== port_test_s3 EXIT $? $(date +%H:%M)"
env NZ_SD_CHECK=5 NZ_SD_TIGHT=1 HUCAP=1 GEO=5 S=10 XSTR="{4:200;7:200;12:300}" TAU=-3151.60 TLIM=300 LOG=60 KW="(resp=true, dd=true)" TAG=_portchk \
    julia -t 12,1 --project="$P" "$D/orcl_replay.jl" > orep_g5_portchk300.log 2>&1
echo "== orep_g5_portchk300 EXIT $? $(date +%H:%M)"; grep -E "^RESULT|DD 선별|^전선:|dd_sdcut" orep_g5_portchk300.log | cut -c1-600
echo "QUEUE DONE $(date +%H:%M)"
