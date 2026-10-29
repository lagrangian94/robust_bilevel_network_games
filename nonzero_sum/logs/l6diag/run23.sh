# run23 (2026-10-30): S50 1 시간, infeas_pre + parent_basis + 재시도 마감 + node_lp_cap=120. x̄={2,3}, seed 1, 3600 s
#   같이 NZ_INFEAS_DUMP: 사전 판정이 못 잡고 전체 LP 가 INFEASIBLE 로 끝난 노드를 infdump/S50miss 에 최대 8 개 저장
#   (최적 solve 마다 pre_*.bas 를 쓰는 비용이 조금 듦)
cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag/lpm
D=nonzero_sum/logs/l6diag/infdump/S50miss
B="range_lp=true,dd=false,farkas=true"
rm -f $L/cap120_1h_S50.nodelp
NZ_INFEAS_DUMP=$D NZ_INFEAS_DUMP_MAX=8 NZ_NODELP_LOG=$L/cap120_1h_S50.nodelp DL_SEED=1 DL_S=50 DL_X="2,3" DL_TL=3600 DL_PRE=0 DL_SOLVERS=abbdr \
  DL_ABBKW="$B,infeas_pre=true,parent_basis=true,node_lp_cap=120.0" DL_WORKERS=12 \
  julia -t 14,1 nonzero_sum/diag_l6.jl > $L/cap120_1h_S50.log 2>&1
echo RUN23_DONE
