# run25 (2026-10-30): S50 1 시간 = run24 와 같은 조건, 재시도 조건 수정 (NUMERIC·검증 실패 INFEASIBLE 일 때만). x̄={2,3}, seed 1
cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag/lpm
B="range_lp=true,dd=false,farkas=true,infeas_pre=true,parent_basis=true,node_lp_cap=120.0"
rm -f $L/retry2_1h_S50.nodelp
NZ_NODELP_LOG=$L/retry2_1h_S50.nodelp DL_SEED=1 DL_S=50 DL_X="2,3" DL_TL=3600 DL_PRE=0 DL_SOLVERS=abbdr \
  DL_ABBKW="$B,infeas_pre_retry=true" DL_WORKERS=12 julia -t 14,1 nonzero_sum/diag_l6.jl > $L/retry2_1h_S50.log 2>&1
echo RUN25_DONE
