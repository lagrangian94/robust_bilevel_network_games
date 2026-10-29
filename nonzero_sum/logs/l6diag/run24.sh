# run24 (2026-10-30): infeas_pre_retry. (1) S20 smoke 300 s + NZ_PRE_CHECK (사전 판정으로 가지친 노드를 전체 LP 로 재확인),
#   (2) S50 1 시간 = run23 (cap120_1h_S50) 과 같은 조건 + infeas_pre_retry (덤프 없음). x̄={2,3}, seed 1
cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag/lpm
B="range_lp=true,dd=false,farkas=true,infeas_pre=true,parent_basis=true,node_lp_cap=120.0"
rm -f $L/smoke_retry_S20.nodelp $L/retry_1h_S50.nodelp
NZ_PRE_CHECK=1 NZ_NODELP_LOG=$L/smoke_retry_S20.nodelp DL_SEED=1 DL_S=20 DL_X="2,3" DL_TL=300 DL_PRE=0 DL_SOLVERS=abbdr \
  DL_ABBKW="$B,infeas_pre_retry=true" DL_WORKERS=12 julia -t 14,1 nonzero_sum/diag_l6.jl > $L/smoke_retry_S20.log 2>&1
echo SMOKE_DONE
grep -q "^RESULT" $L/smoke_retry_S20.log || { echo SMOKE_FAILED; exit 1; }
NZ_NODELP_LOG=$L/retry_1h_S50.nodelp DL_SEED=1 DL_S=50 DL_X="2,3" DL_TL=3600 DL_PRE=0 DL_SOLVERS=abbdr \
  DL_ABBKW="$B,infeas_pre_retry=true" DL_WORKERS=12 julia -t 14,1 nonzero_sum/diag_l6.jl > $L/retry_1h_S50.log 2>&1
echo RUN24_DONE
