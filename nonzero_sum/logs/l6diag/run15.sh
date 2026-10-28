# run15 (2026-10-29): cutoff (분리 건너뛰기만, Gurobi Cutoff 안 씀) 처리량. NZ_PRUNE_CHECK 로 S20 가지치기 대조 먼저
cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag
KW="range_lp=true,dd=false,farkas=true,cutoff=true"
NZ_PRUNE_CHECK=1 DL_SEED=1 DL_S=20 DL_X="2,3" DL_TL=300 DL_PRE=0 DL_SOLVERS=abbdr DL_ABBKW="$KW" DL_WORKERS=12 \
  julia -t 14,1 nonzero_sum/diag_l6.jl > $L/prune_check_S20_s1.log 2>&1
grep -qE '^\s+prune_disagree = 0$' $L/prune_check_S20_s1.log || { echo "STOP: 가지치기 불일치 또는 실행 실패"; exit 1; }
for S in 20 50; do
  DL_SEED=1 DL_S=$S DL_X="2,3" DL_TL=600 DL_PRE=0 DL_SOLVERS=abbdr DL_ABBKW="$KW" DL_WORKERS=12 \
    julia -t 14,1 nonzero_sum/diag_l6.jl > $L/co_S${S}_s1.log 2>&1
done
