cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag
# 2026-10-28: 노드 처리량 vs S — L6 seed 1 x̄={2,3} 고정, α-B&B (range_lp) 600 s, S = 3, 20, 50 (S=10 은 rlp_S10_s1.log)
for S in 3 20 50; do
DL_SEED=1 DL_S=$S DL_X="2,3" DL_TL=600 DL_PRE=0 DL_SOLVERS=abbdr DL_ABBKW="range_lp=true,dd=false" DL_WORKERS=12 \
  julia -t 14,1 nonzero_sum/diag_l6.jl > $L/rlp_S${S}_s1.log 2>&1
done
