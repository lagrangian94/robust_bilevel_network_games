cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag
# 2026-10-28: L6 Benders (range_lp) S=20 → S=50, 시드 1~3, 1 h (S=10 시드 1~3 과 같은 구성). 인증 (run10) 은 S3 seed 5 에서 중단 — 다시 돌리면 끝난 7 개는 건너뜀
KW="range_lp=true,dd=false"
for S in 20 50; do
PP_PRE=0 PP_BNBKW="$KW" PP_NLOC=25 PP_NA=5 PP_NB=5 PP_NBMAX=5 PP_QB=750 PP_LAMBDA=2000 PP_S=$S PP_SEEDS="1,2,3" PP_TL=3600 PP_WORKERS=12 \
  julia -t 14,1 nonzero_sum/pilot_pooled_store.jl > $L/bend_L6_S${S}_rlp.log 2>&1
done
