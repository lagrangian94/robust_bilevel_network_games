cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag
# 2026-10-28: 꼭짓점 LP 범위 (range_lp) 검증 → Benders L6 S=3 × 10 (600 s) → S=10 × 3 (1 h)
julia -t 2 nonzero_sum/test_corner_ranges.jl > $L/test_corner_ranges.log 2>&1
KW="range_lp=true,dd=false"
PP_PRE=0 PP_BNBKW="$KW" PP_NLOC=25 PP_NA=5 PP_NB=5 PP_NBMAX=5 PP_QB=750 PP_LAMBDA=2000 PP_S=3 PP_SEEDS="1,2,3,4,5,6,7,8,9,10" PP_TL=600 PP_WORKERS=12 \
  julia -t 14,1 nonzero_sum/pilot_pooled_store.jl > $L/bend_L6_S3_rlp.log 2>&1
PP_PRE=0 PP_BNBKW="$KW" PP_NLOC=25 PP_NA=5 PP_NB=5 PP_NBMAX=5 PP_QB=750 PP_LAMBDA=2000 PP_S=10 PP_SEEDS="1,2,3" PP_TL=3600 PP_WORKERS=12 \
  julia -t 14,1 nonzero_sum/pilot_pooled_store.jl > $L/bend_L6_S10_rlp.log 2>&1
