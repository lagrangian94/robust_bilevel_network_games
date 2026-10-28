cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag
PP_PRE=0 PP_BNBKW="dual_range=false" PP_NLOC=25 PP_NA=5 PP_NB=5 PP_NBMAX=5 PP_QB=750 PP_LAMBDA=2000 PP_S=3 PP_SEEDS="1,2,3,4,5,6,7,8,9,10" PP_TL=600 PP_WORKERS=12 \
  julia -t 14,1 nonzero_sum/pilot_pooled_store.jl > $L/bend_L6_S3_nodr.log 2>&1
PP_PRE=0 PP_BNBKW="dual_range=false" PP_NLOC=25 PP_NA=5 PP_NB=5 PP_NBMAX=5 PP_QB=750 PP_LAMBDA=2000 PP_S=10 PP_SEEDS="1,2,3" PP_TL=3600 PP_WORKERS=12 \
  julia -t 14,1 nonzero_sum/pilot_pooled_store.jl > $L/bend_L6_S10_nodr.log 2>&1
