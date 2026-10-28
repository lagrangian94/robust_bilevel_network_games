cd /c/Users/user/robust_bilevel_network_games
DL_S=10 DL_PRE=0 DL_SOLVERS=abbdr DL_TL=600 DL_ABBKW="dual_range=false" julia -t 14,1 nonzero_sum/diag_l6.jl > nonzero_sum/logs/l6diag/S10_abb_nodr_s1_x23.log 2>&1
DL_S=10 DL_PRE=0 DL_SOLVERS=grb DL_TL=600 julia -t 2 nonzero_sum/diag_l6.jl > nonzero_sum/logs/l6diag/S10_grb_s1_x23.log 2>&1
