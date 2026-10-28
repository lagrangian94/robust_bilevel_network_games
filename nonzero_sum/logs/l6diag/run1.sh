cd /c/Users/user/robust_bilevel_network_games
for c in "default|" "nodr|dual_range=false" "nodd|dd=false"; do
  n=${c%%|*}; kw=${c#*|}
  DL_PRE=0 DL_SOLVERS=abbdr DL_TL=600 DL_ABBKW="$kw" julia -t 14,1 nonzero_sum/diag_l6.jl > nonzero_sum/logs/l6diag/abb_${n}_s1_x23.log 2>&1
done
DL_PRE=0 DL_SOLVERS=grb DL_TL=600 julia -t 2 nonzero_sum/diag_l6.jl > nonzero_sum/logs/l6diag/grb_s1_x23.log 2>&1
