cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag
# 2026-10-28 아침: Benders L6 S=3 (range_time=120, dd=false) 에서 못 끝난 시드의 마지막 x* 고정 — α-B&B vs Gurobi Ω, 600 s
for sx in "5:1,2" "6:1,3,4" "7:4,5" "10:2,5"; do
  s=${sx%%:*}; x=${sx#*:}
  DL_SEED=$s DL_S=3 DL_X="$x" DL_TL=600 DL_GAP=1e-3 DL_PRE=0 DL_SOLVERS=info,abbdr,grb DL_ABBKW="range_time=120.0,dd=false" \
    julia -t 14,1 nonzero_sum/diag_l6.jl > $L/fx_S3_s${s}.log 2>&1
done
