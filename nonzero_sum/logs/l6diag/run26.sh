# run26 (2026-10-30): 작은 S 오버헤드 확인. L6 x̄={2,3}, seed 1~3, S=3·10, 600 s.
#   A = 지금 L6 구성 (range_lp, dd=false, farkas)  vs  B = A + infeas_pre + infeas_pre_retry + parent_basis + node_lp_cap 120
cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag/lpm
A="range_lp=true,dd=false,farkas=true"
B="$A,infeas_pre=true,infeas_pre_retry=true,parent_basis=true,node_lp_cap=120.0"
run() {  # 이름 S seed KW
  DL_SEED=$3 DL_S=$2 DL_X="2,3" DL_TL=600 DL_PRE=0 DL_SOLVERS=abbdr DL_ABBKW="$4" DL_WORKERS=12 \
    julia -t 14,1 nonzero_sum/diag_l6.jl > $L/$1.log 2>&1
}
for S in 3 10; do for s in 1 2 3; do
  run ovA_S${S}_s$s $S $s "$A"
  run ovB_S${S}_s$s $S $s "$B"
done; done
echo RUN26_DONE
