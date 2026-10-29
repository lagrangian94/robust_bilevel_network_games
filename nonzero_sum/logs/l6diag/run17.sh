# run17 (2026-10-29): L6 노드 LP 방법 (barrier) 과 warm start (dive vs pop) 진단. 고정 x̄={2,3}, seed 1, 600 s, NZ_NODELP_LOG
cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag/lpm
B="range_lp=true,dd=false,farkas=true"
run() {  # 이름 S workers KW
  rm -f $L/$1.nodelp
  NZ_NODELP_LOG=$L/$1.nodelp DL_SEED=1 DL_S=$2 DL_X="2,3" DL_TL=600 DL_PRE=0 DL_SOLVERS=abbdr DL_ABBKW="$4" DL_WORKERS=$3 \
    julia -t $(($3 + 2)),1 nonzero_sum/diag_l6.jl > $L/$1.log 2>&1
}
run base_S50 50 12 "$B"
run bar_S50 50 12 "$B,lp_method=2,lp_presolve=-1,lp_crossover=0"
run base_S20 20 12 "$B"
run bar_S20 20 12 "$B,lp_method=2,lp_presolve=-1,lp_crossover=0"
run base_S50_w6 50 6 "$B"
echo RUN17_DONE
