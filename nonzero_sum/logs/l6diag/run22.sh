# run22 (2026-10-29): S50 infeas_pre + parent_basis + 재시도 마감 적용 (cap 없음, node_lp_cap=120). x̄={2,3}, seed 1, 600 s
cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag/lpm
B="range_lp=true,dd=false,farkas=true"
run() {  # 이름 S workers KW
  rm -f $L/$1.nodelp
  NZ_NODELP_LOG=$L/$1.nodelp DL_SEED=1 DL_S=$2 DL_X="2,3" DL_TL=600 DL_PRE=0 DL_SOLVERS=abbdr DL_ABBKW="$4" DL_WORKERS=$3 \
    julia -t $(($3 + 2)),1 nonzero_sum/diag_l6.jl > $L/$1.log 2>&1
}
run dl_S50 50 12 "$B,infeas_pre=true,parent_basis=true"
run cap120_S50 50 12 "$B,infeas_pre=true,parent_basis=true,node_lp_cap=120.0"
echo RUN22_DONE
