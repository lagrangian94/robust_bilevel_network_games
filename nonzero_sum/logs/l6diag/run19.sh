# run19 (2026-10-29): infeas_pre + parent_basis (부모 basis 를 자식에 싣고 pop 때 시작 basis 로) vs run18 (infeas_pre). 같은 조건
cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag/lpm
B="range_lp=true,dd=false,farkas=true"
run() {  # 이름 S workers KW
  rm -f $L/$1.nodelp
  NZ_NODELP_LOG=$L/$1.nodelp DL_SEED=1 DL_S=$2 DL_X="2,3" DL_TL=600 DL_PRE=0 DL_SOLVERS=abbdr DL_ABBKW="$4" DL_WORKERS=$3 \
    julia -t $(($3 + 2)),1 nonzero_sum/diag_l6.jl > $L/$1.log 2>&1
}
run pb_S20 20 12 "$B,infeas_pre=true,parent_basis=true"
run pb_S50 50 12 "$B,infeas_pre=true,parent_basis=true"
echo RUN19_DONE
