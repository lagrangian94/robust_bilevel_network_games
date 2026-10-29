# run20 (2026-10-29): infeas_pre + parent_basis, 분리 행 공유 없음 (share_rows=false) vs run19 (공유 있음)
cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag/lpm
B="range_lp=true,dd=false,farkas=true"
run() {  # 이름 S workers KW
  rm -f $L/$1.nodelp
  NZ_NODELP_LOG=$L/$1.nodelp DL_SEED=1 DL_S=$2 DL_X="2,3" DL_TL=600 DL_PRE=0 DL_SOLVERS=abbdr DL_ABBKW="$4" DL_WORKERS=$3 \
    julia -t $(($3 + 2)),1 nonzero_sum/diag_l6.jl > $L/$1.log 2>&1
}
run pbn_S50 50 12 "$B,infeas_pre=true,parent_basis=true"
run pbn_S20 20 12 "$B,infeas_pre=true,parent_basis=true"
echo RUN20_DONE
