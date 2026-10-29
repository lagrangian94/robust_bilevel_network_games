# run18 (2026-10-29): infeas_pre (recourse 블록을 뺀 부분 LP 로 실행 불가능 노드 사전 판정) vs run17 기준 (base_S20, base_S50). 같은 조건
cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag/lpm
B="range_lp=true,dd=false,farkas=true"
run() {  # 이름 S workers KW
  rm -f $L/$1.nodelp
  NZ_NODELP_LOG=$L/$1.nodelp DL_SEED=1 DL_S=$2 DL_X="2,3" DL_TL=600 DL_PRE=0 DL_SOLVERS=abbdr DL_ABBKW="$4" DL_WORKERS=$3 \
    julia -t $(($3 + 2)),1 nonzero_sum/diag_l6.jl > $L/$1.log 2>&1
}
run pre_S20 20 12 "$B,infeas_pre=true"
run pre_S50 50 12 "$B,infeas_pre=true"
echo RUN18_DONE
