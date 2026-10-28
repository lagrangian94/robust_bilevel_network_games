# range_time + MIP 마감 (마감에 걸린 MIP 는 bound 만 사용), dd 끔
cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag
run() { DL_PRE=0 DL_SOLVERS=abbdr DL_TL=600 DL_S=$1 DL_SEED=$2 DL_X=$3 DL_ABBKW="$4" julia -t 14,1 nonzero_sum/diag_l6.jl > $L/$5 2>&1; }
run 10 1 "2,3" "range_time=120.0,dd=false" S10_rtd120_s1_x23.log
run 3 1 "2,3" "range_time=60.0,dd=false"   S3_rtd60_s1_x23.log
run 3 3 "3"   "range_time=120.0,dd=false"  S3_rtd120_s3_x3.log
