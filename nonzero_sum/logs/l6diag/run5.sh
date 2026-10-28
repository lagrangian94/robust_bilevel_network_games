# 고정 x̄: 범위 MIP 를 처음 200 s 만 (range_time), 그 뒤는 부모 범위
cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag
run() { DL_PRE=0 DL_SOLVERS=abbdr DL_TL=600 DL_S=$1 DL_SEED=$2 DL_X=$3 DL_ABBKW="$4" julia -t 14,1 nonzero_sum/diag_l6.jl > $L/$5 2>&1; }
run 3 3 "3"   "range_time=200.0"          S3_rt200_s3_x3.log
run 3 1 "2,3" "range_time=200.0"          S3_rt200_s1_x23.log
run 10 1 "2,3" "range_time=200.0"         S10_rt200_s1_x23.log
run 3 3 "3"   "range_time=200.0,dd=false" S3_rt200nodd_s3_x3.log
