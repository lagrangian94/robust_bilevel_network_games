# 고정 x̄ oracle: range_mip=false (MIP 없는 강화만) 를 nodr·기본과 비교. bend_L6_S3_nodr.log 가 끝난 뒤 시작.
cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag
until [ "$(grep -c '^RESULT' $L/bend_L6_S3_nodr.log)" -ge 10 ]; do sleep 30; done
until ! powershell -NoProfile -Command "if (Get-Process julia -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 20; done
run() { DL_PRE=0 DL_SOLVERS=abbdr DL_TL=600 DL_S=$1 DL_SEED=$2 DL_X=$3 DL_ABBKW="$4" julia -t 14,1 nonzero_sum/diag_l6.jl > $L/$5 2>&1; }
run 3 1 "2,3" "range_mip=false,dd=false" S3_rmf_s1_x23.log
run 3 3 "3"   "range_mip=false,dd=false" S3_rmf_s3_x3.log
run 3 3 "3"   "dual_range=false"         S3_nodr_s3_x3.log
run 3 3 "3"   ""                         S3_default_s3_x3.log
run 10 1 "2,3" "range_mip=false,dd=false" S10_rmf_s1_x23.log
