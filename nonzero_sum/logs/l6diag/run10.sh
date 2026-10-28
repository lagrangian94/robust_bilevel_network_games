cd /c/Users/user/robust_bilevel_network_games
# 2026-10-28: L6 Benders (range_lp) 결과의 λ 무관 인증 (x ∈ X 32 개마다 witness, 제한 10·120·600 s)
CF_LOGDIR=l6diag CF_LOGS="bend_L6_S3_rlp.log,bend_L6_S10_rlp.log" CF_OUT=nonzero_sum/logs/l6diag/cert_L6_rlp.log CF_BNBKW="range_lp=true,dd=false" \
  CF_TOL=5e-3 CF_TLS="10,120,600" CF_WORKERS=12 julia -t 14,1 nonzero_sum/cert_lambda_free.jl > nonzero_sum/logs/l6diag/cert_L6_rlp.out 2>&1
