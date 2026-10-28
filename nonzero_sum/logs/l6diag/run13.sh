cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag
# 2026-10-29: Farkas 검증 (farkas=true). (1) S20 대조 (NZ_FARKAS_CHECK=1, 불일치 있으면 멈춤) → (2) 고정 x̄ 노드 처리량 S20·S50 → (3) Benders S20·S50 seed 1~3
KW="range_lp=true,dd=false,farkas=true"
NZ_FARKAS_CHECK=1 DL_SEED=1 DL_S=20 DL_X="2,3" DL_TL=300 DL_PRE=0 DL_SOLVERS=abbdr DL_ABBKW="$KW" DL_WORKERS=12 \
  julia -t 14,1 nonzero_sum/diag_l6.jl > $L/fk_check_S20_s1.log 2>&1
grep -qE '^\s+farkas_disagree = 0$' $L/fk_check_S20_s1.log || { echo "STOP: farkas 불일치 또는 실행 실패"; exit 1; }
for S in 20 50; do
DL_SEED=1 DL_S=$S DL_X="2,3" DL_TL=600 DL_PRE=0 DL_SOLVERS=abbdr DL_ABBKW="$KW" DL_WORKERS=12 \
  julia -t 14,1 nonzero_sum/diag_l6.jl > $L/fk_S${S}_s1.log 2>&1
done
for S in 20 50; do
PP_PRE=0 PP_BNBKW="$KW" PP_NLOC=25 PP_NA=5 PP_NB=5 PP_NBMAX=5 PP_QB=750 PP_LAMBDA=2000 PP_S=$S PP_SEEDS="1,2,3" PP_TL=3600 PP_WORKERS=12 \
  julia -t 14,1 nonzero_sum/pilot_pooled_store.jl > $L/bend_L6_S${S}_fk.log 2>&1
done
