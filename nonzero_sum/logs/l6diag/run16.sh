# run16 (2026-10-29): 노드 LP Presolve (lp_presolve = :warm / :cold) 처리량. S20 은 가지치기 대조 (NZ_PRUNE_CHECK) 겸
cd /c/Users/user/robust_bilevel_network_games
L=nonzero_sum/logs/l6diag
for P in warm cold; do
  KW="range_lp=true,dd=false,farkas=true,lp_presolve=:$P"
  NZ_PRUNE_CHECK=1 DL_SEED=1 DL_S=20 DL_X="2,3" DL_TL=300 DL_PRE=0 DL_SOLVERS=abbdr DL_ABBKW="$KW" DL_WORKERS=12 \
    julia -t 14,1 nonzero_sum/diag_l6.jl > $L/pre${P}_check_S20_s1.log 2>&1
  grep -qE '^\s+prune_disagree = 0$' $L/pre${P}_check_S20_s1.log || { echo "STOP: $P 가지치기 불일치 또는 실행 실패"; exit 1; }
done
for P in warm cold; do
  KW="range_lp=true,dd=false,farkas=true,lp_presolve=:$P"
  for S in 20 50; do
    DL_SEED=1 DL_S=$S DL_X="2,3" DL_TL=600 DL_PRE=0 DL_SOLVERS=abbdr DL_ABBKW="$KW" DL_WORKERS=12 \
      julia -t 14,1 nonzero_sum/diag_l6.jl > $L/pre${P}_S${S}_s1.log 2>&1
  done
done
