#!/usr/bin/env bash
# 2026-11-28 새벽: zero-sum 네트워크 차단을 B&P (gen_zs.py 가 ccgbp.jl v17 에서 만든 ccgbp_zs.jl) 로 (사용자 "제로섬도 ccg bnp로 풀어보고").
#   설정: S = 10, ε̂ = ε̃ = 0.3, β = 0.4, δ = 0.1, λᵁ 1e4 (oracle·모형), TOL 5e-3, TL 1800 s, DD 끔. 비교: Benders (menu) true_dro/factor_5/logs/value_eps0p3/S_10/<net>_menu[_s<seed>].log
#   사용: ./zs_queue.sh "42" "grid5x5 polska abilene nobel_us sioux_falls"
SPD="C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/310f4530-ff69-4912-9a26-69df1bdd76ba/scratchpad"; cd "$SPD" || exit 1
export GRB_LICENSE_FILE=C:/Users/user/gurobi.lic
P="C:/Users/user/robust_bilevel_network_games"; D="$P/docs/nz_ccg_bnp_scripts"
PYTHONIOENCODING=utf-8 python "$D/gen_zs.py" "$SPD/ccgbp_zs.jl" || exit 1
run() { local tag=$1; shift; env "$@" julia -t 12,1 --project="$P" "$SPD/ccgbp_zs.jl" > "$tag.log" 2>&1; echo "== $tag EXIT $? $(date +%H:%M)"; grep -E "^RESULT" "$tag.log" | cut -c1-330; }
echo "시작 $(date +%H:%M)"
for sd in $1; do for net in $2; do run zs_${net}_s${sd} NET=$net ZSEED=$sd S=10 TL=${TL:-1800}; done; done
echo "QUEUE DONE $(date +%H:%M)"
