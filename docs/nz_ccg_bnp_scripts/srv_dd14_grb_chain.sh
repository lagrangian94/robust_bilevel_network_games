#!/usr/bin/env bash
# 2026-11-28 밤 (사용자 "수제 알파비앤비 말고 구로비 솔바로 푼 거랑도 비교", md 11.34): 서버 cykim.
#   1,800 s DD 끔·켬 큐 (dd14L_A·B.out) 가 끝나기를 기다린 뒤, 돌고 있는 스크립트를 덮어쓰지 않으려고 그때 새 orcl_replay (SOLVER=grb) 와
#   큐 스크립트 (구성 G) 를 제자리에 놓는다. 그다음 Gurobi 구성 G 를 두 줄로 동시에 (8 스레드씩, A·B 와 같은 부하 모양).
#   줄 1 = 지리 1·15 (1,800 s) → 10·11 (600 s), 줄 2 = 지리 8·6 (1,800 s) → 4 (600 s).
#   올려 둘 것: srv_src/orcl_replay_local.jl (= orcl_replay.jl), srv_src/srv_dd14_queue_next.sh (= srv_dd14_queue.sh)
#   사용: cd ~/rbng_runs && setsid nohup bash srv_src/srv_dd14_grb_chain.sh > dd14G_chain.out 2>&1 < /dev/null &
set -u
R=$HOME/rbng_runs; cd "$R" || exit 1
until grep -q "QUEUE DONE" dd14L_A.out && grep -q "QUEUE DONE" dd14L_B.out; do sleep 60; done
echo "A·B 끝 $(date +%H:%M)"
sed -e 's|^root = "C:/Users/user/robust_bilevel_network_games"|root = "/home/swkim/rbng_runs/wt_dd"|' \
    -e 's|include(joinpath(root, "nonzero_sum", "loc_tkp.jl"))|include("/home/swkim/rbng_runs/srv_src/loc_tkp_srv_wt.jl")|' \
    srv_src/orcl_replay_local.jl > orcl_replay_wt.jl
grep -q '^root = "/home/swkim/rbng_runs/wt_dd"' orcl_replay_wt.jl && grep -q 'srv_src/loc_tkp_srv_wt.jl' orcl_replay_wt.jl || { echo "orcl_replay_wt.jl 생성 실패"; exit 1; }
cp srv_src/srv_dd14_queue_next.sh srv_dd14_queue.sh
( GEOS="1 15" TLIM=1800 bash srv_dd14_queue.sh G; GEOS="10 11" TLIM=600 bash srv_dd14_queue.sh G ) > dd14G_1.out 2>&1 &
( GEOS="8 6" TLIM=1800 bash srv_dd14_queue.sh G; GEOS="4" TLIM=600 bash srv_dd14_queue.sh G ) > dd14G_2.out 2>&1 &
wait
echo "CHAIN DONE $(date +%H:%M)"
