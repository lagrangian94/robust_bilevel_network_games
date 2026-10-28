# run11 (Benders S20·S50, bash PID 20584) 이 끝나면 run12 (노드 처리량) 시작
while tasklist //FI "PID eq 20584" | grep -q bash.exe; do sleep 60; done
bash /c/Users/user/robust_bilevel_network_games/nonzero_sum/logs/l6diag/run12.sh
