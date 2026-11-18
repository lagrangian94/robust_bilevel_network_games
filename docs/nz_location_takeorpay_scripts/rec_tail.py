# -*- coding: utf-8 -*-
# report: the yardstick is the bad tail (the leader minimizes a worst-case CVaR), so Coupled is the better one when the demand distribution is unknown.  Section 6 row, conclusion row 5.
import io, shutil
D = "C:/Users/user/robust_bilevel_network_games/docs/"; p = "report_short_main.md"; s = io.open(p, encoding="utf-8").read()
a = "| 쓸 수 있음 (조건 명시) | Coupled 는 Data-DR 과 같은 평균 대가로 꼬리를 더 줄인다. Belief-only 보다는 평균을 1 가량 더 내고 꼬리를 2 가량 더 줄이므로 어느 쪽이 낫다고는 못 한다. Coupled 의 결정은 점 탐색으로 고른 것이다 (4.5, 5 절의 3) |"; assert s.count(a) == 1
s = s.replace(a, "| 쓸 수 있음 | **수요 분포도 모를 때는 Coupled 가 가장 낫다.** 리더의 목적은 최악 경우의 CVaR 을 낮추는 것이므로 판정 기준은 나쁜 쪽 꼬리이고 평균은 그 대가다. Coupled 의 꼬리 이득 ($-2.61$) 은 Belief-only ($-0.51$, 반지름 0.2 는 $-1.23$) 와 Data-DR ($-0.66$, 반지름 0.2 는 $-1.74$) 보다 크고, 평가 30 칸 모두에서 Nominal 보다 낫다. 평균 대가 ($+1.12$) 는 같은 반지름의 Data-DR 과 같다. 남는 조건은 Coupled 의 결정이 점 탐색으로 고른 것이라는 점이다 (4.5, 5 절의 3) |")
a = "Belief-only 보다는 평균을 1 가량 더 낸다 (4.5, 새 집합)"; assert s.count(a) == 1
s = s.replace(a, "판정 기준인 꼬리로는 Belief-only 보다도 낫고, 그 대가로 평균을 1 가량 더 낸다 (4.5, 새 집합)")
io.open(p, "w", encoding="utf-8", newline="").write(s); shutil.copyfile(p, D + "nz_location_takeorpay_scripts/report_short_main.md"); shutil.copyfile("rec_tail.py", D + "nz_location_takeorpay_scripts/rec_tail.py"); print("ok")
