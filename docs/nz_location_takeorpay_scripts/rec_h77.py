# -*- coding: utf-8 -*-
# records the verdict of H77 (radius sweep, ledger 4.31): both copies of cf_predictions.txt, ledger status, hypothesis doc 9.34, memory, script copies.
# also repairs "\rho" in hypothesis doc 9.33 (an earlier record script wrote a carriage return + "ho").
import io, shutil, os
SP2 = "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad/"
DOC = "C:/Users/user/robust_bilevel_network_games/docs/"
MEM = "C:/Users/user/.claude/projects/C--Users-user-robust-bilevel-network-games/memory/nz_location_takeorpay.md"
V77 = r"""
 H77 판정 (2026-11-18 19:40. `agg_sweep.py` → `agg_sweep_out.txt`, 그림 `docs/figs/nz_location_takeorpay/radius_frontier.png`, 분해 `agg_mechx_out.txt`. 150 건, 실패 0, 용량비 0.4, 30 칸 평균, ρ = 1, Nominal 대비 %. `geo/zss3m` 과 `geo/zss3x` 의 Nominal 값 차이 최대 0.002%p):
  a) 성립. Coupled (0.2) q95 −4.08 ± 0.45, 평균 +2.79 ± 0.36. Coupled (0.3) q95 −4.35 ± 0.47, 평균 +3.98 ± 0.52.
  b) 성립. 같은 반지름끼리 Coupled − Data-DR 의 q95 가 −1.94 ± 0.58, −2.33 ± 0.60, −2.61 ± 0.57 (0.1, 0.2, 0.3). Coupled − Belief-only 는 −2.09 ± 0.47, −2.85 ± 0.48, −3.05 ± 0.53.
  c) 성립. Coupled (0.2) − Data-DR (0.3): 평균 −0.40 ± 0.60, q95 −2.34 ± 0.62. 평균 대가가 같은데 꼬리가 2.3 좋다.
  d) 거의 성립. Data-DR (0.3) 평균 +3.19 ± 0.69 (예측 +3.2 ~ +4.5 의 아래 끝에서 0.01 벗어남), q95 −1.74 ± 0.73. Belief-only (0.3) 평균 +0.41 ± 0.11, q95 −1.30 ± 0.42, 아는 조건 q95 −1.96 ± 0.52.
  e) 성립. 조용한 칸 (γ = 10, ζ = 0) 에서 Coupled (0.3) 의 q95 +1.43 ± 0.39. Coupled (0.2) 는 +0.43 ± 0.22, Coupled (0.1) 은 −0.77 ± 0.17.
  예측에 없던 것: 꼬리 이득은 세 모형 모두 반지름 0.2 근처에서 포화된다. 0.2 → 0.3 의 q95 변화가 Coupled −0.27 ± 0.19, Data-DR +0.01 ± 0.18, Belief-only −0.07 ± 0.19 인데 평균 대가는 +1.19, +0.73, +0.21 더 든다.
"""
S934 = r"""
### 9.34 반지름 스윕: 세 모형 × 반지름 0.1·0.2·0.3 (2026-11-18 저녁, 150 건, `agg_sweep.py`, H77)

설정은 `dwps3` 그대로 (가격 기준, 집합 = TV ∩ 상자 $r/\sqrt S$, Coupled 의 δ 는 반지름의 3분의 1), 지리 15 × 시드 10 = 150 건, 실패 0, 용량비 0.4. 평가는 30 칸 평균, 값은 Nominal 대비 % (Nominal 의 학습 비용 기준), 표준오차는 지리 단위. 그림은 `docs/figs/nz_location_takeorpay/radius_frontier.png`.

**수요 분포도 모르는 조건 ($\rho = 1$): 꼬리 ($q_{95}$) 와 평균 대가.**

| 반지름 | Belief-only 꼬리 | 평균 | Data-DR 꼬리 | 평균 | Coupled 꼬리 | 평균 |
|---|---|---|---|---|---|---|
| 0.1 | −0.51 ± 0.20 | +0.01 | −0.66 ± 0.55 | +1.21 | −2.61 ± 0.47 | +1.12 |
| 0.2 | −1.23 ± 0.31 | +0.21 | −1.74 ± 0.70 | +2.46 | −4.08 ± 0.45 | +2.79 |
| 0.3 | −1.30 ± 0.42 | +0.41 | −1.74 ± 0.73 | +3.19 | −4.35 ± 0.47 | +3.98 |

**수요 분포를 아는 조건: 꼬리.** Belief-only −0.89, −1.75, −1.96. Data-DR +0.25, +0.97, +1.37. Coupled −0.46, +0.19, +1.08.

**결정과 손잡이 (학습 belief 에서의 수준, Nominal 은 총용량 701·약정 594·여유 106).**

| 모형 | 총용량 0.1 / 0.2 / 0.3 | 여유 용량 | 수요만 틀릴 때 피해 (최악 10) | belief 만 | 함께 |
|---|---|---|---|---|---|
| Nominal | 701 | 106 | 5.62 | 5.07 | 11.05 |
| Belief-only | 694 / 685 / 681 | 104 / 98 / 96 | 5.64 / 5.29 / 5.12 | 3.33 / 2.93 / 3.03 | 9.53 / 9.06 / 9.07 |
| Data-DR | 673 / 653 / 646 | 77 / 61 / 53 | 3.90 / 2.87 / 2.51 | 5.10 / 4.90 / 4.68 | 10.46 / 9.47 / 9.00 |
| Coupled | 667 / 641 / 629 | 83 / 61 / 54 | 4.43 / 3.15 / 2.74 | 3.26 / 2.81 / 2.64 | 7.92 / 6.49 / 5.96 |

읽기 (꼬리로 판정, 평균은 대가).

- **같은 반지름에서 Coupled 의 꼬리가 가장 좋다.** Data-DR 과의 짝 차이가 −1.94 ± 0.58, −2.33 ± 0.60, −2.61 ± 0.57, Belief-only 와는 −2.09 ± 0.47, −2.85 ± 0.48, −3.05 ± 0.53 이다.
- **같은 평균 대가에서도 Coupled 가 Data-DR 보다 낫다.** Coupled (0.2) 와 Data-DR (0.3) 은 평균 대가가 같고 (차 −0.40 ± 0.60) 꼬리는 Coupled 가 2.34 ± 0.62 좋다. Data-DR 은 반지름을 키워도 Coupled (0.1) 의 꼬리 (−2.61) 에 닿지 못한다 (가장 좋을 때 −1.74).
- **꼬리 이득은 반지름 0.2 근처에서 포화된다.** 0.2 → 0.3 의 꼬리 변화가 Coupled −0.27 ± 0.19, Data-DR +0.01 ± 0.18, Belief-only −0.07 ± 0.19 인데 평균 대가는 +1.19, +0.73, +0.21 더 든다.
- **반지름이 크면 조용한 세계에서 손해다.** 진짜 분포가 추정 근처인 칸 (γ = 10, ζ = 0) 에서 Coupled 의 꼬리는 −0.77 ± 0.17, +0.43 ± 0.22, +1.43 ± 0.39 (0.1, 0.2, 0.3) 이다. 수요 분포를 아는 조건에서도 Coupled (0.3) 은 Nominal 보다 +1.08 나쁘고 Belief-only 가 가장 좋다.
- **손잡이는 반지름을 따라 같은 방향으로 더 간다.** Data-DR 은 여유 용량을 줄여 수요 쪽 피해만 줄이고 (5.62 → 2.51) belief 쪽은 거의 그대로다 (5.07 → 4.68). Belief-only 는 belief 쪽만 줄인다 (5.07 → 3.03). Coupled 는 둘 다 줄여 함께 틀릴 때의 피해가 11.05 → 5.96 이 된다. Coupled (0.3) 의 감소 5.09 는 Data-DR (0.3) 의 2.05 와 Belief-only (0.3) 의 1.98 을 더한 것보다 크다.
- 한계: Coupled 와 Belief-only 는 후보 안에서 고른 결정이고 Coupled 의 값은 점 탐색 하한이다. 반지름 0.5 (대장 4.33) 와 δ 스윕 (4.34) 은 실행 예약 상태다.
"""
MEMB = r"""
- **반지름 스윕 0.1·0.2·0.3 결과 (2026-11-18 저녁, 가설 문서 9.34, H77 성립, 150 건).** ρ = 1 의 꼬리 (q95)·평균: Coupled −2.61·+1.12, −4.08·+2.79, −4.35·+3.98. Data-DR −0.66·+1.21, −1.74·+2.46, −1.74·+3.19. Belief-only −0.51·+0.01, −1.23·+0.21, −1.30·+0.41. 같은 반지름에서 Coupled 가 Data-DR 보다 꼬리 1.9~2.6 좋음, 같은 평균 대가 (Coupled 0.2 대 Data-DR 0.3) 에서도 2.34 ± 0.62 좋음. 꼬리 이득은 0.2 근처에서 포화, 평균 대가만 늚. 조용한 칸에서 Coupled 0.3 은 +1.43 손해. 그림 `docs/figs/nz_location_takeorpay/radius_frontier.png`. 반지름 0.5 (대장 4.33, H80) 와 δ 스윕 (4.34, H81, 반지름 0.3·δ 0/0.1/0.3/0.6·ρ 다섯 값) 은 예약 실행 중 (2026-11-19 오후 끝 예정).
"""
for p in (SP2 + "cf_predictions.txt", DOC + "nz_location_takeorpay_scripts/cf_predictions.txt"):
    s = io.open(p, encoding="utf-8").read(); assert "H77 판정" not in s, p
    k = s.index("\nH78 ("); s = s[:k].rstrip("\n") + V77 + s[k:]
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
p = DOC + "nz_location_config_log.md"; s = io.open(p, encoding="utf-8").read()
a = "- **예측.** H77.\n- **상태.** 실행 중."; assert s.count(a) == 1
s = s.replace(a, "- **예측.** H77.\n- **상태 (2026-11-18 19:18).** 끝남, 150 건, 실패 0 (Belief-only 0.3 13:02, Coupled 0.2 14:17, Coupled 0.3 15:19, 평가 16:24, 분해 19:11). 결과는 가설 문서 9.34, H77 판정.")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
p = DOC + "nz_location_capacity_hypothesis_2026-11-14.md"; s = io.open(p, encoding="utf-8").read(); assert "### 9.34" not in s
n = s.count("\rho"); s = s.replace("\rho", "\\rho")
io.open(p, "w", encoding="utf-8", newline="\n").write(s.rstrip("\n") + "\n" + S934)
s = io.open(MEM, encoding="utf-8").read(); assert "반지름 스윕 0.1·0.2·0.3 결과" not in s
io.open(MEM, "w", encoding="utf-8", newline="\n").write(s.rstrip("\n") + "\n" + MEMB)
for f in ("agg_sweep.py", "agg_sweep_out.txt", "agg_mechx_out.txt", "fig_sweep.py", "rec_h77.py"):
    assert os.path.exists(SP2 + f), f; shutil.copy(SP2 + f, DOC + "nz_location_takeorpay_scripts/" + f)
print("H77 verdict, ledger 4.31 status, hypothesis doc 9.34 and memory recorded; repaired %d broken rho in the hypothesis doc" % n)
