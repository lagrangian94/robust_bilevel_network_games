# -*- coding: utf-8 -*-
# record: H78 verdict (why the belief add-on of Data-DR's decisions is larger), hypothesis document 9.32, report 4.6, memory
import io, shutil
D = "C:/Users/user/robust_bilevel_network_games/docs/"; SP = "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad/"
V = """ H78 판정 (2026-11-18 오후. `q4_look.py`, `q4_align.py`, `q4_carry.py`, 출력 `q4_align_out.txt`, `q4_carry_out.txt`. Data-DR ≠ Nominal 50 건, 값은 Nominal 의 학습 비용 대비 %):
  a) 성립. belief 만 틀릴 때 피해의 차이 (Data-DR 결정 − Nominal 결정) 는 섭동 최악 10 으로 +0.10 ± 1.13, adv0.1 로 +0.09 ± 1.07. 다만 표준오차가 1 이라 +1 안팎의 차이는 배제하지 못한다.
  b) 성립. Nominal 결정에서 수요 쪽 최악 10 섭동과 belief 쪽 최악 10 섭동은 평균 1.86 개만 겹친다. 방향이 달라 놓치는 피해는 Nominal 결정 3.25, Data-DR 결정 0.80, 차이 −2.45 ± 0.75.
  c) 성립. 수요 쪽 피해 감소 5.17 가운데 함께 틀릴 때 남는 것은 1.77, 사라지는 것은 3.40. 그중 방향 2.45 (72%), 교차항 0.86 ± 0.46 (25%), belief 쪽 0.10 (3%).
  d) 성립. VC − VD 에서 adv0.1 을 뺀 값이 Nominal 결정 −0.65, Data-DR 결정 +1.76, 차이 +2.41 ± 0.64.
  읽기: +2.50 은 Data-DR 결정이 belief 오류에 더 약해서 생긴 값이 아니다. 묶인 상대는 한 방향만 고를 수 있어 Nominal 결정에서는 두 피해를 다 내지 못한다. 수요 쪽 피해를 지우면 상대는 belief 쪽 최악 방향을 그대로 고른다. 한쪽 헤징이 결합 집합에서 남는 몫은 Data-DR 46% (−4.65 → −2.15), Belief-only 92% (−4.46 → −4.11).
"""
for p in (D + "nz_location_takeorpay_scripts/cf_predictions.txt", SP + "cf_predictions.txt"):
    s = io.open(p, encoding="utf-8").read(); assert "H78 판정" not in s; io.open(p, "w", encoding="utf-8", newline="").write(s.rstrip("\n") + "\n" + V)
H = """
### 9.32 Data-DR 결정의 belief 추가 피해가 큰 이유 (2026-11-18 오후, 150 건, `q4_align.py`·`q4_carry.py`)

9.31 (가) 가 남긴 물음이다. Data-DR (0.1) 이 고른 결정은 결합 집합에서 belief 추가 피해 (VC − VD) 가 Nominal 결정보다 +2.50 ± 0.95 크다. 용량 탓은 아니었다. 예측은 H78. 새 실행 없이 기존 자료 (`geo/dw/cp_dwps3`, `geo/dw/dw_dwps3`, `geo/mechc`) 만 썼다. 값은 Nominal 의 학습 비용 대비 %, ± 표준오차 (지리 군집).

**답: Data-DR 결정이 belief 오류에 더 약한 것이 아니다. 가장 나쁜 방향이 서로 달라서다.**

결합 집합에서 상대는 수요 분포 $a$ 와 belief $d$ 를 거의 같은 쪽으로만 움직일 수 있다 ($TV(a, d) \le \delta$). VC − VD 는 belief 피해가 아니라, 결합 최악 피해에서 수요만 틀릴 때의 최악 피해를 뺀 나머지다.

Data-DR ≠ Nominal 인 50 건, 섭동 90 가지 (한 쌍 옮기기).

| | Nominal 결정 | Data-DR 결정 | 차이 |
|---|---|---|---|
| 수요만 틀림, 가장 나쁜 10 | 7.96 | 2.79 | −5.17 ± 0.47 |
| belief 만 틀림, 가장 나쁜 10 | 5.37 | 5.47 | +0.10 ± 1.13 |
| 둘 다 틀림, 가장 나쁜 10 | 12.44 | 10.67 | −1.77 ± 1.13 |
| 그 10 가지에서 수요 쪽 | 6.61 | 2.50 | −4.11 ± 0.67 |
| 그 10 가지에서 belief 쪽 | 3.47 | 4.96 | +1.48 ± 1.12 |
| 그 10 가지에서 교차항 | 2.36 | 3.21 | +0.86 ± 0.46 |
| 방향이 달라 상대가 놓치는 피해 | 3.25 | 0.80 | −2.45 ± 0.75 |
| 수요 쪽 최악 10 과 belief 쪽 최악 10 이 겹치는 수 | 1.86 | 2.28 | +0.42 ± 0.38 |
| 결합 집합: VD − V0 | 7.23 | 2.58 | −4.65 ± 0.36 |
| 결합 집합: VC − VD | 4.26 | 6.76 | +2.50 ± 0.95 |
| belief 만 틀릴 때 최악 (adv0.1) | 4.91 | 5.00 | +0.09 ± 1.07 |
| 결합 집합: VC − V0 | 11.49 | 9.34 | −2.15 ± 0.92 |

"방향이 달라 놓치는 피해" 는 (수요만 틀릴 때 최악 10 + belief 만 틀릴 때 최악 10) 에서 (둘 다 틀릴 때 최악 10 에서의 수요 쪽 + belief 쪽) 을 뺀 것이다.

- belief 만 틀릴 때의 피해는 두 결정이 같다 (+0.10 ± 1.13, adv0.1 로 +0.09 ± 1.07). 표준오차가 1 이라 +1 안팎의 차이는 배제하지 못한다.
- Nominal 결정에서 수요 쪽으로 가장 나쁜 섭동과 belief 쪽으로 가장 나쁜 섭동은 10 개 중 1.9 개만 겹친다. 묶인 상대는 한 방향을 골라야 해서 3.25 를 놓친다.
- Data-DR 결정은 수요 쪽 피해가 어느 방향으로도 작다. 상대는 belief 쪽으로 가장 나쁜 방향을 그대로 고르고 0.80 만 놓친다.
- 수요 쪽 피해를 5.17 줄였는데 둘 다 틀릴 때는 1.77 만 준다. 사라진 3.40 = 방향 2.45 + 교차항 0.86 + belief 쪽 0.10.

**한쪽 헤징이 결합 집합에서 남는 몫** (각 리더가 Nominal 과 다른 결정을 고른 건).

| 리더 | 건수 | 수요만 틀릴 때 최악 | belief 만 틀릴 때 최악 | 결합 집합 최악 | 헤징한 쪽 대비 남는 몫 |
|---|---|---|---|---|---|
| Data-DR (0.1) | 50 | −4.65 ± 0.36 | +0.09 ± 1.07 | −2.15 ± 0.92 | 46% (잃는 것 +2.50 ± 0.95) |
| Belief-only (0.1) | 70 | +0.08 ± 0.44 | −4.46 ± 0.49 | −4.11 ± 0.53 | 92% (잃는 것 +0.35 ± 0.31) |
| Coupled | 82 | −2.03 ± 0.42 | −3.42 ± 0.50 | −6.47 ± 0.57 | 119% (두 쪽 합 −5.44 보다 −1.02 ± 0.62 더) |

- Belief-only 도 방향 때문에 1.08 ± 0.67 을 잃지만 교차항이 0.65 ± 0.25 줄어 메운다. Data-DR 은 방향으로 2.45 를 잃고 교차항도 0.86 늘어난다.
- 두 리더의 차이가 분명한 것은 교차항의 부호다. 방향 쪽 차이 (2.45 대 1.08) 는 표준오차 0.7 안팎이라 뚜렷하지 않다.
- 한계: 섭동 분해는 수요와 belief 를 똑같이 옮긴 경우 ($a = d$) 로 결합 집합을 흉내 낸 것이다. VC 는 점 탐색 하한이다. 리더마다 건의 집합이 다르다 (50, 70, 82 건).
"""
p = D + "nz_location_capacity_hypothesis_2026-11-14.md"; s = io.open(p, encoding="utf-8").read(); assert "### 9.32" not in s
io.open(p, "w", encoding="utf-8", newline="").write(s.rstrip("\n") + "\n" + H)
R = """**한쪽만 헤징한 보호는 둘 다 틀릴 때 얼마나 남나.** 각 리더가 Nominal 과 다른 결정을 고른 건에서, 헤징한 쪽의 최악 피해를 줄인 폭과 결합 집합 (수요 분포와 belief 가 함께 움직이는 집합, 2.3) 의 최악 피해를 줄인 폭이다.

| 리더 | 건수 | 헤징한 쪽 피해 감소 | 결합 집합 최악 피해 감소 | 남는 몫 |
|---|---|---|---|---|
| Data-DR | 50 | $-4.65 \pm 0.36$ | $-2.15 \pm 0.92$ | 46% |
| Belief-only | 70 | $-4.46 \pm 0.49$ | $-4.11 \pm 0.53$ | 92% |
| Coupled | 82 | $-5.44$ (수요 쪽 $-2.03$, belief 쪽 $-3.42$) | $-6.47 \pm 0.57$ | 119% |

- **Data-DR 의 보호는 절반이 사라진다** ($+2.50 \pm 0.95$). Data-DR 의 결정이 belief 오류에 더 약해서가 아니다. belief 만 틀릴 때의 피해는 Nominal 의 결정과 같다 ($+0.09 \pm 1.07$).
- 이유는 가장 나쁜 방향이 서로 다르다는 데 있다. Nominal 의 결정에서 수요 쪽으로 가장 나쁜 섭동 10 가지와 belief 쪽으로 가장 나쁜 10 가지는 평균 1.9 개만 겹친다. 둘을 함께 움직여야 하는 상대는 한 방향을 골라야 해서 3.25 를 놓친다. Data-DR 이 수요 쪽 피해를 줄여 놓으면 상대는 belief 쪽으로 가장 나쁜 방향을 그대로 고르고 0.80 만 놓친다 (차이 $-2.45 \pm 0.75$).
- 교차항도 반대로 움직인다. Data-DR 의 결정에서는 늘고 ($+0.86 \pm 0.46$), Belief-only 의 결정에서는 준다 ($-0.65 \pm 0.25$). 그래서 Belief-only 의 보호는 거의 그대로 남는다.

"""
p = SP + "report_short_main.md"; s = io.open(p, encoding="utf-8").read(); a = "**belief 오류의 피해는 꼬리에만 남는다.**"; assert s.count(a) == 1 and "얼마나 남나" not in s
io.open(p, "w", encoding="utf-8", newline="").write(s.replace(a, R + a))
M = "\n- **Data-DR 결정의 belief 추가 피해 +2.50 의 원인 (2026-11-18 오후, 가설 문서 9.32, H78 성립).** Data-DR 결정이 belief 에 더 약한 것이 아님 (belief 만 틀릴 때 피해 차 +0.09 ± 1.07). Nominal 결정에서는 수요 쪽 최악 방향과 belief 쪽 최악 방향이 달라 (최악 10 중 1.9 개 겹침) 묶인 상대가 3.25 를 놓치는데, 수요 쪽 피해를 지우면 0.80 만 놓침. 한쪽 헤징이 결합 집합에서 남는 몫: Data-DR 46%, Belief-only 92%, Coupled 는 두 쪽 합보다 −1.02 ± 0.62 더. 교차항은 Data-DR +0.86, Belief-only −0.65. 스크립트 `q4_align.py`, `q4_carry.py`.\n"
p = "C:/Users/user/.claude/projects/C--Users-user-robust-bilevel-network-games/memory/nz_location_takeorpay.md"; s = io.open(p, encoding="utf-8").read(); io.open(p, "w", encoding="utf-8", newline="").write(s.rstrip("\n") + "\n" + M)
for f in ("q4_look.py", "q4_align.py", "q4_carry.py", "rec_932.py"): shutil.copy(SP + f, D + "nz_location_takeorpay_scripts/" + f)
print("recorded")
