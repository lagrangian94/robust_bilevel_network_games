# -*- coding: utf-8 -*-
# report: new section 4.6 (hedging strategies of the four leaders), conclusion row 6; hypothesis document 9.28; memory
import io, shutil
D = "C:/Users/user/robust_bilevel_network_games/docs/"; SP = "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/7fb80576-e557-4e86-9816-eed8962afe26/scratchpad/"
p = SP + "report_short_main.md"; s = io.open(p, encoding="utf-8").read()
a = "**Coupled 의 결정.** 두 손잡이 (4.4) 를 다 쓴다."; b = "## 5. 한계와 남은 일"; assert s.count(a) == 1 and s.count(b) == 1
NEW = r"""### 4.6 네 리더의 헤징 전략: Coupled 는 두 손잡이를 나눠 쓴다

리더가 모르는 것은 수요의 분포와 소매상의 belief 둘이다. 네 리더는 이 가운데 무엇을 모른다고 두느냐만 다르다 (2.3). 4.4 의 두 손잡이로 각 리더의 결정을 읽는다. 숫자는 4.5 의 결정이다 (새 집합, 반지름 0.1, $c/f = 0.4$, 150 건 평균).

**여유 용량은 두 가지 일을 한다.** 이것이 네 전략을 가른다.

- 수요가 높을 때 추가 주문을 받는다. 수요 분포가 저수요 쪽이면 이 값어치가 줄어든다.
- 소매상이 덜 약정했을 때 모자란 만큼을 정가로 판다. belief 가 틀렸을 때 잃은 약정 수입을 되찾는 길이다.

Data-DR 은 앞의 일만 보고 여유 용량을 줄인다. 그러면 뒤의 일도 함께 줄어든다. Coupled 는 두 일을 함께 보므로 여유 용량을 덜 줄이고, 모자라는 보호는 거점으로 채운다.

**전략.**

| 리더 | 모른다고 두는 것 | 거점: belief 가 틀릴 때 빠지는 약정 | 여유 용량 | Nominal 과 다른 결정 | 그중 총용량이 같은 건 |
|---|---|---|---|---|---|
| Nominal | 없음 | 26.5 | 106 | | |
| Belief-only | 소매상의 belief | 23.6 | 104 | 47% | 69% |
| Data-DR | 수요 분포 | 32.7 | 77 | 33% | 26% |
| Coupled | 둘 다 | 22.0 | 83 | 55% | 39% |

"빠지는 약정" 은 리더에게 가장 나쁜 belief 10 가지에서 약정이 줄어드는 양이다 (4.4 의 섭동 90 가지 중).

- **Belief-only 는 거점만 옮긴다.** 용량은 그대로 두고 약정이 덜 흔들리는 거점 조합으로 간다.
- **Data-DR 은 여유 용량만 줄인다.** 거점은 belief 를 보지 않고 고르므로 약정이 오히려 더 흔들리는 조합으로 간다 (26.5 → 32.7).
- **Coupled 는 둘을 나눠 쓴다.** 약정이 가장 덜 흔들리는 거점 조합을 고르고 (22.0), 여유 용량은 Data-DR 보다 덜 줄인다. 결정이 Nominal 과 다른 건에서 Data-DR 은 여유 용량을 88 줄이고 Coupled 는 43 줄인다. 다른 결정의 87% 에서 어떤 거점은 키우거나 새로 열고 어떤 거점은 줄이거나 닫는다.

![네 리더의 헤징 전략](figs/nz_location_takeorpay/four_strategies.png)

*그림 3. (가) 두 손잡이를 쓴 정도. (나) 그 결과로 두 오류에 약한 정도.*

**결과.** 같은 섭동을 수요 분포에, belief 에, 둘 다에 넣었을 때의 피해다 (가장 나쁜 10 가지 평균, Nominal 의 학습 비용 대비 %). 괄호는 Nominal 과의 차이다.

| 리더 | 수요 분포만 틀림 | belief 만 틀림 | 둘 다 틀림 |
|---|---|---|---|
| Nominal | 5.6 | 5.1 | 11.1 |
| Belief-only | 5.6 ($+0.0$) | 3.3 ($-1.7$) | 9.5 ($-1.5$) |
| Data-DR | 3.9 ($-1.7$) | 5.1 ($+0.0$) | 10.5 ($-0.6$) |
| Coupled | 4.4 ($-1.2$) | 3.3 ($-1.8$) | 7.9 ($-3.1$) |

- 한쪽만 헤징하면 그쪽 피해만 1.7 줄고 다른 쪽은 그대로다.
- Coupled 는 belief 쪽 보호를 Belief-only 만큼 얻고 ($-1.8$), 수요 쪽 보호는 Data-DR 의 70% 를 얻는다 ($-1.2$ 대 $-1.7$).
- 두 오류가 함께 오면 피해가 더해진다 (Nominal 11.1 ≈ 5.6 + 5.1). 이때 Coupled 가 줄이는 피해 ($-3.1$) 는 두 헤징을 따로 한 것 ($-1.5$, $-0.6$) 을 더한 것보다 크다.

**여유 용량을 다 줄이지 않는 이유.** 수요 분포는 $\hat q$ 로 두고 belief 만 틀리게 해서 약정이 줄어든 경우에, 리더가 잃은 약정 수입을 추가 주문으로 얼마나 되찾는지 쟀다.

| 리더 | 잃은 약정 수입 | 되찾는 몫: 평균 | 수요가 약정에 못 미치는 시나리오 | 그 밖 시나리오 | CVaR 에 남는 피해 |
|---|---|---|---|---|---|
| Nominal | 9.5 | 92% | 24% | 113% | $+2.49$ |
| Belief-only | 9.1 | 99% | 32% | 119% | $+1.52$ |
| Data-DR | 10.5 | 88% | 27% | 112% | $+3.09$ |
| Coupled | 8.5 | 98% | 38% | 119% | $+1.46$ |

- 평균으로는 잃은 약정 수입의 대부분을 추가 주문이 되찾는다. 추가 주문이 나오는 시나리오에서는 정가가 예약료보다 높아 리더가 오히려 번다.
- 수요가 약정에도 못 미치는 시나리오 (10 개 중 2~3 개) 에서는 추가 주문이 없어 되찾지 못한다. CVaR 은 그 시나리오를 본다. belief 헤징이 평균은 비기고 꼬리에서만 이기는 이유다.
- Data-DR 의 결정은 belief 오류에 가장 약하다 ($+3.09$). 되찾는 길인 여유 용량을 줄였고 약정이 더 흔들리는 거점으로 갔기 때문이다. Coupled 의 결정이 가장 덜 약하다 ($+1.46$).

**평가에서의 성과 (4.5, 수요 분포도 모르는 조건, $\rho = 1$).** Belief-only 평균 $+0.01$, $q_{95}$ $-0.51$. Data-DR $+1.21$, $-0.66$. Coupled $+1.12$, $-2.61$. Coupled 는 Data-DR 과 같은 평균 대가를 내고 꼬리를 네 배 줄인다. 아래 표는 그 이유를 비용의 수준으로 보인다. 같은 섭동 90 가지에서 점포비까지 넣은 비용을 Nominal 과 견주었다 (결정 − Nominal, %).

| 결정 | 학습 분포 | 가장 나쁜 10 가지: 수요 분포만 움직임 | belief 만 | 둘 다 | 90 가지 평균 (둘 다) |
|---|---|---|---|---|---|
| Belief-only (0.1) | $+0.94$ | $+0.95$ | $-0.80$ | $-0.58 \pm 0.14$ | $+0.28$ |
| Data-DR (0.1) | $+0.73$ | $-0.99$ | $+0.77$ | $+0.14 \pm 0.38$ | $+0.89$ |
| Coupled | $+1.75$ | $+0.55$ | $-0.07$ | $-1.39 \pm 0.22$ | $+1.17$ |

- Coupled 는 Data-DR 보다 덜 보수적인 것이 아니다. 학습 분포에서는 더 낸다. 대비한 경우가 다르다.
- Data-DR 은 수요 분포가 움직이고 소매상이 $\hat q$ 를 그대로 믿을 때만 이긴다. 소매상이 움직인 분포를 따라가면 Nominal 과 같아진다. 평가에서는 소매상의 belief 가 $\hat q$ 에 머물지 않으므로 Data-DR 의 보험료는 꼬리에서 돌아오지 않는다.
- Coupled 는 둘이 함께 움직일 때 이긴다. 평가의 $\rho = 1$ 이 그 경우다. $\rho = 0$ 에서 꼬리 이득이 줄어드는 것과 맞는다.

**원고에 쓸 요지.** 한쪽 불확실성만 헤징하는 리더는 한 손잡이만 쓴다. Belief-only 는 거점을 옮기고 Data-DR 은 여유 용량을 줄인다. Data-DR 이 줄인 여유 용량은 소매상이 덜 약정했을 때 수입을 되찾는 길이기도 하므로, Data-DR 의 결정은 belief 오류에 Nominal 보다 약해진다. 둘 다 모른다고 두는 Coupled 는 여유 용량을 덜 줄이고 약정이 덜 흔들리는 거점을 고른다. 그 결과 두 오류가 함께 올 때의 피해를 가장 많이 줄인다.

한계는 다음과 같다. 섭동은 한 쌍 옮기기 90 가지다. 예약료 0.75 한 구성이다. Belief-only 와 Coupled 의 결정은 점 탐색으로 고른 것이다. 약정이 덜 흔들리는 거점의 특성은 밝히지 못했다.

"""
i, j = s.index(a), s.index(b); s = s[:i].rstrip("\n") + "\n\n" + NEW + s[j:]
r5 = "| 5 | **Coupled 는 Data-DR 과 같은 평균 대가로"; k = s.index(r5); k2 = s.index("\n", k) + 1
ROW = "| 6 | **Coupled 는 두 손잡이를 나눠 쓴다.** 여유 용량은 수요가 높을 때의 추가 주문도 받고 소매상이 덜 약정했을 때 수입을 되찾는 길이기도 하다. Data-DR 은 앞의 일만 보고 줄여서 belief 오류에 더 약해지고, Coupled 는 덜 줄이는 대신 약정이 덜 흔들리는 거점을 고른다 (4.6) | 여유 용량 Nominal 106, Data-DR 77, Coupled 83. belief 가 틀릴 때 빠지는 약정 26.5, 32.7, 22.0. 두 오류가 함께 올 때의 피해 11.1, 10.5, 7.9 |\n"
s = s[:k2] + ROW + s[k2:]; io.open(p, "w", encoding="utf-8", newline="").write(s)
shutil.copyfile(p, D + "nz_location_takeorpay_scripts/report_short_main.md")
for f in ("fig_strategy.py", "rec_46.py"): shutil.copyfile(SP + f, D + "nz_location_takeorpay_scripts/" + f)
H = """
### 9.28 네 리더의 헤징 전략 정리 (2026-11-18, 사용자 요청, 보고서 4.6·그림 3)

새 실행 없음. 9.24·9.25·9.27 의 숫자를 한 줄기로 묶어 가치 보고서 4.6 절에 넣었다 (그림 `figs/nz_location_takeorpay/four_strategies.png`, 스크립트 `fig_strategy.py`).

- 핵심: 여유 용량은 두 가지 일을 한다. 수요가 높을 때 추가 주문을 받고, 소매상이 덜 약정했을 때 수입을 되찾는다. Data-DR 은 앞의 일만 보고 줄여서 belief 오류에 더 약해진다. Coupled 는 덜 줄이고 약정이 덜 흔들리는 거점을 고른다 (사용자: 좋은 인사이트, 논문에 넣는 방향).
- 요약 숫자 (150 건, 반지름 0.1): 여유 용량 N 106·B 104·D 77·C 83, belief 가 틀릴 때 빠지는 약정 26.5·23.6·32.7·22.0, 피해 (수요만 / belief 만 / 둘 다) N 5.6/5.1/11.1, B 5.6/3.3/9.5, D 3.9/5.1/10.5, C 4.4/3.3/7.9. Coupled 는 belief 쪽 보호를 Belief-only 만큼, 수요 쪽 보호는 Data-DR 의 70% 를 얻는다.
"""
p = D + "nz_location_capacity_hypothesis_2026-11-14.md"; t = io.open(p, encoding="utf-8").read(); io.open(p, "w", encoding="utf-8", newline="").write(t.rstrip("\n") + "\n" + H)
p = "C:/Users/user/.claude/projects/C--Users-user-robust-bilevel-network-games/memory/nz_location_takeorpay.md"; t = io.open(p, encoding="utf-8").read()
M = "\n- **네 리더의 전략 정리 (2026-11-18, 보고서 4.6·그림 3, 가설 문서 9.28).** 사용자가 좋은 인사이트로 꼽고 논문에 넣자고 한 줄기: 여유 용량은 두 가지 일 (고수요 때 추가 주문, 소매상이 덜 약정했을 때 수입 되찾기) 을 함. Data-DR 은 앞의 일만 보고 줄여 belief 오류에 더 약해짐, Coupled 는 덜 줄이고 약정이 덜 흔들리는 거점을 고름. Belief-only = 거점만, Data-DR = 여유 용량만, Coupled = 둘을 나눠 씀. 원고 tex 반영은 아직 안 함 (허락 필요).\n"
io.open(p, "w", encoding="utf-8", newline="").write(t.rstrip("\n") + "\n" + M); print("recorded")
