# Experimental Design: Value of Full Model and Source Decomposition

## 0. Notation and Model Structure

### 0.1 Key Notation

| Symbol | Description |
|--------|-------------|
| $G = (V, A)$ | Directed network |
| $x \in \{0,1\}^{\|A\|}$ | Leader's interdiction decision |
| $h$ | Follower's here-and-now recovery decision |
| $y(\xi)$ | Follower's wait-and-see flow decision |
| $\hat{P}$ | Leader의 empirical distribution ($S$개 scenario) |
| $\varepsilon_1$ | Follower belief에 대한 Wasserstein ball 반경 |
| $\varepsilon_2$ | True distribution에 대한 Wasserstein ball 반경 |
| $S$ | In-sample scenario 수 (leader가 관측한 data 크기) |
| $K_{\text{test}}$ | Out-of-sample test scenario 수 |
| $\gamma$ | Interdiction budget |

### 0.2 Model Structure — 핵심 원칙

Leader는 **자기 자신의 empirical distribution $\hat{P}$만** 가지고 모델을 푼다. Follower의 실제 belief $\tilde{P}$도, true distribution $P_{\text{true}}$도 leader의 모델에 직접 들어가지 않는다.

Leader의 모델은 $\hat{P}$를 중심으로 두 개의 Wasserstein ball을 설정한다:

$$\mathcal{Q}_1 = B_{\varepsilon_1}(\hat{P}): \quad \text{"follower의 belief이 내 empirical에서 최대 } \varepsilon_1 \text{만큼 벗어날 수 있다"}$$

$$\mathcal{Q}_2 = B_{\varepsilon_2}(\hat{P}): \quad \text{"true distribution이 내 empirical에서 최대 } \varepsilon_2 \text{만큼 벗어날 수 있다"}$$

양쪽 모두 **같은 $\hat{P}$ 중심**이다. $\varepsilon = 0$으로 설정하면 해당 방향의 hedging을 하지 않는 것이고, $\varepsilon > 0$이면 해당 방향의 distributional deviation에 대해 robust하게 의사결정하는 것이다.

$\tilde{P}$와 $P_{\text{true}}$는 **오직 out-of-sample evaluation 환경**을 구성할 때만 사용된다. 즉, leader가 구한 $x^*$를 고정한 후, "실제 세계에서 follower가 $\tilde{P}$에 기반해 반응하고, capacity가 $P_{\text{true}}$에서 실현될 때, leader의 의사결정이 얼마나 잘 작동하는가"를 측정하는 데 쓰인다.


---

## 1. Experiment 1: Value of Full Model (VFM)

### 1.1 Research Question

> "왜 follower belief ambiguity와 true distribution ambiguity를 동시에 hedge해야 하는가?"

한쪽만 hedge하거나 아예 hedge하지 않은 모델과 full model의 out-of-sample 성능을 비교하여, 논문의 존재 이유를 증명한다.

### 1.2 Model Variants

네 가지 모델을 정의한다. 모두 동일한 $\hat{P}$ (leader의 $S$개 scenario)로 풀되, $\varepsilon$ 설정만 다르다.

| Model | $\varepsilon_1$ (follower belief hedge) | $\varepsilon_2$ (true dist. hedge) | 해석 |
|-------|:---:|:---:|------|
| **N** (Nominal) | 0 | 0 | Hedging 없음. $\hat{P}$를 그대로 사용 |
| **FO** (Follower-Only) | $\varepsilon$ | 0 | Follower belief deviation만 hedge |
| **TO** (True-Only) | 0 | $\varepsilon$ | True distribution deviation만 hedge |
| **FM** (Full Model) | $\varepsilon$ | $\varepsilon$ | 양쪽 모두 hedge |

$\varepsilon_1 = \varepsilon_2 = \varepsilon$로 동일하게 설정한다. Leader가 두 방향의 deviation을 동일한 수준으로 우려한다는 가정이며, 이를 통해 FO와 TO의 OOS를 직접 비교할 수 있다.

### 1.3 Data Generation Protocol

**Step 1. 모델 풀기 (in-sample)**

1. DGP에서 $S$개 scenario를 생성한다. 이것이 leader의 empirical $\hat{P}$이다.
2. 네 모델 (N, FO, TO, FM)을 각각 $\hat{P}$와 해당 $(\varepsilon_1, \varepsilon_2)$로 풀어 $x^*_{\text{N}}, x^*_{\text{FO}}, x^*_{\text{TO}}, x^*_{\text{FM}}$을 얻는다.

**Step 2. OOS 환경 구성**

3. 동일 DGP에서 독립적으로 $S_{\text{follower}}$개 scenario를 뽑는다. 이것이 follower의 actual belief $\tilde{P}$이다. (Leader는 이 $\tilde{P}$를 모름. OOS evaluation용.)
4. 동일 DGP에서 독립적으로 $K_{\text{test}}$개 test scenario를 뽑는다. 이것이 실제로 실현되는 capacity이다.

**Step 3. OOS evaluation**

5. 각 $x^*$를 고정한 채, 각 test scenario $\xi^{(k)}$ ($k = 1, \ldots, K_{\text{test}}$)에 대해:
   - Follower가 $\tilde{P}$에 기반해 here-and-now recovery $h^*$를 결정
   - Capacity $\xi^{(k)}$가 실현된 후 follower가 max-flow $y^*$를 결정
   - Leader가 경험하는 actual flow를 기록
6. $K_{\text{test}}$개에 대한 평균 (혹은 worst-case)이 해당 $x^*$의 OOS objective.

**Step 4.** 서로 다른 random seed로 위 과정을 $R$회 반복.

### 1.4 Parameter Settings

| Parameter | Values | Rationale |
|-----------|--------|-----------|
| Network | Grid $5 \times 5$, ABILENE, POLSKA, Sioux-Falls | Lei et al. (2018) Table 3 참고 |
| DGP | Uniform$[0,10]$ i.i.d. | Lei et al. (2018)과 동일 |
| | Log-normal factor model ($K_{\text{factor}} = 2$) | Sadana & Delage (2022) 스타일 |
| $S$ | 10 | Sadana의 $\|K\| = 10$ 과 일관 |
| $\varepsilon$ | $\{0.25, 0.5, 1.0, 2.0\}$ | $\varepsilon = 0$은 N 모델이 커버 |
| $\gamma$ | $\lceil 0.05\|A\|\rceil$, $\lceil 0.10\|A\|\rceil$ | Lei et al. Table 4 참고 |
| $\rho$ (recovery ratio) | 0.2 | Lei et al. 참고 |
| $K_{\text{test}}$ | 1000 | Lei et al.의 OOS 설정 |
| $R$ (replications) | 30 | 평균 및 std 보고 |

### 1.5 Result Table Format

각 (Network, DGP, $\gamma$) 조합에 대해:

| $\varepsilon$ | $Z^*_{\text{N}}$ | $Z^*_{\text{FO}}$ | $Z^*_{\text{TO}}$ | $Z^*_{\text{FM}}$ | $\text{OOS}_{\text{N}}$ | $\text{OOS}_{\text{FO}}$ | $\text{OOS}_{\text{TO}}$ | $\text{OOS}_{\text{FM}}$ |
|---|---|---|---|---|---|---|---|---|
| 0.25 | | | | | | | | |
| 0.5 | | | | | | | | |
| 1.0 | | | | | | | | |
| 2.0 | | | | | | | | |

왼쪽 4열: in-sample optimal objective. 오른쪽 4열: out-of-sample performance. 각 셀은 $R$회 replication의 평균 (± std).

### 1.6 Expected Observations

- **In-sample**: $Z^*_{\text{FM}} \geq Z^*_{\text{FO}}, Z^*_{\text{TO}} \geq Z^*_{\text{N}}$은 trivially 성립 (더 큰 ambiguity set → 더 높은 worst-case). 이것 자체는 아무것도 증명하지 못함.
- **Out-of-sample**: 핵심은 $\text{OOS}_{\text{FM}} \leq \text{OOS}_{\text{TO}}$ 및 $\text{OOS}_{\text{FM}} \leq \text{OOS}_{\text{FO}}$ 성립 여부. FM의 $x^*$가 실제 환경에서 더 낮은 max flow를 달성하면, 양쪽 모두 hedge하는 것이 더 나은 의사결정을 이끈다는 증거.
- $\text{OOS}_{\text{FO}}$ vs $\text{OOS}_{\text{TO}}$ 비교가 자연스럽게 "어느 방향의 hedge가 더 효과적인가"의 1차 답변을 제공.
- $\varepsilon$가 너무 크면 overconservatism으로 OOS 악화 예상 (sweet spot 존재).


---

## 2. Experiment 2: Source Decomposition

### 2.1 Research Question

> "True distribution uncertainty와 follower belief uncertainty 중 어느 것이 leader의 성능에 더 큰 영향을 미치는가?"

Experiment 1이 "full model이 필요하다"는 정성적 답을 주었다면, Experiment 2는 두 uncertainty source의 상대적 중요도를 정량적으로 분리한다.

### 2.2 설계 원칙

두 시나리오를 설계한다. **모델 측면**에서는 $(\varepsilon_1, \varepsilon_2)$ 설정이 다르고, **OOS evaluation 환경**에서 $\tilde{P}$와 $P_{\text{true}}$의 관계가 다르다. 각 시나리오는 하나의 uncertainty source를 isolate한다.

모든 경우 모델의 input은 leader의 $\hat{P}$ ($S$개 scenario)뿐이다. $\tilde{P}$, $P_{\text{true}}$는 모델에 들어가지 않으며, OOS evaluation에서만 사용된다.

### 2.3 Scenario A: "Leader knows true dist. — follower belief만 문제"

**해석**: Leader의 empirical $\hat{P}$가 실제로 true distribution과 일치하는 환경. 유일한 risk source는 follower가 다른 belief을 갖는 것.

**모델**: $\varepsilon_2 = 0$ 고정, $\varepsilon_1$ vary. Leader는 true dist. deviation은 hedge하지 않고, follower belief deviation만 hedge한다.

**OOS 환경**: $P_{\text{true}} = \hat{P}$ (leader의 empirical이 곧 truth). Follower는 독립적으로 뽑은 $S$개 scenario에 기반한 $\tilde{P} \neq \hat{P}$로 반응.

**Protocol:**
1. DGP에서 $S$개 → leader의 $\hat{P}$.
2. 동일 DGP에서 독립적으로 $S$개 → follower의 $\tilde{P}$ (OOS용).
3. 각 $\varepsilon_1 \in \{0, 0.25, 0.5, 1.0, 2.0\}$에 대해 모델 풀기 ($\varepsilon_2 = 0$) → $x^*_A(\varepsilon_1)$.
4. OOS: $\hat{P}$ (= $P_{\text{true}}$)에서 $K_{\text{test}}$개 test scenario 생성. Follower는 $\tilde{P}$에 기반해 반응. Actual flow 측정.

**Message**: $\varepsilon_1$을 키울수록 OOS가 개선되면, 이것은 순수하게 **game-theoretic contribution** (follower의 private belief을 모델링하는 것)의 가치이다.

### 2.4 Scenario B: "Follower knows true dist. — leader의 정보 부족만 문제"

**해석**: Follower의 belief과 true distribution이 동일한 환경. Leader의 유일한 risk source는 자기 empirical $\hat{P}$가 $P_{\text{true}}$와 다르다는 것.

**모델**: $\varepsilon_1 = \varepsilon_2 = \varepsilon$, 같이 vary. Leader는 follower와 nature가 aligned되어 있되, 그 공통 분포가 자신의 empirical과 다를 수 있다고 우려하므로 동일한 $\varepsilon$로 양쪽을 hedge.

**OOS 환경**: $P_{\text{true}} \neq \hat{P}$ (leader의 empirical은 $P_{\text{true}}$의 subsample). $\tilde{P} = P_{\text{true}}$ (follower는 true dist.를 안다).

**Protocol:**
1. DGP에서 large pool 생성 → 이 전체를 $P_{\text{true}}$로 간주.
2. 이 중 $S = 10$개를 추출하여 leader에게 제공 → $\hat{P}$.
3. 각 $\varepsilon \in \{0, 0.25, 0.5, 1.0, 2.0\}$에 대해 모델 풀기 ($\varepsilon_1 = \varepsilon_2 = \varepsilon$) → $x^*_B(\varepsilon)$.
   - 모델은 여전히 $\hat{P}$ ($S=10$개)만 사용.
4. OOS: $P_{\text{true}}$에서 $K_{\text{test}}$개 test scenario 생성. Follower는 $P_{\text{true}}$를 알고 최적 반응. Actual flow 측정.

**Message**: $\varepsilon$를 키울수록 OOS가 개선되면, 이것은 순수하게 **distributional robustness** (leader의 유한 표본에 의한 정보 부족을 hedge하는 것)의 가치이다.

### 2.5 Parameter Settings

Experiment 1과 동일한 network/DGP 조합 중 가장 interesting한 2–3개에 집중.

| Parameter | Values |
|-----------|--------|
| $\varepsilon$ | $\{0, 0.25, 0.5, 1.0, 2.0\}$ |
| Network | Experiment 1에서 선택 |
| $R$ (replications) | 30 |
| 나머지 | Experiment 1과 동일 |

### 2.6 Result Presentation

각 네트워크에 대해, Scenario A와 B의 curve를 같은 figure에 겹쳐 그린다.

- x축: $\varepsilon$ (Scenario A에서는 $\varepsilon_1$, Scenario B에서는 $\varepsilon$)
- y축: Baseline ($\varepsilon = 0$) 대비 OOS improvement rate

$$\Delta_A(\varepsilon_1) = \frac{\text{OOS}_A(\varepsilon_1 = 0) - \text{OOS}_A(\varepsilon_1)}{\text{OOS}_A(\varepsilon_1 = 0)}, \quad \Delta_B(\varepsilon) = \frac{\text{OOS}_B(\varepsilon = 0) - \text{OOS}_B(\varepsilon)}{\text{OOS}_B(\varepsilon = 0)}$$

$\Delta > 0$: hedging이 OOS를 개선. $\Delta < 0$: overconservatism.

### 2.7 Expected Observations and Interpretation

| 패턴 | 해석 |
|------|------|
| $\max \Delta_A \gg \max \Delta_B$ | Follower belief uncertainty가 dominant. Game-theoretic 모델링의 가치가 크다. |
| $\max \Delta_A \ll \max \Delta_B$ | Leader의 정보 부족이 dominant. Standard DRO의 가치가 크다. |
| $\max \Delta_A \approx \max \Delta_B$ | 양쪽 모두 중요. Full model이 필수. |
| 네트워크마다 패턴이 다름 | "어디에 집중해야 하는가"는 network topology에 depend한다는 practical insight. |

각 curve에서 $\Delta$가 최대가 되는 $\varepsilon^*$의 위치도 보고한다. $\varepsilon^*$가 작으면 약간의 hedging으로도 효과가 크고, $\varepsilon^*$가 크면 큰 ambiguity set이 필요하다는 뜻.


---

## 3. Computational Considerations

### 3.1 Total Instance Count

**Experiment 1**: 4 networks × 2 DGPs × 2 budgets × 4 $\varepsilon$ × 4 models × 30 replications = 3,840 solves + baselines. 소규모 네트워크 먼저 수행.

**Experiment 2**: 2–3 networks × 2 scenarios × 5 $\varepsilon$ × 30 replications = 600–900 solves.

### 3.2 Implementation Notes

- **$\varepsilon_1 \neq \varepsilon_2$ 지원**: S-lemma 제약에서 leader 측 LMI의 $\varepsilon^2$과 follower 측 LMI의 $\varepsilon^2$을 별도 파라미터로 분리. 구조적 변경은 최소이며, `build_robust_counterpart_matrices`에서 epsilon 인자를 두 개로 확장하면 됨.
- **OOS evaluation**: 주어진 $x^*$에서 follower의 best response는 LP (max-flow with recovery)이므로 빠르게 계산 가능. $K_{\text{test}} = 1000$개에 대해 반복.
- **Scenario A에서 $P_{\text{true}} = \hat{P}$의 OOS test 생성**: leader의 empirical DGP 파라미터로 새 scenario를 생성하거나, 동일 DGP에서 재샘플링.

### 3.3 Solver Configuration

- SDP solver: MOSEK (primary), Hypatia (backup)
- MIP (interdiction): Gurobi
- Benders decomposition: nested trust-region variant (기존 구현)
- Time limit per instance: 3600s
- Optimality tolerance: 0.01%


---

## 4. Summary

| | Experiment 1 (VFM) | Experiment 2 (Source Decomposition) |
|---|---|---|
| **질문** | Full model이 필요한가? | 두 source 중 어디가 더 중요한가? |
| **비교 대상** | N vs FO vs TO vs FM | Scenario A vs Scenario B |
| **모델의 input** | 항상 leader의 $\hat{P}$ ($S$개)만 사용 | 동일 |
| **$\varepsilon$ 설정** | $\varepsilon_1 = \varepsilon_2 = \varepsilon$, vary $\varepsilon$ | A: $\varepsilon_2 = 0$, vary $\varepsilon_1$. B: $\varepsilon_1 = \varepsilon_2 = \varepsilon$, vary |
| **OOS 환경** | $P_{\text{true}}, \tilde{P}$ 모두 $\hat{P}$와 독립 | A: $P_{\text{true}} = \hat{P}$, $\tilde{P} \neq \hat{P}$. B: $\tilde{P} = P_{\text{true}} \neq \hat{P}$ |
| **결과 형태** | Table (in-sample + OOS) | Figure ($\Delta$ curves) |
| **Story** | 논문의 존재 이유 증명 | Practical insight 제공 |

Experiment 1이 coarse한 질문 ("양쪽 모두 hedge해야 하는가?")에 답하고, Experiment 2가 fine-grained insight ("어느 방향의 hedge가 더 효과적인가?")을 제공하는 계층적 구조이다.
