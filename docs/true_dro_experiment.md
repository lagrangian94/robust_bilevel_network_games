# Out-of-Sample Test & Value of Robustness: Experiment Design Specification

---

## 1. Model Structure Recap

### 1.1 Decision Timeline

```
x (leader, 확정) → h (follower, 확정) → ξ 실현 → y (flow, wait-and-see)
```

- **Leader**: binary interdiction `x ∈ X`, budget `γ`
- **Follower**: capacity recovery `h ∈ H` under follower's belief `P̃`
- **Nature**: scenario `ξ` realized according to true distribution `P*`
- **Leader's objective**: evaluated under `P*`

### 1.2 Two Layers of Distributional Uncertainty

Leader는 두 분포를 모른다:

1. **P\* (true/nature)**: 실제 scenario 실현 확률. Leader의 OOS objective를 결정.
2. **P̃ (follower's belief)**: Follower가 `h`를 결정할 때 사용하는 분포.

Leader는 자신의 empirical `q̂ = (1/|S|)·1`만 알고, TV-distance 기반 ambiguity set 구성:

$$P^* \in \mathcal{B}_{\varepsilon_1}(\hat{q}), \quad \tilde{P} \in \mathcal{B}_{\varepsilon_2}(\hat{q})$$

### 1.3 세 분포의 삼각관계

```
         P* (true/nature)
        /              \
       /                \
    d(P̂,P*)          d(P̃,P*)
     /                    \
    /                      \
   P̂ (leader) ——————————— P̃ (follower)
               d(P̂,P̃)
```

- ε₁은 d(P̂, P\*)를 커버
- ε₂는 d(P̂, P̃)를 커버
- d(P̃, P\*)는 직접 제어하지 않음 — ε₁, ε₂의 결과로 간접적으로 커버

---

## 2. Instance Generation

### 2.1 Network Structure

**Grid networks** (Cormican et al. 1998, Sadana & Delage 2022):
- m × n grid, source `s` → column 1 → ... → column n → sink `t`
- Within-column arcs: upward/downward with equal probability
- Between-column arcs: always toward sink

**Real-world networks**: Sioux-Falls, NOBEL-US, ABILENE, POLSKA

**Non-interdictable arcs** (standard convention since Cormican et al. 1998):
- Source에서 나가는 arc
- Sink으로 들어가는 arc
- 첫째/마지막 column 내 수직 arc (grid에서)
- 마지막 column으로 들어오는 수평 arc (grid에서)
- Real-world: source/sink 인접 arc

**논거**: 양 끝단을 보호해야 trivial solution (source/sink 바로 앞 차단) 방지. 이건 network interdiction의 표준 관행이며 Cormican et al. (1998), Royset & Wood (2007), Janjarassuk & Linderoth (2008), Atamtürk et al. (2020), Sadana & Delage (2022) 모두 동일.

### 2.2 Capacity Scenario Generation

**Factor model** (Sadana & Delage 2022):
- **Interdictable arcs만** factor model 적용: `c^k = F ξ^k`
  - `F ∈ R₊^{|A_I| × 2}`, entries ~ Uniform(0,1)
  - `ξᵢ ~ Exp(μᵢ)`, `μ ~ Uniform(0,1)`, `k = 2` factors
- **Non-interdictable arcs**: capacity = 100 (모든 scenario 동일)
- |S| = 10 scenarios

**Uniform i.i.d. model** (Lei et al. 2018):
- 각 interdictable arc capacity ~ Uniform(0, 10) per scenario, i.i.d.
- Arc 간 상관구조 없음

**Scenario 고정 원칙**: Factor model로 |S|개 capacity vector를 생성한 뒤 **고정**. 이후 distributional uncertainty는 scenario 위의 probability vector `q ∈ Δ^{|S|}`에 대한 것만 다룬다.

**논거**: Sadana도 동일한 구조. Section 5.3.1에서는 capacity scenario를 여러 번 새로 뽑았지만 (100 instances), Section 5.3.2 (OOS)에서는 capacity를 고정하고 probability만 Dirichlet에서 draw. 우리도 OOS에서는 후자를 따름.

### 2.3 Instance 수

Sadana와 동일하게 각 setting당 10개 random instances.

---

## 3. Parameter Settings

### 3.1 Interdiction Budget γ

`γ = ⌈γ_ratio × |A_I|⌉`

- Grid: `γ_ratio = 0.30` (Sadana: `B = ⌊0.3m⌋`)
- Real-world: `γ_ratio ∈ {0.03, 0.05, 0.10}` (Lei et al.)

### 3.2 Recovery Budget w

`w = ρ × γ × c̄`, where `c̄` = interdictable arc capacity의 scenario 평균

Baseline: **ρ = 0.2**, sensitivity: ρ ∈ {0.1, 0.2, 0.3}

**논거**: Recovery `h`는 uncertainty 실현 전에 이루어지는 확정적 (deterministic) 투자다. ρ가 크면 follower의 deterministic investment가 stochastic capacity를 dominate하여, distributional uncertainty — 그리고 DRO — 가 무의미해진다.

구체적으로 Lei et al.은 ρ ≈ 1.0 수준 (h₀ = 10, Uniform(0,10), γ=2 → w/(γ·c̄) = 10/10 = 1.0)을 사용했는데, 이 경우 interdiction된 arc 중 하나는 항상 완전 복구 가능하다. Lei의 sensitivity analysis에서도 "h₀ > 0이면 결과가 거의 안 변한다"고 보고했는데, 이는 deterministic recovery가 uncertainty를 흡수해버렸기 때문이다.

ρ = 0.2이면 follower의 recovery가 존재하지만 uncertainty를 dominate하지 않는 regime이므로, bilevel 구조와 DRO 모두 의미 있게 작동한다.

### 3.3 Interdiction Effectiveness

`v = 1.0` (완전 차단)

### 3.4 Parameter Summary Table

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Network (grid) | m × n | 5×5, 10×10 | Sadana |
| Scenarios | \|S\| | 10 | Sadana |
| Factors | k | 2 | Sadana |
| Interdiction budget ratio | γ_ratio | 0.30 (grid) | Sadana |
| Recovery power ratio | ρ | 0.2 | See §3.2 |
| Interdiction effectiveness | v | 1.0 | Full interdiction |
| Instances per setting | — | 10 | Sadana |

---

## 4. Dirichlet Distribution: 역할과 해석

### 4.1 뭘 뽑는가

Dirichlet는 **arc capacity를 뽑는 게 아니라**, 고정된 |S|개 scenario 위의 **probability vector** `q ∈ Δ^{|S|}`를 뽑는다.

### 4.2 Concentration Parameter β의 의미 (|S|=10 기준)

Bayesian 관점에서 β는 pseudo-count로 해석된다. 총 pseudo-count = β·|S|이므로, |S|=10일 때 β=0.3이면 총 3개, β=1.0이면 10개, β=50이면 500개의 pseudo-observation에 대응된다. Data가 적을수록 posterior가 퍼지므로 "β 낮음 → ambiguity 큼"이 성립.

| β | 총 pseudo-count | q의 행동 | E[‖q − q̂‖₁] |
|---|----------------|--------|-------------|
| 0.1 | 1 | simplex 꼭짓점 근처 (한 scenario 독식) | ~1.6 |
| 0.3 | 3 | 1~2개 scenario 지배적 | ~1.2 |
| 0.5 | 5 | 일부 scenario에 치우침 | ~1.0 |
| 0.8 | 8 | 약간 치우침 | ~0.7 |
| 1.0 | 10 | simplex 위에서 uniform (어떤 q든 equally likely) | ~0.6 |
| 5.0 | 50 | 중심 근처에 몰림 | ~0.33 |
| 50 | 500 | 거의 중심 | ~0.11 |
| ∞ | ∞ | 정확히 q = (1/|S|)·1 | 0 |

**핵심**: TV distance E[‖q − q̂‖₁]는 β가 감소할수록 **단조롭게 증가**한다. β=1에서 최대가 아니다.

**Nominal model** (q̂ = (1/|S|)·1을 하나의 점으로 고정)은 β → ∞의 극한에 대응된다.

**주의**: β=1.0에서도 E[‖q − q̂‖₁] ≈ 0.6이며, 이는 TV distance 최대값 대비 약 33%로 상당한 deviation이 가능. β에 "hard/easy" 같은 사전 라벨을 붙이지 않고, 실험 결과를 보고 판단한다.

---

## 5. Ambiguity Set Calibration

### 5.1 Symmetric Ignorance 가정

Leader는 P\*와 P̃ 모두 관측 불가능하며, 둘 중 어느 하나에 대해 추가 정보를 갖지 않는다. 따라서:

$$\varepsilon_1 = \varepsilon_2 = \varepsilon$$

**논거**: 이건 calibration의 결과가 아니라, **모델링 원칙에서 직접 따라오는 선택**이다. ε₁ ≠ ε₂를 정당화하려면 follower의 information structure에 대한 추가 가정이 필요한데, 이는 일반적인 방법론 논문에서 부과하기 어렵다. Baseline으로 ε₁ = ε₂를 두고, asymmetric case는 Phase 2 (Information Asymmetry Scenarios)에서 다룬다.

### 5.2 Calibration 절차

ε의 절대적 크기는 Sadana & Delage (2022)의 coverage-based 방법을 TV distance에 맞게 적용:

1. `q⁽¹⁾, ..., q⁽¹⁰'⁰⁰⁰⁾ ~ Dirichlet(β·1_{|S|})` sampling
2. `dᵢ = ‖q⁽ⁱ⁾ − q̂‖₁` 계산
3. `ε = 95th percentile of {d₁, ..., d₁₀,₀₀₀}`

**논거**: Sadana는 n_cal = 100을 사용했으나, 95th percentile을 100개 sample에서 추정하면 sampling error가 무시할 수 없다. Dirichlet sampling은 비용이 거의 0이므로 n_cal = 10,000으로 올려 ε 추정의 안정성을 확보한다. 특히 Phase 1의 ε sweep에서 결과가 ε 값에 민감하므로 (inverted-U curve), ε 자체의 추정이 불안정하면 문제가 된다.

---

## 6. Comparison Models

| Model | ε₁ (OOS) | ε₂ (follower) | 의미 |
|-------|---------|--------------|------|
| Nominal | 0 | 0 | Robustness 없음 |
| Single-layer DRO | ε | 0 | P\*만 robust; P̃ = q̂ 가정 |
| **Two-layer DRO** | ε | ε | 둘 다 robust (proposed model) |

- Nominal → Two-layer DRO gap = **VOR** (전체 robustness 가치)
- Single-DRO → Two-layer DRO gap = follower belief uncertainty 모델링의 추가 가치

별도 metric 이름 (VFR 등)을 붙이지 않고, 세 모델의 OOS performance 테이블에서 reader가 직접 gap을 읽도록 한다.

**논거**: VOR만 정의하고 나머지는 테이블에서 자연스럽게 드러나게 한다. 메트릭이 많으면 reviewer도 헷갈리고 notation이 무거워진다.

---

## 7. Out-of-Sample Evaluation

### 7.1 왜 OOS만 하는가

In-sample에서 "VOR" 같은 metric은 의미가 없다.
- DRO의 in-sample objective가 nominal보다 나쁜 건 **당연하다** — 더 보수적이니까.
- 이건 "가치"가 아니라 "비용"이다.
- DRO의 가치는 P\* ≠ P̂일 때 nominal보다 덜 손해보는 것에서 나오며, 이건 **OOS에서만 확인 가능**하다.

cf. Sadana의 VRS (Value of Randomized Strategy)는 같은 ambiguity set에서 두 solution method를 비교한 것이므로 in-sample에서도 의미가 있었음. 우리의 VOR는 다른 evaluation distribution에서 같은 solution을 비교하는 것이므로 본질적으로 OOS metric.

### 7.2 Nested Sampling Design

**왜 nested인가**: Follower optimization (h\* 계산)은 비싸고, OOS evaluation (가중합)은 거의 공짜. 이 비대칭성을 활용.

```
for each model ∈ {Nominal, Single-DRO, Two-layer DRO}:
    x* = solve_leader(ε₁, ε₂)

    for j = 1, ..., M=100:                          # outer: follower belief (비싼 부분)
        q̃⁽ʲ⁾ ~ draw_follower_belief(scenario)
        h*⁽ʲ⁾ = solve_follower(x*, q̃⁽ʲ⁾)            # LP/MIP, M번만 수행

        for ℓ = 1, ..., L=1000:                      # inner: nature (공짜)
            q*⁽ʲ·ˡ⁾ ~ draw_true(scenario)
            eval⁽ʲ·ˡ⁾ = Σ_k q*_k · flow(x*, h*⁽ʲ⁾, ξᵏ)
```

| | 1:1 pairing | Nested design |
|---|---|---|
| Follower optimization 횟수 | 1,000 | 100 |
| OOS evaluation 횟수 | 1,000 | 100,000 |
| 계산 비용 지배항 | follower opt × 1,000 | follower opt × 100 |

**논거**: Nested design이 계산은 1/10이면서 test instance는 100배 더 많다. 또한 variance decomposition이 자연스럽게 나온다.

### 7.3 같은 Dirichlet에서 뽑아도 되는가?

된다. q\*와 q̃가 같은 Dirichlet에서 independently 나와도 **방향이 독립적**이므로 follower의 h\*가 실제 q\*에 대해 misaligned되는 효과가 자연스럽게 발생한다.

**논거**: 오히려 같은 Dirichlet에서 뽑는 것이 **가장 보수적인 (conservative) 실험 디자인**이다. q̃를 q\*와 의도적으로 다르게 뽑으면 two-layer DRO의 가치가 인위적으로 부풀려질 수 있다 — reviewer가 "DGP의 artifact"이라고 공격할 여지. 같은 Dirichlet에서 independently 뽑으면 discrepancy는 순수하게 finite-sample variation에서 나오므로, 이 세팅에서 two-layer DRO가 이기면 robust한 evidence.

### 7.4 Metrics

**Primary**: Value of Robustness (VOR)

$$\text{VOR} = \frac{\bar{Y}(\text{Nominal}) - \bar{Y}(\text{Two-layer DRO})}{\bar{Y}(\text{Two-layer DRO})} \times 100\%$$

**Secondary**: 95th percentile of OOS distribution (tail risk)

### 7.5 Variance Decomposition

Law of total variance:

$$\text{Var}(Y) = \underbrace{\text{Var}_{\tilde{q}}[\mathbb{E}_{q^*}[Y \mid \tilde{q}]]}_{\text{follower belief effect}} + \underbrace{\mathbb{E}_{\tilde{q}}[\text{Var}_{q^*}[Y \mid \tilde{q}]]}_{\text{nature effect}}$$

Nested sampling에서의 추정:

- Inner statistics: `Ȳⱼ = (1/L) Σ_ℓ eval⁽ʲ·ˡ⁾`, `Sⱼ² = sample var of eval⁽ʲ·*⁾`
- Follower belief effect: `(1/(M-1)) Σⱼ (Ȳⱼ − Ȳ̄)²`
- Nature effect: `(1/M) Σⱼ Sⱼ²`
- Follower share: follower belief effect / total var

**해석**:
- Follower share가 크면 → follower belief uncertainty가 OOS 변동의 주요 driver → two-layer DRO 가치가 큼
- Follower share가 작으면 → nature uncertainty가 지배적 → single-layer DRO로 충분

---

## 8. Value of Robustness Experiments

### Phase 1: Symmetric Sweep (ε₁ = ε₂ = ε)

표준 DRO 실험: uncertainty set size vs OOS performance curve.

**설정**: β ∈ {0.3, 1.0}, 각 β에서 ε\*를 calibrate

- β = 0.3: 총 pseudo-count = β·|S| = 3. Distributional uncertainty가 큼.
- β = 1.0: 총 pseudo-count = β·|S| = 10. β = 0.3보다 uncertainty가 작으나, E[‖q − q̂‖₁] ≈ 0.6으로 여전히 상당한 deviation이 가능. 실제로 "쉬운" setting인지는 결과를 보고 판단.

**Sweep**: 각 β에 대해 ε / ε\* ∈ {0, 0.25, 0.5, 1.0, 1.5, 2.0}

| ε / ε\* | 의미 |
|--------|------|
| 0 | Nominal (overfitting 예상) |
| 1.0 | Calibrated baseline |
| 2.0 | Overconservative |

각 (β, ε)에서 세 모델 (Nominal, Single-DRO, Two-layer DRO) 풀고 OOS evaluate.

**기대 결과**: inverted-U shape. ε가 너무 작으면 overfitting, 너무 크면 overconservatism. Sweet spot 근처에서 Two-layer DRO > Single-DRO > Nominal. β 간 차이의 크기는 사전에 예단하지 않고 결과에서 확인.

**논거**: β를 하나만 쓰면 "그 β에서만 sweet spot이 존재하는 거 아니냐"는 질문에 취약. β ∈ {0.3, 1.0} 두 개를 보여주면 sweet spot의 존재가 β에 robust하다는 것을 확인 가능. 이건 거의 모든 DRO 논문에 있는 표준 실험.

---

### Phase 2: Information Asymmetry Scenarios

세 분포 (P\*, P̃, P̂)의 삼각관계에서 누가 더 정확한지에 따라 three named scenarios.

#### Scenario S (Symmetric Ignorance) — Baseline

Leader도 follower도 truth를 모르며, 동등한 수준의 무지.

| 분포 | 생성 방법 | 파라미터 |
|------|---------|---------|
| q_true | Dir(β · 1) | β = 0.3 |
| q̃ | Dir(β · 1), independent | β = 0.3 |

DRO 설정: ε₁ = ε₂ = ε\*(β=0.3)

**해석**: Discrepancy가 순수 finite-sample variation에서 발생. Two-layer DRO의 baseline 가치를 측정.

---

#### Scenario L (Leader Advantage): P̂ ≈ P\*, P̃ 부정확

Leader가 충분한 data를 보유하여 truth가 q̂ 근처에 있고, follower는 적은 data로 부정확한 belief.

| 분포 | 생성 방법 | 파라미터 |
|------|---------|---------|
| q_true | Dir(β_H · 1) | β_H = 50 |
| q̃ | Dir(β_L · 1) | β_L = 0.3 |

ε calibration:
- ε₁: Dir(β_H = 50)에서 calibrate → **작음** (truth ≈ q̂)
- ε₂: Dir(β_L = 0.3)에서 calibrate → **큼** (follower가 크게 벗어남)

**왜 β_H = 50인가**: β_H = 5이면 각 component의 SD ≈ 0.042로, truth가 (0.02, 0.18, ...) 같이 q̂에서 상당히 벗어날 수 있다. β_H = 50이면 SD ≈ 0.013으로, truth ≈ (0.09, 0.11, 0.10, ...) 수준이 되어 "leader의 모델이 거의 정확하다"는 상황이 성립한다.

**기대 결과**:
- OOS 위험의 주된 원천이 follower의 misaligned recovery
- Single-DRO (ε₂ = 0)는 이걸 무시 → OOS에서 손해
- Two-layer DRO의 가치가 **가장 큼**
- Variance decomposition에서 follower belief effect 지배적

---

#### Scenario F (Follower Advantage): P̃ ≈ P\*, P̂ 부정확

Follower가 truth에 가까운 belief를 가지고, leader의 q̂는 truth에서 멀리 있는 상황.

| 분포 | 생성 방법 | 파라미터 |
|------|---------|---------|
| q_true | Dir(β_L · 1) | β_L = 0.3 |
| q̃ | Dir(κ · q_true) | κ = 50 |

q̃의 생성이 핵심: q_true를 먼저 뽑고, 이를 center로 Dirichlet에서 q̃를 뽑는다.
- `q̃ ~ Dir(κ · q_true)` → `E[q̃] = q_true`
- κ = 50이면 q̃의 effective sample size = 50, 즉 q̃ ≈ q_true (SD ≈ 0.01~0.02 per component)
- κ는 "follower가 truth에 대해 가진 effective sample size"로 해석

**주의**: q̃를 Dir(β_H · 1)에서 뽑으면 q̃ ≈ q̂가 되는 것이지 q̃ ≈ q_true가 아니다.

ε calibration:
- ε₁: Dir(β_L = 0.3)에서 calibrate → **큼**
- ε₂: Leader는 follower의 DGP를 모르므로, 동일하게 Dir(β_L = 0.3)에서 calibrate → **큼** (실제로는 과대추정)

**기대 결과**:
- Follower의 h\*가 이미 truth에 잘 calibrate됨
- Two-layer DRO가 ε₂를 과대추정하여 **overconservative**
- Single-DRO와 비슷하거나 약간 나쁠 수 있음
- Variance decomposition에서 nature effect 지배적

**논거**: Two-layer DRO가 Scenario F에서 가치가 줄어드는 건 weakness가 아니라 **올바른 behavior**다. Follower belief uncertainty가 실제로 없는 상황에서 방어할 필요가 없는 것이 맞다.

---

### Phase 2 Summary Table

| Scenario | q_true | q̃ | ε₁ | ε₂ | Two-layer 가치 |
|----------|--------|---|---|---|--------------|
| **S** (symmetric) | Dir(0.3) | Dir(0.3), indep. | ε\*(0.3) | ε\*(0.3) | baseline |
| **L** (leader 유리) | Dir(50) | Dir(0.3) | ε\*(50), 작음 | ε\*(0.3), 큼 | **가장 큼** |
| **F** (follower 유리) | Dir(0.3) | Dir(50·q_true) | ε\*(0.3), 큼 | ε\*(0.3), 큼(과대) | **가장 작음** |

---

## 9. OOS Evaluation: Scenario별 Nested Loop 구조

세 scenario의 nested loop 구조가 **비대칭**이다. 이는 각 scenario의 information structure가 다르기 때문이며, 의도된 설계다.

### Scenario S & L: 표준 Nested Design (inner loop 활성)

q_true와 q̃가 independent하므로, inner loop에서 q_true를 매번 새로 draw한다.

```
# Scenario S
for j = 1, ..., M:                              # M = 100
    q̃⁽ʲ⁾ ~ Dir(β · 1)                            # β = 0.3
    q_true⁽ʲ⁾는 여기서 안 뽑음 — inner loop에서 뽑음
    h*⁽ʲ⁾ = solve_follower(x*, q̃⁽ʲ⁾)

    for ℓ = 1, ..., L:                           # L = 1000
        q_true⁽ʲ·ˡ⁾ ~ Dir(β · 1)                 # independent draw
        eval⁽ʲ·ˡ⁾ = dot(q_true⁽ʲ·ˡ⁾, flows)

# Scenario L: 동일 구조, β만 다름
for j = 1, ..., M:
    q̃⁽ʲ⁾ ~ Dir(β_L · 1)                          # β_L = 0.3 (follower 부정확)
    h*⁽ʲ⁾ = solve_follower(x*, q̃⁽ʲ⁾)

    for ℓ = 1, ..., L:
        q_true⁽ʲ·ˡ⁾ ~ Dir(β_H · 1)               # β_H = 50 (truth ≈ q̂)
        eval⁽ʲ·ˡ⁾ = dot(q_true⁽ʲ·ˡ⁾, flows)
```

Variance decomposition이 자연스럽게 적용된다:
- Follower belief effect: Var_q̃[E_{q\*}[Y | q̃]]
- Nature effect: E_q̃[Var_{q\*}[Y | q̃]]

### Scenario F: Inner Loop 불필요 (q_true 고정)

Scenario F의 핵심은 "q̃ ≈ q_true이므로 follower의 h\*가 이미 truth에 잘 calibrate되어 있다"는 것이다.

**만약 inner loop에서 q_true를 새로 draw하면 (option b)**: h\*⁽ʲ⁾는 q̃⁽ʲ⁾ ≈ q_true⁽ʲ⁾에 맞춰 최적화되었는데, 평가는 전혀 다른 q_true⁽ʲ·ˡ⁾에 대해 이루어진다. Follower advantage가 evaluation 단계에서 사라지므로 Scenario S와 구분 불가.

따라서 **Scenario F에서는 q_true를 outer loop에서 한 번 뽑고 inner loop에서 고정**한다 (option a):

```
# Scenario F
for j = 1, ..., M:                              # M = 100
    q_true⁽ʲ⁾ ~ Dir(β_L · 1)                     # β_L = 0.3 (truth ≠ q̂)
    q̃⁽ʲ⁾ ~ Dir(κ · q_true⁽ʲ⁾)                    # κ = 50 (q̃ ≈ q_true)
    h*⁽ʲ⁾ = solve_follower(x*, q̃⁽ʲ⁾)

    # Inner loop는 q_true⁽ʲ⁾ 고정 → eval이 deterministic → L = 1이면 충분
    eval⁽ʲ⁾ = dot(q_true⁽ʲ⁾, flows)
```

**결과**: Inner loop가 deterministic이 되어 variance decomposition에서 nature effect = 0. 이는 "follower가 truth를 알면 nature uncertainty가 follower의 recovery를 통해 이미 흡수된다"는 해석과 정확히 일치한다.

### Scenario별 Loop 구조 요약

| Scenario | Outer (M=100) | Inner | Total evals | Var decomposition |
|----------|--------------|-------|-------------|-------------------|
| S | q̃ draw, q_true 별도 | L=1000, q_true 매번 draw | 100,000 | 양쪽 모두 활성 |
| L | q̃ draw, q_true 별도 | L=1000, q_true 매번 draw | 100,000 | 양쪽 모두 활성 |
| F | (q_true, q̃) paired draw | L=1, q_true 고정 | 100 | nature effect = 0 |

보고 시 이 비대칭성을 명시하고, Scenario F에서 inner loop가 불필요한 이유를 정당화한다.

---

## 10. Reporting

### 10.1 Phase 1 Output

**Plot**: ε/ε\* (x-axis) vs OOS mean flow (y-axis), 세 모델 curves. β = 0.3과 β = 1.0 각각 별도 subplot (or overlaid).

**Expected shape**: Nominal (flat line) vs DRO models (inverted-U). β = 0.3에서 curve 간 gap이 더 크고, β = 1.0에서 gap이 줄어듦.

### 10.2 Phase 2 Output

**Main Table**: Scenario × Model × Metrics

| Scenario | Model | OOS Mean | OOS 95th Pctl | Follower Var Share |
|----------|-------|----------|---------------|-------------------|
| S | Nominal | ... | ... | ... |
| S | Single-DRO | ... | ... | ... |
| S | Two-layer DRO | ... | ... | ... |
| L | Nominal | ... | ... | ... |
| L | Single-DRO | ... | ... | ... |
| L | Two-layer DRO | ... | ... | ... |
| F | ... | ... | ... | ... |

**Boxplots** (à la Sadana Figure 3): per-scenario Ȳⱼ 분포, Scenario × Model

### 10.3 핵심 Story

> Scenario L에서 Two-layer DRO가 가장 큰 가치를 보이고, Scenario F에서 가치가 줄어든다. 이는 모델이 올바르게 작동한다는 evidence다: follower belief uncertainty가 **실제로 존재할 때만** two-layer robustness가 필요하다.

---

## 11. Preliminary Validation: DGP 시각화

**본 실험을 진행하기 전에**, 각 scenario의 DGP가 의도한 information structure를 실제로 만들어내는지 시각화를 통해 확인한다. 이 단계는 실험 결과의 해석 가능성을 보장하고, β, β_H, β_L, κ 등의 파라미터 선택이 적절한지 사전 검증하는 역할을 한다.

### 11.1 시각화 항목

각 scenario (S, L, F)에 대해 다음을 시각화:

**Plot 1: 세 분포의 위치 비교 (Simplex 위)**

|S| = 3인 toy example에서 simplex 위에 scatter plot:
- q̂ = (1/3, 1/3, 1/3): 고정점 (×)
- q_true 1000개 draw: 한 색 (●)
- q̃ 1000개 draw: 다른 색 (▲)

세 scenario를 나란히 놓으면:
- Scenario S: 두 cloud가 비슷한 크기로 q̂ 주변에 퍼짐
- Scenario L: q_true cloud가 q̂에 밀착, q̃ cloud가 크게 퍼짐
- Scenario F: q_true cloud가 크게 퍼지고, q̃ cloud가 각 q_true 근처에 따라붙음

**Plot 2: TV distance 분포 비교 (|S| = 10에서)**

|S| = 10에서 세 거리의 histogram을 scenario별로:

| 거리 | 의미 | 계산 |
|------|------|------|
| d(q̂, q_true) | Leader의 model error | ‖q_true − q̂‖₁ |
| d(q̂, q̃) | Leader가 보는 follower deviation | ‖q̃ − q̂‖₁ |
| d(q̃, q_true) | Follower의 실제 calibration quality | ‖q̃ − q_true‖₁ |

기대되는 패턴:

| Scenario | d(q̂, q_true) | d(q̂, q̃) | d(q̃, q_true) |
|----------|-------------|----------|--------------|
| S | 중간 | 중간 | 중간 (≈ √2 × d(q̂, q_true)) |
| L | **작음** | 중간 | **큼** (follower가 truth에서 멀리) |
| F | 중간 | 중간 | **작음** (follower가 truth에 가까움) |

**Plot 3: Calibrated ε와 실제 거리 분포의 관계**

각 scenario에서 calibrate된 ε₁, ε₂를 histogram 위에 vertical line으로 표시. 95% coverage가 실제로 달성되는지 확인.

### 11.2 구현

```julia
"""
    visualize_dgp(; S=10, β=0.3, β_H=50, β_L=0.3, κ=50, n_draws=1000)

세 scenario의 DGP를 시각화하여 파라미터 선택이 적절한지 사전 검증.
"""
function visualize_dgp(; S=10, β=0.3, β_H=50, β_L=0.3, κ=50, n_draws=1000)
    q_hat = fill(1.0/S, S)
    
    for scenario in [:S, :L, :F]
        d_hat_true = zeros(n_draws)
        d_hat_tilde = zeros(n_draws)
        d_tilde_true = zeros(n_draws)
        
        for i in 1:n_draws
            if scenario == :S
                q_true = rand(Dirichlet(S, β))
                q_tilde = rand(Dirichlet(S, β))
            elseif scenario == :L
                q_true = rand(Dirichlet(S, β_H))
                q_tilde = rand(Dirichlet(S, β_L))
            elseif scenario == :F
                q_true = rand(Dirichlet(S, β_L))
                q_tilde = rand(Dirichlet(κ * q_true))
            end
            
            d_hat_true[i] = norm(q_true - q_hat, 1)
            d_hat_tilde[i] = norm(q_tilde - q_hat, 1)
            d_tilde_true[i] = norm(q_tilde - q_true, 1)
        end
        
        # Plot histograms of three distances
        # Mark calibrated ε on histogram
        # Print summary statistics
        println("=== Scenario $scenario ===")
        println("  d(q̂, q_true): mean=$(mean(d_hat_true)), p95=$(quantile(d_hat_true, 0.95))")
        println("  d(q̂, q̃):     mean=$(mean(d_hat_tilde)), p95=$(quantile(d_hat_tilde, 0.95))")
        println("  d(q̃, q_true): mean=$(mean(d_tilde_true)), p95=$(quantile(d_tilde_true, 0.95))")
    end
end
```

### 11.3 검증 기준

시각화 결과에서 다음을 확인:

1. **Scenario S**: d(q̂, q_true) ≈ d(q̂, q̃) ≈ d(q̃, q_true) — 세 거리가 비슷한 order
2. **Scenario L**: d(q̂, q_true) ≪ d(q̂, q̃) 이고 d(q̃, q_true) 큼 — leader 유리, follower 불리
3. **Scenario F**: d(q̃, q_true) ≪ d(q̂, q_true) — follower가 truth에 가까움

만약 이 패턴이 관찰되지 않으면 β_H, β_L, κ 값을 조정한 뒤 본 실험을 진행한다.

---

## 12. Implementation Checklist

- [ ] **Preliminary**: `visualize_dgp()` — DGP 시각화 및 파라미터 적절성 사전 검증
- [ ] Factor model: k=2 factors, F는 interdictable arcs만, non-interdictable = 100
- [ ] `sample_dirichlet(S, β; n_samples)` — Distributions.jl 래퍼
- [ ] `calibrate_epsilon(S, β; n_cal=10000, coverage=0.95)` — L1 distance 기반
- [ ] `solve_follower_weighted(x, q, network, cap; w)` — 임의 scenario weight q̃ 지원
- [ ] `compute_maxflow_per_scenario(x, h, network, cap)` — 각 scenario별 flow 계산
- [ ] `oos_evaluate(x_star, network, cap, scenario_config; M, L)` — nested sampling
- [ ] Leader solver: (ε₁, ε₂) pair 수용; ε₂=0이면 single-layer DRO로 환원
- [ ] Phase 1: ε sweep loop (β ∈ {0.3, 1.0})
- [ ] Phase 2: three scenario DGPs (S, L, F)
- [ ] Reporting: tables, boxplots, variance decomposition
