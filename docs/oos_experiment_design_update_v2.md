# Out-of-Sample Test & Value of Robustness: Experiment Design Specification (v4)

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
- Source에서 나가는 arc, Sink으로 들어가는 arc
- 첫째/마지막 column 내 수직 arc (grid에서)
- 마지막 column으로 들어오는 수평 arc (grid에서)
- Real-world: source/sink 인접 arc

**논거**: 양 끝단을 보호해야 trivial solution 방지. Network interdiction의 표준 관행 (Cormican et al. 1998, Sadana & Delage 2022 등).

### 2.2 Capacity Scenario Generation

**Factor model** (Sadana & Delage 2022):
- **Interdictable arcs만** factor model 적용: `c^k = F ξ^k`
  - `F ∈ R₊^{|A_I| × 2}`, entries ~ Uniform(0,1)
  - `ξᵢ ~ Exp(μᵢ)`, `μ ~ Uniform(0,1)`, `k = 2` factors
- **Non-interdictable arcs**: capacity = 100 (모든 scenario 동일)
- |S| = 10 scenarios

**Scenario 고정 원칙**: Factor model로 |S|개 capacity vector를 생성한 뒤 **고정**. 이후 distributional uncertainty는 scenario 위의 probability vector `q ∈ Δ^{|S|}`에 대한 것만 다룬다.

### 2.3 Instance 수

각 setting당 **10개 random instances**. Scenario set의 randomness에 의한 변동을 평균화하기 위함.

---

## 3. Parameter Settings

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Network (grid) | m × n | 5×5, 10×10 | Sadana |
| Scenarios | \|S\| | 10 | Sadana |
| Factors | k | 2 | Sadana |
| Interdiction budget ratio | γ_ratio | 0.30 (grid) | Sadana |
| Recovery power ratio | ρ | 0.2 | See below |
| Interdiction effectiveness | v | 1.0 | Full interdiction |
| Instances per setting | — | 10 | Sadana |

**Recovery budget** `w = ρ × γ × c̄`. ρ = 0.2이면 follower의 recovery가 존재하지만 uncertainty를 dominate하지 않는 regime. Sensitivity: ρ ∈ {0.1, 0.2, 0.3}.

---

## 4. Dirichlet Distribution: 역할과 해석

Dirichlet는 고정된 |S|개 scenario 위의 **probability vector** `q ∈ Δ^{|S|}`를 뽑는다.

Bayesian 관점에서 β는 pseudo-count. 총 pseudo-count = β·|S|.

| β | 총 pseudo-count | q의 행동 | E[‖q − q̂‖₁] |
|---|----------------|--------|-------------|
| 0.1 | 1 | simplex 꼭짓점 근처 | ~1.6 |
| 0.3 | 3 | 1~2개 scenario 지배적 | ~1.2 |
| 0.5 | 5 | 일부 scenario에 치우침 | ~1.0 |
| 0.8 | 8 | 약간 치우침 | ~0.7 |
| 1.0 | 10 | simplex 위 uniform | ~0.6 |

**핵심**: E[‖q − q̂‖₁]는 β 감소 시 단조 증가. Nominal model은 β → ∞의 극한에 대응.

---

## 5. Ambiguity Set Calibration

### 5.1 Symmetric Ignorance 가정

$$\varepsilon_1 = \varepsilon_2 = \varepsilon$$

**논거**: 모델링 원칙에서 직접 따라오는 선택. ε₁ ≠ ε₂는 follower의 information structure에 대한 추가 가정 필요.

### 5.2 Coverage-based Calibration

1. `q⁽¹⁾, ..., q⁽¹⁰'⁰⁰⁰⁾ ~ Dirichlet(β·1_{|S|})` sampling
2. `dᵢ = ‖q⁽ⁱ⁾ − q̂‖₁` 계산
3. `ε = 95th percentile of {d₁, ..., d₁₀,₀₀₀}`

n_cal = 10,000 (Dirichlet sampling 비용 ≈ 0이므로 ε 추정의 안정성 확보).

### 5.3 Analytical Calibration: PO/PP/NR/WR (Rahimian et al. 2019)

Coverage-based 방법의 보완으로, TV-DRO 문헌의 analytical calibration을 적용한다. 이 방법은 i.i.d. data 가정에 의존하지 않으며, 모델 자체의 구조로부터 ε의 적절한 range를 결정한다.

#### 5.3.1 세 Solution

| Symbol | ε | 구하는 법 |
|--------|---|---------|
| $x^{\text{neut}}$ | 0 | `solve_leader(ε₁=0, ε₂=0)` — Nominal model |
| $x^*_\varepsilon$ | ε | `solve_leader(ε₁=ε, ε₂=ε)` — 각 ε에서 DRO model |
| $x^{\text{rob}}$ | ε_max | `solve_leader(ε₁=1, ε₂=1)` — Full robust (또는 충분히 큰 ε) |

$x^{\text{neut}}$와 $x^{\text{rob}}$은 한 번만 풀면 되고, $x^*_\varepsilon$은 ε grid의 각 점에서 한 번씩 풀어야 한다.

#### 5.3.2 $f_\varepsilon(x)$ 계산: 임의의 x를 임의의 ε 잣대로 평가

$f_\varepsilon(x)$는 "solution $x$를 ε 수준의 DRO objective로 평가한 값"이다. $x$와 $\varepsilon$은 독립적으로 넣을 수 있다.

$$f_\varepsilon(x) = \sup_{p \in \mathcal{B}_\varepsilon(\hat{q})} \sum_k p_k \cdot \text{flow}_k(x, h^*, \xi^k)$$

**계산 Step 1**: $x$를 고정한 상태에서 follower의 worst-case recovery $h^*$ 결정.

Two-layer 구조에서 follower도 DRO이므로, $h^*$는 ε₂에 의존한다. 여기서는 symmetric (ε₁=ε₂=ε)이므로:
- ε=0일 때: follower는 $\hat{q}$ 하에서 최적화 → `h* = solve_follower(x, q̂)`
- ε>0일 때: follower도 ambiguity set 내 worst-case → `h* = solve_follower_robust(x, ε)`

**간소화**: $f_\varepsilon(x)$ 계산 시 follower의 ε₂를 평가 대상 ε과 같게 놓는다. 즉 $f_\varepsilon(x)$는 "ε₁=ε₂=ε인 세상에서 이 x의 objective value"이다.

**계산 Step 2**: 각 scenario별 flow 계산.

```julia
flows = [maxflow(x, h_star, ξ_k) for k in 1:|S|]
```

**계산 Step 3**: TV ball 위에서 inner worst-case expectation.

$$\sup_p \left\{ \sum_k p_k \cdot \text{flow}_k \;:\; \frac{1}{2}\|p - \hat{q}\|_1 \leq \varepsilon,\; \sum_k p_k = 1,\; p \geq 0 \right\}$$

TV distance에서 이 문제는 **LP**로 풀린다. 또는 closed-form으로: flow를 내림차순 정렬한 뒤, worst-case $p$는 가장 큰 flow의 scenario에 가능한 최대 weight를 부여한다.

```julia
function tv_worst_case_expectation(flows, q_hat, ε)
    # LP formulation
    # max  Σ p_k * flows_k
    # s.t. Σ |p_k - q_hat_k| ≤ 2ε   (TV distance ≤ ε ↔ L1 ≤ 2ε)
    #      Σ p_k = 1
    #      p_k ≥ 0
    #
    # 또는 Greedy:
    # flow를 내림차순 정렬, 큰 flow scenario에 weight를 올리고
    # 작은 flow scenario에서 weight를 빼되, 총 이동량 ≤ 2ε
    
    S = length(flows)
    sorted_idx = sortperm(flows, rev=true)  # 내림차순
    p = copy(q_hat)
    budget = 2ε  # L1 이동 가능량
    
    # 가장 flow가 작은 scenario에서 weight를 빼서
    # 가장 flow가 큰 scenario에 weight를 올림
    # (양방향 이동이므로 L1 변화 = 2 × 이동량)
    
    # ... LP solver 또는 greedy implementation
    return dot(p, flows)
end
```

**주의**: 정확한 구현은 LP solver를 사용하는 것이 안전하다. TV ball constraint는 linear이므로 어떤 LP solver든 사용 가능.

#### 5.3.3 네 가지 Measure

```julia
PO_ε = f_ε(x_neut) - f_ε(x_star_ε)   # Price of Optimism
PP_ε = f_ε(x_rob)  - f_ε(x_star_ε)   # Price of Pessimism
NR_ε = f_0(x_star_ε) - f_0(x_neut)   # Nominal Regret
WR_ε = f_1(x_star_ε) - f_1(x_rob)   # Worst-case Regret
```

**해석**:

| Measure | 질문 | ε 증가 시 |
|---------|------|----------|
| PO | "DRO 세상인데 SP solution 고집하면 얼마나 손해?" | 증가 (낙관의 대가 ↑) |
| PP | "DRO 세상인데 RO solution 고집하면 얼마나 손해?" | 감소 (RO가 점점 적절) |
| NR | "SP 세상인데 DRO solution 쓰면 얼마나 후회?" | 증가 (불필요한 보수성 ↑) |
| WR | "RO 세상인데 DRO solution 쓰면 얼마나 후회?" | 감소 (DRO → RO 수렴) |

모두 항상 ≥ 0 (suboptimality/regret).

#### 5.3.4 두 Indifference Level

**ε^S (solution indifference)**: PO_ε = PP_ε인 가장 작은 ε.
- 이 ε에서 SP solution과 RO solution이 DRO 잣대로 **같은 cost**
- ε < ε^S: 낙관이 비관보다 싸다 → 아직 robust할 필요 적음

**ε^D (distribution indifference)**: NR_ε = WR_ε인 가장 작은 ε.
- 이 ε에서 "nominal이 맞을 때의 후회" = "worst-case가 맞을 때의 후회"
- ε > ε^D: nominal regret > worst-case regret → 불필요하게 보수적

**성질**: ε^S ≤ ε^D ≤ ε^cr (Rahimian et al. 2019, Theorem 4).

**추천 range**: ε ∈ [ε^S, ε^D].

#### 5.3.5 계산 절차: Combined Bisection

Grid search는 DRO를 매 grid point에서 풀어야 하므로 비싸다. 대신 ε^S와 ε^D를 **동시에** 탐색하는 bisection을 사용한다. ε^S ≤ ε^D이므로, 현재 ε이 세 영역 중 어디에 있는지 판별하여 방향을 정한다.

| PO − PP | NR − WR | 위치 | 행동 |
|---------|---------|------|------|
| < 0 | < 0 | ε < ε^S | ε 올려야 |
| ≥ 0 | < 0 | ε^S ≤ ε ≤ ε^D | **범위 안. Stop.** |
| ≥ 0 | ≥ 0 | ε > ε^D | ε 내려야 |

**핵심 비용 절감**: PO − PP = f_ε(x^neut) − f_ε(x^rob)는 **DRO를 풀지 않고** 계산 가능 (x^neut, x^rob은 이미 구해져 있고 evaluate_f만 호출). NR − WR만 x*_ε가 필요하므로 DRO를 풀어야 한다.

따라서 **PO − PP를 먼저 체크하여 ε < ε^S인 경우를 빠르게 걸러내고**, ε ≥ ε^S일 때만 DRO를 풀어 NR − WR을 계산한다.

```julia
function find_epsilon_range(network, scenarios; 
                            ε_start=0.5, ε_max=1.0, tol=0.01, max_iter=15)
    # Step 1: 양 극단 (한 번만)
    x_neut = solve_leader(ε₁=0, ε₂=0)
    x_rob  = solve_leader(ε₁=ε_max, ε₂=ε_max)
    f0_neut = evaluate_f(x_neut, 0)         # 상수
    f1_rob  = evaluate_f(x_rob, ε_max)      # 상수
    
    # Step 2: Combined bisection
    lo, hi = 0.0, ε_max
    ε = ε_start
    history = []   # 탐색 경로 기록
    
    for iter = 1:max_iter
        # PO - PP (싸다: DRO 안 풂)
        po_pp = evaluate_f(x_neut, ε) - evaluate_f(x_rob, ε)
        
        if po_pp < 0
            # ε < ε^S → 올려야. DRO 안 풀어도 됨.
            push!(history, (ε=ε, po_pp=po_pp, nr_wr=NaN, region="below εS"))
            lo = ε
            ε = (ε + hi) / 2
            continue
        end
        
        # PO - PP ≥ 0 → ε ≥ ε^S. NR - WR 체크 (비싸다: DRO 풀어야)
        x_star = solve_leader(ε₁=ε, ε₂=ε)
        nr = evaluate_f(x_star, 0) - f0_neut
        wr = evaluate_f(x_star, ε_max) - f1_rob
        nr_wr = nr - wr
        
        if nr_wr < 0
            # ε^S ≤ ε ≤ ε^D → 범위 안. Stop.
            push!(history, (ε=ε, po_pp=po_pp, nr_wr=nr_wr, region="IN [εS, εD]"))
            
            return (
                ε_recommended = ε,
                x_neut = x_neut,           # ★ 저장: Experiment 3,4,5에서 재활용
                x_rob  = x_rob,            # ★ 저장: 재활용
                x_star = x_star,           # ★ 저장: Two-layer DRO solution
                PO = evaluate_f(x_neut, ε) - evaluate_f(x_star, ε),
                PP = evaluate_f(x_rob, ε) - evaluate_f(x_star, ε),
                NR = nr,
                WR = wr,
                history = history,
            )
        else
            # ε > ε^D → 내려야
            push!(history, (ε=ε, po_pp=po_pp, nr_wr=nr_wr, region="above εD"))
            hi = ε
            ε = (lo + ε) / 2
        end
    end
    
    # max_iter 도달 시 현재 ε 반환
    return (ε_recommended=ε, history=history)
end
```

**비용**: 매 iteration에서 최대 DRO 1번 + evaluate_f 4번. PO − PP < 0인 iteration에서는 DRO를 풀지 않으므로 더 싸다. 보통 **3~5회**면 범위 안에 도달.

**evaluate_f 함수**:

```julia
function evaluate_f(x, ε)
    # x를 고정, ε 수준에서 DRO objective value 계산
    
    # Step 1: follower의 response (ε₂ = ε에서)
    if ε ≈ 0
        h_star = solve_follower(x, q_hat)              # nominal follower
    else
        h_star = solve_follower_worst_case(x, ε)       # robust follower
    end
    
    # Step 2: 각 scenario별 flow
    flows = [maxflow(x, h_star, ξ_k) for k in 1:|S|]
    
    # Step 3: inner worst-case expectation over TV ball
    if ε ≈ 0
        return dot(q_hat, flows)                        # nominal expectation
    else
        return tv_worst_case_expectation(flows, q_hat, ε)  # LP
    end
end
```

#### 5.3.6 보고

**Table**: 탐색 경로 (history)

| Iter | ε | PO−PP | NR−WR | Region |
|------|---|-------|-------|--------|
| 1 | 0.50 | +0.12 | +0.08 | above ε^D |
| 2 | 0.25 | −0.05 | — | below ε^S |
| 3 | 0.375 | +0.03 | −0.02 | **IN [ε^S, ε^D]** |

**최종 출력**: ε_recommended, PO, PP, NR, WR at that point.

**Optional plot**: 탐색 과정에서 얻은 점들로 PO−PP와 NR−WR의 sign을 ε 축에 표시.

**이 실험이 OOS보다 먼저 수행되어야 한다**: ε_recommended를 결정한 뒤, 이 ε으로 Phase A/B OOS 실험을 진행. 각 β의 ε_coverage와 ε_recommended를 비교 보고.

---

## 6. Comparison Models

| Model | ε₁ | ε₂ | 의미 |
|-------|---|---|------|
| Nominal | 0 | 0 | Robustness 없음 |
| Single-layer DRO | ε | 0 | P\*만 robust |
| **Two-layer DRO** | ε | ε | 둘 다 robust (proposed) |

---

## 7. Out-of-Sample Evaluation

### 7.1 구조적 제약: Linear Collapse 문제

우리 모델의 OOS evaluation은 $\text{eval} = \sum_k q_{\text{true},k} \cdot \text{flow}_k(x^*, h^*)$로 $q_{\text{true}}$에 **linear**이다. Symmetric Dirichlet에서 $q_{\text{true}}$를 sampling하면 $\mathbb{E}[q_{\text{true}}] = (1/|S|) \cdot \mathbf{1} = \hat{q}$이므로:

$$\mathbb{E}[\text{eval}] = \sum_k \frac{1}{|S|} \cdot \text{flow}_k(x^*, h^*) = f_0(x^*)$$

이는 nominal objective로 평가한 값과 동일하다. Nominal model의 $x^{\text{neut}}$이 $f_0$를 minimize하므로 **OOS mean에서 nominal이 항상 이긴다.** 이 현상은 β 무관하고, Normal sampling에서도 동일하다 ($\mathbb{E}[p] = \hat{q}$이면 항상 발생).

**문헌적 선례**: Ben-Tal et al. (2013)도 phi-divergence DRO에서 동일한 구조를 가지며, mean 대신 **range (min, max)**를 보고하여 해결했다. Sadana & Delage (2022)는 CVaR objective ($q$에 비선형)를 사용하여 이 문제를 자연스럽게 우회했다. Rahimian et al. (2019)은 OOS를 하지 않고 analytical calibration (PO/PP/NR/WR)을 사용했다.

**해결 방향**: 두 가지 보완적 방법을 사용한다.

### 7.2 OOS Phase A: Symmetric Dirichlet + Tail Metrics

Ben-Tal et al. (2013) Section 6.4의 프로토콜. OOS mean 대신 **distributional statistics**로 DRO의 가치를 측정한다.

**원리**: $q_{\text{true}}$가 $\hat{q}$에서 크게 벗어난 "나쁜 realization"에서 DRO가 nominal보다 flow를 잘 줄이는지 확인. Mean에서는 보이지 않지만 **tail (p95, max)**에서 드러난다.

DRO의 $x^*$는 모든 scenario의 flow를 고르게 낮추므로, $q_{\text{true}}$가 worst scenario에 weight를 몰릴 때 nominal보다 나은 protection을 제공한다. 이건 보험과 같은 구조: 평균 수익은 떨어지지만 재난 시 파산을 막는다.

```
for each β ∈ {0.1, 0.3, 0.5, 0.8}:
  ε = calibrate_epsilon(β)

  for each model ∈ {Nominal, Single-DRO, Two-layer DRO}:
    x* = solve_leader(model, ε)

    for j = 1, ..., M=100:                         # follower belief
      q_tilde ~ Dirichlet(β · 1)
      h* = solve_follower(x*, q_tilde)
      flows = [maxflow(x*, h*, ξ^k) for k in 1:|S|]

      for ℓ = 1, ..., L=100:                       # nature
        q_true ~ Dirichlet(β · 1)
        eval[j*L + ℓ] = dot(q_true, flows)

  # Metrics
  metrics[model] = (
    mean  = mean(eval),            # 참고용 (모델 간 비슷할 것)
    p5    = quantile(eval, 0.05),  # best-case (min 문제이므로)
    p95   = quantile(eval, 0.95),  # worst-case → DRO가 나아야 함
    min   = minimum(eval),
    max   = maximum(eval),
  )

  # 모델 간 비교
  win_rate = mean(eval_dro .< eval_nom)   # DRO가 이기는 비율
```

**보고**: Mean + p5 + p95 + Min + Max + Win Rate.
- p95/Max에서 DRO < Nominal (minimization) → worst-case 방어 효과
- Win rate > 50% → 과반 realization에서 DRO 승

### 7.3 OOS Phase B: Asymmetric Dirichlet + OOS Mean

$\hat{q} = \text{uniform}$이 misspecified된 상황을 만들어, **OOS mean 자체가 의미 있게** 작동하도록 한다.

**원리**: Dirichlet parameter $\alpha$에 noise를 줘서 $\mathbb{E}[q_{\text{true}}] = \alpha/\Sigma\alpha \neq \text{uniform}$을 만든다. Leader의 $\hat{q}$가 틀린 세상에서 DRO가 nominal보다 평균적으로 나은지 측정.

**주의**: Noise를 **매번** 새로 뽑으면 (Option B) $\mathbb{E}[\alpha] = \beta \cdot \mathbf{1}$이 되어 다시 uniform 수렴. Noise를 outer loop에서 **한 번 고정** (Option A)해야 $\mathbb{E}[q_{\text{true}}] \neq \text{uniform}$이 보장된다.

```
for each β ∈ {0.1, 0.3, 0.5, 0.8}:
  ε = calibrate_epsilon(β)   # symmetric β로 calibrate

  for each model ∈ {Nominal, Single-DRO, Two-layer DRO}:
    x* = solve_leader(model, ε)

  for m = 1, ..., M=100:                            # outer: noise realization
    α = β .* (1.0 .+ 0.5 * randn(|S|))             # asymmetric
    α = max.(α, 0.01)                               # 양수 보장
    p_center = α / sum(α)                           # 이 세상의 true mean

    for r = 1, ..., R=100:                           # inner: Dir sampling
      p_true  ~ Dirichlet(α)
      q_tilde ~ Dirichlet(α)

      for each model:
        h* = solve_follower(x*_model, q_tilde)
        flows = [maxflow(x*_model, h*, ξ^k) for k in 1:|S|]
        eval[model, m, r] = dot(p_true, flows)

    # 이 세상의 OOS mean (α 고정 → p_center로 수렴 ≠ uniform)
    gap_two_vs_nom[m] = mean(eval[Two-layer, m, :]) - mean(eval[Nominal, m, :])

  # M개 세상에 걸친 gap 통계
  results = (
    gap_mean  = mean(gap_two_vs_nom),
    gap_p5    = quantile(gap_two_vs_nom, 0.05),
    gap_p95   = quantile(gap_two_vs_nom, 0.95),
    dro_wins  = mean(gap_two_vs_nom .< 0),   # gap < 0이면 DRO 승 (min 문제)
  )
```

**보고**: Gap mean + Gap interval (p5, p95) + DRO win rate.
- Gap mean < 0: 평균적으로 DRO가 나음
- DRO Win% > 50%: 과반의 "세상"에서 DRO 승

### 7.4 Computational 절약

f_share ≈ 0 (이전 실험에서 확인)인 경우, Phase B의 inner loop을 생략:

```
for m = 1, ..., M:
  α = generate_asymmetric_alpha(β, |S|)
  p_center = α / sum(α)

  for model in [Nominal, Single-DRO, Two-layer]:
    h* = solve_follower(x*_model, p_center)   # 한 번만
    flows = [maxflow(x*_model, h*, ξ^k) for k in 1:|S|]
    cost[model, m] = dot(p_center, flows)     # 직접 계산

  gap[m] = cost[Two-layer, m] - cost[Nominal, m]
```

이러면 M × 3 = 300번의 follower optimization만 필요.

### 7.5 Variance Decomposition (Phase A에서)

Law of total variance:

$$\text{Var}(Y) = \underbrace{\text{Var}_{\tilde{q}}[\mathbb{E}_{q^*}[Y \mid \tilde{q}]]}_{\text{follower belief effect}} + \underbrace{\mathbb{E}_{\tilde{q}}[\text{Var}_{q^*}[Y \mid \tilde{q}]]}_{\text{nature effect}}$$

- Follower share가 크면 → two-layer DRO 가치가 큼
- Follower share가 작으면 → single-layer DRO로 충분

---

## 8. Experiments

**실행 순서**: Experiment 2 (PO/PP calibration) → ε 결정 → Experiment 1, 3, 4, 5.

**Computational flow**: Experiment 2에서 $x^{\text{neut}}$, $x^{\text{rob}}$, $x^*_\varepsilon$ (Two-layer DRO)를 구하고 저장한다. Experiment 3, 4, 5에서는 이 solution들을 재활용하며, 추가로 Single-layer DRO의 $x^*_{\text{single}}$만 한 번 풀면 된다. **총 DRO solve 횟수**: Experiment 2의 bisection ~5회 + $x^{\text{neut}}$ 1회 + $x^{\text{rob}}$ 1회 + $x^*_{\text{single}}$ 1회 ≈ **~8회** (per network instance).

### 8.1 Experiment 1: ε Sweep (Symmetric, Phase A Metrics)

ε의 효과를 확인하는 표준 DRO 실험.

**설정**: β ∈ {0.3, 1.0}, ε/ε\* ∈ {0, 0.25, 0.5, 1.0, 1.5, 2.0}

각 (β, ε)에서 세 모델을 풀고 Phase A OOS (tail metrics) 보고.

**기대 결과**: ε가 너무 작으면 overfitting, 너무 크면 overconservatism. Win rate 기준 inverted-U shape 예상.

### 8.2 Experiment 2: PO/PP/NR/WR Calibration (★ 먼저 실행)

Rahimian et al. (2019) 방식의 analytical calibration. **이 실험이 다른 OOS 실험보다 먼저 수행되어야 한다** — ε의 적절한 range를 결정한 뒤 나머지 실험을 진행.

Combined bisection으로 [ε^S, ε^D] 안의 ε을 찾는다. 상세 절차와 pseudocode는 **Section 5.3.5**를 참조.

**실행**:
```
result = find_epsilon_range(network, scenarios, ε_start=0.5)
ε_recommended = result.ε_recommended
```

**출력**: ε_recommended + PO/PP/NR/WR at that point + 탐색 경로. 이후 실험 (8.3, 8.4, 8.5)에서 사용할 ε = ε_recommended.

**각 β의 ε_coverage와 비교**: ε_recommended vs calibrate_epsilon(β)를 나란히 보고.

**8.1과의 관계**: 8.1의 ε sweep에서 win rate가 가장 높은 ε과, 8.2의 ε_recommended가 일관된 range를 주는지 확인.

### 8.3 Experiment 3: Symmetric OOS (Phase A, fixed ε)

Experiment 2에서 저장된 solution을 재활용한다. **DRO를 다시 풀지 않는다.**

```
# Experiment 2에서 이미 구한 것들:
# result.x_star    → Two-layer DRO solution at ε_recommended
# x_neut           → Nominal solution (ε=0)
# x_rob            → Full robust solution (ε=1)
# (Single-layer만 별도로 풀어야 함: solve_leader(ε₁=ε_recommended, ε₂=0))

x_single = solve_leader(ε₁=ε_recommended, ε₂=0)  # 한 번만

for β in [0.1, 0.3, 0.5, 0.8]:
    oos_phase_a(x_neut, x_single, result.x_star, scenarios, β; M=100, L=100)
```

**보고**: Network × β × Model 테이블 (Mean, p5, p95, Min, Max, Win Rate, f_share).

### 8.4 Experiment 4: Asymmetric OOS (Phase B, fixed ε)

마찬가지로 Experiment 2에서 저장된 solution 재활용. DRO를 다시 풀지 않는다.

```
for β in [0.1, 0.3, 0.5, 0.8]:
    oos_phase_b(x_neut, x_single, result.x_star, scenarios, β; M=100, R=100)
```

**보고**: Network × β 테이블 (Gap Mean, Gap p5, Gap p95, DRO Win%).

### 8.5 Experiment 5: Information Asymmetry Scenarios

Phase A metrics를 사용하여 세 information scenario를 비교.

#### Scenario S (Symmetric Ignorance) — Baseline

| 분포 | 생성 방법 | 파라미터 |
|------|---------|---------|
| q_true | Dir(β · 1) | β = 0.3 |
| q̃ | Dir(β · 1), independent | β = 0.3 |

DRO 설정: ε₁ = ε₂ = ε\*(β=0.3)

#### Scenario L (Leader Advantage)

| 분포 | 생성 방법 | 파라미터 |
|------|---------|---------|
| q_true | Dir(β_H · 1) | β_H = 50 |
| q̃ | Dir(β_L · 1) | β_L = 0.3 |

ε₁: Dir(β_H=50)에서 calibrate (작음). ε₂: Dir(β_L=0.3)에서 calibrate (큼).

**기대**: Two-layer DRO의 가치 **가장 큼**. Follower belief effect 지배적.

#### Scenario F (Follower Advantage)

| 분포 | 생성 방법 | 파라미터 |
|------|---------|---------|
| q_true | Dir(β_L · 1) | β_L = 0.3 |
| q̃ | Dir(κ · q_true) | κ = 50 |

ε₁ = ε₂ = ε\*(β_L=0.3).

**Scenario F의 loop 구조**: q_true를 outer loop에서 고정, L = 1 (inner loop 불필요). Nature effect = 0.

**기대**: Two-layer DRO의 가치 **가장 작음**. 올바른 behavior.

#### Summary

| Scenario | q_true | q̃ | ε₁ | ε₂ | Two-layer 가치 |
|----------|--------|---|---|---|--------------|
| **S** | Dir(0.3) | Dir(0.3), indep. | ε\*(0.3) | ε\*(0.3) | baseline |
| **L** | Dir(50) | Dir(0.3) | ε\*(50) | ε\*(0.3) | **가장 큼** |
| **F** | Dir(0.3) | Dir(50·q_true) | ε\*(0.3) | ε\*(0.3) | **가장 작음** |

---

## 9. Reporting

### 9.1 Experiment 1 Output

**Plot**: ε/ε\* (x-axis) vs win rate & p95 (y-axis), 세 모델.

### 9.2 Experiment 2 Output

**Table**: Bisection 탐색 경로 (iter, ε, PO−PP, NR−WR, region).

**Final output**: ε_recommended + PO, PP, NR, WR at that point.

**Comparison table**: ε_recommended vs ε_coverage(β) for each β.

| β | ε_coverage (95%) | ε_recommended (PO/PP) | Ratio |
|---|---|---|---|
| 0.1 | ... | ... | ... |
| 0.3 | ... | ... | ... |
| 0.8 | ... | ... | ... |

### 9.3 Experiment 3 Output (Phase A)

**Table**: Network × β × Model × {Mean, p5, p95, Max, Win Rate, f_share}

### 9.4 Experiment 4 Output (Phase B)

**Table**: Network × β × {Gap Mean, Gap p5, Gap p95, DRO Win%}

**Boxplot**: β별 M개의 gap 분포.

### 9.5 Experiment 5 Output

**Table**: Scenario × Model × {Mean, p95, Win Rate, f_share}

**Boxplot** (à la Sadana Figure 3): per-scenario eval 분포.

### 9.6 핵심 Story

Phase A: "DRO는 평균적으로 nominal과 동등하지만, worst-case realization에서 보호한다 (보험 효과)."

Phase B: "$\hat{q}$가 misspecified된 상황에서 DRO는 평균적으로도 nominal보다 나은 performance를 보인다."

Experiment 5: "Follower belief uncertainty가 실제로 존재할 때만 two-layer robustness가 필요하다."

---

## 10. Preliminary Validation: DGP 시각화

본 실험 전에 각 scenario의 DGP가 의도한 information structure를 실제로 만들어내는지 확인.

```julia
function visualize_dgp(; S=10, β=0.3, β_H=50, β_L=0.3, κ=50, n_draws=1000)
    q_hat = fill(1.0/S, S)
    for scenario in [:S, :L, :F]
        # ... 세 거리의 histogram + calibrated ε 표시
        # d(q̂, q_true), d(q̂, q̃), d(q̃, q_true)
    end
end
```

검증 기준:
1. Scenario S: 세 거리 비슷
2. Scenario L: d(q̂, q_true) ≪ d(q̂, q̃), d(q̃, q_true) 큼
3. Scenario F: d(q̃, q_true) ≪ d(q̂, q_true)

---

## 11. Parameter Summary

| Parameter | Phase A | Phase B |
|-----------|---------|---------|
| β | {0.1, 0.3, 0.5, 0.8} | {0.1, 0.3, 0.5, 0.8} |
| ε calibration | Dir(β·1), 95th pctl | 동일 |
| q_true sampling | Dir(β·1) — symmetric | Dir(α) — asymmetric |
| noise_scale | — | 0.5 |
| OOS samples | M=100 × L=100 = 10,000 | M=100 outer × R=100 inner |
| Primary metric | **p95, max, win rate** | **OOS mean gap** |
| Secondary metric | mean (참고) | gap interval, win rate |

---

## 12. 주의사항

1. **Phase A의 mean은 모델 간 차이를 보여주지 못한다.** Risk-neutral DRO의 구조적 성질 (linear collapse). Ben-Tal et al. (2013)도 동일한 현상을 보고하고 range로 해결.

2. **Phase B에서 noise를 매번 새로 뽑으면 안 된다.** Symmetric i.i.d. noise면 $\mathbb{E}[\alpha] = \beta \cdot \mathbf{1}$ → uniform 수렴. Noise를 outer loop에서 한 번 고정 (Option A).

3. **ε은 symmetric β로 calibrate.** Leader는 noise를 모른다. Phase B의 noise는 "leader가 모르는 진짜 세상의 misspecification".

4. **Instance 수 = 10**: Scenario set의 운에 의한 변동을 평균화.

5. **f_share ≈ 0이면 Phase B 절약 가능**: Inner loop 생략, p_center로 직접 계산.

---

## 13. Implementation Checklist

**실행 순서가 중요하다**: Experiment 2 (PO/PP) → ε 결정 → 나머지 실험.

- [ ] `visualize_dgp()` — DGP 시각화 및 파라미터 검증
- [ ] `calibrate_epsilon(S, β; n_cal=10000, coverage=0.95)` — L1 distance 기반 coverage calibration
- [ ] `evaluate_f(x, ε)` — 임의의 x를 임의의 ε 잣대로 평가 (Section 5.3.2)
  - follower worst-case h* 계산 (ε₂ = ε)
  - scenario별 flow 계산
  - TV ball 위 inner worst-case LP
- [ ] `tv_worst_case_expectation(flows, q_hat, ε)` — LP로 TV ball 위 worst-case 가중합
- [ ] `find_epsilon_range(network, scenarios; ε_start=0.5)` — Section 5.3.5의 combined bisection
  - PO−PP 체크 (싸다: DRO 안 풂, evaluate_f만)
  - NR−WR 체크 (비싸다: DRO 풀어야)
  - [ε^S, ε^D] 안에 들어가면 stop → ε_recommended 반환
- [ ] `oos_phase_a(x_star, scenarios, β; M=100, L=100)` — symmetric Dirichlet + tail metrics
- [ ] `oos_phase_b(x_star, scenarios, β; M=100, R=100, noise_scale=0.5)` — asymmetric Dirichlet + OOS mean
- [ ] **Experiment 2** (★ 먼저): `find_epsilon_range()` → ε_recommended 결정
- [ ] **Experiment 1**: ε sweep with Phase A metrics (β ∈ {0.3, 1.0}) → ε^S, ε^D와 일관성 확인
- [ ] **Experiment 3**: Phase A OOS, fixed ε (β ∈ {0.1, 0.3, 0.5, 0.8})
- [ ] **Experiment 4**: Phase B OOS, fixed ε (β ∈ {0.1, 0.3, 0.5, 0.8})
- [ ] **Experiment 5**: Information asymmetry (S, L, F)
- [ ] Reporting: tables, boxplots, PO/PP curves

---

## References (for OOS methodology)

- Ben-Tal, A., den Hertog, D., De Waegenaere, A., Melenberg, B., & Rennen, G. (2013). Robust solutions of optimization problems affected by uncertain probabilities. *Management Science*, 59(2), 341–357.
- Rahimian, H., Bayraksan, G., & Homem-de-Mello, T. (2019). Controlling risk and demand ambiguity in newsvendor models. *European Journal of Operational Research*, 279(3), 854–868.
- Park, J., & Bayraksan, G. (2022). A multistage distributionally robust optimization approach to water allocation under climate uncertainty. *European Journal of Operational Research*, 306(2), 849–871.
- Sadana, U., & Delage, E. (2022). The value of randomized strategies in DRMFNI problems. *INFORMS Journal on Computing*, 35(1), 216–232.
