# Joint Scenario Generation: Capacity × Interdiction Success

## 1. Modeling Stance

Finite-support φ-divergence (TV) DRO를 사용한다는 사실로부터 다음 modeling primitive를 채택한다.

**Primitive 1 (support).** Uncertainty는 modeler가 specify한 finite list
$$
\Xi = \{(u^k, v^k) : k = 1, \ldots, |K|\}
$$
로 represent된다. Support 자체는 ambiguity ball에 의해 변경되지 않으며, ambiguity는 *list 위의 probability assignment*에만 적용된다.

**Primitive 2 (construction protocol).** List $\Xi$를 만드는 *protocol*은 modeling input이지 random parameter가 아니다. 구체적으로:
- Capacity vector $u^k$는 factor model로 generate ([Sadana & Delage, 2022] 따름)
- Interdiction success vector $v^k$는 i.i.d. Bernoulli로 generate ([Lei et al., 2018] 따름)
- Bernoulli rate $p$, factor matrix $F$, 그리고 mean vector $\mu$는 protocol parameter이며 모델의 random parameter가 아니다.

이 stance에서 "$p=0.75$를 왜 robustify하지 않는가?"라는 질문은 다음과 같이 답변된다:

> $p$는 list construction parameter이다. 모델 안의 random object는 'list의 어느 element가 realize되는가'이고 이 위에 ambiguity ball을 둔다. $p$의 변화는 *list 자체를 다시 construct*하는 것에 해당하며, 이는 별도의 sensitivity analysis로 다룬다.

## 2. Why Joint Generation, and Why i.i.d. per Scenario

### 2.1 Motivation for adding $v$ randomness

Capacity-only setting에서는 일부 interdiction plan이 다른 plan에 대해 *deterministic하게* dominate하는 경우가 발생하며, 이 경우 ambiguity ball을 어떻게 설정하더라도 dominant plan이 항상 winner가 된다. 즉 DRO의 effect가 numerically 관찰되지 않는다. Interdiction success $v$를 random으로 두면 dominance가 stochastic해져 ambiguity ball이 의미 있게 작동한다.

### 2.2 Independence assumption

$u \perp v$를 가정한다. 정당화: capacity는 traffic/infrastructure 상태를 반영하는 반면 interdiction success는 enforcement technology에 의존하는 별개의 mechanism이다. 이 가정은 reformulation을 단순하게 유지한다 — bilinear term이 추가로 발생하지 않는다 ($v_{ij}^k$가 parameter이므로 $u_{ij}^k(1-v_{ij}^k)$가 새로운 effective capacity coefficient).

### 2.3 i.i.d. Bernoulli per scenario

각 scenario $k$에서 $v^k_{ij}$를 모든 interdictable arc에 대해 *독립적으로* Bernoulli sample한다.

### 2.4 Sample size의 한계와 ambiguity ball의 역할

$|K|=20$ scenarios로 $|A|$차원 binary $v$ 공간 전체를 explicitly enumerate할 수 없다. 이 사실에 대한 본 framework의 stance를 명확히 한다.

본 framework는 두 종류의 uncertainty를 *분리*한다:

1. **List 위 probability assignment $q^*$ 에 대한 epistemic uncertainty.** Leader는 list $\Xi$를 specify한 후에도, 각 scenario가 어떤 frequency로 realize되는지 (즉 $q^*$) 를 알지 못한다. Reference $\hat{q}$는 leader의 best guess이며, 실제 $q^*$가 $\hat{q}$와 다를 가능성이 *ambiguity ball이 직접 hedge하는 대상*이다.

2. **List $\Xi$ 자체의 선택에 대한 uncertainty.** "다른 list로 protocol을 다시 돌렸으면 어땠을까"에 해당하는 우려. 이는 ambiguity ball이 hedge하지 *않는다* — Primitive 1에 의해 list는 modeling input이기 때문이다. 이 종류의 uncertainty는 별도의 *sensitivity analysis* (seed, $p$ 변화) 로 점검한다.

이 분리에 따라 "$|K|=20$이 $|A|$차원을 cover하지 못한다"는 우려는 (2)에 속하며, ambiguity ball의 작동 영역 밖에 있다. Ambiguity ball은 (1) — list가 fix된 상태에서 그 위 probability가 어디로 갈지에 대한 epistemic uncertainty — 만을 hedge한다.

이 stance는 finite-support DRO 문헌의 표준 framing이며 (Sadana & Delage 2022 등), modeling primitive로 받아들여진다. 한계의 *영향*은 §5의 sensitivity analysis로 평가한다.

## 3. Generation Procedure (수도코드)

### 3.1 Inputs

| Symbol | Meaning | Default |
|---|---|---|
| `num_arcs` | $\|E\| = \|A\| + 1$ (regular arcs + dummy) | from network |
| `interdictable_arcs::Vector{Bool}` | length `num_arcs`, dummy/source-incident/sink-incident는 `false` | from network |
| `num_scenarios` | $\|K\|$ | 20 |
| `k_factors` | factor 수 | 2 (Sadana 표준) |
| `p_bernoulli` | Bernoulli rate | 0.75 (Lei et al. 2018) |
| `seed` | random seed | optional |

### 3.2 Pseudocode

```
function generate_joint_scenarios(num_arcs, interdictable_arcs, num_scenarios;
                                   k_factors=2, p_bernoulli=0.75, seed=nothing)

    if seed !== nothing then set_random_seed(seed) end

    num_regular_arcs ← num_arcs - 1                  # exclude dummy arc
    num_interdictable ← count(interdictable_arcs)

    # ---- Protocol parameters (drawn once per instance) ----
    # F ∈ R_+^{num_regular_arcs × k_factors}, entries ~ Uniform(0,1)
    F ← rand(num_regular_arcs, k_factors)
    # μ ∈ R_+^{k_factors}, entries ~ Uniform(0,1)
    μ ← rand(k_factors)

    capacity_scenarios ← zeros(num_arcs, num_scenarios)
    interdiction_success_scenarios ← zeros(num_arcs, num_scenarios)

    for k in 1:num_scenarios do
        # (i) Capacity via factor model (Sadana 2022)
        # ξ^k_i ~ Exp(μ_i) independently
        ξ ← [sample_exponential(μ[i]) for i in 1:k_factors]
        capacity_scenarios[1:num_regular_arcs, k] ← F * ξ
        # Dummy arc capacity = sum of regular capacities (numerically stable upper bound)
        capacity_scenarios[end, k] ← sum(capacity_scenarios[1:num_regular_arcs, k])

        # (ii) Interdiction success via i.i.d. Bernoulli (Lei et al. 2018)
        # Non-interdictable arcs: irrelevant (v multiplied by x; x=0 for those)
        #                          → set to 0 by convention.
        for e in 1:num_arcs do
            if interdictable_arcs[e] then
                interdiction_success_scenarios[e, k] ← sample_bernoulli(p_bernoulli)
            else
                interdiction_success_scenarios[e, k] ← 0     # convention; never used
            end
        end
    end

    return (capacity_scenarios,                      # u^k for all k
            interdiction_success_scenarios,          # v^k for all k
            F, μ)                                    # protocol parameters (logging)
end
```

### 3.3 Notes for implementer

- **Existing function `generate_capacity_scenarios_factor_model`** in `network_generator.jl`은 capacity 부분을 그대로 수행하고 있다. 새 함수는 그 위에 $v$ generation을 추가한 형태이거나, 내부에서 기존 함수를 호출하는 wrapper로 구현 가능.
- **`interdictable_arcs`** field는 이미 `GridNetworkData` struct에 존재 (`network_generator.jl` 참조).
- **Dummy arc index**: 항상 마지막 arc (`network.arcs[end] == ("t", "s")`).
- **`v^k`의 0 값 (non-interdictable arc) 의 정당화**: interdiction constraint
  $$y_{ij} \le u_{ij}^k(1 - v_{ij}^k x_{ij}) + h_{ij}$$
  에서 non-interdictable arc는 $x_{ij}=0$으로 fix되므로 $v_{ij}^k$ 값과 무관. Convention상 0으로 두는 것이 안전.
- **Reproducibility**: protocol parameters $(F, \mu, p)$와 seed를 logging해야 sensitivity analysis가 가능.

## 4. Calling Convention 변경 사항

기존 코드에서 capacity만 사용하던 부분을 다음과 같이 확장:

| 기존 | 수정 후 |
|---|---|
| `cap_scenarios, F = generate_capacity_scenarios_factor_model(...)` | `cap_scenarios, v_scenarios, F, μ = generate_joint_scenarios(...)` |
| Constraint: `y ≤ diag(cap[:,k]) * (1 .- x) + h` | Constraint: `y ≤ diag(cap[:,k]) * (1 .- v_scenarios[:,k] .* x) + h` |

위 constraint 변경에서 element-wise product `v_scenarios[:,k] .* x`의 의미는: "scenario $k$에서 interdiction이 성공한 arc에 한해 $x$가 effect를 가짐". 기존 코드 ($v \equiv 1$) 와 비교하면 $v_{ij}^k = 0$인 arc에서는 $x_{ij}=1$이어도 capacity reduction이 발생하지 않는다 (단속 실패).

## 5. Sensitivity Analysis (실험 시 보고할 항목)

논문/디펜스에서 modeling primitive를 정당화하기 위해 다음 sensitivity를 보고한다.

| Axis | Range | 목적 |
|---|---|---|
| `seed` | 3~5개의 다른 seed | 동일 protocol에서 list가 재생성될 때 결과가 일관적인지 |
| `p_bernoulli` | $\{0.5, 0.75, 0.9\}$ | Bernoulli rate에 sensitive한 artifact가 아닌지 |
| `num_scenarios` | $\{10, 20, 40\}$ (가능 instance에서) | $\|K\|$ 증가 시 결과의 질적 양상이 유지되는지 |

각 sensitivity에 대해 list가 재생성됨에 유의: $p$를 바꿀 때는 list 자체를 다시 construct, 결과 비교는 *여러 list에 걸친 robustness check*로 해석.

## 6. Out-of-Sample Evaluation (변경 없음)

`true_dro.tex`의 nested sampling protocol은 $v$ 추가에 따라 변경할 필요 없음. 이유:
- Out-of-sample은 *probability assignment* 위에서 정의됨 ($q^*, \tilde{q} \sim \mathrm{Dirichlet}(\beta\mathbf{1}_{|K|})$)
- Support $\Xi = \{(u^k, v^k)\}_{k=1}^{|K|}$는 in-sample과 out-of-sample 모두 동일 (Primitive 1)
- $v^k$가 추가되어도 support의 atom 수 $|K|$는 변하지 않음

따라서 기존 calibration procedure와 evaluation loop는 그대로 유지된다.

## 7. Limitations (논문에 명시)

1. **Support primitive의 한계.** $|K|=20$은 $|A|$차원 binary $v$ 공간의 작은 부분만 represent. 결과는 *constructed scenario list에 conditional*하게 해석되어야 함. Underlying full distribution에 대한 statement는 making하지 않음.

2. **$u \perp v$ 가정.** 현실에서 capacity와 interdiction success가 약하게 correlated될 수 있는 상황 (예: 악천후 시 capacity 감소 + 단속 difficulty 증가) 은 다루지 않음. Future work.

3. **Marginal independence 비강제.** Ambiguity ball 안의 worst-case $q$는 $u$와 $v$의 marginal independence를 violate할 수 있음. 이는 Sadana-Delage 2022와 동일한 stance.

## References

- Sadana, U., & Delage, E. (2022). The Value of Randomized Strategies in Distributionally Robust Risk-Averse Network Interdiction Problems. *INFORMS Journal on Computing*, 35(1), 216–232.
- Lei, X., Shen, S., & Song, Y. (2018). Stochastic maximum flow interdiction problems under heterogeneous risk preferences. *Computers & Operations Research*, 90, 97–109.
