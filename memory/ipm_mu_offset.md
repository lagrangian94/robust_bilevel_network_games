# IPM Analytic Center Artifact in Conic Benders Decomposition

## 배경: Inner Benders Decomposition 구조

Inner master problem (IMP)는 α를 결정하고, inner subproblem (ISP)이 V(α)를 계산한다.
ISP의 optimal value V(α)의 **subgradient μ**를 이용해 IMP에 cut을 추가:

```
t ≤ intercept + α' · μ      (Benders optimality cut)
```

여기서 `intercept + α*' · μ = V(α*)` (현재 α*에서 tight).

## 두 가지 ISP 구현

### Dual ISP (기존)
- **Max** 문제. 변수: (U, β, Z, P, M, Γ)
- Coupling constraint: `βhat2[k] ≤ α[k]`
- μ 추출: `shadow_price(coupling_constraint)` → LP simplex basis에서 정확한 marginal value

### Primal ISP (dual of dual ISP)
- **Min** 문제. 변수: (η, μ, Φ, Ψ, Π, ϑ, Λ, M)
- **Objective**: `min (1/S)·η + Σ_k α_k · μ_k`
- μ 추출: `value(μ)` → solver가 반환하는 optimal point에서 직접 읽음

## 핵심 관찰

Primal ISP에서 `α_k = 0`이면 **μ_k의 objective coefficient = 0**.
즉 μ_k는 제약조건만 만족하면 어떤 값이든 동일한 objective를 준다.

Primal ISP의 μ_k 관련 제약조건:
```
(1) μ_k ≥ 0                              (변수 하한)
(2) Λ₂·r - Φ₀ + μ ≥ 0                   (βhat2 ≥ 0의 dual constraint)
```

α_k = 0이면 objective에서 μ_k가 사라지므로, μ_k는 `max(0, Φ₀[k] - (Λ₂·r)[k])` 이상이기만 하면 된다.

## IPM (Interior Point Method)의 Analytic Center

Mosek은 IPM 기반 solver다. IPM은 feasible region의 **analytic center**를 찾는다.

Min 문제에서 barrier subproblem:
```
min  f(x) - τ · Σ log(slack_i)
```

Zero-cost 변수 μ_k에 대해서는 f(x) 기여가 0이므로,
barrier function `-τ·[log(μ_k) + log(slack_of_constraint_2)]`이 μ_k의 위치를 결정한다.

**단순한 경우** (μ_k ≥ 0, μ_k ≥ L_k 두 제약만 있을 때):
```
min -τ·log(μ_k) - τ·log(μ_k - L_k)
→ μ_k = L_k/2 + √(L_k²/4 + ...)   (analytic center)
```

하지만 실제로는 μ_k가 Φ, Ψ, Λ, M (PSD cone) 등과 **간접적으로 연결**되어 있다.
연결 경로:

```
ε → [ε² constraint: tr(M₁₁) ≤ M₂₂·ε²]
  → ϑ (이 constraint의 dual variable)
  → [M₁₁ = ϑ·I - D'·(Φ_L - v·Ψ_L)]
  → Φ_L, Φ₀ (SDP linking을 통해 연결)
  → [constraint (2): Λ₂·r - Φ₀ + μ ≥ 0]
  → μ_k
```

이 체인에서 **ε가 feasible region의 scale을 결정**한다:
- ε가 크면 → `tr(M₁₁) ≤ M₂₂·ε²`이 더 loose → ϑ의 feasible range 확대
- ϑ 확대 → M₁₁의 diagonal이 커질 수 있음 → Φ의 range 확대
- Φ range 확대 → constraint (2)에서 μ_k의 lower bound이 달라짐
- IPM의 analytic center가 이 scale에 비례하여 μ_k를 **ε만큼 offset**

## 실험적 검증

| ε 값 | μ_primal - μ_dual (uniform offset) | Inner iterations (primal) | Inner iterations (dual) |
|------|-----------------------------------|--------------------------|------------------------|
| 0.5  | +0.5                              | 23                       | 2                      |
| 0.3  | +0.3                              | 12                       | 2                      |

Offset이 정확히 ε에 비례함을 확인.

## 왜 Cut Quality가 나빠지는가

Inner Benders cut: `t ≤ intercept + α' · μ`

**정확한 μ** (dual ISP): sparse, 대부분 0, 필요한 arc만 nonzero
```
μ_dual = [0, 0, 0, ..., 1.0, ..., 0.384, ..., 0, 0]   (11/47 nonzero)
```

**IPM의 μ** (primal ISP): 모든 component에 +ε offset
```
μ_primal = [0, 0.5, 0.5, ..., 1.5, ..., 0.884, ..., 0.5, 0.5]   (46/47 nonzero)
```

IMP에서 다른 α를 시도할 때:
- 정확한 cut: `t ≤ intercept + Σ α_k · μ_k^true` → α가 바뀌면 tight하게 반응
- IPM cut: `t ≤ intercept + Σ α_k · (μ_k^true + ε)` → 모든 방향에 +ε bias
  → cut RHS가 `ε · Σα_k`만큼 높아짐 → **looser upper bound** → IMP가 덜 constrained

## 보정 방법

```julia
# 1. Primal ISP를 평소대로 풀고, objective_value(model)에서 obj_val 추출
obj_val = intercept_raw + α' · μ_raw

# 2. μ에서 ε offset 제거
μ_corrected = max.(μ_raw .- ε, 0.0)

# 3. Intercept를 재계산하여 cut tightness 유지
#    핵심: obj_val은 변경하지 않음 (ISP의 진짜 최적값)
#    intercept_new + α' · μ_corrected = obj_val 이 되도록
intercept_corrected = obj_val - α' · μ_corrected
```

**왜 이게 valid한가:**
- V(α*)에서의 subgradient는 optimal dual variable의 하나
- α_k = 0인 component에서 μ_k는 free (어떤 optimal 값이든 valid subgradient)
- `max(0, μ - ε)`는 feasible한 다른 optimal point → valid subgradient
- α_k > 0인 component에서: μ_k - ε도 여전히 올바른 값
  (IPM이 이 component에도 +ε offset을 추가했으므로)

## 보정 후 결과 (5×5 grid, S=1)

| 항목 | Dual ISP | Primal ISP (보정 전) | Primal ISP (보정 후) |
|------|---------|-------------------|-------------------|
| Inner iterations | 2 | 23 | 2 |
| 수렴 값 | 17.7072 | 17.7072 | 17.7072 |
| μ nonzeros | 11 | 46 | 9 |
| μ diff vs dual | - | ~0.5 uniform | ~1e-5 |

## 의의

1. **Conic Benders에서 IPM solver의 subgradient bias**: LP Benders에서는 simplex가 vertex solution을
   주므로 이 문제가 없음. Conic (SDP/SOCP) Benders에서만 발생.
2. **문제 구조에 의존하는 bias**: offset이 ε (uncertainty set radius)에 정확히 비례.
   다른 conic Benders 문제에서도 유사한 bias가 있을 수 있음.
3. **간단한 보정**: `max(0, μ - ε)` + intercept 재계산. 구현 1줄.
