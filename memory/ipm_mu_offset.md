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

## Offset 정밀 분석 (test_offset_by_alpha.jl)

α density별 offset 측정 결과 (5×5 grid, S=1, ε=0.5):

| α_k 값 | μ offset (primal - dual) | 비고 |
|---------|-------------------------|------|
| α_k = 0 | **정확히 +ε (+0.5)** | zero-cost → IPM analytic center |
| α_k > 0 | **≈ 0 (1e-5)** | nonzero cost → IPM이 정확한 값 반환 |

이 패턴은 α의 density(1/47 ~ 47/47)에 관계없이 일관적.

## 보정 방법

### V1 (blanket subtraction) — 부분적으로만 작동
```julia
μ_corrected = max.(μ_raw .- ε, 0.0)
```
문제: α_k > 0인 component에서 offset이 0인데도 ε를 빼버림 → μ가 과도하게 작아짐.
첫 inner loop (α 전부 0)에서는 완벽하지만, 이후 dense α에서 성능 저하.

### V2 (conditional subtraction) — 올바른 보정 ✓
```julia
ε = uncertainty_set[:epsilon]
for k in eachindex(subgradient)
    if α_sol[k] < 1e-8  # zero-cost component만 보정
        subgradient[k] = max(subgradient[k] - ε, 0.0)
    end
end
# Intercept 재계산: cut tightness 유지
intercept = obj_val - α_sol' * subgradient
```

**핵심**: `obj_val = objective_value(model)` (ISP의 진짜 최적값)은 변경하지 않음.

**왜 이게 valid한가:**
- V(α*)에서의 subgradient는 optimal dual variable의 하나
- α_k = 0인 component에서: μ_k는 free (어떤 optimal 값이든 valid subgradient).
  `max(0, μ - ε)`는 feasible한 다른 optimal point → valid subgradient
- α_k > 0인 component에서: offset ≈ 0이므로 보정 불필요, 그대로 사용

## 보정 후 결과 (5×5 grid, S=1)

### 단일 inner loop (test_inner_cut_quality.jl)
| 항목 | Dual ISP | Primal (보정 전) | Primal (V1) | Primal (V2) |
|------|---------|----------------|-------------|-------------|
| Inner iterations | 2 | 23 | 2 | 2 |
| 수렴 값 | 17.7072 | 17.7072 | 17.7072 | 17.7072 |

### Full Benders (test_hybrid_benders.jl)
| 항목 | Original (dual) | Hybrid (V1 blanket) | Hybrid (V2 conditional) |
|------|----------------|--------------------|-----------------------|
| Outer iters | 60 | 58 | 62 |
| Inner iters | 242 | 371 | **273** |
| Time | 61.9s | 78.6s | **64.2s** |

V2가 original과 거의 동등한 성능 (inner 273 vs 242, time 64s vs 62s).

### Full Primal
| 항목 | V1 blanket | V2 conditional |
|------|-----------|---------------|
| Outer | 93 | 93 |
| Inner | 563 | **432** |
| Time | 100.7s | **79.6s** |

Full Primal의 outer iteration이 많은 것은 outer cut quality 차이 (별도 이슈).

## 의의

1. **Conic Benders에서 IPM solver의 subgradient bias**: LP Benders에서는 simplex가 vertex solution을
   주므로 이 문제가 없음. Conic (SDP/SOCP) Benders에서만 발생.
2. **문제 구조에 의존하는 bias**: offset이 ε (uncertainty set radius)에 정확히 비례.
   α_k = 0인 component에서만 발생 (zero-cost variable → analytic center).
3. **Conditional 보정**: α_k ≈ 0인 component에서만 `max(0, μ - ε)`. 간단하고 정확.
