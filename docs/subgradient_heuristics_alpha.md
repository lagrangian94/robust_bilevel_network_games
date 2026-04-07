# H2: NBD Inner Loop에서 IMP를 Subgradient로 대체

## 1. 맥락: NBD Inner Loop의 비용 구조

### 1.1 현재 NBD: Inner Loop가 비싼 이유

```
Outer iteration k:
  OMP → χ̄ₖ

  ┌─ Inner loop (V(χ̄ₖ) 계산) ─────────────────────────────────────┐
  │  IMP → α₁                                                       │
  │  ISP(α₁; χ̄ₖ) 풀기 (2S회) → inner cut to IMP                   │
  │  IMP → α₂                                                       │
  │  ISP(α₂; χ̄ₖ) 풀기 (2S회) → inner cut to IMP                   │
  │  IMP → α₃                                                       │
  │  ISP(α₃; χ̄ₖ) 풀기 (2S회) → inner cut to IMP                   │
  │  ...                                                             │
  │  IMP → α* (수렴)                                                 │
  │  ISP(α*; χ̄ₖ) 풀기 (2S회) → inner cut (gap ≤ ε, 종료)          │
  └──────────────────────────────────────────────────────────────────┘
  ISP 총 호출: 2S × inner_iter (≈ 4-5) = 8-10S회

  ISP(α*; χ̄ₖ)의 dual variables로 outer cut 구성 → OMP에 추가
```

**Inner loop가 하는 일**: 고정된 χ̄ₖ에서 V(χ̄ₖ) = max_α f(α; χ̄ₖ)를 정확히 계산.
**비용**: IMP ↔ ISP를 4-5번 왕복. 매 왕복마다 ISP를 2S회 풀어야 함.
**절약 기회**: 이 왕복 횟수를 줄이는 것.

### 1.2 Inexact IMP: 왕복을 줄이되, 매번 리셋

```
Outer iteration k:
  OMP → χ̄ₖ
  IMP → α₁ → ISP(α₁) → inner cut → IMP → α₂ → ISP(α₂) → STOP (2회에서 끊음)
  ISP 총: 2S × 2 = 4S회. α₂는 suboptimal.

Outer iteration k+1:
  OMP → χ̄ₖ₊₁
  (이전 inner cut 삭제, IMP 처음부터)
  IMP → α₁' → ISP(α₁') → inner cut → IMP → α₂' → ISP(α₂') → STOP
  ISP 총: 4S회. 또 suboptimal. 이전 α₂ 정보는 버림.
```

**문제**: 매 outer iteration마다 α 탐색을 리셋. 이전에 찾은 좋은 α를 버린다.
4S회 써서 얻은 α가 여전히 나쁘고, 다음에 또 4S회 써서 비슷하게 나쁜 α를 얻음.

### 1.3 H2의 핵심 아이디어: 왕복을 1회로 줄이되, α를 across outer iters로 누적

```
Outer iteration k:
  OMP → χ̄ₖ
  ISP(αₖ; χ̄ₖ) 풀기 (2S회)          ← 1 라운드만
    → outer cut (Û, Ũ, β̂, ... 에서 구성)
    → μₖ = shadow_price(coupling)   ← 부산물, 추가 비용 없음
  αₖ₊₁ = Proj_Δ(αₖ + γ·μₖ)         ← simplex projection, 무시할 비용

Outer iteration k+1:
  OMP → χ̄ₖ₊₁
  ISP(αₖ₊₁; χ̄ₖ₊₁) 풀기 (2S회)     ← αₖ₊₁은 이전 μ로 개선된 것
    → outer cut
    → μₖ₊₁                          ← 새로운 supergradient
  αₖ₊₂ = Proj_Δ(αₖ₊₁ + γ·μₖ₊₁)

Outer iteration k+2:
  OMP → χ̄ₖ₊₂
  ISP(αₖ₊₂; χ̄ₖ₊₂) 풀기 (2S회)     ← αₖ₊₂는 2번 개선된 것
    → outer cut
    → μₖ₊₂
  αₖ₊₃ = Proj_Δ(αₖ₊₂ + γ·μₖ₊₂)
```

**μₖ를 저장해서 재사용하는 것이 아님. 매 outer iteration마다 ISP를 새로 풀어서 새 μ를 얻음.**

**절약하는 것**: inner loop의 왕복 (4-5 라운드 → 1 라운드).
**보존하는 것**: α 자체. 이전 outer iteration의 μ 정보가 α에 이미 반영되어 있음.

---

## 2. 비용 비교: 뭘 절약하는가

### 2.1 매 outer iteration의 ISP 호출

| 전략 | Inner loop 구조 | ISP calls / outer iter |
|------|----------------|------------------------|
| Exact IMP | IMP↔ISP 4-5 왕복 | 2S × 4.5 ≈ **9S** |
| Inexact IMP (2회) | IMP↔ISP 2 왕복 | 2S × 2 = **4S** |
| **H2** | **ISP 1 라운드 (왕복 없음)** | **2S** |

### 2.2 Exact IMP와의 총 비용 비교

5×5 grid, S=1 기준:

```
Exact IMP:
  outer 60 iters × 9 ISP/iter = 540 ISP calls

H2:
  outer ? iters × 2 ISP/iter = ? ISP calls
  → outer가 270 미만이면 순이득 (4.5배 여유)
```

Cut quality가 떨어져서 outer iteration이 늘어나지만,
ISP 1회 = 1.5초(5×5)이므로 ISP 호출 총수만 줄이면 시간 절감.

### 2.3 ISP를 줄이는 게 왜 중요한가

```
총 시간 ≈ 총 ISP 호출 × ISP 1회 시간
  ISP 1회 = ~1.5초 (5×5, PSD 51×51)
  전체의 ~85%가 ISP 시간

  IMP (Gurobi LP) = 무시할 수준 (<1%)
  OMP (Gurobi MIP) = 무시할 수준 (<1%)
  Julia overhead = ~10-15%
```

ISP 호출을 N회 줄이면 ≈ 1.5N초 절약. 이것이 지배적.

---

## 3. 수학적 기초

### 3.1 f(α; χ)의 concavity

$$V(\chi) = \max_{\alpha \in \Delta} f(\alpha;\chi), \quad f(\alpha;\chi) = \sum_s [Z_1^{L,s}(\alpha;\chi) + Z_1^{F,s}(\alpha;\chi)]$$

각 $Z_1^{L,s}(\alpha;\chi) = \min_p \{(1/S)\eta + \sum_k \alpha_k \mu_k + \cdots\}$:
affine 함수들의 infimum → **concave** in α.

(vertex_optimality_investigation.md에서 실험적으로도 확인: interior optimal is generic.)

### 3.2 Supergradient

$f$가 concave이므로 supergradient $g$ at $\tilde\alpha$:
$$f(\alpha) \leq f(\tilde\alpha) + g^\top(\alpha - \tilde\alpha), \quad \forall \alpha$$

Envelope theorem: $g_k = \sum_s (\hat\mu_k^{*s} + \tilde\mu_k^{*s})$

이것은 dual ISP의 coupling constraint shadow price → ISP를 풀면 공짜로 나옴.

### 3.3 Outer cut validity (α 무관)

임의의 feasible $\tilde\alpha \in \Delta$에 대해 $f(\tilde\alpha;\chi) \leq V(\chi)$, ∀χ.
따라서 suboptimal α에서도 outer cut은 항상 **valid**. Quality만 다를 뿐.

---

## 4. H2 알고리즘: 정밀 Step-by-Step

### 4.0 초기화

```
α₁ ← w/|A| · 1   (uniform on simplex)
k ← 0
```

### 4.1 매 Outer Iteration k

```
k ← k + 1

──── Step 1: OMP ────
Solve OMP → χ̄ₖ = (x̄, h̄, λ̄, ψ̄0)
LBₖ ← OMP objective value

──── Step 2: ISP 풀기 (2S회, 이번 iter의 유일한 ISP 호출) ────
For s = 1, ..., S:
  dual ISP에 parameter 설정:
    objective coefficients ← (x̄ₖ, h̄ₖ, λ̄ₖ, ψ̄0ₖ)
    coupling constraint RHS ← αₖ
  Solve dual ISP_leader(s):
    → obj_Lˢ
    → variable values: Ûˢ, β̂ˢ, Ẑˢ, P̂ˢ, ...   (outer cut 재료)
    → shadow_price(coupling): μ̂ˢ                (supergradient 성분)
  Solve dual ISP_follower(s):
    → obj_Fˢ
    → variable values: Ũˢ, β̃ˢ, Z̃ˢ, P̃ˢ, ...   (outer cut 재료)
    → shadow_price(coupling): μ̃ˢ                (supergradient 성분)

  ※ outer cut 재료와 supergradient는 ISP 풀이의 동시 산출물.
    μ를 위해 ISP를 추가로 푸는 것이 아님.

──── Step 3: Outer Cut 구성 → OMP에 추가 ────
기존 evaluate_master_opt_cut과 동일:
  cut₁ = -ϕU · Σₛ (Û₁ˢ + Ũ₁ˢ) .* diag(x)E
  cut₂ = -ϕU · Σₛ (Û₃ˢ + Ũ₃ˢ) .* (E - diag(x)E)
  cut₃ = Σₛ Z̃₁₃ˢ .* (diag(λ-v·ψ0) · diag(ξ̄ˢ))
  cut₄ = Σₛ (d₀ᵀ β̃₁₁ˢ) · λ
  cut₅ = -Σₛ (h + diag(λ-v·ψ0)·ξ̄ˢ)ᵀ β̃₁₃ˢ
  intercept = Σₛ (intercept_Lˢ + intercept_Fˢ)
OMP에 추가: t₀ ≥ cut₁ + cut₂ + cut₃ + cut₄ + cut₅ + intercept

──── Step 4: α Update (추가 비용 ≈ 0) ────
gₖ ← Σₛ (μ̂ˢ + μ̃ˢ)                   (Step 2에서 이미 얻은 것)
αₖ₊₁ ← Proj_Δ(αₖ + γₖ · gₖ)           (simplex projection, O(|A| log|A|))

──── Step 5: 수렴 판정 ────
(Section 6 참조)
```

### 4.2 Exact IMP와 교대 사용 (정밀 검증)

```
if k mod N_verify == 0 or LB stagnation:
    ┌─ 기존 inner Benders loop (tr_imp_optimize!) 호출 ──────────┐
    │  IMP(αₖ) → ISP → inner cut → IMP → ... → α* (수렴)       │
    │  V(χ̄ₖ) 정확히 계산                                        │
    └────────────────────────────────────────────────────────────┘
    UBₖ = f(χ̄ₖ) + V(χ̄ₖ)
    αₖ₊₁ ← α*  (exact IMP의 결과로 α 재보정)
    if UBₖ - LBₖ ≤ ε: TERMINATE
```

**α 재보정의 효과**: exact IMP에서 나온 α*는 현재 χ̄ₖ에 대한 global maximizer.
이것이 다음 H2 step들의 warm start가 됨 → subgradient drift 교정.

---

## 5. Inexact IMP 대비 H2의 이점

### 5.1 α trajectory 비교

```
Inexact IMP (inner 2회, 매번 리셋):
  Outer k:   [α₁ → α₂]  ← 2S×2=4S ISP, α₂ suboptimal
  Outer k+1: [α₁'→ α₂'] ← 4S ISP, α₂' suboptimal (이전 α₂ 버림)
  Outer k+2: [α₁"→ α₂"] ← 4S ISP, α₂" suboptimal (이전 α₂' 버림)

H2 (across outer iters 누적):
  Outer k:   ISP(αₖ) → μₖ → αₖ₊₁ = Proj(αₖ + γμₖ)     ← 2S ISP
  Outer k+1: ISP(αₖ₊₁) → μₖ₊₁ → αₖ₊₂ = Proj(αₖ₊₁ + γμₖ₊₁)  ← 2S ISP
  Outer k+2: ISP(αₖ₊₂) → μₖ₊₂ → αₖ₊₃ = Proj(αₖ₊₂ + γμₖ₊₂)  ← 2S ISP
```

Inexact IMP: 12S ISP로 3개의 독립적인 suboptimal α.
H2: 6S ISP로 3 step 누적 개선된 α. **절반 비용, 더 좋은 α.**

### 5.2 Inner cut 보존 문제의 우회

Inner cut을 across outer iterations로 보존하면 IMP warm-start도 가능하지만:
- inner cut = {t ≤ intercept(χ) + α'μ(χ)}: intercept와 μ가 χ에 의존
- χ가 바뀌면 이전 cut은 invalid → 관리 복잡

H2는 inner cut이 아예 없음. α 자체만 보존. 깔끔함.

---

## 6. 수렴 판정

### 6.1 문제: V(χ̄ₖ)를 모른다

H2에서 매 iteration 얻는 f(αₖ; χ̄ₖ)는 V(χ̄ₖ)의 **하한**:
$$f(\alpha_k;\bar\chi_k) \leq V(\bar\chi_k)$$

따라서 f(χ̄ₖ) + f(αₖ; χ̄ₖ)는 valid upper bound가 **아님** → gap 계산 불가.

### 6.2 해결: 주기적 정밀 검증 (Section 4.2)

매 N_verify회마다 기존 inner Benders를 정밀하게 풀어서 V(χ̄ₖ) 확보.
이때만 valid UB 갱신 + 종료 판정 가능.

### 6.3 Heuristic 종료 (보조)

- LB 변화량 < ε이 N_stall회 연속
- ‖αₖ₊₁ - αₖ‖ < ε_α

---

## 7. Step Size

### 7.1 표준 이론과의 차이

표준 projected subgradient: 고정된 concave f에 대해 수렴 보장.
H2: f(·; χₖ)가 매 outer iter마다 바뀜 → 직접 적용 불가.

### 7.2 Trust Region과의 시너지

Outer TR이 있으면:
- **Null step 동안 χ = χ_center (고정)**
- f(·; χ_center)가 동일한 함수
- → 연속된 null step = 같은 concave function에 대한 projected subgradient ascent
- → α가 α*(χ_center)로 수렴 → cut quality 개선 → serious step 촉진

**Null step이 α 개선 기회. H2 + TR의 핵심 상보성.**

### 7.3 실용적 선택

```julia
γ_k = γ₀ / sqrt(k_local)
```
- k_local: 마지막 serious step 이후 null step 수 (serious step에서 리셋)
- γ₀ ≈ w / ‖g₁‖ (첫 step이 simplex diameter 대비 적절한 크기)

---

## 8. Trust Region 통합

### 8.1 Serious Step 판정

현재: `improvement = past_major_obj - V(χ̄)` (V는 exact).
H2: `improvement = past_major_obj - f(αₖ; χ̄ₖ)` (f ≤ V이므로 과소평가).

→ serious step 판정 보수적 → null step 많아짐.
→ 그러나 null step 동안 α 개선 → 자기 수정적.

### 8.2 α 재보정과의 연동

정밀 검증(Section 4.2)이 serious step 판정과 자연스럽게 결합:
```
매 N_verify회: exact IMP → V exact → serious/null 정확 판정 + α 재보정
그 외: H2 step → 항상 null step 취급 (보수적) + α 점진 개선
```

---

## 9. Simplex Projection 구현

$$\text{Proj}_\Delta(z) = \argmin_{\alpha \geq 0, \mathbf{1}^\top\alpha = w} \|\alpha - z\|_2^2$$

```julia
function project_simplex(z::Vector{Float64}, w::Float64)
    n = length(z)
    u = sort(z, rev=true)
    cssv = cumsum(u)
    rho = findlast(i -> u[i] > (cssv[i] - w) / i, 1:n)
    theta = (cssv[rho] - w) / rho
    return max.(z .- theta, 0.0)
end
```

|A| = 50 (5×5 grid) → 마이크로초. 무시.

---

## 10. 기대 효과 추정

### 10.1 ISP 호출 (5×5, S=1)

| 전략 | ISP/outer | Est. outer iters | 총 ISP | vs Exact IMP |
|------|-----------|------------------|--------|-------------|
| Exact IMP (현재) | 9 | 60 | 540 | baseline |
| **H2 (verify 매 10회)** | **2.9** | **~100** | **~290** | **-46%** |
| **H2 (verify 매 5회)** | **3.8** | **~80** | **~304** | **-44%** |
| Inexact IMP (2회) | 4 | ? | ? | 불확실 |

### 10.2 시간 추정

ISP 1회 ≈ 1.5초:
- 현재: 540 × 1.5 ≈ 810초 (ISP only), 총 ~950초
- H2: 290 × 1.5 ≈ 435초, 총 ~510초 → **~46% 절감**

---

## 11. 구현 체크리스트

### 재사용 (변경 없음)

- `isp_leader_optimize!`, `isp_follower_optimize!`
- `evaluate_master_opt_cut`
- `update_outer_trust_region_constraints!`
- OMP 전체 구조

### 제거

- `build_imp`, `initialize_imp` (IMP 관련 전부)
- `tr_imp_optimize!` / `tr_imp_optimize_hybrid!` (inner loop 전부)
- Inner cut 관리 로직

### 새로 작성

- `project_simplex(z, w)`: ~10줄
- α state 관리 + update: ~5줄
- 주기적 검증 wrapper (기존 `tr_imp_optimize!` 호출): ~10줄

---

## 12. 요약

| 관점 | Exact IMP | Inexact IMP | H2 |
|------|-----------|-------------|-----|
| ISP/outer | 8-10S | 2-4S | **2S** |
| α quality | Optimal | 나쁨 (리셋) | **점진적 개선 (누적)** |
| IMP 필요 | ✓ (4-5회/outer) | ✓ (1-2회/outer) | **불필요** |
| α 정보 보존 | N/A (매번 exact) | ❌ 리셋 | **✓ across outers** |
| Valid UB | 매 iter | 매 iter | 주기적 검증 필요 |
| TR 상보성 | 없음 | 없음 | **✓ (null step = α 개선)** |
| 구현 복잡도 | IMP + inner cut | IMP + inner cut | **α + projection** |
