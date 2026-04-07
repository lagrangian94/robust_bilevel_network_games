# Pareto-Optimal Benders Cut 강화 기법

## 1. 배경: 왜 Cut 강화가 필요한가

Benders decomposition에서 subproblem의 최적해가 degenerate할 때, 같은 α*에서 **무한히 많은 valid cut**이 존재.
IPM (interior point method) solver는 analytic center를 반환하는데, barrier 크기에 따라 norm이 달라짐:
- Full OSP (큰 모델): barrier ↑ → `||Uhat1||` 팽창 → x=0 방향에서 급락하는 약한 cut
- ISP decomp (작은 모델): barrier ↓ → 작은 norm → 더 robust한 cut

**목표**: degenerate 자유도를 활용하여 Pareto-optimal (nondominated) cut을 생성.

---

## 2. 문제 구조 (ISP)

우리 ISP는 **master 변수가 objective에** 들어가는 conic subproblem:

```
z*(x, λ, h, ψ0; α) = max_Y  f(x, λ, h, ψ0; Y)
                       s.t.   g(Y, α) ≥ 0   (coupling: α fixed)
                              Y ∈ K           (conic)
```

Leader ISP objective (per scenario s):
```
max  -ϕU·Σ(Ûhat1 ⊙ diag(x)·E)           ← x-dependent term 1
     -ϕU·Σ(Ûhat3 ⊙ (E - diag(x)·E))     ← x-dependent term 2
     + d0'·β̂hat1_1                         ← intercept (P-terms)
     - ϕU·Σ(P̂hat1_Φ) - πU·Σ(P̂hat1_Π)     ← intercept (bounds)
     - ϕU·Σ(P̂hat2_Φ) - πU·Σ(P̂hat2_Π)
```

Follower ISP objective (추가 terms):
```
     + Σ(Z̃tilde1_3 ⊙ (diag(λ-v·ψ0)·diag(ξ̄)))  ← λ,ψ0-dependent
     + λ·d0'·β̃tilde1_1                             ← λ-dependent
     - (h + diag(λ-v·ψ0)·ξ̄)'·β̃tilde1_3            ← h,λ,ψ0-dependent
```

**핵심**: master 변수 (x, λ, h, ψ0)가 constraint RHS가 아니라 **objective coefficient**에 등장.

---

## 3. 기법 A: Magnanti-Wong (MW) Cut

### 이론
Magnanti & Wong (1981): core point x̄ ∈ relint(conv(X))에서 cut value를 최대화하면 **Pareto-optimal cut**.

### 알고리즘 (2-solve per ISP)
```
Phase 1: ISP(x_sol, α*) → z*, 기본 해 Y*
Phase 2: max  f(x̄_core; Y)        ← core point에서의 cut value 최대화
         s.t. f(x_sol; Y) ≥ z*    ← optimality constraint (원래 점에서 여전히 최적)
              g(Y, α*) ≥ 0
              Y ∈ K
         → Y_MW: Pareto-optimal cut coefficients
```

### 구현: `evaluate_mw_opt_cut` (nested_benders_trust_region.jl:590)

```julia
function evaluate_mw_opt_cut(
    isp_leader_instances, isp_follower_instances, isp_data, cut_info, iter;
    x_sol, λ_sol, h_sol, ψ0_sol,        # 현재 점 (optimality constraint)
    x_core, λ_core, h_core, ψ0_core,    # core point (MW objective)
    multi_cut=false)
```

**Per scenario s, leader:**
1. `z_star_l = objective_value(model_l)` (Phase 1 결과 재사용)
2. `orig_obj_l` 재구성 (x_sol 기준 objective expression)
3. `@constraint(model_l, orig_obj_l >= z_star_l - 1e-6)` — optimality constraint 추가
4. `core_obj_l` 구성 (x_core로 교체) → `@objective(model_l, Max, core_obj_l)`
5. `optimize!(model_l)` — Phase 2 solve

**Per scenario s, follower:** 동일 + λ_core, h_core, ψ0_core 사용

**Cleanup (필수):**
- MW constraint 삭제: `delete(model_l, mw_con_l)`
- 원래 objective (x_sol 기준) 복원
- **Re-solve**: 다음 core point 호출 시 `objective_value`가 원래 z*를 반환해야 함
- → ISP instance를 재사용하기 위한 state 복원

**비용**: Leader 2회 + Follower 2회 = 4 conic solves per scenario per core point

---

## 4. 기법 B: Sherali ζ-Perturbation Cut

### 이론
Sherali & Lunday (2011): subproblem objective를 ζ만큼 core point 방향으로 perturbation하면,
secondary optimization 없이 **ε₀-optimal maximal nondominated cut**을 1회 solve로 추출 가능.

원래 논문은 LP의 RHS perturbation (`b → b + ζb̄`):
```
min c'y  s.t.  Ay ≥ b + ζb̄  →  dual: max (b + ζb̄)'π  s.t. A'π ≤ c
```

### 우리 구조로의 적용 (Objective Perturbation)
ISP는 master 변수가 objective에 있으므로, RHS perturbation 대신 **objective perturbation**:
```
x_pert = x_sol + ζ · x_core
λ_pert = λ_sol + ζ · λ_core
h_pert = h_sol + ζ · h_core
ψ0_pert = ψ0_sol + ζ · ψ0_core
```

Perturbed ISP:
```
max  f(x_pert, λ_pert, h_pert, ψ0_pert; Y)
s.t. g(Y, α*) ≥ 0
     Y ∈ K
→ Y_pert: maximal cut coefficients
```

**ζ = 1e-8** (충분히 작아서 face를 크게 벗어나지 않으면서, degenerate 방향에서 maximal을 선택)

### Cut validity 보장
어떤 ISP-feasible Y에서든 valid cut이 성립:
```
z*(x) = max_Y f(x; Y) ≥ f(x; Ỹ*)  ∀ feasible Ỹ*
```
Sherali perturbed solve의 결과 Y_pert도 feasible → cut(x) ≤ z*(x) → valid.

### Core point 조건
Sherali의 PWV (Positive Weight Vector) 조건: core point의 모든 component가 양수.
이는 MW의 relint(conv(X)) 조건보다 **약한** 조건 → 기존 `generate_core_points`의 `:interior` 전략으로 충족.

### 구현: `evaluate_sherali_opt_cut` (nested_benders_trust_region.jl:792)

```julia
function evaluate_sherali_opt_cut(
    isp_leader_instances, isp_follower_instances, isp_data, cut_info, iter;
    x_sol, λ_sol, h_sol, ψ0_sol,
    x_core, λ_core, h_core, ψ0_core,
    ζ=1e-8, multi_cut=false)
```

**구현 차이점 (vs MW):**
1. `isp_leader/follower_optimize!`를 **호출하지 않음** — 직접 `@objective` 설정 + `optimize!`
   - 이유: perturbed x가 fractional → `isp_leader_optimize!` 내부의 strong duality assertion (`abs(dual_obj - primal_obj) < 1e-4`) 위반
2. **Constraint 추가/삭제 없음** — objective만 변경
3. **State 복원 불필요** — 다음 iteration에서 새 (x, λ, h, ψ0)로 objective가 갱신되므로

**비용**: Leader 1회 + Follower 1회 = 2 conic solves per scenario per core point
(MW의 절반)

### ζ 선택 근거: 왜 1e-8인가

Additive perturbation `x_pert = x_sol + ζ·x_core`에서, `x_sol[i] = 1`인 arc에 대해:
```
x_pert[i] = 1 + ζ·x_core[i] > 1
→ (1 - x_pert[i]) < 0
→ Uhat3 term: -ϕU·Uhat3·(E - diag(x_pert)·E) 의 부호 반전
→ Max 문제에서 Uhat3 → +∞ → DUAL_INFEASIBLE (primal unbounded)
```

**실험 결과**:
| ζ | 결과 |
|---|------|
| 1e-8 | 정상 동작 (solver가 수치적으로 무시하는 수준) |
| 1e-7 | DUAL_INFEASIBLE (leader + follower 모두) |
| 1e-6 | DUAL_INFEASIBLE |

**대안**: convex combination `x_pert = (1-ζ)·x_sol + ζ·x_core` → x_pert ∈ [0,1] 보장.
현재는 additive 방식 + ζ=1e-8로 고정.

### IPM face jumping 현상
ζ=1e-8로 매우 작은 perturbation이지만, conic solver (Mosek)는 완전히 다른 analytic center로 이동할 수 있음:
```
pert_obj = 8.0294, orig_obj = 7.8462  (차이 0.18 despite ζ=1e-8)
```
이는 **정상** — IPM이 다른 optimal face의 중심으로 점프한 것.
중요한 것은 `pert_obj`가 아니라 **`cut@x̄`** (원래 점에서의 cut value).

### Cut quality 진단
```julia
slack = cut@x̄ - z*(x̄)    # z* = cut_info[:obj_val]
```

| slack 범위 | 판정 | 의미 |
|-----------|------|------|
| > 1e-3 | INVALID (assert error) | cut이 z*보다 큼 → 위반 |
| > -1e-2 | near-optimal | 거의 tight — 최상 |
| > -1e-1 | slightly loose | 약간 loose하지만 acceptable |
| ≤ -1e-1 | LOOSE | 약한 강화 — core point 재검토 필요 |

---

## 5. Core Point 생성: `generate_core_points` (nested_benders_trust_region.jl:545)

```julia
function generate_core_points(network, γ, λU, w, v;
    interdictable_idx=nothing, strategy=:interior_and_arcs)
```

### Strategy 옵션

| Strategy | 생성 수 | 내용 |
|----------|---------|------|
| `:interior` | 1개 | x̄ᵢ = γ/\|A_I\| (fractional), λ̄ = λU/2, McCormick ψ0 |
| `:arc_directed` | min(γ, \|A_I\|)개 | interdictable arc별 eᵢ (binary, 1개만 1) |
| `:interior_and_arcs` | 1 + min(γ, \|A_I\|)개 | 위 둘 합산 |

### McCormick ψ0 계산
```julia
ψ0_bar[k] = min(λU * x_bar[k], λ_bar, max(λ_bar - λU * (1 - x_bar[k]), 0.0))
```

---

## 6. Solver 통합

### Parameter: `strengthen_cuts::Symbol`

| 값 | 동작 | Per-iter conic solves (S=1) |
|----|------|-----------------------------|
| `:none` | 기본 cut만 | Leader 1 + Follower 1 = 2 |
| `:mw` | 기본 cut + MW cut | 2 + 2(MW phase2) + 2(MW restore) = 6 |
| `:sherali` | Sherali cut만 (기본 cut 생략)* | Leader 1 + Follower 1 = 2 |

*Strict Benders에서만 기본 cut 생략. Nested/Hybrid에서는 기본 cut + Sherali cut 병행.

### Strict Benders (strict_benders.jl) — `:sherali` 경로
```
OSP solve → α* 획득 → (ISP 원본 solve 생략) → Sherali perturbed cut만 추가
```
ISP 원본 solve + ISP-based cut을 완전히 건너뛰고, Sherali 1개만 추가.

### Strict Benders — `:mw` 경로
```
OSP solve → α* 획득 → ISP solve → ISP-based cut 추가 → MW cut 추가
```
2개 cut per iteration (ISP base + MW).

### Nested/Hybrid (nested_benders_trust_region.jl)
```
Inner Benders loop → outer cut 생성 → (if strengthen) additional MW/Sherali cut 추가
```
기존 outer cut은 inner Benders에서 자연스럽게 나오므로, MW/Sherali는 추가 cut.

---

## 7. 관련 파일

| 파일 | 역할 |
|------|------|
| `nested_benders_trust_region.jl` | `generate_core_points`, `evaluate_mw_opt_cut`, `evaluate_sherali_opt_cut` 정의 |
| `strict_benders.jl` | `strict_benders_optimize!` — `:mw`/`:sherali` 분기 |
| `compare_cuts.jl` | 6-way 비교: Strict/Nested/Hybrid/MW-interior/MW-arc/Sherali |
| `compare_benders.jl` | 벤치마크 — `strengthen_cuts=:mw` 또는 `:sherali` |

---

## 8. MW vs Sherali 비교

| | MW | Sherali |
|---|---|---|
| Cut quality | **정확히 Pareto-optimal** (2nd optimization 보장) | ε₀-optimal (근사적 maximal) |
| 추가 비용 (ISP 원본 cut 외) | Phase2 + restore = **4 extra conic**/scenario | **2 extra conic**/scenario |
| 구현 복잡도 | Constraint 추가/삭제/objective 복원 | Objective만 변경 |
| Cleanup | 필수 (ISP state 복원) | 불필요 |

**결론**: MW가 cut quality 우위. Sherali는 절반 비용으로 near-optimal. 둘 다 유지하고 실험적으로 비교 권장.

## 9. 알려진 이슈

1. **Sherali only (기본 cut 생략) 시 LB 개선 저하**: Sherali cut이 현재 점에서 original cut보다 loose할 수 있음. ISP 원본 cut + Sherali를 함께 쓰는 게 안전.
2. **IPM face jumping**: ζ=1e-8에서도 conic solver가 다른 face로 이동 → pert_obj가 크게 달라질 수 있으나, cut validity에는 영향 없음.
3. **MW cleanup 비용**: Phase 2 solve 후 원래 objective 복원 + re-solve 필요 → solve 횟수 증가.
4. **ζ ≥ 1e-7 → DUAL_INFEASIBLE**: additive perturbation에서 x_pert > 1 → objective 부호 반전 → unbounded. ζ=1e-8로 고정 필요.
