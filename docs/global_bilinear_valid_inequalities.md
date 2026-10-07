# Ω (global bilinear program) 용 valid inequality 정리 (2026-10-07)

Benders subproblem Ω (`V*(x̄) = sup_{(α,ζ)∈Ω} F(x̄, α, ζ)`) 를 전역으로 풀 때 쓰는 부등식을 한곳에 모은다.
측정 근거는 `docs/alpha_bnb_notes.md` (zero-sum, Abilene) 와 `docs/nonzero_sum_location_results.md` §9 (비제로섬, location).

- zero-sum Ω: `true_dro/true_dro_build_subproblem.jl` (`build_true_dro_subproblem`)
- 비제로섬 Ω: `nonzero_sum/nz_omega.jl` (`build_nz_omega`)
- α-B&B 노드 완화: `true_dro/global_bilinear_solver.jl` (zero-sum), `nonzero_sum/nz_alpha_bnb.jl` (비제로섬)

## 0. 구조와 분류

**bilinear 항.** 전부 α × (belief 또는 인증서) 다. α 를 고정하면 Ω 는 LP.

| 항 | 의미 | zero-sum | 비제로섬 |
|---|---|---|---|
| ζL_ks = α_k r_s (CVaR, 아니면 α_k a_s) | 리더 측 용량 × 리더 가중치 | O | O |
| ζF_ks = α_k d_s | follower 측 용량 × follower belief | O | O |
| ζW_js = α_{h(j)} ϖ_js | H 행 j 의 follower 인증서 × 그 행의 α | − | O (html Step 4′ 의 α_k ϖ_k) |

**유효성 분류.** Benders 에 쓸 때 요구 조건이 다르다.

| 종류 | 뜻 | Benders 에서 |
|---|---|---|
| **A. implied** | Ω 의 모든 가능해가 만족 (기존 제약의 곱·결합) | cut·UB 모두 그대로 유효 |
| **B. 값 보존 restriction** | 가능해 일부를 자르지만 모든 x̄ 에서 최적값을 바꾸지 않음 | 해는 Ω 가능해 → cut 유효. 최적값 보존 → UB 유효. x̄ 와 무관해야 함 |
| **C. 최적성 기반** | x̄ 마다 전역 최적해를 하나 남김 (x̄ 에 의존) | 전역 풀이에서만. 로컬 풀이·시간 제한 incumbent 와 섞을 때 주의 |

## 1. 변수 범위 (A)

TV 볼과 CVaR 에서 바로 나오는 시나리오별 범위. spatial B&B 와 McCormick 의 출발점이다.

| 변수 | 범위 | 근거 |
|---|---|---|
| a_s | [max(0, q̂_s − 2ε̂), min(1, q̂_s + 2ε̂)] | ‖a − q̂‖₁ ≤ 2ε̂ |
| d_s | [max(0, q̂_s − 2ε̃), min(1, q̂_s + 2ε̃)] | 같음 |
| r_s | [0, a_s^max / (1 − β)] | r_s ≤ a_s / (1−β) |
| b_s, e_s | [0, 2ε̂], [0, 2ε̃] | Σ ≤ 2ε |
| α_k | [0, w] (location: [0, hᵁ_i]) | Σα ≤ w 에서 개별 상한 |
| ζL, ζF, ζW | [0, αᵁ · (짝 변수 상한)] | 곱의 상한 |

`true_dro_v5.md` §10.1 의 "tight per-scenario TV bounds". ε 가 작으면 a, d 의 범위가 4ε 로 좁아 McCormick 이 거의 exact.

## 2. McCormick (A)

곱 α·y (y ∈ [y^min, y^max]) 의 4 부등식. 아래 RLT 에서 "범위 행 × 상자 인수" 의 특수한 경우다.
`true_dro_v5.md` §10.2.

한계: belief 상한 r^max ≈ 0.68 이 1/S 보다 훨씬 커서 **root gap 이 S 에 비례해 커진다** (alpha_bnb_notes §1).

## 3. level-1 RLT: belief 다면체 × α 상자 (A)

α 상자 l ≤ α ≤ u 에서, belief 다면체의 선형 행 g_j(y) ≥ 0 마다
(α_i − l_i)·g_j(y) ≥ 0, (u_i − α_i)·g_j(y) ≥ 0 을 곱 변수 Z_is = α_i y_s 로 선형화한다.

곱하는 행:
- 범위 행 (y − y^min ≥ 0, y^max − y ≥ 0) — 이것만 곱하면 McCormick
- TV 볼: q̂_s + b_s − a_s ≥ 0, a_s + b_s − q̂_s ≥ 0, 2ε̂ − Σ_s b_s ≥ 0 (d, e, ε̃ 동일)
- CVaR: a_s/(1−β) − r_s ≥ 0

곱 변수가 필요한 족: a, b, r, d, e (각 K × S). ζL = Z[r] (CVaR) 또는 Z[a], ζF = Z[d].

효과 (zero-sum Abilene, alpha_bnb_notes §1~2): TV 볼 RLT 가 "초과 질량 합 ≤ 2ε" 를 보존해서
**root gap 이 S 와 무관하게 3~6%**. S=50 에서 α-B&B gap 0.07~0.26% (Gurobi global 1.9~6.2%).

적용 방식 (같은 노트 §3):
- 행 전부 넣으면 S=200 에서 21.6만 행, 노드당 47s → **위반 행만 분리**. root 값 같고 행은 7~12%.
- 부모 상자에서 만든 행은 자식에서도 유효. **행을 지우면 basis 를 잃는다** (13k iter) → 무효 행은 rhs = −∞ 로 비활성.
- Gurobi global 에 root RLT 를 통째로 주입하면 UB 는 좋아지지만 S=200 에서 LB 가 무너짐 (모델 비대, local cut 불가).

## 4. 정적 RLT (A)

상자와 무관하게 항상 유효한 곱. 노드 LP 에 한 번만 넣는다.

| 부등식 | 출처 | 비고 |
|---|---|---|
| Σ_s Z^f_ks = α_k, f ∈ {a, r, d} | α_k × (Σ_s y_s = 1) | 등식 RLT |
| Σ_k W_lk Z^f_ks ≤ w_l y_s | (w_l − W_l α) × y_s ≥ 0 | zero-sum: W = 1ᵀ, f ∈ {a, r, d}. 비제로섬: f ∈ {a, b, r, d, e} |
| (w − Σα)·g_j(y) ≥ 0 ("단체 RLT") | 예산 × belief 행 | **시도, 효과 미미** (root 18.026 → 18.006) |

## 5. 비제로섬: 인증서 스케일 상한 Wcap (B) ← 이번에 찾은 것

### 5.1 부등식

H 행 j (location: 예약 행) 와 시나리오 s 마다

  **ϖ_js ≤ ϖᵁ_j · r_s**    (location: ϖᵁ = p, 현장구매 프리미엄)

이전에는 상수 상한 ϖ_js ≤ ϖᵁ_j · r_s^max 만 있었다. `nz_omega.jl` 의 `Wcap`.

### 5.2 왜 값을 보존하나 (location)

ϖˢ 는 r_s 로 스케일된 follower recourse dual 이고 (N1: Aᵀϖˢ ≥ r_s c), Ω 에서는 N1 과 목적함수에만 나온다.
Ω 의 아무 가능해에서 예약 행 성분을 ϖ′_js = min(ϖ_js, p·r_s) 로 바꾸면:
- N1 유지: 예약분 열 y^R_ij 의 dual 제약 −μ_j + ϖ_res,i + ϖ_cap,i ≥ −r_s·dist_ij 는, 같은 (i, j) 의 현장구매 열 제약
  −μ_j + ϖ_cap,i ≥ −r_s·(dist_ij + p) 에 p·r_s 를 더한 것으로 보장된다. 다른 열에는 예약 행이 없다.
- 목적함수 불감소: 예약 행 ϖ 의 목적 계수는 −θ·(u + Bα)_j = −θ α_i ≤ 0 이므로 ϖ 를 줄이면 F 는 줄지 않는다.

→ 모든 가능해를 F 가 줄지 않는 Wcap 만족 해로 옮길 수 있다. x̄ 와 무관하므로 모든 x̄ 에서 최적값 보존 (B).
수치 확인: λᵁ ∈ {50, 100, 1000}, 16 개 x 모두 Ω = KKT V* (최대 차이 1.3e-7).

일반 원리: **"인증서 성분을 belief 가중치에 비례한 상한으로 자르는" 지배 논증.** 필요한 것은 (i) 상한으로 잘라도
dual 가능성이 유지되는 열 구조 (여기서는 현장구매 열이 예약 열을 덮음), (ii) 그 행의 목적 계수 부호 (≤ 0).
"모든 최적 dual 이 ϖᵁ 이하" 일 필요는 없다.

### 5.3 Wcap 의 RLT (A, Wcap 이 있는 Ω 에서)

인증서 행 ϖ_js ≥ 0, ϖᵁ_j r_s − ϖ_js ≥ 0 을 **자기 α_{h(j)} 하고만** 곱한다 (α_i ϖ_js, i ≠ h(j) 는 Ω 에 없는 곱이라 만들지 않음).
i = h(j), U = ϖᵁ_j 라 두면:

| 곱 | 선형화 |
|---|---|
| (α_i − l_i) ϖ_js ≥ 0 | ζW_js ≥ l_i ϖ_js |
| (u_i − α_i) ϖ_js ≥ 0 | ζW_js ≤ u_i ϖ_js |
| (α_i − l_i)(U r_s − ϖ_js) ≥ 0 | U ζL_is − ζW_js ≥ l_i (U r_s − ϖ_js) |
| (u_i − α_i)(U r_s − ϖ_js) ≥ 0 | u_i (U r_s − ϖ_js) ≥ U ζL_is − ζW_js |

뒤의 두 행이 ζW 를 ζL = α r 에 묶는다 (α ϖ ≤ U α r). 상수 상한 McCormick 에는 없는 결합이다.
`nz_alpha_bnb.jl` 의 `_nz_rows`.

### 5.4 효과 (seed 4, S=3)

| | Wcap 없음 | Wcap |
|---|---|---|
| Gurobi Ω, x=[3,4], λᵁ=1000 | 66~87s | 0.0s |
| Gurobi Ω, x=[3,4], λᵁ=50 | 2,400s 에도 gap 2.2e-4 | 1.5s |
| 표준 Benders (Gurobi oracle), λᵁ=1000 | 3,754s | 2.4s |
| belief-menu Benders, λᵁ=1000 | 69.5s | 0.1s |
| 표준 Benders, λᵁ=50 | 8,485s (Stalled) | 27.3s |

Gurobi NonConvex 는 RLT 를 쓰지 않고 원래 Ω 를 푼 것이다. 빨라진 이유는 ϖ 가 r_s 와 묶여 Gurobi 의 α·ϖ 완화가
조여진 것으로 보이지만, Gurobi 내부의 어떤 경로 (범위 축소 / 자체 RLT cut) 인지는 확인하지 않았다.
α-B&B 는 Wcap 이 있는 상태로만 돌려 5.3 의 RLT 행 기여는 따로 재지 않았다.

### 5.5 후보 (미검증)

- 같은 지배 논증으로 다른 행에도 스케일 상한: location 의 용량 행 ϖ_cap ≤ (c^max + p) r_s, 수요 행 μ ≤ (c^max + p) r_s
  (A 점포에 항상 여유가 있다는 nz_data.jl 논증). 이 행들은 bilinear 에 안 들어가서 효과는 LP 완화 결합 쪽뿐일 것.
- follower 블록의 π̃ (λ 로 스케일된 follower dual) 에도 π̃_ks ≤ λᵁ π_Fᵁ_k d_s 형태가 가능한지. 지금은 상수 상한 λᵁ π_Fᵁ_k.
- zero-sum Ω 에는 ϖ 같은 인증서 bilinear 이 없어 Wcap 은 해당 없음. 다만 "dual 상한을 belief 가중치에 비례하게" 거는
  강화가 리더·follower 블록의 다른 변수에도 되는지는 볼 만하다.

## 6. big-M 상수 (Ω 의 McCormick 범위)

부등식은 아니지만 완화 강도를 직접 정한다. 유효 범위 안에서 작을수록 강하다.

| 상수 | 역할 | location 에서 | 근거 |
|---|---|---|---|
| θᵁ | follower 최적성 벌점 (리더 블록) | Lemma 2: 1200 → circuit 상계 **50** (달성됨) | results §4 |
| π̂ᵁ_k | x·π̂ McCormick (리더) | 점포별 θᵁ·max_j (c_Aj + p − c_ij)⁺ (`bigm=:tight`, 이전 θᵁ(c^max+p)+v) | nz_data.jl, results §9d |
| π_Fᵁ_k | x·π̃ McCormick (follower, × λᵁ) | 점포별 max_j (c_Aj + p − c_ij)⁺ (이전 c^max + p) | 같은 곳 |
| λᵁ | follower exact penalty | 경계 **50** (< 50 이면 과대평가 10·(50−λᵁ)) | results §8 |

λᵁ 는 경계 바로 근처면 Ω 가 평평해져 Gurobi·α-B&B 모두 느려진다 (α-B&B x=[3,4]: 1000 → 노드 1, 50 → 노드 4,669).
경계의 2 배 이상 (λᵁ ≥ 100) 이면 Wcap 이 있는 Ω 에서 Gurobi·α-B&B 모두 1 초 안팎.

## 7. 최적성 기반 (C) — zero-sum 코드에 있음

`true_dro_build_subproblem.jl` 의 `add_objF_vi` / `add_objF_vi_arcwise`.
follower 목적 F^F 의 각 항 (−φ̃ᵁ x_k ρ̃¹, −φ̃ᵁ (1−x_k) ρ̃³, −λᵁ x_k ρ⁰¹, −λᵁ (1−x_k) ρ⁰³) 이 구조적으로 ≤ 0 이고 전역 최적에서
F^F = 0 이므로 각 항 ≥ 0 을 강제한다 (arcwise 판은 해당 변수 상한을 0 으로 고정).
- x̄ 에 의존 (C). 코드 주석대로 **전역 풀이에서만** 쓴다 (로컬 해에서는 F^F < 0 이 가능해 잘못 자름).
- "전역 최적에서 F^F = 0" 은 λᵁ 가 exact 일 때만 성립한다. λᵁ 가 경계 아래면 최적해가 반응집합 밖에 있을 수 있다.
- 효과 측정 기록은 없음 (`compare_objF_vi.jl` 스크립트만 있음). 비제로섬에는 아직 옮기지 않았다.

## 8. 시도했지만 버린 것 (zero-sum, alpha_bnb_notes §3)

| 시도 | 결과 |
|---|---|
| OBBT (root) | 고정되는 α 없음. S=50 일부 이득, S=200 손해 (LP 1 개 ~12s) |
| 단체 RLT (w − Σα)·g ≥ 0 | 미미 |
| Gurobi global + root RLT 전체 주입 | UB 개선, S=200 LB 붕괴 |
| Gurobi + root RLT user cut callback | S=50 UB 개선, S=200 root cut 루프에 시간 소진 |
| PWL (α 구간 binary) | 모델 25 만 행 (S=50) |
| Gurobi α BranchPriority | 무시됨 (연속 변수 spatial 분기에 미적용) |

## 9. 요약

| 부등식 | 대상 | 종류 | 코드 | 측정된 효과 |
|---|---|---|---|---|
| TV·CVaR 변수 범위 | 둘 다 | A | 두 빌더 | McCormick 의 출발점 |
| McCormick | 둘 다 | A | (RLT 범위 행) | root gap ∝ S |
| level-1 RLT (TV 볼·CVaR × α 상자) | 둘 다 | A | 두 α-B&B | root gap S 무관 3~6% (zero-sum) |
| 정적 RLT (Σ Z = α, Σ W Z ≤ w y) | 둘 다 | A | 두 α-B&B | (분리 RLT 와 같이 사용) |
| **Wcap ϖ ≤ p·r** | 비제로섬 | **B** | `nz_omega.jl` | Gurobi Ω 수백 초 → ~1s, Benders 3,754s → 2.4s |
| Wcap 의 RLT (자기 α 만) | 비제로섬 | A | `nz_alpha_bnb.jl` | 단독 효과 미측정 |
| 작은 big-M (θᵁ, λᵁ 를 유효 경계 근처까지) | 둘 다 | − | nz_data.jl | θᵁ 1200 → 50 으로 수치 오차 제거. λᵁ 는 경계 바로 근처 피하기 |
| objF 항별 VI | zero-sum | C | `add_objF_vi` | 기록 없음 |

다음에 볼 것:
1. Wcap 의 RLT 행 단독 기여 (α-B&B 에서 Wcap 행만 빼고 비교).
2. 5.5 의 스케일 상한 후보 (용량·수요 행, follower π̃).
3. 실험 머신 (worker 12 + 휴리스틱) 에서 α-B&B 와 Gurobi 속도 비교, 더 큰 인스턴스 (S, 고객 수).
