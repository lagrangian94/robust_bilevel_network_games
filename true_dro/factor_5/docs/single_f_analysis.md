# Single-F 분석: ε̂=0, ε̃>0 (follower-only DRO)

## Setup
- Network: Polska, factor_additive (k=5), S=20, γ=2, seed=42
- Single-F: ε̂=0 (leader nominal), ε̃>0 (follower robust)
- q̂: uniform, sample1, sample3, sample8
- ε̃: 0.1, 0.3, 1.0

## 1. Benders 결과: 모든 single_F x* = NOM x*

8개 configuration (4 q̂ × 2 ε̃) + ε̃=1.0 (sample8) 추가:

| q̂ | NOM x* | ε̃=0.1 x* | ε̃=0.3 x* | ε̃=1.0 x* |
|---|--------|-----------|-----------|-----------|
| uniform | [3,6] | [3,6] | [3,6] | — |
| sample1 | [6,34] | [6,34] | [6,34] | — |
| sample3 | [6,34] | [6,34] | [6,34] | — |
| sample8 | [3,6] | [3,6] | [3,6] | [3,6] |

**모든 경우 x*(single_F) = x*(NOM).**

## 2. In-sample Z₀ 불변 확인

### Global subproblem (sample8, x=[3,6])

| ε̃ | Z₀ | status | TV(d*,q̂) | Σα |
|-----|-----------|---------|-----------|--------|
| 0.0 | 18.472646 | OPTIMAL | 0.000 | 31.3436 |
| 0.1 | 18.472646 | OPTIMAL | 0.100 | 31.3436 |
| 0.3 | 18.472646 | OPTIMAL | 0.300 | 31.3436 |
| 1.0 | 18.472646 | OPTIMAL (Benders) | — | — |

Z₀ 완전 동일. d*는 움직이고 (TV = ε̃), α 배분도 바뀌지만, Z₀ 불변.

### 8개 x에 대한 ε̃=0 vs ε̃=1.0 비교

| x | Z₀(ε̃=0) | Z₀(ε̃=1) | ΔZ₀ |
|---|----------|----------|-----|
| [3,6] optimal | 18.472646 | 18.472646 | 0.000000 |
| [17,34] | 23.773255 | 23.773255 | 0.000000 |
| [13,34] | 22.410661 | 22.410662 | +0.000001 |
| [14,34] | 23.060578 | 23.060576 | -0.000002 |
| [5,6] | 21.406877 | 21.406877 | 0.000000 |
| [22,33] | 22.617521 | 22.617520 | -0.000001 |
| [9,22] | 24.048657 | 24.048657 | -0.000000 |
| [14,22] | 23.340620 | 23.340620 | +0.000001 |

**모든 x에서 ΔZ₀ < 1e-5 (solver tolerance).** 이건 instance-specific이 아니라 구조적.

## 3. 구조적 원인: single_F ≡ NOM

### Game-theoretic 설명

Single_F의 bilevel 구조:
- Leader: min_x Z₀(x), q̂이 truth (ε̂=0)
- Subproblem: fixed x에서 worst-case (adversary가 Z₀(x) maximize)

Subproblem adversary는 d와 α를 jointly 선택. 이때:

1. **ε̃=0**: d=q̂ 강제, follower가 q̂ 기준 최적 α 선택 → Z₀ = E_q̂[flow(x, α*)]
2. **ε̃>0**: d가 자유이지만, d≠q̂은 **follower의 비합리적 의사결정** (true distribution이 q̂인데 다른 분포로 복구 계획)
3. 비합리적 의사결정 → follower 복구 효율 ↓ → flow ↓ → **leader에게 이득**
4. Subproblem adversary는 leader에게 이득을 주는 선택을 안 함 → **항상 d=q̂ (또는 동치) 선택**
5. 따라서 ε̃ 확대는 adversary에게 "쓸모없는 옵션"만 추가 → **Z₀ 불변**

### KKT reformulation 관점

Subproblem objective = obj_L + obj_F:
- obj_L = Σ_s q̂_s · ρ_s (d 무관, q̂ 고정)
- obj_F = follower primal − dual = 0 (strong duality at optimum)

ε̃는 obj_F의 constraint (TV ball on d)에만 영향. obj_F=0이고 obj_L은 d 무관이므로,
d-related KKT에서 φ̃ (TV dual)가 자유도를 흡수 → α에 대한 실질적 제약 변화 없음.

### 요약

**single_F에서 follower의 DRO는 vacuous.**
- Adversarial subproblem이 follower를 "비합리적"으로 만들 인센티브가 없음
- d=q̂이 항상 subproblem maximizer → ε̃ 무의미
- 이는 모든 x, 모든 q̂, 모든 ε̃에서 성립하는 **formulation의 구조적 성질**

## 4. 함의

### Single_F OOS 분석 불필요
- x*(single_F) = x*(NOM) 이므로 OOS 결과도 동일
- h*도 동일한 x*에서 동일한 q̂로 계산 → 모든 것이 NOM과 동치

### Two-layer DRO (ε̂>0, ε̃>0)에 대한 시사점
- Single_L (ε̂>0, ε̃=0): leader의 DRO가 x*를 바꿈 → OOS에서 효과 관찰 가능
- Single_F (ε̂=0, ε̃>0): follower의 DRO가 x*를 안 바꿈 → 효과 없음
- **Two-layer에서 ε̃의 역할**: ε̃는 ε̂과 결합될 때만 의미 있을 가능성
  - ε̂>0이면 leader의 worst-case q≠q̂ → subproblem에서 d≠q̂이 adversary에게도 유의미해질 수 있음
  - 단독 ε̃는 구조적으로 무력

## 5. Double-layer에서 q* = d* (ε̂ ≤ ε̃)

### 수학적 결과
ε̂ ≤ ε̃이면, double-layer subproblem의 optimal solution에서 **d* = a* (follower의 worst-case = leader의 worst-case)인 해가 반드시 존재**한다.

이유: minimax (zero-sum) 구조. Leader의 worst-case q_true를 maximize하는 방향과 follower의 worst-case q_tilde를 maximize하는 방향이 일치. ε̃ ≥ ε̂이면 follower의 TV ball이 leader의 TV ball을 포함하므로, d=a가 항상 feasible이고 optimal.

### 수치 검증 (3x3 grid, S=3, OPTIMAL 보장)

**Z₀(free) vs Z₀(d=a 고정):**

| (ε̂, ε̃) | Z₀(free) | Z₀(d=a) | gap |
|----------|----------|---------|-----|
| (0.1, 0.1) | 20.545038 | 20.545038 | 0.000000 |
| (0.1, 0.3) | 20.545038 | 20.545038 | 0.000000 |
| (0.2, 0.2) | 21.913067 | 21.913067 | 0.000000 |
| (0.3, 0.3) | 23.281095 | 23.281095 | 0.000000 |
| (0.5, 0.5) | 26.017152 | 26.017152 | 0.000000 |

**모든 ε̂ ≤ ε̃에서 gap = 0 → d=a도 optimal solution (multiple optima).**

참고: Gurobi가 자유 solve에서 찾는 해는 d≠a일 수 있음 (multiple optima 중 하나). 그러나 d=a를 강제해도 Z₀ 동일.

### 함의
- Double-layer에서 ε̃ > ε̂ 부분은 **redundant**: d*=a* optimal이 존재하므로, follower가 leader보다 넓은 ambiguity set을 가져도 실질적 추가 효과 없음
- ε̃ = ε̂ 이 sufficient — ε̃를 더 키워도 Z₀ 불변 (위 표에서 (0.1,0.1)과 (0.1,0.3)의 Z₀ 동일)
- **Effective parameter**: ε̂만이 Z₀를 결정. ε̃ ≥ ε̂이기만 하면 됨.

## 6. TODO: 수리적 증명

위 결과는 수치 실험 + game-theoretic 직관에 기반. Formulation 수준에서의 엄밀한 증명이 필요:
- Single_F subproblem에서 ε̃가 feasible set을 확장하지만 optimal value를 변화시키지 않음을 KKT/strong duality로 증명
- 특히 φ̃ (TV dual)의 자유도가 d-α coupling을 완전히 흡수함을 보일 것
- NOM subproblem과 single_F subproblem의 optimal value 동치성을 formal proposition으로 작성
- Double-layer에서 d*=a* optimality: ε̃ ≥ ε̂ → B_TV(q̂, ε̃) ⊇ B_TV(q̂, ε̂) → d=a feasible, zero-sum → d=a optimal

## Diagnostic scripts
- `diag_global_sub.jl` — sample8 x=[3,6]에서 global subproblem solve (ε̃=0/0.1/0.3)
- `diag_eps_effect_multi_x.jl` — 8개 x에 대해 ε̃=0 vs ε̃=1.0 비교
- `diag_alpha_compare.jl` — uniform q̂에서 α* 변화 + per-scenario flow 비교
- `run_single_f_batch.jl` — 8개 configuration Benders batch
- `diag_qtrue_eq_qtilde.jl` — Polska에서 a* vs d* 비교 (TIME_LIMIT 문제 있음)
- `diag_qtrue_eq_qtilde_grid3x3.jl` — 3x3 grid에서 a* vs d* 비교 (OPTIMAL 보장)
- `diag_fix_d_eq_a.jl` — d=a 고정 후 Z₀ 비교 (multiple optima 확인)

## Log files
- `logs/polska_eps0p1_single_f_factor.log` — uniform ε̃=0.1
- `logs/polska_eps0p3_single_f_factor.log` — uniform ε̃=0.3 (이전 존재)
- `logs/polska_eps0p1_single_f_factor_sample1.log` — sample1 ε̃=0.1
- `logs/polska_eps0p3_single_f_factor_sample1.log` — sample1 ε̃=0.3
- `logs/polska_eps0p1_single_f_factor_sample3.log` — sample3 ε̃=0.1
- `logs/polska_eps0p3_single_f_factor_sample3.log` — sample3 ε̃=0.3
- `logs/polska_eps0p1_single_f_factor_sample8.log` — sample8 ε̃=0.1
- `logs/polska_eps0p3_single_f_factor_sample8.log` — sample8 ε̃=0.3
- `logs/polska_eps1p0_single_f_factor_sample8.log` — sample8 ε̃=1.0
