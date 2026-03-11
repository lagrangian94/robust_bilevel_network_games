# Bottleneck Analysis & Manuscript Summary

## 1. 문제 개요 (from Manuscript)

### 1.1 Problem: Network Interdiction Game with Private Uncertainty Model

**Bilevel structure**: Leader(interdiction) vs Follower(max-flow)

```
Leader x → Follower h=h(x) → Uncertainty ξ → Follower y=y(x,h,ξ)
```

- **Leader**: interdiction `x ∈ {0,1}^|A|`, budget `1ᵀx ≤ γ`
- **Follower Stage 1** (here-and-now): capacity recovery `h`, budget `1ᵀh ≤ w`
- **Follower Stage 2** (wait-and-see): flow `y` on network
- **Coupling**: `y_ij ≤ u_ij(ξ)(1 - v_ij(ξ)x_ij) + h_ij`
- **Uncertainty**: arc capacity `u(ξ)`, interdiction effectiveness `v(ξ)`

### 1.2 Reformulation Pipeline

1. **Stochastic Bilevel Program** (1): `min_x sup_{h,ŷ}` with arg max constraints
2. **arg max 제거** (2): Proposition 2.1 ([Yanikoglu 2018]) → equivalent reformulation with `z*(x)` value function
3. **Dualization** (3-5): inner sup → inf (dual), variable transform `ỹ←λỹ, h←λh`
4. **Network structure 적용** (6): `c=0, d₁=d₀` → 구체적 formulation with N, I₀, B matrices
5. **Affine Decision Rules** (ADR): `π(ξ) = Πξ`, `ϕ(ξ) = Φξ` etc. → finite-dimensional
6. **Robust Counterpart** (SOC): linear constraints → `ΛR = Q, Λr̄ ≥ q, [Λ]_m ∈ K_SOC`
7. **S-Lemma** (SDP): quadratic worst-case objective → PSD constraint (|A|+1)×(|A|+1)

### 1.3 Key Variables & Notation

| Symbol | Description | Space |
|--------|-------------|-------|
| `x` | Interdiction (binary) | `{0,1}^|A|` |
| `h` | Recovery (continuous) | `R+^|A|` |
| `λ` | Budget coupling scalar | `R+` |
| `ψ⁰` | McCormick linearization of `λx` | `R^|A|` |
| `Π̂ˢ, Π̃ˢ` | Dual flow coefficients (leader/follower) | `R^(|V|-1)×(|A|+1)` |
| `Φ̂ˢ, Φ̃ˢ` | Dual capacity coefficients | `R^|A|×(|A|+1)` |
| `Ψ̂ˢ, Ψ̃ˢ` | McCormick aux for `Φ·x` | `R^|A|×(|A|+1)` |
| `Ỹˢ, ỹ_ts` | Follower flow decision rules | varies |
| `Λ̂, Λ̃` | SOC robust counterpart multipliers | varies |
| `M̂ˢ, M̃ˢ` | SDP dual matrices | `S^(|A|+1)` |
| `ϑ̂ˢ, ϑ̃ˢ` | S-Lemma multipliers | `R+` |
| `α_k` | Inner master coupling variable | `R+` |
| `μ̂ˢ, μ̃ˢ` | Scenario-coupling epigraph vars | `R^|A|` |

### 1.4 Uncertainty Set

```
ξ = ξ̄ˢ(1 + ζ),  ‖ζ‖₂ ≤ ε
Ξₛ = {ζ ∈ R^|A| : Rζ ≥ r̄ˢ}  (SOC representation)
R = [0; I],  r̄ˢ = (-ε; 0)
Dₛ = diag(ξ̄ˢ)
```

- `ε ≤ 1` → worst-case capacity stays non-negative
- `ε`의 의미: nominal scenario 대비 최대 변화 비율

## 2. Full Model 구조 (2DRNDP, Eq. 16)

```
min  t + wν
s.t. 1ᵀh ≤ λw,  x ∈ X,  h ≥ 0
     Σ(η̂ˢ + η̃ˢ) ≤ St                          (epigraph)
     [ϑ̂ˢI - D'(Φ̂ˢ-diag(v)Ψ̂ˢ)  ...] ≥_PSD 0    (leader SDP, per s)
     [ϑ̃ˢI - D'(Φ̃ˢ-diag(v)Ψ̃ˢ)  ...] ≥_PSD 0    (follower SDP, per s)
     McCormick (Ψ̂, Ψ̃ for x coupling)
     Σ(μ̃ˢ + μ̂ˢ) ≤ Sν  ∀k                       (scenario coupling)
     Λ̂₁R = Q̂, Λ̂₁r̄ ≥ q̂, [Λ̂₁]_m ∈ K_SOC       (leader SOC RC)
     Λ̂₂R = -Φ̂ˢ_L, ...                           (leader SOC RC 2)
     Λ̃₁R = Q̃, Λ̃₁r̄ ≥ q̃, [Λ̃₁]_m ∈ K_SOC       (follower SOC RC)
     Λ̃₂R = -Φ̃ˢ_L, ...                           (follower SOC RC 2)
     ψ⁰ ≤ λᵁx, ψ⁰-λ ≤ 0, λ-ψ⁰ ≤ λᵁ(1-x)      (McCormick for λx)
```

## 3. Decomposition 계층 (Algorithm Hierarchy)

### 3.1 Outer Master Problem (OMP, Eq. 17)
- **Solver**: Gurobi MIP
- **Variables**: `x` (binary), `h`, `λ`, `ψ⁰`, `t₀`
- **Cuts**: Benders optimality cuts from OSP

### 3.2 Outer Subproblem (OSP, Eq. 18) = Primal
- **Solver**: Mosek (MISOCP+SDP via Pajarito)
- **Input**: fixed `(x, h, λ, ψ⁰)`
- **Variables**: `η̂, η̃, ν, Φ̂, Ψ̂, Φ̃, Ψ̃, ỹ_ts, μ, Λ̂, Λ̃`
- **Cones**: PSD `(|A|+1)×(|A|+1)`, SOC

### 3.3 Dualized Outer Subproblem (Eq. 19) → Nested Benders로 분해
- **Dual variables**: `M̂ˢ, M̃ˢ` (PSD), `Û, Ũ` (≥0), `β̂, β̃` (≥0), `Γ̂, Γ̃` (SOC), `α` (≥0), `Z` (free), `P` (≥0)
- **Adjoint operators** `A*`: SDP variable에 대한 linear operator의 adjoint를 trace로 유도

### 3.4 Inner Master Problem (IMP, Eq. 45)
- **Solver**: Gurobi LP
- **Variables**: `α_k` (coupling), `t₁ˢ` (epigraph)
- **Constraint**: `Σα_k = w/S`
- **Cuts**: Benders cuts from ISP

### 3.5 Inner Subproblem (ISP, Eq. 46-47)
- **Leader ISP** `Z₁^{L,s}(α)`: SDP with `M̂ˢ ∈ K_PSD`
- **Follower ISP** `Z₁^{F,s}(α)`: SDP with `M̃ˢ ∈ K_PSD`
- **Solver**: Mosek (conic)
- **PSD 크기**: `(|A|+1) × (|A|+1)` → **병목의 근본 원인**
- 각 scenario `s`에 대해 독립적으로 풀 수 있음

### 3.6 Dual of ISP (Eq. 48-49)
- Original primal과 동일 구조 but:
  - Objective에 `Σ α_k μ_k^s` 항 추가 (α는 parameter)
  - `ν ≥ Σ(μ̂+μ̃)` constraint는 master로 이동

## 4. Known Issues (from Manuscript)

1. **h=0, λ>0 infeasible**: `y(ξ)-h ≤ λξ(1-vx)`에서 worst-case ξ가 negative 가능 (ε 크기 따라)
   - 해결: `ξ = ξ̄(1+ζ)` 모델링으로 ε≤1에서 non-negative 보장
   - SDP part의 경우 별도 lemma 필요

2. **λ가 크면 infeasible**: LDR coefficient의 upper bound 초과
   - `λ ≤ π̃_ts(ξ) ≤ ... + ϕ̃(ξ)` worst-case 과정에서 발생

3. **ε sensitivity**: magnitude가 예민
   - 너무 작으면 `ε²` 때문에 numerically unstable
   - 너무 크면 negative worst-case → infeasible

4. **λ, ψ⁰을 subproblem으로 내리는 방안**: inner master에서 dual variable들을 master로 넣어야 함

---

## 5. 실험 결과 요약 (5×5 Grid, S=2)

| Algorithm | Wall time | ISP 풀이 횟수 | Est. Mosek 시간 | Outer iters |
|---|---|---|---|---|
| Strict Benders | 1101s | 627 | 941s (85%) | 677 |
| Nested Benders | 1198s | 815 | ~1222s | 146 |
| TR None (F,F) | 1189s | 815 | ~1222s | 146 |
| TR Outer only (T,F) | 1136s | 776 | ~1164s | 141 |
| **TR Inner only (F,T)** | **997s** | **652** | **~978s** | 151 |
| TR Both (T,T) | 1019s | 665 | ~998s | 154 |

## 6. 병목 구조

```
Strict:  OMP(Gurobi MIP) → OSP(Mosek conic)                    ← 2-level
Nested:  OMP(Gurobi MIP) → IMP(Gurobi LP) → ISP(Mosek conic)  ← 3-level
```

### ISP (Inner Subproblem) = Mosek conic solve
- PSD 행렬 크기: (|A|+1) × (|A|+1) = 51×51 (5x5 grid)
- **1회 풀이 시간: ~1.5초**
- 전체 시간의 **85%+** 차지
- Interior point method → O(n³) per iteration, 15-30 barrier iterations

### OMP / IMP
- OMP (Gurobi MIP): 무시할 수준 (<1%)
- IMP (Gurobi LP): 무시할 수준 (<1%)

### Julia overhead
- 행렬 연산, cut 생성 등: ~10-15%

## 7. 스케일링 분석

| Grid | Arcs |A| | PSD 크기 | Mosek 1회 | 이론 비율 |
|------|---------|----------|----------|-----------|
| 3×3 | 18 | 19×19 | ~0.04s | 1x |
| 5×5 | 50 | 51×51 | ~1.5s | (51/19)³ ≈ 374x |
| 7×7 (예상) | ~98 | 99×99 | ~10s | (99/19)³ ≈ 7000x |

## 8. Inner TR이 효과적인 이유

- ISP 풀이 횟수를 815 → 652로 **20% 감소**
- TR이 inner master를 안정화 → inner iteration 수 감소
- 1회 절약 = ~1.5초 → 총 ~244초 절약
- Outer iteration은 약간 증가 (146→151)하지만 ISP 감소 효과가 압도

## 9. 연구 방향 우선순위

### 1순위: SDP 풀이 시간 줄이기
- **레버리지**: 모든 알고리즘 variant에 곱하기로 작용
- **스케일링**: O(n³)이 근본 원인 — 7×7 이상에서는 현재 접근 불가능
- 가능한 접근:
  - SOC relaxation: PSD → SOC 근사 (차원 선형 감소)
  - Chordal decomposition: 네트워크 sparsity 활용한 PSD 분해
  - SDP-free reformulation: equivalent LP/SOCP formulation 탐색
  - Warm-starting: 근사 시작점 제공

### 2순위: Inner iteration 줄이기 (SDP 빨라진 후)
- Inner TR로 20% 달성 — 추가 개선 여지는 점진적
- Bundle method, level method 등 검토 가능
- SDP가 빨라지면 iteration 수가 다시 주요 병목이 됨

## 10. 3×3 Grid 비교 데이터

| Algorithm | S=2 | S=10 | S=20 |
|---|---|---|---|
| Strict | 2.87 | 17.32 | 34.96 |
| Nested | 5.98 | 39.77 | 82.85 |
| TR None | 6.00 | 39.83 | 82.63 |
| Inner only | **5.38** | **34.33** | **71.56** |
| Outer only | 6.23 | 57.51 | 122.03 |
| Both | 6.28 | 47.52 | 97.64 |

- 3×3에서는 Strict가 가장 빠름
- 5×5에서는 Inner TR only가 Strict보다 ~10% 빠름 → 네트워크 커지면 nested의 이점 발현
