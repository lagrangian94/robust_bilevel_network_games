# Strict Benders OSP DUAL_INFEASIBLE 디버깅 결과

## 문제
- 4×4 grid, ϕU=2 (=1/ε), λU=10에서 strict benders의 OSP가 DUAL_INFEASIBLE (unbounded Max)
- ϕU=10으로 하면 정상 동작
- Nested benders (primal ISP)는 ϕU=2에서도 정상

## 진단 방법
- `debug_test/diagnose_osp_ray_v2.jl`: Benders loop 돌면서 DUAL_INFEASIBLE 발생 시점 포착
- `debug_test/diagnose_direct.jl`: 실패하는 (x, λ, h, ψ0) 직접 입력하여 ray 추출
- Mosek의 INFEASIBILITY_CERTIFICATE (primal ray) 추출 → 어떤 변수가 unbounded direction인지 분석

## 핵심 결과

### 실패 조건
- Benders iteration 21에서 λ_sol = 2.3813 > ϕU = 2.0 일 때 발생
- Critical ϕU* = 2.003584 (이 값 이하에서 unbounded)

### Ray 구조
| 변수 | Ray magnitude | 역할 |
|------|--------------|------|
| βtilde1_1[32] | 2130.2 | dummy arc의 tilde block dual — **ray의 주 성분** |
| Ptilde2_Π | 2127.2 | Πtilde_0 constraint 보상용 P 변수 |
| Γtilde1 | 1065.1 | Λtilde1 constraint 흡수용 SOC 변수 |
| M, α, βhat2, βtilde2 | ≈0 | bounded (PSD/trace/sum constraint) |

### Objective 분해
| Term | Value | ϕU 의존? |
|------|-------|---------|
| βt1_1·λ·d0 | **+5072.64** | No (= λ_sol × βtilde1_1[32]) |
| P_lb_tilde | -4394.29 | Yes (×ϕU) |
| U3+Ut3 | -421.43 | Yes (×ϕU) |
| 기타 | -148.9 | Mixed |
| **NET** | **+9.0 > 0** | → Unbounded |

### 메커니즘
1. βtilde1_1[32] (dummy arc tilde dual)를 Δ만큼 증가시키면:
   - Objective gain = **λ_sol × Δ**
   - Πtilde_0 constraint (N_ts[sink]=+1)에서 Ptilde2_Π로 보상 → cost = **ϕU × Δ**
   - Λtilde1 constraint에서 Γtilde1 (SOC recession)로 흡수 → 무료
2. Net per unit: λ_sol - ϕU = 2.38 - 2.0 = +0.38 > 0 → **unbounded ray**

### 왜 이런 구조가 생기나
- βtilde1_1[32]는 I_0=[I|0] 구조 때문에 Φtilde_0 constraint에 직접 연결 안됨
- Πtilde_0 (flow conservation)과 Λtilde1만으로 연결
- N_ts는 sink 노드에 +1 하나뿐 → Ptilde2_Π 하나로 보상 가능
- 보상 cost = ϕU, gain = λ_sol → λ_sol > ϕU이면 unbounded

## 왜 Nested Benders에선 오류가 안 뜨나
- ISP follower도 **동일한 구조**를 가짐 (line 240 주석: `#이거만 maximize하면 dual infeasible`)
- 차이점: IMP의 epigraph 변수 `t_1_f[s]`에 `upper_bound = flow_upper`가 있음
- ISP follower가 unbounded여도 IMP는 bounded → OMP로 전파 안됨
- 또한 IMP cut이 λ > ϕU 영역을 간접적으로 차단하여 해당 영역 탐색을 방지
- **잠재적 위험**: λ > ϕU에 도달하면 ISP follower도 DUAL_INFEASIBLE → `error()` crash 가능

## 결론
**듀얼 코딩 에러가 아니라, ϕU와 λU 간의 관계에서 오는 구조적 문제.**
- ϕU는 LDR coefficient의 bound
- λ가 ϕU를 초과하면, tilde part의 dummy arc dual이 unbounded ray를 형성
- hat part는 coefficient가 1이라 ϕU ≥ 1이면 bounded

## 해결: λ ≤ ϕU 제약 추가
- OMP에 `λ ≤ ϕU` 추가 → strict benders 수렴 확인됨
- **이론적 근거**: 아래 "이론적 분석" 참조

## 이론적 분석: manuscript.md 수식 매핑

### 1. Primal 관점: 왜 λ > πU이면 infeasible인가

**Eq (6d)** — tilde part dummy arc flow conservation:
```
Nts⊺π̃(ξ) ≥ λ,  P̃-a.s.
```
LDR 적용: `π̃(ξ) = π̃₀ + Π̃_L·ζ`, so `Nts⊺π̃(ξ) = π̃₀_sink + Nts⊺Π̃_L·ζ`

Robust counterpart (worst case over `||ζ|| ≤ ε`):
```
π̃₀_sink - ε·||Nts⊺Π̃_L||₂ ≥ λ
```

**P bound** (manuscript lines 1350-1354, tilde part):
```
-πU·E ≤ [Π̃ₛ_L  π̃₀ₛ] ≤ πU·E   (element-wise)
```
따라서 `π̃₀_sink ≤ πU`.

**결합하면**: `πU - ε·||Nts⊺Π̃_L||₂ ≥ λ` 필요.
- Π̃_L을 0으로 놓으면 best case: `πU ≥ λ` → **λ ≤ πU 필요**
- Π̃_L ≠ 0이면 LHS가 더 작아짐 → 조건이 더 강해짐
- **코드에서 πU = ϕU** (모든 P bound에 ϕU 사용) → **λ ≤ ϕU 필요**

cf) **hat part (eq 6f)**: `Nts⊺π̂(ξ) ≥ 1` → `πU ≥ 1` 필요 → ϕU = 1/ε ≥ 1 (ε ≤ 1)이면 항상 만족

### 2. Dual 관점: Ray가 왜 생기는가 (manuscript eq 매핑)

**Eq (19)** — 듀얼 목적함수의 핵심 항:
```
+λd₀ᵀβ̃₁ₛ,₁           ← gain = λ per unit β̃₁[dummy]
-πU·||P̃₂ₛ,π||_F       ← cost = πU per unit P̃₂_Π  (코드에서 πU = ϕU)
```

**Eq (39)** — Π̃ₛ에 대한 듀얼 제약 (intercept block, j = |A|+1):
```
N·β̃₁ₛ,₁ + β̃₁ₛ,₄ + P̃₁ₛ,π - P̃₂ₛ,π = 0
```
N의 dummy arc 열 = Nts에서 sink row만 +1 (eq 9-10):
→ `β̃₁[dummy]` 증가 시 sink row에서 `+β̃₁[dummy]` 발생
→ `P̃₂_Π[sink, |A|+1]`로 보상 → 목적함수 cost = **πU = ϕU**

**Eq (33)** — Φ̃ₛ에 대한 듀얼 제약:
```
... + [-I₀·Z̃₁ₛ,₁ - Z̃₁ₛ,₅ + Z̃₂ₛ | I₀·β̃₁ₛ,₁ + β̃₁ₛ,₅ - β̃₂ₛ] + P̃₁ϕ - P̃₂ϕ = 0
```
I₀ = [I|0] → `I₀·β̃₁ₛ,₁`에서 **dummy arc 성분 제외** → β̃₁[dummy]와 무관

**Eq (43)** — Λ̃ₛ₁ 제약:
```
Z̃₁ₛR⊺ + β̃₁ₛ(r̄ₛ)⊺ + Γ̃ₛ₁ = 0
```
Γ̃ₛ₁ ∈ KSOC → recession cone에서 흡수 가능 → **추가 비용 0**

**종합 (ray per unit Δ)**:
| 경로 | 수식 참조 | 기여 |
|------|----------|------|
| β̃₁[dummy] ↑Δ | Eq (19): `λd₀ᵀβ̃₁` | +λΔ (gain) |
| → Eq (39) sink row 보상 | Eq (39): `Nβ̃₁ + P̃₁π - P̃₂π = 0` | P̃₂_Π ↑Δ |
| → P̃₂_Π objective cost | Eq (19): `-πU·||P̃₂π||` | -πUΔ (cost) |
| → Eq (43) SOC 흡수 | Eq (43): `β̃₁r̄⊺ + Γ̃₁ = 0` | 0 (free) |
| → Eq (33) 무관 | Eq (33): `I₀β̃₁` dummy=0 | 0 |
| **Net** | | **(λ - πU)Δ** |

λ > πU (= ϕU in code) → Net > 0 → **unbounded ray** → dual infeasible
λ > πU (= ϕU in code) → **primal infeasible** (strong duality)

### 3. 코드 vs Manuscript 차이

Manuscript (lines 1346-1360)에서는 **별도 bound** 사용:
| Bound | 적용 대상 | Manuscript |코드 (`build_dualized_outer_subprob.jl`) |
|-------|----------|-----------|-------|
| ϕU | Φ̂, Φ̃ (LDR capacity coeff) | ϕU | ϕU |
| πU | Π̂, Π̃ (LDR flow conservation coeff) | πU | **ϕU** (동일) |
| yU | Ỹ (LDR flow coeff) | yU | **ϕU** (동일) |
| ytsU | Ỹts (LDR dummy flow coeff) | ytsU | **ϕU** (동일) |

코드에서 모든 P bound를 ϕU로 통일 (line 174-175):
```julia
obj_term_ub_tilde = [-ϕU * sum(Ptilde1_Φ) - ϕU * sum(Ptilde1_Π) - ϕU * sum(Ptilde1_Y) - ϕU * sum(Ptilde1_Yts)]
```

**실제 필요한 조건은 λ ≤ πU**이지만, πU = ϕU이므로 λ ≤ ϕU가 됨.

### 4. 정확한 결론

- λ ≤ ϕU는 **모델링에서 놓친 것이 아니라**, LDR + Big-M reformulation의 **구조적 결과**
- 원래 문제 (eq 6)에서 π̃(ξ)는 free variable이므로 λ에 제한 없음
- 하지만 LDR approximation + P bound (boxing) 적용 시, π̃₀_sink ≤ πU 제약이 생김
- eq (6d) `Nts⊺π̃(ξ) ≥ λ`를 만족시키려면 **λ ≤ πU** 필요
- 코드에서 πU = ϕU → **λ ≤ ϕU** 필요

**대안**:
1. `λ ≤ ϕU` 제약 추가 (현재 방식, 가장 간단)
2. πU를 ϕU와 별도로 설정 (πU = λU ≥ λ로 하면 unbounded 발생 안함)
3. Benders에서 feasibility cut으로 처리 (OSP unbounded 시 λ > ϕU 영역 차단)

### 물리적 해석
| 파라미터 | 의미 |
|---------|------|
| ϕU (= πU in code) | LDR coefficient 크기 제한 → uncertainty에 대한 **follower 반응 능력의 한계** |
| λ | interdiction 강도 → uncertainty **증폭 계수** |

- λ > πU → eq (6d)에서 요구하는 `π̃_sink ≥ λ`를 LDR bound 내에서 달성 불가
- → follower inner problem이 LDR로 feasible한 해를 구성 불가 → infeasible

### Full model vs Decomposition
- **Full model**: λ와 inner problem이 동시에 최적화 → λ > πU 영역은 inner problem infeasible이므로 자연스럽게 회피
- **Benders decomposition**: OMP가 λ를 독립적으로 탐색 → λ > πU에 도달 가능 → OSP가 unbounded(=primal infeasible) 반환
- **놓친 것**: decomposition 시 암묵적 feasibility 조건을 명시적으로 추가해야 함
- `λ ≤ ϕU`는 원래 문제의 feasible region을 축소하는 것이 아니라, **LDR reformulation에서 원래부터 필요한 조건을 명시한 것**
- Benders에서 feasibility cut으로 처리할 수도 있지만, λ ≤ ϕU를 직접 넣는 것이 더 깔끔

## 관련 파일
- `debug_test/diagnose_direct.jl` — 직접 ray 추출 스크립트
- `debug_test/diagnose_osp_ray_v2.jl` — Benders loop 기반 진단
- `debug_test/test_build_dualized_outer_subprob.jl` — OSP 테스트 복사본
- `debug_test/test_strict_benders.jl` — strict benders 테스트 복사본
