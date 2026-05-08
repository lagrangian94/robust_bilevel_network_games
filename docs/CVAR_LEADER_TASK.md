# Task: Extend Expectation-Case Codebase to CVaR-Leader Case

## Context

기존 코드는 expectation-leader / expectation-follower DRO 네트워크 인터딕션 문제를 Benders decomposition 으로 풀고 있다. 본 task 는 leader 의 risk functional 을 expectation 에서 **CVaR$_\beta$** 로 확장하는 것.

**수학적 reference**: `cvar_leader_merged.pdf` (특히 §7, §8, §9, §10).

**구현 원칙**: $\beta$ 를 파라미터로 추가. $\beta = 0$ 에서 기존 expectation 코드와 **수치적으로 정확히 동일** 해야 한다 (이게 첫 번째 acceptance test).

---

## Pre-work: 코드 구조 파악

먼저 다음을 읽고 정리한 메모를 출력할 것 (수정 시작 전):

1. `build_primal_isp.jl` 의 `primal_isp_leader_optimize!` --- ISP-L primal LP 구성
2. `build_dual_isp.jl` (또는 dual 빌더) --- ISP-L dual LP 구성
3. `evaluate_master_opt_cut` --- cut intercept/slope 추출
4. ISP-F 관련 파일들 (이건 변경하지 않을 것이지만 호출 인터페이스 확인 필요)
5. OMP 빌더 / solver

**메모 형식** (이걸 먼저 출력):
```
## Primal ISP-L 구조
- 파일: <경로>
- 변수: σ^L±, μ^L, η^L (이름은 코드에 맞춰), π̂, φ̂, ψ̂
- 제약: (EL-1)~(EL-7), McCormick
- 목적: min Σ q̂_s(σ⁺ - σ⁻) + 2ε̂ μ^L + η^L

## Dual ISP-L 구조
- ...

## Cut extraction
- 파일: <경로>
- Slope formula: -φ^U Σ (ρ̂^{s,1} - ρ̂^{s,3}) - ...
```

이 메모가 정확하지 않으면 이후 모든 변경이 잘못된다. 확실하지 않은 부분은 grep 으로 직접 확인할 것.

---

## 변경 범위 요약

| 컴포넌트 | 변경 여부 | 비고 |
|---|---|---|
| OMP | **변경 X** | 변수, 제약, cut 형태 모두 동일 |
| ISP-L primal | **변경 O** | R-U 변수 $(\eta_\beta, \vartheta_s)$ 추가, obj 수정 |
| ISP-L dual | **변경 O** | $r_s$ 추가, (DL-2C)(DL-3C) 수정, (DL-RU1)(DL-RU2) 추가 |
| ISP-F (primal/dual) | **변경 X** | 완전히 그대로 |
| Cut intercept 추출 | **변경 X** | $Z^{L*}$ 값 자체만 사용 |
| Cut slope 추출 | **변경 X** | McCormick 듀얼 $\hat\rho^{s,1}, \hat\rho^{s,3}$ 동일 |
| 외부 인터페이스 | **변경 O** | $\beta$ 파라미터 추가 (기본값 0.0) |

핵심 메시지: **ISP-L 만 수정하면 끝**. 나머지는 인터페이스 파라미터만 추가.

---

## Step-by-Step Implementation

### Step 1. $\beta$ 파라미터 추가

`isp_data` (또는 글로벌 config) 에 `beta::Float64` 필드 추가. 기본값 `0.0`. 어디에서 받아오는지 명확히 (예: command-line arg, config file).

**Acceptance**: $\beta$ 가 모든 ISP-L 호출 경로로 전파되는지 확인.

### Step 2. ISP-L Primal 수정

`primal_isp_leader_optimize!` (또는 빌더) 에 다음 추가:

**새 변수**:
- `η_β` : `JuMP.@variable(model, η_β)` (free; box bound 으로 `[0, M̄]` 추가 권장 — Remark 참조)
- `ϑ[s in 1:S]` : `JuMP.@variable(model, ϑ[1:S] >= 0)` (또한 box bound `[0, M̄]`)

**기존 obj 수정** (Form 2 사용 권장; PDF §9):
```julia
# 기존:
# @objective(model, Min, sum(q̂[s]*(σ⁺[s] - σ⁻[s]) for s) + 2ε̂*μ^L + η^L_TV)

# 새:
@objective(model, Min,
    η_β + (1/(1-β)) * (sum(q̂[s]*(σ⁺[s] - σ⁻[s]) for s) + 2ε̂*μ^L + η^L_TV)
)
```

⚠️ `η^L` 의 이름이 코드에 어떻게 되어 있는지 확인. PDF 에서는 $\etaTV := \eta^L_{\mathrm{TV}}$ 로 표기 (TV envelope 의 simplex 제약 dual).

**기존 (EL-1) 제약 split**:
```julia
# 기존 (EL-1): σ⁺ - σ⁻ + η^L ≥ Σ_k ξ̄^s(φ̂ - v_k ψ̂) + Σ_k α_k φ̂

# 새 (EL-1C): σ⁺[s] - σ⁻[s] + η^L_TV - ϑ[s] ≥ 0   ∀s
@constraint(model, [s=1:S],
    σ⁺[s] - σ⁻[s] + η^L_TV - ϑ[s] >= 0
)

# 새 (EL-5C): η_β + ϑ[s] - Σ_k ξ̄^s_k(φ̂[k,s] - v_k ψ̂[k,s]) - Σ_k α[k] φ̂[k,s] ≥ 0   ∀s
@constraint(model, [s=1:S],
    η_β + ϑ[s] - sum(ξ̄[k,s]*(φ̂[k,s] - v[k]*ψ̂[k,s]) for k) - sum(α[k]*φ̂[k,s] for k) >= 0
)
```

**나머지 (EL-2)~(EL-McC3) 변경 없음.**

**Acceptance**: $\beta = 0$ 에서 기존 코드와 obj value 가 일치 (소수점 6자리까지).

### Step 3. ISP-L Dual 수정

`isp_leader_optimize!` (dual 버전) 에 다음 추가:

**새 변수**:
- `r[s in 1:S]` : `JuMP.@variable(model, r[1:S] >= 0)`. Box bound `[0, 1/(1-β)]` 추가.

**기존 (DL-2), (DL-3) 에서 $a_s$ 를 $r_s$ 로 교체** ($\widetilde{\text{DL-2C}}$, $\widetilde{\text{DL-3C}}$):
```julia
# 기존 (DL-2): û[k,s] - (ξ̄[k,s] + α[k])*a[s] + ρ̂[k,s,2] - ρ̂[k,s,3] ≤ 0
# 새 (DL-2C):  û[k,s] - (ξ̄[k,s] + α[k])*r[s] + ρ̂[k,s,2] - ρ̂[k,s,3] ≤ 0
#               ↑ a[s] → r[s] 만 변경

# 마찬가지로 (DL-3) → (DL-3C): a[s] → r[s]
```

**(DL-4)~(DL-7) 은 변환된 $(\tilde a, \tilde b)$ 를 사용하면 RHS 동일** (PDF §9 참조). 즉 코드에서 변수 이름은 `a`, `b` 그대로 두되, **수학적으로 이는 $\tilde a = (1-\beta)a^{\text{orig}}$ 에 해당**한다고 이해하면 됨. 결과적으로 (DL-4)~(DL-7) 은 expectation 코드와 **글자 그대로 동일**:
```julia
# 변경 없음:
# (DL-4): a[s] - b[s] ≤ q̂[s]
# (DL-5): a[s] + b[s] ≥ q̂[s]
# (DL-6): Σ b[s] ≤ 2ε̂
# (DL-7): Σ a[s] = 1
```

**새 제약 (DL-RU1), (DL-RU2)**:
```julia
# (DL-RU1): r[s] ≤ a[s] / (1-β)   ∀s
@constraint(model, [s=1:S], r[s] <= a[s] / (1-β))

# (DL-RU2): Σ r[s] = 1
@constraint(model, sum(r[s] for s) == 1)
```

**Dual obj 변경 없음**:
```julia
@objective(model, Max,
    sum(σ̂[s] for s) - φU * sum(x̄[k] * ρ̂1[s,k] for s, k) - φU * sum((1-x̄[k]) * ρ̂3[s,k] for s, k)
)
```

**Acceptance**: $\beta = 0$ 에서 (DL-RU1) 가 $r_s \leq a_s$ 가 되고 $\sum r = \sum a = 1$ 와 결합해 $r_s = a_s$ 강제 → expectation dual 정확히 회복.

### Step 4. ISP-F 변경 X

`isp_follower_optimize!` 와 그 primal 버전: **건드리지 말 것**. PDF §10.3 의 pseudocode 참조.

### Step 5. Cut 추출 변경 X

`evaluate_master_opt_cut` 의 slope formula:
```julia
slope[k] = -φU * sum(ρ̂1[s,k] - ρ̂3[s,k] for s) +
           -φU * sum(ρ̃1[s,k] - ρ̃3[s,k] for s) +
           -λU * (ρ0_1[k] - ρ0_3[k])
```
**변경 없음**. R-U 보조변수 $(\eta_\beta, \vartheta, r)$ 는 master 변수 $\bar x$ 와 곱해지지 않으므로 slope 에 등장 X (PDF §10.2 참조).

Intercept = `ZL_star + ZF_star` 도 그대로. 다만 `ZL_star` 자체가 R-U 변수가 포함된 LP 의 optimal value 라는 점만 다름 — 추출 코드 변경은 없다.

---

## Tests (작성 + 통과 필수)

### Unit Test 1: $\beta = 0$ Sanity (가장 중요)

```julia
@testset "CVaR-leader reduces to expectation at β=0" begin
    isp_data_exp = load_test_instance("small")
    isp_data_cvar = deepcopy(isp_data_exp)
    isp_data_cvar[:beta] = 0.0
    
    # Solve both with same x̄, ᾱ
    x̄ = test_x()
    ᾱ = test_α()
    
    Z_exp = solve_ISP_L_expectation(isp_data_exp, x̄, ᾱ)
    Z_cvar = solve_ISP_L_cvar(isp_data_cvar, x̄, ᾱ)
    
    @test Z_exp ≈ Z_cvar atol=1e-6
    
    # Cut slopes should also match
    cut_exp = extract_cut_expectation(...)
    cut_cvar = extract_cut_cvar(...)
    
    @test cut_exp.slope ≈ cut_cvar.slope atol=1e-6
    @test cut_exp.intercept ≈ cut_cvar.intercept atol=1e-6
end
```

### Unit Test 2: Known CVaR Value

작은 instance ($S = 4$ scenario, manually computable max-flow 값들) 에서 $\beta = 0.5$ 로 직접 계산 가능한 CVaR 값과 일치 확인.

```julia
@testset "CVaR matches manual computation at β=0.5" begin
    # MF^s values at fixed (x̄, ᾱ): [10, 20, 30, 40]
    # q̂ = [0.25, 0.25, 0.25, 0.25], β = 0.5
    # CVaR_0.5 = mean of top 50% = (30 + 40) / 2 = 35
    # (assuming TV ball ε̂ = 0)
    
    Z_cvar = solve_ISP_L_cvar(...)
    @test Z_cvar ≈ 35.0 atol=1e-4
end
```

### Unit Test 3: Primal-Dual Consistency

```julia
@testset "ISP-L primal and dual values match" begin
    Z_primal = solve_ISP_L_primal_cvar(...)
    Z_dual = solve_ISP_L_dual_cvar(...)
    @test Z_primal ≈ Z_dual atol=1e-6
end
```

### Integration Test: Benders Convergence

기존 expectation 케이스로 수렴하는 instance 를 $\beta = 0$ 으로 CVaR 코드 돌려서 동일한 optimal $x^*$ 와 동일한 obj 가 나오는지.

---

## Pitfalls (미리 인지)

1. **$\beta \to 1$ 시 numerical blow-up**: $1/(1-\beta)$ 가 폭발. $\beta \leq 0.95$ 정도로 제한하는 것을 권장. 더 큰 $\beta$ 가 필요하면 별도 numerical scaling 필요.

2. **$r_s$ 의 box bound**: $r_s \in [0, 1/(1-\beta)]$. McCormick of $\alpha_k r_s$ (만약 spatial B&B 사용 시) 에 사용. expectation 의 $r_s \in [0, 1]$ 보다 박스가 넓어 relaxation 이 약해질 수 있음.

3. **IPM dual degeneracy** (Mosek 사용 시): 기존 코드에서 이미 알려진 이슈 (`memory/ipm_mu_offset.md` 참조). R-U 변수 추가로 dual degeneracy 가 더 심해질 수 있음. `evaluate_master_opt_cut_from_primal` 가 아니라 dual ISP 기반 cut extraction 을 사용할 것 (기존 권장사항 유지).

4. **변수 이름 충돌**: $\beta$ 가 dual 측에 이미 사용되고 있을 수 있음 (예: McCormick 듀얼 이름). grep 으로 확인 후, 충돌 시 risk parameter 는 `beta_risk` 같은 이름으로.

5. **$\beta = 0$ 경계 케이스**: $1/(1-\beta) = 1$ 이라 numerically 안전하지만, 코드 logic 이 $\beta > 0$ 가정으로 작성되었다면 division 등 문제 가능. 명시적 if-else 로 처리하지 말고 그냥 일반 식으로 처리할 것 (LP solver 가 알아서 처리).

---

## Acceptance Criteria

다음 모두 통과해야 task 완료:

- [x] Pre-work 메모 정확 (코드 구조 파악)
- [x] Step 1~5 모두 구현 (ISP-L dual 기반, primal ISP-L은 미사용)
- [x] Unit Test 1 ($\beta=0$ sanity) 통과 — grid3x3 S=3 Z₀/x* 완전 일치
- [ ] Unit Test 2 (known CVaR) 통과
- [ ] Unit Test 3 (primal-dual) 통과
- [x] Integration test ($\beta=0$ Benders 수렴 동일) 통과
- [x] Pitfalls 1~5 각각에 대한 코드 코멘트 또는 핸들링 존재
- [x] $\beta = 0.3$ 에서 ISP-L solve 가 numerically 안정 (warning 없이 종료)

---

## 작업 순서 권장

1. Pre-work 메모 출력 → 사용자 확인
2. Step 1 (β 파라미터 전파) → 빌드 통과 확인
3. Step 2 (Primal) → Unit Test 1 partial pass (primal 만)
4. Step 3 (Dual) → Unit Test 1 full pass
5. Unit Test 2, 3 추가
6. Integration test
7. Pitfalls 검토 및 코멘트 추가

각 단계 완료 시 commit 권장. 한 번에 모두 변경하면 디버깅 어려움.

---

## 구현 결과 (2026-05-08)

### 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `true_dro/true_dro_data.jl` | `TrueDROData.beta` 필드, `make_true_dro_data(beta=0.0)` kwarg, assert |
| `true_dro/true_dro_build_isp_leader.jl` | `r[s]` 변수, DL-RU1/RU2, DL-2/DL-3에서 `a→r`, `update_isp_leader_alpha!` r 기반, `solve_isp_leader!` r_val 반환 |
| `true_dro/true_dro_build_subproblem.jl` | full/single: `r[s]`, `ζL=α·r`, DL-RU1/RU2. α-step LP: `r_val` 파라미터. `solve_true_dro_subproblem!` r_val 반환 |
| `true_dro/true_dro_benders.jl` | `_use_nominal_compact` 조건에 `td.beta==0.0` 추가. `last_r_val` 저장, fix/unfix `r`, α-step LP에 `r_val` 전달 |

### 테스트 결과 (grid3x3, S=3)

| Test | ε̂ | ε̃ | β | 기능 | Z₀ | Iters |
|------|-----|-----|-----|------|------|-------|
| β=0 sanity (Run1 vs Run2) | 0.3 | 0 | 0 | regression | 12.233333=12.233333 | 8 |
| β>0 smoke | 0.3 | 0 | 0.3 | CVaR 수렴 | 12.809524 | 8 |
| nominal compact | 0 | 0 | 0 | ε̂=ε̃=0 path | 11.333333 | 8 |
| full double-layer | 0.3 | 0.3 | 0 | full variant | 12.233333 | 8 |
| mini-benders | 0.3 | 0 | 0 | mini-benders path | 12.233333 | 3 |
| mini-benders + MW β=0 | 0.3 | 0 | 0 | MW cuts | 12.233333 | 3 |
| mini-benders + MW β=0.3 | 0.3 | 0 | 0.3 | MW + CVaR | 12.809524 | 3 |

β=0.3 > β=0 (12.81 > 12.23): CVaR 보수성 방향 정상.

---

## 참고자료 위치

- 수학적 derivation: `cvar_leader_merged.pdf`
- 핵심 수식: §9.3 (Form 2 코딩 가이드), §10.2 (cut formula)
- 기존 expectation 코드: 위 Pre-work 에서 파악한 파일들

이 MD 파일과 PDF 둘 다 작업 시 항상 참조할 것.
