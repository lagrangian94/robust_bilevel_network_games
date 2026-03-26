# Nested Benders Engineering Features

`nested_benders_trust_region.jl`의 `tr_nested_benders_optimize!`에 추가된 엔지니어링 기법 4가지.

모든 하드코딩 변수는 outer_tr 블록 초반부 (line ~1337)에 위치:
```julia
mini_benders_last_n = 0      # (1) mini-benders
inexact_every_n = 3          # (2) inexact IMP / subgradient
use_subgradient = true       # (2b) inexact phase에서 IMP 대신 subgradient
subgrad_step_size = 0.1      # (2b) subgradient step size
use_best_lb_cut = true       # (3) best-LB intermediate cut
α_persistent = nothing       # (2b) subgradient용 α state
```

---

## 1. Mini-Benders (α-fixed extra cuts)

### 개념
IMP에서 수렴한 α를 고정하고, OMP를 다시 풀어 새 (x,h,λ,ψ0)를 얻은 뒤 ISP를 풀어 추가 cut을 OMP에 넣는 phase.
α 고정 → ISP가 LP/SDP로 축소 → 빠르게 여러 cut 생성.

### 하드코딩 변수
| 변수 | 의미 | 예시 |
|------|------|------|
| `mini_benders_last_n` | 마지막 N개 stage에서만 활성화 | `0`=전체, `1`=마지막만, `2`=마지막 2개 |

### 활성화 조건
```julia
mini_benders_stage_ok = inexact_every_n > 1 ||
    !outer_tr || mini_benders_last_n == 0 || B_bin_stage > n_stages - mini_benders_last_n
mini_benders_active = mini_benders && !is_inexact && leader_instances !== nothing && mini_benders_stage_ok
```

**규칙:**
- `inexact_every_n > 1`이면 → exact iteration마다 항상 mini-benders (stage 무관)
- `inexact_every_n = 1`이면 → `mini_benders_last_n` 기준으로 stage 제한
- inexact/subgradient iteration에서는 절대 실행 안 함 (`!is_inexact`)

### 끄는 법
- 함수 인자 `mini_benders=false` (기본값)
- 또는 `mini_benders_last_n`으로 stage 제한

---

## 2. Inexact Inner Solve

`inexact_every_n > 1`이면 N번에 1번만 exact IMP, 나머지는 inexact.
Inexact phase의 구현은 두 가지 중 선택:

### 2a. Inexact IMP (`use_subgradient = false`)

IMP를 loose tolerance + iteration cap으로 풀어 suboptimal α를 빠르게 구함.

| 항목 | Exact | Inexact IMP |
|------|-------|-------------|
| IMP tolerance | `tol` (1e-6) | `inexact_tol` (0.1) |
| IMP max iter | ∞ | `max_inexact_iter` (5) |
| B_conti 확장 | 수렴 후 확장 계속 | 수렴 즉시 반환 |
| ISP calls / outer iter | 2S × 4~5 ≈ 9S | 2S × 2 = 4S |

### 2b. Subgradient Heuristic (`use_subgradient = true`)

IMP를 아예 풀지 않고, ISP 1라운드(2S회)만 풀어서 μ(supergradient)로 α를 simplex projection.
α를 outer iteration에 걸쳐 누적 개선 (inexact IMP는 매번 리셋).

```
Outer iter k:
  ISP(αₖ; χ̄ₖ) 풀기 (2S회)
    → outer cut (re-solve 없이 extract)
    → μₖ = shadow_price(coupling)
  αₖ₊₁ = Proj_Δ(αₖ + γ·μₖ)

Outer iter k+1 (exact):
  tr_imp_optimize! → α* (수렴)
  α_persistent ← α* (drift 교정)
```

| 항목 | Exact IMP | Inexact IMP | Subgradient |
|------|-----------|-------------|-------------|
| ISP/outer | ~9S | ~4S | **2S** |
| α quality | Optimal | 나쁨 (매번 리셋) | 점진적 개선 (누적) |
| IMP 필요 | O | O | **X** |
| α 정보 보존 | N/A | ❌ 리셋 | ✓ across outers |
| TR 상보성 | 없음 | 없음 | ✓ (null step = α 개선) |

### 하드코딩 변수
| 변수 | 의미 | 예시 |
|------|------|------|
| `inexact_every_n` | N번에 1번 exact | `3`=3번에 1번 exact, `1`=항상 exact |
| `use_subgradient` | inexact phase 구현 선택 | `true`=subgradient, `false`=inexact IMP |
| `subgrad_step_size` | subgradient γ₀ | `0.1` |

### Inexact 판정
```julia
use_inexact = inexact_every_n > 1 && (iter % inexact_every_n != 0) && iter > 1 && !stage_just_changed
```
- 첫 iteration과 stage 전환 직후는 항상 exact

### Subgradient 함수
```julia
function project_simplex(z, w)  # O(|A| log|A|), 무시할 비용
function subgradient_alpha_step!(α_current, isp_leader_instances, isp_follower_instances, isp_data; ...)
    # Step 1: ISP 풀기 (2S회)
    # Step 2: obj, μ 수집 (ISP 부산물, 추가 비용 없음)
    # Step 3: outer cut 추출 (extract_outer_cut_from_current_isp, re-solve 없음)
    # Step 4: α_new = project_simplex(α + γ·μ, w/S)
    # Returns: tr_imp_optimize! 호환 인터페이스
```

### Exact solve 후 α 재보정
```julia
if !is_inexact && use_subgradient
    α_persistent = copy(cut_info[:α_sol])  # drift 교정
end
```

### Subgradient 시 outer cut 구성
ISP를 이미 풀었으므로 `evaluate_master_opt_cut` re-solve를 건너뛰고
`cut_info[:outer_cut_info]`를 직접 사용 → ISP 호출 추가 절약.

### 주의: imp_cuts 보존
Subgradient는 IMP를 쓰지 않으므로 `imp_cuts[:old_cuts]`를 덮어쓰면 안 됨.
덮어쓰면 다음 exact IMP가 이전 exact의 stale cuts를 삭제하지 못해 IMP underestimate → invalid UB.
```julia
if !is_subgradient
    imp_cuts[:old_cuts] = cut_info[:cuts]
end
```

### 끄는 법
- `inexact_every_n = 1` → 항상 exact (subgradient/inexact 둘 다 비활성화)
- `use_subgradient = false` → inexact IMP로 전환

---

## 3. Best-LB Intermediate Cut

### 개념
IMP inner loop에서 ISP를 여러 번 풀면서 lower bound가 갱신됨.
최종 수렴 α뿐 아니라, inner loop 도중 best LB를 달성한 시점의 ISP solution에서도 outer cut coefficients를 추출하여 OMP에 추가.
추가 solve 없이 `value.()`만 호출하므로 비용 거의 없음.

### 하드코딩 변수
| 변수 | 의미 |
|------|------|
| `use_best_lb_cut` | `true`=추가, `false`=기존대로 |

### 스냅샷 저장 (`tr_imp_optimize!`)
```julia
if subprob_obj > lower_bound + 1e-8
    result[:best_lb_outer_cut] = extract_outer_cut_from_current_isp(...)
    result[:best_lb_α] = copy(α_sol)
end
```

### OMP에 추가 (outer loop)
```julia
if use_best_lb_cut && haskey(cut_info, :best_lb_outer_cut) && cut_info[:best_lb_outer_cut] !== nothing
    best_lb_cut = add_optimality_cuts!(...; prefix="best_lb_cut", ...)
end
```

### 끄는 법
`use_best_lb_cut = false`

---

## 공통 동작 요약

### Exact / Inexact / Subgradient 비교

| 항목 | Exact | Inexact IMP | Subgradient |
|------|-------|-------------|-------------|
| IMP 풀기 | ✓ (tol=1e-6) | ✓ (tol=0.1, cap=5) | ✗ |
| ISP calls | ~9S | ~4S | 2S |
| UB 업데이트 | ✓ | ✗ | ✗ |
| SS/NS 판정 | ✓ | ✗ | ✗ |
| Mini-benders | ✓ (조건부) | ✗ | ✗ |
| Best-LB cut | ✓ | ✓ | ✗ (inner loop 없음) |
| imp_cuts 업데이트 | ✓ | ✓ | ✗ (보존) |
| outer cut 구성 | evaluate_master_opt_cut (re-solve) | evaluate_master_opt_cut (re-solve) | extract (re-solve 없음) |

### outer_tr=false 호환
- `inexact_every_n`은 outer_tr 블록 밖에서 정의 → outer_tr=false에서도 동작
- mini-benders: `!outer_tr` → `mini_benders_stage_ok = true` (stage 제한 없음)
- best-lb, subgradient: outer_tr와 무관하게 동작

---

## 벤치마크 (5x5 Grid, S=10, :mw)

| 설정 | 시간 |
|------|------|
| mini-benders ON, `inexact_every_n=1` (매번 exact) | 6540초 |
| mini-benders ON, `inexact_every_n=5` (5번에 1번 exact, exact일 때만 mini-benders) | 5500초 |
| mini-benders ON, `inexact_every_n=3` (3번에 1번 exact, exact일 때만 mini-benders, inexact일땐 subgradient l2 norm, step size 0.1) | 5000초 |
