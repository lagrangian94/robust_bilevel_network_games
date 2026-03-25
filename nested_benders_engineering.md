# Nested Benders Engineering Features

`nested_benders_trust_region.jl`의 `tr_nested_benders_optimize!`에 추가된 엔지니어링 기법 3가지.

모든 하드코딩 변수는 outer_tr 블록 초반부 (line ~1240)에 위치:
```julia
mini_benders_last_n = 0      # (1) mini-benders
inexact_every_n = 3          # (2) inexact IMP
use_best_lb_cut = true       # (3) best-LB intermediate cut
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

### 활성화 조건 (line ~1676)
```julia
mini_benders_stage_ok = inexact_every_n > 1 ||
    !outer_tr || mini_benders_last_n == 0 || B_bin_stage > n_stages - mini_benders_last_n
mini_benders_active = mini_benders && !is_inexact && leader_instances !== nothing && mini_benders_stage_ok
```

**규칙:**
- `inexact_every_n > 1`이면 → exact iteration마다 항상 mini-benders (stage 무관)
- `inexact_every_n = 1`이면 → `mini_benders_last_n` 기준으로 stage 제한
- inexact iteration에서는 절대 실행 안 함 (`!is_inexact`)
  - 이유: suboptimal α → 약한 cut → LB 개선 미미

### 호출 (line ~1680)
```julia
n_mini = alpha_fixed_benders_phase!(omp_model, omp_vars, cut_info[:α_sol],
    leader_instances, follower_instances, isp_data;
    max_iter=max_mini_benders_iter, ...)
```

### 끄는 법
- 함수 인자 `mini_benders=false` (기본값)
- 또는 `mini_benders_last_n`으로 stage 제한

---

## 2. Inexact IMP (Inner Master Problem)

### 개념
매 outer iteration마다 IMP를 global optimality까지 풀 필요 없음.
N번에 1번만 exact하게 풀고, 나머지는 loose tolerance + iteration cap으로 suboptimal α를 빠르게 구함.
Inexact α도 valid Benders cut을 생성 (약하지만 유효).

### 하드코딩 변수
| 변수 | 의미 | 예시 |
|------|------|------|
| `inexact_every_n` | N번에 1번 exact | `3`=3번에 1번 exact, `1`=항상 exact |

### Inexact 판정 (line ~1366)
```julia
use_inexact = inexact_every_n > 1 && (iter % inexact_every_n != 0) && iter > 1 && !stage_just_changed
```
- 첫 iteration(`iter==1`)과 stage 전환 직후(`stage_just_changed`)는 항상 exact

### `tr_imp_optimize!` 파라미터 (line ~284)
```julia
function tr_imp_optimize!(...; inexact=false, inexact_tol=0.1, max_inexact_iter=5)
    effective_tol = inexact ? inexact_tol : tol         # 0.1 vs 1e-6
    effective_max_iter = inexact ? max_inexact_iter : typemax(Int)  # 5 vs ∞
```

### Inexact 시 동작 차이

| 항목 | Exact | Inexact |
|------|-------|---------|
| IMP tolerance | `tol` (1e-6) | `inexact_tol` (0.1) |
| IMP max iter | ∞ | `max_inexact_iter` (5) |
| B_conti 확장 | 수렴 후 확장 계속 | 수렴 즉시 반환 (확장 없음) |
| UB 업데이트 | `upper_bound = min(upper_bound, subprob_obj)` | 스킵 |
| SS/NS 판정 | 정상 수행 | 스킵 (center 유지) |
| Mini-benders | 활성화 가능 | 비활성화 |

### Inexact 조기 반환 (line ~360)
```julia
if inexact && inner_converged
    # B_conti 확장 없이 즉시 반환
    return (:OptimalityCut, result)
end
```
이 early return이 없으면 B_conti 확장 branch에 들어가서 iteration을 더 소모함.

### 끄는 법
`inexact_every_n = 1`

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

### 스냅샷 저장 (`tr_imp_optimize!`, line ~352)
```julia
if subprob_obj > lower_bound + 1e-8
    result[:best_lb_outer_cut] = extract_outer_cut_from_current_isp(
        isp_leader_instances, isp_follower_instances, S)
    result[:best_lb_α] = copy(α_sol)
end
```
- ISP를 푼 직후, LB가 갱신될 때마다 현재 solution의 cut coefficients를 덮어씀
- 최종 수렴 시점의 α와 다른 α에서의 cut이 저장될 수 있음

### `extract_outer_cut_from_current_isp` (line ~574)
```julia
function extract_outer_cut_from_current_isp(isp_leader_instances, isp_follower_instances, S)
    # value.()로 Uhat1, Utilde1, Uhat3, Utilde3, Ztilde1_3, βtilde1_1, βtilde1_3, intercept 추출
    # re-solve 없음 → 비용 ≈ 0
    return Dict(:Uhat1=>..., :Utilde1=>..., ..., :intercept_l=>..., :intercept_f=>...)
end
```

### OMP에 추가 (outer loop, line ~1622)
```julia
if use_best_lb_cut && haskey(cut_info, :best_lb_outer_cut) && cut_info[:best_lb_outer_cut] !== nothing
    best_lb_cut = add_optimality_cuts!(omp_model, omp_vars, best_lb_cut_info, ...;
        prefix="best_lb_cut", result_cuts=result[:cuts])
end
```

### 끄는 법
`use_best_lb_cut = false`

---

## 상호작용 요약

| 조합 | 동작 |
|------|------|
| exact + mini-benders + best-lb | cut 3종: opt_cut + best_lb_cut + mini-benders cuts |
| exact + mini-benders only | cut 2종: opt_cut + mini-benders cuts |
| inexact | cut 1종: opt_cut만 (UB/SS/mini-benders 모두 스킵) |
| inexact + best-lb | cut 2종: opt_cut + best_lb_cut (inexact에서도 best-lb는 추가됨) |

---

## 벤치마크 (5x5 Grid, S=10, :mw)

| 설정 | 시간 |
|------|------|
| mini-benders ON, `inexact_every_n=1` (매번 exact) | 6540초 |
| mini-benders ON, `inexact_every_n=5` (5번에 1번 exact, exact일 때만 mini-benders) | 5500초 |

---

### outer_tr=false 호환
- `inexact_every_n`은 outer_tr 블록 밖에서 정의 → outer_tr=false에서도 동작
- mini-benders: `!outer_tr` → `mini_benders_stage_ok = true` (stage 제한 없음)
- best-lb: outer_tr와 무관하게 동작
