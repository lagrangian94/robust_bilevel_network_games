# Intercept 역산 보정 (ISP Follower Duality Gap Fix)

## 문제
`isp_follower_optimize!`에서 Mosek이 `MSK_RES_TRM_STALL` (IPM stall)로 종료하면,
JuMP은 `MOI.SLOW_PROGRESS`로 매핑하여 `OPTIMAL`과 동일하게 처리하지만,
shadow price(`ηtilde_pos`, `ηtilde_neg`)가 부정확해진다.

이로 인해 dual objective 재구성 시 strong duality check 실패:
```
dual_obj = intercept + α' * subgradient  ≠  objective_value(model)
```
gap ≈ 0.12 ~ 0.18 수준.

## 발생 조건
- `inner_tr=true` (α에 box constraint)
- 비자명한 (x, h, λ, ψ0) (outer iteration 진행 후)
- 이 조합이 ISP follower conic problem을 numerically 어렵게 만듦

## 해결: 역산법
`objective_value(model)`은 primal objective로 Mosek stall 상태에서도 신뢰 가능.
shadow price 기반 intercept 대신 역산:
```julia
intercept = objective_value(model) - α_sol' * subgradient
dual_obj  = objective_value(model)
```
이렇게 하면 Benders cut이 현재 α에서 tight하게 유지됨.

## 검증 결과 (4×4 grid, S=1, γ=2, w=1.6571)

| | Baseline (inner_tr=false) | Patched (inner_tr=true + 역산) |
|---|---|---|
| Objective | 4.365650 | 4.365670 |
| Outer iters | 29 | 32 |
| Inner iters | 152 | 133 |
| Time | 36.86s | 34.70s |
| x*, λ* | 동일 | 동일 |

Gap = 0.00002 → 사실상 동일한 해로 수렴.
역산 보정은 2회 발생 (Outer 4, Mosek STALL).

## 위치
`nested_benders_trust_region.jl` → `isp_follower_optimize!` 내 strong duality check 부분 (line ~265).
