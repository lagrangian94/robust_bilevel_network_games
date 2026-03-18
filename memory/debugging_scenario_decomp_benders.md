# Scenario-Decomposed Benders 디버깅 기록

## 알고리즘 구조

| | Strict | **Scenario-Decomposed** | Nested |
|---|---|---|---|
| 구조 | OMP → 1 OSP(S개) | OMP → IMP → S × OSP(1개) | OMP → IMP → S × (ISP_l + ISP_f) |
| α 공유 | 전체 공유 | IMP에서 공유 (inner Benders) | IMP에서 공유 |
| 병렬화 | 불가 | S개 병렬 | S개 병렬 |
| Inner iter | 없음 | 있음 (IMP ↔ OSP) | 있음 (IMP ↔ ISP) |

- 함수: `scenario_benders_optimize!` in `strict_benders.jl`
- Inner loop: `build_imp` → `osp_inner_optimize!` per scenario
- Inner Benders cut: `t_1_l[s] <= intercept_l + μhat' α`, `t_1_f[s] <= intercept_f + μtilde' α`

## 검증 완료 사항

1. **hat/tilde 독립성**: α 고정 시 hat(leader)과 tilde(follower)는 완전 독립 (SDP 별개). μhat = ∂V_l/∂α, μtilde = ∂V_f/∂α 정확. 별도 inner cut 유효.
2. **Joint α → 분해 일치**: joint OSP의 α*를 per-scenario OSP에 고정 → sum = joint obj (Step 4, diff < 1e-4)
3. **고정 (x*,h*,λ*,ψ0*)에서 inner loop 정상 수렴**: optimal 해에서 inner Benders → 8 iter, 1.1251 ≈ joint OSP (Step 5)
4. **Free-α 분해 = relaxation**: per-scenario free-α sum (1.313) > joint (1.125) — 기대대로

## 발견된 버그: Inner Benders가 특정 (x,h,λ,ψ0)에서 잘못 수렴

### 현상
Abilene (S=2, MW cuts)에서 `scenario_benders_optimize!` 실행 시:
- Outer iter 31: decomposed=1.117, joint=1.126, **mismatch=0.009**
- Outer iter 44: decomposed=0.617, joint=1.126, **mismatch=0.510**
- `upper_bound = min(upper_bound, subprob_obj)` 에서 0.617이 UB를 영구적으로 끌어내림
- 최종: SD UB=0.617, SD LB=1.103 (LB > UB, 비정상)

### Inner loop 상태 (iter 44)
- inner_iter = 6, gap = 2.9e-5 → "수렴" 판정
- converged α: arc 26에 1.5 집중 (나머지 ≈ 0) — **극단적 corner solution**
- joint optimal α: 여러 arc에 분산 (arcs 5, 7, 15, 26)
- x_sol = arcs 8, 20, 28 interdicted (비최적 x)

### 근본 원인 (미확정)
Inner Benders cut이 **invalid** (overestimator가 아니라 underestimator). IMP가 optimal α 영역을 차단하여 suboptimal α로 수렴.

가능한 원인:
1. **Shadow price 수치 오류**: 극단적 α (한 arc에 집중)에서 Mosek의 coupling constraint shadow price가 부정확
2. **Slater's condition 위반**: α ≈ 0인 arc에서 `βhat2 <= 0` (tight bound) → conic 문제의 strong duality 불성립 → shadow price ≠ supergradient
3. **Intercept 계산 오류**: V_l, V_f decomposition이 특정 (x,h,λ,ψ0)에서 불일치

### UB drop 감지 로그
`strict_benders.jl`의 `scenario_benders_optimize!`에 UB drop 시 joint OSP 검증 코드 추가됨:
```julia
if subprob_obj < upper_bound - 1e-3
    # joint OSP를 같은 (x,h,λ,ψ0)에서 풀어서 비교
    ...
    println("  MISMATCH = $(verify_cc[:obj_val] - subprob_obj)")
end
```

## 디버그 도구
- `debug_sd_benders.jl`: 6-step 진단 스크립트 (strict 해 하드코딩)
- Steps 2-5: inner loop 격리 검증 (정상)
- Step 6: 전체 `scenario_benders_optimize!` 실행 (버그 재현)

## TODO
- [ ] UB drop 시 각 inner Benders cut을 joint α*에서 평가 → `cut(joint_α*) vs Q_s(joint_α*)` 비교
- [ ] cut이 underestimate하면 shadow price 오류 확인
- [ ] 극단적 α에서 Mosek solver status, duality gap 확인
- [ ] Grid network (정상 동작)과 Abilene (버그)의 inner loop 차이 비교

## 관련 수정 이력
- `build_full_model.jl`: constraint (14l) per-scenario bound 수정 (Full Model 2x 문제 해결)
- `nested_benders_trust_region.jl`: `tr_imp_optimize!`에서 R, r_dict 등 global → local 변수 추출 (parallel 에러 수정)
- `strict_benders.jl`: combined cut 시도 후 revert (hat/tilde 독립 확인)
