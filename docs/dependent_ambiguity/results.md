# 종속 ambiguity set (belief 결합 δ) — 구현과 수치 확인 (2026-10-07 ~ 10-08)

이론 정리: `docs/dependent_ambiguity/dependent_ambiguity_note.tex` (PDF 같은 폴더).
모델: 𝒟_δ = {(p, p̃) ∈ D̂ × D̃ : d_TV(p, p̃) ≤ δ}, 두 TV 볼 모두 중심 q̂. δ = Inf 는 기존 rectangular, δ ≥ ε̂ + ε̃ 도 rectangular 와 같음.

## 1. 구현

| 파일 | 변경 |
|---|---|
| `nonzero_sum/nz_data.jl` | `nz_delta`, `nz_coupled`, `nz_with_delta(nd, δ)` (δ 는 `meta[:delta]`, 없으면 Inf) |
| `nonzero_sum/nz_omega.jl` | Ω 에 결합 `|a_s − d_s| ≤ cpl_s, Σ cpl_s ≤ 2δ` (global / fixed_alpha / relax 모드. belief_lp 는 belief 고정이라 제외). `nz_clean_belief` 는 (a, d) 를 q̂ 쪽으로 같은 비율로 줄여 ε̂·ε̃·δ 세 제약을 함께 만족 |
| `nonzero_sum/nz_alpha_bnb.jl` | RLT 의 Γ 에 결합 행 3 종 + cpl 상하한, 곱 변수 `Zc = α_i cpl_s`, 정적 RLT (w − Wα) cpl ≥ 0 |
| `nonzero_sum/nz_kkt_eval.jl` | KKT 독립 평가기에 같은 결합 (big-M 없음 → λᵁ exact penalty 의 독립 확인) |
| `nonzero_sum/nz_benders.jl` | OMP 추가 카디널리티 `meta[:omp_card]` (원고 실험의 source/sink cut). **기존 버그 2 개 수정** (아래) |
| `true_dro/true_dro_build_subproblem.jl` | `build_true_dro_subproblem(...; delta_couple=Inf)` (원고 zero-sum Ω 의 교차검증용) |
| `nonzero_sum/nz_paper_instances.jl` | 원고 실험 (factor_5/run_baseline_batch.jl) 과 같은 max-flow 인스턴스를 NZData 로 |
| 스크립트 | `test_nz_coupling.jl` (Ω·KKT·원고 Ω 3 중 비교), `screen_coupling.jl` (KKT 탐색), `sweep_coupling_kkt.jl` (KKT 전수 열거 δ sweep), `run_coupling_benders.jl` (δ 별 Benders), `test_coupling_alpha_bnb.jl`, `test_coupling_paper_x.jl` |

원고 zero-sum Benders (`true_dro_benders_optimize!`) 는 쓰지 않았다. mini-Benders 가 α 고정 후 ISP-L, ISP-F 를
**따로** 푸는데, 이 분리가 결합에서 정확히 깨지는 부분이라 그대로 쓰면 무효 cut 이 나온다. zero-sum 실험은
일반 코드에 `nz_from_true_dro` 로 넣어 돌렸다 (일반 Ω = 원고 Ω 는 결합 포함해 아래 2 절에서 확인).

### 고친 기존 버그 (결합과 무관, 비제로섬 Benders 공통)

1. **MW 허용오차와 menu 위반 판정이 같은 크기** (둘 다 1e-6 상대) → t₀ 가 그 사이에 끼면 같은 menu cut 이 무한 반복
   (Abilene δ=0.1 에서 x=[15,26] 에 270 회). MW 제약을 1e-9 로, 채택은 x̄ 값이 z − 1e-7 이상일 때만.
2. **같은 x̄ 세 번째 oracle 호출이면 무조건 :Stalled** → target_stop (목표값 조기 종료) 에서는 같은 x̄ 재호출이 정상인데
   멈춤 (Abilene δ=0.1 에서 16 초 만에 LB 14.80 / UB 19.64 로 종료). 시간 제한에 걸린 호출만 stall 판정에 세도록 수정.

3. **α-B&B 노드 LP `NUMERICAL_ERROR`** (결합에서만 관찰: SGB128 pair, x=∅, δ=0.1. 같은 점 δ=Inf 는 41 초 수렴).
   기존 코드는 OPTIMAL/INFEASIBLE/TIME_LIMIT 외 상태에서 `error` 로 종료. `_nz_solve_lp!` 에 재시도
   (NumericFocus=3 → barrier) 를 넣고, 설정 복원은 다음 optimize 직전에 (해를 읽은 뒤 속성을 바꾸면 JuMP 가 해를 무효로 봄).
   그래도 실패하면 부모 상한 유지 + 가장 넓은 α_i 반분할 (유효성 유지), α 고정 LP 는 "값 모름" (NaN) 처리.
   발생 횟수는 결과 Dict 의 `:numerr`, `:numerr_fail`. 수정 후 같은 점 LB 0, UB 9.8e-5 (참값 0), 1,209 노드 322 초.
   검증 실행에서 9 회 발생, 모두 재시도로 해결 (실패 0). 원인 분석은 아래 §3.

## 2b. 원래 zero-sum 코드 (`true_dro/`) 의 결합 지원 (2026-10-08)

키워드 인자 `delta_couple` (기본 `Inf` = 기존 동작). TrueDROData 는 그대로.

| 파일 | 변경 |
|---|---|
| `true_dro_data.jl` | `reduce_coupling(td, δ)` (δ ≥ ε̂+ε̃ → rect, ε̂=0 → ε̃←min(ε̃,δ), ε̃=0 → ε̂←min(ε̂,δ) 로 결합 행 없는 문제로), `td_with_eps`, `couple_clean(a, d, q, ε̂, ε̃, δ)` (결합까지 맞추는 belief 정리), `clean_r` |
| `true_dro_build_subproblem.jl` | `build_true_dro_subproblem(...; delta_couple)` (이전부터) |
| `global_bilinear_solver.jl` (α-B&B) | 결합 RLT 행 + 곱 변수 `Zc`, 하위 빌더에 δ 전달, `envs`·`heuristic` 키워드 (학술 WLS 에서 Env 재사용), 노드 LP 수치 오류 재시도 (nz 와 같음), `_fix_alpha!` 분리 |
| `belief_menu_benders.jl` | `belief_menu_benders_optimize!(...; delta_couple, bnb_envs, bnb_heuristic)`, `_clean_belief(...; delta_couple)` |
| `true_dro_benders.jl` | `true_dro_benders_optimize!(...; delta_couple, boost_envs, boost_heuristic)`. 결합이면 **mini-Benders 를 α 고정 joint LP 로** (ISP-L/ISP-F 분리는 결합에서 성립하지 않음), MW 도 joint LP 에서 (joint Pareto). α-step 의 고정 belief 는 `couple_clean`·`clean_r` 로. subproblem (전역·local·Sherali)·α-B&B boost·belief menu 에 δ 전달 |

검증 (`true_dro/test_zs_coupling_benders.jl`, `nonzero_sum/logs/zs_coupling_benders_grid3x4.log`):
원고 생성 grid3x4 (S=5, seed 1, β=0.7, ε̂=0.1, ε̃=0.3), δ=0.05 에서 결합이 binding (KKT: x=∅ 45.68 → 43.60, x=[7,13] 45.68 → 42.01,
`nonzero_sum/find_zs_binding.jl`). 원고 실험 구성 / 기본 구성 그대로, α-B&B worker 1 + 휴리스틱.

| δ | 방법 | x* | LB | UB | 시간 |
|---|---|---|---|---|---|
| 0.05 | `true_dro_benders_optimize!` (joint mini-Benders, MW, min-cut VI, inexact, α-B&B boost) | [11,15] | 29.604600 | 29.604601 | 27 s |
| 0.05 | `belief_menu_benders_optimize!` (α-B&B oracle, local_first, menu MW, min-cut VI) | [11,15] | 29.604601 | 29.604601 | 12 s |
| 0.05 | 기준: nz belief-menu (결합 검증 완료) | [11,15] | 29.604594 | 29.604601 | 17 s |
| Inf | 위 세 방법 | [11,15] | 29.604599–29.604601 | 29.604601 | 4–22 s |

KKT V*([11,15]) = 29.604602 (δ=0.05, Inf 모두). 이 인스턴스는 결합이 다른 x (∅, [7,13]) 의 값은 바꾸지만 최적점의 값은 바꾸지 않음.

## 3. α-B&B 노드 LP NUMERICAL_ERROR 원인 (2026-10-08)

**결론: 결합 행 자체가 아니라, 한 α 좌표의 상자가 극단적으로 좁아져 RLT bound-factor 행 쌍이 거의 선형종속이 된 것.
결합은 B&B 트리를 깊게 만들어 (x=∅: δ=Inf 821 노드 → δ=0.1 1,209 노드) 그 상황에 도달하게 했을 뿐.**

재현·분석 도구 (디버그 전용, 평소 실행에는 영향 없음): `nonzero_sum/diag_numerr_run.jl` (환경변수 `NZ_NUMERR_DUMP=<폴더>`
면 실패 노드 LP 3 개를 Gurobi `.mps/.bas` 로, `NZ_LP_DUMP_EVERY=N` 이면 정상 노드 LP 도 N 번째마다 저장),
`diag_numerr_analyze.jl`, `diag_numerr_kappa.jl`, `diag_numerr_inactive.jl`, `diag_numerr_box.jl`.
결합 RLT 행에는 `rltc_*`, Ω 의 결합 기본 행에는 `cpl_ad/cpl_da/cpl_sum` 이름을 붙임.
로그 `nonzero_sum/logs/numerr_analyze.log`, `numerr_kappa.log`, `numerr_inactive.log`, `numerr_box.log`.

1. 재현: SGB128 pair, x=∅, δ=0.1, 기본 구성 → 매번 같은 노드 수 (1,209) 에서 같은 9 회 발생 (결정적).
2. 결합 행은 무관: 실패 LP 3 개에 분리된 결합 bound-factor 행은 0 개 (정적 (w−Wα)·cpl ≥ 0 행 15 개뿐). 이 15 개를 지워도 결과 동일.
3. 오프라인 재현 안 됨: 저장 basis warm start, cold dual / primal / barrier 모두 OPTIMAL (같은 목적값).
   → 메모리 안의 warm-start 경로 (bound 변경 + 행 추가 직후 dual simplex) 에서만 실패.
4. 조건수: 정상 노드 LP 와 실패 노드 LP 의 최적 basis 조건수

   | LP | 행 | 비활성 행 (rhs ±1e30) | Kappa | 최소 α 상자 폭 (전체 300) |
   |---|---|---|---|---|
   | δ=Inf 정상 노드 6 개 | 1.3k–3.9k | 0.5k–2.0k | 2.6e6–6.2e7 | 0.8–68 |
   | δ=0.1 정상 노드 5 개 | 1.7k–5.5k | 0.8k–2.8k | 6.3e7–7.0e8 | 1.6–183 |
   | δ=0.1 실패 numerr_1 | 11,028 | 5,323 | 1.3e12 | 7.9e-3 |
   | δ=0.1 실패 numerr_2 | 13,017 | 6,306 | 1.0e13 | 1.6e-3 |
   | δ=0.1 실패 numerr_3 | 14,357 | 6,611 | 3.1e19 | 3.3e-5 |

5. 비활성 행 (무효 RLT 행을 rhs = −1e30 으로 끈 것) 도 무관: 삭제하거나 rhs 를 ±1e100 (Gurobi 무한) 으로 바꿔도
   KappaExact 가 그대로 (1.5e10 → 1.6e10, 3.5e11 → 3.3e11, 3.0e16 → 1.5e16).

메커니즘: 상자 [l, u] 의 bound-factor 행 (α_i − l)·g ≥ 0 과 (u − α_i)·g ≥ 0 은 u − l → 0 이면 부호만 반대인 거의 같은 행
(합 = (u − l)·g ≥ 0) 이 되어 basis 가 특이에 가까워진다. 이 인스턴스는 big-M 규모 (McCormick 승수 ρ⁰ ≈ 4e5, 목적 계수 1e5)
때문에 기본 조건수가 이미 1e6–1e8 이고, 폭 비율 (300 / 3e-5 ≈ 1e7) 이 곱해져 1e12–1e19 가 된다.
δ=Inf 도 깊이 내려가면 같은 일이 생길 수 있는 구조다.

대응: 현재 재시도 (NumericFocus 3 → barrier, 설정 복원은 다음 optimize 직전) 가 이 원인에 맞는 처방이고 실패 0 회.
근본 대책 후보 (미구현): (a) 상대 폭 하한 — 폭이 hU 의 1e-4 정도 미만인 좌표는 더 분기하지 않음 (현재 `min_width = 1e-7` 은
절대값, 상대 3e-10 까지 내려감). (b) 좁은 상자에서 그 좌표의 bound-factor 쌍 대신 다른 처리 (유효성 증명 필요).

## 2. 유효성 검증 (2026-10-08, 기본 구성, 이 PC 는 α-B&B worker 1 + 휴리스틱)

성능은 재지 않았다 (이 PC 의 학술 WLS 는 worker 2 이상이면 Error 10009 → 실험 머신에서 재야 함).
판정 기준: 독립 KKT 평가기 (big-M 없음) 의 V*_δ(x) 를 정답으로, cut(x) ≤ V*_δ(x), bound 가 정답을 포함하는가.

### α-B&B + RLT (결합 행 포함), SGB128 pair, 각 120 s (`logs/coupling_validity_alphabnb_sgb128.log`)

| x | δ | KKT V* | α-B&B LB | 포함 |
|---|---|---|---|---|
| [1] | Inf | −871.6667 | −871.6680 | ✓ |
| [1] | 0.1 | −893.7457 | −893.7514 | ✓ |
| [1,4] | Inf | −871.6667 | −871.6737 | ✓ |
| [1,4] | 0.1 | −893.7463 | −893.7479 | ✓ |

(상한은 120 s 안에 닫히지 않음 — SGB128 의 알려진 난점. 휴리스틱을 끄면 LB 가 −882, −907 로 나빠짐.)

### Benders cut, SGB128 pair δ=0.1 (결합이 binding: V*_0.1([1]) = −893.75 < V*_rect([1]) = −871.67)

`logs/coupling_validity_sgb128_d0p1.log`, `logs/coupling_validity_sgb128_std_d0p1.log`. 변형마다 10~15 분 (수렴 아님, 유효성만).

| 변형 | cut | 확인 x | max (cut − V*) / max(1,|V*|) | 판정 |
|---|---|---|---|---|
| belief-menu + MW + α-B&B (기본) | belief 4, restricted 5 (MW 9) | 5 | −1.0e-9 | 유효 |
| belief-menu, MW 끔 | belief 5, restricted 5 | 5 | +2.2e-12 | 유효 |
| 표준 (restricted cut) + MW | restricted 2 (MW 2) | 2 | 0 | 유효 |
| belief-menu + MW + Gurobi oracle | omega 1 | 5 | +7.1e-5 (x=∅) | 허용오차 (아래) |

- belief cut exactness: 만든 x̄ 에서 |Ω − LP| ≤ 3e-11.
- 기본·MW 끔 모두 LB = −925.0556 (= 610 + V*_0.1([1,2]) = 610 − 1535.0556, KKT 와 일치).
- Gurobi 의 +7.1e-5: x=∅ (참값 0) 에서 600 s 시간 제한 incumbent 가 0 보다 큼. **δ=Inf (기존 rectangular) 에서도 3.1e-5**
  (`logs/coupling_validity_aux.log`) → 결합과 무관한 기존 수치 특성 (SGB128 의 큰 big-M). 값 규모 (~900) 대비 1e-7.

### Benders cut, zero-sum grid 3×3 (원고 생성, S=5, seed 1, β=0.7) δ=0.1

`logs/coupling_validity_zs_grid3x3_d0p1.log`. 네 변형 모두 Optimal, x* = [9,10], 22.961563 (KKT 22.961564).
cut 41 개 × x 6: max 초과 +5e-15 → 전부 유효. 단, 확인한 6 개 x 모두 V*_Inf = V*_0.1 = V*_0 → 이 인스턴스에서는
결합이 binding 하지 않음 (코드 경로 점검 수준).

### 이전 정확성 확인 (2026-10-07)

- zero-sum grid 3×3 (uniform 용량, w=1): x 11 × δ 7 에서 일반 Ω = 원고 zero-sum Ω (`delta_couple`) = KKT, 최대 상대차 3.2e-7.
  δ 효과 없음 (회복 예산이 작아 follower 효과 자체가 없음). `logs/coupling_zs_grid3x3_S3.log`
- location random pooled seed 4: 39 행 Ω = KKT (OPTIMAL 37 행 차 ≤ 1.8e-6, 시간 제한 2 행은 포함). δ 효과 없음.
  `logs/coupling_loc_random_pooled_seed4_S3_gap1e-6_partial.log`
- Abilene (원고 β=0.7, ε=0.2) δ=Inf: belief-menu Benders x* = [5,27], LB 16.215183 (원고 로그 LB 와 동일), UB 16.2954
  (gap 0.49%, 원고 실행은 2.8% 에서 종료), 2,724 s. `logs/coupling_benders_abilene_b07.log` (δ=0.1 은 중단)
- SGB128 pair KKT δ sweep (부분, `logs/sweep_coupling_kkt_sgb128_pair_seed1_S3.log`): x=[1] 에서
  δ ≥ 0.15: −871.67, 0.1: −893.75, 0.05: −919.00, 0.02: −934.15, 0: −944.25. [2], [1,2], [3] 은 δ 무관.
