# α-공간 B&B: Ω subproblem 의 큰 S dual bound (작업 노트, 2026-10-02~03)

목표: JOC 규모 (S≈200) 에서 Ω subproblem V*(x̄) 의 tight 한 dual bound (UB) 와 좋은 해 (LB).
인스턴스: Abilene, K=30, β=0.4, ε̂=ε̃=0.2, λU=10, x̄ ∈ {∅, [8,11]}, 시간 300s (별도 표기 제외).
모든 코드는 `true_dro/` 의 신규 파일. 기존 파일 미수정.

## 1. 핵심 관찰

- Ω 의 bilinear 항은 전부 α_k × (r_s | a_s | d_s). α 고정 → LP. α 상자만 줄이면 McCormick/RLT 가 exact 로 수렴.
  → α (K=30 차원, S 와 무관) 만 분기하는 spatial B&B.
- McCormick (+ Σy=1 RRLT) 만으로는 root gap 이 S 에 비례해 커짐: belief 상한 r_max ≈ 0.68 ≫ 1/S.
  TV 볼 제약 (a−b≤q̂, Σb≤2ε, r≤a/(1−β)) × 상자 인수 (α_k−l_k), (u_k−α_k) 의 level-1 RLT 를 넣으면
  root gap 이 S 와 무관하게 3~6%.
- RLT 행은 18만 개 (S=200) 지만 root 에서 dual≠0 은 ~2%. 절반 이상이 degenerate 하게 tight.

## 2. 최종 비교 (300s, [LB, UB], 괄호 = 노드 수)

| S | x̄ | Gurobi global (Ω) | α-B&B RLT 분리 (순차) | **α-B&B 병렬 12 workers** | **병렬 + global LB 병행** |
|---|---|---|---|---|---|
| 50 | ∅ | [21.908, 22.326] | [21.905, 21.980] (249) | [21.908, **21.924**] (938) | [21.908, 21.925] |
| 50 | [8,11] | [15.833, 16.817] | [15.615, 15.970] (175) | [15.838, **15.878**] (629) | [**15.843**, 15.880] |
| 200 | ∅ | [22.698, 24.003] | [22.524, 23.131] (18) | [22.559, 23.131] (33) | [**22.698**, **23.131**] gap 1.9% |
| 200 | [8,11] | [17.365, 19.834] | [16.915, 17.602] (14) | [16.692, **17.509**] (21) | [**17.365**, 17.612] gap 1.4% |

- S=50: 병렬 α-B&B 가 gap 0.07~0.26% 로 global (1.9~6.2%) 을 압도.
- S=200: UB 는 global 대비 크게 tight (23.13 vs 24.00, 17.5 vs 19.8). LB 는 global 이 강함 → 병행 시 gap 1.4~1.9%.
- S=200 은 노드 LP (RLT 분리, ~3.6만+α 행) 가 노드당 15~45s 라 300s 에 노드 수십 개가 한계.

### 1200s (S=200)

| x̄ | Gurobi global 단독 | **병렬 α-B&B + global LB 병행** |
|---|---|---|
| ∅ | [22.706, 23.762] gap 4.65% | [22.698, **22.841**] gap **0.63%** (노드 144) |
| [8,11] | [17.365, 18.798] gap 8.25% | [**17.401**, **17.422**] gap **0.12%** (노드 90) |

- 초반 ~300s 는 root + worker 별 cold start 로 노드가 적고, 이후 warm 상태에서 노드 처리가 빨라지며 gap 이 빠르게 닫힘.
- S=200 [8,11] 에서는 B&B 가 찾은 LB (17.401) 가 global (17.365) 보다 좋음.

### Gurobi global 없이: 병렬 α-B&B + 로컬 NLP primal heuristic (300s)

로컬 heuristic = Gurobi 로컬 (OptimalityTarget=1, ρ≤10) 1회 → 상한 높은 노드 α̂ 에서 Ipopt 반복.
로컬 해의 α 를 고정해 정확한 LP 로 재평가한 값만 LB 로 사용 (엄밀히 valid).
단독 측정: 로컬 1회로 global 300s LB 와 같거나 더 좋음 (S=200 ∅ Ipopt 22.707 > global 22.698).

| S | x̄ | 병렬 + global LB 병행 | **병렬 + 로컬 heuristic** |
|---|---|---|---|
| 50 | ∅ | [21.908, 21.925] | [21.908, 21.925] gap 0.08% |
| 50 | [8,11] | [15.843, 15.880] | [15.839, 15.880] gap 0.26% |
| 200 | ∅ | [22.698, 23.131] | [22.698, 23.131] gap 1.9% |
| 200 | [8,11] | [17.365, 17.612] | [17.364, 17.644] gap 1.6% |

→ Gurobi global 병행 없이 같은 수준. 논문에는 "α-B&B + 로컬 NLP primal heuristic" 으로 기술 가능.
(S=200 은 진행 중 노드 마무리로 ~340s 에 종료 — 시간 엄수 처리 필요)

## 3. 시도한 것과 결과

| 시도 | 결과 | 원인 / 메모 |
|---|---|---|
| α-B&B `:basic` (McCormick+RRLT) | global 에 짐 | root gap ∝ S |
| α-B&B `:full` (TV 볼 RLT, 행 전부) | S≥20 에서 global 보다 tight UB | 노드 LP 21.6만 행, 47s/노드 (S=200) |
| 라그랑지 분해 + subgradient | 실패 | L(ν,μ) 유한 영역이 얇음 (dual 변수 상한 없음) → 모든 step 이 unbounded |
| Dantzig–Wolfe (α 복사) | 실패 | K·S 복사 행 consensus → master 정체 (60 iter 동안 값 불변) |
| Gurobi global + root RLT 주입 | UB 개선, S=200 LB 붕괴 | 모델 비대, local cut 불가 |
| PWL (α 구간 binary) | 실패 | 모델 25만 행 (S=50) |
| **RLT 분리 (cut separation)** | root 값 동일, 행 7~12% | **행 삭제 금지** (삭제 → basis 손실, 13k iter). 무효 행은 rhs −∞ |
| 정체 시 라운드 중단 + diving | 노드 1.7~2배 | |
| 부모 basis 복원 (VBasis/CBasis) | 효과 없음 | Gurobi 가 무시 (Presolve=0 으로 일부만) — 형제 전환 ~25s 유지 |
| Zhen et al. (IJOC 2022) binding-scenario + mountain climbing | LB 일부 개선, global 미달 | LDR=RLT (Theorem 1) 이므로 bound 쪽은 이미 보유 |
| OBBT (root) | 고정되는 α 없음, S=50 일부 이득, S=200 손해 | OBBT LP 1개 ~12s (S=200) |
| 단체 RLT (w−Σα)·g ≥ 0 | 무시할 수준 | root 18.026→18.006 |
| Gurobi α BranchPriority | 무시됨 (root RLT 유무 모두 UB 6자리까지 동일) | 연속 변수 spatial 분기에 미적용 |
| Gurobi + root RLT user cut callback (`global_rlt_cb.jl`) | S=50·S=200 ∅ 에서 전체 주입보다 UB 좋음, LB 유지. S=200 [8,11] 은 root cut 루프에 300s 소진, 해 없음 | UB S=50 22.022/16.022, S=200 23.257/18.014 |
| 위 + root cut 5라운드 상한 + MIP start (mountain climbing) | S=200: ∅ [22.532, 23.459], [8,11] [17.001, 17.967]. 해는 생기지만 LB 는 시작해에서 개선 안 됨, 노드 4~9개 | Gurobi 자체 노드 처리가 S=200 에서 너무 느림 |
| **노드 병렬 (12 workers)** | S=50 노드 3~4배, gap 0.07~0.26% | worker 별 Gurobi env, root 행 공유 |
| **병렬 + global LB 병행** | S=200 gap 1.4~1.9% (한 번 실행) | Gurobi MIPSOL callback 으로 LB 공유 |

## 4. 별도 발견: Ω 의 λU 는 벌점 완화

λU=10 인 Ω 는 follower 비최적 α 를 λU×gap 만 벌점 → V* 과대평가 가능
(S=7 [8,11]: Ω 18.188 vs 진짜 18.176; λU=100 이면 18.176). 위 비교는 모두 같은 Ω (λU=10) 기준.

## 5. 파일

| 파일 | 내용 |
|---|---|
| `true_dro/alpha_bnb.jl` | α-B&B (McCormick/full RLT, 행 전부) |
| `true_dro/alpha_sep.jl` | RLT 분리 α-B&B, diving, 정체 중단, OBBT, 단체 RLT 옵션 |
| `true_dro/alpha_par.jl` | 병렬 α-B&B (+ global LB 병행) |
| `true_dro/alpha_heur.jl` | binding-scenario + mountain climbing LB heuristic |
| `true_dro/alpha_lagrange.jl`, `alpha_dw.jl` | 라그랑지 / DW 분해 (실패, 기록용) |
| `true_dro/global_rlt.jl` | Gurobi global + root RLT, PWL (실패, 기록용) |
| `true_dro/test_alpha_bnb.jl` | 테스트 스크립트 |

## 6. 다음 후보

- S=200 노드 LP 비용 절감: 형제 전환 비용 (~25s) 이 최대 병목. worker 별 subtree 할당 (전환 최소화).
- LB: 병행 global 이 가장 효과적. Benders 통합 시 "global(LB) ∥ α-B&B(UB)" 구성.
- λU 정확성 문제 별도 정리 (논문 재정식화 exactness).


## 추가 (2026-10-04): 최종판 `global_bilinear_solver.jl` 수정 사항
- 보고 UB 에 허용오차 (rel_gap) 로 가지치기한 노드의 상한 포함 (이전엔 최대 rel_gap 만큼 과소 → Benders 에서 LB > UB).
- `julia -t N,1` (interactive 스레드) 필수: 스레드 1 을 worker 가 점유하면 메인 루프·sleep 타이머가 멈춰 시간 제한이 깨짐.
  시작 시 `Threads.nthreads(:interactive) ≥ 1` 확인, worker 도 종료 조건 직접 판정.
- `target` 인자: LB ≥ target 또는 UB ≤ target 이면 종료 (Benders 판정용).
- Benders 통합 결과는 `docs/benders_belief_menu_design.md` §8~10.
