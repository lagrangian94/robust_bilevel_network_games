# Belief-menu Benders 실험 리포트 (2026-10-03 ~ 10-06)

이 문서는 **측정 결과**만 다룬다. 정식화·명제·증명 (Ω, belief 고정 restriction, 충분 menu, NP-hard 평가 상한,
설계 B 의 big-M 없는 belief 블록 C&CG) 은 이론 문서로 옮겼다.

- 이론 문서: `docs/belief_menu_benders_theory.html` (게시본: https://claude.ai/artifact/YUEKzSdhhDxewboxdUpa9H)
- 코드: `true_dro/belief_menu_benders.jl` (설계 A), `true_dro/true_dro_benders.jl` (standard, `belief_menu=true` 가 하이브리드),
  `true_dro/global_bilinear_solver.jl` (α-B&B), 측정 스크립트 `true_dro/archive_alpha_bnb/run_improve_*.jl`

## 용어

| 이름 | 뜻 |
|---|---|
| standard | 기존 Benders (`true_dro_benders.jl`) + mini-Benders (α 고정 LP cut, MW) + α-B&B boost |
| A | 설계 A: belief-menu Benders (belief 고정 LP cut, menu 수렴 시에만 전역 oracle) |
| A2 | A + 로컬 해 우선 oracle (`local_first`) + 반복 x̄ 시간 제한 boost (`repeat_boost`). 현재 기본값 |
| 하이브리드 | standard + menu cut (`belief_menu=true`) |
| 설계 B | belief 를 LP 쌍대 블록째 master 에 넣는 C&CG (이론 문서 §10, 미구현) |
| menu 크기 \|M\| | 한 실행에서 모은 belief 수 = 최적값 인증에 쓰인 (충분) menu 의 크기. cover 정리의 \|Θ*\| 가 아님 (이론 문서 §4) |

공통 설정: β=0.4, ε̂=ε̃=0.2, tol 0.5%, sub_time_limit 15s, mincut VI, ss cut, inexact. `julia -t 14,1`.

## 0. 사전 진단: α* 다양성 (기존 로그)

| 로그 (Abilene, double, β=0.4) | 값 |
|---|---|
| Benders 반복 | 299 (boost 6회, TIME_LIMIT 37회) |
| 고유 α* (소수 2자리 / 1자리 / 정수 반올림) | 241 / 232 / 216 |
| 고유 지지집합 (α_k > 0 인 arc 집합) | 145 |
| α* 지지집합 크기 | 2~9 (평균 4.7) |
| subproblem 시간 | 합 18,661s, 중앙값 0.2s, 상위 10% 15s, 최대 3,609s (boost) |

→ α 쪽 menu 는 작지 않다 (반복의 70~80% 가 새 α*). belief 쪽 \|M\| ≤ 11 (아래 측정) 과 대비되는 비대칭 (이론 문서 §11).

(§1~5 의 설계 서술은 이론 문서로 옮겼다. 이전 판은 git 이력 커밋 25197ff 참고. 절 번호는 기존 참조를 유지하려고 그대로 둔다.)

## 6. 첫 실험 결과 (2026-10-03, Abilene S=10, double, β=0.4, ε=0.2, tol 5e-3 — factor_5 배치와 동일 설정)

| 구성 | wall | 반복 | 전역 평가 | 최종 |
|---|---|---|---|---|
| ① 설계 A + α-B&B oracle | 647.7s | 154 | 6회 (2~7s ×5, **600.5s ×1**: x̄ 하나에서 gap 1.6% 미종료) | 14.122941, x*=[5,27] |
| ② 설계 A + Gurobi oracle | 1,306.1s | 146 | 7회 (15s ×6, boost 1,214s ×1) | 동일 |
| ③ 기존 Benders + α-B&B boost | **213.9s** | 32 | boost 1회 **7.0s** (exact) | 동일 |
| ④ 기존 Benders + Gurobi boost | 1,877.8s | 32 | boost 1회 1,147.7s | 동일 |
| (참고) 이전 배치 로그 | 1,521s | 37 | boost 912.8s | 동일 |

- 가장 큰 효과: 기존 Benders 의 boost 만 α-B&B 로 교체 (③ vs ④: 8.8배, boost 1,148s → 7s).
- 설계 A 는 전역 평가 횟수를 6~7회로 줄였고 Gurobi oracle 끼리는 기존보다 빠름 (② 1,306s vs ④ 1,878s).
  그러나 α-B&B oracle 에서는 ③ 보다 느림: menu 가 수렴한 x̄ (최적이 아닌 점 포함) 에서 oracle 을 목표 gap 0.5% 로
  풀다가 한 번 600s 를 다 씀.
- 개선안: oracle 에 목표값 기반 조기 종료 — incumbent > t₀ + tol (새 belief 발견 → cut) 이거나
  상한 ≤ t₀ + tol (menu 가 x̄ 에서 exact) 이면 즉시 중단. 최적이 아닌 x̄ 에서는 정확한 값이 필요 없음.


## 7. 목표값 조기 종료 + S=50 (2026-10-03)

oracle 목표값: incumbent ≥ t₀ + ½·tol·|t₀| (새 belief) 또는 상한 ≤ 같은 값 (menu exact) 이면 즉시 중단.
Gurobi: BestObjStop/BestBdStop, α-B&B: `global_bilinear_solve(...; target)`. menu belief 는 실현 가능하게 정리 (`_clean_belief`).

| S | 구성 | 상태 | 시간 | [LB, UB] | 전역 평가 |
|---|---|---|---|---|---|
| 10 | 기존 + Gurobi boost | 최적 | 1,877.8s | 14.1229 | boost 1,147.7s |
| 10 | 기존 + α-B&B boost | 최적 | 213.9s | 14.1229 | boost 7.0s |
| 10 | 설계 A + α-B&B (조기 종료 없음) | 최적 | 647.7s | 14.1229 | 6회 |
| 10 | **설계 A + α-B&B (목표값 조기 종료)** | 최적 | **51.6s** | 14.1229 | 7회 |
| 10 | 설계 A + Gurobi (조기 종료 없음) | 최적 | 1,306.1s | 14.1229 | 7회 |
| 10 | 설계 A + Gurobi (목표값 조기 종료) | 최적 | 1,974.7s | 14.1229 | 16회 |
| 50 | 기존 + Gurobi boost | **2h 미수렴** | 7,721s | [14.311, 14.731] gap 2.9% | boost 3,600s × 2 |
| 50 | **기존 + α-B&B boost** | 최적 | **1,206.7s** | 14.3094 | boost 최대 769.8s |
| 50 | 설계 A + α-B&B (목표값) | 최적 | 2,683.0s | 14.3094 | 12회 (최적점에서 600+603+730s) |
| 50 | 설계 A + Gurobi (목표값) | **정체 종료** | 7,445s | [14.311, 14.892] gap 4.1% | 12회 |

- S=50 에서 Gurobi 를 전역 평가로 쓰는 두 구성은 2시간 내 미수렴, α-B&B 를 쓰는 두 구성은 수렴.
- 설계 A + α-B&B: S=10 최선 (51.6s), S=50 에서는 기존+α-B&B boost 보다 느림 — 목표값 (t₀ + 0.25%) 이
  수렴 기준 (0.5%) 보다 엄격해 최적점에서 상한 증명에 시간 소모. → 목표값을 t₀ + tol·|t₀| 로 맞추는 것이 다음 수정.
- **주의 (수치):** S=50 의 수렴한 실행에서 LB 가 UB 보다 약간 큼 (14.3106 vs 14.3094, 상대 ~1e-4).
  cut (허용오차 있는 LP/barrier 해에서 계산) 또는 α-B&B 상한의 수치 오차로 추정. 원인 확인 필요.


## 8. 다중 네트워크 비교와 결론 (2026-10-04)

설정: β=0.4, ε=0.2, tol 0.5%, factor_5 배치와 동일 (sub_time_limit 15s, mini-Benders, mw, mincut, inexact, ss cut).
α-B&B = `global_bilinear_solver.jl` (julia -t 14,1). 모든 비교에서 세 방식의 x* 동일.

### 전역 평가 solver (기존 Benders 구조)
| 인스턴스 | Gurobi boost | α-B&B boost |
|---|---|---|
| Abilene S=10 | 1,878s (boost 1,148s) | **214s** (boost 7s) |
| Abilene S=50 | 2시간 미수렴, gap 2.9% | **1,226s** |

### S=10 (최종 재측정, 단독 실행)
| 네트워크 | 설계 A + α-B&B | 기존 + α-B&B boost |
|---|---|---|
| Abilene | **46s** | 189s |
| Polska | **31s** | 85s |
| grid 5×5 | **99s** | 267s |
| Nobel-US | **91s** | 341s |
| Sioux Falls | 872s (반복 756) | **566s** |

### S=50
| 네트워크 | 설계 A + α-B&B | 기존 + α-B&B boost | 하이브리드 (기존 + menu cut) |
|---|---|---|---|
| Abilene | 2,035s | **1,226s** | 2,079s |
| Polska | 1,492s | **564s** | 677s |
| grid 5×5 | 1,451s | **1,018s** | 1,089s |

(S=50 Polska/grid 와 S=200 은 interactive 스레드 수정 전 측정 — 두 방식 모두 같은 영향.)

### S=200 Abilene
- 설계 A: 2시간 제한, [15.160, 15.259] gap 0.65%.  기존 Benders: 측정 중단 (미완).

### 결론
- **기본: 기존 Benders + α-B&B boost.** S 가 커질수록 유리, 구조 변경은 boost 하나.
- 설계 A: S≈10 에서 5개 중 4개 네트워크 2.7~4.1배 빠름 → 소규모 S 옵션으로 유지.
- 하이브리드 (`belief_menu=true`): 반복 수가 거의 줄지 않음 (mini-Benders 가 비슷한 역할) → 이득 없음.

## 9. 설계 A 가 S 에 스케일하지 않는 이유 (Abilene, 시간 분해)

| 구성 요소 | 설계 A S=10 | 설계 A S=50 | 기존 S=50 |
|---|---|---|---|
| 전체 | 52s | 2,035s | 1,226s |
| menu LP 단계 | 22s | 176s (9%) | — |
| 최적이 아닌 x̄ 의 α-B&B oracle | 2~7s × 6 | 33~123s × 9 = 560s | (15s Gurobi·로컬 해, 일반 반복 전체 436s) |
| 최적점 증명 | 7s × 1 | 604s (600s 제한 실패) + 694s = 1,298s | boost 1회 790s |

1. α-B&B 호출 1회 비용이 S 에 따라 급증 (노드 LP O(K·S), 호출마다 LP 모델 13개 재생성 + root 부터 재시작).
   설계 A 는 이를 ~10회, 기존은 최적 근처 1회.
2. 목표값 조기 종료는 나쁜 x̄ 에선 빠르지만, 값이 t₀ 에 가까운 좋은 x̄ 에선 "t₀+tol 보다 큰 값 없음" 증명이 필요 → 최적 근처 x̄ 들에서 반복.
3. 같은 x̄ 에서 시간 제한 실패 후 재호출 시 트리를 처음부터 다시 만듦 (604s 낭비).

### 개선안 (미구현)
| 개선 | 내용 | 겨냥 |
|---|---|---|
| oracle 계층화 | 먼저 로컬 해 (Gurobi 로컬 / Ipopt) 로 t₀ 보다 큰 belief 탐색, 실패 시에만 α-B&B | 1 |
| 모델 재사용 | Ω 제약은 x̄ 무관 → worker LP 를 Benders 전체에서 한 번만 만들고 목적만 갱신, 같은 x̄ 에선 트리 이어 쓰기 | 1, 3 |
| 시간 제한 조정 | 같은 x̄ 재등장 시 처음부터 긴 제한 | 3 |
이 개선들을 넣으면 설계 A 는 "기존 Benders + belief menu" 에 수렴 — S 가 크면 두 설계가 같은 지점으로 모인다.

## 10. 이 과정에서 고친 버그
| 문제 | 영향 | 수정 |
|---|---|---|
| α-B&B UB 에 허용오차로 가지치기한 노드 상한 누락 | 보고 UB 가 최대 0.5% 과소 → Benders LB > UB | `pruned_ub` 추적해 UB 에 포함 |
| x̄ 반올림의 −0.0 | 같은 x̄ 를 다른 Dict 키로 인식, oracle 시간 제한 boost 미작동 | x̄ 를 0/1 로 정리 |
| 스레드 1 점유로 메인 루프·sleep 타이머 정지 | α-B&B 시간 제한 미준수, 병렬 효율 저하 | `julia -t N,1` (interactive 스레드) 필수 + 시작 시 확인, worker 측 종료 판정·yield |
| barrier 해의 belief 허용오차 | menu LP infeasible | `_clean_belief` 로 실현 가능한 점으로 정리 |
| Gurobi OBJECTIVE_LIMIT / LOCALLY_SOLVED 상태 | oracle 오류 | 전용 `_oracle_solve!`, bound 없으면 +∞ |

## 11. 개선안 3·1 구현 (2026-10-05)

`belief_menu_benders_optimize!` 옵션 (둘 다 기본값으로 켜짐):

| 옵션 | 기본 | 동작 |
|---|---|---|
| `repeat_boost` | `true` | oracle 을 이미 부른 x̄ 가 다시 나오면 처음부터 `boost_time_limit` (3,600s). 600s 실패 후 처음부터 재시작하는 낭비 제거 (§9 의 604s) |
| `local_first` | `oracle == :alpha_bnb` | α-B&B 전에 Ipopt 로컬 해 (출발점: menu 최선 LP 해의 α, 균등 α; 각 `local_time`=60s 이내) → α 고정 LP 로 정확히 재평가해 목표값 이상이면 그 해로 cut·belief, α-B&B 생략. 로컬 해는 상한이 없으므로 UB 는 α-B&B 에서만 갱신 |

개선 2 (worker LP·트리 재사용) 는 아래 측정 결과를 보고 결정.

### 첫 확인 (Abilene S=10)
| | 개선 전 A | A2 (개선 3+1) |
|---|---|---|
| wall | 46.0s | 45.8s |
| oracle 호출 | 7 (모두 α-B&B) | 7 (로컬 6 / 17s, α-B&B 1 / 9s = 최적점 증명) |
| 해 | [5, 27], LB 14.122941 | 동일 |

### 진행 중인 측정 (`true_dro/archive_alpha_bnb/run_improve_batch.jl`, `julia -t 14,1`, 순차)
1. S=10 × 5개 네트워크: A2 (기준선 A·standard 는 §8)
2. S=50 × Abilene·Polska·grid 5×5: A2, standard, A (개선 전) — 같은 조건에서 재측정
3. S=200 Abilene: A2 (wall 7,200s), standard (wall 제한 없음)

### 결과 (2026-10-05, 모든 방식 같은 x*)

S=10 (A·standard 는 §8 값)
| 네트워크 | A | **A2** | standard |
|---|---|---|---|
| Abilene | 46s | **45s** | 189s |
| Polska | 31s | **31s** | 85s |
| grid 5×5 | 99s | **84s** | 267s |
| Nobel-US | **91s** | 99s | 341s |
| Sioux Falls | 872s | 794s | **566s** |

S=50 (세 방식 같은 조건에서 재측정)
| 네트워크 | A | **A2** | standard | A2 oracle 내역 |
|---|---|---|---|---|
| Abilene | 1,350s | 1,088s (−19%) | **1,014s** | 로컬 8 / 192s, α-B&B 1 / 680s |
| Polska | 1,540s | 1,183s (−23%) | **580s** | 로컬 10 / 316s, α-B&B 1 / 83s → menu 단계 ~780s (반복 428) |
| grid 5×5 | 1,507s | **778s** (−48%) | 1,024s | 로컬 7 / 372s, α-B&B 1 / 167s |

S=200 Abilene
| | A (개선 전, §8) | **A2** | standard |
|---|---|---|---|
| 결과 | 7,912s wall 제한, gap 0.65% | 6,547s 수렴 | **6,145s** 수렴 |
| 내역 | — | 로컬 10 / 785s, α-B&B 1 / 3,177s, menu ~2,585s | boost 1회 3,493s |

### 해석
- 개선 3+1 로 α-B&B 호출이 모든 인스턴스에서 최적점 증명 1회로 줄었다 (S=50 에서 8~11회 → 1회). A 대비 19~48% 단축, S=200 도 수렴.
- 그래도 A2 vs standard 는 네트워크마다 갈린다: grid 는 A2, Polska 는 standard (2배), Abilene 은 비슷 (S=50·200 모두 standard 가 4~7% 빠름).
- 이제 두 방식의 공통 비용은 "최적점 증명 1회" (S=200 에서 3,177s vs 3,493s). 차이는 A2 의 menu 단계 (반복 수백 회 × menu LP) 와 로컬 해 시간.
- 개선 2 (worker LP·트리 재사용) 는 α-B&B 호출이 이미 1회라 효과가 작을 것 → 보류.
- 다음 병목 후보: (i) 최적점 증명 자체 (α-B&B 노드 LP, 두 방식 공통), (ii) A2 의 menu 단계 반복 수 (Polska), (iii) Ipopt 시간 (grid).


## 12. 5개 네트워크 확장 측정 (2026-10-05~06, 중단)

`true_dro/archive_alpha_bnb/run_improve_batch2.jl`. S=50: wall 2h·boost 1h, S=200: wall 4h·boost 3h (두 방식 동일).
grid 5×5 S=200 A2 진행 중 (1.5h) 사용자 요청으로 중단.

### S=50 (wall 시간)
| 네트워크 (아크) | A | A2 | standard |
|---|---|---|---|
| Abilene (30) | 1,350s | 1,088s | **1,014s** |
| Polska (36) | 1,540s | 1,183s | **580s** |
| Nobel-US (38) | — | **854s** | 1,072s |
| grid 5×5 (47) | 1,507s | **778s** | 1,024s |
| Sioux Falls (76) | — | **실패**: 반복 1,000 상한 (6,836s), LB 14.93, UB 없음, oracle 2회 | 15,830s (wall 7,200s 초과), LB 18.882 UB 18.965 gap 0.44%, x=[35, 72] |

### S=200 (wall 시간, 둘 다 수렴, 같은 x*)
| 네트워크 | A2 | standard |
|---|---|---|
| Abilene | 6,547s | **6,145s** |
| Polska | 9,313s (로컬 8 / 671s, α-B&B 1 / 1,698s) | **7,424s** (boost 2,208s) |
| grid 5×5, Nobel-US, Sioux Falls | 미측정 (중단) | 미측정 |

### 결론
- **기본은 standard + α-B&B boost 유지.** A2 는 S=50 에서 2/5 네트워크 (Nobel-US, grid) 만 빠르고, S=200 측정한 2개 모두 standard 가 빠름.
- **A2 는 큰 네트워크에서 실패**: Sioux Falls (아크 76) 에서 menu 단계만 1,000회 돌고 LB 가 최적값 (~18.9) 의 79% 에 머묾.
  menu cut (belief 고정 LP cut) 이 약해서 OMP 가 x 를 계속 바꾸며 반복만 늘어남 (S=10 에서도 739회).
  standard 는 mini-Benders 가 x̄ 마다 α 를 갱신해 cut 이 강함 → 45회.

### 미해결
1. **standard 의 wall 제한 초과**: 제한은 반복 시작 시에만 확인 (`true_dro_benders.jl:262`). Sioux S=50 은 7,200s 제한에 15,830s.
   로그에 시간이 찍힌 단계 (Sub 444s, mini/alt-Benders ~1,940s, OMP 재풀이 120s, boost 138s) 합계 ~2,650s →
   나머지 ~13,000s 는 시간 표시가 없는 단계 (α-step 등) 로 추정, 원인 미확인.
2. A2 menu 단계의 K 확장성 (위).
3. S=200 grid·Nobel-US·Sioux Falls 비교.


## 13. Sioux Falls 에서 A 가 수렴하지 못한 원인 (2026-10-06 진단)

### 반복이 어디서 쌓였나 (A2, S=50, verbose, max_iter=200)
- 200회 중 oracle 은 첫 반복 1회 (로컬 해, belief 1개). 나머지 199회는 모두 menu cut (belief 1개 LP) 만.
- LB: 4.16 → 8.90 (반복 20) → **8.905 에서 22~70회 정체** (V_menu ≈ 19.58 반복) → 11.83 (반복 200). 최적값 ≈ 18.9.
- γ=2, K=76 → 후보 x 는 1 + 76 + 2,850 = 2,927개.

### menu cut 계수 측정 (belief 1개, 아크 2개짜리 x′ 2,850개에서 cut 값)

| cut 을 만든 x̄ | x̄ 에서 값 | 다른 x′ 에서 중앙값 | 0 보다 큰 비율 | π_k < −100 인 아크 |
|---|---|---|---|---|
| [35, 73] | 19.8 | −250 | 0.0% | 32/76 |
| [10, 50] | 18.5 | −210 | 0.2% | 32/76 |

→ cut 하나가 실질적으로 막는 점은 x̄ 하나뿐. OMP 가 X 를 한 점씩 지운다 (정체 구간 = 같은 하한의 x 들을 하나씩 소거).
기전: x̄_k = 0 인 아크에서 ρ¹ 의 목적 계수가 0 → 쌍대 퇴화 → LP 꼭짓점이 ρ¹ 을 크게 두면 π_k ≪ 0 (이론 문서 §8).

### standard 와의 비교 시 주의
standard 의 "45 반복" 은 바깥 반복이고, 그 안에서 mini/alt-Benders 가 OMP 를 381회 더 풀었다.
x̄ 평가 횟수는 비슷한 규모이며, 차이는 cut 의 질 (mini-Benders 는 MW 강화, menu cut 은 강화 없음).

### 다음 후보
1. 설계 B (belief 블록 master): 반복 수가 \|X\| 가 아니라 belief 수에 묶임 (이론 문서 명제 6). S=10~50 시제품 → Sioux Falls 에서 반복 수 비교.
2. menu cut 에 MW 강화 (비교 기준선).
3. standard 의 wall 제한 초과: master (OMP MIP) 풀이에 시간 제한이 없음. 반복마다 찍히는 "OMP (0.003s)" 는 실제 master 풀이 시간이 아님
   (Polska 요약의 OMP 평균 7s 와 불일치) → 남은 wall 시간을 OMP/subproblem TimeLimit 으로 넘기고 master 시간을 기록하는 수정 필요.


## 14. menu cut 에 MW 강화 (2026-10-06)

`belief_menu_benders_optimize!(...; menu_mw=true)` (`_mw_menu_cut!`): LP_θ 최적값 z* 를 유지하는 해 중 core point
(차단 가능 아크에 γ/n 균등, mini-Benders 와 같은 점) 에서 목적이 최대인 해로 cut 을 만든다. §15 결과로 기본값 `true`.

### cut 강도 (Sioux Falls S=50, belief 1개, 아크 2개짜리 x′ 2,850개)
| x̄ | cut | x̄ 에서 값 | π 범위 | 다른 x′ 중앙값 | LB(8.9) 보다 큰 x′ | 0 보다 큰 x′ |
|---|---|---|---|---|---|---|
| [35, 73] | 기본 | 19.79 | [−228, 54] | −250 | 0.0% | 0.0% |
| [35, 73] | MW | 19.79 | [−64, 52] | −76 | 0.0% | 0.0% |
| [10, 50] | 기본 | 18.54 | [−228, 23] | −187 | 0.0% | 0.2% |
| [10, 50] | MW | 18.54 | [−82, 14] | −9 | 1.7% | 25.2% |

### Sioux Falls S=50 전체 실행
| | A2 (MW 없음) | **A2 + MW** | standard |
|---|---|---|---|
| 상태 | 반복 1,000 상한, LB 14.93 | **수렴** | wall 제한 초과 종료, gap 0.44% |
| wall | 6,836s | **2,875s** | 15,830s |
| 반복 | 1,000 | 391 | 45 (내부 OMP 약 426회) |
| oracle | 2 | 5 (로컬 4 / 381s, α-B&B 1 / 266s) | — |
| menu | 2 | 5 | — |
| 결과 | — | LB 18.870, UB 18.965, x=[35, 72] | LB 18.882, UB 18.965, x=[35, 72] |

- MW 적용 563/564 회 (실패 1회는 기본 cut 으로 대체).
- 첫 menu 수렴 (oracle 호출) 이 MW 없이는 1,000회 안에 없었고, MW 로는 233회째.
- (정정) 처음엔 "새 belief 직후 첫 반복만 수 분씩 걸린다" 고 보았으나, §15 의 단계별 시간 측정 결과 출력 버퍼링 때문에
  로그 줄이 몰려 찍힌 것이었다. 실제로는 모든 반복이 7.3s 이하.

### 다음
1. 새 belief 첫 MW LP 지연 확인 (기본 LP 첫 풀이와 MW LP 첫 풀이 시간 분리 측정) → dual simplex 강제 또는 MW 시간 상한.
2. `menu_mw` 를 다른 네트워크 (S=10, 50) 에서 확인 후 기본값 결정.


## 15. MW 후속: 단계별 시간과 다른 네트워크 (2026-10-06)

### 단계별 시간 (Sioux Falls S=50, A2+MW, 반복 240 까지)
| 단계 (menu 반복 238회 합) | 시간 | 비율 |
|---|---|---|
| OMP (master MIP) | 656s | 66% |
| MW LP | 200s | 20% |
| menu LP | 133s | 13% |

- 가장 긴 반복 7.3s. 새 belief 직후 (반복 233~240) 도 6~7s 로 다른 반복과 같음.
- §14 의 "새 belief 직후 지연" 은 출력 버퍼링 착시 (menu 줄은 flush 없이 쌓였다가 oracle 줄에서 한꺼번에 기록). menu 줄마다 flush 하도록 수정.
- 남은 병목은 OMP: cut 이 쌓이며 반복당 0.4s → 4s 이상.

### A2 vs A2+MW vs standard (wall, 괄호 = 반복)
| 네트워크 (아크) | S=10 A2 | S=10 A2+MW | S=10 standard | S=50 A2 | S=50 A2+MW | S=50 standard |
|---|---|---|---|---|---|---|
| Abilene (30) | 45s (150) | **42s** (90) | 189s | 1,088s (295) | 1,112s (140) | **1,014s** |
| Polska (36) | 31s (113) | **28s** (68) | 85s | 1,183s (428) | 782s (169) | **580s** |
| Nobel-US (38) | 99s (376) | **50s** (170) | 341s | 854s (419) | **795s** (198) | 1,072s |
| grid 5×5 (47) | 84s (287) | **77s** (203) | 267s | **778s** (256) | 826s (189) | 1,024s |
| Sioux Falls (76) | 794s (739) | **288s** (290) | 566s | 실패 (1,000) | **2,875s** (391) | 15,830s† |

† wall 제한 7,200s 초과, gap 0.44%. 모든 수렴 실행에서 같은 x*.

- MW 로 반복 수가 모든 인스턴스에서 26~61% 감소. 시간은 7곳 단축 + Sioux S=50 은 실패→수렴, 2곳 (Abilene·grid S=50) +2~6% (MW LP 비용).
- A2+MW vs standard: S=10 은 5/5 에서 A2+MW 가 빠름 (2.0~6.8배). S=50 은 Nobel-US·grid·Sioux Falls 에서 A2+MW, Abilene (10%)·Polska (35%) 에서 standard.
- Abilene S=50 은 두 방식 모두 최적점 증명 α-B&B 1회가 대부분 (A2+MW 735s, standard boost ~580s).
- **결정: `menu_mw` 기본값 true.**


## 16. 실험 목표: MW 기본값 기준 재측정 (계획, 2026-10-06)

`menu_mw=true` 가 기본이 되면서 설계 A 의 이전 측정 (§11~12, A2) 은 현재 기본 구성과 다르다.
standard 는 mini-Benders 에 이미 MW 를 쓰므로 영향 없음 → 다시 잴 것은 설계 A 쪽과, 아직 없는 조합뿐이다.

### 이미 현재 구성 (A2+MW) 으로 있는 것
| S | 네트워크 |
|---|---|
| 10 | Abilene, Polska, Nobel-US, grid 5×5, Sioux Falls (§15) |
| 50 | Abilene, Polska, Nobel-US, grid 5×5 (§15), Sioux Falls (§14) |

### 재측정 / 신규 측정 목표
| 우선 | S | 네트워크 | 구성 | 이유 | 시간 제한 |
|---|---|---|---|---|---|
| 1 | 200 | Abilene, Polska | A2+MW | 기존 값은 A2 (MW 없음): Abilene 6,547s, Polska 9,313s. standard (6,145s, 7,424s) 와 다시 비교 | wall 4h, boost 3h |
| 2 | 200 | grid 5×5, Nobel-US | A2+MW, standard | 미측정 (§12 중단) | wall 4h, boost 3h |
| 3 | 200 | Sioux Falls | A2+MW, standard | 미측정. S=50 에서 A2+MW 2,875s vs standard 15,830s 로 차이가 가장 큼 | wall 4h, boost 3h |

예상 시간: 우선 1 약 4~5h, 우선 2 약 8~12h, 우선 3 최대 8h+ (standard 의 wall 초과 가능).

### 선행 조건
1. **standard 의 wall 제한 준수** (§13): master (OMP MIP) 와 subproblem 에 남은 wall 시간을 Gurobi TimeLimit 으로 넘기고 master 시간을 기록.
   고치지 않으면 Sioux Falls·큰 S 에서 standard 가 제한을 크게 넘겨 (S=50 에서 7,200s → 15,830s) 같은 조건 비교가 안 됨. 코드 수정이라 승인 필요.
2. 측정은 `julia -t 14,1`, 다른 계산 작업과 겹치지 않게 단독 실행 (WSL 작업 포함 확인).

### 범위 밖 (다시 돌리지 않아도 되는 것)
- 논문 실험 (`true_dro/factor_5/`, OOS 등) 은 standard Benders (`true_dro_benders.jl`) 만 쓰므로 menu MW 와 무관.
  논문의 계산 실험을 설계 A 로 바꾸기로 하면 그때 별도 계획.
- S=10·50 의 설계 A 측정은 §14~15 로 완료.
