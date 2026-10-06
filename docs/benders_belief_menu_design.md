# Belief 유한성을 이용한 Benders 개선 설계 (2026-10-03)

대상: `true_dro_benders.jl` (outer Benders, OMP ↔ Ω subproblem).
배경: ccg.tex / `docs/ccg_degeneracy.md` 의 belief cover 이론, `docs/alpha_bnb_notes.md` 의 α-공간 B&B.

## 0. 출발점: Ω 의 구조 (코드로 확인)

`build_true_dro_subproblem` 은 "제약은 x 와 무관, x 는 목적함수 계수에만" 들어간다 (Benders 주석: *Subproblem is built ONCE (constraints x-independent); only objective updates*).

  V*(x) = max { c(x)ᵀ y : y ∈ F },   F = Ω 의 가능영역 (x 무관, bilinear 때문에 비볼록),   c(x) 는 x 에 대해 affine.

따라서

1. **V*(x) 는 x 에 대해 볼록** (affine 함수들의 상한). 임의의 ŷ ∈ F 는 전역적으로 유효한 cut η ≥ c(x)ᵀŷ 를 준다. 지금 Benders 가 이것을 쓴다.
2. 어려움은 전부 **V*(x̄) 평가** (비볼록 F 위의 최대화) 에 있다.
3. **belief 를 고정하면 F 가 다면체가 된다.** b = (a, b, r; d, e) (leader / follower belief) 를 고정하면 bilinear 항 α·r, α·d 가 선형이 되어

   v_b(x) := max { c(x)ᵀ y : y ∈ F(b) }   는 **LP** (α 와 쌍대 변수는 자유).

   F(b) ⊆ F 이므로 **v_b(x) ≤ V*(x) ∀x** (restriction) 이고, v_b 도 x 에 대해 볼록.
4. **유한성 (ccg.tex + minimal cover 이론):** 각 x 에서 최적 belief 는 유한 집합 (vertex cover / 극대 패턴의 대표점, leader 쪽은 𝒬 의 꼭짓점) 에서 고를 수 있다. x ∈ X 는 유한 (binary) 이므로 전체 합집합 𝔅 도 유한.

   **V*(x) = max_{b ∈ 𝔅} v_b(x)  ∀x ∈ X.**

   즉 belief menu 𝔅 를 알면 V* 는 **LP 들의 최대값**이다. 비싼 전역 해법은 menu 를 찾는 데만 필요하다.

## 1. 설계 A — Belief-menu Benders (LP cut, 기본안)

### 알고리즘
```
𝔅 ← ∅ (또는 TV 볼 꼭짓점 몇 개로 초기화)
loop
  OMP → x̄, LB
  [저렴] 모든 b ∈ 𝔅 에 대해 LP v_b(x̄) (병렬). V_𝔅(x̄) = max_b v_b(x̄).
         각 b (또는 상위 몇 개) 의 LP 해로 cut η ≥ c(x)ᵀ ŷ_b 추가.      ← 유효 (ŷ_b ∈ F)
  if  LB ≥ V_𝔅(x̄) − tol  (menu 기준으로는 수렴)  또는  x̄ 가 반복
      [비쌈] 전역 평가 V*(x̄) (α-B&B 또는 Gurobi): UB ← min(UB, 상한), 최적 belief b* 획득
      if V*(x̄) > V_𝔅(x̄) + tol:  𝔅 ← 𝔅 ∪ {b*}      (menu 확장)
      else: menu 가 x̄ 에서 exact → UB = V*(x̄) 로 수렴 판정
until UB − LB ≤ tol
```

### 정당성
- cut: ŷ_b ∈ F(b) ⊆ F → 기존과 같은 이유로 전역 유효.
- LB (OMP 값) 는 유효한 하한, UB 는 전역 해법의 상한 (α-B&B 는 dual bound) → 유효.
- **유한 종료:** 전역 평가가 새 belief 를 내놓을 때마다 |𝔅| 가 늘고 𝔅 는 유한. 새 belief 가 안 나오면 menu 가 그 x̄ 에서 exact → 표준 C&CG 종료 논리.

### 기대 효과
- 전역 평가 횟수 ≈ 필요한 belief 수 (+ 확인 1회). 지금은 반복마다 (또는 boost 때) 전역 평가.
- LP v_b(x̄) 는 S=200 에서도 수 초 (α 고정 LP 와 같은 규모). menu 가 작으면 대부분 반복이 LP 만으로 진행.
- 기존 mini-Benders (α 고정 LP) 와 대칭: **α 대신 belief 를 고정**. 둘을 같이 쓸 수도 있음 (menu 원소 = (α, b) 쌍).

### 위험
- menu 크기: S=7 에서 극대 패턴 20~29개 (x̄ 하나 기준). Benders 경로에서 실제로 필요한 것은 x* 근처의 worst belief 뿐이라 훨씬 적을 것으로 예상하나 **측정 필요**.
- 전역 평가가 낸 b* 가 근사해 (incumbent) 이면 menu 에 들어가도 무해 (restriction 이므로 여전히 유효), 단 종료 판정에는 dual bound 를 써야 함.

## 2. 설계 B — Master 에서 belief-scenario C&CG (강한 master, 소규모 S 옵션)

menu 의 각 belief 에 대해 cut 하나가 아니라 **v_b(x) 전체**를 master 에 넣는다 (Zeng–Zhao C&CG 의 master 확장과 같은 역할).

v_b(x) = max { c(x)ᵀ y : A_b y ≤ β_b } = min { β_bᵀ π : A_bᵀ π = c(x), π ≥ 0 }   (LP 쌍대; 제약 형태에 맞게 부호 조정)

master:
```
min_{x ∈ X, η, π_b}  η
s.t.  η ≥ β_bᵀ π_b,   A_bᵀ π_b = c(x),   π_b ≥ 0      ∀ b ∈ 𝔅
```
- **x 가 c(x) (목적 계수) 에만 있으므로 쌍대 실현성 제약이 x 에 대해 선형** → big-M 없이 MILP.
  (x 가 제약 쪽에 있었다면 x·π 곱의 big-M 이 필요했을 것 — 여기서는 0절의 구조 덕분에 불필요.)
- η ≥ v_b(x) 를 **모든 x 에 대해 정확히** 표현 → 같은 belief 로 만들 수 있는 모든 Benders cut 을 한 번에 넣은 것과 같음.
- master 하한 = min_x max_{b∈𝔅} v_b(x). 𝔅 ⊇ (최적 belief) 이면 정확.

### 비용 / 위험
- belief 하나당 쌍대 블록 크기 ≈ Ω LP 크기 (변수·제약 O(K·S)). S≤20 이면 블록 수천 개 변수 → 수십 개 belief 까지 무난. S=200 이면 블록당 ~9만 → 실용 어려움.
- 쌍대 블록에서 Ω 의 자유/상한 없는 변수 (ρ, ω, …) 에 대응하는 쌍대 제약이 등식/부등식으로 정확히 나와야 함 → JuMP 로 F(b) LP 를 만들고 `dualize` (Dualization.jl) 로 생성하는 것이 안전. (설치 여부 확인 필요)

## 2.5 Claude Doc 「ccg.tex 구조 활용 ISP 평가」 와의 관계, 그리고 사전 진단

**바깥 층에서는 max 쪽 열이 정당하다.** 그 문서의 핵심 교훈은 "subproblem 안에서 belief 를 열로 쓰면 master 가
restriction 이 되어 NP-hard 인증이 필요하고, follower 반응 h̄ 를 열로 써야 relaxation + LP pricing 이 된다" 였다.
바깥 층 (min_x V*(x)) 에서는 방향이 반대다: master 는 min 문제의 relaxation 이어야 하고, max 쪽 대상 (belief b 또는 α) 을
고정한 v_b(x), g_α(x) 는 모두 V*(x) 의 하한이므로 master 에 넣어도 relaxation 이 유지된다 (Zeng–Zhao 의 시나리오와 같은 편).
인증 (UB) 은 최종 x̄ 에서 전역 해법 한 번으로 충분하다.

**문서의 제안 "바깥 층 CCG: follower 반응 α_j 를 master 에 열로 넣어 g_α(x) 를 정확히 담기"** 는 이 설계의 α 판이다.
g_α(x) = (α 고정 시 ISP-L ∥ ISP-F LP 값) — 기존 mini-Benders 가 cut 으로 쓰는 것과 같은 LP.
문서가 권한 사전 진단 (반복별 α* 중복도) 을 기존 로그로 했다:

| 로그 (Abilene, double, β=0.4) | 값 |
|---|---|
| Benders 반복 | 299 (boost 6회, TIME_LIMIT 37회) |
| 고유 α* (소수 2자리 / 1자리 / 정수 반올림) | 241 / 232 / 216 |
| 고유 지지집합 (α_k > 0 인 arc 집합) | 145 |
| α* 지지집합 크기 | 2~9 (평균 4.7) |
| subproblem 시간 | 합 18,661s, 중앙값 0.2s, 상위 10% 15s, 최대 3,609s (boost) |

→ **α menu 는 작지 않다** (반복의 70~80% 가 새 α*). α 판 C&CG 는 기대 효과가 낮다.
→ **시간의 대부분은 boost 소수 회** (3,600s 급). 이미 구현한 boost=:alpha_bnb 가 직접 겨냥하는 부분.
→ belief 판의 menu 크기는 로그에 belief 가 없어 미확인 — C1 구현 시 계측해서 판단.

## 3. 설계 C — 하이브리드 (권장 구현 순서)

| 단계 | 내용 | 비용 | 기대 |
|---|---|---|---|
| C1 | 설계 A (menu + LP cut). 전역 평가는 기존 boost 규칙 대신 "menu 수렴 시" 호출. 전역 해법 = `boost_solver` 그대로 (:gurobi / :alpha_bnb) | 작음 (LP builder 는 α 고정 LP 와 동일 패턴) | 전역 평가 횟수 감소 |
| C2 | menu warm start: 전역 평가가 낸 belief 뿐 아니라 α-B&B 노드에서 본 belief (노드 LP 해의 a, r, d) 중 값이 큰 것도 후보로 추가 | 작음 | menu 가 빨리 참 cover 에 도달 |
| C3 | S 작을 때 설계 B (master 쌍대 블록) 옵션 | 중간 | outer 반복 수 감소 |
| C4 | follower 반응 조각 θ = Q(x̄, h̄) 의 x̄ 간 재사용 (ccg_degeneracy §8.4 열린 과제 b) → minimal cover / response CCG 를 전역 해법으로 쓸 때 warm start | 중간 | S≤5~7 에서 전역 평가 자체 가속 |

## 4. 실험 계획

- 인스턴스: Abilene / Polska, β=0.4, ε=0.2, S ∈ {10, 50, 200}.
- 비교: (i) 기존 Benders (boost=:gurobi), (ii) boost=:alpha_bnb, (iii) C1, (iv) C1+C2, (v) C3 (S≤20).
- 지표: 전역 평가 횟수, outer 반복 수, wall time, 최종 gap, |𝔅|.
- 확인할 가설: Benders 경로에서 필요한 |𝔅| 가 작다 (x* 근처 worst belief 소수).

## 5. 주의

- 전부 λU=10 인 Ω 기준. Ω 자체가 λU 벌점 완화라는 문제 (`docs/alpha_bnb_notes.md` §4) 는 별개로 남음.
- belief 고정 LP 는 α 를 자유롭게 두므로 v_b 는 "그 belief 에서의 최악 α" 까지 포함 — ccg.tex 의 W(v) 와 같은 양 (Ω 표현).


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
