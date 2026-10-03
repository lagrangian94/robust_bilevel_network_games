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
