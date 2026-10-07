# 비제로섬 확장: location 구현과 belief LP cut 검증 (2026-10-06 ~ 10-07)

이 문서는 측정 결과만 다룬다. 수식과 유도는 `docs/nonzero_sum_extension.html`
(게시본 https://claude.ai/artifact/B2bkdVY57TmeXDh2GRtH27) 에 있다.

- 코드: `nonzero_sum/`
  - `nz_data.jl` — 일반 LP recourse 데이터 (`NZData`), location 인스턴스, 원고 max-flow 변환
  - `nz_omega.jl` — 비제로섬 Ω (html Step 4′, θ = θᵁ 고정), 확장 belief 고정 LP, follower 인증서
  - `nz_kkt_eval.jl` — big-M 없는 독립 평가기 (KKT 상보성 indicator + r·φ bilinear) 와 θ 진단
  - `nz_benders.jl` — 표준 Benders, belief-menu Benders (확장 belief), oracle = Gurobi | α-B&B
  - `nz_alpha_bnb.jl` — α 만 분기하는 전역 해법 (zero-sum α-B&B 를 비제로섬 Ω 로 확장, §9)
  - `test_nz_omega.jl`, `test_nz_belief_menu.jl`, `screen_location_params.jl` — 검증·탐색 스크립트
  - `logs/` — 실행 로그
- 실행 환경: 이 PC 의 Julia 1.12 (juliaup), Gurobi 12 (Gurobi_jll), 기본 환경에 JuMP·Gurobi·HiGHS·Ipopt 설치.

## 1. 구현 요약

일반 recourse: `Q = max{cᵀy : Ay ≤ r_s(x,h)}`, `r_s(x,h)_k = u_kˢ + hcoef_k·h_{h(k)} + g_kˢ·x_{ι(k)}`
(행마다 h, x 가 많아야 하나씩). 리더 평가 `φ_s = max{ℓᵀy : y ∈ argmax Q}` (pessimistic).
follower 1단계 목적 `c0ᵀh` 를 허용 (location 의 예약비 `−fᵀh`). 리더 위험은 CVaR_β + TV 볼 (기존 실험과 같은 형태).

Ω 는 html Step 4′ 파라미터 버전 그대로다.
- 리더 블록: `F^L = Σ(ℓ+θc)ᵀŷ − θΣ r_s(x,α)ᵀϖ − π̂ᵁ McCormick`, 제약에 N1 `Aᵀϖˢ ≥ r_s c` 만 추가.
  bilinear 은 α·r (기존), α·ϖ (신규, H 행만) 두 종류.
- follower 블록: 원고 그대로이고, 1단계 목적 때문에 λ 의 dual 제약 우변에 `c0ᵀα` (α 에 선형) 가 붙는다.
- 확장 belief 고정 LP: (a, b, r, d, e) 와 인증서 σ 를 고정하고 ϖˢ = r_s σˢ 로 둔다. 고정 범위 두 가지:
  `:hrows` (bilinear 에 들어가는 H 행의 ϖ 만 고정, 나머지는 LP 변수) / `:all` (html 그대로 ϖ 전체).
  인증서 두 가지: `:recompute` (html (b), (x̄, α̂) 에서 follower LP simplex dual) / `:solution` (html (a), ϖ̂/r̂).

## 2. zero-sum 교차검증

원고 max-flow (grid 3×3, S=3, β=0.4, ε̂=ε̃=0.2, λᵁ=10) 를 일반 recourse 로 옮겨 ℓ = c = e_ts, θ = 0 으로 두면
기존 `build_true_dro_subproblem` 과 Ω(x) 가 같아야 한다.

| x | 기존 Ω | 일반 Ω | 차이 |
|---|---|---|---|
| ∅ | 11.777778 | 11.777778 | 1.8e-14 |
| [6, 10] | 9.000000 | 9.000000 | 1.1e-14 |
| [9, 10] | 4.888889 | 4.888889 | 0 |
| [6, 11] | 8.000000 | 8.000000 | 4.4e-15 |

→ 일반 Ω 의 리더·follower 블록, CVaR·TV, McCormick 코딩이 원고와 일치.

## 3. location 인스턴스 (Goyal et al. 2023 §7.1 기본 사례 규모)

Goyal 의 문제 파라미터만 가져오고, 불확실성은 원고 설정 (support 에서 샘플링한 이산 시나리오, q̂ 균등, 양쪽 TV 볼, 리더 CVaR) 을 쓴다.
Goyal 의 moment ambiguity 는 쓰지 않는다.

| 항목 | 값 | 출처 |
|---|---|---|
| 위치 | 8곳, [0,1]² 균등 좌표 (SGB128 대신) | Goyal 기본 사례 d=8 |
| A 점포 | 1곳, 용량 720 | Goyal |
| B 후보 | 4곳, 용량 360, 출점비 305, 출점 수 ≤ 4 | Goyal |
| 고객 | 나머지 3곳, 수요 U[30, 240] 에서 S 개 샘플 | Goyal (support) |
| 수송비 | Euclidean 거리, 0.1 단위 반올림 | Goyal (Euclidean), 반올림은 θᵁ 용 |
| 리더 손실 ℓ | B 점포에서 나가는 flow 에 −5 | Goyal v = −5 |
| 예약 (확장) | 단위 예약비 f = 0.1, 현장구매 프리미엄 p = 0.3, 한도 1ᵀh ≤ 100 | 신규 (html) |
| ambiguity | S = 3, ε̂ = ε̃ = 0.2, β = 0.4 | 원고 실험 공통 설정 |
| λᵁ | 1000 | 설정값 (KKT 대조로 확인) |

## 4. θᵁ: Lemma 2 상계는 24배 느슨, circuit 상계는 정확

- html Lemma 2 (Cramer): θᵁ = D‖ℓ‖₁ = 10 × 5 × 24 = **1200**.
- location 구조 상계 (circuit 논증, `nz_data.jl`): Y 의 edge 방향은 TU 행렬의 circuit (성분 ±1).
  circuit 하나가 바꾸는 B 총판매량은 1 단위 이하라 Δ(ℓᵀy) ≤ v, follower 비용 변화는 0 이 아니면 해상도 0.1 이상.
  따라서 θ* ≤ v / 0.1 = **50**.
- 진단 (`nz_theta_diag`: 2단계 LP 로 정확한 φ 와 최적성 제약 dual = 필요한 θ): Ω 해의 α 에서 필요한 θ 가
  여러 x 에서 **정확히 50** (나머지 10, 12.5, 16.67). 즉 50 은 달성되는 값이고 더 줄일 수 없다.
- θᵁ = 1200 의 부작용: F 가 `θ(cᵀŷ − rᵀϖ)` 처럼 ~2×10⁵ 규모 두 항의 상쇄를 포함해서, Gurobi 허용오차만으로
  x = ∅ (참값 0) 에서 Ω = 0.108 이 나왔다 (과대평가 → LB 오류 가능). θᵁ = 50 + FeasibilityTol/OptimalityTol 1e-9 로
  같은 점에서 9e-13.
- π̂ᵁ 도 같은 논증 (A 점포에 항상 여유 → 수요 dual ≤ θ(cmax+p)) 으로 θᵁ(cmax+p)+v = 65~75. 진단에서 실제 π̂/π̂ᵁ ≤ 0.53.

## 5. Ω vs 독립 평가 (seed 1, S=3, 모든 x ∈ X, θᵁ=50)

`logs/test_omega_S3.log`. 16 개 x 전부 |Ω − KKT| ≤ 2.3e-4 (값 −1360 대비 상대 2e-7, Ω MIP gap 1e-6 범위).
Ω 는 x 당 2~90s (두 개는 600s 시간 제한, incumbent 는 KKT 와 일치), KKT 평가는 0.1~12s.

관찰: seed 1 은 B 점포가 하나라도 있으면 V*(x) = −1360 으로 모두 같다. A 점포가 모든 고객에게 멀고 (0.8~1.1)
예약 한도가 커서, pessimistic follower 가 A 에 예약하면 A 예약가 = B 현장가 (예: 0.6+0.3 = 0.9) 동률이 되고
동률에서 B 가 불리하게 배정되는 것으로 추정 (face-selection 층). 리더 결정이 자명해서 belief-menu 검증에는 seed 4 를 쓴다.

## 6. 인스턴스 탐색 (`screen_location_params.jl`, KKT 평가기, S=3)

| seed | 최적 x | q·x + V* | 2등 | V* 고유값 수 /16 |
|---|---|---|---|---|
| 1 | 동률 | −1055.0 | −1055.0 | 2 |
| 2 | 동률 | −784.5 | −784.5 | 6 |
| 3 | ∅ | 0.0 | 5.4 | 4~5 |
| **4** | **[3, 4]** | **−1786.0** | **−1481.0** | 6 |
| 5 | 동률 | −875.6 | −875.6 | 3 |
| 6 | [3, 4] | −380.8 | −190.1 | 5 |

p ∈ {0.2, 0.3}, 예약 한도 ∈ {100, 200} 은 결과를 바꾸지 않았다. Goyal scalability 설정 (d=15, 고객 10) 은
KKT 평가가 120s 안에 끝나지 않아 접었다.

## 7. 확장 belief LP cut 검증 (seed 4, S=3)

### (A) exactness: x 에서 얻은 (belief, 인증서) 를 고정한 LP 의 x 값 = Ω(x) ?

`logs/belief_menu_base_seed4_S3_tableA_run3.log`. 열: KKT 독립 평가, 전역 Ω (MIP gap 1e-5, 300s), 인증서 × 고정 범위 4 가지.

| x | KKT V*(x) | Ω(x) | recompute/hrows | recompute/all | solution/hrows | solution/all | max(Ω − LP) |
|---|---|---|---|---|---|---|---|
| ∅ | 0 | 0 | 0 | 0 | 0 | 0 | 4e-9 |
| [1], [2], [1,2] | −480.6667 | −480.6666 | −480.6667 | −480.6667 | −480.6667 | −480.6667 | ≤ 5.1e-5 |
| [3] | −725.0000 | −725.0000 | 동일 | 동일 | 동일 | 동일 | 4.9e-5 |
| [1,3], [2,3], [1,2,3] | −1239.8333 | −1239.8333 | 동일 | 동일 | 동일 | 동일 | ≤ 6.4e-5 |
| [4], [1,4], [2,4], [1,2,4] | −1520.4444 | −1520.4444 | 동일 | 동일 | 동일 | 동일 | ≤ 8.5e-5 |
| [3,4], [1,3,4], [2,3,4], [1,2,3,4] | −2396.0000 | −2396.0000 | 동일 | 동일 | 동일 (−2396.0001) | 동일 | ≤ 2.2e-5 |

- 16 개 x 모두, 네 가지 변형 모두 Ω(x) 를 Ω 자체의 MIP gap (1e-5 상대) 안에서 복원. html 의 "최적 ϖ/r 는 follower dual
  꼭짓점이므로 인증서 하나로 제한해도 손실 없음" 이 수치로 확인됨.
- Ω 와 KKT 도 모든 x 에서 일치 (seed 4 에서도 Ω 유도 검증).
- 인증서 수치 문제: solver dual (simplex shadow price, ϖ̂/r̂) 은 1e-7~1e-8 만큼 Aᵀσ ≥ c 를 어길 수 있고, ϖ 를 고정하면
  belief LP 가 INFEASIBLE 이 된다 (`:solution` + `:all` 에서 발생). `nz_repair_cert` (min‖σ′−σ‖₁ LP + 계수 ≥ 0 행 dual 을
  올리는 정확 보정) 로 해결. 보정은 restriction 을 바꿀 뿐이므로 cut 유효성에는 영향 없음.

### (B) validity, (C) 충분성 (`analyze_nz_menu_cache.jl`, `logs/analyze_cache_seed4_S3.log`)

(A) 에서 x 마다 얻은 원소 (belief + 인증서 σ:recompute, `:hrows`) 16 개를 모든 x₂ 에서 풀었다 (LP 256 개).

- validity: 모든 쌍에서 LP ≤ Ω(x₂). 최대 초과 1.2e-4 (x₂=[3,4], Ω incumbent 자체가 −2396.0001 로 참값보다 1e-4 낮음, 상대 5e-8).
  → 다른 x 에서 얻은 인증서는 weak duality 로 벌점만 커진다는 html 의 유효성 논증과 일치.
- 충분성: 모든 x₂ 에서 max_원소 LP = Ω(x₂) (gap ≤ 8.5e-5). 16 개 x 를 모두 exact 로 덮는 greedy 최소 menu 는 **3 개**
  (x₁ = [1], [3], [4] 에서 얻은 원소).
- 원소가 exact 인 범위는 x 에 따라 갈린다.

| 원소를 얻은 x₁ | exact 인 x₂ |
|---|---|
| ∅ | ∅ |
| [1] | ∅, [1], [2], [1,2] |
| [3], [1,3] | ∅ 와 3 을 포함하는 모든 x (9 개) |
| [4] 계열 | ∅, [4], [1,4], [2,4], [1,2,4] |
| [3,4] | [3,4], [1,3,4] |
| [1,3,4] 등 | 3, 4 를 모두 포함하는 4 개 |

- 인증서 패턴: 시나리오마다 고유 σˢ 9 개 / 원소 16, 고유 belief (r, d) 12 개 / 16. 같은 belief 라도 x 에 따라 min-cut 이 달라서
  belief 만 저장하면 다른 x 에서 exact 가 아니다 → html "belief 하나에 시나리오당 min-cut 하나, 다른 x̄ 에서 다른 min-cut 이 필요하면
  oracle 이 새 쌍을 추가" 의 구조가 그대로 보인다. 다만 이 인스턴스에서는 3 개면 충분.

### (D) 대조: 인증서 없이 belief 만 고정

belief (a, b, r, d, e) 만 고정하면 α·ϖ 가 남아 QCP 다 (Gurobi NonConvex). x = ∅, [1], [2] 에서 값은 Ω 와 같다
(0, −480.6667, −480.6667). 즉 belief 만으로도 값은 충분하지만 LP 가 아니고, 인증서를 붙여야 LP 가 된다 (html 명제 2).

### (E) Benders 비교 (`logs/belief_menu_base_seed4_S3.log`, tol 1e-5, Ω 시간 제한 600s)

| | x* | LB = UB | 반복 | 전역 Ω 호출 | menu | wall |
|---|---|---|---|---|---|---|
| 표준 Benders (매 반복 전역 Ω) | [3, 4] | −1786.0001 | 17 | 16 | − | 3,754s |
| belief-menu (확장 belief, σ:recompute, `:hrows`) | [3, 4] | −1786.0001 | 10 | **2** | **2** | **69.5s** |

전수 열거 최적 ([3, 4], q·x + V* = 610 − 2396 = −1786) 과 같다.

- 표준 Benders 는 16 개 x 를 사실상 전부 방문했다. LB 가 −4.5×10⁶ 에서 시작해 −4.6×10⁵ 까지밖에 안 오르는데,
  cut 기울기가 λᵁ = 1000 이 곱해진 McCormick 항 (ρ̃, ρ⁰) 에 지배되기 때문이다. 원고 실험의 big-M tightening 이슈와 같은 종류.
- belief-menu 는 x̄ = ∅ 에서 얻은 원소 하나로 반복 2~8 에서 menu cut 을 냈고 (모두 유효, exact 는 아님:
  예 x=[1,2,3,4] 에서 −2667 < Ω −2396), 반복 9 의 x̄=[3,4] 에서 menu 가 수렴해 oracle 을 불러 두 번째 원소를 얻었다.
  그 원소가 [3,4] 에서 exact (gap −1.2e-4, Ω incumbent 오차) 라 반복 10 에서 LB = UB.
- menu cut 은 같은 반복 수에서 LB 를 더 많이 올렸다 (반복 8: −3.8×10⁴ vs 표준 반복 15: −4.6×10⁵). belief LP 의 해가
  McCormick 쪽 ρ 를 덜 쓰는 꼭짓점을 고르는 것으로 보인다 (확인 안 함).
- oracle 이 x̄ 에서 새 원소를 넣을 때마다 그 원소의 LP 값이 x̄ 에서 Ω(x̄) 와 같았다 (exactness 기록 2/2).

## 8. λᵁ (follower exact penalty) 민감도 (`test_nz_lambda.jl`, seed 4, S=3)

follower 블록은 `V̄^F = inf_{λ≤λᵁ} λ·gap` 이라 λᵁ 가 유한하면 반응집합 밖 α 에 유한 벌점만 준다.
λᵁ 가 작으면 Ω 가 V* 를 과대평가하고 cut 이 무효가 된다.

`logs/lambda_seed4_S3.log`: x = [3], [4], [1,3] 은 λᵁ = 0.1 부터 exact, x = [3,4] 만 민감.

| λᵁ | Ω([3,4]) | Ω 상한 | V* (KKT) | Ω − V* |
|---|---|---|---|---|
| 0.1 | −1897.0 | | −2396.0 | +499 |
| 1 | −1906.0 | | −2396.0 | +490 |
| 10 | −1996.0 | | −2396.0 | +400 |
| 40 | −2296.0 | −2296.0 (OPTIMAL) | −2396.0 | +100 |
| 49 | −2386.0 | −2386.0 (OPTIMAL) | −2396.0 | +10 |
| 49.9 | −2395.0 | −2395.0 (OPTIMAL) | −2396.0 | +1 |
| 50 | −2396.0 | −2394.3 (300s) | −2396.0 | 0 |
| 51, 60, 100, 1000 | −2396.0 | | −2396.0 | 0 |

과대평가량은 정확히 **10·(50 − λᵁ)⁺** (`logs/lambda_edge_seed4_S3.log`). 리더는 반응집합 밖으로 예약 100 단위를 옮겨
단위당 v = 5 를 얻고 follower 는 단위당 0.1 (거리 해상도) 을 잃는다 → 경계 λ* = v / 해상도 = 50 으로 θᵁ 와 같은 값.
follower 2단계 LP (h, y) 에도 θᵁ 와 같은 circuit 논증이 통하는 것으로 보이지만, follower 쪽 gap 은 belief d 로 가중되고
(d_s 최솟값이 작으면 단위당 손실도 작아짐) 리더 쪽은 CVaR 가중 r 이라 일반 상계는 아직 유도하지 않았다.
기존 원고 실험의 λᵁ = 10 (max-flow) 도 같은 방식으로 경계를 확인해 볼 만하다.

### λᵁ 별 Benders — Wcap 이전 Ω (`logs/lambda_benders_seed4_S3_stopped.log`, tol 1e-4, oracle 600s, 재방문 시 2400s)

> **주의**: 이 절의 시간은 ϖ_ks ≤ p·r_s (Wcap, §9) 를 넣기 전 Ω 로 잰 것이다. Wcap 을 넣으면 Gurobi Ω 의 느림이
> 사라져서 아래의 "λᵁ 를 줄이면 oracle 이 어려워진다" 는 결론은 대부분 그 강화가 없던 탓이었다. 갱신된 결과는 §9.

λᵁ = 50 은 exact 경계 그 자체 (§8), Ω 는 λᵁ 에 단조 비증가라 50 이상은 모두 exact.

| λᵁ | 방법 | 상태 | x* | UB | 반복 | 전역 Ω | menu | wall |
|---|---|---|---|---|---|---|---|---|
| 50 | belief-menu | Optimal | [3,4] | −1785.96 | 11 | 3 | 2 | 3,010s |
| 50 | 표준 | Stalled (gap 2.2e-4) | [3,4] | −1785.61 | 18 | 17 | − | 8,485s |
| 100 | belief-menu | Optimal | [3,4] | −1785.97 | 10 | 2 | 2 | 600s |
| 100 | 표준 | 중단 (반복 13, LB −6.8×10⁴) | [3,4] (UB) | −1785.96 | 13+ | 13 | − | ~7,000s+ |
| 1000 | belief-menu (tol 1e-5, §7) | Optimal | [3,4] | −1786.00 | 10 | 2 | 2 | 69.5s |
| 1000 | 표준 (tol 1e-5, §7) | Optimal | [3,4] | −1786.00 | 17 | 16 | − | 3,754s |

- λᵁ 를 줄이면 표준 Benders 의 첫 cut 뒤 LB 가 10 배 좋아진다 (λᵁ=1000: −4.5×10⁶, 50: −4.3×10⁵, 100: −5.7×10⁵) 그러나
  여전히 McCormick big-M 항이 지배해서 x 를 거의 전부 방문한다.
- 대신 전역 Ω (oracle) 가 어려워진다: x=[3,4] 에서 λᵁ=1000 은 66~87s 에 OPTIMAL, 100 은 600s 에 gap 1.4e-5,
  50 은 2400s 보강 뒤에도 gap 2.2e-4. 경계에 가까운 λᵁ 에서는 반응집합 밖 α 의 벌점이 리더 이득과 거의 같아
  (§8: 기울기 10·(50−λᵁ)) Ω 의 최적해 근처가 평평해지는 것으로 보인다.
- 이 인스턴스에서는 cut 강화보다 oracle 난이도 증가가 커서 λᵁ=1000 (경계의 20 배) 이 가장 빨랐다.
- λᵁ=100 표준 Benders 는 반복마다 Ω 가 600s 시간 제한에 걸려 반복 13 에서 중단했다 (결론에 영향 없음).
  λᵁ=1000 의 tol 1e-4 재측정도 생략 (tol 1e-5 결과가 §7 에 있음).

### α-B&B 는 λᵁ 에 덜 민감한가 (측정 전 추론 — 결과는 §9)

위 oracle 은 전부 Gurobi NonConvex 다. α 만 분기하는 `true_dro/global_bilinear_solver.jl` 은 zero-sum Ω 전용이라
비제로섬의 새 bilinear α·ϖ 를 다루지 못한다. 예상은 양쪽 모두 민감하다는 쪽이다.
- λᵁ 가 클 때: 노드 RLT 완화에서 ζF = α·d 가 어긋날 여지를 follower 블록이 이용하는데, 그 블록의 벌점 계수가
  λᵁ 크기라 노드 상한의 느슨함이 대략 λᵁ × 상자 폭에 비례할 것 → 더 잘게 분기.
- λᵁ 가 경계 근처일 때: 반응집합 밖 α 가 최적값과 거의 같은 평평한 영역은 Ω 자체의 성질이라 α-B&B 도 덮어야 함.
- 유리한 점: α (여기선 5 차원, S 무관) 만 분기하고 상자가 점이면 정확한 LP. zero-sum 에서 Gurobi boost 1,148s → 7s 전례.
→ 비제로섬에 α-B&B 를 연결해 λᵁ ∈ {50, 100, 1000} 에서 Gurobi 와 비교하는 작업을 진행 (§9).

## 9. 비제로섬 α-B&B 와 Wcap 강화 (2026-10-07)

### 구현 (`nz_alpha_bnb.jl`)

`true_dro/global_bilinear_solver.jl` 의 α-공간 B&B (병렬 best-first + diving, 위반 RLT 행 분리, Ipopt 휴리스틱) 를
비제로섬 Ω 로 옮겼다. 바뀐 것:
- bilinear 이 α·r, α·d 에 더해 **α_{h(j)}·ϖ_js** (H 행 j 의 인증서). 상자가 점이면 Ω 는 정확한 LP.
- 노드 완화: belief 행 (a, b, r, d, e 상·하한, TV 볼, CVaR) × 모든 α_i, **인증서 행 (ϖ ≥ 0, ϖ ≤ ϖᵁ r) × 자기 α_{h(j)}**.
  α_i·ϖ_js (i ≠ h(j)) 는 Ω 에 없으므로 만들지 않는다.
- Ω 빌더에 mode (`:global`, `:belief_lp`, `:fixed_alpha`, `:relax`) 추가. α-B&B 는 `:relax` (노드) 와 `:fixed_alpha` (평가) 를 쓴다.
- Benders 에 `oracle = :gurobi | :alpha_bnb` (α-B&B 는 incumbent α 를 고정한 LP 의 해로 cut·belief), belief-menu 에
  목표값 조기 종료 `target_stop` (α-B&B 전용) 추가.
- Gurobi Env 는 프로세스 안에서 풀로 재사용 (`_nz_envs`, 성능 중립). 선택 인자 `envs`, `heuristic` 은 이 PC 의 학술 WLS
  라이선스 (동시 세션 2 개) 에서 worker 1 개로 검증할 때만 쓴다. 기본값은 worker 12 + 휴리스틱 그대로.

### Wcap: ϖ_ks ≤ p·r_s

α·ϖ 의 RLT 를 만들려고 Ω 에 ϖ_ks ≤ ϖᵁ_k r_s (H 행) 를 넣었다. ϖˢ/r_s 는 follower 최적 dual 이고 예약 dual ≤ p 인
최적 dual 이 존재하므로 (nz_data.jl 논증) 최적값을 바꾸지 않는다. 확인: λᵁ ∈ {50, 100, 1000} 에서 16 개 x 모두
Ω = KKT V* (최대 차이 1.3e-7). 이전에는 ϖ ≤ p·r_max 만 있어서 α·ϖ 의 McCormick 이 매우 느슨했다.

효과 (Gurobi Ω, x=[3,4]): λᵁ=1000 66~87s → 0.0s, λᵁ=50 2400s 에도 미종료 → 1.5s.

### α-B&B 정확성과 λᵁ 민감도 (`test_nz_alpha_bnb.jl`, `logs/alpha_bnb_lambda_seed4_S3.log`, worker 1, 휴리스틱 끔)

12 개 (λᵁ, x) 모두 α-B&B 의 [LB, UB] 가 KKT V* 를 포함. 대부분 root 노드 하나 (0.2s) 에서 exact.
민감한 곳은 x=[3,4] 하나:

| λᵁ | α-B&B 노드 | α-B&B 시간 | Gurobi Ω (Wcap) |
|---|---|---|---|
| 1000 | 1 | 0.2s | 0.0s |
| 100 | 34 | 0.6s | 0.1s |
| 50 | 4,669 | 94s | 1.5s |

→ α-B&B 는 λᵁ 에 덜 민감하지 않다. λᵁ 가 경계 (50) 에 가까우면 반응집합 밖 α 가 평평해져 노드가 폭증한다 (§8 의 추론 중
두 번째). λᵁ 가 클 때 노드 상한이 느슨해지는 효과 (첫 번째 추론) 는 보이지 않았다 (1000 에서 root 하나로 종료).
worker 1 개·휴리스틱 없이 잰 것이라 Gurobi 와의 속도 비교는 실험 머신에서 다시 해야 한다.

### λᵁ × oracle × Benders (Wcap 적용, `logs/lambda_benders_wcap_seed4_S3.log`, tol 1e-4)

| λᵁ | oracle | belief-menu | 표준 |
|---|---|---|---|
| 50 | Gurobi | 6.2s (Ω 2, menu 2) | 27.3s (Ω 16) |
| 50 | α-B&B (worker 1) | 602.7s (Ω 2, menu 1) | 1,202s (Ω 10) |
| 100 | Gurobi | 0.2s | 3.4s (Ω 16) |
| 100 | α-B&B | 0.7s | 2.4s (Ω 9) |
| 1000 | Gurobi | 0.1s | 2.4s (Ω 16) |
| 1000 | α-B&B | 0.5s | 2.0s (Ω 9) |

모두 x* = [3, 4], −1786.0 (λᵁ=50 α-B&B 의 UB 는 −1785.89, gap 6e-5 < tol).

- Wcap 하나로 이전 측정 (§7 E: 표준 3,754s, belief-menu 69.5s, λᵁ=1000) 이 2.4s, 0.1s 가 됐다.
- α-B&B oracle 을 쓴 표준 Benders 는 반복 10 (Ω 9 회) 로, Gurobi oracle (반복 17, Ω 16 회) 보다 적다. α-B&B 는 incumbent α
  를 고정한 LP 의 해 (꼭짓점) 로 cut 을 만드는데, 이 cut 이 Gurobi 비볼록 해의 cut 보다 강하다 (반복 9 의 LB: −1,925 vs
  Gurobi 쪽은 같은 시점 −10⁴~10⁵ 규모). 원인 확인 안 함.
- λᵁ 는 이제 경계 근처 (50) 만 피하면 Gurobi·α-B&B 모두 빠르다. λᵁ 를 경계의 2 배 (100) 이상으로 잡으면 충분.

## 9b. aggregate follower 의 균형 해석 (2026-10-07)

예약을 고객×점포별 선주문 h_ij (y^R_ij ≤ h_ij, 점포별 선판매 할당량 Σ_j h_ij ≤ w_i) 로 두면 (`make_location_instance(reservation=:pair)`),
follower 의 aggregate 2단계 LP 는 개별 고객 선주문의 가격 균형과 정확히 같다 (용량 dual = 시나리오별 혼잡 가격,
할당량 dual = 예약 슬롯 가격). 명제와 증명은 논문 supplementary `joc-non_zero_sum/supplementary.tex` 의
"Equilibrium interpretation of the aggregate follower in the location instance" 절.

수치 확인 (`test_nz_equilibrium.jl`, `logs/equilibrium_seed4_S3.log`, x 16 × belief 4, 세 경우 192 개):
- aggregate 해가 가격 아래 각 고객 문제의 최적해: 192/192 (상대 차이 ≤ 2e-16), 강쌍대 ≤ 4e-16.
- 가격 0 이면 결합 제약 (용량·할당량) 위반 192/192 → 혼잡 가격이 균형의 필수 요소.
- 고객별로 따로 푼 최적해 묶음은 열린 점포 용량이 묶일 때 청산 실패 (48/48). 예약 없는 벤치마크 (할당량 0) 에서도
  같은 48/48 → 가격이 배분을 하나로 정하지 못하는 것은 YK/Goyal 수송 LP 에 원래 있는 성질. pessimistic 선택이 처리.
- 점포별 예약 풀 (pooled) 의 follower 값은 항상 pair 보다 큼 (완화).
- 닫힌 B 후보지의 용량 dual 은 혼잡 가격이 아니므로 고객 선택지와 집계에서 제외.

## 10. 요약

1. 비제로섬 Ω (html Step 4′) 는 zero-sum 에서 원고 Ω 와 일치하고, location 에서 big-M 없는 독립 평가와 모든 x 에서 일치한다.
2. θᵁ 는 Lemma 2 대신 구조 상계 (location: v/거리해상도 = 50) 를 써야 한다. 1200 은 수치 상쇄로 Ω 를 과대평가했다.
3. 확장 belief (belief + follower 인증서) 를 고정하면 Ω 가 LP 가 되고, (i) 얻은 x 에서 exact, (ii) 다른 x 에서 유효,
   (iii) 모은 원소의 max 가 모든 x 에서 Ω 를 복원한다. html "확장 belief" 절의 검증 항목이 이 인스턴스에서 성립.
4. belief-menu Benders 는 전역 Ω 2 회, menu 2 개로 수렴해 표준 Benders 보다 54 배 빨랐다.
5. λᵁ (follower exact penalty) 는 이 인스턴스에서 경계가 정확히 50 (= v/해상도, θᵁ 와 같음). 그 미만이면 Ω 가 V* 를 10·(50−λᵁ) 만큼 과대평가 (cut 무효), 이상이면 exact (λᵁ=50 에서 16 개 x 모두 확인, `logs/lambda50_allx_seed4_S3.log`). λᵁ 를 줄이면 cut 은 강해지지만 Gurobi 전역 Ω 가 크게 어려워져 이 인스턴스에서는 λᵁ=1000 이 가장 빨랐다 (belief-menu 69.5s, 100: 600s, 50: 3,010s).
6. α-B&B 를 비제로섬에 연결했다 (§9). 정확성 확인. α-B&B 도 λᵁ 경계 근처에서 노드가 폭증해 덜 민감하지 않다.
7. α·ϖ RLT 를 위해 넣은 Wcap (ϖ ≤ p·r) 이 Ω 를 크게 강화해서 Gurobi Ω 가 수백 초 → 1 초 안팎, Benders 전체가 수 초가 됐다.
   §7(E)·§8 의 시간 측정은 Wcap 이전 Ω 기준이다.
