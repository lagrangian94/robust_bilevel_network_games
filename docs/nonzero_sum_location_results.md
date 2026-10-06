# 비제로섬 확장: location 구현과 belief LP cut 검증 (2026-10-06 ~ 10-07)

이 문서는 측정 결과만 다룬다. 수식과 유도는 `docs/nonzero_sum_extension.html`
(게시본 https://claude.ai/artifact/B2bkdVY57TmeXDh2GRtH27) 에 있다.

- 코드: `nonzero_sum/`
  - `nz_data.jl` — 일반 LP recourse 데이터 (`NZData`), location 인스턴스, 원고 max-flow 변환
  - `nz_omega.jl` — 비제로섬 Ω (html Step 4′, θ = θᵁ 고정), 확장 belief 고정 LP, follower 인증서
  - `nz_kkt_eval.jl` — big-M 없는 독립 평가기 (KKT 상보성 indicator + r·φ bilinear) 와 θ 진단
  - `nz_benders.jl` — 표준 Benders, belief-menu Benders (확장 belief)
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

### (B)~(E)

(재실행 중 — 첫 실행은 (B) 에서 스크립트 스코프 오류로 중단, `main()` 으로 감싸 재실행)
