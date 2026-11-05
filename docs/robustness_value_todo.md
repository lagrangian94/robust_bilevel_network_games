# 로버스트니스의 가치: 해야 할 것 (2026-11-04)

원고 6 절에서 우리 모형 (Coupled, D_δ) 이 nominal 보다 낫다는 OOS 증거가 두 게임 어디에도 없다. 아래 세 가지를 한다.

## 0'. 결정 (사용자, 2026-11-05 오전): ε̂ = ε̃ = 0.3, β = 0.4, δ = 0.1 로 두 게임 통일 (아래 0 절 대체)

- 근거: ε = 0.1 에서는 두 게임 모두 Coupled = Product (5 절). 비제로섬은 ε ≈ 0.3·β 중간에서만 결합이 결정을 바꿈. β 는 논문에서 두 게임 같은 값.
- Weissman 반경 (α = 0.05): ε = 0.3 ↔ N ≈ 28 (S = 3), 55 (S = 10), 95 (S = 20), 200 (S = 50).
- 비제로섬 계산 실험 (L1~L6, E, 2×2, δ·β, BD-G) 은 이미 이 설정 (독립 수요) → 그대로 씀. "요인 수요로 전부 다시" (0 절) 는 폐기.
- zero-sum 계산 실험은 다섯 네트워크 모두 ε = 0.3 으로 다시 (예전은 0.1 / 0.2).
- OOS 원칙 (사용자, 2026-11-05): 모형은 q̂ 로 한 번 풀고 x* 고정, OOS 는 진짜 분포·follower 믿음만 뽑아 평가 (원고 방식). 반복마다 표본으로 q̂ 를 새로 만들어 다시 푸는 데이터 기반 설계는 쓰지 않음 (파일럿 dd_pilot.jl 은 중단, 결과 폐기).

## 0. (대체됨) 결정 (사용자, 2026-11-04 밤): ε̂ = ε̃ = 0.1, δ = 0.1 로 두 게임 통일

- 지금까지: zero-sum 은 네트워크별 ε* (grid·polska·sioux_falls 0.1, abilene·nobel_us 0.2), 비제로섬은 ε = 0.3, δ 는 모두 0.1 절대값.
  → 결합 세기 ρ = δ/(2ε) 가 0.5 / 0.25 / 0.17 로 실험마다 달라 비교가 안 됨. ε* 는 Product 의 q₉₅ 기준으로 골라 한 모형에 유리.
  (2026-10-20 에 δ 를 "입지 실험과 같게" 절대값 0.1 로 바꾸면서 생긴 것. 그 전 zero-sum 결합 실험은 δ = ρ·2ε.)
- 새 규칙: 두 게임 모두 ε̂ = ε̃ = 0.1, δ = 0.1 (ρ = 0.5), β = 0.4. 값을 미리 고정하고 결과를 보고 바꾸지 않는다.
  근거: Weissman 반경 ε(S, N) = √((S ln 2 + ln(1/α)) / (2N)), α = 0.05 에서 ε = 0.1 ↔ N ≈ 500 (S = 10), 850 (S = 20), 1,900 (S = 50).
  ε 민감도 (예: 0.05, 0.1, 0.2, 0.3) 는 그림·보충자료로 따로.
- 비제로섬에서 ε = 0.1 로 가치가 안 보이면 ε 을 키우지 말고 시나리오 생성을 고친다.
- zero-sum 다시 돌릴 것 (abilene, nobel_us 를 ε = 0.1 로): Ours 시드 5 개, nob 시드 5 개, BD-G·BD-KKT seed 42. 나머지 세 네트워크 계산 결과는 그대로.
- 비제로섬은 요인 수요 + ε = 0.1 로 계산 실험 (L1~L6, E, 2×2, δ·β, BD-G, 인증) 과 가치 실험을 전부 다시 돌린다.
- 큐 v5 는 2026-11-04 22:33 에 멈춤 (δ·β 10 설정 끝, BD-G 나머지·인증 안 돌림). 기존 ε = 0.3 결과는 보관만.

## 1. Zero-sum: OOS 표에 Coupled 열 추가

- 지금 원고 tab:oos-summary 는 예전 실험이라 Nominal / Data-DR / Product 세 열뿐이고, 우리 모형 (Coupled, δ) 열이 없다.
- ε* 를 네트워크마다 "Product 의 q₉₅ 를 최소로 만드는 반경" 으로 골랐다 (Grid·Polska·Sioux Falls 0.1, Abilene·Nobel-US 0.2). 비교하는 모형 하나에 유리하게 고른 것이다. 계산 실험도 같은 네트워크별 ε* 로 돌렸고 (Abilene·Nobel-US 0.2), 비제로섬 (ε = 0.3 고정) 과는 규칙이 다르다.
- 기존 OOS 결과: `true_dro/factor_5/plots/eps_{0p1,0p2,0p5}_beta_0p4/*_factor_phaseAB.jls` (다섯 네트워크 모두).
- 계산 실험의 Coupled x* (ε = 0.1, 단 abilene·nobel_us 는 0.2, δ = 0.1, β = 0.4, 시드 42·1~4): `true_dro/factor_5/logs/nig_coupled/S_10/*_menu*.log`.

## 2. 비제로섬: 시나리오를 다시 제대로 생성해서 로버스트니스의 가치를 보인다

- 지금 pair 인스턴스 (SGB128, w = 150, S = 3) 에서는 Nominal / Data-DR / Coupled 가 모두 x* = {1,2} 로 같다. OOS 에서 로버스트가 nominal 을 이기는 장면이 없고, Product 만 {2} 로 16% 손해.
- pooled_store 기본 인스턴스는 모든 모형이 x* = {1,2}, −889.50 (영 결과).
- 독립 수요에서는 follower 모호성이 결정을 거의 안 바꿈 (메모 nz_follower_value: 560 개 중 3). 요인 수요 (`demand=:factor`) 에서는 follower 벌점 비가 0.19 → 0.83 으로 커짐.
- 시나리오 생성 방식을 바꾸면 계산 실험 (L1~L6, E, 2×2, BD-G, δ·β) 과의 일관성도 같이 정해야 한다.

## 3. 비제로섬: x 의 범위가 거의 같다 → Goyal 은 어떻게 다르게 했나 찾아본다

- 지금 인스턴스는 후보지가 사실상 둘이라 |𝒳| = 4 ({}, {1}, {2}, {1,2}) 이고, 모형마다 고를 수 있는 x 가 거기서 거기다.
- Goyal (비제로섬 목표 인스턴스 1~10 의 출처, 메모 nz_target_instances) 의 인스턴스는 후보지 수·출점 수·수요 구조를 어떻게 잡았는지, 리더 결정 공간을 어떻게 만들었는지 확인한다.
- **정정 (2026-11-05)**: SGB128 가치 인스턴스는 B 후보지 4 곳, N_B = 4 라 |𝒳| = 16 (loc_insample CSV 에 16 개 x). "|𝒳| = 4" 는 틀림 — 모든 모형이 {1,2} 를 고를 뿐.
- **Yanıkoğlu–Kuhn (SIOPT 2018) 7 절 확인 (2026-11-05, Optimization Online 판 15~19 쪽)**:
  - 예시 (SGB15): SGB128 에서 **인구가 가장 많은 15 개 도시** (처음 15 개 아님). 거리는 도로 거리 (Table 7.1, 마일). 15 곳 모두 출점 가능, A 는 Saint Louis·San Jose 두 곳 (용량 각 3/4 Σμ), B 는 나머지 13 곳 중 최대 3 곳 (|𝒳| = 378), 출점비 500, 용량 1/2 Σμ, 단위 수익 5. 수요 ξ_i ~ U[μ_i/2, 3μ_i/2] 독립, μ_i 는 인구에 따라 50~100 (연속분포, 결정규칙으로 풂). follower = 실현 수요를 보고 수송비를 최소화하는 가상 고객 (사전 결정 없음).
  - 무작위 실험 (Table 7.2): d ∈ {15, 20, 25, 30, 35}, 후보지 ‖s‖₁ ∈ {5, 10}, A 점포 ‖t‖₁ ∈ {2, 3, 5}, r ∈ {2, 3, 5}, q ∈ {750, 1000}, 좌표 [0,1]² 균등·유클리드 거리, 수요 U[50, 150], v = 5, 용량 150·(d − ‖s‖₁)/‖t‖₁, 설정마다 25 개. **Goyal 무작위 설정 1~10 과 우리 L1~L6 은 이 표의 행** (예: L1 = (15, 5, 2, 3, 750), L6 = (25, 10, 5, 5, 750)).
- **SGB15 가치 인스턴스 (2026-11-05, 사용자 결정: SGB15 + Kuhn 인구 기반 수요 S = 10 독립, 직선거리, OOS 는 공통 설계)**: 생성기 `make_sgb15_instance` (nz_data.jl), 풀이 `nonzero_sum/sgb15_insample.jl`, 데이터 `nonzero_sum/data/sgb15/` (Table 7.1, (Toronto, San Francisco) = 480 은 오기로 보임).
  - 1 마일 해상도·tol 1e-3 (`logs/sgb15/S10/`): Nominal·Data-DR 은 10~20 분에 Optimal, Product·Coupled 는 1 시간에 gap 1~2% (θ̄ = 5, θ̄·c^max ≈ 17,000 으로 상한이 느슨). L1·L3·L5 (같은 활성 차원) 의 Coupled S10 은 평균 30~54 초.
  - **10 마일 해상도·tol 5e-3 (계산 실험과 같음), θ̄ = 0.5, λ = 40θ̄ = 20 (`logs/sgb15/S10_res10/`)**: 네 모형 10~21 분에 Optimal. x*: N {San Francisco, Wichita, Toledo}, D {Washington, Sacramento, Wichita}, P {Toronto, San Francisco, Wichita}, C {Sacramento, Wichita, Toledo}.
  - 고정 x 교차 평가 (α-B&B gap 1e-3, scratchpad sgb15_cross.jl): 결정은 두 무리. N·P 의 x 는 N·P 목적 모두에서 비김 (구간 겹침), D·C 의 x 는 D 목적에서 비김. **Coupled 목적에서 x_C 가 다른 세 x 보다 확실히 좋음 (≥ 0.7%), Product 목적에서 x_C 는 2.6% 나쁨 → C ≠ P, C ≠ N 이 인증됨.** Coupled = Data-DR 무리, Product = Nominal 무리.
  - **공통 설계 OOS (loc_oos.jl LO_RES = sgb15, (γ, 구간) 마다 1,000 쌍, `logs/loc_oos_common/sgb15_S10_seed1_res10/`)**: Nominal 대비 개선 % (+ 가 좋음), 평균 / 가장 나쁜 5%, γ 전체.
    Coupled: Close −4.1 / +1.2, Moderate −3.6 / +1.7, Misspecified −3.3 / +2.7. Data-DR: −6.3 / +0.4, −6.0 / +1.0, −5.8 / +2.6. Product: −0.6 / −0.1 (Nominal 과 거의 같음).
    γ 별 Coupled 꼬리: γ ≤ 0.3 (진짜 분포가 멀리) +1.9~+3.7%, γ = 1 약 0, γ = 5 (ε 공 안) −2.5~−4.5%. 평균은 모든 γ 에서 −2.7~−5.5%.
    → Coupled 는 Data-DR 보다 평균·꼬리 모두 나음. Nominal 대비는 평균 3~4% 를 내주고 꼬리 1~3% 를 얻음 (꼬리 이득이 평균 손해보다 작음). seed 1 하나.
  - 남은 것: λ = 20 의 λ 무관 인증, seed 2~10.

## 4. 점검 결과 (2026-11-05)

기존 결과를 다시 확인한 것. 원고 6 절과 위 1~3 절의 전제를 바꾼다.

| # | 내용 | 근거 |
|---|---|---|
| 1 | 원고 tab:oos-summary 의 Nobel-US 행이 틀림. 원고 Product = [5.09, 12.87], CSV = **[7.24, 13.84]** (q₉₅ 가 Nominal 12.87 보다 나쁨). "Product attains the smallest q₉₅ on every network" 는 거짓 | `true_dro/factor_5/tables/oos_test.csv`, 나머지 네 행은 일치 |
| 2 | 원고 "five seeds, 100 cells" → 실제는 OOS 시드 42 하나, 네트워크당 20 칸 | CSV 행 수, jls 의 `:oos_seed` |
| 3 | 원고 "the follower recovers optimally under the drawn distribution" → 코드는 q̃ 를 q 와 **독립** 으로 뽑음. 결합 가정 d_TV(q, q̃) ≤ δ 가 거의 늘 깨지는 평가 | `plot_phaseAB_3way.jl:207-208, 224-230` |
| 4 | Coupled x* (δ = 0.1) = Product x* (다섯 네트워크 모두, 그때의 ε) | `logs/nig_coupled/S_10/*_menu.log` vs jls `:x_arcs` |
| 5 | ε = 0.1 에서 Nobel-US 는 세 모형 모두 {6,9}, Abilene 은 Data-DR = Product = {26,27} | `plots/eps_0p1_beta_0p4/*.jls` |
| 6 | 예전 OOS 의 Nominal x* 가 ε 폴더마다 다름 (grid: 0.1 폴더 {28,32}, 0.2 폴더 {28,37}). 지금 코드 (menu, tol 5e-3) 는 {28,37}. 허용오차 0.5% 가 모형 간 목적값 차이 (~0.6%) 와 비슷 → 가치 실험 x* 는 NS_TOL = 1e-3 | 스모크 2026-11-04 |
| 7 | 비제로섬 L1~L6 은 전부 독립 수요 (`pilot_pooled_store.jl` 에 demand 인자 없음) | grep |
| 8 | 원고 OOS 의 Phase B 는 α = γ(1 + ζ·randn) 을 0.01 로 자름 (ζ = 5 면 성분 ~40% 가 잘림), 원고에는 "Dir(γ q̃_ζ)" 로만 씀 | `plot_phaseAB_3way.jl:224-225` |

결정 (사용자, 2026-11-04): OOS 는 두 게임 모두 belief 를 진짜 분포에 묶어 추출 (p̃ = (1−τ)p + τu), d_TV(p, p̃) 세 구간 (≤ δ, (δ, 2ε], > 2ε) 으로 보고. 반복은 인스턴스 시드 5 개 (42, 1~4).
- zero-sum: `run_nig_coupled_scaling.jl` 에 `NS_EPS`·`NS_MODEL` 추가, 새 `oos_coupled_eval.jl`. x* 로그 `true_dro/factor_5/logs/value_eps0p1/S_10/`.
- 비제로섬: `loc_insample.jl`·`loc_oos.jl` 에 `NZ_DEMAND = iid | factor`, loc_oos 의 Moderate 경계 ε̃ → 2ε.
- 순서: 모형 간 x* 가 갈리는지 먼저 보고, 갈릴 때만 OOS. 계산 실험 재실행은 가치가 보인 뒤.

## 5. 밤사이 탐색 (2026-11-05 새벽): Coupled ≠ Product 인 설정 찾기

사용자 허락: ε·δ·β·인스턴스를 넓혀 Coupled 와 Product 의 결정이 갈리고 결합이 가치 있는 설정을 찾을 것. 논문에서 β 는 두 게임 같은 값.

### 5.3 zero-sum ε = 0.1 vs 0.3 x* 를 같은 결합 추출로 (2026-11-05 낮)

지표는 위쪽 꼬리 CVaR (draw 중 가장 나쁜 5% 평균) 로 바꾸기로 함 (사용자 동의). scratchpad `oos_eps_compare.jl` (oos_coupled_eval_sp.jl 사본, 구간 경계는 ε = 0.3 기준으로 고정해 두 ε 의 x* 에 같은 (p, p̃) 쌍), 출력 scratchpad `epscmp/`. seed 42 의 ε = 0.3 결과는 밤 실행과 draw 단위로 같음 (확인). 22 인스턴스 (polska s2, abilene s42, sioux_falls s4 는 ε = 0.3 x* gap 0.66~1.8% 라 제외). **Coupled = Product 22/22 (두 ε 모두).**

Close 구간, Nominal 대비 % (인스턴스마다 draw 합쳐 계산 후 시드 평균, 비용이라 음수가 좋음). Moderate·Misspecified 도 같은 방향.

| 네트워크 | 시드 | Data-DR ε=0.3 평균 / 꼬리 | Coupled ε=0.3 평균 / 꼬리 / 최대 | Coupled ε=0.1 평균 / 꼬리 / 최대 |
|---|---|---|---|---|
| Grid 5×5 | 5 | +4.0 / −0.3 | +1.3 / −3.6 / −4.3 | +2.2 / −1.9 / −2.7 |
| Polska | 4 | +5.3 / +2.5 | +5.8 / −10.3 / −12.0 | +1.3 / −4.7 / −6.2 |
| Sioux Falls | 4 | +2.4 / −2.3 | +5.2 / −8.5 / −10.7 | −0.7 / −3.8 / −2.8 |
| Abilene | 4 | −0.1 / −14.1 | −2.1 / −16.7 / −19.3 | −3.6 / −11.8 / −11.6 |
| Nobel-US | 5 | +0.9 / +2.5 | +7.2 / −2.6 / +1.7 | −0.3 / −0.1 / −0.1 |

- ε = 0.3: 꼬리 −4~−17%, 평균 +1~+7% (abilene 은 평균도 이득). polska·sioux·abilene 은 "평균 조금, 꼬리 확연히". Nobel-US 만 과보수 (평균 +7%, 꼬리 −3%, 최대 +2%).
- ε = 0.1: 평균은 거의 공짜 (−4~+2%), 꼬리 −2~−12%. Nobel-US 는 Nominal 과 거의 같음.
- seed 42 의 sioux (평균 +12.7%) 는 운이 나쁜 시드였음 (시드 평균 +5.2%).
- Data-DR (진짜 분포만 헤지) 은 꼬리 이득이 작거나 오히려 나쁨 (polska·nobel +2.5%) → zero-sum 에서 꼬리를 줄이는 것은 follower 믿음의 헤지. 결합은 결정에 영향 없음 (P = C).
- 결론: zero-sum 에 더 작은 ε 가 필요하다는 근거는 없음. ε = 0.3 이 꼬리 그림이 더 분명함.

### 5.0 아침 요약
1. **zero-sum 은 Coupled = Product.** ε = 0.1 의 25 개, ε = 0.3 의 25 개 (네트워크 5 × 시드 42·1~4) 모두 같은 결정 (50/50). β = 0 (ε̂ = ε̃) 이면 증명됨 (follower 의 목적 = 리더의 손실이라 최악의 belief 가 truth). β > 0 의 CVaR 경로는 고정 x 에서 gap 이 생길 수 있지만 (Polska {3,33}, ε = 0.3 에서 0.7%) 최적 결정은 안 바뀜. ε = 0.3 에서 Data-DR ≠ Product 16/25, Nominal ≠ Product 20/25 → follower 모호성은 결정을 바꾸지만 결합은 아님.
2. **비제로섬은 ε ≈ 0.3, β ∈ [0.2, 0.6] 에서만 Coupled ≠ Product.** ε ≤ 0.2 이면 retailer 예약이 belief 공 안에서 안 바뀌고, β 가 크면 CVaR 가 truth 와 무관해져 결합이 할 일이 없음. 원고의 SGB128 pair 는 ε = 0.3, β = 0.4 에서 δ ≤ 0.3 이면 갈림 (Product {2}, Coupled {1,2}, OOS 에서 Coupled 가 평균 +19~29%). 무작위 인스턴스는 5~10% 만 갈림.
3. **갈릴 때 대부분 Product 가 과보수 (점포를 덜 엶) 이고 Coupled 가 OOS 에서 이김.** 30 개 전체 평균 Coupled vs Product: 요인 수요 평균 +1.4% / 꼬리 +1.2~1.5%, 독립 수요 평균 +3.2% / 꼬리 +2.8~3.4%. **하지만 Coupled vs Nominal 은 짐:** 요인 수요 평균 −5.5% (꼬리 +1~2.5%), 독립 수요 평균 −5.7% · 꼬리 −5~6%. OOS 진짜 분포를 ε 만큼 멀리 뽑아도, retailer 응답을 pessimistic 으로 바꿔도 같은 결론. 원인은 OOS 진짜 분포가 q̂ 중심 대칭이라 Nominal 이 평균적으로 잘 맞는 구조로 보임 → Nominal 대비 가치는 데이터 기반 OOS (q̂ = 표본 추정) 로 봐야 함. **데이터 기반 파일럿 (SGB128 pair, p* = (0.2, 0.3, 0.5), q̂·p̃ 각각 28 개 표본) 에서 Coupled 가 9/9 반복 1 위 또는 공동 1 위** (Nominal 은 q̂ 잡음으로 3 번 틀린 결정, Product 는 8 번 과보수). 균등 p* 에서는 Coupled = Nominal (10/10 같은 결정), Product 만 과보수.
4. **추천 공통 설정: ε = 0.3, β = 0.4, δ = 0.1** (= 비제로섬 원래 설정, L1~L6 계산 결과 그대로 씀). zero-sum 계산 실험은 ε = 0.3 으로 다시 (abilene 은 ε = 0.3 에서 gap 1.3~1.8% 로 Stalled — 솔버에게 어려움).
5. **정할 것:** (가) 공통 설정 채택 여부 (ε = 0.1 의 Weissman N ≈ 500 근거를 버리고 ε = 0.3 ↔ N ≈ 55 (S = 10) 소표본 근거로), (나) 두 게임 OOS 진짜 분포 통일, (다) `oos_coupled_eval.jl` 에 "Stalled 이고 gap ≤ 0.5% 면 x* 사용" 규칙 넣기, (라) Nominal 대비 가치를 보일 OOS 설계 (데이터 기반: 진짜 분포 고정, q̂ = N 개 표본 추정, 표본마다 재풀이 — 비용 큼), (마) zero-sum 계산 실험 재실행.
탐색 스크립트·로그는 세션 scratchpad (pp_value.jl = pilot_pooled_store.jl 사본에 ε̂·ε̃·δ·수요 인자, zs_coupling_bind.jl = 고정 x 의 V_δ). 저장소 코드는 위 4 절 diff 외에 안 고침.

### 5.1 zero-sum: 결합 효과 없음 (이론 + 숫자)
- ε = 0.1, β = 0.4: 25 개 인스턴스 (네트워크 5 × 시드 42·1~4) 모두 Coupled x* = Product x* (`true_dro/factor_5/logs/value_eps0p1/S_10/`). 값도 1e-6 수준까지 같음.
- 고정 x 에서 V_δ (δ = 0, 0.02~0.1, ∞): Polska {13,34}·{3,33}, grid {27,28}·{28,31} 에서 ε ∈ {0.1, 0.2, 0.3}, β ∈ {0.4, 0.9} 모두 V_0 = V_∞ (차이 ≤ 1e-5 상대, B&B 허용오차 안).
- 이유: zero-sum 에서는 follower 의 오인이 리더에게 이득 → adversary 에게 최악은 belief = truth. β = 0, ε̂ = ε̃ 이면 V_δ = V_∞ (모든 δ) 를 증명할 수 있음 (h*(a) 가 E_a[flow] 최대화 → sup_d 는 d = a 에서). β > 0 은 belief 를 CVaR 기울기 r* 쪽으로 미는 경로만 남는데, 숫자로는 0 (belief 공 안에서 follower 의 복구 h 가 거의 안 바뀜).
- 원고: zero-sum 은 "Coupled = Product, 결합은 비제로섬에서만 의미" 로 정직하게. OOS 표는 Nominal / Data-DR / Product (= Coupled).
- 증명 스케치 (β = 0, ε̃ ≥ ε̂): 고정 x 에서 F_s(h) = 시나리오 s 의 최대 흐름, h*(d) ∈ argmax_h E_d[F(h)] (동률이면 리더에게 불리하게).
  V_δ(x) = sup_{(a,d) ∈ D_δ} E_a[F(h*(d))]. 집합이 포함 관계라 V_0 ≤ V_δ ≤ V_∞.
  임의의 (a, d) 에서 E_a[F(h*(d))] ≤ max_h E_a[F(h)] = E_a[F(h*(a))], 그리고 ε̃ ≥ ε̂ 이면 (a, a) ∈ D_0 → V_∞ ≤ V_0. 따라서 모든 δ 에서 V_δ = V_∞.
  핵심은 "follower 의 목적 = 리더의 손실" (zero-sum). β > 0 이면 리더 손실이 CVaR_β^a 라 follower 의 E_d 와 달라져 이 논증이 깨짐 (CVaR 기울기 경로). 비제로섬은 애초에 목적이 달라 belief ≠ truth 가 리더에게 더 나쁠 수 있음 → 결합이 의미를 가짐.
- ε = 0.3, β = 0.4, δ = 0.1, 25 개 (네트워크 5 × 시드 42·1~4, `true_dro/factor_5/logs/value_eps0p3/S_10/`, 표는 scratchpad `zs_xstar_table.sh value_eps0p3`): **Coupled x* = Product x* 25/25.** Data-DR = Product 9/25, Nominal = Product 5/25.
  - 많은 실행이 Optimal 이 아니라 Stalled (gap ≤ 0.5% 이 대부분). 0.5% 를 넘는 것: polska s2 (P 1.7%, C 1.6%), abilene s42 (P 1.8%, C 1.3%), sioux_falls s4 (C 0.66%). 이 셋은 P·C 의 x 가 같아도 최적 인증이 안 됨.
  - ε = 0.3 에서 zero-sum 계산은 ε = 0.1 보다 확실히 어려움 → 공통 설정을 ε = 0.3 으로 하면 zero-sum 계산 실험의 Optimal 비율이 떨어짐.

### 5.2 비제로섬: ε 가 커야 하고 β 는 중간
SGB128 pair (원고 가치 인스턴스, 독립 수요, S = 3, KKT 전수). Coupled ≠ Product 인 δ 범위:

| ε \ β | 0 | 0.2 | 0.4 | 0.6 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|
| 0.1 | | | 없음 (모두 {1,2}) | | 없음 | 없음 |
| 0.2 | | 없음 | 없음 | | | |
| 0.3 | 없음 (Product 도 {1,2}) | δ ≤ 0.3 | δ ≤ 0.3 | δ ≤ 0.2 | δ ≤ 0.03 | 없음 |
| 0.3 (요인 수요) | | δ ≤ 0.1 | δ = 0 만 | | 없음 | 없음 |

- 갈릴 때 Product = {2} (1 번 점포를 닫는 과보수), Coupled = Nominal = Data-DR = {1,2}.
- ε 이 작으면 retailer 의 예약 (LP 꼭짓점) 이 belief 공 안에서 안 바뀌어 follower 모호성 자체가 결정에 영향 없음 (ε̂ = 0.3 이어도 ε̃ ≤ 0.2 면 모두 {1,2}).
- β 가 크면 CVaR 가 truth a 와 무관하게 최악 시나리오만 봐서, a 를 d 에 묶는 결합이 할 일이 없음. β = 0 이면 Product 도 과보수가 안 됨 → 중간 β 가 최적.
- pooled_store (거점별 한도 300) 는 모든 설정에서 모형 간 같음 (예약이 한도에 붙어 belief 에 둔감).
- 결합 OOS (ε = 0.3, β = 0.4, δ = 0.1, `nonzero_sum/logs/loc_oos/loc_insample_sgb128_pair_e3b4_table/`): B 이익 평균/아래 5% CVaR — Product {2} 1072/942 (Close), 1080/942 (Moderate), 1073/918 (Misspecified) vs Coupled {1,2} 1280/1064, 1309/1024, 1382/965. Coupled 가 세 구간 모두 우세.

L1 계열 (d = 15, A = 2, B = 3, |X| = 8, S = 10, 시드 1~10, Benders):
- ε = 0.1, β = 0.4, 요인 수요, quota 300: P = C 10/10, D = P 10/10, N ≠ D 1/10 (seed 6). quota 750 도 P = C 10/10.
- ε = 0.3, β = 0.4, 독립 수요, quota 300 (원래 설정): **C ≠ P 1/10** (seed 2: Product {2,3}, Coupled δ = 0.1·0 모두 {3}).
- ε = 0.3, β = 0.4, 요인 수요, quota 300: 0/10 (seed 1 은 {1}·{2} 가 같은 값 −749.9985 인 비긴 해).
- B6N2 (d = 15, A = 2, B = 6, N_B ≤ 2, |X| = 22), 독립 수요, ε = 0.3, β = 0.4: 실제 차이 1/10 (seed 3: Product {}, Coupled {4}). seed 2 는 허용오차 안의 비긴 해 (Coupled 값이 Product 보다 높게 나옴 = 허용오차).
- 갈린 시드의 네 모형: L1 seed 2 — N {2,3}, D {3}, P {2,3}, C {3} (Coupled 가 N·P 둘 다와 다름). B6N2 seed 3 — N = D = C = {4}, P = {}.
- 결합 OOS (scratchpad nz_oos_random.jl, loc_oos 규약, S = 10 이라 u ~ Dir(0.2) 로 먼 belief 도 뽑음, 구간마다 300 쌍, B 이익):
  B6N2 s3: P {} 0 vs C {4} 357/307 (평균/아래 5%, Close), 360/230 (Misspecified) → Coupled 우세.
  L1 s2: P = N {2,3} 1746/1610 (Close), 1736/1546 (Misspecified) vs C = D {3} 1627/1507, 1595/1279 → Coupled 열세.
  OOS 진짜 분포 Dir(50·q̂) 는 d_TV(p, q̂) ≈ 0.17 < ε 라 보수적인 결정이 손해. 정리: Product 가 과보수일 때 Coupled 가 이김 (SGB128, B6N2 s3), Coupled 가 더 보수적일 때 짐 (L1 s2).
- 무작위 인스턴스에서 store 한도 (h_i ≤ 300) 는 예약을 한도에 붙여 belief 에 둔감 → pooled total (Σh ≤ 300·600) 시험. pair 는 θ 회로 상계 계산이 경로 수 초과로 실패해 제외.
  pooled total 은 결합이 값을 크게 바꿈 (L1 total 600 s6: −283 → −497, B6N2 total 300 s2: −473 → −781) 하지만 결정은 L1 total 300 1/10, total 600 0/10, B6N2 total 300 1/10.
- B6N2 요인 수요 시드 11~30 추가: 실제 차이 s16 하나 (s25 는 값이 같은 비긴 해) → B6N2 요인 수요 2/30. B6N2 독립 수요 시드 11~30: s24 → 2/30. B6N2 store 계열 합계 4/60.
- β = 0.2 (ε = 0.3): L1 독립 수요 1/10 (s4, β = 0.4 에서 갈린 s2 는 안 갈림), B6N2 독립 0/10, **L1 pooled total 300 2/10** (s1: P {} vs C {1}, s2: P {3} vs C {1}) — 지금까지 가장 높은 비율.

무작위 인스턴스에서 Product 와 Coupled (δ = 0.1) 의 결정이 실제로 갈린 비율 (ε = 0.3, S = 10, 비긴 해 제외):

| 계열 | β = 0.4 | β = 0.2 |
|---|---|---|
| L1 store 독립 (원래 설정) | 1/10 | 1/10 |
| L1 store 요인 | 0/10 | |
| L1 store quota 750 요인 | 0/10 | |
| L1 total 300 독립 | 1/10 | 4/30 (s1, s2, s16, s18) |
| L1 total 600 독립 | 0/10 | |
| B6N2 store 독립 | 2/30 | 0/10 |
| B6N2 store 요인 | 2/30 | |
| B6N2 total 300 독립 | 1/10 | 1/10 |
| B8N2 (d=20, B=8, N_B ≤ 2, \|X\| = 37) store 독립·요인 | 0/10, 0/10 | |
| 합계 | 7/140 (5%) | 6/60 (10%) |

후보를 늘려도 (B8N2) 갈리는 비율이 늘지 않음. 결합은 결정은 드물게 바꾸지만 값은 자주 크게 바꿈 (pooled total 에서 최대 75%).

결정이 갈린 인스턴스의 결합 OOS 정리 (ε = 0.3, β = 0.4 (표시한 것은 0.2), δ = 0.1, B 이익, 평균/아래 5% CVaR):

| 인스턴스 | N | D | P | C | OOS (C vs P) |
|---|---|---|---|---|---|
| SGB128 pair (S=3) | {1,2} | {1,2} | {2} | {1,2} | C 우세 (세 구간 평균 +19~29%, 꼬리 +5~13%) |
| B6N2 독립 s3 | {4} | {4} | {} | {4} | C 우세 (P 는 출점 안 해 이익 0) |
| B6N2 요인 s10 | {1,2} | {1,2} | {6} | {1} | C 가 P 보다 전 구간 우세, N 대비 Close 평균 −7% 지만 Moderate·Misspecified 꼬리 +11·+21% |
| B6N2 요인 s16 | {1,5} | {5} | {1,5} | {1} | P = N 대비 평균 −2~3%, Moderate·Misspecified 꼬리 +1·+12% |
| B6N2 독립 s24 | {6} | {6} | {4} | {6} | 거의 같음 (C 평균 +1%, Close 꼬리는 P 가 나음) |
| L1 독립 β=0.2 s4 | {1,2} | {1,2} | {1} | {1,2} | C 크게 우세 (평균 +57~65%, 꼬리 +21~50%) |
| L1 total β=0.2 s1 | {1} | {1} | {} | {1} | C 우세 (P 는 출점 안 해 이익 0, C 평균 ≈ 248) |
| L1 total β=0.2 s18 | {1,2} | {1,2} | {2} | {1,2} | C 우세 (세 구간 평균 +18%, 꼬리 +15%) |
| L1 독립 s2 | {2,3} | {3} | {2,3} | {3} | C 열세 (평균 −7~8%, 꼬리 −6~17%) |
| L1 total s2 | {1} | {1} | {3} | {1} | 같음 (OOS 의 꼭짓점 응답에서 두 점포가 같은 결과) |
| B6N2 total s10 | {2,6} | {2,6} | {2,6} | {1,2} | 같음 (같은 이유) |

| B6N2 total β=0.2 s2 | {6} | {2} | {2} | {3} | 같음 (단일 점포 결정이 OOS 에서 모두 같은 이익) |

B6N2 요인 수요 30 개 전체 (ε = 0.3, β = 0.4, δ = 0.1) 네 모형 x*: C ≠ P 3 (실제 2), **C ≠ N 8**, D ≠ N 4, P ≠ N 7 (scratchpad nz4table.sh). 결정이 하나라도 다른 9 개 시드의 OOS 를 30 개 평균 (같은 결정은 0) 으로 집계 (nzoos_agg.sh):
- C vs Product: 평균 +1.5 / +1.4 / +1.4%, 아래 5% CVaR +1.5 / +1.2 / +1.2% (Close / Moderate / Misspecified) — Coupled 우세 (s10 의 +45% 가 대부분).
- C vs Nominal: 평균 −5.9 / −5.5 / −5.3%, CVaR +1.2 / +2.2 / +2.5% — 평균을 내주고 꼬리를 조금 얻음. s26 (−35%), s15 (−28%) 처럼 크게 지는 인스턴스도 있음.

같은 OOS 를 진짜 분포 Dir(10·q̂) (E[d_TV(p, q̂)] ≈ 0.35 ≈ ε, NO_CP = 1) 로 다시: C vs P 평균 +1.3~1.4%, CVaR +0.8~1.6%; C vs N 평균 −5.3~5.7%, CVaR +1.9~2.3% → 같은 결론.

B6N2 독립 수요 30 개 전체 네 모형: C ≠ P 3, C ≠ N 8, D ≠ N 11, P ≠ N 10. OOS (NO_CP = 5) 30 개 평균:
- C vs Product: 평균 +3.2%, CVaR +2.8~3.4% (s3 의 +100% 가 대부분).
- C vs Nominal: 평균 −5.6~5.9%, **CVaR 도 −5.1~6.2%** — 로버스트한 결정 (C ≠ N 인 8 개 중 대부분이 D = P = C) 이 꼬리에서도 짐.
- NO_CP = 1 로 다시: C vs P 평균 +3.2~3.3%, CVaR +3.1~3.4%; C vs N 평균 −5.9~6.1%, CVaR −4.3~5.5% → 같은 결론.
- 가설 "OOS 의 꼭짓점 응답이 덜 비관적이라서" 는 **기각**: retailer 응답을 in-sample 처럼 pessimistic (p̃ 아래 최적값을 유지하는 (h, y) 중 진짜 p 아래 B 손실 최대, scratchpad nz_oos_random.jl NO_PESS = 1) 으로 바꿔도 독립 수요 C vs N 평균 −5.3~5.7%, 꼬리 −5.0~6.4%; 요인 수요 평균 −5.8~6.2%, 꼬리 +1.2~2.3%; C vs P 는 +0.8~3.4% 그대로.
- 더 그럴듯한 원인: OOS 진짜 분포가 q̂ 중심 대칭 (평균 = q̂) 이라 q̂ 에 맞춘 Nominal 이 평균적으로 잘 맞음. 데이터 기반 DRO 의 가치는 q̂ 가 유한 표본의 잡음 섞인 추정이라 진짜와 체계적으로 어긋날 때 드러남 → "Nominal 대비 가치" 는 진짜 분포 고정 + N 개 표본으로 q̂ 생성 + 표본마다 모형 재풀이 (메모 data_driven_oos_design) 가 필요.

**데이터 기반 OOS 파일럿 (SGB128 pair, scratchpad dd_pilot.jl):** 진짜 p* 고정, 반복마다 리더 q̂ = 28 개 표본의 경험분포 (S = 3 에서 ε = 0.3 ↔ N ≈ 28), retailer belief p̃ = 따로 28 개 표본 (둘 다 p* 에서 → 자연스러운 결합), q̂ 로 네 모형 KKT 전수, x* 를 p*·p̃ 아래 평가.
- p* = (0.2, 0.3, 0.5), 10 반복: **Coupled 가 매번 1 위 또는 공동 1 위.** 평균 / 최저 20%: Nominal 1157.8 / 1065.2, Data-DR 1184.8 / 1124.3, Product 1065.2 / 1065.2, **Coupled 1196.6 / 1183.4** (Nominal 대비 +3.3% / +11%). Nominal 은 q̂ 잡음 때문에 3 번 {2} 를 골라 매번 −135, Product 는 거의 늘 {2}. 단서: KKT 평가 25 번이 시간 제한 120 s 에 걸려 incumbent 값으로 argmin.
- p* = 균등, 10 반복 (시간 제한 600 s, 걸린 것 8 번): **Coupled = Nominal = Data-DR**, 매번 {1,2} (평균 / 최저 20% 1300.5 / 1230.0). Product 는 9/10 번 {2} (1115.4 / 1076.6, −14% / −12%). 균등이면 q̂ 잡음이 Nominal 의 결정을 바꾸지 않아 Coupled 의 이점이 없음.
- p* = (0.5, 0.3, 0.2): KKT 가 느려 (시간 제한 900 s) 진행 중. 지금까지 3 반복 모두 네 모형 {1,2}.
- 요약: 데이터 기반 설계에서 **Coupled 는 Nominal 에 지지 않음** (q̂ 잡음이 Nominal 을 틀린 결정으로 끌 때 이기고, 아니면 같음). Product 는 진짜 분포와 상관없이 과보수. 대칭 OOS 에서 Nominal 에 지던 결과와 다름 → 원고의 비제로섬 가치 실험은 데이터 기반 OOS 로 하는 것이 맞아 보임. 다만 "Coupled 가 Nominal 을 이긴다" 가 아니라 "Product 의 과보수 없이 q̂ 잡음에 강하다" 로 써야 함.

**두 게임의 OOS 진짜 분포가 다름 (정해야 함):** 비제로섬 p ~ Dir(5·S·q̂) 는 q̂ 근처 (d_TV(p, q̂) ≈ 0.17 < ε) → Nominal 에 유리. zero-sum 은 Dir(γ·1), γ ≤ 1 → q̂ 에서 멀리 (d_TV 0.5~0.9) → 로버스트 모형에 유리. 통일 후보: 두 게임 모두 E[d_TV(p, q̂)] ≈ ε 이 되도록 집중도를 맞춤.

OOS 평가의 한계: retailer 예약을 꼭짓점 하나로 정함 (in-sample 모형은 같은 최적값의 예약 중 리더에게 가장 불리한 것). pooled total 의 세 사례는 OOS 에서 구분이 안 됨 → pooled total 에서 in-sample 로 보인 결정 차이는 pessimistic follower 의 동률 처리에서 나왔을 가능성이 큼 (가치 근거로 쓰지 말 것). store 한도·SGB128 pair 의 사례는 OOS 에서도 차이가 보임.

zero-sum ε = 0.3, β = 0.4, seed 42: 다섯 네트워크 모두 P = C (grid {28,33}, polska {6,33}, sioux {35,72}, abilene {5,11}, nobel {22,25}).
Data-DR 은 다섯 네트워크 모두 P 와 다름 (grid {28,31}, polska {13,34}, sioux {54,72}, abilene {26,27}, nobel {6,9}) → ε = 0.3 의 zero-sum 에서 follower 모호성 자체는 결정을 바꾸지만 (D ≠ P) 결합은 바꾸지 못함 (P = C). 위의 이론 (최악 belief 가 truth 에 붙음) 과 맞음.
Nominal (seed 42): grid {28,37}, polska {3,33}, sioux {35,73}, abilene {5,26}, nobel {6,9}.

zero-sum 결합 OOS (seed 42, ε = 0.3, β = 0.4, δ = 0.1, 구간마다 200 쌍 × γ 4 × ζ 5, 리더 CVaR 비용, 낮을수록 좋음, `true_dro/factor_5/logs/oos_coupled_eps0p3/`, scratchpad 사본 oos_coupled_eval_sp.jl = Stalled gap ≤ 5e-3 허용). abilene 은 Product x* gap 1.8% 라 제외.

| 네트워크 | 구간 | Nominal q95 / 폭 / 평균 | Data-DR | Product = Coupled |
|---|---|---|---|---|
| grid | Close | 38.10 / 19.17 / 28.80 | 39.23 / 16.49 / 30.55 | **35.89 / 12.97** / 29.98 |
| | Misspecified | 36.62 / 19.79 / **27.03** | **35.45** / 14.82 / 28.68 | 36.50 / 14.97 / 29.37 |
| polska | Close | 20.11 / 6.82 / **15.88** | 20.29 / 7.21 / 15.68 | **18.23 / 3.99** / 16.31 |
| | Misspecified | 17.55 / 6.04 / **14.57** | **16.97 / 4.56** / 15.05 | 17.40 / 5.85 / 15.10 |
| sioux | Close | 21.53 / 10.34 / 16.10 | 21.36 / 9.69 / **15.69** | **20.59 / 4.92** / 18.15 |
| | Misspecified | 21.18 / 10.47 / 15.21 | 19.89 / 9.89 / **14.78** | **19.66 / 5.47** / 17.35 |
| nobel | Close | 16.01 / 7.37 / **11.13** | = Nominal | **15.87 / 5.90** / 12.14 |
| | Misspecified | **12.22** / 5.68 / **9.53** | = Nominal | 13.72 / 5.60 / 10.90 |

요약: Product (= Coupled) 는 Close 구간에서 q95 를 1~9%, 폭을 20~52% 줄이고 평균은 3~13% 내줌. belief 가 멀면 (Misspecified) Data-DR 의 꼬리가 더 좋은 경우가 많음. nobel 은 ε = 0.3 이 과보수 (평균·q95 모두 손해, Close q95 만 조금 이득). abilene 은 gap 1.3~1.8% 에서 Stalled (Coupled 실행 중 109 반복에 LB 15.78 / UB 17.52 까지 벌어진 적 있음) → ε = 0.3 의 zero-sum 은 솔버에게도 어려움.
- zero-sum ε = 0.3, β = 0.4 x* (네 모형, `true_dro/factor_5/logs/value_eps0p3/`) 진행 중. Polska Product x* = {6,33} (ε = 0.1 에서는 {13,34}). 고정 x 에서 처음으로 gap: Polska {3,33} 의 V_∞ 20.454 vs V_0.1 20.313 (0.7%, d_TV(a, d) = 0.21) — ε = 0.3 이면 zero-sum 에서도 CVaR 기울기 경로가 열림.

Goyal (arXiv 2206.03531) 확인: 리더 = 진입 기업 B, 결정은 출점 위치만, X = {x : x + l^A ≤ l^S, ‖x‖₁ ≤ N_B}. 무작위 설정 1~10 은 N_B = 적격지 − A 점포라 |X| = 2^{N_B} = 8 / 4 / 32 (우리 L1~L6 과 같음). 결정 비교는 d = 8 기본 사례에서만 (γ₁ = 0 → {3,4}, γ₁ ≥ 1 → {2}), 무작위 설정은 gap·시간만 보고. Yanıkoğlu–Kuhn 의 SGB15 예는 후보 13, r = 3 → |X| = 378 (유일하게 출점 수 제한이 걸리는 예).
