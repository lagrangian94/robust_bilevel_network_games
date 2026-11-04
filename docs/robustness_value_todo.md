# 로버스트니스의 가치: 해야 할 것 (2026-11-04)

원고 6 절에서 우리 모형 (Coupled, D_δ) 이 nominal 보다 낫다는 OOS 증거가 두 게임 어디에도 없다. 아래 세 가지를 한다.

## 1. Zero-sum: OOS 표에 Coupled 열 추가

- 지금 원고 tab:oos-summary 는 예전 실험이라 Nominal / Data-DR / Product 세 열뿐이고, 우리 모형 (Coupled, δ) 열이 없다.
- ε* 를 네트워크마다 "Product 의 q₉₅ 를 최소로 만드는 반경" 으로 골랐다 (Grid·Polska·Sioux Falls 0.1, Abilene·Nobel-US 0.2). 비교하는 모형 하나에 유리하게 고른 것이고, 계산 실험 (ε = 0.1 고정) 과 비제로섬 (ε = 0.3 고정) 의 규칙과도 다르다.
- 기존 OOS 결과: `true_dro/factor_5/plots/eps_{0p1,0p2,0p5}_beta_0p4/*_factor_phaseAB.jls` (다섯 네트워크 모두).
- 계산 실험의 Coupled x* (ε = 0.1, δ = 0.1, β = 0.4, 시드 42·1~4): `true_dro/factor_5/logs/nig_coupled/S_10/*_menu*.log`.

## 2. 비제로섬: 시나리오를 다시 제대로 생성해서 로버스트니스의 가치를 보인다

- 지금 pair 인스턴스 (SGB128, w = 150, S = 3) 에서는 Nominal / Data-DR / Coupled 가 모두 x* = {1,2} 로 같다. OOS 에서 로버스트가 nominal 을 이기는 장면이 없고, Product 만 {2} 로 16% 손해.
- pooled_store 기본 인스턴스는 모든 모형이 x* = {1,2}, −889.50 (영 결과).
- 독립 수요에서는 follower 모호성이 결정을 거의 안 바꿈 (메모 nz_follower_value: 560 개 중 3). 요인 수요 (`demand=:factor`) 에서는 follower 벌점 비가 0.19 → 0.83 으로 커짐.
- 시나리오 생성 방식을 바꾸면 계산 실험 (L1~L6, E, 2×2, BD-G, δ·β) 과의 일관성도 같이 정해야 한다.

## 3. 비제로섬: x 의 범위가 거의 같다 → Goyal 은 어떻게 다르게 했나 찾아본다

- 지금 인스턴스는 후보지가 사실상 둘이라 |𝒳| = 4 ({}, {1}, {2}, {1,2}) 이고, 모형마다 고를 수 있는 x 가 거기서 거기다.
- Goyal (비제로섬 목표 인스턴스 1~10 의 출처, 메모 nz_target_instances) 의 인스턴스는 후보지 수·출점 수·수요 구조를 어떻게 잡았는지, 리더 결정 공간을 어떻게 만들었는지 확인한다.
