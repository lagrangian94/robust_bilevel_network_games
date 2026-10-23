# 비제로섬 원고: 나중에 반영할 것

대상: `paper/joc-non_zero_sum/INFORMS-IJOC-Template.tex`, `supplementary.tex` (Overleaf). 반영할 때는 Overleaf 를 먼저 pull.

## 1. Benders 에서 spatial B&B 가 언제, 몇 번 불리는가 (2026-10-23 논의)

### 관찰 (S=20, 시간 제한 1 시간, `nonzero_sum/logs/loc_base_v2_seed*_S20_ALGL.log`, `loc_base_seed*_S20_ALG.log`)

| 실행 | B&B 호출 | 각 호출 시간 | local 로 얻은 type |
|---|---|---|---|
| ALGL seed 1 | 2 | 63, 61 s | 5 |
| ALGL seed 2 | 4 | 33, 72, 85, 487 s | 5 |
| ALGL seed 4 | 5 | 31, 74, 71, 349, 241 s | 5 |
| ALGL seed 3 | 2 | 47, 3,565 s (시간 초과) | 4 |
| ALG seed 2 (local 없음) | 7 | 11, 6, 22, 12, 75, 361, 246 s | 0 |
| ALG seed 3 (local 없음) | 7 | 12, 6, 4, 3, 60, 600, 2,406 s | 0 |

- ALGL 은 처음 12 iteration 정도를 type LP + local 해만으로 진행하고, B&B 는 그 뒤 2~5 번.
- B&B 는 마지막 후보 근처에 몰리고, 같은 x̄ 에서 여러 번 불리기도 한다. seed 2 는 4 번 모두 x = {1,2} 에서,
  V*({1,2}) 추정이 local −1327 → −1315 → −1274 → −1210 → −1210 (인증) 으로 계단식으로 오름.
  호출마다 τ⁻ = t̄₀ + m/2 를 넘는 점을 찾는 즉시 멈추기 때문. 끝까지 증명하는 건 마지막 호출뿐이라 그게 가장 김.
- local 해의 정확도는 인스턴스마다 다름: seed 1 의 x = {1} 은 local −990.92 대 전역 −990.96 (사실상 정확),
  seed 2 의 x = {1,2} 는 local 이 최악을 약 9% 과소평가.

### 설명 (원고에 넣을 내용)

후보 x* 의 최적성 증명에는 (a) 다른 모든 x 의 하한 — 아무 valid cut, local 해로 충분 — 과 (b) x* 한 점의 상한 — 전역 해법만 가능 — 이 필요하다.
local cut 은 항상 valid 한 하한이라 부정확해도 하한이 낮아질 뿐이다. 따라서 B&B 는 (1) 마지막 후보의 인증과
(2) local 이 어떤 후보를 지나치게 낮게 평가해 master 가 그 x 를 계속 고를 때만 필요하다. |X| 가 작으면 (후보지 4 곳, 결정 16 개)
master 가 고르는 유망한 x 가 몇 개뿐이라, local 이 그 몇 점에서 웬만큼 맞으면 하한이 일찍 최적에 붙고 B&B 는 주로 인증에 쓰인다.
원고 1164~1171 행 ("most iterations are completed by solving linear programs, and Algorithm abb is called mainly to certify the final candidate")
을 이 비대칭과 위 표로 보강할 것.

### 확인할 것 (미실행)

- local 해가 왜 이 문제에서 꽤 정확한가. 추측: 최악이 D 의 꼭짓점 쪽이라 경쟁하는 local 최적이 적음. 검증 방법:
  seed 몇 개에서 16 개 x 전부에 대해 local Ω 값과 전역 V*(x) 를 비교.

## 2. Algorithm 상자와 실험의 불일치: local 단계

- Algorithm (Benders with belief cuts) 상자에는 local 단계가 없고, 새 type 은 B&B 가 돌려준 ĥ 의 restricted ISP 에서만 나온다.
- 그런데 940 행은 "h̄ 는 local interior-point 해 또는 B&B incumbent", 실험 절 1381~1383 행은 "모든 Benders 변형이 같은 local solver 설정,
  Algorithm 은 SB-A 와 restricted ISP·belief cut 만 다르다" 고 써서, 실험은 local 단계가 있는 판 (코드의 ALGL, `local_first = true`) 으로 읽힌다.
- 결정할 것: 상자에 local 단계를 한 줄 넣거나, 실험을 local 없는 판으로 맞추기.
- 넣는다면 Prop finite 증명에 한 줄: local 단계는 매번 어떤 x̄ 에서 master 하한을 tol 이상 올리고, X 가 유한하며 각 x 에서 하한은 V*(x) 를
  넘지 못하므로 유한 번. (local 로 얻은 type 은 저장된 모든 type 보다 x̄ 에서 tol 이상 커서 중복은 없지만, Prop belief (iii) 의 P* 원소라는 보장은 없음.)
- 발표자료 Benders 슬라이드 마지막 줄 ("Spatial B&B at x̄ only to find a new type or to certify optimality") 도 이 결정에 맞출 것.

## 3. 같은 날 나온 다른 표현 문제

- Lemma phi (ii): "fixed at θᵁ" → "fixed at any θ ≥ θᵁ" (증명에서 L(θ) 비증가 + L ≥ φ 이므로 성립). θᵁ 는 uniform exact penalty parameter 중
  하나 (존재·poly encoding 용, 일반적으로 계산 불가), 계산 가능한 값은 θ_C, θ_G (보충 C.2), zero-sum 0, TU 상한.
- 본문 616 행 "the circuits are the cycles of the network", 1340 행 "whose circuits are the simple cycles of the network":
  보충의 circuit 정의 ((Ag, g) support 최소) 로는 단순 경로도 circuit 이 됨 → 보충 C.2 (모서리 방향 = 단순 사이클) 를 참조하도록.
  보충 Remark theta-circuit 의 location 단락은 2026-10-23 에 C.2 참조로 고침.
