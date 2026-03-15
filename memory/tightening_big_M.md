# Big-M / P-bound Tightening Analysis

## 문제
Strict Benders의 lower bound가 0에서 stall → OSP dual의 P-bound penalty term이 과도하게 차감되어 cut이 약함.

## 논문 vs 코드: P-bound 상한 비교

| 파라미터 | 적용 변수 | 의미 | 코드 사용값 |
|---------|----------|------|-----------|
| **ϕU** | Φ̂, Φ̃ (flow LDR) | flow LDR 계수 범위 | ϕU ✅ |
| **πU** | Π̂, Π̃ (node price LDR) | node price dual LDR 범위 | ϕU ❌ |
| **yU** | Ỹ (follower recovery LDR) | recovery LDR 범위 | ϕU ❌ |
| **ytsU** | Ỹ_ts (dummy arc LDR) | dummy arc LDR 범위 | ϕU ❌ |

## 코드에서 ϕU가 잘못 사용되는 위치

### OSP (`build_dualized_outer_subprob.jl`)
- L166-167: `Uhat1`, `Uhat3` — ϕU ✅
- L172-173: `Phat1_Π`, `Phat2_Π` — should be πU ❌
- L174-175: `Ptilde1_Π/Y/Yts`, `Ptilde2_Π/Y/Yts` — should be πU/yU/ytsU ❌

### Full model (`build_full_model.jl`)
- L127,129: `Φhat`, `Φtilde` bounds — ϕU ✅
- L134-135: `Πhat`, `Πtilde` bounds — should be πU ❌
- L138: `Ytilde` bounds — should be yU ❌
- L141: `Yts_tilde` bounds — should be ytsU ❌
- L255-262: Big-M (`Ψ ≤ ϕU·x`) — ϕU ✅ (Ψ = diag(x)Φ이므로)

### ISP leader (`nested_benders_trust_region.jl`)
- L1268-1269: `Phat1_Π`, `Phat2_Π` — should be πU ❌

### ISP follower (`nested_benders_trust_region.jl`)
- L1467-1468: `Ptilde1_Π/Y/Yts` — should be πU/yU/ytsU ❌

### IMP (`nested_benders_trust_region.jl`)
- L197-198 (leader part): Π에 πU ❌
- L243-244 (follower part): Π/Y/Yts에 πU/yU/ytsU ❌

## Lower bound stall 메커니즘
OSP(max)에서 `-ϕU * sum(P)` term은 penalty. ϕU가 실제보다 크면:
1. P 변수 여유 공간 증가
2. 목적함수 penalty 과다 차감
3. Benders cut 기울기 약화 → LB stall at 0

## Tight bound 추정 아이디어

### Idea 1: Full model solution에서 역산
Full model (Pajarito)을 풀면 Φ, Π, Y, Yts의 실제 optimal 값을 알 수 있음.
→ `max|Π*| / max|Φ*|` 비율로 πU/ϕU 비율 추정 가능.
**장점**: 정확. **단점**: full model을 먼저 풀어야 함 (대형 인스턴스에서 불가).

### Idea 2: LP relaxation에서 추정
Full model의 binary relaxation (x ∈ [0,1])을 풀면 빠르게 LDR 범위 추정 가능.
→ LP relaxation의 optimal LDR 값 × safety factor (e.g., 2배).

### Idea 3: 물리적 해석 기반 analytic bound
- **πU**: node price = flow conservation dual. max-flow value로 bound 가능.
  `πU ≤ max_flow_value` (capacity sum upper bound 등)
- **yU**: follower recovery variable. `yU ≤ w` (recovery budget bound)
- **ytsU**: dummy arc 1개이므로 `ytsU ≤ max_flow_value`
- 이 방법이 가장 실용적: full model 없이도 network 구조에서 유도 가능.

### Idea 4: Adaptive tightening (iterative)
1. 초기에는 큰 ϕU 사용
2. 첫 몇 iteration의 ISP solution에서 Π, Y, Yts 범위 관측
3. 관측 범위 × safety factor로 bound 축소
4. 축소된 bound로 재시작
**주의**: bound를 optimal 아래로 줄이면 infeasible → safety factor 필요.

### Idea 5: 논문 참조
Manuscript에서 πU, yU, ytsU의 유도 과정이 있는지 확인 필요.
LP strong duality나 complementary slackness로부터 유도할 수 있을 가능성.
