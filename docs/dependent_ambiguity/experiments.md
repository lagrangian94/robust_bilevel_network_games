# 원고 6 장 실험 계획 — 종속 ambiguity set (belief 결합 δ) 반영 (2026-10-08 갱신)

원고: `paper/joc-non_zero_sum/INFORMS-IJOC-Template.tex` 6.1 (location, 커밋 eb78f45 "실험 설계 골격", 38f0ab0 "인스턴스 크기 방침").
이 문서는 원고 설계를 코드·실행 단위로 옮긴 것과 지금까지의 결과.

## 0. 확정된 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| δ 실험 방식 | **A: ε̂ = ε̃ = ε 고정, δ sweep** (끝점 δ ≥ 2ε = product) | 주변분포 (투영) 를 고정하고 결합만 바꿈 → 결합 고유 효과 분리. 원고 문장 "Coupled for a range of δ, whose endpoint δ ≥ ε̂+ε̃ is the product set" 와 일치 |
| type 수 \|𝒫\| | Figure loc-delta 에서 **뺌** | Theorem game-theoretic (iii): 𝒫 는 ε̃' = min(ε̃, ε̂+δ) 로만 바뀜. A 에서는 ε̃' = ε 상수 |
| 기본 인스턴스 | Goyal 기본 사례 (SGB128 8 도시) + 예약, **quota w_i = 150** (기존 100 에서 변경) | quota 100 은 최적점에서 결합 효과 0, 150 은 δ 가 결정을 바꿈 (§2). 기존 결과는 표 1 개 + Benders 1 회라 재실행 비용 작음 |
| 인스턴스 역할 | 기본 = 모형 해석 (loc-values, loc-insample, loc-delta, loc-oos) + S 스케일링 비교. 무작위 = 크기 스케일링 (loc-comp) | 원고 6.1 Instances and Setup, Goyal §7.2 / §7.3 구성 |
| E (KKT 전수) | **기본 인스턴스에서만 보고**. 개선 시도 후 안 풀리면 "미해결" 로 정직하게 | 무작위 d = 15 에서 S = 3 부터 시간 제한 (§3) |
| 시간 제한 | 방법당 3,600 s (모든 oracle 호출을 남은 시간으로 자름), E 는 x 하나당 300 s. 작은 S 에서 실패한 방법은 큰 S 생략 | 안 풀리는 문제에 시간 쓰지 않음 |
| 남은 결정 | ε = 0.3 의 근거 (validation 또는 표본 수 규칙) | 현재는 효과가 보이는 값으로 고른 것 |

## 1. 방법 (원고 6.1 Algorithms)

| 이름 | 내용 | 코드 |
|---|---|---|
| E | 모든 x 에서 V*(x) 를 KKT 단일수준 MINLP 로 (Benders·minimax·big-M 없음) | `nz_kkt_value` (원래), `nz_kkt_value_reduced` (축소판, §3) |
| SB-G | 표준 Benders: Ω local 해 (Gurobi OptimalityTarget=1) cut 먼저, 위반 cut 없을 때만 Gurobi 전역 Ω. 전역 호출에는 남은 시간 전체 | `nz_standard_benders(...; oracle=:gurobi, local_first=true, oracle_remaining=true)` |
| SB-A | 같음, 전역은 α-B&B (남은 시간 전체) | `nz_standard_benders(...; oracle=:alpha_bnb, local_first=true, oracle_remaining=true)` |
| Algorithm 2 | belief-menu + α-B&B. oracle 600 s, 같은 후보 재방문 시 2,400 s (첫 호출이 부정확했다는 뜻이므로), 남은 시간으로 자름 | `nz_belief_menu_benders` |

공통: ε = 1e-4. Magnanti–Wong: Algorithm 2 (menu cut, restricted cut) 와 SB-A 의 전역 (restricted) cut 에 적용. SB-G 의 전역 cut (Ω 해) 과 local cut 에는 없음 (MW 를 하려면 bilinear 문제를 한 번 더 풀어야 함).
유한 수렴: local cut 은 x̄ 에서 허용오차 이상 위반될 때만 추가 → 같은 x̄ 에서 t₀ 가 매번 tol 이상 오르고 V*(x̄) 로 막혀 있음 → x̄ 마다 유한, 𝒳 유한.

## 2. 기본 인스턴스 (SGB128 + 예약 quota 150, S = 3, seed 1, β = 0.4, ε = 0.3)

### Table loc-insample — 완료 (`nonzero_sum/loc_insample.jl`, 로그 `logs/loc_insample_sgb128_pair_q150.log`, KKT 160 회 모두 OPTIMAL)

| 모형 | ε̂ | ε̃ | δ | w = 0: x* | 값 | w = 150: x* | 값 |
|---|---|---|---|---|---|---|---|
| Nominal | 0 | 0 | – | {1} | −1328.06 | {1,2} | −873.39 |
| Data-DR | 0.3 | 0 | – | {1} | −1194.50 | {1,2} | −860.50 |
| Coupled | 0.3 | 0.3 | 0.1 | {1} | −1194.50 | {1,2} | −860.50 |
| Coupled | 0.3 | 0.3 | 0 | {1} | −1194.50 | {1,2} | −865.08 |
| Coupled (product) | 0.3 | 0.3 | ≥ 0.6 | {1} | −1194.50 | **{2}** | −836.00 |

- w = 0: belief 가 영향을 못 줌 → Data-DR = Coupled = product (원고 설계대로).
- w = 150: product 만 결정을 바꿈 ({1,2} → {2}, 과도하게 보수적). 결합하면 {1,2} 로 돌아옴.

### Figure loc-delta 데이터 — 완료 (A 방식, ε̂ = ε̃ = 0.3)

| δ | 0 | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 | ≥ 0.6 (product) |
|---|---|---|---|---|---|---|---|---|---|---|
| x* | {1,2} | {1,2} | {1,2} | {1,2} | {1,2} | {1,2} | {1,2} | **{2}** | {2} | {2} |
| 값 | −865.1 | −861.4 | −860.5 | −860.0 | −854.8 | −849.6 | −844.4 | −836.0 | −836.0 | −836.0 |

결정 전환 δ ∈ (0.3, 0.4). (같은 로그의 ε̃ = 0.1, 0.2 선은 B 방식이라 쓰지 않음.)

### quota 선택 근거 (`screen_coupling_decision.jl`, `logs/screen_coupling_decision_sgb128_pair.log`, ε̂ = ε̃)

| quota | 최적점에서 순수 결합 효과 |
|---|---|
| 100 (기존 원고) | 없음. δ 효과는 ε̂ ≠ ε̃ 일 때 유효 반경을 통해서만 (`logs/sweep_coupling_kkt_sgb128_pair_q100_seed1_S3.log`) |
| 150 | ε = 0.3 이면 β ∈ {0.2, 0.4, 0.6} 모두 δ ≤ 0.1 에서 x* 변경 |
| 200 | β = 0.2 에서 값만 변화 (최대 3.4%) |
| 300 | 없음 |

참고: 원고 tex 의 TODO (−871.67 vs −911.8) 는 quota 불일치 (결합 로그 300, 원고 100) 가 원인.

### S=3 스모크 (`logs/loc_base_scaling_smoke_S3.log`, 방법당 900 s, δ = 0.1) — 2026-10-08 저녁

| 방법 | 결과 | LB 가 최적 (−860.5) 에 도달 | 막힌 곳 |
|---|---|---|---|
| E (KKT 축소판) | **Optimal 14 s** (16 개 x 전부) | — | — |
| SB-G | TimeLimit, LB −860.50, UB −639.4 (gap 35%) | local cut 5 개, 수 초 | x={1,2} 인증용 Gurobi 전역 1 회가 남은 시간 전부 |
| SB-A | TimeLimit, LB −860.52, UB +1848 | local cut 5 개, 수 초 | x={1,2} 인증용 α-B&B 1 회 |
| Algorithm 2 | TimeLimit, LB −860.50, UB +2169 | belief 5 개, ~10 s | x={1,2} α-B&B 600 s + 290 s |

수정한 버그: Gurobi local 모드는 `LOCALLY_SOLVED` 를 반환하는데 `nz_solve!` 가 이를 오류로 처리했고, local 단계의 try/catch 가 오류를 삼켜
SB-G/SB-A 가 사실상 전역만 쓰는 Benders 로 돌았다 (이전 두 실행 `loc_base_scaling_v0/v1_aborted.log` 는 무효). try/catch 제거, 상태 명시 검사.

### α-B&B 상한 진단 (`diag_abb_base.jl`, x = {1,2}, ε = 0.3, 600 s, worker 12, `logs/diag_abb_base_x12.log`)

| quota | δ | KKT V* | α-B&B LB | α-B&B UB | gap | 노드 |
|---|---|---|---|---|---|---|
| 100 | Inf | −1283.5 | −1283.54 | −142.2 | 89% | 37,836 |
| 100 | 0.1 | −1283.5 | −1283.54 | +248.1 | 119% | 19,283 |
| 150 | Inf | −1437.0 | −1437.06 | +980.9 | 168% | 32,219 |
| 150 | 0.1 | −1470.5 | −1470.52 | +1531.2 | 204% | 17,960 |

- α-B&B 는 해는 즉시 찾지만 상한이 느슨하다. quota (h^r 상자 폭) 가 클수록, 결합이 있을수록 (노드 처리 속도 절반) 나빠짐.
- 같은 점에서 Gurobi Ω (spatial B&B) 상한 −1249 (900 s) 가 α-B&B 보다 훨씬 좋고, KKT 는 수 초에 정확한 값.
- 3,600 s 추이 (quota 150, δ 0.1, `logs/diag_abb_base_x12_q150_d0p1_3600.log`): α-B&B 상한 60 s +3611 → 1 h +540 (gap 137%, 35k 노드),
  Gurobi Ω 상한 1 h −1405.5 (gap 4.4%, 8,774 만 노드, 루트 +213,849). **시간을 늘려도 α-B&B 는 닫히지 않음.**
- λᵁ = 10 (원고에서 정확함을 확인한 값, π̃ 상한 = λᵁ·piFU 가 10 배 작아짐) 도 개선 없음: 600 s α-B&B +1869, Gurobi −919
  (`logs/diag_abb_base_x12_q150_d0p1_lam10.log`) → 원인은 λᵁ 가 아님. 후보: 예약 dual ϖ (상한 p = 200) × h^r (0~150) bilinear, θᵁ 쪽 항.
- 결론: 기본 인스턴스에서 병목은 최적해 탐색이 아니라 고정 x 의 상한 (인증). KKT (E) 는 같은 값을 수 초에. 알고리즘 설계 논의 필요.

### α-B&B 상한이 느슨한 원인: θᵁ (2026-10-09 새벽)

**결론: 비제로섬 Ω 의 리더 블록에는 θ(cᵀŷ − rᵀϖ) 라는 큰 두 항의 상쇄가 있다 (zero-sum 은 θ = 0 이라 없음).
SGB128 은 거리 (c) 가 마일 단위 (~1,000) 이고 θᵁ = v / 거리 해상도 = 5 라 θc ≈ 5,000 / 단위 흐름 → 완화 오차가 증폭.**

1. 루트 McCormick 분해 (`diag_root_relax.jl`, `logs/diag_root_relax_x12_q150_d0p1.log`, V* = −1470.5): 세 곱 모두 McCormick 상한 461,393,
   ζW (= h^r ϖ) 만 정확 267,321, ζL·ζF 만 정확 322,366 → 특정 곱 하나가 아니라 θ 스케일 전체 문제.
2. θ 비교 (`logs/diag_abb_base_x12_q150_d0p1_theta.log`, x = {1,2}, 600 s, worker 4 / Gurobi 4 스레드):

   | θᵁ | α-B&B UB (gap) | Gurobi Ω UB (gap) |
   |---|---|---|
   | 5 (기존 circuit 상계) | +3549.4 (341%) | −67.2 (95%) |
   | 0.625 (경험적 필요값 0.3125 의 2 배, 증명 없음) | −950.8 (35%) | −1443.0 (2.0%) |

3. **증명 가능한 정확 θ̄**: `nz_theta_circuit_exact(nd)` (nz_data.jl). Lemma phi (ii) 는 θᵁ ≥ max{ℓᵀg / (−cᵀg) : g 는 [A; −I] 의 circuit, cᵀg < 0} 이면 충분
   (비관적 최적 꼭짓점의 접뿔이 모서리 방향으로 생성되고 최적면 안 방향은 ℓᵀg ≤ 0). pair 행렬의 circuit 은 점포–고객 이분 다중그래프 (R, S 호) 의
   단순 교대 경로·짝수 사이클이고, 사이클과 고객–고객 경로는 ℓᵀg = 0 → 모든 단순 경로 열거로 정확히 계산.
   SGB128: **θ̄ = 1.0** (경로 79,140 개, 0.1 s; 최대 비율 circuit +R(5,1) −R(1,1) +R(1,3) −R(3,3) +S(3,2), 비용 증가 5 마일, B 판매 1 단위).
   quota·S·x 와 무관. 경험값 0.3125 ≤ 1 로 일관. 기존 θᵁ = 5 는 비용 변화 하한을 해상도 1 마일로 둔 것.
4. θᵁ = 1 (증명 가능) 로 x = {1,2} 600 s (`logs/diag_abb_base_x12_q150_d0p1_theta1.log`, worker 12 / Gurobi 전체 스레드):
   α-B&B UB −1057.2 (gap 28%, θ=5 에서 204%), **Gurobi Ω UB −1467.25 (gap 0.22%)**.
   → θ 가 주원인. α-B&B 에는 다른 약점이 남음 (후보: h^r 로만 분기해 h^r × ϖ, ϖ ∈ [0, 200 r] 곱이 안 조여짐. Gurobi 는 ϖ 로도 분기).
5. Remark 초안 (원고 supplement 용): `docs/dependent_ambiguity/remark_theta_circuit.tex`.

### α-B&B 가 zero-sum 에선 압도적인데 비제로섬에선 안 되던 원인과 해결 (2026-10-09)

**원인.** 리더 블록 F^L = ℓᵀŷ + θ(cᵀŷ − 쌍대목적). 정확한 점에서는 약쌍대성으로 θ 항 ≤ 0 (최적점에서 = 0) 이지만, 완화에서는
primal 용량 (ζL = α r) 과 쌍대목적 (ζW = α ϖ) 이 따로 완화돼 **벌점이 보상으로 바뀐다**. 목적은 −158k (ŷ 항) 와 +190k (ϖ 항) 의
상쇄로 −1470 이 되는 구조라 오차가 θ × 비용 (마일) 으로 증폭. zero-sum 은 θ = 0 (ℓ = c) 이라 이 항 자체가 없고, 곱의 상대가 확률 (a, r, d) 이다.
`diag_box_width.jl`: 노드 상한 − V* 가 α 상자 폭에 비례 (기울기 54.6 / 단위 폭) → 허용오차까지 활성 좌표마다 폭 ~0.003 필요.

**해결 1 — 정확한 circuit θᵁ** (위). **해결 2 — 약쌍대 부등식 (WD)**: 시나리오마다 cᵀŷˢ ≤ (uˢ + gˢx̄)ᵀϖˢ + Σ_H hcoef ζWˢ
(`build_nz_omega(weak_duality=true)`, :global·:relax 모드, x 행 계수는 `nz_set_objective!` 가 x̄ 로 갱신). 최적점은 만족 → 값 불변,
줄인 집합의 점은 Ω 의 점 → cut 유효. 16 개 x 에서 Ω = KKT, 위반 0 (`logs/check_theta_validity_q150_d0p1_wd.log`).

| x = {1,2}, θᵁ = 1 | WD 없음 | WD 있음 |
|---|---|---|
| 노드 완화 상한 − V*: 루트 / 폭 1 / 폭 0.3 | 19,700 / 54.6 / 16.4 | 629 / 5.4 / 1.6 |
| Gurobi Ω 60 s gap | 10% | 0.04% |
| α-B&B 600 s gap (worker 12) | 28% (θᵁ=5 면 204%) | 0.9% |

**효과 없음 / 역효과** (옵션으로 남김): ϖ 분기 (`varpi_branch`, 기본 끔: gap 204% / 79%), 닫힌 점포 프리솔브 (`presolve`, 28.3% vs 28.1%,
해는 없으니 기본 켬 유지), 비용이 A 현장구매보다 비싼 예약의 지배 (x 무관 6/15, 미구현).

**남은 약점.** WD 후에도 기울기 5.4 / 단위 폭 (허용오차까지 폭 ~0.03). 같은 ŷ·ζW 쌍에서 오차 (w=0.3: ŷ 항 −15.8, ζW 항 +17.4).
다음 후보: full RLT (α_i × 모든 ϖ_k, α_i × ŷ_j 곱 변수 + 쌍대·primal 행 × α 의 RLT).

### S = 3 벤치마크, θᵁ = 1 + WD (`logs/loc_base_scaling_theta_wd.log`, 방법당 3,600 s)

| 방법 | θᵁ = 5 | θᵁ = 1 | θᵁ = 1 + WD |
|---|---|---|---|
| E (KKT) | 14 s | 14.5 s | 14.7 s |
| SB-G | TimeLimit, gap 8.2% | Optimal 1,816 s | **Optimal 153 s** |
| SB-A | TimeLimit, gap 175% | (중단) | TimeLimit, gap 0.89% |
| Algorithm 2 | Stalled, gap 165% | (중단) | 진행 중 |

### 밤샘 실행 (`logs/loc_base_scaling.log`, S = 3, 20, 50, 방법당 3,600 s) — S = 20 의 E 후 중단 (θ 문제 먼저)

| S | 방법 | 결과 |
|---|---|---|
| 3 | E | Optimal 14 s |
| 3 | SB-G | TimeLimit, gap 8.2% (LB 는 local cut 5 개로 수 초에 최적) |
| 3 | SB-A | TimeLimit, gap 175% |
| 3 | Algorithm 2 | Stalled 3,039 s, gap 165% (LB 최적 도달 37 s) |
| 20 | E | 미해결 (x = {1} 300 s 시간 제한) |

(SB 의 RESULT 줄 t_LB 는 hist 시각 버그로 과대. 수정 커밋 470dd84)

E 만 "작은 S 에서 미해결이면 큰 S 생략". Benders 계열은 모든 S 를 시간 제한까지 돌려 최종 gap 과 t_LB (LB 가 최종값에 처음 도달한 시각) 기록.

### 남은 작업 (기본 인스턴스)

1. **S 스케일링 비교**: `run_loc_base_scaling.jl`, S = 3 → 20 → 50 → 200, E / SB-G / SB-A / Algorithm 2, δ = 0.1. (파일럿 끝난 뒤 실행)
2. Table loc-values (quota 150 에서 모든 x) 와 Benders 서술을 새 인스턴스로 다시.
3. λᵁ 수치 확인을 quota 150 에서 다시 (θᵁ = v 논증은 quota 무관).
4. Table loc-oos: location OOS 평가기 새로 작성 (진실 Dirichlet, belief = 진실 + 노이즈, 거리 Close / Moderate / Misspecified).

## 3. 무작위 인스턴스 (Goyal §7.3)

Goyal Table 1, 2 의 설정 (각 10 개, w 는 C=V=0 / C,V≠0 일 때 값):

| 설정 | d | 적격지 | A 점유 | N_B | w | 우리 \|𝒳\| |
|---|---|---|---|---|---|---|
| 1–4 | 15 | 5 | 2 / 3 | 3 / 2 | 750 / 1000 | 8 / 4 |
| 5–8 | 20 | 5 | 2 / 3 | 3 / 2 | 750 / 1000 | 8 / 4 |
| 9–10 | 25 | 10 | 5 | 5 | 750 / 1000 | 32 |

생성: 좌표 U[0,1]², 수요 U[50,150] (비적격 위치), 적격지 용량 b_i = 150 (d − ‖l_S‖) / ‖l_A‖.
원고 방침: 최소 설정 (d = 15, 적격 5, A 2, N_B 3) 에서 출발, 주로 S 를 늘리고 |𝒳|, n_h 는 파일럿으로 조정.
예약: pooled (n_h = 적격지 수) 또는 pair (n_h = 적격지 × 고객, d = 15 에서 50).

### 파일럿 (`pilot_loc_random.jl`, `logs/pilot_loc_random_d15.log`, d = 15, quota/wres 300, ε = 0.3, δ = 0.1) — 진행 중

| 예약 | seed | S | E | Algorithm 2 |
|---|---|---|---|---|
| pooled | 1 | 3 / 5 / 10 | 미해결 (S=3, x={1} 300 s) | Optimal 16 s / 19 s / 5 s (x* = {1}, {1}, ∅) |
| pooled | 2 | 3 / 5 / 10 | 미해결 | Optimal 3 s / 1 s / 17 s (x* = {3}) |
| pair | 1 | 3 | 미해결 (x=∅ 에서도 300 s) | 진행 중 |

관찰
- E 는 S 가 아니라 고객 수 (3 → 10) 에서 막힘: 상보성 이진변수가 시나리오당 ~106 → ~240 (pooled) / ~330 (pair).
- seed 1 은 값이 작고 S = 10 에서 출점 안 함 → 출점비 750 이 예약 확장과 함께 B 에 불리할 수 있음. 경제 파라미터 점검 필요.

### KKT 개선 시도 (`nz_kkt_value_reduced`, `test_kkt_reduced.jl`, `logs/test_kkt_reduced.log`)

follower 블록이 recourse 최적성을 보장하므로 pessimistic 복사본은 "Aŷ ≤ r_s(x̄, α), cᵀŷ ≥ cᵀyˢ" 만 요구 → 시나리오당 (m + ny) 개 상보성 이진변수 제거.
recourse dual 을 d 로 스케일하지 않고 1단계 조건에 d_s·πˢ (bilinear) 를 써서 d_s = 0 도 정확.
- 기본 인스턴스: 모든 x 에서 원래와 같은 값 (상대차 ≤ 1.5e-6), 시간은 2~3 배 느림.
- 무작위 d = 15: incumbent 는 찾지만 (원래는 incumbent 없음) x = {1} 에서 여전히 300 s 시간 제한 → E 는 무작위에서 미해결로 보고.

### 남은 작업 (무작위)

1. 파일럿 완료 후 n_h (pooled / pair) 와 경제 파라미터 확정, S 범위 결정.
2. Goyal 설정 1–10 을 따라 d, |𝒳| 확장 + S 확장, 설정당 10 개.

## 4. zero-sum (NIG) — 코드만 준비, 원고 위치 미정

`true_dro/factor_5/run_coupling_batch.jl` (원고 baseline 과 같은 Enhanced BD + VIs, δ = ρ(ε̂+ε̃), ρ ∈ {1, 0.5, 0.25, 0}, 5 네트워크 × β × ε),
요약 `summarize_coupling_batch.jl`. 스모크 (grid5x5, β = 0.4, ε = 0.1): ρ = 0 / 1 모두 Optimal, x* 같음, Z₀ 차 0.24% (허용오차 0.5% 이내 → 구별 불가).
값 비교용으로는 `CPL_TOL` 을 줄여 따로 돌려야 함. 원고 6.2 에 δ 실험을 넣을지는 미정.

## 5. 실행 명령

```bash
# 기본 인스턴스 표·그림
julia nonzero_sum/loc_insample.jl > nonzero_sum/logs/loc_insample_sgb128_pair_q150.log
# 기본 인스턴스 S 스케일링
julia -t 14,1 nonzero_sum/run_loc_base_scaling.jl > nonzero_sum/logs/loc_base_scaling.log
# 무작위 파일럿
julia -t 14,1 nonzero_sum/pilot_loc_random.jl > nonzero_sum/logs/pilot_loc_random_d15.log
# zero-sum δ 배치
cd true_dro/factor_5 && julia -t 14 run_coupling_batch.jl && julia summarize_coupling_batch.jl
```
