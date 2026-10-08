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
| 시간 제한 | 방법당 3,600 s, E 는 x 하나당 300 s. 작은 S 에서 실패한 방법은 큰 S 생략 | 안 풀리는 문제에 시간 쓰지 않음 |
| 자원 | 이 PC (Ryzen 9 9950X) 는 α-B&B worker 12, 휴리스틱 켬 (`julia -t 14,1`) | worker 1 제한은 다른 PC 의 WLS 사정 |
| 남은 결정 | ε = 0.3 의 근거 (validation 또는 표본 수 규칙) | 현재는 효과가 보이는 값으로 고른 것 |

## 1. 방법 (원고 6.1 Algorithms)

| 이름 | 내용 | 코드 |
|---|---|---|
| E | 모든 x 에서 V*(x) 를 KKT 단일수준 MINLP 로 (Benders·minimax·big-M 없음) | `nz_kkt_value` (원래), `nz_kkt_value_reduced` (축소판, §3) |
| SB-G | 표준 Benders: Ω local 해 (Gurobi OptimalityTarget=1) cut 먼저, 위반 cut 없을 때만 Gurobi 전역 Ω | `nz_standard_benders(...; oracle=:gurobi, local_first=true)` |
| SB-A | 같음, 전역은 α-B&B | `nz_standard_benders(...; oracle=:alpha_bnb, local_first=true)` |
| Algorithm 2 | belief-menu + α-B&B | `nz_belief_menu_benders` |

공통: ε = 1e-4, oracle 600 s (재방문 2,400 s), worker 12.

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
