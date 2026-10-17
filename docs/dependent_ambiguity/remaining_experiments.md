# 남은 실험 계획 (2026-10-17 11:20 일시 정지 시점)

비제로섬 입지 (retailer + 거점별 예약) 본 실험. 계획 전체와 결정 경위는 `alpha_bnb_nonzero_analysis.md` §10.24~10.25, 메모리 `nz_pooled_retailer`.
공통: Algorithm 2 (ALGL), 1 시간, 허용오차 0.5%, λᵁ = 2,000, 시드 1~10, ε̂ = ε̃ = 0.3, δ = 0.1, β = 0.4. 규칙: 어떤 S 에서 Optimal < 5/10 이면 그 설정의 더 큰 S 는 안 돌림.

## 1. 끝난 것

| 설정 | d | \|A\| / B | S=10 | S=20 | S=50 |
|---|---|---|---|---|---|
| L1 | 15 | 2 / 3 | 10/10 | 10/10 | 9/10 |
| L2 | 15 | 3 / 2 | 10/10 | 10/10 | **seed 1~6 까지** (5 Optimal, seed 5 TimeLimit). seed 7 은 풀던 중 사용자가 다른 세션에서 종료 |
| L3 | 20 | 2 / 3 | 10/10 | 10/10 | — |
| L4 | 20 | 3 / 2 | 10/10 | 10/10 | — |
| L5 | 25 | 2 / 3 | 10/10 | 10/10 | 10/10 |
| L6 | 25 | 5 / 5 | 스크리닝 S=3 1/10 → 범위 밖 (본 실험 없음) | | |
| 보충 \|A\|=3 | 16 | 3 / 3 | 10/10 | 10/10 | — |
| 보충 \|A\|=4 | 17 | 4 / 3 | 7/10 | — | — |

## 2. 남은 것 (이 순서로)

| 순서 | 작업 | 로그 | 예상 |
|---|---|---|---|
| 1 | **L2 S=50 seed 7~10** (방법 (a): 같은 로그에 이어 붙임) | `main_list_L2_d15_A3B2_S50.log` | 1~3 h |
| 2 | L3 S=50 | `main_list_L3_d20_A2B3_S50.log` | 약 2 h |
| 3 | L4 S=50 | `main_list_L4_d20_A3B2_S50.log` | 약 4.5 h |
| 4 | 보충 \|A\|=3 S=50 | `main_pooled_store_A3_d16_S50.log` | 6~10 h |
| 5 | 보충 \|A\|=4 S=20 (S=10 이 7/10 이라 실행) | `main_pooled_store_A4_d17_S20.log` | 5~10 h |
| 6 | 보충 \|A\|=4 S=50 (5 가 Optimal ≥ 5 일 때만) | `main_pooled_store_A4_d17_S50.log` | 10 h 이상 |
| 7 | λ 무관 인증 (모든 인스턴스) | `cert_lambda_free.log` | 몇 시간 (아직 한 번도 끝까지 안 돌아 봄) |

### 실행 방법

1 (seed 7~10 을 같은 로그에 이어 붙이기 — RESULT 가 10 줄이 되어야 큐가 이 로그를 건너뛴다):

```bash
cd /c/Users/user/robust_bilevel_network_games
PP_NLOC=15 PP_NA=3 PP_NB=2 PP_NBMAX=2 PP_QB=750 PP_LAMBDA=2000 PP_S=50 PP_SEEDS="7,8,9,10" PP_TL=3600 PP_WORKERS=12 \
  julia -t 14,1 nonzero_sum/pilot_pooled_store.jl >> nonzero_sum/logs/main_list_L2_d15_A3B2_S50.log 2>&1
```

2~6 (큐 스크립트를 다시 실행하면 끝난 로그는 건너뛰고 이어서 함. 요약은 `logs/run_pooled_store_queue.out`):

```bash
bash nonzero_sum/run_pooled_store_queue.sh >> nonzero_sum/logs/run_pooled_store_queue.out 2>&1
```

7 (큐가 끝난 뒤. 이미 CERT 줄이 있는 인스턴스는 건너뜀):

```bash
julia -t 14,1 nonzero_sum/cert_lambda_free.jl >> nonzero_sum/logs/cert_lambda_free.stdout 2>&1
```

### 주의

- **1 을 하지 않고 큐를 다시 실행하면** L2 S=50 로그의 RESULT 가 10 줄이 아니라서 seed 1 부터 다시 돌리고 로그를 덮어쓴다.
- julia 두 개가 동시에 돌지 않게 할 것 (worker 12 개씩이라 시간이 부풀려짐). 시작 전에 PowerShell `Get-Process julia` 로 확인 (Git Bash `ps` 에는 부모가 끝난 julia 가 안 보임).
- 다른 세션 (zonal_consistency) 의 실험과도 겹치지 않게. 2026-10-17 의 zonal 실험 1~4 는 06:05 에 끝남.
- L2~L4 의 3,600 s λ 인증 로그 (`lab_lambda_certify_list_L*_tl3600.log`) 는 "건너뜀" 표시가 들어 있어 큐가 다시 돌리지 않는다 (λ 무관 인증으로 대체, 사용자 결정).

## 3. 큐 밖에 남은 일

| 일 | 상태 |
|---|---|
| 기본 인스턴스 (SGB128) 를 거점별 예약으로 다시 (원고 표 loc-values, loc-insample, loc-oos) + λᵁ = 100 인증 | 미실행 |
| 비교 대상 (SB-G, SB-A, E) | 범위 미정. 허용오차는 모두 0.5% 로 맞춤 (`run_loc_base_scaling.jl`). 무작위 인스턴스용 스크립트는 아직 없음 |
| value of robustness (ε·δ 를 바꾼 실험), out-of-sample | 계산 성능 실험 뒤 |
| 원고: §6.1.2 의 λ 서술을 λ 무관 인증 결과로, 계산 성능 표 (3 단계), 보충 §G·§H 결과 표, Algorithms 단락의 ε = 10⁻⁴ 문구 | 결과가 나온 뒤 |
| `lab_lambda_certify.jl` 이 `nz_exact_eval.jl` 을 include 하도록 정리 (지금은 같은 함수 복사본) | 큐가 그 파일을 안 쓸 때 |
