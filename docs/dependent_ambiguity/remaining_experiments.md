# 남은 실험 계획 (2026-10-20 11:12 일시 정지 시점)

비제로섬 입지 (retailer + 거점별 예약) 본 실험. 계획 전체와 결정 경위는 `alpha_bnb_nonzero_analysis.md` §10.24~10.25, 메모리 `nz_pooled_retailer`.
공통: Algorithm 2 (ALGL), 1 시간, 허용오차 0.5%, λᵁ = 2,000, 시드 1~10, ε̂ = ε̃ = 0.3, δ = 0.1, β = 0.4. 규칙: 어떤 S 에서 Optimal < 5/10 이면 그 설정의 더 큰 S 는 안 돌림.

## 1. 끝난 것

| 설정 | d | \|A\| / B | S=10 | S=20 | S=50 |
|---|---|---|---|---|---|
괄호는 SGM (shift 10 s, 못 푼 것은 3,600 s).

| 설정 | d | \|A\| / B | S=10 | S=20 | S=50 |
|---|---|---|---|---|---|
| L1 | 15 | 2 / 3 | 10/10 (45 s) | 10/10 (128 s) | 9/10 (568 s) |
| L2 | 15 | 3 / 2 | 10/10 (103 s) | 10/10 (197 s) | 7/10 (735 s) |
| L3 | 20 | 2 / 3 | 10/10 (61 s) | 10/10 (114 s) | 10/10 (364 s) |
| L4 | 20 | 3 / 2 | 10/10 (163 s) | 10/10 (324 s) | 7/10 (920 s) |
| L5 | 25 | 2 / 3 | 10/10 (89 s) | 10/10 (228 s) | 10/10 (825 s) |
| L6 | 25 | 5 / 5 | 스크리닝 S=3 1/10 → 범위 밖 (본 실험 없음) | | |
| 보충 \|A\|=3 | 16 | 3 / 3 | 10/10 (217 s) | 10/10 (519 s) | **seed 1~6 까지** (4 Optimal, seed 1·5 TimeLimit). seed 7 은 풀던 중 사용자 요청으로 멈춤 |
| 보충 \|A\|=4 | 17 | 4 / 3 | 7/10 | — | — |

**본문 목록 L1~L5 는 S=10·20·50 모두 끝남 (150 개 중 142 개 Optimal).**

## 2. 남은 것 (이 순서로)

| 순서 | 작업 | 로그 | 예상 |
|---|---|---|---|
| 1 | **보충 \|A\|=3 S=50 seed 7~10** (같은 로그에 이어 붙임) | `main_pooled_store_A3_d16_S50.log` | 3~4 h |
| 2 | 보충 \|A\|=4 S=20 (S=10 이 7/10 이라 실행) | `main_pooled_store_A4_d17_S20.log` | 5~10 h |
| 3 | 보충 \|A\|=4 S=50 (2 가 Optimal ≥ 5 일 때만) | `main_pooled_store_A4_d17_S50.log` | 10 h 이상 |
| 4 | λ 무관 인증 (모든 인스턴스) | `cert_lambda_free.log` | 몇 시간 (아직 한 번도 끝까지 안 돌아 봄) |

### 실행 방법

1 (seed 7~10 을 같은 로그에 이어 붙이기 — RESULT 가 10 줄이 되어야 큐가 이 로그를 건너뛴다. 보충 축은 출점비 = 거점 용량이라 PP_QB=cap):

```bash
cd /c/Users/user/robust_bilevel_network_games
PP_NLOC=16 PP_NA=3 PP_NB=3 PP_NBMAX=3 PP_QB=cap PP_LAMBDA=2000 PP_S=50 PP_SEEDS="7,8,9,10" PP_TL=3600 PP_WORKERS=12 \
  julia -t 14,1 nonzero_sum/pilot_pooled_store.jl >> nonzero_sum/logs/main_pooled_store_A3_d16_S50.log 2>&1
```

2~3 (큐 스크립트를 다시 실행하면 끝난 로그는 건너뛰고 이어서 함. 요약은 `logs/run_pooled_store_queue.out`):

```bash
bash nonzero_sum/run_pooled_store_queue.sh >> nonzero_sum/logs/run_pooled_store_queue.out 2>&1
```

4 (큐가 끝난 뒤. 이미 CERT 줄이 있는 인스턴스는 건너뜀):

```bash
julia -t 14,1 nonzero_sum/cert_lambda_free.jl >> nonzero_sum/logs/cert_lambda_free.stdout 2>&1
```

### 주의

- **1 을 하지 않고 큐를 다시 실행하면** 보충 |A|=3 S=50 로그의 RESULT 가 10 줄이 아니라서 seed 1 부터 다시 돌리고 로그를 덮어쓴다.
- julia 두 개가 동시에 돌지 않게 할 것 (worker 12 개씩이라 시간이 부풀려짐). 시작 전에 PowerShell `Get-Process julia` 로 확인 (Git Bash `ps` 에는 부모가 끝난 julia 가 안 보임).
- 다른 세션 (zonal_consistency, energy_community 등) 의 실험과도 겹치지 않게 (python·Gurobi 프로세스도 확인).
- L2~L4 의 3,600 s λ 인증 로그 (`lab_lambda_certify_list_L*_tl3600.log`) 는 "건너뜀" 표시가 들어 있어 큐가 다시 돌리지 않는다 (λ 무관 인증으로 대체, 사용자 결정).

## 3. 큐 밖에 남은 일

| 일 | 상태 |
|---|---|
| 기본 인스턴스 (SGB128) 를 거점별 예약으로 다시 (원고 표 loc-values, loc-insample, loc-oos) + λᵁ = 100 인증 | 미실행 |
| 비교 대상 (SB-G, SB-A, E) | 범위 미정. 허용오차는 모두 0.5% 로 맞춤 (`run_loc_base_scaling.jl`). 무작위 인스턴스용 스크립트는 아직 없음 |
| value of robustness (ε·δ 를 바꾼 실험), out-of-sample | 계산 성능 실험 뒤 |
| 원고: §6.1.2 의 λ 서술을 λ 무관 인증 결과로, 계산 성능 표 (3 단계), 보충 §G·§H 결과 표, Algorithms 단락의 ε = 10⁻⁴ 문구 | 결과가 나온 뒤 |
| `lab_lambda_certify.jl` 이 `nz_exact_eval.jl` 을 include 하도록 정리 (지금은 같은 함수 복사본) | 큐가 그 파일을 안 쓸 때 |

## 4. 걸어 둔 순서 (2026-10-20 22:10)

zonal_consistency 세션의 큐 (`exp2_chain.sh` → `exp2_rep.sh`, 대표일 시각별 실험) 가 끝난 뒤 자동으로 1 → 2 를 돈다 (scratchpad `after_zonal_chain.sh`, 세션이 끝나면 사라짐).
- 시작 조건: zonal 체인 셸이 없고 python·julia 도 없는 상태가 2 분 간격으로 두 번 연속 (정상 종료든 중간 정지든).
- 1 (|A|=3 S=50 seed 7~10) 은 로그의 RESULT 가 6 줄일 때만 이어 붙인다. 2 (|A|=4 S=20) 는 로그가 이미 있으면 덮어쓰지 않고 멈춘다.
- 큐 스크립트는 쓰지 않는다 (그러면 3 까지 이어서 돌기 때문). 3 (|A|=4 S=50) 과 4 (λ 무관 인증) 는 사용자가 나중에.
- 진행·종료 표시는 `logs/run_pooled_store_queue.out` 의 `=== ` 줄.
- 그 뒤에 이어서 (2026-10-20 사용자 결정): **§6.2 네트워크 차단 계산 성능을 결합 집합으로 다시** — δ = 0.1, β = 0.4, ε̂ = ε̃ = ε*(네트워크) (grid·polska·sioux_falls 0.1, abilene·nobel_us 0.2), S = 10·20·50.
  S = 10 은 네 변형 (Plain BD, Enhanced BD, Enhanced BD + VIs, Algorithm 2 의 zero-sum 판 = belief menu), S = 20·50 은 belief menu 만 (사용자 결정 2026-10-20). 시간 한도 3,600 s (입지 실험과 통일, 사용자 결정), 허용오차 0.5%. 전역 subproblem 은 zero-sum α-B&B (global_bilinear_solver.jl).
  스크립트 `true_dro/factor_5/run_nig_coupled_scaling.jl` (기존 스크립트는 그대로), 로그 `true_dro/factor_5/logs/nig_coupled/S_<S>/<net>_<variant>.log`.
  out-of-sample 은 이번에 안 함. 시작 조건: 보충 사슬·zonal 체인이 없고 python·julia 도 없을 때 (scratchpad `nig_after.sh`).
  참고: 기존 tab:computational 의 숫자는 지금 남은 로그 (true_dro/factor_5/logs/factor/...) 로 다시 계산하면 맞지 않는다 (예: Grid Plain BD 1,162 s vs 원고 924 s) — 더 이전 실행에서 만든 표로 보임. 이번 결과로 표를 새로 만든다.
