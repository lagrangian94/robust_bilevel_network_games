# Experiment 2 분석: grid_5x5 DRO over-conservatism 진단

## 1. Experiment 2 Calibration 결과

ε bisection으로 [ε^S, ε^D] 범위 탐색 (Rahimian et al. 2019).

| Iter | ε | PO-PP | NR-WR | Region |
|------|-------|---------|---------|--------|
| 1 | 0.5000 | -0.211 | --- | below ε^S |
| 2 | 0.7500 | -0.135 | --- | below ε^S |
| 3 | 0.8750 | -0.008 | --- | below ε^S |
| 4 | 0.9375 | +0.055 | +0.464 | above ε^D |
| 5 | 0.9062 | +0.024 | +0.464 | above ε^D |
| 6 | 0.8906 | +0.007 | +0.464 | above ε^D |
| 7 | 0.8828 | -0.001 | --- | below ε^S |

- **ε_recommended = 0.8867**, bounds [0.8828, 0.8906]
- [ε^S, ε^D] ≈ [0.883, 0.889] — 극도로 좁은 구간
- Coverage ε (β=0.1) = 0.8439 < ε^S → analytical calibration이 coverage보다 **높은** ε 제시 (ratio=1.05)

## 2. 해 비교

| 설정 | ε (approx) | x* (arcs) | nominal과 동일? |
|------|-----------|-----------|----------------|
| Nominal (ε=0) | 0 | 24, 30, 37 | baseline |
| DRO β=0.8 | 0.44 | 24, 30, 37 | 동일 |
| DRO β=0.5 | 0.58 | 24, 30, 37 | 동일 |
| DRO β=0.3 | 0.67 | 24, 30, 37 | 동일 |
| DRO ε=0.8867 | 0.89 | 24, **25**, 37 | **다름** |
| Robust (ε=1.0) | 1.00 | 24, **25**, 37 | 다름 (= ε=0.8867) |

- x*가 달라지는 임계점: ε ≈ 0.883
- 차이: arc 30 (node_3_2→node_3_3) vs arc 25 (node_3_1→node_3_2)

## 3. OOS Phase A 결과 (x_nom vs x_dro at ε=0.8867)

β=0.1, M=100, L=100, seed=42. Block-mean 기준 CI.

| β | Model | Mean [95% CI] | p5 [CI] | p95 [CI] | f_share | Win% |
|---|-------|--------------|---------|----------|---------|------|
| 0.1 | nominal | 11.296 [11.224, 11.368] | 7.491 [7.353, 7.630] | 14.964 [14.818, 15.109] | 0.024 | --- |
| 0.1 | dro | 11.797 [11.735, 11.860] | 7.843 [7.715, 7.971] | 15.624 [15.494, 15.753] | 0.017 | 0.0% |
| 0.3 | nominal | 11.486 [11.447, 11.524] | 8.974 [8.892, 9.056] | 13.859 [13.784, 13.934] | 0.016 | --- |
| 0.3 | dro | 11.942 [11.899, 11.985] | 9.342 [9.269, 9.414] | 14.398 [14.319, 14.478] | 0.019 | 0.0% |
| 0.5 | nominal | 11.534 [11.496, 11.572] | 9.578 [9.507, 9.650] | 13.371 [13.318, 13.424] | 0.025 | --- |
| 0.5 | dro | 12.002 [11.966, 12.038] | 9.950 [9.882, 10.018] | 13.948 [13.892, 14.003] | 0.021 | 0.0% |

- 모든 β에서 CI 겹치지 않음 — nominal이 **확실히** 우세
- DRO Win% = 0.0% (block mean 기준 100개 중 0개 승리)
- Gap ≈ +0.47 (DRO가 nominal보다 약 0.5 높음, minimization)

## 4. OOS Phase B 결과 (Asymmetric Dirichlet)

noise_scale=0.5, M=100, R=100.

| β | Gap (DRO-Nom) [CI] | DRO wins |
|---|-------------------|----------|
| 0.1 | +0.477 [0.464, 0.491] | 0.0% |
| 0.3 | +0.478 [0.464, 0.492] | 0.0% |
| 0.5 | +0.464 [0.452, 0.477] | 0.0% |

- Phase A와 거의 동일한 gap — asymmetric 분포에서도 결론 불변

## 5. Paired Comparison: 왜 DRO가 항상 지는가

### 5.1 시나리오별 max-flow 비교 (Block 1, β=0.1)

같은 q_follower, 같은 q_true에서 DRO - Nominal diff:
- **mean = +0.318**, std = 0.130
- **min = +0.028** — L=100개 전부 양수 (DRO가 한 번도 안 이김)
- 분포 전체가 zero 오른쪽

### 5.2 시나리오 단위 승패

| | Block 1 | 전체 M=100 |
|--|---------|-----------|
| Nom wins | 18/20 | 1408/2000 (70%) |
| DRO wins | 2/20 | 496/2000 (25%) |
| Block mean에서 DRO 승 | - | **0/100** |

### 5.3 Arc capacity: arc 30 vs arc 25

| | arc 25 (node_3_1→3_2) | arc 30 (node_3_2→3_3) |
|--|---|---|
| mean cap | ~3.1 | ~4.1 |
| max cap | 6.34 | **10.57** |
| min cap | 0.44 | 0.62 |

arc 30이 **거의 모든 시나리오에서** arc 25보다 capacity가 큼. arc 25는 arc 30에 대해 **dominated alternative**.

DRO가 arc 25를 선택하는 이유: ε=0.89의 adversary가 TV distance 내에서 arc 25가 유리한 극소수 시나리오(S15, S19)에 확률을 집중시키는 비현실적 분포를 구성.

## 6. Recovery (h*) 분석: f_share가 낮은 이유

w = 11.5491 (interdictable arc 최대 capacity). Follower가 arc 하나를 **완전히 복구** 가능한 수준.

### 6.1 h* 배분의 q_follower 감도 (x_nom, β=0.1, M=100)

| arc | mean h | std h | CV | 역할 |
|-----|--------|-------|-----|------|
| 24 (interdicted) | 2.19 | 0.84 | 0.39 | |
| 30 (interdicted) | **2.74** | **0.35** | **0.13** | 가장 안정 |
| 37 (interdicted) | 2.84 | 0.87 | 0.31 | |
| 25 | 0.96 | 0.55 | 0.57 | |
| 29 | 0.71 | 0.53 | 0.74 | |
| 31 | 1.40 | 0.48 | 0.34 | |

- Budget 항상 풀로 사용 (11.5491 ± 0.0000)
- **arc 30의 CV = 0.13**: q_follower가 뭐든 항상 ~2.7 recovery 투입
- 네트워크 min-cut 구조가 recovery 배분을 강하게 결정 → follower belief에 둔감
- **f_share = 0.02 (2%)**: follower belief가 결과 분산의 2%만 설명

### 6.2 해석

1. **Interdiction >> Recovery 비대칭**: v=1.0으로 arc capacity 완전 제거. w=11.55로 하나만 완전 복구 가능 (3개 중). 2개는 무방비.
2. **Min-cut 구조적 결정**: 이 네트워크에서 최적 recovery 배분이 belief에 거의 무관 → follower 불확실성의 실질적 영향 미미
3. **DRO의 보호 대상 부재**: DRO가 보호하려는 "follower의 잘못된 recovery로 인한 손실"이 거의 발생 안 함

## 7. Open Questions

### (1) γ (interdiction budget) 조정
- 현재 γ = 3 (ceil(0.10 × 25))
- γ를 키우거나 줄이면 [ε^S, ε^D] 구간이 변할 수 있음

### (2) Capacity scenario 생성 방식
- 현재: factor model (seed=42, S=20) — 시나리오 간 상관이 높음
- seed 변경 또는 uniform 분포로 실험 필요?
- 시나리오 diversity가 [ε^S, ε^D] 구간 폭에 영향

### (3) 현재 ε calibration은 문제 구조를 전혀 활용하지 않음
- Coverage-based든 analytical (PO/PP/NR/WR)든, ambiguity set 크기만 조절
- 네트워크 구조, 시나리오 분포 특성 미반영

### (4) f_share를 높이는 설정 탐색
- DRO의 이점이 드러나려면 follower belief가 결과에 더 큰 영향을 미쳐야 함
- v < 1.0 (부분 interdiction), w 증가, 다른 네트워크 구조 등

## 8. 파일 참조

| 파일 | 내용 |
|------|------|
| `exp2_grid5x5_console_log.txt` | bisection 전체 로그 + x*(ε=0.8867) 결과 |
| `oos_quick_compare_phase_a.jls` | Phase A OOS raw data (q_follower, q_true, h, flows 포함) |
| `oos_block1_beta01_histogram.png` | Block 1 분포 비교 히스토그램 |
| `oos_block1_beta01_paired_diff.png` | Block 1 paired diff 히스토그램 |
| `run_oos_quick_compare.jl` | Phase A 비교 스크립트 |
| `run_oos_quick_compare_b.jl` | Phase B 비교 스크립트 |
