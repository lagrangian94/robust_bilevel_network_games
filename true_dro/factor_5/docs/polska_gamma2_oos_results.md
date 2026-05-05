# Polska γ=2 OOS 3-Way Comparison

## Setup
- Network: Polska (12 nodes, 36 arcs + dummy)
- γ = 2, uniform capacity scenarios, S = 20, seed = 42
- Source-sink connectivity cut 적용
- OOS: Phase A symmetric Dirichlet, M = 500, seed = 42

## Interdiction Solutions

| Model | ε̂ | ε̃ | x arcs | Description |
|-------|----|----|--------|-------------|
| Nominal | 0 | 0 | [3, 6] | Stochastic program |
| Single-layer | 1.0 | 0 | [6, 14] | Leader-only DRO |
| Double-layer | 1.0 | 1.0 | [18, 33] | Full bilevel DRO |

## OOS Results (mean cost, lower = better)

| β | Nominal [3,6] | Single [6,14] | Double [18,33] |
|---|---------------|---------------|----------------|
| 0.5 | 9.137 | 9.414 | **9.011** |
| 1.0 | 9.250 | 9.539 | **9.087** |
| 5.0 | 9.353 | 9.640 | **9.108** |

## Pairwise Win Rates

### Double vs Nominal
| β | Gap (mean) | Double wins |
|---|-----------|-------------|
| 0.5 | -0.126 | 259/500 (51.8%) |
| 1.0 | -0.164 | 290/500 (58.0%) |
| 5.0 | -0.245 | 395/500 (79.0%) |

### Double vs Single
| β | Gap (mean) | Double wins |
|---|-----------|-------------|
| 0.5 | -0.403 | 338/500 (67.6%) |
| 1.0 | -0.452 | 400/500 (80.0%) |
| 5.0 | -0.532 | 495/500 (99.0%) |

### Single vs Nominal
| β | Gap (mean) | Single wins |
|---|-----------|-------------|
| 0.5 | +0.277 | 151/500 (30.2%) |
| 1.0 | +0.289 | 116/500 (23.2%) |
| 5.0 | +0.287 | 23/500 (4.6%) |

## Detailed Statistics

### β = 0.5
```
Nominal [3,6]       : median=9.104358, mean=9.137492
  [p05, p95] = [7.839817, 10.477179]  [q25, q75] = [8.645764, 9.588714]
Single [6,14]       : median=9.403504, mean=9.414238
  [p05, p95] = [8.252589, 10.479011]  [q25, q75] = [8.989867, 9.864992]
Double [18,33]      : median=9.012098, mean=9.011047
  [p05, p95] = [8.052277, 10.062873]  [q25, q75] = [8.635392, 9.402911]
```

### β = 1.0
```
Nominal [3,6]       : median=9.255359, mean=9.250293
  [p05, p95] = [8.367217, 10.079626]  [q25, q75] = [8.916758, 9.573419]
Single [6,14]       : median=9.542065, mean=9.539029
  [p05, p95] = [8.770654, 10.232309]  [q25, q75] = [9.284747, 9.821824]
Double [18,33]      : median=9.083815, mean=9.086699
  [p05, p95] = [8.402652, 9.791161]  [q25, q75] = [8.838982, 9.339765]
```

### β = 5.0
```
Nominal [3,6]       : median=9.330429, mean=9.353392
  [p05, p95] = [8.986684, 9.752419]  [q25, q75] = [9.202960, 9.510424]
Single [6,14]       : median=9.629813, mean=9.640460
  [p05, p95] = [9.295056, 10.001607]  [q25, q75] = [9.507092, 9.782848]
Double [18,33]      : median=9.099086, mean=9.108047
  [p05, p95] = [8.844982, 9.380123]  [q25, q75] = [8.995465, 9.222278]
```

## Key Findings

1. **Double-layer [18,33]이 모든 β에서 최고 성능** — mean cost 가장 낮음
2. **Single-layer [6,14]이 가장 나쁨** — Nominal보다도 열등 (wins 4.6~30.2%)
3. β 커질수록 (분포 불확실성 감소) Double의 우위 더 명확: β=5에서 Nom 대비 79%, Single 대비 99% 승리
4. Single이 Nominal보다 나쁜 원인: ε̃=0으로 follower ambiguity 무시 → interdiction 선택 왜곡

## Plots
- `plots/polska_gamma2_3way_beta0.5.png`
- `plots/polska_gamma2_3way_beta1.0.png`
- `plots/polska_gamma2_3way_beta5.0.png`
