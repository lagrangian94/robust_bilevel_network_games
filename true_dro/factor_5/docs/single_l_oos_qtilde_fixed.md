# Single-L OOS: qtilde=qhat 고정, qtrue ~ Dir(β)

## Setup
- Network: Polska, factor_additive (k=5), S=20, γ=2, seed=42
- Follower belief q̃=q̂ 고정, h* 1회 계산
- OOS: qtrue ~ Dir(β·1_S), M=5000, seed=42
- Metric: E_qtrue[flow] = dot(qtrue, flows), flows 고정

### q̂ = uniform (기본)
- NOM = [3,6] (nominal solution), DRO = [6,18] (single-layer ε̂=0.1)

## Δ (DRO − NOM): negative = DRO better

| β | Δmean | Δp90 | Δp95 | Δp99 | Δmax | win% | TV mean |
|---|-------|------|------|------|------|------|---------|
| 0.1 | +0.38 | +0.42 | **-0.10** | **-1.86** | **-4.24** | 39% | 0.733 |
| 0.3 | +0.36 | +0.35 | +0.14 | **-0.33** | **-2.16** | 35% | 0.559 |
| 0.5 | +0.38 | +0.33 | +0.22 | **-0.16** | **-2.23** | 31% | 0.470 |
| 1.0 | +0.38 | +0.31 | +0.25 | +0.01 | **-0.44** | 26% | 0.359 |
| 5.0 | +0.39 | +0.35 | +0.33 | +0.30 | +0.24 | 8% | 0.171 |

## Paired gap 분포 (Δ = DRO − NOM)

| β | p01 | p05 | median | p95 | p99 |
|---|-----|-----|--------|-----|-----|
| 0.1 | **-3.97** | -2.30 | +0.56 | +2.52 | +3.25 |
| 0.3 | **-2.37** | -1.43 | +0.44 | +1.91 | +2.39 |
| 0.5 | **-1.70** | -1.04 | +0.42 | +1.65 | +2.05 |
| 1.0 | **-1.21** | -0.69 | +0.42 | +1.36 | +1.65 |
| 5.0 | -0.26 | -0.06 | +0.39 | +0.84 | +1.00 |

## Absolute costs

| β | mean_nom | mean_dro | p95_nom | p95_dro | p99_nom | p99_dro | max_nom | max_dro |
|---|----------|----------|---------|---------|---------|---------|---------|---------|
| 0.1 | 19.284 | 19.669 | 22.875 | 22.771 | 25.197 | 23.337 | 28.078 | 23.839 |
| 0.3 | 19.355 | 19.717 | 21.688 | 21.829 | 22.807 | 22.475 | 25.799 | 23.638 |
| 0.5 | 19.347 | 19.726 | 21.233 | 21.454 | 22.092 | 21.932 | 24.983 | 22.756 |
| 1.0 | 19.307 | 19.689 | 20.761 | 21.008 | 21.458 | 21.468 | 22.522 | 22.084 |
| 5.0 | 19.321 | 19.709 | 20.000 | 20.332 | 20.247 | 20.552 | 20.759 | 20.999 |

## Key Findings

1. **Mean/median에서는 NOM이 항상 우세** (Δmean ≈ +0.38, 모든 β)
2. **β≤0.5에서 p99/max는 DRO가 우세** — β=0.1: Δp99=-1.86, Δmax=-4.24
3. **Paired gap이 negative skew**: DRO가 이길 때의 이득 폭이 NOM이 이길 때보다 큼
   - β=0.1: p01=-3.97 vs p99=+3.25 (|p01| > |p99|)
4. β 커질수록 skew 감소, β=5.0에서는 NOM이 모든 quantile에서 우세

## Interpretation

- DRO는 빈도에서 지지만 (win≈39%), 극단 상황에서의 손실 폭을 줄여줌
- NOM의 max cost (28.08)가 DRO의 max cost (23.84)보다 훨씬 높음 → worst-case protection
- 이는 DRO의 전형적 특성: average에서는 약간 손해, tail에서 보호

## Phase A vs Phase B (noise perturbation)

Phase B: qtrue ~ Dir(β * (1 + noise * randn(S))), noise=0이면 Phase A와 동일.
Noise가 커질수록 α가 비대칭 → qtrue의 TV distance from q̂ 증가.

### β=0.1 Phase A/B 비교

| 방법 | Δmean | Δp95 | Δp99 | Δmax | win% | TV mean |
|------|-------|------|------|------|------|---------|
| PhaseA | +0.38 | **-0.10** | **-1.86** | **-4.24** | 39% | 0.733 |
| PhaseB n=0.5 | +0.41 | **-0.17** | **-2.07** | **-4.21** | 37% | 0.740 |
| PhaseB n=1.0 | +0.40 | **-0.12** | **-1.95** | **-4.03** | 38% | 0.742 |
| PhaseB n=2.0 | +0.39 | **-0.18** | **-1.41** | **-4.20** | 38% | 0.728 |
| PhaseB n=5.0 | +0.37 | +0.01 | **-1.17** | **-3.52** | 38% | 0.686 |

β=0.1에서는 이미 TV가 0.7 이상이라 Phase B noise 효과 미미.

### β별 Phase B 효과 요약

β=0.3~1.0에서 noise를 키우면 spread 확대, lower whisker가 더 길어짐.
β=5.0에서도 n=2.0~5.0이면 lower whisker가 0 아래로 내려감 — noise가 분포 불확실성을 키우면 DRO tail protection 복원.

### Paired Δ skewness 해석

Paired Δ boxplot에서 upper whisker (p99)가 lower whisker (p01)보다 짧음:
- NOM이 이길 때의 최대 이득 < DRO가 이길 때의 최대 이득
- DRO는 빈도에서 지지만, 극단 상황에서의 손실 폭을 줄여줌 (negative skew)
- β가 작을수록, noise가 클수록 이 비대칭이 강해짐

## Non-uniform q̂ 실험: sample1, sample3, sample8

q̂를 Dir(1,...,1)에서 3개 샘플링 (uniform에서 가장 먼 TV distance).
각 sample에 대해 nominal과 single-L DRO를 Benders로 풀고, 동일한 OOS protocol 적용.

### x* solutions per sample

| Model | sample1 | sample3 | sample8 |
|-------|---------|---------|---------|
| NOM (ε=0) | [6,34] | [6,34] | [3,6] |
| ε=0.1 | [6,34] | [6,34] | [6,18] |
| ε=0.3 | [6,18] | [6,18] | [6,18] |

- sample1, sample3: NOM = ε=0.1 (같은 x*). 비교 대상: **[6,34] vs [6,18]**
- sample8: ε=0.1 = ε=0.3 (같은 x*). 비교 대상: **[3,6] vs [6,18]**
- q̂ TV from uniform: sample1=0.457, sample3=0.400, sample8=0.410

### Sample1: NOM/ε01 [6,34](A) vs ε03 [6,18](B)

#### Δ (B − A): negative = B better

| β | Δmean | Δp95 | Δp99 | Δmax | win%(A) | TV mean |
|---|-------|------|------|------|---------|---------|
| 0.1 | **-0.56** | +1.71 | +2.45 | +3.60 | 44.6% | 0.793 |
| 0.3 | **-0.52** | +1.02 | +1.57 | +2.90 | 35.5% | 0.679 |
| 0.5 | **-0.49** | +0.80 | +1.24 | +2.06 | 31.5% | 0.626 |
| 1.0 | **-0.50** | +0.44 | +0.76 | +1.21 | 22.1% | 0.562 |
| 5.0 | **-0.51** | -0.05 | +0.14 | +0.54 | 3.3% | 0.481 |

#### Paired gap 분포

| β | p01 | p05 | median | p95 | p99 |
|---|-----|-----|--------|-----|-----|
| 0.1 | **-5.19** | -3.92 | -0.18 | +1.71 | +2.45 |
| 0.3 | **-3.52** | -2.57 | -0.36 | +1.02 | +1.57 |
| 0.5 | **-2.93** | -2.10 | -0.40 | +0.80 | +1.24 |
| 1.0 | **-2.12** | -1.64 | -0.45 | +0.44 | +0.76 |
| 5.0 | **-1.24** | -1.01 | -0.50 | -0.05 | +0.14 |

#### Absolute costs

| β | mean_A | mean_B | p95_A | p95_B | p99_A | p99_B | max_A | max_B |
|---|--------|--------|-------|-------|-------|-------|-------|-------|
| 0.1 | 19.770 | 19.212 | 22.790 | 21.997 | 24.352 | 22.787 | 26.619 | 23.551 |
| 0.3 | 19.759 | 19.244 | 21.772 | 21.154 | 22.684 | 21.851 | 24.899 | 23.135 |
| 0.5 | 19.732 | 19.239 | 21.293 | 20.845 | 22.049 | 21.327 | 24.354 | 22.312 |
| 1.0 | 19.762 | 19.264 | 20.919 | 20.480 | 21.524 | 20.900 | 22.482 | 21.593 |
| 5.0 | 19.767 | 19.254 | 20.298 | 19.855 | 20.531 | 20.075 | 21.020 | 20.524 |

### Sample3: NOM/ε01 [6,34](A) vs ε03 [6,18](B)

#### Δ (B − A): negative = B better

| β | Δmean | Δp95 | Δp99 | Δmax | win%(A) | TV mean |
|---|-------|------|------|------|---------|---------|
| 0.1 | **-0.94** | +1.38 | +2.01 | +2.99 | 34.3% | 0.778 |
| 0.3 | **-0.90** | +0.70 | +1.18 | +2.23 | 22.2% | 0.655 |
| 0.5 | **-0.87** | +0.48 | +0.92 | +1.84 | 16.5% | 0.596 |
| 1.0 | **-0.87** | +0.11 | +0.44 | +1.01 | 7.6% | 0.529 |
| 5.0 | **-0.89** | **-0.41** | **-0.21** | +0.10 | **0.1%** | 0.439 |

#### Paired gap 분포

| β | p01 | p05 | median | p95 | p99 |
|---|-----|-----|--------|-----|-----|
| 0.1 | **-6.25** | -4.42 | -0.54 | +1.38 | +2.01 |
| 0.3 | **-4.17** | -3.00 | -0.72 | +0.70 | +1.18 |
| 0.5 | **-3.57** | -2.59 | -0.74 | +0.48 | +0.92 |
| 1.0 | **-2.62** | -2.08 | -0.81 | +0.11 | +0.44 |
| 5.0 | **-1.65** | -1.42 | -0.88 | **-0.41** | **-0.21** |

#### Absolute costs

| β | mean_A | mean_B | p95_A | p95_B | p99_A | p99_B | max_A | max_B |
|---|--------|--------|-------|-------|-------|-------|-------|-------|
| 0.1 | 19.537 | 18.599 | 22.143 | 21.704 | 23.595 | 22.488 | 25.580 | 23.289 |
| 0.3 | 19.529 | 18.632 | 21.325 | 20.739 | 22.140 | 21.495 | 23.979 | 22.742 |
| 0.5 | 19.501 | 18.634 | 20.952 | 20.407 | 21.563 | 20.987 | 23.590 | 22.001 |
| 1.0 | 19.524 | 18.654 | 20.595 | 20.013 | 21.029 | 20.378 | 21.756 | 21.235 |
| 5.0 | 19.532 | 18.643 | 20.014 | 19.297 | 20.213 | 19.550 | 20.621 | 20.012 |

### Sample8: NOM [3,6](A) vs ε01/ε03 [6,18](B)

#### Δ (B − A): positive = A better

| β | Δmean | Δp95 | Δp99 | Δmax | win%(A) | TV mean |
|---|-------|------|------|------|---------|---------|
| 0.1 | **+0.75** | +3.19 | +3.67 | +4.32 | **67.4%** | 0.769 |
| 0.3 | **+0.75** | +2.56 | +2.96 | +3.47 | **74.0%** | 0.643 |
| 0.5 | **+0.73** | +2.24 | +2.65 | +3.07 | **77.8%** | 0.586 |
| 1.0 | **+0.75** | +1.83 | +2.18 | +2.71 | **84.9%** | 0.520 |
| 5.0 | **+0.74** | +1.24 | +1.42 | +1.72 | **99.3%** | 0.440 |

#### Paired gap 분포

| β | p01 | p05 | median | p95 | p99 |
|---|-----|-----|--------|-----|-----|
| 0.1 | **-3.90** | -2.66 | +0.99 | +3.19 | +3.67 |
| 0.3 | -2.27 | -1.33 | +0.85 | +2.56 | +2.96 |
| 0.5 | -1.64 | -0.93 | +0.80 | +2.24 | +2.65 |
| 1.0 | -1.03 | -0.47 | +0.79 | +1.83 | +2.18 |
| 5.0 | +0.03 | +0.22 | +0.75 | +1.24 | +1.42 |

#### Absolute costs

| β | mean_A | mean_B | p95_A | p95_B | p99_A | p99_B | max_A | max_B |
|---|--------|--------|-------|-------|-------|-------|-------|-------|
| 0.1 | 18.818 | 19.569 | 22.559 | 22.146 | 24.512 | 22.636 | 27.081 | 23.179 |
| 0.3 | 18.848 | 19.602 | 21.359 | 21.391 | 22.424 | 21.863 | 24.981 | 22.933 |
| 0.5 | 18.857 | 19.590 | 20.910 | 21.094 | 21.697 | 21.593 | 24.164 | 22.415 |
| 1.0 | 18.867 | 19.615 | 20.396 | 20.760 | 20.951 | 21.123 | 22.035 | 21.632 |
| 5.0 | 18.865 | 19.608 | 19.601 | 20.160 | 19.866 | 20.370 | 20.420 | 20.755 |

### Uniform vs Sample8: 같은 x* pair [3,6] vs [6,18] 비교

Sample8과 uniform은 동일한 NOM [3,6] vs DRO [6,18] 비교이므로 직접 대조 가능.

#### Δ (DRO − NOM) 비교

| β | uniform Δmean | sample8 Δmean | uniform win%(NOM) | sample8 win%(NOM) |
|---|------|------|------|------|
| 0.1 | +0.38 | +0.75 | 39% | 67% |
| 0.3 | +0.36 | +0.75 | 35% | 74% |
| 0.5 | +0.38 | +0.73 | 31% | 78% |
| 1.0 | +0.38 | +0.75 | 26% | 85% |
| 5.0 | +0.39 | +0.74 | 8% | 99% |

#### Paired gap skewness 비교 (β=0.1)

| | uniform | sample8 |
|---|---------|---------|
| p01 | **-3.97** | **-3.90** |
| median | +0.56 | +0.99 |
| p99 | +3.25 | +3.67 |
| \|p01\|/\|p99\| | 1.22 (skewed) | 1.06 (≈ symmetric) |

#### Absolute max 비교 (β=0.1)

| | uniform NOM | uniform DRO | sample8 NOM | sample8 DRO |
|---|---|---|---|---|
| max | 28.08 | 23.84 | 27.08 | 23.18 |

질적으로 비슷: NOM이 mean 우세, absolute max는 DRO가 낮음, p01 negative (극단 tail에서 DRO 승리).
차이: sample8에서 NOM 우세 2배 강함 (Δmean +0.75 vs +0.38), negative skew 약화 (1.06 vs 1.22).
원인: non-uniform q̂로 인한 follower h* 편향이 NOM 우세를 강화하고 DRO tail protection skew를 약화.

### Non-uniform q̂ Key Findings

1. **Sample1, Sample3: DRO(ε=0.3)가 mean에서도 우세** — uniform q̂와 반대
   - Sample1: Δmean ≈ -0.5 (모든 β), β=5.0에서도 A wins 3.3%
   - Sample3: Δmean ≈ -0.9 (모든 β), β=5.0에서 A wins **0.1%** — DRO 압도적
2. **Sample8: NOM이 모든 β에서 우세** — uniform과 질적으로 유사하나 2배 강함
   - Δmean ≈ +0.75 (모든 β), β=5.0에서 A wins 99.3%
   - β=0.1에서만 p01 whisker가 0 아래 (극소수 DRO 승리)
3. **q̂ 비대칭성이 결과 방향을 결정**: uniform q̂에서의 "mean에서 NOM 우세 + tail에서 DRO 보호" 패턴이 non-uniform q̂에서는 성립하지 않음
4. **NOM과 ε=0.1이 같은 x*를 선택** (sample1, sample3) — ε=0.1이 너무 작아서 nominal과 구별 불가
5. **ε=0.1과 ε=0.3이 같은 x*를 선택** (sample8) — [6,18]이 robust solution으로 수렴

### DRO tail protection과 β/ε 관계의 괴리

DRO가 worst-case tail에서 NOM보다 우세해지는 조건 요약:

| q̂ | NOM mean 우세 여부 | tail에서 DRO 우세 시작 | 해당 β의 TV(q_true, q̂) |
|---|---|---|---|
| uniform | O (Δmean≈+0.38) | β≤0.5 (Δp99 negative) | TV≈0.47~0.73 |
| sample8 | O (Δmean≈+0.75) | β≤0.1 (p01만 negative) | TV≈0.77 |
| sample1 | X (DRO mean 우세) | 모든 β | — |
| sample3 | X (DRO mean 우세) | 모든 β | — |

**문제**: ε=0.1~0.3인데, DRO tail protection이 나타나려면 TV(q_true, q̂) ≈ 0.5~0.8이 필요.
이는 ε의 **5~8배**에 해당하며, β=0.1의 Dir은 사실상 random (한 component에 weight 집중).

즉, DRO의 ambiguity set 반경(ε) 내의 분포 변동에서는 NOM과 DRO 차이가 미미하고,
ε를 훨씬 초과하는 극단적 분포 이동에서야 DRO의 "flatter cost profile"이 효과를 발휘.
이는 single-layer DRO (ε̃=0)의 구조적 한계를 시사:
- DRO가 x*를 통해 달성하는 variance reduction은 worst-case q에 대한 것이지, OOS variance 전반에 대한 것이 아님
- non-uniform q̂에서는 follower h*의 편향이 지배적이며, x* 선택만으로는 이를 제어 불가
- **two-layer DRO (ε̃ > 0)가 follower h* 분산을 제어**하여 이 한계를 보완할 수 있는지가 관건

### Worst-case q* 집중도 분석 (sample8, DRO [6,18])

DRO 모델의 worst-case q*가 얼마나 집중되는지를 Dir(β=0.1)과 비교.
q* = argmax_q Σ q_s·flow_s  s.t. TV(q, q̂) ≤ ε, q̂=sample8.

#### Worst-case q* 집중도

| 지표 | DRO q* (ε=0.1) | DRO q* (ε=0.3) | Dir(β=0.1, S=20) |
|------|----------------|----------------|------------------|
| N_eff (1/Σp²) | **8.8** | **5.0** | **3.2** |
| \|q≥0.01\| | 13 | 9 | ~6.2 |
| \|q≥0.05\| | 6 | 5 | ~3.9 |
| top-3 share | 47.2% | 67.1% | 84.6% |
| max(q) | 0.178 (s=4) | 0.365 (s=9) | ~0.505 (mean) |

#### Worst-case q* 이동 패턴

ε=0.1: q̂에서 거의 움직이지 않음. s=9 (flow=23.55, 최대급)에만 +0.1 이동.
ε=0.3: s=9에 weight를 0.065→0.365로 대폭 증가, s=12 (flow=15.24, 최소급)에서 0.159→0.129로 감소.

| s | flow | q̂ | q*(ε=0.1) | q*(ε=0.3) | 역할 |
|---|------|---|-----------|-----------|------|
| 9 | 23.55 | 0.065 | 0.165 | **0.365** | 최대 flow → weight 증가 |
| 4 | 20.27 | 0.178 | 0.178 | 0.178 | q̂ 최대 → 변동 없음 |
| 17 | 22.58 | 0.127 | 0.127 | 0.127 | 변동 없음 |
| 12 | 15.24 | 0.159 | 0.129 | 0.129 | 최소 flow → weight 감소 |
| 18 | 7.16 | 0.070 | 0.070 | 0.070 | 최저 flow인데 변동 없음 |

#### 핵심 관찰

1. **DRO q*는 Dir(0.1)보다 훨씬 덜 집중됨**: ε=0.3에서도 N_eff=5.0 vs Dir(0.1)의 3.2, top-3=67% vs 85%
2. **TV ball 내의 이동은 제한적**: q̂ 구조를 대부분 유지하며, 최대 flow 시나리오에 weight를 집중시키는 방향으로만 이동
3. **DRO가 대비하는 worst-case와 Dir(0.1) 실현값의 괴리**: Dir(0.1)은 1-2개 시나리오에 90%+ weight가 몰리지만, DRO q*는 5~9개 시나리오에 분산 → DRO가 대비하지 않는 극단적 집중이 Dir(0.1)에서 발생
4. **β=0.1에서만 DRO tail protection이 나타나는 이유**: Dir(0.1)의 실현값이 TV ball **바깥**으로 나가는 경우가 빈번 (TV ≈ 0.77 >> ε=0.1~0.3), 이때 DRO x*의 "flatter flow profile"이 우연히 도움이 되는 것

## 네 가지 q̂ 비교 종합

### 1. q̂가 승패 방향 자체를 결정
- uniform, sample8 → NOM mean 우세 (Δmean > 0)
- sample1, sample3 → DRO mean 우세 (Δmean < 0)
- 같은 x* pair ([6,34] vs [6,18])인데도 q̂에 따라 결과가 뒤집힘
- 원인: h*가 q̂로 1회 계산되므로, q̂가 flows profile 자체를 바꿈

### 2. DRO variance reduction이 non-uniform q̂에서 약화 가능
- uniform q̂: DRO boxplot이 NOM보다 짧음 (전형적 φ-divergence DRO 특성)
- non-uniform q̂ (sample3): DRO boxplot이 NOM보다 더 길어지는 현상 관찰
- 가설: TV-DRO가 줄이는 건 q̂-weighted variance이지 전체 variance가 아닐 수 있음
- **추후 분석 필요**: 다른 네트워크/q̂에서도 재현되는지, 원인이 q̂ 비대칭 vs h* 편향인지

### 3. NOM이 mean 우세인 경우 (uniform, sample8): OOS가 q̂에서 크게 벗어나야 DRO가 이김
- ε=0.1~0.3인데, tail protection 발현에는 TV(q_true, q̂) ≈ 0.5~0.8 필요 (ε의 5~8배)
- DRO worst-case q*는 N_eff=5~9로 완만, Dir(0.1)은 N_eff=3.2로 극단 집중
- DRO가 "대비하지 않는" 분포가 실제로 발생 → TV ball 바깥에서야 효과

### 4. NOM = ε=0.1: nominal solution의 안정성
- sample1, sample3에서 NOM = ε=0.1 (같은 x*), sample8에서 ε=0.1 = ε=0.3
- 이는 ε=0.1이 작아서가 아니라, **nominal x*가 이미 small TV ball 내에서 robust**하다는 증거
- ε ∈ (0.1, 0.3) 어딘가에서 전환 threshold 존재 — 후보 x*가 discrete이므로 threshold도 discrete

### 5. 문제 구조 (네트워크 topology)의 지배적 영향
- Polska γ=2에서 유효한 interdiction plan이 극소수: [6,34], [6,18], [3,6] 정도
- q̂나 ε보다 **네트워크 topology + budget constraint가 x* 선택을 지배**
- NOM이든 DRO든 비슷한 x*로 수렴 — distributional robustness가 leader decision에 미치는 영향이 제한적
- 반면 **follower h*는 continuous decision**이라 distributional assumption에 훨씬 민감
- → **two-layer DRO (ε̃ > 0)가 h*를 robust하게 만드는 것이 single-layer x* 변경보다 유의미할 가능성**

## Plots

### q̂ = uniform
- `plots/polska_factor_beta01_phaseAB.png` — β=0.1 Phase A/B boxplot
- `plots/polska_factor_beta0p3_phaseAB.png` — β=0.3 Phase A/B boxplot
- `plots/polska_factor_beta0p5_phaseAB.png` — β=0.5 Phase A/B boxplot
- `plots/polska_factor_beta1p0_phaseAB.png` — β=1.0 Phase A/B boxplot
- `plots/polska_factor_beta5p0_phaseAB.png` — β=5.0 Phase A/B boxplot
- `plots/polska_factor_paired_hbox_both.png` — conditional paired Δ (nom-tail vs dro-tail)

### q̂ = sample1 (NOM/ε01 [6,34] vs ε03 [6,18])
- `plots/sample1/beta0p1_phaseAB.png` — β=0.1 Phase A/B boxplot
- `plots/sample1/beta0p3_phaseAB.png` — β=0.3 Phase A/B boxplot
- `plots/sample1/beta0p5_phaseAB.png` — β=0.5 Phase A/B boxplot
- `plots/sample1/beta1p0_phaseAB.png` — β=1.0 Phase A/B boxplot
- `plots/sample1/beta5p0_phaseAB.png` — β=5.0 Phase A/B boxplot

### q̂ = sample3 (NOM/ε01 [6,34] vs ε03 [6,18])
- `plots/sample3/beta0p1_phaseAB.png` — β=0.1 Phase A/B boxplot
- `plots/sample3/beta0p3_phaseAB.png` — β=0.3 Phase A/B boxplot
- `plots/sample3/beta0p5_phaseAB.png` — β=0.5 Phase A/B boxplot
- `plots/sample3/beta1p0_phaseAB.png` — β=1.0 Phase A/B boxplot
- `plots/sample3/beta5p0_phaseAB.png` — β=5.0 Phase A/B boxplot

### q̂ = sample8 (NOM [3,6] vs ε01/ε03 [6,18])
- `plots/sample8/beta0p1_phaseAB.png` — β=0.1 Phase A/B boxplot
- `plots/sample8/beta0p3_phaseAB.png` — β=0.3 Phase A/B boxplot
- `plots/sample8/beta0p5_phaseAB.png` — β=0.5 Phase A/B boxplot
- `plots/sample8/beta1p0_phaseAB.png` — β=1.0 Phase A/B boxplot
- `plots/sample8/beta5p0_phaseAB.png` — β=5.0 Phase A/B boxplot
