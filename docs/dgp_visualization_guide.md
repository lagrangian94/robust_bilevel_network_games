# DGP Visualization Guide

`visualize_dgp()` 실행 시 생성되는 3개 그림 설명.

## 전제

- 네트워크 무관. 순수 Dirichlet sampling으로 DGP 파라미터 검증.
- S=10 (scenario 수)만 관여. capacity scenario/topology 무관.
- 세 거리 전부 **TV distance = L1/2** scale.

## 세 분포의 관계

```
         P* (true/nature)
        /              \
       /                \
    TV(q̂,q*)          TV(q̃,q*)
     /                    \
    /                      \
   q̂ (leader) ——————————— q̃ (follower)
             TV(q̂,q̃)
```

- `q̂ = (1/S)·1`: leader의 empirical (uniform, 고정)
- `q*` (q_true): 실제 nature 분포
- `q̃` (q_tilde): follower의 belief

---

## 그림 1: `dgp_fig1_who_is_close.png`

### 무엇을 보여주나
**"세 scenario에서 누가 누구에게 가까운가?"**

### 구조
3개 패널 (S, L, F). 각 패널에 히스토그램 3개 겹침:
- **파란색**: TV(q̂, q*) — leader의 q̂가 truth에서 얼마나 먼가
- **빨간색**: TV(q̂, q̃) — follower belief가 leader의 q̂에서 얼마나 먼가
- **초록색**: TV(q̃, q*) — follower belief가 truth에서 얼마나 먼가

### 읽는 법
왼쪽에 몰려있을수록 = 가깝다 = 정확하다.

### 기대 패턴

| 패널 | 기대 | 의미 |
|------|------|------|
| S | 세 색 다 비슷한 위치 | 둘 다 모른다 (symmetric) |
| L | **파란색만 왼쪽** | leader만 truth를 잘 안다 |
| F | **초록색만 왼쪽** | follower만 truth를 잘 안다 |

### 검증 기준
이 패턴이 나오면 β_H=50, β_L=0.3, κ=50이 세 scenario의 information 구조를 제대로 만들고 있다는 뜻.

---

## 그림 2: `dgp_fig2_epsilon_coverage.png`

### 무엇을 보여주나
**"calibrated ε이 실제 거리를 잘 커버하는가?"**

### 구조
6개 패널 (3 scenario × 2). 각 패널:
- **회색 히스토그램**: 실제 TV distance 분포 (10,000개 draw)
- **빨간 점선**: leader가 calibrate한 ε 값

왼쪽 열 = ε₁ (leader의 model error 커버), 오른쪽 열 = ε₂ (follower deviation 커버).

### 읽는 법
빨간 선 왼쪽에 히스토그램의 95%가 있으면 = ε이 적절.

### 각 패널 의미

| 패널 | ε | 값 | 의미 |
|------|---|-----|------|
| S 왼쪽 | ε₁ = ε(β=0.3) | ≈0.70 | leader의 model error 커버 |
| S 오른쪽 | ε₂ = ε(β=0.3) | ≈0.70 | follower deviation 커버 |
| L 왼쪽 | ε₁ = ε(β_H=50) | ≈0.077 | 작은 ε — truth≈q̂이니까 작아도 됨 |
| L 오른쪽 | ε₂ = ε(β_L=0.3) | ≈0.70 | 큰 ε — follower가 크게 벗어남 |
| F 왼쪽 | ε₁ = ε(β_L=0.3) | ≈0.70 | 큰 ε — leader가 truth에서 멂 |
| F 오른쪽 | ε₂ = ε(β_L=0.3) | ≈0.70 | 과대추정 — 실제론 follower가 정확하지만 leader는 이를 모름 |

### 핵심
F행 오른쪽의 ε₂ 과대추정은 **의도된 동작**. Leader는 follower가 정확한지 모르므로 큰 ε₂를 설정하고, 이로 인해 two-layer DRO가 overconservative해짐.

---

## 그림 3: `dgp_fig3_follower_calibration.png`

### 무엇을 보여주나
**"follower가 truth에 얼마나 가까운가? — 세 scenario 한눈에 비교"**

### 구조
1개 패널에 히스토그램 3개 겹침. 전부 같은 거리: **TV(q̃, q*)** = follower가 truth에서 얼마나 먼가.
- **회색**: Scenario S (baseline)
- **빨간색**: Scenario L
- **초록색**: Scenario F

점선은 각 분포의 mean.

### 읽는 법
- 빨간색(L)이 오른쪽 → follower 부정확 → two-layer DRO 가치 큼
- 초록색(F)이 왼쪽 → follower 정확 → two-layer DRO 불필요
- 회색(S)이 중간 → baseline

### 검증 기준
**빨간-회색-초록 순서로 오른쪽→왼쪽** 나열되면 파라미터 적절.

### 해석
이 그림이 "two-layer DRO가 언제 가치 있는가"의 직관:
- follower가 부정확할 때(L) → two-layer DRO 가장 유용
- follower가 정확할 때(F) → two-layer DRO 불필요

---

## ε Calibration 방법

`oos_dirichlet.jl`의 `calibrate_epsilon(K, β)`:

1. Dir(β·1_K)에서 10,000개 샘플 q 생성
2. 각 q와 uniform q̂=(1/K)·1의 L1 distance 계산
3. 95th percentile 구함
4. **ε = L1_quantile / 2** (TV = L1/2)

제약 조건에서 `Σ|b_s| ≤ 2ε` (L1 ≤ 2ε) 이므로 TV ≤ ε.

---

## 실행

```julia
include("true_dro/visualize_dgp.jl")
visualize_dgp()                    # 기본: S=10, β=0.3, β_H=50, β_L=0.3, κ=50
visualize_dgp(S=20)                # S=20으로 변경
visualize_dgp(β_H=100.0, κ=100.0) # 파라미터 조정
```
