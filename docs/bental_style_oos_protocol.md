# Ben-Tal Style OOS Protocol (q̂ = uniform, Normal sampling)

## 목표 정리

| 목표 | 달성 여부 | Note |
|---|---|---|
| X: Ben-Tal 2013 style calibration/sampling framework 채택 | ✅ | 방법론적 citation 확보 |
| Z: Mean OOS에서 DRO가 nominal을 이김 | ❌ (구조적 제약) | Uniform-centered sampling에서 $\mathbb{E}[p^{(r)}] = \hat{q}$ → nominal이 mean OOS에서 보호됨 |

**이 protocol은 X 목표에 집중**. Z는 별도 data-driven pivot 경로 (문서 `data_driven_oos_protocol.md`)에서 해결.

**부수적 기대**: Normal sampling이 Dir symmetry를 깨면서 two-layer vs single-layer 차이가 드러날 가능성 (primary가 아닌 bonus).

---

## 핵심 구분: Calibration vs OOS Sampling

이 protocol에서 $\varepsilon$-ball 두 가지 **별개 용도로 등장**:

### 용도 1: DRO Radius (tuning parameter)

Leader solve에서 사용하는 ambiguity ball radius $\varepsilon_1, \varepsilon_2$.

- **역할**: Optimization model의 hyperparameter
- **결정 방식**: **Tuning**. 고정된 공식에 묶일 필요 없음. Sensitivity analysis 축.
- **Sweep**: $\varepsilon \in \{0.05, 0.1, 0.2, 0.3, 0.5\}$ 또는 $N$ 역변환 값

### 용도 2: OOS Sampling 영역

Out-of-sample evaluation에서 hypothetical $p^{(r)}$를 뽑는 "confidence region"의 크기.

- **역할**: OOS metric이 "얼마나 넓은 영역에 대한 성능인가"를 통제
- **결정 방식**: **Concentration inequality로 원칙적 공식 적용**. Ben-Tal-style calibration.
- **Fix**: 실험 세팅의 일부로 principled하게 결정, 그 위에서 DRO radius를 sweep.

**두 용도가 같은 $\varepsilon$ 값을 공유할 필요 없음**. Reviewer 방어 포인트: "calibration은 evaluation 영역을 정의하고, DRO radius는 최적화 model의 choice로 분리된다".

---

## TV-Specific Calibration (용도 2)

**Ben-Tal 2013의 $\chi^2$-based asymptotic formula는 TV에 직접 적용 불가** — TV의 generating function $\phi(t) = \frac{1}{2}|t-1|$이 $t=1$에서 미분 불가능하여 $\phi''(1)$가 존재하지 않음.

TV-native calibration 두 옵션을 protocol에 모두 포함:

### Option A: Weissman Concentration Bound (primary)

Weissman et al. (2003) tight $\ell_1$ bound for categorical empirical:

$$\Pr\!\left(\|\hat{q}_N - p\|_1 \geq \varepsilon\right) \leq (2^S - 2)\exp\!\left(-\frac{N\varepsilon^2}{2}\right)$$

$(1-\alpha)$ coverage 만족:

$$\boxed{\varepsilon_{\text{Weissman}}(N, S, \alpha) = \sqrt{\frac{2}{N}\!\left(S\log 2 + \log\!\tfrac{1}{\alpha}\right)}}$$

**(Weissman의 TV 반감은 $\frac{1}{2}\|\cdot\|_1$을 쓰면 factor 1/2 추가; 본 protocol은 $\varepsilon = \|\cdot\|_1$ 정의로 통일)**

특징:
- TV-native (TV에 대한 직접 concentration bound)
- Finite-sample valid (asymptotic 아님)
- $S$ dependence 명시적 — $O(\sqrt{S/N})$
- $\alpha = 0.05$ 기본

Citation: "TV-based adaptation of the Ben-Tal et al. (2013) confidence-set calibration, using the Weissman et al. (2003) concentration inequality appropriate for the non-smooth TV divergence."

### Option B: Pinsker Inequality via KL

Pinsker: $\text{TV}(p, q) \leq \sqrt{\frac{1}{2}\text{KL}(p \| q)}$.

KL에는 Ben-Tal asymptotic 공식 적용 가능 ($\phi_{\text{KL}}''(1) = 1$):

$$\rho_{\text{KL}} = \frac{1}{2N}\chi^2_{S-1, 1-\alpha}$$

이를 TV로 upper bound:

$$\boxed{\varepsilon_{\text{Pinsker}}(N, S, \alpha) = \sqrt{\frac{\chi^2_{S-1, 1-\alpha}}{N}}}$$

**(여기서 $\|\cdot\|_1 \leq 2 \cdot \text{TV}$로 scaling 맞춤; 필요 시 2배 또는 1/2 factor 조정)**

특징:
- Ben-Tal asymptotic framework 재활용
- 보수적 (Pinsker inequality loose)
- $S$ dependence implicit (via $\chi^2$ d.o.f.)
- Large $S$에서 Weissman보다 큰 $\varepsilon$ 생성 (보통)

Citation: "Following Ben-Tal et al. (2013) for KL-divergence calibration, extended to TV via the Pinsker inequality."

### 두 option 비교 실험

두 calibration 모두 protocol에 포함하여 **robustness check**. 예상:

- $\varepsilon_{\text{Pinsker}} > \varepsilon_{\text{Weissman}}$ typically (보수적)
- OOS sampling region이 다르면 결과의 quantitative value는 달라짐
- **Qualitative 결론 (nominal vs DRO ordering)은 같아야 함** — 같으면 robustness 확인, 다르면 흥미로운 발견

비교는 appendix에.

### 참고 — Dir quantile (기존 manuscript 방식)

$$\varepsilon_{\text{Dir}}(\beta) = 95\%\text{-quantile of } \|q - \hat{q}\|_1,\; q \sim \text{Dir}(\beta \cdot \mathbf{1})$$

이건 Bayesian credible region 해석. Frequentist confidence set과 다름. 기존 결과와 비교용으로만 유지.

---

## Protocol

### Step 1: Fix $\hat{q}$ and DRO radius

$$\hat{q} = \frac{1}{S}\mathbf{1}_S$$

DRO radius는 **tuning parameter** (Ben-Tal calibration에 묶이지 않음):

$$\varepsilon_1^{\text{DRO}}, \varepsilon_2^{\text{DRO}} \in \text{sweep range}$$

Sweep 예: $\{0.05, 0.1, 0.2, 0.3, 0.5\}$ 또는 $N \in \{10, 30, 100, 300\}$에 해당하는 Weissman 값들.

### Step 2: Solve three leader models

각 $\varepsilon^{\text{DRO}}$에 대해:

- **Nominal**: $\varepsilon_1 = \varepsilon_2 = 0$, center $\hat{q}$
- **Single-layer DRO**: $\varepsilon_1 = \varepsilon^{\text{DRO}}, \varepsilon_2 = 0$
- **Two-layer DRO**: $\varepsilon_1 = \varepsilon_2 = \varepsilon^{\text{DRO}}$

Output: $x_{\text{nom}}^*, x_{\text{SL}}^*, x_{\text{TL}}^*$ per radius setting.

### Step 3: Fix OOS sampling region

**별개 parameter** (DRO radius와 독립적으로 선택):

$$\varepsilon^{\text{OOS}} = \varepsilon_{\text{Weissman}}(N_{\text{cal}}, S, 0.05) \quad \text{or} \quad \varepsilon_{\text{Pinsker}}(N_{\text{cal}}, S, 0.05)$$

$N_{\text{cal}}$은 hypothetical sample size로서 confidence region의 반경을 결정하는 **radius scaling knob**. "Leader가 $N_{\text{cal}}$개의 데이터를 관측하여 $\hat{q}$를 구성했다고 가정했을 때의 95% confidence region 반경"이라는 해석. $\hat{q}$가 실제 empirical이 아니라 uniform prior이지만, Ben-Tal 2013 (Section 6.4)도 동일하게 $q_N$을 reference distribution으로 고정하고 $N$을 sweep parameter로 사용. Fix (예: $N_{\text{cal}} = 100$), 또는 별도 sweep.

### Step 4: Normal sampling of hypothetical $p^{(r)}$

Ben-Tal Section 6.4 방식:

```
for r in 1..J:
    repeat:
        for i in 1..S-1:
            p_i ~ Normal(1/S, σ)
        p_S = 1 - sum(p[1..S-1])
    until (p ≥ 0 for all components) AND (‖p - q̂‖_1 ≤ ε^OOS)
```

$\sigma$는 acceptance rate 높이도록 pre-tune. 경험식: $\sigma \approx \varepsilon^{\text{OOS}} / (2\sqrt{S})$를 초기값으로, rejection rate 모니터링하며 조정.

**Acceptance 95% target**: $\sigma$를 iterative하게 조정하여 "약 95%의 sampled $p$가 TV ball 안"이 되도록.

### Step 5: Per-sample evaluation — Option b' (Ben-Tal-consistent, nested, 채택)

**Nested 구조**: $\tilde{p}$ outer (비싼 follower LP, $M$번), $p_{\text{true}}$ inner (싼 weighted sum, $L$번).

```
for j in 1..M:                                    # outer: follower belief
    q̃^(j) ← sample_bental_normal(ε^OOS)
    h*^(j) = follower(x*, q̃^(j))                  # expensive (LP solve)
    flows^(j) = maxflow_per_scenario(x*, h*^(j))

    for ℓ in 1..L:                                 # inner: nature's truth
        p_true^(j,ℓ) ← sample_bental_normal(ε^OOS)  # same σ, independent
        Y^(j,ℓ) = Σ_k p_true^(j,ℓ)_k · flows^(j)_k  # cheap (dot product)

    Ȳ^(j) = (1/L) Σ_ℓ Y^(j,ℓ)

Ȳ = (1/M) Σ_j Ȳ^(j)
```

$p_{\text{true}}$와 $\tilde{p}$는 **같은** $\mathcal{N}(\hat{q}, \sigma)$에서 **독립** draw (symmetric ignorance).

**Target estimand**:
$$\bar{Y}(x^*) = \mathbb{E}_{\tilde{p}}\!\left[\mathbb{E}_{p_{\text{true}}}\!\left[\sum_k p_{\text{true},k}\, Q(h^*(x^*, \tilde{p}), x^*, \xi^k) \mid \tilde{p}\right]\right]$$

**Variance decomposition** (manuscript 식 12):
$$\text{Var}(Y) = \underbrace{\text{Var}_{\tilde{p}}[\mathbb{E}_{p_{\text{true}}}[Y \mid \tilde{p}]]}_{\text{follower belief effect}} + \underbrace{\mathbb{E}_{\tilde{p}}[\text{Var}_{p_{\text{true}}}[Y \mid \tilde{p}]]}_{\text{nature effect}}$$

Empirical estimators:
- $\widehat{\text{Var}}_{\tilde{p}} = \frac{1}{M-1}\sum_j (\bar{Y}^{(j)} - \bar{\bar{Y}})^2$
- $\widehat{\mathbb{E}[\text{Var}_{p}]} = \frac{1}{M}\sum_j \frac{1}{L-1}\sum_\ell (Y^{(j,\ell)} - \bar{Y}^{(j)})^2$

**Rationale**: Leader가 관측한 $\hat{q}$로부터 confidence set $\{p : \|p - \hat{q}\|_1 \leq \varepsilon^{\text{OOS}}\}$을 구성. Nature의 $p_{\text{true}}$와 follower의 $\tilde{p}$가 각각 독립적으로 이 ball 안 어딘가에 위치. Follower가 $\hat{q}$를 직접 관측하는 것이 아니라, follower의 실제 distribution이 $\hat{q}$ 근처에 있을 것이라는 해석. Two-layer DRO의 $\varepsilon_2$가 follower belief uncertainty radius로 자연스럽게 대응.

**σ에 noise를 주지 않는 이유**: $\tilde{p}$에 추가 noise를 주면 $\varepsilon_2 > \varepsilon_1$ asymmetric 가정을 imply → symmetric ignorance framework와 충돌. 비대칭 실험은 sensitivity section에서 별도 처리.

> **Note**: 기존 Step 5 (Option a, flat $p = \tilde{p}$)는 follower가 truth를 안다는 가정이며 variance decomposition 불가. Option b' nested가 two-layer 모델과 더 consistent.

### Step 6: Metrics

각 $(x^*, \varepsilon^{\text{DRO}})$에 대해:

- **Mean**: $\bar{\bar{Y}} = (1/M) \sum_j \bar{Y}^{(j)}$ — primary comparison
- **Worst-case**: $\max_j \bar{Y}^{(j)}$ — Ben-Tal style
- **Range**: $\max_j \bar{Y}^{(j)} - \min_j \bar{Y}^{(j)}$
- **Std**: $\sqrt{\text{Var}(\bar{Y}^{(j)})}$ — outer-level variability
- **Variance decomposition**: follower share = $\widehat{\text{Var}}_{\tilde{p}} / (\widehat{\text{Var}}_{\tilde{p}} + \widehat{\mathbb{E}[\text{Var}_{p}]})$
- **Quantiles**: 90%, 95%, 99%

---

---

## Normal sampling 구현 세부

### $\sigma$ tuning

Target: ~95% acceptance rate, samples의 ~95%가 TV ball 안.

Pre-experiment calibration:
```
σ_init = ε^OOS / (2 * sqrt(S))
for σ in {σ_init * k : k = 0.5, 0.75, 1.0, 1.5, 2.0}:
    sample 1000 p's, measure acceptance rate + TV ball coverage
    choose σ closest to targets
```

실제 Ben-Tal 2013 (Section 6.4)에선 component별 $\sigma_i$를 hand-tune했지만 우리는 symmetric 세팅 ($\hat{q}$ = uniform)이라 single $\sigma$로 충분.

### Degenerate cases

- $p_S = 1 - \sum_{i<S} p_i < 0$: rejection
- $\varepsilon^{\text{OOS}}$가 매우 작으면 acceptance rate 낮음 → $\sigma$ 더 축소
- $\varepsilon^{\text{OOS}}$가 매우 크면 simplex boundary에 자주 걸림 → Dirichlet fallback 고려

### Validation

Pilot에서 generated $p^{(r)}$ distribution의 empirical moments를 체크:
- $\mathbb{E}[p^{(r)}] \approx \hat{q}$ (대칭성 확인)
- $\text{Var}(p^{(r)}_i) \approx \sigma^2$
- $\|p^{(r)} - \hat{q}\|_1$의 95% quantile $\approx \varepsilon^{\text{OOS}}$

---

## Calibration 독립성의 방어 논리

Reviewer 예상 질문: "왜 DRO radius는 tuning이고 OOS는 Weissman fixed인가?"

**답변 template**:

> "The DRO radius $\varepsilon^{\text{DRO}}$ is a modeling parameter of the optimization formulation, whose choice balances robustness and in-sample performance — it is naturally a hyperparameter to be swept or tuned via cross-validation. In contrast, the OOS evaluation region $\varepsilon^{\text{OOS}}$ serves a distinct purpose: it defines the statistical scope over which out-of-sample performance is measured, answering 'how does each solution perform across plausible deviations from the nominal distribution?'. Following Ben-Tal et al. (2013), we fix this evaluation region using a concentration-based confidence set, which provides a principled 95% coverage guarantee. Decoupling these two uses of $\varepsilon$ allows us to separately examine (i) how DRO radius choice affects the optimized solution and (ii) how each solution generalizes to a well-defined statistical neighborhood."

---

## 예상 결과 및 해석

### 예상 패턴 (Ben-Tal 재현)

| Metric | Nominal | Single-L DRO | Two-L DRO |
|---|---|---|---|
| Mean OOS | **최소 or 동등** (uniform unbiased) | 비슷 | 비슷 |
| Worst-case OOS | 나쁨 | 좋음 | 더 좋음 (기대) |
| Range / Std | 큼 | 작음 | 더 작음 |

**Honest reporting**: "Mean OOS에서 nominal이 여전히 경쟁력 있음. DRO의 value는 worst-case, variance reduction에서 드러남." 이건 Ben-Tal 2013의 framing 그대로.

### Two-layer가 single-layer를 이길 조건

Follower의 recovery가 $\tilde{q}$에 대해 충분히 비선형 (convex)이면 Two-L이 Single-L을 mean에서 이길 가능성. Polska 진단에서는 이 비선형성이 약했음. Normal sampling이 이 구조를 바꿀지는 empirical question.

### Null result 대응

만약 Two-L ≈ Single-L이 모든 setting에서:
- Two-layer의 **variance reduction** 효과로 narrative 전환 (식 12의 variance decomposition 활용)
- Two-layer가 **worst-case에서만** 차별적 가치 있다면 그대로 정직하게 report

---

## Section 재작성 여부

**이 protocol은 기존 manuscript Section 4를 거의 유지**:

- $\hat{q}$ = uniform: 기존 그대로
- Symmetric ignorance logic: 기존 그대로 유지 가능 (두 $\varepsilon$가 모두 ball 안에서 uncertainty)
- Calibration 부분만 "Dir quantile → Weissman (+ Pinsker)" 교체
- OOS sampling 부분 "Dir → Normal + rejection" 교체

변경 범위: Section 4.1–4.2의 calibration/OOS 2–3 페이지 정도. Full rewrite 아님.

---

## Open Questions / TODOs

1. **$\sigma$ 선택 방식**: single $\sigma$ vs. component별 $\sigma_i$. Uniform $\hat{q}$에선 single 충분. Confirm empirically.
2. **$N_{\text{cal}}$ 결정**: fixed (e.g., 100) vs. sweep. Main experiment에선 fixed, appendix에서 $N_{\text{cal}}$ sensitivity.
3. **Weissman의 $\log(1/\alpha)$ 항**: $\alpha=0.05$에서 $\log 20 \approx 3$, 작은 $S$에서 무시 못 함. Formula에 명시.
4. **Acceptance rate too low**: Polska $S$ 크면 Normal rejection 비효율적. Fallback으로 Dir-on-TV-ball로 바꾸는 안.
5. **Pinsker의 TV vs $\ell_1$ convention**: factor 2 주의. Manuscript notation과 consistent하게.

---

## 분석 히스토리

1. **기존 (Dir-based) 세팅 결과의 문제점**:
   - Polska factor_additive에서 nominal이 mean OOS에서 dominate
   - 원인: $\mathbb{E}_{p \sim \text{Dir}(\beta \cdot \mathbf{1})}[p] = \hat{q} = $ uniform → nominal이 mean OOS의 unbiased minimizer

2. **Ben-Tal 2013 방식 검토**:
   - Section 6.4 newsvendor 실험: $q_N$-centered Normal sampling
   - 결과: mean에서 non-robust ≥ robust, worst-case에서 robust가 이김
   - **Mean OOS 우위는 그들도 주장 안 함** — "average 유지 + worst-case 개선" framing

3. **결정**: X 목표 (Ben-Tal style framework 채택)에 집중. Z 목표는 data-driven pivot 문서에서 별도 처리.

4. **TV-$\phi$의 Ben-Tal calibration 문제**:
   - $\phi''(1)$ undefined for TV
   - Weissman (TV-native concentration) + Pinsker (KL via inequality) 두 옵션 채택

5. **Calibration vs OOS sampling 분리**:
   - DRO radius는 tuning parameter로 free
   - OOS region은 principled (Weissman/Pinsker) fixed
   - 이 분리가 reviewer 방어 논리의 핵심

6. **예상 결론 tone**: Mean에서 tie, worst/variance에서 DRO 우위. Honest reporting.
