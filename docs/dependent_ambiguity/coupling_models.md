# 진실 분포와 follower belief 의 종속 ambiguity set — 모델링 후보 (2026-10-07)

원고 (비제로섬 JOC 판) 의 Assumption 4 는 곱집합 𝒟 = D̂ × D̃ (두 TV 볼 모두 중심 q̂) 이다. 즉 진실 분포 p 와
follower belief p̃ 가 서로 독립적으로 움직인다. 실제로는 둘이 어느 정도 상관되어 있을 것이므로 결합된 집합
𝒟 ⊆ D̂ × D̃ 를 쓰는 후보를 정리한다.

공통 성질
- 어떤 결합 집합이든 투영이 같으면 𝒟 ⊆ proj₁𝒟 × proj₂𝒟 이므로 V*_𝒟(x) ≤ V*_rect(x). 현재 모델은 "보수적 rectangular hull".
- 유한 support 에서 결합 제약이 (p, p̃) 에 선형이면 Ω 에 행 몇 개만 추가된다. bilinear 항 (α × belief), F, x 무관성,
  Benders cut 공식은 그대로다. 증명 수정은 `dependent_ambiguity_note.tex` (같은 폴더) 에 정리.
- 반응집합 표현은 V*(x) = max_α max_{p ∈ D̂_𝒟(x,α)} R^p[φ(x,α,ξ)] 로 바뀐다.
  D̂_𝒟(x,α) = {p : ∃ p̃ 가 α 를 합리화, (p, p̃) ∈ 𝒟}. follower 의 반응을 보면 진실 분포의 후보가 줄어드는 구조다.

---

## 1. 거리 결합 (belief-coupled set) — 주 후보

𝒟_δ = {(p, p̃) : d_TV(p, q̂) ≤ ε̂,  d_TV(p̃, q̂) ≤ ε̃,  d_TV(p, p̃) ≤ δ}

- 해석: δ 는 follower 가 진실을 얼마나 정확히 아는지. 두 볼 모두 q̂ 중심 (리더는 자기 belief 만 앎).
- 특수 경우: δ ≥ ε̂ + ε̃ 이면 현재 rectangular 와 정확히 같음 (삼각부등식). δ = 0 이면 p̃ = p (공통이지만 ambiguous 한 prior).
  δ ≤ ε̃ − ε̂ 이면 follower 볼 제약이 자동 성립 → nested 형태 p̃ ∈ B_δ(p), p ∈ B_ε̂(q̂).
- 유효 반경: proj_p̃ 𝒟_δ = TV 볼 반경 ε̃' = min(ε̃, ε̂ + δ), proj_p 𝒟_δ = 반경 ε̂' = min(ε̂, ε̃ + δ).
  허용 반응은 rectangular 를 ε̃' 로 줄인 것과 같고, 그 위에 type 마다 리더 집합이 줄어드는 효과가 더해짐.
- Ω 추가: |a_s − d_s| ≤ g_s, Σ_s g_s ≤ 2δ (변수 S, 행 2S+1). (a = 리더 TV 분포 p, d = follower belief p̃)
- Bertsimas–Sim 과의 유비: rectangular = budget 없는 box, δ = "belief-alignment budget".
  다만 합 budget d(p,q̂) + d(p̃,q̂) ≤ Γ 가 아니라 차이 budget d(p,p̃) ≤ δ 라서, 시점 간 변화량에 budget 을 거는
  dynamic uncertainty set (Lorca & Sun 2015) 에 더 가깝다.
- δ 보정: follower belief 가 진실에서 뽑은 N_f 개 표본의 경험분포라면
  P(d_TV(P̂_f, P*) ≥ δ) ≤ (2^S − 2) e^{−2 N_f δ²} (Weissman et al. 2003) → δ = sqrt((S ln2 + ln(1/η)) / (2 N_f)).
  S 가 크면 느슨하므로 해석용.
- 결과 (note): 모든 정리가 확장됨. 새 증명이 필요한 것은 Lemma certificate (iii) 하나. Σ₂ᵖ-hard 는 모든 ε̃, δ > 0 에서 성립,
  saturation (zero-sum, 위험중립, ε̂ ≤ ε̃) 은 모든 δ ≥ 0 에서 성립. δ = 0 의 복잡도는 미해결.
- 수치 (results.md): SGB128 pair, ε̂ = ε̃ = 0.2, β = 0.4, x = [1] 에서 V*_δ 가 rect −871.67 → δ=0.1 −893.75 →
  δ=0 −944.25 (유효 반경으로 설명되지 않는 순수 결합 효과). 많은 인스턴스 (grid 3×3, location random pooled,
  Abilene δ=0.1) 에서는 rect 의 최악 (p, p̃) 가 이미 가까워 값이 바뀌지 않음 → 효과는 인스턴스 의존적.

## 2. Signal / likelihood-ratio 결합

follower 가 확률 κ 이상인 private signal θ 를 관측하고 진실 분포의 조건부 분포를 belief 로 가진다:
p̃ = P(· | θ),  dp̃/dp = π(θ|ξ)/π(θ) ≤ 1/π(θ).

𝒟_κ = {(p, p̃) : p ∈ D̂,  p̃ ∈ D̃,  κ p̃_s ≤ p_s ∀s}

- 정확한 특성화 (유한 support): {확률 κ 이상인 signal 실현 아래의 posterior} = {p̃ ∈ Δ : κ p̃ ≤ p}.
  역방향은 이진 signal π(θ=1 | ξ_s) = κ p̃_s / p_s (≤ 1) 로 구성: π(θ=1) = κ, posterior = p̃.
  → Bayesian 근거가 있는 결합.
- 해석: κ = 1 이면 p̃ = p (정보 없는 signal, common prior). κ → 0 이면 p̃ ≪ p 만 남음.
  1/κ = follower 의 private 정보가 belief 를 얼마나 날카롭게 만들 수 있는지의 상한.
- 구조: CVaR 위험 envelope 과 같은 꼴. Ω 에 이미 있는 (1−β) r_s ≤ a_s 와 같은 형태의 행 κ d_s ≤ a_s (S 개, 변수 없음).
  α-B&B 의 RLT 처리를 그대로 복사하면 됨. κ = 1 − β 이면 리더의 위험 재가중과 follower belief 가 같은 envelope 위에 놓임
  (CVaR = 확률 1−β 이상 사건 아래 최악 조건부 기댓값).
- 한계: 단측이라 follower 가 일부 시나리오를 무시 (p̃_s = 0) 하는 것은 막지 못하고, 진실이 배제한 시나리오
  (p_s = 0) 에는 belief 를 줄 수 없음 (절대연속).
- 변형 (모두 선형):
  - 양측 density ratio: L p ≤ p̃ ≤ U p (DeRobertis & Hartigan 1981) — dogmatic belief 방지.
  - ε-contamination: p̃ ≥ (1 − θ) p (follower belief = 진실 + 편향 θ, Huber; Berger & Berliner 1986).
- rectangular 포함: κ → 0 으로 보내도 절대연속이 남으므로 정확히 포함하려면 p̃ ∈ D̃ 와 교집합으로 둠.

## 3. Bayes plausibility (여러 follower type)

follower type 이 여러 개 (belief-menu / Bayesian Stackelberg 확장) 이고 type k 의 belief 가 p̃^k, 비중 w_k 일 때
진실이 type belief 들의 평균이어야 한다:

Σ_k w_k p̃^k = p

- 근거: signal 에 대한 posterior 들의 평균은 prior (splitting lemma, Aumann–Maschler; Kamenica & Gentzkow 2011).
  정보 구조 전체에 대해 worst-case 를 취하면 Bergemann–Morris 의 informationally robust 관점.
- 단일 follower (현재 모델) 에서는 type 이 하나라 걸 수 없고, type 이 여러 개일 때 의미가 있음.
- w 가 고정이면 선형 → Ω 에 그대로 들어감. w 도 ambiguous 하면 w_k p̃^k 이 bilinear (perspective 변수로 선형화 가능).
- 2 번 (likelihood-ratio) 과의 관계: 각 type 의 posterior 는 자동으로 w_k p̃^k ≤ p 를 만족 (κ = w_k 인 2 번).

## 기타 후보 (간단히)

- misspecified model (Berk–Nash, Esponda & Pouzo 2016): p̃ = argmin_{Q ∈ M_f} KL(p ‖ Q). 결정적 사상, ambiguity 는 M_f 에.
- 공통 factor (Goyal & Grand-Clément 2023 의 factor matrix 발상): 두 분포가 regime 내 조건부 분포는 공유하고 regime 가중치만 다름.
- KL 결합: KL(p̃ ‖ p) ≤ δ (Hansen–Sargent 식). 볼록이지만 polyhedral 이 아니라 현재 LP 기반 Ω 와는 맞지 않음.

## 선행연구 (대화에서 정리한 것)

- rectangularity: Epstein & Schneider (2003, JET, recursive multiple priors), Iyengar (2005, MOR),
  Nilim & El Ghaoui (2005, OR), Shapiro (2016, OR, Rectangular sets of probability measures).
- non-rectangular: Wiesemann, Kuhn & Rustem (2013, MOR; 일반 non-rectangular RMDP 는 NP-hard),
  Mannor, Mebel & Xu (2016, MOR; k-rectangular), Goyal & Grand-Clément (2023, MOR; factor model, doi 10.1287/moor.2022.1259).
- 확률변수 간 dependence (개념이 다름): Gao & Kleywegt (2017, Distributionally robust stochastic optimization with dependence structure).
- 게임과 ambiguity: Liu, Xu, Yang & Zhang (2018, EJOR; 플레이어별 독립 ambiguity set), Kajii & Ui (2005; multiple priors),
  Morris (1995; common prior assumption), Bergemann & Morris (2016, TE; BCE), Brooks & Du (2021, Econometrica),
  Dworczak & Pavan (2022, Econometrica).
- interdiction 비대칭 정보: Bayrak & Bailey (2008, Networks), Nguyen & Smith (2022, EJOR; follower 가 실제 비용 관측 → p̃ = δ_ξ 극단).
- survey: Beck, Ljubić & Schmidt (2023, EJOR).
- 원고 결합 (truth, follower belief) 을 bilevel DRO 에서 직접 다룬 논문은 찾지 못함 (2026-10 기준 검색 범위).
