# True-DRO-Exact via Lagrangian Decomposition: Full Derivation with TV Reformulation

---

# Part I: Problem Formulation

## 1. Problem Description

Maximum flow network interdiction game on directed network G = (V, A) with source s and sink t.

**Players and decisions:**
- **Leader (interdictor):** binary interdiction x ∈ X = {x ∈ {0,1}^|A| : 1ᵀx ≤ γ}
- **Follower (flow player):**
  1. Here-and-now capacity recovery h ∈ H = {h ∈ R₊^|A| : 1ᵀh ≤ w}
  2. Wait-and-see flow y(x, h, ξ) after observing uncertainty ξ

**Decision chronology:** x → h(x) → ξ → y(x, h, ξ)

**Heterogeneous beliefs:** Leader evaluates under P̂ (leader's belief), follower decides under P̃ (follower's belief). In general P̂ ≠ P̃.

**Notation:** Node-arc incidence matrix N = [N_y | N_ts] ∈ R^{m×(|A|+1)}, m = |V|-1. Arc capacity ξ̄_k^s under scenario s, interdiction effectiveness v_k ∈ [0,1].


## 2. Pessimistic Bilevel Formulation

### 2.1 Bilevel Program

Following the pessimistic bilevel framework (cf. Yanıkoğlu–Kuhn 2018, Goyal–Zhang–He 2022):

```
min_{x ∈ X}  sup_{h̃, ŷ}  E^{P̂}[d₀ᵀ ŷ(ξ)]

s.t.  h̃ ∈ arg max_{h̄ ∈ H} { cᵀh̄ + E^{P̃}[Q(h̄, x, ξ)] }         (λ)
      ŷ(ξ) ∈ arg max_{ȳ} { d₀ᵀȳ : Aȳ ≤ b_x(ξ) - Bh̃ }    P̂-a.s.
```

where Q(h̄, x, ξ) = max_{ȳ} {d₁ᵀȳ : Aȳ ≤ b_x(ξ) - Bh̄} and b_x(ξ) = {ξ_k(1 - v_k x_k)}.

**Remark (Heterogeneous distributions).** Follower optimizes h under P̃, leader evaluates ŷ under P̂. This two-distribution structure distinguishes our model from single-distribution settings.


### 2.2 Strong Duality of the Inner Problem

**Assumption 1 (Shared Support).** The distributions P̂ and P̃ are supported on a common compact set Ξ ⊂ R^k, and satisfy E^{P̂}[‖ξ‖²] < ∞, E^{P̃}[‖ξ‖²] < ∞.

**Remark (Relatively Complete Recourse).** In the network interdiction setting, the follower's feasible set {y ∈ R₊ⁿ : Ay + Bh ≤ b_x(ξ)} is non-empty and bounded for all ξ ∈ Ξ, x ∈ X, and h ∈ H. Non-emptiness: y = 0 is always feasible. Boundedness: arc capacities are non-negative and finite on compact Ξ. Hence relatively complete recourse (Yanıkoğlu–Kuhn 2018) is automatically satisfied under both P̂ and P̃.

**Proposition 1 (Strong Duality).** Under Assumption 1, with c = 0 and d₁ = d₀, the inner pessimistic problem (P) for fixed x ∈ X:

```
(P)  sup_{h ∈ H, ỹ ∈ L²ₙ(P̃), ŷ ∈ L²ₙ(P̂)}  E^{P̂}[d₀ᵀŷ(ξ)]

s.t.  E^{P̃}[d₀ᵀỹ(ξ)] ≥ z*(x)                      (λ)
      Aỹ(ξ) + Bh ≤ b_x(ξ)         P̃-a.s.            (π̃(ξ))
      Aŷ(ξ) + Bh ≤ b_x(ξ)         P̂-a.s.            (π̂(ξ))
      Wh ≤ w                                          (ν)
```

has strong duality with its Lagrangian dual:

```
(D)  inf_{π̃ ∈ L²ₘ(P̃), π̂ ∈ L²ₘ(P̂), λ,ν≥0}

     E^{P̂}[b_x(ξ)ᵀπ̂(ξ)] + E^{P̃}[b_x(ξ)ᵀπ̃(ξ)] - λz*(x) + wν

s.t.  Aᵀπ̃(ξ) - d₀λ ≥ 0        P̃-a.s.
      Aᵀπ̂(ξ) ≥ d₀               P̂-a.s.
      E^{P̃}[Bᵀπ̃(ξ)] + E^{P̂}[Bᵀπ̂(ξ)] + Wᵀν ≥ 0
      π̃(ξ) ≥ 0 P̃-a.s.,  π̂(ξ) ≥ 0 P̂-a.s.
```

That is, val(P) = val(D), and the infimum in (D) is attained.

**Proof.** Three steps:

**Step 1 (Product space lifting).** Define P⊗ = P̃ ⊗ P̂ on Ξ × Ξ. Lift ȳ₁(ξ̃,ξ̂) := ỹ(ξ̃), ȳ₂(ξ̃,ξ̂) := ŷ(ξ̂). By Fubini–Tonelli, both are in L²ₙ(P⊗). Rewrite (P) on the single measure P⊗ as (P'), with implicit measurability constraints that ȳ₁ depends only on ξ̃ and ȳ₂ depends only on ξ̂.

**Step 2 (Measurability redundancy).** In (P'), ȳ₂ may depend on both (ξ̃,ξ̂), but the constraints and objective for ȳ₂ involve only ξ̂. By pointwise optimization, optimal ȳ₂* depends only on ξ̂. Same for ȳ₁. So relaxing measurability doesn't change the value → (P') is a standard stochastic LP on a single probability space.

**Step 3 (Strong duality).** Verify conditions of Shapiro et al. (2021) Theorem 7.62: (i) primal feasible (y=0 works), (ii) dual feasible (exists π̂₀ with Aᵀπ̂₀ ≥ d₀), (iii) bounded (compact Ξ, bounded flows). The dual (D') on the product space reduces to (D) by the same separability argument: optimal dual multipliers depend only on the relevant component of ξ. □


### 2.3 Single-Level Reformulation

For network interdiction, c = 0 and d₀ = d₁, so the value function constraint becomes redundant: for each ξ, the optimal flow ŷ(ξ) attains Q(h,x,ξ) exactly. The pessimistic structure over h remains — the follower chooses h under P̃ while the leader evaluates under P̂. Applying Proposition 1 and the variable transform ỹ ← λy, h ← λh:

```
V(x, P̂, P̃) = inf  Σ_k E^{P̂}[ξ_k(1-v_k x_k)φ̂_k(ξ)]
                    + Σ_k E^{P̃}[ξ_k(1-v_k x_k)φ̃_k(ξ)]
                    - E^{P̃}[ỹ_ts(ξ)] + wν

s.t.

  (C-L)  N_yᵀπ̂(ξ) + φ̂(ξ) ≥ 0,  N_tsᵀπ̂(ξ) ≥ 1,  π̂ free, φ̂ ≥ 0     P̂-a.s.

  (C-F)  N_yᵀπ̃(ξ) + φ̃(ξ) ≥ 0,  N_tsᵀπ̃(ξ) ≥ λ,  π̃ free, φ̃ ≥ 0     P̃-a.s.

  (C-P)  N_y ỹ(ξ) + N_ts ỹ_ts(ξ) = 0,  ỹ_k(ξ) ≤ h_k + λξ_k(1-v_k x_k) ∀k   P̃-a.s.

  (C-B)  1ᵀh ≤ λw,  λ,ν ≥ 0, h ≥ 0

  ┌──────────────────────────────────────────────────────────────┐
  │(CC)  E^{P̃}[φ̃_k(ξ)] + E^{P̂}[φ̂_k(ξ)] ≤ ν,  ∀k = 1,...,|A| │
  └──────────────────────────────────────────────────────────────┘
```

Constraint (CC) is the **only** constraint linking expectations under both P̂ and P̃.


## 3. DRO Formulation

### 3.1 Ambiguity Sets (TV balls)

```
D̂ := { q ∈ R₊^S : Σ_s q_s = 1,  ½‖q - q̂‖₁ ≤ ε̂ }
D̃ := { q ∈ R₊^S : Σ_s q_s = 1,  ½‖q - q̂‖₁ ≤ ε̃ }
```

### 3.2 True DRO

```
V*(x) := sup_{(P̂, P̃) ∈ D̂ × D̃} V(x, P̂, P̃)
```

### 3.3 Direct Route (Conservative Approximation)

Apply worst-case robustification to each sup_P **separately** → V^Dir(x) ≥ V*(x). Gap: different worst-case distributions for objective and each coupling constraint.


---

# Part II: Exact Reformulation via Lagrangian Decomposition

## 4. Lagrangian Relaxation of (CC)

### 4.1 Relaxation and Decomposition

Relax (CC) with multiplier α ∈ R₊^|A| (using α_k instead of μ_k to avoid collision with TV dual μ^L):

```
L(α) = g₁(x, α, P̂) + g₂(x, α, P̃, λ, h) + (w - Σ_k α_k)ν
```

where:
- g₁(x, α, P̂) := inf_{π̂,φ̂: (C-L)} Σ_k E^{P̂}[(ξ_k(1-v_k x_k) + α_k)φ̂_k(ξ)]
  - Independent of (λ, h, P̃)
- g₂(x, α, P̃, λ, h) := inf_{π̃,φ̃,ỹ: (C-F),(C-P)} {Σ_k E^{P̃}[(ξ_k(1-v_k x_k) + α_k)φ̃_k] - E^{P̃}[ỹ_ts]}
  - Depends on (λ, h) as parameters

Infimizing over ν ≥ 0: optimal ν* = 0 when Σ_k α_k ≤ w.

### 4.2 Strong Duality

By Slater's condition:

```
V(x, P̂, P̃) = sup_{α ≥ 0, 1ᵀα ≤ w}  inf_{λ,h: 1ᵀh ≤ λw}  [g₁(x,α,P̂) + g₂(x,α,P̃,λ,h)]
```


## 5. Interchange Chain

Starting from True DRO:

```
V*(x) = sup_{(P̂,P̃)} sup_α inf_{λ,h} inf_{rest} [g₁ + g₂]
```

**Step 1: Merge two sup's** (free):
```
= sup_{α, (P̂,P̃)}  inf_{λ,h}  inf_{rest}  [g₁ + g₂]
```

**Step 2: Cartesian separation.** g₁ ⊥ (P̃, λ, h) and D̂ × D̃ is product:
```
= sup_α  [ sup_{P̂} g₁(x,α,P̂)  +  inf_{λ,h} sup_{P̃} inf_{rest^F} g₂ ]
```

**Step 3: sup_{P̃} inf_{λ,h} interchange** (Sion).

R(λ,h,P̃) := inf_{rest^F} g₂. Sion conditions:
- R linear (concave) in P̃ for fixed (λ,h) ✓
- R convex in (λ,h) for fixed P̃ (LP value function) ✓
- D̃ compact convex ✓
- {(λ,h) : λ ∈ [0,λ^U], h ≥ 0, 1ᵀh ≤ λw} compact convex ✓

**Remark:** Extends Goyal et al. Theorem 1 to two-distribution setting. Alternatively, embed via product measure P̂ ⊗ P̃.

### Final Form

```
┌───────────────────────────────────────────────────────────────────────┐
│ V*(x) = sup_{α ≥ 0, 1ᵀα ≤ w}  inf_{λ,h,ψ⁰}                       │
│           [ Piece-L(α, x)  +  Piece-F'(α, x, λ, h, ψ⁰) ]          │
│                                                       (True-DRO-Exact)│
└───────────────────────────────────────────────────────────────────────┘
```


---

# Part III: TV Reformulation and Computation

## 6. TV Reformulation of Each Piece

### 6.1 Piece-L(α, x)

```
Piece-L := min  Σ_s q̂_s(σ_s^{L+} - σ_s^{L-}) + 2ε̂ μ^L + η^L

s.t.
(EL-1)  σ_s^{L+} - σ_s^{L-} + η^L ≥ Σ_k ξ̄_k^s(φ̂_k^s - v_k ψ̂_k^s) + Σ_k α_k φ̂_k^s    ∀s
(EL-2)  σ_s^{L+} + σ_s^{L-} ≤ μ^L                                                          ∀s
(EL-3)  [N_yᵀπ̂ˢ]_k + φ̂_k^s ≥ 0                                                     ∀k, s
(EL-4)  N_tsᵀπ̂ˢ ≥ 1                                                                   ∀s
(EL-5/6/7) McCormick ψ̂_k^s ≈ x_k φ̂_k^s

Signs: π̂ˢ free; φ̂, ψ̂, σ^{L±}, μ^L ≥ 0; η^L free.
```

**Difference from Direct route:** Only (EL-1) changed: +Σ_k α_k φ̂_k^s. Entire TV-ν leader block absent.


### 6.2 Piece-F'(α, x, λ, h, ψ⁰) — Parameterized

(λ, h, ψ⁰) as **parameters** from outer inf:

```
Piece-F' := min  Σ_s q̂_s(σ_s^{F+} - σ_s^{F-}) + 2ε̃ μ^F + η^F

s.t.
(EF-1)  σ_s^{F+} - σ_s^{F-} + η^F ≥ Σ_k ξ̄_k^s(φ̃_k^s - v_k ψ̃_k^s) + Σ_k α_k φ̃_k^s - ỹ_ts^s  ∀s
(EF-2)  σ_s^{F+} + σ_s^{F-} ≤ μ^F                                                              ∀s
(EF-3)  [N_yᵀπ̃ˢ]_k + φ̃_k^s ≥ 0                                                         ∀k, s
(EF-4)  N_tsᵀπ̃ˢ ≥ λ                                                                       ∀s
(EF-5)  N_y ỹˢ + N_ts ỹ_ts^s = 0                                                           ∀s
(EF-6)  ỹ_k^s ≤ h_k + (λ - v_k ψ_k⁰) ξ̄_k^s                                              ∀k, s
(MT1/2/3) McCormick ψ̃_k^s ≈ x_k φ̃_k^s

Signs: π̃ˢ free; φ̃, ψ̃, ỹ, ỹ_ts, σ^{F±}, μ^F ≥ 0; η^F free.
```

**No (EF-7), no (MP1–3)** — these belong to the outer inf_{λ,h,ψ⁰}.


## 7. Merging and Dualization

### 7.1 Merged Piece-F: Full Primal

Piece-L independent of (λ,h,ψ⁰) → inf pushes inside Piece-F'. The merged Piece-F absorbs (h,λ,ψ⁰) as primal variables, together with the outer constraints (EF-7) and (MP1–3). With x̄ fixed from OMP:

```
Piece-F(α, x̄) := min  Σ_s q̂_s(σ_s^{F+} - σ_s^{F-}) + 2ε̃ μ^F + η^F

Variables: (σ_s^{F±}, μ^F, η^F, π̃ˢ, φ̃_k^s, ψ̃_k^s, ỹ_k^s, ỹ_ts^s, h_k, λ, ψ_k⁰)

All constraints in ≥ 0 form for dualization:

TV envelope (duals d_s ≥ 0, e_s ≥ 0):
(1)  σ_s^{F+} - σ_s^{F-} + η^F - Σ_k(ξ̄_k^s + α_k)φ̃_k^s + Σ_k v_k ξ̄_k^s ψ̃_k^s + ỹ_ts^s ≥ 0   ∀s
(2)  μ^F - σ_s^{F+} - σ_s^{F-} ≥ 0                                                              ∀s

Follower dual feasibility (duals ũ_k^s ≥ 0, σ̃ˢ ≥ 0):
(3)  [N_yᵀπ̃ˢ]_k + φ̃_k^s ≥ 0                                                             ∀k, s
(4)  N_tsᵀπ̃ˢ - λ ≥ 0                                                                       ∀s

Follower primal feasibility (duals ωˢ FREE, β_k^s ≥ 0):
(5)  N_y ỹˢ + N_ts ỹ_ts^s = 0                                                               ∀s
(6)  -ỹ_k^s + h_k + λξ̄_k^s - v_k ψ_k⁰ ξ̄_k^s ≥ 0                                          ∀k, s

Budget (dual δ ≥ 0):
(7)  λw - Σ_k h_k ≥ 0

McCormick ψ̃_k^s ≈ x̄_k φ̃_k^s (duals ρ̃_k^{s,1}, ρ̃_k^{s,2}, ρ̃_k^{s,3} ≥ 0):
(8)   -ψ̃_k^s ≥ -φ^U x̄_k                    ∀k, s
(9)   φ̃_k^s - ψ̃_k^s ≥ 0                     ∀k, s
(10)  ψ̃_k^s - φ̃_k^s ≥ -φ^U(1 - x̄_k)        ∀k, s

McCormick ψ_k⁰ ≈ λ x̄_k (duals ρ_k^{0,1}, ρ_k^{0,2}, ρ_k^{0,3} ≥ 0):
(11)  -ψ_k⁰ ≥ -λ^U x̄_k                      ∀k
(12)  λ - ψ_k⁰ ≥ 0                            ∀k
(13)  ψ_k⁰ - λ ≥ -λ^U(1 - x̄_k)              ∀k

Signs: π̃ˢ FREE; φ̃, ψ̃, ỹ, ỹ_ts ≥ 0; σ^{F±}, μ^F ≥ 0; η^F free; h, λ, ψ⁰ ≥ 0.
```


### 7.2 Dualization: ISP-F(α, x̄)

**Dual objective** (only McCormick constraints have x̄-dependent RHS):

| Constraint | RHS | Dual var |
|---|---|---|
| (8) | -φ^U x̄_k | ρ̃_k^{s,1} |
| (10) | -φ^U(1-x̄_k) | ρ̃_k^{s,3} |
| (11) | -λ^U x̄_k | ρ_k^{0,1} |
| (13) | -λ^U(1-x̄_k) | ρ_k^{0,3} |

```
Z^F(α) = max  -φ^U Σ_{s,k} x̄_k ρ̃_k^{s,1} - φ^U Σ_{s,k}(1-x̄_k) ρ̃_k^{s,3}
              -λ^U Σ_k x̄_k ρ_k^{0,1} - λ^U Σ_k(1-x̄_k) ρ_k^{0,3}
```

**Dual constraints** (from each primal variable, Aᵀy ≤ c):

```
From σ_s^{F+} ≥ 0:   d_s - e_s ≤ q̂_s                                    ∀s         (DF-1)
From σ_s^{F-} ≥ 0:   d_s + e_s ≥ q̂_s                                    ∀s         (DF-2)
From μ^F ≥ 0:         Σ_s e_s ≤ 2ε̃                                                  (DF-3)
From η^F free:         Σ_s d_s = 1                                                    (DF-4)

From π̃ˢ FREE:         N_y ũ^s + N_ts σ̃ˢ = 0                             ∀s         (DF-5)

From φ̃_k^s ≥ 0:      -(ξ̄_k^s + α_k) d_s + ũ_k^s + ρ̃_k^{s,2} - ρ̃_k^{s,3} ≤ 0    ∀k,s  (DF-6)
From ψ̃_k^s ≥ 0:      v_k ξ̄_k^s d_s - ρ̃_k^{s,1} - ρ̃_k^{s,2} + ρ̃_k^{s,3} ≤ 0    ∀k,s  (DF-7)

From ỹ_k^s ≥ 0:      [N_yᵀωˢ]_k - β_k^s ≤ 0                            ∀k,s       (DF-8)
From ỹ_ts^s ≥ 0:     d_s + N_tsᵀωˢ ≤ 0                                  ∀s         (DF-9)

From h_k ≥ 0:         Σ_s β_k^s ≤ δ                                      ∀k         (DF-h)

From λ ≥ 0:           Σ_s σ̃ˢ ≥ Σ_{s,k} ξ̄_k^s β_k^s + wδ
                              + Σ_k ρ_k^{0,2} - Σ_k ρ_k^{0,3}                      (DF-λ)

From ψ_k⁰ ≥ 0:       v_k Σ_s ξ̄_k^s β_k^s + ρ_k^{0,1} + ρ_k^{0,2} ≥ ρ_k^{0,3}   ∀k   (DF-ψ)
```

**Dual variable signs:** d_s, e_s, ũ_k^s, σ̃ˢ, β_k^s, δ, ρ̃_k^{s,i}, ρ_k^{0,i} ≥ 0; ωˢ **free**.

**Remark (McCormick for ψ_k⁰).** Without McCormick, substituting ψ_k⁰ = λx̄_k puts x̄_k into the coefficient of λ in constraint (6), making (DF-λ) depend on x̄. McCormick keeps x̄_k in RHS only → dual objective only.


### 7.3 ISP-L(α, x̄): Leader Dual

Non-zero RHS: (EL-4) gives RHS = 1 (dual σ̂ˢ); McCormick (EL-5/7) give RHS = -φ^U x̄_k and -φ^U(1-x̄_k).

```
Z^L(α) = max  Σ_s σ̂ˢ - φ^U Σ_{s,k} x̄_k ρ̂_k^{s,1} - φ^U Σ_{s,k}(1-x̄_k) ρ̂_k^{s,3}

s.t.
(DL-1)  N_y û^s + N_ts σ̂ˢ = 0                                            ∀s
(DL-2)  -(ξ̄_k^s + α_k) a_s + û_k^s + ρ̂_k^{s,2} - ρ̂_k^{s,3} ≤ 0        ∀k, s
(DL-3)  v_k ξ̄_k^s a_s - ρ̂_k^{s,1} - ρ̂_k^{s,2} + ρ̂_k^{s,3} ≤ 0        ∀k, s
(DL-4)  a_s - b_s ≤ q̂_s                                                   ∀s
(DL-5)  a_s + b_s ≥ q̂_s                                                   ∀s
(DL-6)  Σ_s b_s ≤ 2ε̂
(DL-7)  Σ_s a_s = 1
(DL-8)  all ≥ 0
```


## 8. Non-Concavity in α

**Counterexample.** 2-scenario, 2-arc series network, q̂ = (0.5, 0.5), ε̂ = 0.3, x = 0, ξ̄¹ = (3,1), ξ̄² = (1,3):

```
Piece-L(α) = 1 + max(0.2α₁ + 0.8α₂, 0.8α₁ + 0.2α₂)    (convex, not concave)
```

At w = 2: f(2,0) = f(0,2) = 2.6 but f(1,1) = 2.0 < 2.6. Extreme-point enumeration fails.


## 9. Benders Decomposition

### 9.1 Variable Assignment

| Level | Variables | Problem |
|---|---|---|
| OMP | x ∈ X, t₀ | MILP |
| Subproblem | α, d^L ∈ F^L(α), d^F ∈ F^F(α) | Bilinear |

F^L(α) = ISP-L feasible region (DL-1 through DL-8). F^F(α) = ISP-F feasible region (DF-1 through DF-ψ). Both depend on α through (DL-2) and (DF-6).

### 9.2 Convexity of Z₀(x)

```
Z₀(x) = sup_{(α, d^L, d^F) ∈ Ω} [obj^L(x, d^L) + obj^F(x, d^F)]
```

where Ω := {(α, d^L, d^F) : α ≥ 0, 1ᵀα ≤ w, d^L ∈ F^L(α), d^F ∈ F^F(α)}, independent of x.

**Proposition.** Z₀(x) is convex in x (pointwise sup of affine functions). Ω is nonconvex (bilinear coupling α × a_s, α × d_s) but this doesn't affect convexity in x.

### 9.3 Cut

```
t₀ ≥ Z₀* + Σ_k π_{x_k}(x_k - x̄_k)

π_{x_k} = -φ^U Σ_s (ρ̂_k^{s,1*} + ρ̃_k^{s,1*})
         + φ^U Σ_s (ρ̂_k^{s,3*} + ρ̃_k^{s,3*})
         - λ^U ρ_k^{0,1*} + λ^U ρ_k^{0,3*}
```

**Validity:** Any feasible (α, d) ∈ Ω → valid cut ℓ(x) ≤ Z₀(x) for all x.

**Tightness:** If (α*, d*) globally optimal at x̄ → ℓ(x̄) = Z₀(x̄). Since dual obj is affine in x (not just convex approximation), no linearization error — nonlinearity is in α, internal to subproblem.

### 9.4 Generating Valid Cuts via Fixed-ᾱ Mini-Benders

Solving the bilinear subproblem to global optimality may be expensive. Any feasible α produces a valid cut:

1. **Obtain candidate ᾱ** by heuristic (extreme points, alternating opt, partial B&B).
2. **Fix ᾱ, solve mini-Benders.** Bilinear terms become constants → ISP-L(ᾱ, x̄) ∥ ISP-F(ᾱ, x̄) as **independent LPs**.
3. **Return valid cut** t₀ ≥ Z^L(ᾱ) + Z^F(ᾱ) + Σ_k π_{x_k}(x_k - x̄_k).

Cut is valid but not tight (Z^L(ᾱ) + Z^F(ᾱ) ≤ Z₀(x̄)). Slope π_{x_k} is exact for the given ᾱ.

**Practical use:** Interleave cheap valid cuts (heuristic ᾱ) with occasional expensive tight cuts (B&B). Valid-but-weak cuts still help prune OMP's binary search tree.

#### Strengthening via Magnanti–Wong Cuts

When dual subproblem is degenerate, MW selects the strongest cut among alternative optima.

**Procedure.** Choose core point x̂ ∈ ri(conv(X)), e.g., x̂_k = γ/|A|. After solving ISP-L(ᾱ, x̄) to get Z^{L*}, solve MW secondary problem:

```
MW-L:  max_{d^L ∈ F^L(ᾱ)}  obj^L(x̂, d^L)   s.t.  obj^L(x̄, d^L) = Z^{L*}
```

Same for MW-F. Each is original ISP + one equality constraint, objective at x̂ instead of x̄.

**Properties:**
- MW-L ∥ MW-F independent LPs (L/F decomposition preserved)
- Resulting cut is Pareto-optimal (nondominated)
- Cut "tilted" toward core point x̂ → tighter at x away from x̄
- Cost: one additional LP per piece with one extra equality constraint


## 10. Solving the Bilinear Subproblem

### 10.1 Structure

Bilinear terms in (DL-2) and (DF-6):
- (DL-2): -(ξ̄_k^s + α_k)·a_s  →  α_k · a_s
- (DF-6): -(ξ̄_k^s + α_k)·d_s  →  α_k · d_s

Key observations:
1. α only in constraint coefficients, **not** objective
2. For fixed α: ISP-L ∥ ISP-F independent LPs
3. a_s ∈ [max(0, q̂_s - 2ε̂), min(1, q̂_s + 2ε̂)], similarly d_s with ε̃
4. α_k ∈ [0, w], Σ_k α_k ≤ w
5. Bilinear = "capacity augmentation" × "scenario weight"

### 10.2 McCormick LP Relaxation (Upper Bound)

ζ_{ks}^L = α_k a_s with McCormick envelopes using bounds α_k ∈ [0,w], a_s ∈ [a_s^min, a_s^max]:

```
ζ ≥ a_s^min · α_k
ζ ≥ w · a_s + a_s^max · α_k - w · a_s^max
ζ ≤ a_s^max · α_k
ζ ≤ w · a_s + a_s^min · α_k - w · a_s^min
```

Replace α_k a_s → ζ_{ks}^L in (DL-2), α_k d_s → ζ_{ks}^F in (DF-6). Single LP → upper bound.

**Remark.** When ε̂ small → a_s^max - a_s^min ≈ 4ε̂ small → McCormick tight. At ε̂ = 0: a_s = q̂_s fixed, exact.

### 10.3 Lower Bounds

1. **Extreme points:** α = 0, we₁, ..., we_{|A|}. Total 2(|A|+1) LPs. Best = initial incumbent.
2. **Alternating optimization:** Fix α → LP; fix (a,d,rest) → LP in α (linear when a,d fixed). Both LP. Monotone convergence but to coordinate-wise optimum, not global.

### 10.4 Spatial Branch-and-Bound

Branch on α_k ∈ [0,w]. At each node [α_k^L, α_k^U]:
- **UB:** McCormick LP with tightened α_k bounds
- **LB:** Fix α at midpoint, solve ISP-L + ISP-F
- **Branch:** largest gap (α_k^U - α_k^L)(a_s^max - a_s^min)
- **Fathom:** McCormick UB ≤ best LB

|A| branching variables only. ISP-L ∥ ISP-F parallel at each node.

**Trade-off:** Fixed α → L/F decomposition. B&B over α couples them. Direct route: full decomposition but inexact. True-DRO: exact but bilinear.

### 10.5 Practical Algorithm

1. Extreme points → incumbent α⁰, initial LB
2. Alternating opt from α⁰ → α¹, improved LB
3. McCormick LP at root → initial UB
4. If UB - LB < ε_tol: terminate with α¹
5. Else: spatial B&B with α¹ as incumbent


---

# Part IV: Alternative Approaches

## 11. Pointwise Maximum Reformulation

### 11.1 Idea

At optimality, ν* = max_k [E^{P̂}[φ̂_k] + E^{P̃}[φ̃_k]]. Substitute:

```
wν → w · max_{k ∈ A} [E^{P̂}[φ̂_k] + E^{P̃}[φ̃_k]]
```

### 11.2 Epigraph Decomposition

τ ≥ g + w·max_k G_k for all P ⟺ τ ≥ g + w·G_k for all P, for all k.

For fixed k: F_k = g + wG_k is linear in P → Cartesian separation:
```
sup_P F_k = sup_{P̂} F_k^L + sup_{P̃} F_k^F
```
Standard single-distribution TV-DRO, **no bilinear terms** (w is constant).

### 11.3 Bound Ordering

**Proposition.** V*(x) ≤ V^PW(x) ≤ V^Dir(x).

### 11.4 Gap Analysis

**Source 1 (minimax):** max_k F_k(z,P) is convex in P → Sion fails for sup_P inf_z max_k F_k. Interchange inf_z max_k sup_P F_k is relaxation.

**Source 2 (per-k separation):** Different k's use different worst-case P_k*. True DRO uses one joint P.

Sharing z across k tightens but doesn't close gap (Source 1 persists).


---

## 12. Summary

```
V*(x)  ≤  V^PW(x)  ≤  V^Dir(x)
```

| Formulation | Exactness | Subproblem | Bilinear? | L/F Decomp |
|---|---|---|---|---|
| True DRO (V*) | Exact | Spatial B&B + LP | Yes (α_k a_s) | ✗ (B&B couples) |
| Pointwise-max (V^PW) | UB (tight) | LP (per-k TV) | No | ✓ |
| Direct route (V^Dir) | UB (loose) | LP | No | ✓ |

**Paper positioning:**
1. **Main contribution:** V^PW — LP tractable, L/F decomposition, tighter than Direct route
2. **Theoretical:** True-DRO-Exact via Lagrangian decomposition, bound ordering
3. **Numerical:** Compute V^PW − V* gap via B&B to empirically validate V^PW quality
