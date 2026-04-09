# TV Reformulation and Benders Decomposition: Derivation

## 1. Notation

- G = (V, A): directed network, N = [N_y | N_ts] ∈ R^{m×(|A|+1)}, m = |V|-1
- x ∈ X = {x ∈ {0,1}^|A| : 1ᵀx ≤ γ}: interdiction
- h ∈ R₊^|A|: recovery
- λ ≥ 0: coupling scalar
- ψ_k⁰ := λ x_k (McCormick, exact for binary x)
- v ∈ R₊^|A|: interdiction effectiveness
- ξ̄ˢ ∈ R₊^|A|, s ∈ [S]: capacity scenarios, q̂_s > 0 nominal probability

**Shorthand (used only in Sections 1–6):**
- c_k^s := ξ̄_k^s (1 - v_k x_k)
- r_k^s := h_k + (λ - v_k ψ_k⁰) ξ̄_k^s

From Section 7 onward, these shorthands are **not used**; all expressions are expanded in terms of (x, h, λ, ψ⁰, ξ̄ˢ).


## 2. TV Ambiguity Sets

Leader (P̂) and follower (P̃) each use separate TV balls:

```
Q̂ := { q ∈ R₊^S : Σ_s q_s = 1,  ½‖q - q̂‖₁ ≤ ε̂ }
Q̃ := { q ∈ R₊^S : Σ_s q_s = 1,  ½‖q - q̂‖₁ ≤ ε̃ }
```

ε̂, ε̃ ∈ [0,1]: robustness parameters for leader and follower respectively.


## 3. TV Dual of sup_q

For f ∈ Rˢ and radius ε, the constraint `sup_{q ∈ Q_ε} qᵀf ≤ τ` is equivalent to:

∃ σ_s⁺, σ_s⁻ ≥ 0, μ ≥ 0, η free such that:

```
(TV-D1)  Σ_s q̂_s(σ_s⁺ - σ_s⁻) + 2ε μ + η  ≤  τ
(TV-D2)  σ_s⁺ - σ_s⁻ + η  ≥  f_s              ∀s
(TV-D3)  σ_s⁺ + σ_s⁻  ≤  μ                     ∀s
```

### Derivation

Write q = q̂ + z, introduce auxiliary: -z_s ≤ q_s - q̂_s ≤ z_s, Σ z_s ≤ 2ε.
Primal LP in (q, z). Take LP dual with multipliers:
- (σ_s⁺) for q_s - z_s ≤ q̂_s
- (σ_s⁻) for -q_s - z_s ≤ -q̂_s
- (μ) for Σ z_s ≤ 2ε
- (η) for Σ q_s = 1

Dual objective: Σ_s q̂_s(σ_s⁺ - σ_s⁻) + 2ε μ + η.
Dual constraint from q_s: σ_s⁺ - σ_s⁻ + η ≥ f_s.
Dual constraint from z_s: -σ_s⁺ - σ_s⁻ + μ ≥ 0.


## 4. Pre-Dual Formulation (with sup_q)

```
min_{x ∈ X}  inf   t + wν

s.t.  sup_{q ∈ Q̂} Σ_s q_s g_s^L  +  sup_{q ∈ Q̃} Σ_s q_s g_s^F  ≤  t      (PD-1)

      N_yᵀ π̂ˢ + φ̂ˢ  ≥  0                    ∀s                               (PD-2)
      N_tsᵀ π̂ˢ  ≥  1                          ∀s                               (PD-3)
      N_yᵀ π̃ˢ + φ̃ˢ  ≥  0                    ∀s                               (PD-4)
      N_tsᵀ π̃ˢ  ≥  λ                          ∀s                               (PD-5)

      sup_{q ∈ Q̂} Σ_s q_s φ̂_k^s  +  sup_{q ∈ Q̃} Σ_s q_s φ̃_k^s  ≤  ν    ∀k  (PD-6)

      N_y ỹˢ + N_ts ỹ_ts^s  ≤  0              ∀s                               (PD-7)
      ỹ_k^s  ≤  h_k + (λ - v_k ψ_k⁰) ξ̄_k^s   ∀k, s                           (PD-8)
      1ᵀh  ≤  λw                                                                (PD-9)

      π̂ˢ, φ̂ˢ, π̃ˢ, φ̃ˢ, ỹˢ, ỹ_ts^s ≥ 0  ∀s;  λ, ν ≥ 0
```

where g_s^L := Σ_k ξ̄_k^s (1 - v_k x_k) φ̂_k^s, g_s^F := Σ_k ξ̄_k^s (1 - v_k x_k) φ̃_k^s - ỹ_ts^s.

### Linearization

Introduce ψ̂_k^s ≈ x_k φ̂_k^s, ψ̃_k^s ≈ x_k φ̃_k^s, ψ_k⁰ ≈ λ x_k with McCormick:

```
0 ≤ ψ̂_k^s ≤ φ^U x_k,   ψ̂_k^s ≤ φ̂_k^s,   ψ̂_k^s ≥ φ̂_k^s - φ^U(1-x_k)   ∀k,s   (MC-hat)
0 ≤ ψ̃_k^s ≤ φ^U x_k,   ψ̃_k^s ≤ φ̃_k^s,   ψ̃_k^s ≥ φ̃_k^s - φ^U(1-x_k)   ∀k,s   (MC-tilde)
0 ≤ ψ_k⁰ ≤ λ^U x_k,    ψ_k⁰ ≤ λ,          ψ_k⁰ ≥ λ - λ^U(1-x_k)          ∀k     (MC-psi0)
```

After linearization:
- g_s^L = Σ_k ξ̄_k^s (φ̂_k^s - v_k ψ̂_k^s)
- g_s^F = Σ_k ξ̄_k^s (φ̃_k^s - v_k ψ̃_k^s) - ỹ_ts^s


## 5. Full TV Reformulation

Apply TV dual (Section 3) to each sup_q. TV dual variables:

| sup_q term | Variables |
|---|---|
| (PD-1) leader, radius ε̂ | σ_s^{L+}, σ_s^{L-}, μ^L, η^L |
| (PD-1) follower, radius ε̃ | σ_s^{F+}, σ_s^{F-}, μ^F, η^F |
| (PD-6) leader per k, radius ε̂ | σ_{s,k}^{Lν+}, σ_{s,k}^{Lν-}, μ_k^{Lν}, η_k^{Lν} |
| (PD-6) follower per k, radius ε̃ | σ_{s,k}^{Fν+}, σ_{s,k}^{Fν-}, μ_k^{Fν}, η_k^{Fν} |

```
min_{x ∈ X}  inf   t + wν

s.t.

=== Epigraph (TV dualized) ===

(T1)  Σ_s q̂_s(σ_s^{L+} - σ_s^{L-}) + 2ε̂ μ^L + η^L
      + Σ_s q̂_s(σ_s^{F+} - σ_s^{F-}) + 2ε̃ μ^F + η^F  ≤  t

(T2)  σ_s^{L+} - σ_s^{L-} + η^L  ≥  Σ_k ξ̄_k^s(φ̂_k^s - v_k ψ̂_k^s)     ∀s
(T3)  σ_s^{L+} + σ_s^{L-}  ≤  μ^L                                         ∀s

(T4)  σ_s^{F+} - σ_s^{F-} + η^F  ≥  Σ_k ξ̄_k^s(φ̃_k^s - v_k ψ̃_k^s) - ỹ_ts^s   ∀s
(T5)  σ_s^{F+} + σ_s^{F-}  ≤  μ^F                                         ∀s


=== Leader dual feasibility ===

(T6)  [N_yᵀ π̂ˢ]_k + φ̂_k^s  ≥  0                        ∀k, s
(T7)  N_tsᵀ π̂ˢ  ≥  1                                     ∀s


=== Follower dual feasibility ===

(T8)  [N_yᵀ π̃ˢ]_k + φ̃_k^s  ≥  0                        ∀k, s
(T9)  N_tsᵀ π̃ˢ  ≥  λ                                     ∀s


=== ν coupling (TV dualized) ===

(T10) Σ_s q̂_s(σ_{s,k}^{Lν+} - σ_{s,k}^{Lν-}) + 2ε̂ μ_k^{Lν} + η_k^{Lν}
      + Σ_s q̂_s(σ_{s,k}^{Fν+} - σ_{s,k}^{Fν-}) + 2ε̃ μ_k^{Fν} + η_k^{Fν}  ≤  ν    ∀k

(T11) σ_{s,k}^{Lν+} - σ_{s,k}^{Lν-} + η_k^{Lν}  ≥  φ̂_k^s     ∀s, k
(T12) σ_{s,k}^{Lν+} + σ_{s,k}^{Lν-}  ≤  μ_k^{Lν}              ∀s, k

(T13) σ_{s,k}^{Fν+} - σ_{s,k}^{Fν-} + η_k^{Fν}  ≥  φ̃_k^s     ∀s, k
(T14) σ_{s,k}^{Fν+} + σ_{s,k}^{Fν-}  ≤  μ_k^{Fν}              ∀s, k


=== Follower primal feasibility ===

(T15) N_y ỹˢ + N_ts ỹ_ts^s  ≤  0                        ∀s
(T16) ỹ_k^s  ≤  h_k + (λ - v_k ψ_k⁰) ξ̄_k^s              ∀k, s

=== Budget ===

(T17) 1ᵀh  ≤  λw

=== Signs ===

(T18) all σ, μ ≥ 0;  η's free;  π̂, φ̂, π̃, φ̃, ỹ, ỹ_ts ≥ 0;  λ, ν ≥ 0
      + (MC-hat), (MC-tilde), (MC-psi0)
```

> **Note:** In (T2), (T4), x does not appear in any constraint coefficient. x appears only in the RHS of McCormick constraints (MC-hat)–(MC-psi0). This is the key structural property enabling globally valid Benders cuts.


## 6. OMP (Outer Master Problem)

```
min   t₀

s.t.  1ᵀh ≤ λw,   x ∈ X,   (MC-psi0)
      t₀ ≥ Z₀(x, h, λ, ψ⁰)    (Benders cuts)
```

OMP variables: t₀, x, h, λ, ψ⁰.

> **Note:** Budget constraint is 1ᵀh ≤ λw (not 1ᵀh ≤ w). Bilinear but linearizable via McCormick since λ ≤ λ^U.


## 7. OSP Primal: Z₀(x̄, h̄, λ̄, ψ̄⁰)

(x̄, h̄, λ̄, ψ̄⁰) fixed. **c_k^s, r_k^s shorthands are not used from here on.**

**OSP variables:**
- Existing: σ_s^{L±}, μ^L, η^L (free); σ_s^{F±}, μ^F, η^F (free); π̂ˢ, φ̂_k^s, π̃ˢ, φ̃_k^s, ỹˢ, ỹ_ts^s, ν; TV-ν variables.
- **New:** ψ̂_k^s ≥ 0 (≈ x̄_k φ̂_k^s), ψ̃_k^s ≥ 0 (≈ x̄_k φ̃_k^s) ∀k, s.

Each constraint is labeled with its dual variable in parentheses.

```
Z₀ = min  Σ_s q̂_s(σ_s^{L+} - σ_s^{L-}) + 2ε̂ μ^L + η^L
          + Σ_s q̂_s(σ_s^{F+} - σ_s^{F-}) + 2ε̃ μ^F + η^F + wν

s.t.

--- Leader TV obj (modified: ξ̄ instead of c) ---

(a_s)         σ_s^{L+} - σ_s^{L-} + η^L
              - Σ_k ξ̄_k^s φ̂_k^s + Σ_k v_k ξ̄_k^s ψ̂_k^s  ≥  0     ∀s     (P2)
(b_s)         μ^L - σ_s^{L+} - σ_s^{L-}  ≥  0                      ∀s     (P3)

--- Follower TV obj (modified) ---

(d_s)         σ_s^{F+} - σ_s^{F-} + η^F
              - Σ_k ξ̄_k^s φ̃_k^s + Σ_k v_k ξ̄_k^s ψ̃_k^s
              + ỹ_ts^s  ≥  0                                        ∀s     (P4)
(e_s)         μ^F - σ_s^{F+} - σ_s^{F-}  ≥  0                      ∀s     (P5)

--- Leader dual feas (unchanged) ---

(û_k^s)       [N_yᵀ π̂ˢ]_k + φ̂_k^s  ≥  0                  ∀k, s          (P6)
(σ̂ˢ)          N_tsᵀ π̂ˢ  ≥  1                                ∀s            (P7)

--- Follower dual feas (unchanged) ---

(ũ_k^s)       [N_yᵀ π̃ˢ]_k + φ̃_k^s  ≥  0                  ∀k, s          (P8)
(σ̃ˢ)          N_tsᵀ π̃ˢ  ≥  λ̄                               ∀s            (P9)

--- ν coupling (unchanged) ---

(α_k)         ν - Σ_s q̂_s(σ_{s,k}^{Lν+} - σ_{s,k}^{Lν-}) - 2ε̂ μ_k^{Lν} - η_k^{Lν}
                - Σ_s q̂_s(σ_{s,k}^{Fν+} - σ_{s,k}^{Fν-}) - 2ε̃ μ_k^{Fν} - η_k^{Fν}
              ≥  0                                            ∀k            (P10)

--- Leader TV-ν (unchanged) ---

(a_{s,k}^ν)   σ_{s,k}^{Lν+} - σ_{s,k}^{Lν-} + η_k^{Lν} - φ̂_k^s  ≥  0  ∀s,k  (P11)
(b_{s,k}^ν)   μ_k^{Lν} - σ_{s,k}^{Lν+} - σ_{s,k}^{Lν-}  ≥  0          ∀s,k  (P12)

--- Follower TV-ν (unchanged) ---

(d_{s,k}^ν)   σ_{s,k}^{Fν+} - σ_{s,k}^{Fν-} + η_k^{Fν} - φ̃_k^s  ≥  0  ∀s,k  (P13)
(e_{s,k}^ν)   μ_k^{Fν} - σ_{s,k}^{Fν+} - σ_{s,k}^{Fν-}  ≥  0          ∀s,k  (P14)

--- Follower primal feas (EXPANDED — no r_k^s) ---

(ωˢ)          -N_y ỹˢ - N_ts ỹ_ts^s  ≥  0                  ∀s            (P15)
(β_k^s)       -ỹ_k^s  ≥  -h̄_k - (λ̄ - v_k ψ̄_k⁰) ξ̄_k^s    ∀k, s        (P16)

--- McCormick for ψ̂_k^s ≈ x_k φ̂_k^s (NEW) ---

(ρ̂_k^{s,1})   -ψ̂_k^s  ≥  -φ^U x̄_k                        ∀k, s        (MH1)
(ρ̂_k^{s,2})   φ̂_k^s - ψ̂_k^s  ≥  0                         ∀k, s        (MH2)
(ρ̂_k^{s,3})   ψ̂_k^s - φ̂_k^s  ≥  -φ^U(1 - x̄_k)            ∀k, s        (MH3)

--- McCormick for ψ̃_k^s ≈ x_k φ̃_k^s (NEW) ---

(ρ̃_k^{s,1})   -ψ̃_k^s  ≥  -φ^U x̄_k                        ∀k, s        (MT1)
(ρ̃_k^{s,2})   φ̃_k^s - ψ̃_k^s  ≥  0                         ∀k, s        (MT2)
(ρ̃_k^{s,3})   ψ̃_k^s - φ̃_k^s  ≥  -φ^U(1 - x̄_k)            ∀k, s        (MT3)
```

> **OMP variable dependence in RHS:**
>
> | Constraint | RHS | OMP variables |
> |---|---|---|
> | (P9) | λ̄ | λ |
> | (P16) | -h̄_k - (λ̄ - v_k ψ̄_k⁰) ξ̄_k^s | h_k, λ, ψ_k⁰ |
> | (MH1), (MT1) | -φ^U x̄_k | x_k |
> | (MH3), (MT3) | -φ^U(1 - x̄_k) | x_k |
>
> No OMP variable appears in any constraint coefficient.


## 8. Dual of OSP

### 8.1 Objective

Non-zero RHS contributions (dual variable × RHS):
- (P7): RHS = 1 → +Σ_s σ̂ˢ
- (P9): RHS = λ̄ → +λ̄ Σ_s σ̃ˢ
- (P16): RHS = -h̄_k - (λ̄ - v_k ψ̄_k⁰) ξ̄_k^s → -Σ_{s,k} [h̄_k + (λ̄ - v_k ψ̄_k⁰) ξ̄_k^s] β_k^s
- (MH1): RHS = -φ^U x̄_k → -φ^U Σ_{s,k} x̄_k ρ̂_k^{s,1}
- (MH3): RHS = -φ^U(1-x̄_k) → -φ^U Σ_{s,k} (1-x̄_k) ρ̂_k^{s,3}
- (MT1): RHS = -φ^U x̄_k → -φ^U Σ_{s,k} x̄_k ρ̃_k^{s,1}
- (MT3): RHS = -φ^U(1-x̄_k) → -φ^U Σ_{s,k} (1-x̄_k) ρ̃_k^{s,3}

```
max  Σ_s σ̂ˢ  +  λ̄ Σ_s σ̃ˢ
     − Σ_{s,k} h̄_k β_k^s
     − λ̄ Σ_{s,k} ξ̄_k^s β_k^s
     + Σ_{s,k} v_k ψ̄_k⁰ ξ̄_k^s β_k^s
     − φ^U Σ_{s,k} x̄_k (ρ̂_k^{s,1} + ρ̃_k^{s,1})
     − φ^U Σ_{s,k} (1 − x̄_k)(ρ̂_k^{s,3} + ρ̃_k^{s,3})
```

> Each OMP variable's contribution is directly readable from the objective.


### 8.2 Dual Constraints

Right side "(p: ·)" indicates the primal variable. Free → "="; ≥0 → "≤".

#### Complicating:
```
Σ_k α_k  ≤  w                                          (p: ν ≥ 0)       (D-nu)
```

#### Leader per s:
```
N_y û^s + N_ts σ̂ˢ  ≤  0                    ∀s          (p: π̂ˢ ≥ 0)      (D-pihat)
```

φ̂_k^s appears in (P2) (-ξ̄_k^s), (P6) (+1), (P11) (-1), (MH2) (+1), (MH3) (-1):
```
-ξ̄_k^s a_s + û_k^s - a_{s,k}^ν
  + ρ̂_k^{s,2} - ρ̂_k^{s,3}  ≤  0            ∀k, s       (p: φ̂_k^s ≥ 0)   (D-phihat)
```

ψ̂_k^s appears in (P2) (+v_k ξ̄_k^s), (MH1) (-1), (MH2) (-1), (MH3) (+1):
```
v_k ξ̄_k^s a_s - ρ̂_k^{s,1}
  - ρ̂_k^{s,2} + ρ̂_k^{s,3}  ≤  0            ∀k, s       (p: ψ̂_k^s ≥ 0)   (D-psihat)
```

#### Follower per s:
```
N_y ũ^s + N_ts σ̃ˢ  ≤  0                    ∀s          (p: π̃ˢ ≥ 0)      (D-pitilde)
```

φ̃_k^s appears in (P4) (-ξ̄_k^s), (P8) (+1), (P13) (-1), (MT2) (+1), (MT3) (-1):
```
-ξ̄_k^s d_s + ũ_k^s - d_{s,k}^ν
  + ρ̃_k^{s,2} - ρ̃_k^{s,3}  ≤  0            ∀k, s       (p: φ̃_k^s ≥ 0)   (D-phitilde)
```

ψ̃_k^s appears in (P4) (+v_k ξ̄_k^s), (MT1) (-1), (MT2) (-1), (MT3) (+1):
```
v_k ξ̄_k^s d_s - ρ̃_k^{s,1}
  - ρ̃_k^{s,2} + ρ̃_k^{s,3}  ≤  0            ∀k, s       (p: ψ̃_k^s ≥ 0)   (D-psitilde)
```

Follower flow (unchanged):
```
[N_yᵀ ωˢ]_k + β_k^s  ≥  0                  ∀k,s       (p: ỹ_k^s ≥ 0)   (D-ytilde)
N_tsᵀ ωˢ  ≥  d_s                            ∀s          (p: ỹ_ts^s ≥ 0)  (D-yts)
```

#### Leader TV duals (radius ε̂):
```
a_s - b_s  ≤  q̂_s                           ∀s          (p: σ_s^{L+} ≥ 0)  (D-sigLp)
a_s + b_s  ≥  q̂_s                           ∀s          (p: σ_s^{L-} ≥ 0)  (D-sigLm)
Σ_s b_s  ≤  2ε̂                                          (p: μ^L ≥ 0)       (D-muL)
Σ_s a_s  =  1                                            (p: η^L free)      (D-etaL)
```

#### Follower TV duals (radius ε̃):
```
d_s - e_s  ≤  q̂_s                           ∀s          (p: σ_s^{F+} ≥ 0)  (D-sigFp)
d_s + e_s  ≥  q̂_s                           ∀s          (p: σ_s^{F-} ≥ 0)  (D-sigFm)
Σ_s e_s  ≤  2ε̃                                          (p: μ^F ≥ 0)       (D-muF)
Σ_s d_s  =  1                                            (p: η^F free)      (D-etaF)
```

#### Leader TV-ν duals (radius ε̂), ∀k:
```
a_{s,k}^ν - b_{s,k}^ν  ≤  q̂_s α_k          ∀s,k       (p: σ_{s,k}^{Lν+} ≥ 0)  (D-sigLnup)
a_{s,k}^ν + b_{s,k}^ν  ≥  q̂_s α_k          ∀s,k       (p: σ_{s,k}^{Lν-} ≥ 0)  (D-sigLnum)
Σ_s b_{s,k}^ν  ≤  2ε̂ α_k                    ∀k         (p: μ_k^{Lν} ≥ 0)        (D-mukLnu)
Σ_s a_{s,k}^ν  =  α_k                        ∀k         (p: η_k^{Lν} free)       (D-etakLnu)
```

#### Follower TV-ν duals (radius ε̃), ∀k:
```
d_{s,k}^ν - e_{s,k}^ν  ≤  q̂_s α_k          ∀s,k       (p: σ_{s,k}^{Fν+} ≥ 0)  (D-sigFnup)
d_{s,k}^ν + e_{s,k}^ν  ≥  q̂_s α_k          ∀s,k       (p: σ_{s,k}^{Fν-} ≥ 0)  (D-sigFnum)
Σ_s e_{s,k}^ν  ≤  2ε̃ α_k                    ∀k         (p: μ_k^{Fν} ≥ 0)        (D-mukFnu)
Σ_s d_{s,k}^ν  =  α_k                        ∀k         (p: η_k^{Fν} free)       (D-etakFnu)
```

#### Signs:
All dual variables ≥ 0: α_k, û_k^s, σ̂ˢ, ũ_k^s, σ̃ˢ, ωˢ, β_k^s, a_s, b_s, d_s, e_s, a_{s,k}^ν, b_{s,k}^ν, d_{s,k}^ν, e_{s,k}^ν, ρ̂_k^{s,1}, ρ̂_k^{s,2}, ρ̂_k^{s,3}, ρ̃_k^{s,1}, ρ̃_k^{s,2}, ρ̃_k^{s,3}.

> No OMP variable (x̄, h̄, λ̄, ψ̄⁰) appears in dual constraints (D-pihat)–(D-etakFnu). Hence the dual feasible region is completely independent of all OMP variables.


## 9. Decomposition

### 9.1 Coupling analysis

Fixing α_k yields complete leader/follower separation:
- **Leader variables:** σ̂ˢ, û_k^s, a_s, b_s, a_{s,k}^ν, b_{s,k}^ν, ρ̂_k^{s,1}, ρ̂_k^{s,2}, ρ̂_k^{s,3}
- **Follower variables:** σ̃ˢ, ũ_k^s, ωˢ, β_k^s, d_s, e_s, d_{s,k}^ν, e_{s,k}^ν, ρ̃_k^{s,1}, ρ̃_k^{s,2}, ρ̃_k^{s,3}

McCormick duals ρ̂ appear only in (D-phihat), (D-psihat); ρ̃ appear only in (D-phitilde), (D-psitilde) → separation preserved.


### 9.2 Inner Master Problem (IMP)

```
max  θ^L + θ^F

s.t. Σ_k α_k ≤ w,  α_k ≥ 0  ∀k
     θ^L ≤ (leader Benders cuts)
     θ^F ≤ (follower Benders cuts)
```

IMP variables: α_k, θ^L, θ^F. Size O(|A|).
Inner cut structure unchanged: α_k appears only in TV-ν RHS, not in McCormick constraints.


### 9.3 Leader Subproblem ISP-L(α) — LP

Given α_k, x̄_k (from OMP).

```
Z^L = max  Σ_s σ̂ˢ
           − φ^U Σ_{s,k} x̄_k ρ̂_k^{s,1}
           − φ^U Σ_{s,k} (1 − x̄_k) ρ̂_k^{s,3}

s.t. N_y û^s + N_ts σ̂ˢ  ≤  0                          ∀s           (L1)
     -ξ̄_k^s a_s + û_k^s - a_{s,k}^ν
       + ρ̂_k^{s,2} - ρ̂_k^{s,3}  ≤  0                   ∀k, s       (L2)
     v_k ξ̄_k^s a_s - ρ̂_k^{s,1}
       - ρ̂_k^{s,2} + ρ̂_k^{s,3}  ≤  0                   ∀k, s       (L3)

     a_s - b_s  ≤  q̂_s                                  ∀s           (L4)
     a_s + b_s  ≥  q̂_s                                  ∀s           (L5)
     Σ_s b_s  ≤  2ε̂                                                  (L6)
     Σ_s a_s  =  1                                                    (L7)

     a_{s,k}^ν - b_{s,k}^ν  ≤  q̂_s α_k                  ∀s, k       (L8)
     a_{s,k}^ν + b_{s,k}^ν  ≥  q̂_s α_k                  ∀s, k       (L9)
     Σ_s b_{s,k}^ν  ≤  2ε̂ α_k                            ∀k          (L10)
     Σ_s a_{s,k}^ν  =  α_k                                ∀k          (L11)

     σ̂ˢ, û_k^s, a_s, b_s, a_{s,k}^ν, b_{s,k}^ν,
     ρ̂_k^{s,1}, ρ̂_k^{s,2}, ρ̂_k^{s,3}  ≥  0                          (L12)
```

> No OMP variable in constraints (L1)–(L11). x̄_k appears only in the objective.


### 9.4 Follower Subproblem ISP-F(α) — LP

Given α_k, x̄_k, h̄_k, λ̄, ψ̄_k⁰ (from OMP).

```
Z^F = max  λ̄ Σ_s σ̃ˢ
           − Σ_{s,k} [h̄_k + (λ̄ − v_k ψ̄_k⁰) ξ̄_k^s] β_k^s
           − φ^U Σ_{s,k} x̄_k ρ̃_k^{s,1}
           − φ^U Σ_{s,k} (1 − x̄_k) ρ̃_k^{s,3}

s.t. N_y ũ^s + N_ts σ̃ˢ  ≤  0                          ∀s           (F1)
     -ξ̄_k^s d_s + ũ_k^s - d_{s,k}^ν
       + ρ̃_k^{s,2} - ρ̃_k^{s,3}  ≤  0                   ∀k, s       (F2)
     v_k ξ̄_k^s d_s - ρ̃_k^{s,1}
       - ρ̃_k^{s,2} + ρ̃_k^{s,3}  ≤  0                   ∀k, s       (F3)
     [N_yᵀ ωˢ]_k + β_k^s  ≥  0                          ∀k, s       (F4)
     N_tsᵀ ωˢ  ≥  d_s                                    ∀s           (F5)

     d_s - e_s  ≤  q̂_s                                  ∀s           (F6)
     d_s + e_s  ≥  q̂_s                                  ∀s           (F7)
     Σ_s e_s  ≤  2ε̃                                                  (F8)
     Σ_s d_s  =  1                                                    (F9)

     d_{s,k}^ν - e_{s,k}^ν  ≤  q̂_s α_k                  ∀s, k       (F10)
     d_{s,k}^ν + e_{s,k}^ν  ≥  q̂_s α_k                  ∀s, k       (F11)
     Σ_s e_{s,k}^ν  ≤  2ε̃ α_k                            ∀k          (F12)
     Σ_s d_{s,k}^ν  =  α_k                                ∀k          (F13)

     σ̃ˢ, ũ_k^s, ωˢ, β_k^s, d_s, e_s, d_{s,k}^ν, e_{s,k}^ν,
     ρ̃_k^{s,1}, ρ̃_k^{s,2}, ρ̃_k^{s,3}  ≥  0                          (F14)
```

> No OMP variable in constraints (F1)–(F13). All OMP variables (x̄_k, h̄_k, λ̄, ψ̄_k⁰) appear only in the objective.


## 10. OMP Benders Cut

### 10.1 Sensitivities

Z₀* = Z^{L*} + Z^{F*} after inner loop converges. Sensitivities are read directly by differentiating the dual objective (Section 8.1) w.r.t. each OMP variable:

```
π_{x_k}  = −φ^U Σ_s (ρ̂_k^{s,1*} + ρ̃_k^{s,1*})
           + φ^U Σ_s (ρ̂_k^{s,3*} + ρ̃_k^{s,3*})           (from ISP-L + ISP-F)

π_{h_k}  = −Σ_s β_k^{s*}                                    (from ISP-F)

π_λ      = Σ_s σ̃^{s*} − Σ_{s,k} ξ̄_k^s β_k^{s*}            (from ISP-F)

π_{ψ_k⁰} = +v_k Σ_s ξ̄_k^s β_k^{s*}                        (from ISP-F)
```

> **Derivation from dual objective:**
> - Coefficient of h̄_k: −Σ_s β_k^s → π_{h_k}
> - Coefficient of λ̄: +Σ_s σ̃ˢ − Σ_{s,k} ξ̄_k^s β_k^s → π_λ
> - Coefficient of ψ̄_k⁰: +v_k Σ_s ξ̄_k^s β_k^s → π_{ψ_k⁰}
> - Coefficient of x̄_k: −φ^U Σ_s (ρ̂^1 + ρ̃^1) + φ^U Σ_s (ρ̂^3 + ρ̃^3) → π_{x_k}

### 10.2 Cut

```
t₀ ≥ Z₀* + Σ_k π_{x_k}(x_k − x̄_k) + Σ_k π_{h_k}(h_k − h̄_k)
           + π_λ(λ − λ̄) + Σ_k π_{ψ_k⁰}(ψ_k⁰ − ψ̄_k⁰)
```

This cut is **globally valid** for all (x, h, λ, ψ⁰): the dual feasible region (D-pihat)–(D-etakFnu) is independent of OMP variables, so any cut derived from a dual feasible point is valid for all OMP variable values.

### 10.3 Sensitivity sources

| Sensitivity | ISP-L | ISP-F |
|---|---|---|
| π_{x_k} | ρ̂_k^{s,1*}, ρ̂_k^{s,3*} | ρ̃_k^{s,1*}, ρ̃_k^{s,3*} |
| π_{h_k} | — | β_k^{s*} |
| π_λ | — | σ̃^{s*}, β_k^{s*} |
| π_{ψ_k⁰} | — | β_k^{s*} |

Z₀* = Z^{L*} + Z^{F*}: constant term requires **both** ISP-L and ISP-F.


## 11. Summary

```
OMP (MILP)          →   IMP (LP, α_k)        →   Leader LP (all s, radius ε̂)
binary x, h, λ, ψ⁰      O(|A|) vars               + Follower LP (all s, radius ε̃)
```

| | Wasserstein | TV |
|---|---|---|
| OMP | MILP | MILP |
| IMP | LP, O(\|A\|) | LP, O(\|A\|) |
| Leader/Follower Sub | **SDP** | **LP** |

| | Before (incorrect) | After (corrected) |
|---|---|---|
| x in OSP | coefficient (c_k^s) | RHS only (McCormick) |
| h, λ, ψ⁰ in OSP | hidden in r_k^s | expanded in (P16) |
| ISP-L/F extra vars | none | ρ̂, ρ̃: +3\|A\|S each |
| π_{x_k} | undefined | from ρ̂, ρ̃ |
| π_λ | +Σ c_k^s β_k^s (wrong) | −Σ ξ̄_k^s β_k^s |
| π_{ψ_k⁰} | undefined | +v_k Σ ξ̄_k^s β_k^s |
| OMP budget | 1ᵀh ≤ w | 1ᵀh ≤ λw |
| Benders cut | local (only at x̄) | **globally valid** |
