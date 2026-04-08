# TV Reformulation and Benders Decomposition: Derivation

## 1. Notation

- G = (V, A): directed network, N = [N_y | N_ts] ∈ R^{m×(|A|+1)}, m = |V|-1
- x ∈ X = {x ∈ {0,1}^|A| : 1ᵀx ≤ γ}: interdiction
- h ∈ R₊^|A|: recovery, 1ᵀh ≤ w
- λ ≥ 0: coupling scalar
- ψ_k⁰ := λ x_k (McCormick, exact for binary x)
- v ∈ R₊^|A|: interdiction effectiveness
- ξ̄ˢ ∈ R₊^|A|, s ∈ [S]: capacity scenarios, q̂_s > 0 nominal probability
- c_k^s := ξ̄_k^s (1 - v_k x_k)
- r_k^s := h_k + λ c_k^s (for fixed x, h, λ)


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

Replacing Wasserstein ball in manuscript (6) by Q_TV. All functions evaluated at ξ̄ˢ:

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

where g_s^L := Σ_k c_k^s φ̂_k^s,  g_s^F := Σ_k c_k^s φ̃_k^s - ỹ_ts^s.


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

(T2)  σ_s^{L+} - σ_s^{L-} + η^L  ≥  g_s^L              ∀s
(T3)  σ_s^{L+} + σ_s^{L-}  ≤  μ^L                       ∀s

(T4)  σ_s^{F+} - σ_s^{F-} + η^F  ≥  g_s^F               ∀s
(T5)  σ_s^{F+} + σ_s^{F-}  ≤  μ^F                        ∀s


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
```


## 6. OMP (Outer Master Problem)

```
min   t₀

s.t.  1ᵀh ≤ λw,   x ∈ X,   McCormick(ψ⁰ = λx)
      t₀ ≥ (optimality cuts)
```

OMP variables: t₀, x, h, λ, ψ⁰.

### 6.1 Outer Optimality Cut

At the current solution (x̄, h̄, λ̄, ψ̄⁰), solve the inner loop to get Z₀*.
The cut is:

```
t₀ ≥ intercept + π_h'h + π_λ λ + π_{ψ⁰}'ψ⁰ + π_x'x
```

where:

```
π_{h_k}   = -Σ_s β_k^s                                         (P16 RHS: r_k^s = h_k + ...)
π_λ       = Σ_s σ̃^s - Σ_{s,k} ξ̄_k^s β_k^s                     (P9 RHS: λ̄, P16 RHS: r_k^s)
π_{ψ⁰_k}  = v_k Σ_s ξ̄_k^s β_k^s                                (P16 RHS: r_k^s = ... - v_k ψ⁰_k ξ_k^s)
π_{x_k}   = -v_k Σ_s ξ̄_k^s (a_s φ̂_k^s + d_s φ̃_k^s)           (P2/P4 coeff: c_k^s = ξ̄_k^s(1-v_k x_k))

intercept = Z₀* - π_h'h̄ - π_λ λ̄ - π_{ψ⁰}'ψ̄⁰ - π_x'x̄
```

Sources:
- β_k^s, σ̃^s: ISP-F primal values
- a_s, φ̂_k^s: ISP-L primal value(a[s]), shadow_price(L2[k,s])
- d_s, φ̃_k^s: ISP-F primal value(d[s]), shadow_price(F2[k,s])

**Note on π_x**: x appears in c_k^s = ξ̄_k^s(1 - v_k x_k), which enters as coefficient
in (P2) and (P4). The sensitivity ∂Z₀/∂x_k = -v_k Σ_s ξ̄_k^s (a_s φ̂_k^s + d_s φ̃_k^s).
Without π_x, OMP has no information that interdiction reduces adversary flow → x=0 고착.


## 7. OSP Primal: Z₀(x̄, h̄, λ̄, ψ̄⁰)

(x̄, h̄, λ̄, ψ̄⁰) fixed ⇒ c_k^s, r_k^s are constants.
Variable signs: η^L, η^F, η_k^{Lν}, η_k^{Fν} free; all others ≥ 0.

Each constraint is labeled with its dual variable in parentheses.

```
Z₀ = min  Σ_s q̂_s(σ_s^{L+} - σ_s^{L-}) + 2ε̂ μ^L + η^L
          + Σ_s q̂_s(σ_s^{F+} - σ_s^{F-}) + 2ε̃ μ^F + η^F + wν

s.t.

--- Leader TV obj ---

(a_s)         σ_s^{L+} - σ_s^{L-} + η^L - Σ_k c_k^s φ̂_k^s  ≥  0       ∀s     (P2)
(b_s)         μ^L - σ_s^{L+} - σ_s^{L-}  ≥  0                          ∀s     (P3)

--- Follower TV obj ---

(d_s)         σ_s^{F+} - σ_s^{F-} + η^F - Σ_k c_k^s φ̃_k^s + ỹ_ts^s  ≥  0  ∀s  (P4)
(e_s)         μ^F - σ_s^{F+} - σ_s^{F-}  ≥  0                          ∀s     (P5)

--- Leader dual feas ---

(û_k^s)       [N_yᵀ π̂ˢ]_k + φ̂_k^s  ≥  0                    ∀k, s            (P6)
(σ̂ˢ)          N_tsᵀ π̂ˢ  ≥  1                                  ∀s              (P7)

--- Follower dual feas ---

(ũ_k^s)       [N_yᵀ π̃ˢ]_k + φ̃_k^s  ≥  0                    ∀k, s            (P8)
(σ̃ˢ)          N_tsᵀ π̃ˢ  ≥  λ̄                                 ∀s              (P9)

--- ν coupling ---

(α_k)         ν - Σ_s q̂_s(σ_{s,k}^{Lν+} - σ_{s,k}^{Lν-}) - 2ε̂ μ_k^{Lν} - η_k^{Lν}
                - Σ_s q̂_s(σ_{s,k}^{Fν+} - σ_{s,k}^{Fν-}) - 2ε̃ μ_k^{Fν} - η_k^{Fν}
              ≥  0                                              ∀k              (P10)

--- Leader TV-ν ---

(a_{s,k}^ν)   σ_{s,k}^{Lν+} - σ_{s,k}^{Lν-} + η_k^{Lν} - φ̂_k^s  ≥  0  ∀s,k  (P11)
(b_{s,k}^ν)   μ_k^{Lν} - σ_{s,k}^{Lν+} - σ_{s,k}^{Lν-}  ≥  0          ∀s,k  (P12)

--- Follower TV-ν ---

(d_{s,k}^ν)   σ_{s,k}^{Fν+} - σ_{s,k}^{Fν-} + η_k^{Fν} - φ̃_k^s  ≥  0  ∀s,k  (P13)
(e_{s,k}^ν)   μ_k^{Fν} - σ_{s,k}^{Fν+} - σ_{s,k}^{Fν-}  ≥  0          ∀s,k  (P14)

--- Follower primal feas ---

(ωˢ)          -N_y ỹˢ - N_ts ỹ_ts^s  ≥  0                    ∀s              (P15)
(β_k^s)       r_k^s - ỹ_k^s  ≥  0                             ∀k, s           (P16)
```


## 8. Dual of OSP

### 8.1 Objective

RHS ≠ 0: (P7) = 1, (P9) = λ̄, (P16) = -r_k^s.

```
max  Σ_s σ̂ˢ  +  λ̄ Σ_s σ̃ˢ  -  Σ_s Σ_k r_k^s β_k^s
```


### 8.2 Dual Constraints

Right side "(p: ·)" indicates the primal variable. Free → "="; ≥0 → "≤".

#### Complicating:
```
Σ_k α_k  ≤  w                                          (p: ν ≥ 0)       (D-nu)
```

#### Leader per s:
```
N_y û^s + N_ts σ̂ˢ  ≤  0                    ∀s          (p: π̂ˢ ≥ 0)      (D-pihat)
-c_k^s a_s + û_k^s - a_{s,k}^ν  ≤  0       ∀k,s       (p: φ̂_k^s ≥ 0)   (D-phihat)
```

#### Follower per s:
```
N_y ũ^s + N_ts σ̃ˢ  ≤  0                    ∀s          (p: π̃ˢ ≥ 0)      (D-pitilde)
-c_k^s d_s + ũ_k^s - d_{s,k}^ν  ≤  0       ∀k,s       (p: φ̃_k^s ≥ 0)   (D-phitilde)
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
All dual variables ≥ 0: α_k, û_k^s, σ̂ˢ, ũ_k^s, σ̃ˢ, ωˢ, β_k^s, a_s, b_s, d_s, e_s, a_{s,k}^ν, b_{s,k}^ν, d_{s,k}^ν, e_{s,k}^ν.


## 9. Decomposition

### 9.1 Coupling analysis

α_k를 고정하면 leader/follower로 완전 분리:
- Leader TV duals (a_s, b_s, a_{s,k}^ν, b_{s,k}^ν)는 (D-phihat), (D-sigLp)–(D-etakLnu)에만 등장
- Follower TV duals (d_s, e_s, d_{s,k}^ν, e_{s,k}^ν)는 (D-phitilde)–(D-etakFnu)에만 등장
- Σ_s linking constraints는 각 leader/follower 내부에서 처리


### 9.2 Inner Master Problem (IMP)

```
max  θ^L + θ^F

s.t. Σ_k α_k ≤ w,  α_k ≥ 0  ∀k
     θ^L ≤ (leader Benders cuts)
     θ^F ≤ (follower Benders cuts)
```

IMP variables: α_k, θ^L, θ^F.  Size O(|A|).

**Inner Benders Cut** (added to IMP after solving ISP at α*):

```
θ^L ≤ intercept_L + sg_L' α       (leader cut)
θ^F ≤ intercept_F + sg_F' α       (follower cut)
```

where `intercept = Z(α*) - sg' α*` and sg = ∂Z/∂α (subgradient).

### 9.2.1 Subgradient ∂Z^L/∂α_k

α_k appears in RHS of L7–L10. The subgradient is:

```
∂Z^L/∂α_k = Σ_s q̂_s · (∂Z/∂RHS_L7[s,k])
           + Σ_s q̂_s · (∂Z/∂RHS_L8[s,k])
           + 2ε̂   · (∂Z/∂RHS_L9[k])
           + 1     · (∂Z/∂RHS_L10[k])
```

**⚠ JuMP shadow_price convention**:
JuMP `shadow_price` = sensitivity to *relaxation*, NOT to RHS increase.
- For ≤ constraints: relaxation = RHS↑ → `shadow_price = ∂obj/∂RHS` (use directly)
- For ≥ constraints: relaxation = RHS↓ → `shadow_price = -∂obj/∂RHS` (**must negate**)
- For == constraints: `shadow_price = ∂obj/∂RHS` (use directly)

Therefore in code:

```
sg_k += q[s] *   shadow_price(L7[s,k])      # L7 is ≤
sg_k += q[s] * (-shadow_price(L8[s,k]))     # L8 is ≥ → NEGATE
sg_k += 2ε̂  *   shadow_price(L9[k])         # L9 is ≤
sg_k += 1.0  *   shadow_price(L10[k])        # L10 is ==
```

### 9.2.2 Subgradient ∂Z^F/∂α_k

Same structure with F9–F12:

```
sg_k += q[s] *   shadow_price(F9[s,k])      # F9 is ≤
sg_k += q[s] * (-shadow_price(F10[s,k]))    # F10 is ≥ → NEGATE
sg_k += 2ε̃  *   shadow_price(F11[k])        # F11 is ≤
sg_k += 1.0  *   shadow_price(F12[k])        # F12 is ==
```


### 9.3 Leader Subproblem: Σ_s Z^{L,s}(α) — LP

Given α_k.

```
Σ_s Z^{L,s} = max  Σ_s σ̂ˢ

s.t. N_y û^s + N_ts σ̂ˢ  ≤  0                      ∀s          (L1)
     -c_k^s a_s + û_k^s - a_{s,k}^ν  ≤  0          ∀k, s       (L2)

     a_s - b_s  ≤  q̂_s                              ∀s          (L3)
     a_s + b_s  ≥  q̂_s                              ∀s          (L4)
     Σ_s b_s  ≤  2ε̂                                             (L5)
     Σ_s a_s  =  1                                               (L6)

     a_{s,k}^ν - b_{s,k}^ν  ≤  q̂_s α_k             ∀s, k       (L7)
     a_{s,k}^ν + b_{s,k}^ν  ≥  q̂_s α_k             ∀s, k       (L8)
     Σ_s b_{s,k}^ν  ≤  2ε̂ α_k                       ∀k          (L9)
     Σ_s a_{s,k}^ν  =  α_k                           ∀k          (L10)

     û_k^s, σ̂ˢ, a_s, b_s, a_{s,k}^ν, b_{s,k}^ν ≥ 0             (L11)
```


### 9.4 Follower Subproblem: Σ_s Z^{F,s}(α) — LP

Given α_k.

```
Σ_s Z^{F,s} = max  λ̄ Σ_s σ̃ˢ  -  Σ_s Σ_k r_k^s β_k^s

s.t. N_y ũ^s + N_ts σ̃ˢ  ≤  0                      ∀s          (F1)
     -c_k^s d_s + ũ_k^s - d_{s,k}^ν  ≤  0          ∀k, s       (F2)
     [N_yᵀ ωˢ]_k + β_k^s  ≥  0                     ∀k, s       (F3)
     N_tsᵀ ωˢ  ≥  d_s                               ∀s          (F4)

     d_s - e_s  ≤  q̂_s                              ∀s          (F5)
     d_s + e_s  ≥  q̂_s                              ∀s          (F6)
     Σ_s e_s  ≤  2ε̃                                             (F7)
     Σ_s d_s  =  1                                               (F8)

     d_{s,k}^ν - e_{s,k}^ν  ≤  q̂_s α_k             ∀s, k       (F9)
     d_{s,k}^ν + e_{s,k}^ν  ≥  q̂_s α_k             ∀s, k       (F10)
     Σ_s e_{s,k}^ν  ≤  2ε̃ α_k                       ∀k          (F11)
     Σ_s d_{s,k}^ν  =  α_k                           ∀k          (F12)

     ũ_k^s, σ̃ˢ, ωˢ, β_k^s, d_s, e_s, d_{s,k}^ν, e_{s,k}^ν ≥ 0  (F13)
```


## 10. Summary

```
OMP (MILP)          →   IMP (LP, α_k)        →   Leader LP (all s, radius ε̂)
binary x, h, λ, ψ⁰      O(|A|) vars               + Follower LP (all s, radius ε̃)
```

| | Wasserstein | TV |
|---|---|---|
| OMP | MILP | MILP |
| IMP | LP, O(\|A\|) | LP, O(\|A\|) |
| Leader/Follower Sub | **SDP** | **LP** |
