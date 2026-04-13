# OMP Strengthening via Min-Cut Valid Inequalities for True-DRO Network Interdiction

**Research Note**

---

## 1. Problem Setup and Notation

### 1.1 Network

Directed network G = (V, A) with source s, sink t, |A| = m. Node-arc incidence: N = [N_y | N_ts]. For each arc k ∈ A and scenario j ∈ [S]:

- ξ̄^j_k > 0: nominal capacity
- v_k ∈ {0,1}: interdiction effectiveness

### 1.2 Decisions

Leader: x ∈ X := {x ∈ {0,1}^m : 1ᵀx ≤ γ} (interdiction).
Follower: h ∈ H := {h ∈ ℝ₊^m : 1ᵀh ≤ w} (recovery), then flow ỹ after observing ξ.

### 1.3 Max-Flow

Given (x, h, j), the residual capacity of arc k is u^j_k(x,h) := ξ̄^j_k(1 - v_k x_k) + h_k.
The max-flow is

```
F^j(x, h) := max { ỹ_ts ≥ 0 : ∃ ỹ ≥ 0, N_y ỹ + N_ts ỹ_ts = 0, ỹ_k ≤ u^j_k(x,h) ∀k }
```

### 1.4 True DRO (TV Distance)

Scenario support {ξ̄¹,...,ξ̄ˢ} is finite and fixed. Nominal distribution q̂ = (q̂₁,...,q̂_S), q̂_j > 0, Σ q̂_j = 1. Ambiguity sets:

```
D̂ := { q ∈ ℝ₊ˢ : 1ᵀq = 1, ½‖q - q̂‖₁ ≤ ε̂ }
D̃ := { q ∈ ℝ₊ˢ : 1ᵀq = 1, ½‖q - q̂‖₁ ≤ ε̃ }
```

After removing the argmax operator and exploiting the zero-sum structure (c=0, d₀=d₁), the pessimistic bilevel for fixed x is:

```
V(x, P̂, P̃) = sup_{h̃ ∈ H, ŷ, ỹ}  E^{P̂}[ŷ_ts]
    s.t. E^{P̃}[ỹ_ts] ≥ z*(x), flow/capacity constraints under P̂, P̃.
```

The True DRO objective:

```
V*(x) := sup_{(P̂, P̃) ∈ D̂ × D̃} V(x, P̂, P̃)
```

**Remark (Reformulated form).** After LP dualization and the transform ỹ ← λy, h ← λh, the problem becomes V\*(x) = sup_{α ≥ 0, 1ᵀα ≤ w} inf_{λ,h,ψ⁰} [Piece-L(α,x) + Piece-F'(α,x,λ,h,ψ⁰)]. In this form h appears inside inf. **The proofs below use the original form.**

### 1.5 OMP

```
min_{x ∈ X, t₀}  t₀    s.t. t₀ ≥ c^(ℓ) + Σ_k κ_k^(ℓ) x_k,  ℓ = 1,...,L.
```

Variables: x (binary) and t₀ only. All other variables reside in the subproblem. κ_k^(ℓ) denotes the Benders cut slope coefficient.

---

## 2. Min-Cut Dual of Max-Flow

By LP strong duality, F^j(x, h) = mincut^j(x, h):

```
F^j(x, h) = min_{δ, τ} { Σ_{k∈A} u^j_k(x,h) δ_k : δ_k ≥ τ_{j(k)} - τ_{i(k)} ∀k, τ_s = 0, τ_t = 1, δ ≥ 0 }
```

where δ_k is the cut indicator for arc k, τ_i is the node potential, and i(k), j(k) are the tail and head of arc k.

**Lemma 1 (Bounded variables).** In any optimal solution, δ_k ∈ [0,1] and τ_i ∈ [0,1].

*Proof.* τ_s = 0, τ_t = 1, and δ_k ≥ τ_{j(k)} - τ_{i(k)} imply projecting to [0,1] does not increase the objective. □

---

## 3. Phase 1: Base Valid Inequality (h = 0)

**Proposition 1 (No-recovery min-cut valid inequality).** The following constraints, added to the OMP with auxiliary variables δ^j_k, τ^j_i, w^j_k for each j ∈ [S], are valid:

```
(DC-1)  δ^j_k ≥ τ^j_{j(k)} - τ^j_{i(k)}              ∀k ∈ A, j ∈ [S]
(DC-2)  τ^j_s = 0,  τ^j_t = 1                          ∀j ∈ [S]
(DC-3)  0 ≤ δ^j_k ≤ 1,  0 ≤ τ^j_i ≤ 1                 ∀k, i, j
(MC-1)  w^j_k ≤ x_k                                     ∀k, j
(MC-2)  w^j_k ≤ δ^j_k                                   ∀k, j
(MC-3)  w^j_k ≥ δ^j_k - (1 - x_k)                      ∀k, j
(MC-4)  w^j_k ≥ 0                                        ∀k, j
(DC-link) t₀ ≥ Σ_j q̂_j Σ_k ξ̄^j_k (δ^j_k - v_k w^j_k)
```

*Proof.*

**Step 1 (Nominal distribution).** q̂ ∈ D̂ ∩ D̃, so

V\*(x) ≥ V(x, q̂, q̂).

**Step 2 (Same-distribution simplification).** When P̂ = P̃ = q̂, the value function constraint and the objective are evaluated under the same distribution. Consequently, the optimal h̃ is an argmax of Σ_j q̂_j F^j(x, ·) over H, and

V(x, q̂, q̂) = max_{h ∈ H} Σ_j q̂_j F^j(x, h).

**Step 3 (Recovery relaxation).** h = 0 ∈ H, so max_h ≥ evaluation at h=0:

V\*(x) ≥ Σ_j q̂_j F^j(x, 0).

**Step 4 (LP strong duality).** By min-cut duality with h=0:

F^j(x, 0) = min_{δ^j, τ^j ∈ (DC-1)-(DC-3)} Σ_k ξ̄^j_k (1 - v_k x_k) δ^j_k.

**Step 5 (McCormick).** The objective contains x_k · δ^j_k. Since x_k ∈ {0,1} and δ^j_k ∈ [0,1] (Lemma 1), McCormick (MC-1)-(MC-4) with w^j_k representing x_k δ^j_k is exact at integer x:

x_k ∈ {0,1} ⟹ w^j_k = x_k δ^j_k, and ξ̄^j_k(1-v_k x_k)δ^j_k = ξ̄^j_k(δ^j_k - v_k w^j_k).

**Step 6 (OMP realizes the bound).** The OMP minimizes t₀; (DC-link) forces t₀ ≥ Σ_j q̂_j Σ_k ξ̄^j_k(δ^j_k - v_k w^j_k). Since the OMP also chooses (δ^j, τ^j, w^j) to minimize t₀ over the min-cut feasible region, it obtains exactly Σ_j q̂_j F^j(x, 0) at integer x. Combined: t₀ ≥ Σ_j q̂_j F^j(x,0) ≤ V\*(x). □

**Remark (Why primal embedding fails).** Embedding the max-flow primal (flow variables ỹ^j, balance, capacity, and t₀ ≥ Σ q̂_j ỹ^j_ts) yields a trivial bound: ỹ = 0 is always feasible, and the OMP's minimization sets ỹ^j_ts = 0. The max-flow primal is a maximization inside a minimization master. The min-cut dual is a minimization, matching the OMP's direction.

---

## 4. Phase 2: α*-Enhanced Valid Inequality

### 4.1 α* as Feasible Recovery

**Lemma 2.** Let α^(ℓ) ≥ 0 with 1ᵀα^(ℓ) ≤ w be recovered from the subproblem at iteration ℓ. Then α^(ℓ) ∈ H.

*Proof.* H = {h ≥ 0 : 1ᵀh ≤ w}. Dual nonnegativity gives α^(ℓ) ≥ 0; the Lagrangian constraint gives 1ᵀα^(ℓ) ≤ w. □

### 4.2 Statement

**Proposition 2 (α*-enhanced min-cut valid inequality).** Let α^(ℓ) ∈ H. The following, with fresh auxiliaries δ^{ℓ,j}_k, τ^{ℓ,j}_i, w^{ℓ,j}_k, is valid:

```
(DC-1')  δ^{ℓ,j}_k ≥ τ^{ℓ,j}_{j(k)} - τ^{ℓ,j}_{i(k)}       ∀k, j
(DC-2')  τ^{ℓ,j}_s = 0,  τ^{ℓ,j}_t = 1                        ∀j
(DC-3')  0 ≤ δ^{ℓ,j}_k ≤ 1,  0 ≤ τ^{ℓ,j}_i ≤ 1               ∀k, i, j
(MC')    McCormick (MC-1)-(MC-4) for w^{ℓ,j}_k ≈ x_k δ^{ℓ,j}_k  ∀k, j
(DC-link') t₀ ≥ Σ_j q̂_j Σ_k [ ξ̄^j_k(δ^{ℓ,j}_k - v_k w^{ℓ,j}_k) + α^(ℓ)_k δ^{ℓ,j}_k ]
```

*Proof.*

**Step 1.** By Proposition 1 Steps 1-2: V\*(x) ≥ max_{h ∈ H} Σ_j q̂_j F^j(x, h). Set h̃ = α^(ℓ) ∈ H (Lemma 2):

V\*(x) ≥ Σ_j q̂_j F^j(x, α^(ℓ)).

**Step 2.** Residual capacity: u^j_k = ξ̄^j_k(1-v_k x_k) + α^(ℓ)_k. By min-cut duality:

F^j(x, α^(ℓ)) = min_{δ^j, τ^j} Σ_k [ξ̄^j_k(1-v_k x_k) + α^(ℓ)_k] δ^j_k.

**Step 3.** At integer x: w^j_k = x_k δ^j_k, so [ξ̄^j_k(1-v_k x_k) + α^(ℓ)_k] δ^j_k = ξ̄^j_k(δ^j_k - v_k w^j_k) + α^(ℓ)_k δ^j_k, matching (DC-link').

**Step 4.** OMP minimizes over min-cut feasible region, yielding t₀ ≥ Σ_j q̂_j F^j(x, α^(ℓ)) ≤ V\*(x). □

### 4.3 Cumulative Validity

**Corollary 1.** Constraints for ℓ = 1,...,L use independent auxiliaries and different constants α^(ℓ). All are simultaneously valid. Effective bound: t₀ ≥ max_ℓ Σ_j q̂_j F^j(x, α^(ℓ)).

*Proof.* Each uses an independent h̃ = α^(ℓ) ∈ H. No cross-dependencies. □

**Remark (Monotonicity).** α^(ℓ) ≥ 0 ⟹ u^j_k(x, α^(ℓ)) ≥ u^j_k(x, 0) ⟹ F^j(x, α^(ℓ)) ≥ F^j(x, 0). Every Phase 2 inequality dominates Phase 1.

---

## 5. Comparison with Partial Benders (Crainic et al.)

### 5.1 Why Primal Retention Works for Standard Stochastic Programs

In min_y fᵀy + Σ_s p_s z_s with z_s = min cᵀx^s s.t. Dx^s = d^s - B^s y:

1. **Direction match**: recourse min cᵀx^s and master min are both minimizations.
2. **No intermediate variables**: y → x^s is direct.
3. **Convexity**: z_s(y) is convex in RHS d^s → artificial scenario via d^{s'} = Σ α_s d^s gives valid LB by Jensen.

### 5.2 Why This Fails for Network Interdiction

1. **Direction mismatch**: follower's max-flow (max ỹ_ts) vs. master's min t₀. Primal embedding → ỹ_ts = 0. *Resolved by using min-cut dual.*

2. **Intermediate variables**: x → (h, P) → flow. h, P ∉ OMP → must fix them. *Resolved by setting h = α^(ℓ), P = q̂.*

3. **Concavity**: min-cut value F^j(x,h) = min_{δ ∈ Δ} Σ_k u^j_k δ_k is *concave* in capacity u (pointwise min of affine). Average-capacity scenario gives F(ū) ≥ Σ q̂_j F(u^j), i.e., *wrong* direction for a valid LB. *Componentwise-min scenario is the only valid single-scenario strategy with coefficient 1.*

---

## 6. Single-Scenario Variants

For large S, embedding all scenarios is costly. Two alternatives:

### 6.1 Variant A: Single Scenario s' ∈ [S]

**Proposition 3.** Fix s' ∈ [S], h̃ ∈ H. With auxiliaries for scenario s' only:

```
(SA-link) t₀ ≥ q̂_{s'} Σ_k [ ξ̄^{s'}_k(δ^{s'}_k - v_k w^{s'}_k) + h̃_k δ^{s'}_k ]
```

*Proof.* V\*(x) ≥ Σ_j q̂_j F^j(x, h̃). Since F^j ≥ 0: Σ_j q̂_j F^j ≥ q̂_{s'} F^{s'}. Min-cut + McCormick gives (SA-link). □

### 6.2 Variant B: Componentwise-Minimum Scenario

**Proposition 4.** Define ξ̄^{min}_k := min_j ξ̄^j_k. With auxiliaries for one min-cut instance:

```
(SB-link) t₀ ≥ Σ_k [ ξ̄^{min}_k(δ^{min}_k - v_k w^{min}_k) + h̃_k δ^{min}_k ]
```

*Proof.* Monotonicity: ξ̄^{min}_k ≤ ξ̄^j_k ∀j,k implies F^{min}(x, h̃) ≤ F^j(x, h̃) ∀j. Therefore F^{min} ≤ Σ_j q̂_j F^j ≤ V\*(x). □

**Remark (Using an actual scenario j' with coefficient 1 is invalid).** A real scenario j' can have F^{j'}(x) > Σ_j q̂_j F^j(x) for some x (e.g., when j' has high capacity on unchosen arcs). Only ξ̄^{min}, which is dominated by every scenario pointwise, guarantees F^{min} ≤ F^j ∀j.

### 6.3 Comparison

| | Capacity | Coefficient | Bound |
|---|---|---|---|
| All S scenarios | ξ̄^j_k | q̂_j | Σ_j q̂_j F^j (tightest) |
| Variant A (s') | ξ̄^{s'}_k | q̂_{s'} | q̂_{s'} F^{s'} |
| Variant B (comp. min) | min_j ξ̄^j_k | 1 | F^{min} |

Variant B dominates when S is large (q̂_{s'} ≪ 1). Variant A is competitive when q̂ is non-uniform with a dominant scenario. Both require 3|A| + |V| auxiliary variables; adding both costs 2(3|A| + |V|).

---

## 7. Complexity

Per inequality set (Phase 1, Phase 2 iteration, or single-scenario variant):

| | Variables | Constraints |
|---|---|---|
| Per scenario | 3|A| + |V| | 4|A| + |V| + 2 |
| All S scenarios | S(3|A| + |V|) | S(4|A| + |V| + 2) + 1 |
| Single scenario | 3|A| + |V| | 4|A| + |V| + 3 |

### Summary of Bounds

```
0  ≤  F^{min}(x,0)  ≤  Σ_j q̂_j F^j(x, 0)  ≤  Σ_j q̂_j F^j(x, α^(ℓ))  ≤  V*(x)
       [Var. B]          [Phase 1]               [Phase 2]
```

Gap sources: (i) recovery suboptimality (α^(ℓ) vs. optimal h), (ii) nominal vs. worst-case probability, (iii) DRO layer (sup_P, Lagrangian α-ν coupling).
