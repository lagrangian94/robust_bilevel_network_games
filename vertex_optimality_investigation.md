# Vertex Optimality of α: Investigation Summary

## 1. Conjecture

**Claim:** optimal α* for Q₀(χ) = max_{α ∈ Δ} Q₁(α, χ) lies at a vertex of the simplex Δ = {α ≥ 0 : Σα = w/S}, i.e., α* = (w/S)·eₖ for some k ∈ A.

**Motivation:** If true, IMP becomes combinatorial (|A| candidates), enables scenario-wise multi-cut at OMP level, and opens C&CG approach.

---

## 2. Proof Attempt via Sion's Minimax Theorem

### 2.1 Setup

$$Q_0(\chi) = \max_{\alpha \in \Delta} \min_{p \in \prod_s F_s(\chi)} H(\alpha, p)$$

where H(α, p) = Σ_s [1/S(η̂_s + η̃_s) + α⊤(μ̂ˢ + μ̃ˢ)] is **bilinear** in (α, p).

### 2.2 Sion's Theorem Requires

- X (sup side) = Δ: convex ✓
- Y (inf side) = ∏F_s: **compact** convex ← key question
- H quasi-concave/usc in α ✓ (linear)
- H quasi-convex/lsc in p ✓ (linear)

### 2.3 Compactness Argument (FLAWED)

Initial argument: LDR coefficient bounds (ϕ_U, π_U, y_U) make F_s compact.

**Problem:** These bounds only cover a subset of variables. The ISP dual (= original primal) contains **free/unbounded variables**:

| Variable | Role | Lower bound | Upper bound | Status |
|----------|------|-------------|-------------|--------|
| Φ̂, Π̂, Φ̃, Π̃ | LDR coefficients | -ϕ_U, -π_U | +ϕ_U, +π_U | ✅ bounded |
| Ỹ, ỹ_ts | LDR coefficients | -y_U | +y_U | ✅ bounded |
| μ̂_k, μ̃_k | coupling dual | ≥ 0 | **none** | ❌ unbounded |
| η̂_s | Wasserstein dual | ≥ 0 (leader) | **none** | ❌ unbounded |
| η̃_s | Wasserstein dual | free | **none** | ❌ unbounded |
| ϑ̂_s, ϑ̃_s | trace constraint | ≥ 0 | **none** | ❌ unbounded |
| β̂₁, β̂₂, β̃₁, β̃₂ | uncertainty set | ≥ 0 | **none** | ❌ unbounded |
| Ẑ₁, Ẑ₂, Z̃₁, Z̃₂ | equality duals | free | **none** | ❌ unbounded |
| Λ̂, Λ̃ | SOC vars | SOC cone | **none** | ❌ unbounded |
| Γ̂, Γ̃ | SOC vars | SOC cone | **none** | ❌ unbounded |

Note: Some variables (Z, Λ) are determined as affine functions of others via equality constraints (manuscript eqs 31-44), so they inherit bounds if their "parents" are bounded. But the chain of dependencies needs to be fully traced.

---

## 3. Alternative Proof Attempt (Without Sion)

### 3.1 Argument via SDP Strong Duality

The argument does NOT require Sion:

**Step 1:** Joint SDP strong duality (Slater):
$$\max_{\alpha \in \Delta, z \in C} \text{obj} = \min_{p, \nu} \text{dual obj}$$

**Step 2:** The dual can be written as:
$$\min_{p} \max_{\alpha \in \Delta} H(\alpha, p)$$
because ν (dual of Σα = w/S) and the dual of α ≥ 0 produce exactly the inner LP max over Δ.

**Step 3:** Therefore max_α min_p = min_p max_α (value equality via strong duality, not Sion).

**Step 4:** Saddle point (α*, p*) should exist, with α* = argmax H(α, p*) = vertex (LP on simplex).

### 3.2 Where This Breaks

Value equality holds: both sides = 3.567. But **saddle point with vertex α is not guaranteed**.

A saddle point (α̃, p*) requires TWO conditions simultaneously:
1. α̃ = argmax_α H(α, p*) ← vertex ✓
2. p* = argmin_p H(α̃, p) ← **FAILS**

Experimental proof (Step 3→4 in Test H):
- Fix p* from interior optimal, solve max_α H(α, p*) → vertex α̃ = (w/S)·e₃, H = 5.137
- Re-optimize: min_p H(α̃, p) = -2.0 ≠ 5.137

So (α̃_vertex, p*) is NOT a saddle point. The interior (α*, p*) IS a saddle point:
- α* = argmax H(α, p*) (achieves same max as vertex, H(α*, p*) = 3.567)
- p* = argmin H(α*, p) = Q₁(α*) = 3.567 ✓

**The issue is fundamental:** When α moves to a vertex, μ_{k≠j} become "free" in the objective (coefficient α_k = 0), so the minimizer exploits them to collapse the objective.

---

## 4. Experimental Evidence

### 4.1 Test E: Vertex Sweep (ϕ_U = λ_U = 100)

- **Free α:** obj = 3.567, α distributed on 4 arcs (k=3,18,33,34)
- **All 36 vertices:** obj = -2.0 uniformly
- **Gap = 5.567**

### 4.2 Test F: Shadow Prices at Interior vs Vertex

**Interior optimal (α free):**
- Active arcs (α > 0): μ̂ + μ̃ = 1.5 exactly (= ν, simplex dual)
- Inactive arcs: μ̂ + μ̃ ≈ 1.06-1.09 < 1.5
- Individual μ̂: max ≈ 0.59, individual μ̃: max ≈ 1.09

**Vertex α³ (obj = -2.0):**
- k=18 (Krakow→t): μ̂ = 2.78, μ̃ = 2.80, sum = 5.58
- k=33 (s→Kolobrzeg): μ̂ = 2.60, μ̃ = 2.00, sum = 4.60
- Most other arcs: μ̂ = μ̃ = 0.0

**Interpretation:** At vertex α³, μ on other arcs (especially k=18,33,34) explodes because α_k = 0 removes them from objective, allowing minimizer to exploit them freely.

### 4.3 Test G: μ ≤ 2.0 Bound

- Free α: obj = 3.567 (unchanged, bound non-binding) ✓
- All vertices: still -2.0 ✗

**Conclusion:** μ bound alone is insufficient. Other unbounded variables also contribute.

### 4.4 Test H: Fix-and-Re-optimize

| Step | What | Result |
|------|------|--------|
| 1 | OSP → free α* | obj = 3.567 |
| 2 | Fix α*, solve primal ISP → p*, μ* | obj = 3.567 ✓ |
| 3 | Fix p*, LP over simplex → α̃ | vertex j=3, LP obj = 5.137 |
| 4 | Fix α̃, re-optimize primal ISP | obj = -2.0 ✗ |
| 5 | Full vertex sweep (primal ISP) | all -2.0 |

**This is the definitive diagnostic:**
- Saddle point condition 1: α̃ = argmax H(α, p*) → vertex ✓
- Saddle point condition 2: p* = argmin H(α̃, p) → 5.137 ≠ -2.0 ✗

---

## 5. Mathematical Root Cause

Q₁(α, χ) = min_p H(α, p) where the feasible set F is **independent of α** and H is bilinear.

Since Q₁ is the infimum of affine functions in α → Q₁ is **concave** in α (not convex, not linear).

The outer problem is: max of concave function on simplex → **interior optimal is generic**.

This is fundamentally different from: max of convex/linear function on polytope → vertex optimal.

No amount of bounding can change concavity. The question is whether making ∏F compact can force saddle point existence, which would then imply vertex optimality via a different route.

---

## 6. Open Question: Can ∏F Be Made Compact?

### 6.1 Why It Might Work

If ∏F is compact → Sion applies → saddle point exists → α* at vertex.

The key insight: **if a valid bound on every unbounded variable can be derived from problem structure without changing the optimal value**, then the compactified problem is equivalent to the original, and vertex optimality holds.

### 6.2 What Needs to Be Bounded

Every variable in F_s that lacks an upper bound. From Section 2.5.1:

**Tier 1 — appear in H directly:**
- μ̂_k, μ̃_k: coefficient = α_k in objective
- η̂_s, η̃_s: coefficient = 1/S in objective

**Tier 2 — appear only in constraints (but can be unbounded):**
- ϑ̂_s, ϑ̃_s (trace constraint related)
- β̂₁, β̂₂, β̃₁, β̃₂ (uncertainty set related)
- Ẑ₁, Ẑ₂, Z̃₁, Z̃₂ (equality constraint duals, free)
- Λ̂, Λ̃, Γ̂, Γ̃ (SOC variables)

### 6.3 Possible Sources of Valid Bounds

**μ̂, μ̃:** Dual of β̂_{2,k} = α_k. In network interdiction, μ represents marginal value of Wasserstein budget per arc. Potential bound from max-flow duality: ϕ ∈ [0,1] → μ̂ + μ̃ ≤ 2. (Tested with M_μ = 2.0, insufficient alone.)

**η̂, η̃:** Dual of probability normalization. Bounded by problem's objective range?

**ϑ̂, ϑ̃:** Related to tr(M_{11}) - M_{22}ε² ≤ 0. May be bounded by ε and matrix dimensions.

**β̂, β̃:** Uncertainty set duals. May be bounded by uncertainty set geometry (R, r bounds).

**Z variables:** Determined by equality constraints as affine functions of Φ, Π, Λ, β. If all "parent" variables are bounded, Z is automatically bounded.

**Λ, Γ (SOC):** SOC cone variables. If they appear in equalities with bounded RHS, they may be bounded.

### 6.4 Strategy: Dependency Chain Analysis

Many variables are linked by equality constraints:
```
Φ, Π, Y (bounded by ±ϕ_U, ±π_U, ±y_U)
    ↓ (equalities 31-34)
Z₁, Z₂ = affine(Φ, Π, U, β, Λ)
    ↓ (equalities 41-44)  
Λ = affine(Z, Γ)
    ↓ (SOC constraints)
Γ ∈ SOC
```

If the chain can be traced to show all variables are bounded functions of the already-bounded LDR coefficients, then ∏F is compact **without any additional constraints**.

The key unknowns are:
1. Are the equality constraints sufficient to express ALL unbounded variables as functions of bounded ones?
2. Or do some variables (μ, η, ϑ, β) have genuine degrees of freedom that are unbounded?

### 6.5 Recommended Next Steps

1. **Enumerate all unbounded variables** in ISP primal code (build_primal_isp.jl or equivalent)
2. **Trace equality constraint dependencies** to identify which unbounded variables are determined by bounded ones
3. **For remaining free variables**, attempt to derive valid bounds from problem structure
4. **Add ALL valid bounds** simultaneously and re-run vertex sweep
5. If vertex optimality is restored → Proposition is valid under explicit compactness conditions
6. If still fails → investigate whether some bounds are inherently impossible

### 6.6 Quick Experimental Check

Before doing full analysis, a simple diagnostic: add **arbitrary large bounds** (e.g., M = 1000) on ALL unbounded variables and re-run vertex sweep. If vertex optimality appears → valid tight bounds exist. If still -2.0 → the issue is deeper than compactness.

```julia
# In ISP primal builder, add to ALL unbounded variables:
M_big = 1000.0
@constraint(model, μhat .<= M_big)
@constraint(model, μtilde .<= M_big)
@constraint(model, ηhat .<= M_big)
# η̃ is free, so both directions:
@constraint(model, ηtilde .<= M_big)
@constraint(model, ηtilde .>= -M_big)
@constraint(model, ϑhat .<= M_big)
@constraint(model, ϑtilde .<= M_big)
@constraint(model, βhat1 .<= M_big)
@constraint(model, βhat2 .<= M_big)
@constraint(model, βtilde1 .<= M_big)
@constraint(model, βtilde2 .<= M_big)
# Z variables (free):
@constraint(model, Zhat1 .<= M_big)
@constraint(model, Zhat1 .>= -M_big)
# ... etc for all Z, Λ, Γ
```

If this recovers vertex optimality with M=1000, progressively tighten bounds to find which variables are the binding ones.

---

## 7. Summary of Current Status

| Item | Status |
|------|--------|
| Minimax value equality | ✅ Holds (strong duality) |
| Saddle point existence (general) | ❌ Not guaranteed (∏F non-compact) |
| Vertex optimality | ❌ Experimentally disproved (all vertices = -2.0) |
| Proposition | ⏸ Withdrawn pending compactness resolution |
| C&CG approach | ⏸ Withdrawn pending vertex optimality |
| Scenario multi-cut at OMP | ❌ Invalid (α coupling, independent of vertex question) |
| Nested Benders (IMP+ISP) | ✅ Valid, current working approach |
| Partial inner solve | ✅ Valid (any feasible α gives valid underestimator) |
| Best incumbent tracking | ✅ Valid, should implement |
| multi_cut_lf invalid | ✅ Confirmed (α couples L and F) |
