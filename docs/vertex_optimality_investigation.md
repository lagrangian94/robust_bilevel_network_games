# Vertex Optimality of α: Investigation Summary

## 1. Conjecture

**Claim:** optimal α* for Q₀(χ) = max_{α ∈ Δ} Q₁(α, χ) lies at a vertex of the simplex Δ = {α ≥ 0 : Σα = w/S}, i.e., α* = (w/S)·eₖ for some k ∈ A.

**Motivation:** If true, IMP becomes combinatorial (|A| candidates), enables scenario-wise multi-cut at OMP level, and opens C&CG approach.

**Final verdict: FALSE.** See Section 7.

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

### 2.3 Compactness of ∏F_s

The ISP (Min problem, `build_primal_isp.jl`) variables and their bounds:

| Variable | Role | Lower bound | Upper bound | Status |
|----------|------|-------------|-------------|--------|
| Φ̂, Π̂, Φ̃, Π̃ | LDR coefficients | -ϕ_U, -π_U | +ϕ_U, +π_U | ✅ bounded |
| Ỹ, ỹ_ts | LDR coefficients | -y_U | +y_U | ✅ bounded |
| Ψ̂, Ψ̃ | LDR (interdiction) | 0 | ϕ_U (Big-M) | ✅ bounded |
| μ̂_k, μ̃_k | coupling dual | ≥ 0 | **none** | ❌ unbounded |
| η̂_s | Wasserstein dual | ≥ 0 (leader) | **none** | ❌ unbounded |
| η̃_s | Wasserstein dual | free | **none** | ❌ unbounded |
| ϑ̂_s, ϑ̃_s | trace constraint | ≥ 0 | **none** | ❌ unbounded |
| M̂, M̃ | PSD matrix | PSD cone | **none** | ❌ unbounded |
| Λ̂, Λ̃ | SOC vars | SOC cone | **none** | ❌ unbounded |

Note: In `build_primal_isp.jl`, the Λ (SOC) variables encode what the OSP formulation (`build_dualized_outer_subprob.jl`) represents as separate (β, Z, Γ) variables. Bounding Λ implicitly bounds all of these.

### 2.4 Why Sion Doesn't Help Even With Compactness

Even if ∏F_s is made compact (Test I, Section 4.5), Sion gives a saddle point (α*, p*) with:
- α* ∈ argmax_α H(α, p*) → can choose vertex α̃ with H(α̃, p*) = H(α*, p*)
- **But** (α̃, p*) is NOT a saddle point: p* ≠ argmin_p H(α̃, p)
- So Q₁(α̃) = min_p H(α̃, p) < H(α̃, p*) = Q₁(α*) = V

The vertex gets "punished" when p is re-optimized. This is because Q₁(α) is concave (Section 5).

---

## 3. Alternative Proof Attempt (Without Sion)

### 3.1 Argument via SDP Strong Duality

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

### 4.4 Test H: Fix-and-Re-optimize (Definitive Diagnostic)

| Step | What | Result |
|------|------|--------|
| 1 | OSP → free α* | obj = 3.567 |
| 2 | Fix α*, solve primal ISP → p*, μ* | obj = 3.567 ✓ |
| 3 | Fix p*, LP over simplex → α̃ | vertex j=3, LP obj = 5.137 |
| 4 | Fix α̃, re-optimize primal ISP | obj = -2.0 ✗ |
| 5 | Full vertex sweep (primal ISP) | all -2.0 |

- Saddle point condition 1: α̃ = argmax H(α, p*) → vertex ✓
- Saddle point condition 2: p* = argmin H(α̃, p) → 5.137 ≠ -2.0 ✗

### 4.5 Test I: M=1000 Bounds on ALL Unbounded Variables

**Purpose:** Section 6.1 compactness check — if making ∏F_s compact recovers vertex property, Sion's path is viable.

**Setup:** Added M=1000 box bounds to ALL unbounded variables in `build_primal_isp.jl`:
- Leader: ηhat ≤ M, μhat ≤ M, ϑhat ≤ M, Mhat ∈ [-M,M], Λhat1/2 ∈ [-M,M]
- Follower: ηtilde ∈ [-M,M], μtilde ≤ M, ϑtilde ≤ M, Mtilde ∈ [-M,M], Λtilde1/2 ∈ [-M,M]

Note: Λ in `build_primal_isp.jl` encodes (β, Z, Γ) from the OSP formulation. Bounding Λ implicitly bounds all of them.

**Implementation verification:** α appears ONLY in the primal ISP objective (as μhat/μtilde coefficient via `set_objective_coefficient`). α is NOT a function argument of `build_primal_isp_leader/follower` and does NOT appear in any constraints. This is correct: the primal ISP is the dual of the dual ISP, where the coupling constraint `βhat2[s,k] ≤ α[k]` dualizes to `α[k]` becoming the objective coefficient of `μhat[s,k]`.

**Results (Polska S=1):**

| Setting | obj |
|---------|-----|
| α* (free, no bound) | 3.567 |
| α* (free, M=1000) | 3.567 (bound non-binding ✓) |
| All 36 vertices (M=1000) | **-2.0** uniformly |
| Gap | **-5.567** |

**Conclusion:** Compactification does NOT recover vertex property. Even with compact ∏F_s and Sion's theorem giving a saddle point, vertex optimality still fails.

---

## 5. Mathematical Root Cause

Q₁(α, χ) = min_p H(α, p) where the feasible set F is **independent of α** and H is bilinear.

Since Q₁ is the infimum of affine functions in α → Q₁ is **concave** in α (not convex, not linear).

The outer problem is: max of concave function on simplex → **interior optimal is generic**.

This is fundamentally different from: max of convex/linear function on polytope → vertex optimal.

**Why vertex α fails mechanically:** At vertex α_j = (w/S)·e_j, only μhat[j] and μtilde[j] have nonzero objective coefficients. The minimizer can freely choose μhat[k≠j], μtilde[k≠j] (zero cost) and exploit the constraint system to reduce η (and thus the total objective) down to -λ_sol = -2.0.

**Why Sion + compactness doesn't help:** Sion guarantees a saddle point (α*, p*) where α* can be chosen as a vertex for H(·, p*). But Q₁(α_vertex) = min_p H(α_vertex, p) ≤ H(α_vertex, p*) — the vertex value after re-optimization is generically lower because the minimizer adapts to the vertex.

---

## 6. Formulation Notes

### 6.1 Primal ISP vs OSP Variable Correspondence

The two formulations are duals of each other (both include M as shared PSD variable):

| Primal ISP (Min, `build_primal_isp.jl`) | OSP (Max, `build_dualized_outer_subprob.jl`) |
|---|---|
| ηhat, ηtilde | (dual of SDP linking) |
| μhat, μtilde | (dual of coupling) → α in objective |
| Φhat, Ψhat, Πhat | (dual of Big-M, SOC) → Uhat, Phat |
| ϑhat, ϑtilde | (dual of trace) |
| Mhat, Mtilde (PSD) | Mhat, Mtilde (PSD) — shared |
| **Λhat1/2, Λtilde1/2 (SOC)** | **βhat1/2, Zhat1/2, Γhat1/2** |

Key: Primal ISP's Λ encodes OSP's (β, Z, Γ). They are NOT separate variables in the primal ISP.

### 6.2 Why α Is Only in Objective

In the dual ISP (Max), the coupling constraint is: `βhat2[s,k] ≤ α[k]` with shadow price μhat.

Dualizing to the primal ISP (Min):
- μhat becomes a variable (≥ 0)
- α[k] (the RHS) becomes the objective coefficient of μhat[k]
- No coupling constraint with α in the primal ISP

This is why `set_objective_coefficient(model, μhat[k], α_sol[k])` is sufficient.

---

## 7. Final Conclusion

**Vertex optimality of α is FALSE for this conic formulation.**

### Root Cause
Q₁(α, χ) = min_p H(α,p) is the infimum of affine functions in α → **concave** in α.
Max of concave on simplex → **interior optimal is generic**.

### What Was Tested

| Test | What | Result |
|------|------|--------|
| A-E | OSP vertex sweep (various ϕU, λU) | All vertices = -2.0 |
| F | Shadow prices μ at interior vs vertex | μ explodes at vertex (α_k=0 frees μ_k) |
| G | μ ≤ 2.0 bound (slack+penalty) | Still -2.0 |
| H | Primal ISP → fix vars → LP on simplex → re-optimize | LP gives vertex, re-optimize gives -2.0 (saddle point condition 2 fails) |
| I | M=1000 bounds on ALL unbounded primal ISP vars (incl. Λ = β,Z,Γ) | Still -2.0 (compact ∏F_s doesn't help) |

### Implementation Verification (Test I)

- α is NOT in primal ISP constraints (verified: not a function argument, no grep hits in constraints)
- α is ONLY in objective via `set_objective_coefficient` on μhat/μtilde
- This is correct by duality: coupling `βhat2 ≤ α` → objective coefficient α·μhat
- M=1000 bounds are non-binding at free α* (obj unchanged at 3.567)
- Primal ISP Λ encodes OSP's (β, Z, Γ) → all implicitly bounded

### Implications

| Item | Status |
|------|--------|
| Minimax value equality | ✅ Holds (strong duality) |
| Vertex optimality of α | ❌ **Definitively disproved** (concavity + Tests H,I) |
| Sion/compactness path | ❌ **Closed** (Test I: compact ∏F_s doesn't help) |
| C&CG with vertex α | ❌ Invalid |
| C&CG with free α (IMP pricing) | ✅ Valid (refactored in `ccg_benders.jl`) |
| Nested Benders (IMP+ISP) | ✅ Valid, current working approach |
| Partial inner solve | ✅ Valid (any feasible α gives valid underestimator) |
| multi_cut_lf invalid | ✅ Confirmed (α couples L and F) |
