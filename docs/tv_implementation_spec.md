# TV Network Interdiction DRO: Nested Benders Implementation Spec

## Overview

Three-level nested Benders decomposition for network interdiction DRO with total variation ambiguity set.

```
OMP (MILP)  ──cuts──>  IMP (LP, α_k)  ──cuts──>  ISP-L (LP) + ISP-F (LP)
   x, h, λ, ψ⁰, t₀       α_k, θ^L, θ^F            per α_k, all s together
```

## Notation

- Network: G = (V, A), node-arc incidence N = [N_y | N_ts] ∈ R^{m×(|A|+1)}, m = |V|-1
- Scenarios: ξ̄ˢ ∈ R^|A|, s ∈ [S], nominal prob q̂_s > 0
- TV radii: ε̂ (leader), ε̃ (follower)
- For fixed x: c_k^s := ξ̄_k^s (1 - v_k x_k)
- For fixed x, h, λ: r_k^s := h_k + λ c_k^s

---

## Full Model (for verification)

This is the complete TV reformulation as a single MILP. Solve directly for small instances to verify Benders results.

### Variables

**First-stage (binary + continuous):**
- x ∈ {0,1}^|A| (interdiction)
- h ∈ R₊^|A| (recovery)
- λ ≥ 0 (coupling scalar)
- ψ⁰ ∈ R₊^|A| (McCormick auxiliary, = λx)
- t ∈ R (epigraph, free)
- ν ≥ 0 (recovery budget dual)

**TV dual variables (leader, obj coupling):**
- σ_s^{L+}, σ_s^{L-} ≥ 0, ∀s
- μ^L ≥ 0
- η^L free

**TV dual variables (follower, obj coupling):**
- σ_s^{F+}, σ_s^{F-} ≥ 0, ∀s
- μ^F ≥ 0
- η^F free

**TV dual variables (leader, ν coupling), ∀k:**
- σ_{s,k}^{Lν+}, σ_{s,k}^{Lν-} ≥ 0, ∀s,k
- μ_k^{Lν} ≥ 0, ∀k
- η_k^{Lν} free, ∀k

**TV dual variables (follower, ν coupling), ∀k:**
- σ_{s,k}^{Fν+}, σ_{s,k}^{Fν-} ≥ 0, ∀s,k
- μ_k^{Fν} ≥ 0, ∀k
- η_k^{Fν} free, ∀k

**Scenario recourse (leader), ∀s:**
- π̂^s ∈ R₊^m (node price)
- φ̂^s ∈ R₊^|A| (arc capacity dual)

**Scenario recourse (follower), ∀s:**
- π̃^s ∈ R₊^m (node price)
- φ̃^s ∈ R₊^|A| (arc capacity dual)
- ỹ^s ∈ R₊^|A| (flow)
- ỹ_ts^s ≥ 0 (dummy arc flow)

### Formulation

where: g_s^L := Σ_k c_k^s φ̂_k^s,   g_s^F := Σ_k c_k^s φ̃_k^s - ỹ_ts^s,   c_k^s := ξ̄_k^s(1 - v_k x_k)

```
min  t + w ν

s.t.
=== Epigraph (TV dualized) ===

(T1)  Σ_s q̂_s(σ_s^{L+} - σ_s^{L-}) + 2ε̂ μ^L + η^L
      + Σ_s q̂_s(σ_s^{F+} - σ_s^{F-}) + 2ε̃ μ^F + η^F  ≤  t

(T2)  σ_s^{L+} - σ_s^{L-} + η^L  ≥  g_s^L             ∀s
(T3)  σ_s^{L+} + σ_s^{L-}  ≤  μ^L                      ∀s

(T4)  σ_s^{F+} - σ_s^{F-} + η^F  ≥  g_s^F              ∀s
(T5)  σ_s^{F+} + σ_s^{F-}  ≤  μ^F                      ∀s


=== Leader dual feasibility ===

(T6)  [N_yᵀ π̂^s]_k + φ̂_k^s  ≥  0                      ∀k, s
(T7)  N_tsᵀ π̂^s  ≥  1                                   ∀s


=== Follower dual feasibility ===

(T8)  [N_yᵀ π̃^s]_k + φ̃_k^s  ≥  0                      ∀k, s
(T9)  N_tsᵀ π̃^s  ≥  λ                                   ∀s


=== ν coupling (TV dualized) ===

(T10) Σ_s q̂_s(σ_{s,k}^{Lν+} - σ_{s,k}^{Lν-}) + 2ε̂ μ_k^{Lν} + η_k^{Lν}
      + Σ_s q̂_s(σ_{s,k}^{Fν+} - σ_{s,k}^{Fν-}) + 2ε̃ μ_k^{Fν} + η_k^{Fν}
      ≤  ν                                               ∀k

(T11) σ_{s,k}^{Lν+} - σ_{s,k}^{Lν-} + η_k^{Lν}  ≥  φ̂_k^s     ∀s, k
(T12) σ_{s,k}^{Lν+} + σ_{s,k}^{Lν-}  ≤  μ_k^{Lν}              ∀s, k

(T13) σ_{s,k}^{Fν+} - σ_{s,k}^{Fν-} + η_k^{Fν}  ≥  φ̃_k^s     ∀s, k
(T14) σ_{s,k}^{Fν+} + σ_{s,k}^{Fν-}  ≤  μ_k^{Fν}              ∀s, k


=== Follower primal feasibility ===

(T15) N_y ỹ^s + N_ts ỹ_ts^s  ≤  0                      ∀s
(T16) ỹ_k^s  ≤  h_k + (λ - v_k ψ_k⁰) ξ̄_k^s            ∀k, s


=== Budget and McCormick ===

(T17) 1ᵀh  ≤  λ w
(MC1) 0  ≤  ψ_k⁰  ≤  λ^U x_k                           ∀k
(MC2) ψ_k⁰  ≤  λ                                        ∀k
(MC3) ψ_k⁰  ≥  λ - λ^U(1 - x_k)                        ∀k

=== Integrality and budget ===

(INT) x ∈ {0,1}^|A|,  1ᵀx ≤ γ

=== Sign constraints ===

(T18) π̂^s, φ̂^s, π̃^s, φ̃^s, ỹ^s, ỹ_ts^s ≥ 0  ∀s
      λ, ν ≥ 0
      σ_s^{L±}, σ_s^{F±}, μ^L, μ^F ≥ 0
      σ_{s,k}^{Lν±}, σ_{s,k}^{Fν±}, μ_k^{Lν}, μ_k^{Fν} ≥ 0
      η^L, η^F, η_k^{Lν}, η_k^{Fν} free
```

### Variable count

| Category | Count |
|---|---|
| x (binary) | \|A\| |
| h, ψ⁰ | 2\|A\| |
| λ, ν, t | 3 |
| π̂^s, φ̂^s | S(m + \|A\|) |
| π̃^s, φ̃^s, ỹ^s, ỹ_ts^s | S(m + 2\|A\| + 1) |
| TV obj duals (σ, μ, η) | 4S + 4 |
| TV ν duals (σ, μ, η) | 4\|A\|S + 4\|A\| |
| **Total** | **O(\|A\| · S)** |

### Note on g_s^L, g_s^F

g_s^L and g_s^F are NOT variables — they are shorthand for linear expressions:
- g_s^L = Σ_k c_k^s φ̂_k^s = Σ_k ξ̄_k^s(1 - v_k x_k) φ̂_k^s
- g_s^F = Σ_k c_k^s φ̃_k^s - ỹ_ts^s

Since x is binary, (1 - v_k x_k)φ̂_k^s is bilinear. Use McCormick on x_k φ̂_k^s:
introduce ψ̂_k^s := x_k φ̂_k^s with bounds:
```
0 ≤ ψ̂_k^s ≤ φ^U x_k,   ψ̂_k^s ≤ φ̂_k^s,   ψ̂_k^s ≥ φ̂_k^s - φ^U(1-x_k)    ∀k,s
```
Similarly ψ̃_k^s := x_k φ̃_k^s with same structure.

Then: g_s^L = Σ_k ξ̄_k^s(φ̂_k^s - v_k ψ̂_k^s),  g_s^F = Σ_k ξ̄_k^s(φ̃_k^s - v_k ψ̃_k^s) - ỹ_ts^s

This adds 2|A|S McCormick variables + 6|A|S constraints to the full model.

---

## Benders Decomposition

---

## Level 1: Outer Master Problem (OMP)

### Variables
- x ∈ {0,1}^|A| (binary interdiction)
- h ∈ R₊^|A| (recovery)
- λ ≥ 0 (coupling scalar)
- ψ⁰ ∈ R₊^|A| (McCormick auxiliary for λx)
- t₀ ∈ R (epigraph)

### Formulation

```
min  t₀

s.t. 1ᵀh ≤ w                                         (OMP-1)
     1ᵀx ≤ γ, x ∈ {0,1}^|A|                          (OMP-2)
     0 ≤ ψ_k⁰ ≤ λ^U x_k           ∀k                (OMP-3a)
     ψ_k⁰ ≤ λ                      ∀k                (OMP-3b)
     ψ_k⁰ ≥ λ - λ^U(1-x_k)        ∀k                (OMP-3c)
     t₀ ≥ θ_OMP                    (Benders cuts)     (OMP-4)
```

### Cut generation

After solving OMP, fix (x̄, h̄, λ̄, ψ̄⁰). Solve OSP (= IMP + ISP). Get optimal value Z₀* and dual multipliers w.r.t. (h, λ, ψ⁰). Add optimality cut to OMP.

OMP cut coefficients come from the **dual of the full OSP** evaluated at (x̄, h̄, λ̄, ψ̄⁰). Since OSP is an LP for fixed OMP vars, standard Benders cut:

```
t₀ ≥ Z₀* + π_h ᵀ(h - h̄) + π_λ (λ - λ̄) + π_ψ ᵀ(ψ⁰ - ψ̄⁰)
```

where π_h, π_λ, π_ψ are sensitivities of Z₀ w.r.t. h, λ, ψ⁰.

From the OSP primal, h appears in (P16) RHS = r_k^s = h_k + λ̄ c_k^s, and λ̄ appears in (P9) RHS = λ̄ and in r_k^s. So:

```
π_{h_k} = -∑_s β_k^s           (from P16, sign flip because min)
π_λ    = ∑_s σ̃^s + ∑_s∑_k c_k^s β_k^s    (from P9 + P16)
```

Note: x enters through c_k^s which changes the constraint coefficients, not RHS. So for x we need a different cut structure (integer optimality cuts or Benders with coefficient changes). For simplicity, enumerate x via branch-and-bound and generate cuts in (h, λ, ψ⁰) space.

---

## Level 2: Inner Master Problem (IMP)

This is the OSP decomposed. IMP coordinates α_k.

### Variables
- α_k ≥ 0, ∀k ∈ A (dual of ν)
- θ^L ∈ R (leader value approximation)
- θ^F ∈ R (follower value approximation)

### Formulation

```
max  θ^L + θ^F

s.t. ∑_k α_k ≤ w                                     (IMP-1)
     α_k ≥ 0                      ∀k                  (IMP-2)
     θ^L ≤ (leader Benders cuts)                      (IMP-3)
     θ^F ≤ (follower Benders cuts)                    (IMP-4)
```

### Cut generation

Fix ᾱ_k. Solve ISP-L(ᾱ) and ISP-F(ᾱ). Get optimal values Z^L*, Z^F* and duals w.r.t. α_k.

Leader cut:
```
θ^L ≤ Z^L* + ∑_k (∂Z^L/∂α_k)(α_k - ᾱ_k)
```

Follower cut:
```
θ^F ≤ Z^F* + ∑_k (∂Z^F/∂α_k)(α_k - ᾱ_k)
```

The sensitivities ∂Z/∂α_k come from the ISP dual (see below).

---

## Level 3a: Leader Inner Subproblem ISP-L(α)

Given: α_k (from IMP), c_k^s (from OMP).

### Variables (all s pooled together)
- σ̂^s ≥ 0, ∀s (leader max-flow value per scenario)
- û_k^s ≥ 0, ∀k,s (leader dual flow)
- a_s ≥ 0, b_s ≥ 0, ∀s (TV obj dual)
- a_{s,k}^ν ≥ 0, b_{s,k}^ν ≥ 0, ∀s,k (TV ν-coupling dual)

### Formulation

```
Z^L = max  ∑_s σ̂^s

s.t. N_y û^s + N_ts σ̂^s ≤ 0                  ∀s           (L1)
     -c_k^s a_s + û_k^s - a_{s,k}^ν ≤ 0      ∀k,s         (L2)
     a_s - b_s ≤ q̂_s                          ∀s           (L3)
     a_s + b_s ≥ q̂_s                          ∀s           (L4)
     ∑_s b_s ≤ 2ε̂                                          (L5)
     ∑_s a_s = 1                                            (L6)
     a_{s,k}^ν - b_{s,k}^ν ≤ q̂_s α_k         ∀s,k         (L7)
     a_{s,k}^ν + b_{s,k}^ν ≥ q̂_s α_k         ∀s,k         (L8)
     ∑_s b_{s,k}^ν ≤ 2ε̂ α_k                  ∀k           (L9)
     ∑_s a_{s,k}^ν = α_k                      ∀k           (L10)
     σ̂^s, û_k^s, a_s, b_s, a_{s,k}^ν, b_{s,k}^ν ≥ 0       (L11)
```

### Sensitivity w.r.t. α_k

α_k appears in RHS of (L7), (L8), (L9), (L10). By LP sensitivity:

```
∂Z^L/∂α_k = q̂_s · (dual of L7)_{s,k} - q̂_s · (dual of L8)_{s,k}
             + 2ε̂ · (dual of L9)_k + (dual of L10)_k
```

(summed over s for L7, L8 contributions)

In practice: just read dual values from the LP solver.

---

## Level 3b: Follower Inner Subproblem ISP-F(α)

Given: α_k (from IMP), c_k^s, r_k^s, λ̄ (from OMP).

### Variables (all s pooled together)
- σ̃^s ≥ 0, ∀s (follower max-flow dual per scenario)
- ũ_k^s ≥ 0, ∀k,s (follower dual flow)
- ω^s ≥ 0, ∀s (flow feasibility dual, vector of size m)
- β_k^s ≥ 0, ∀k,s (capacity dual)
- d_s ≥ 0, e_s ≥ 0, ∀s (TV obj dual)
- d_{s,k}^ν ≥ 0, e_{s,k}^ν ≥ 0, ∀s,k (TV ν-coupling dual)

### Formulation

```
Z^F = max  λ̄ ∑_s σ̃^s - ∑_s∑_k r_k^s β_k^s

s.t. N_y ũ^s + N_ts σ̃^s ≤ 0                  ∀s           (F1)
     -c_k^s d_s + ũ_k^s - d_{s,k}^ν ≤ 0      ∀k,s         (F2)
     [N_yᵀ ω^s]_k + β_k^s ≥ 0                ∀k,s         (F3)
     N_tsᵀ ω^s ≥ d_s                          ∀s           (F4)
     d_s - e_s ≤ q̂_s                          ∀s           (F5)
     d_s + e_s ≥ q̂_s                          ∀s           (F6)
     ∑_s e_s ≤ 2ε̃                                          (F7)
     ∑_s d_s = 1                                            (F8)
     d_{s,k}^ν - e_{s,k}^ν ≤ q̂_s α_k         ∀s,k         (F9)
     d_{s,k}^ν + e_{s,k}^ν ≥ q̂_s α_k         ∀s,k         (F10)
     ∑_s e_{s,k}^ν ≤ 2ε̃ α_k                  ∀k           (F11)
     ∑_s d_{s,k}^ν = α_k                      ∀k           (F12)
     σ̃^s, ũ_k^s, ω^s, β_k^s, d_s, e_s, d_{s,k}^ν, e_{s,k}^ν ≥ 0   (F13)
```

### Sensitivity w.r.t. α_k

Same structure as ISP-L: α_k in RHS of (F9)-(F12). Read dual values from solver.

---

## Algorithm

### Outer Loop (OMP ↔ OSP)

```
1. Initialize OMP (no cuts, or with a trivial lower bound)
2. Repeat:
   a. Solve OMP → get (x̄, h̄, λ̄, ψ̄⁰, t₀*)
   b. Compute c_k^s, r_k^s from (x̄, h̄, λ̄)
   c. Solve OSP via inner loop → get Z₀*, dual info
   d. If t₀* ≥ Z₀* - ε_tol: STOP (optimal)
   e. Add Benders cut to OMP: t₀ ≥ Z₀* + sensitivities × (vars - vars̄)
3. Return optimal (x*, h*, λ*)
```

### Inner Loop (IMP ↔ ISP)

```
1. Initialize IMP (no cuts)
2. Repeat:
   a. Solve IMP → get (ᾱ_k, θ^L*, θ^F*)
      - UB_inner = current IMP objective (θ^L* + θ^F*)
   b. Solve ISP-L(ᾱ) → get Z^L*, leader duals
   c. Solve ISP-F(ᾱ) → get Z^F*, follower duals
      - LB_inner = Z^L* + Z^F*
   d. If UB_inner ≤ LB_inner + ε_tol: STOP
   e. Add leader cut to IMP: θ^L ≤ Z^L* + ∑_k (∂Z^L/∂α_k)(α_k - ᾱ_k)
   f. Add follower cut to IMP: θ^F ≤ Z^F* + ∑_k (∂Z^F/∂α_k)(α_k - ᾱ_k)
3. Return Z₀* = Z^L* + Z^F*, and dual info for OMP cut
```

---

## Data Structures

### GridNetworkData (from existing codebase)

```julia
struct GridNetworkData
    num_nodes::Int         # |V|
    num_arcs::Int          # |A|
    N_y::Matrix{Float64}   # (|V|-1) × |A|
    N_ts::Vector{Float64}  # (|V|-1) × 1
    v::Vector{Float64}     # interdiction effectiveness, |A|
    scenarios::Matrix{Float64}  # |A| × S, columns are ξ̄ˢ
    q_hat::Vector{Float64}      # S, nominal probabilities
    gamma::Int             # interdiction budget
    w::Float64             # recovery budget
    lambda_U::Float64      # upper bound on λ
    eps_hat::Float64       # TV radius for leader
    eps_tilde::Float64     # TV radius for follower
end
```

### Model Sizes

For network with |A| arcs, S scenarios:

| Subproblem | Variables | Constraints |
|---|---|---|
| OMP | |A| + |A| + 2 + cuts | 3|A| + 1 + cuts |
| IMP | |A| + 2 | 1 + cuts |
| ISP-L | S(|A|+1) + 2S + 2|A|S = O(|A|S) | S·m + |A|S + 4S + 4|A|S + 2 + 2|A| = O(|A|S) |
| ISP-F | S(|A|+m+2) + 2S + 2|A|S = O(|A|S) | S·m + |A|S + S + 4S + 4|A|S + 2 + 2|A| = O(|A|S) |

All LP. No SDP/SOCP/conic.

---

## Implementation Notes

### 1. Solver
Use HiGHS or Gurobi for all LPs. No conic solver needed (unlike Wasserstein which needs Mosek/Pajarito for SDP).

### 2. Cut management
- Store cuts as (intercept, slope vector) pairs
- Leader and follower cuts are independent
- Purge inactive cuts periodically if iteration count grows

### 3. Warm starting
- IMP: warm start from previous inner loop solution
- ISP: warm start from previous α_k if change is small

### 4. Dual extraction for OMP cuts
After inner loop converges, need sensitivities of Z₀ w.r.t. (h, λ):
- From ISP-F: β_k^s gives ∂Z₀/∂h_k = -∑_s β_k^s (from r_k^s = h_k + λc_k^s)
- From ISP-F: σ̃^s gives ∂Z₀/∂λ = ∑_s σ̃^s + ∑_s∑_k c_k^s β_k^s

These come from the ISP-F optimal solution since h, λ only appear in the follower part.

### 5. Feasibility cuts
If ISP-L or ISP-F is infeasible for given α, add feasibility cut to IMP. In practice this shouldn't happen if α_k ≥ 0 and ∑α_k ≤ w.

### 6. Comparison with Wasserstein codebase
Map to existing code structure:

| Wasserstein module | TV replacement | Key change |
|---|---|---|
| `build_full_model.jl` | `build_full_tv_model.jl` | No LDR vars, no SDP/SOC, add TV dual vars |
| `build_uncertainty_set.jl` | `build_tv_ambiguity.jl` | TV constraints instead of SOC uncertainty set |
| `build_dualized_outer_subprob.jl` | `build_tv_osp.jl` | LP instead of SDP |
| `strict_benders.jl` | `tv_benders.jl` | Same structure, LP subproblems |
| `nested_benders.jl` | `tv_nested_benders.jl` | Same structure, LP subproblems |

### 7. Verifying correctness
- Solve full TV model (Section 5 of tex) directly for small instances
- Compare Z₀ with nested Benders Z₀ — must match
- Compare with Wasserstein solution at ε→0 (should approach nominal SP)

---

## File Structure

```
src/
├── tv_network_generator.jl      # GridNetworkData + scenario generation
├── tv_build_full_model.jl        # Full TV reformulation (T1)-(T18), for verification
├── tv_build_omp.jl               # OMP formulation
├── tv_build_isp_leader.jl        # ISP-L: (L1)-(L11)
├── tv_build_isp_follower.jl      # ISP-F: (F1)-(F13)
├── tv_nested_benders.jl          # Main algorithm: outer + inner loop
├── tv_utils.jl                   # Cut management, dual extraction
└── tv_run.jl                     # Entry point, parameter sweep
```
