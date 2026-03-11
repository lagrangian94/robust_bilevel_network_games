# Compact LDR Matrix Refactoring Analysis

## 1. What is an LDR (Linear Decision Rule) in This Codebase?

In this robust bilevel network interdiction model, uncertain arc capacities are modeled as random parameters. Wait-and-see (recourse) decisions -- flows, dual variables, etc. -- are parameterized as **affine functions of the uncertainty**:

```
phi_hat_k(xi) = sum_{l=1}^{|A|} phi_hat_{k,l} * xi_l + phi_hat_{k,0}
```

This is compactly written as a matrix-vector product:

```
phi_hat(xi) = Phi_hat_L * xi_L + phi_hat_0 = Phi_hat * [xi; 1]
```

where `Phi_hat` is an `|A| x (|A|+1)` matrix. The last column holds intercept terms; the first `|A|` columns (`Phi_hat_L`) hold slopes with respect to each uncertainty dimension `xi_l`.

The full set of LDR matrices in this model:

| Variable | Dimension | Role |
|---|---|---|
| `Phi_hat[s]` | `|A| x (|A|+1)` | Leader's flow dual LDR coefficient |
| `Psi_hat[s]` | `|A| x (|A|+1)` | Leader's interdiction LDR (McCormick of Phi*x) |
| `Pi_hat[s]` | `(|V|-1) x (|A|+1)` | Leader's node price LDR coefficient |
| `Phi_tilde[s]` | `|A| x (|A|+1)` | Follower's flow dual LDR coefficient |
| `Psi_tilde[s]` | `|A| x (|A|+1)` | Follower's interdiction LDR |
| `Pi_tilde[s]` | `(|V|-1) x (|A|+1)` | Follower's node price LDR |
| `Y_tilde[s]` | `|A| x (|A|+1)` | Follower's flow variable LDR |
| `Yts_tilde[s]` | `1 x (|A|+1)` | Follower's dummy arc flow LDR |

**Key insight**: In the current "full" LDR, arc k's decision rule coefficient `Phi_hat[k,l]` depends on the uncertainty of **every** arc l. But physically, arc k's recourse should only depend on uncertainty from **adjacent arcs** (arcs sharing a common node). Most entries in `Phi_hat_L` should therefore be zero.

## 2. Current Implementation: Two Parallel Approaches

### 2.1 Approach 1: Primal Side -- Create Full Variables Then Fix to Zero

**File**: `build_full_model.jl`

LDR variables are created as **full** `|A| x (|A|+1)` matrices at lines 127-141:

```
Line 127: @variable(model, Phi_hat[s=1:S, 1:num_arcs, 1:num_arcs+1], ...)
Line 128: @variable(model, Psi_hat[s=1:S, 1:num_arcs, 1:num_arcs+1], ...)
Line 129: @variable(model, Phi_tilde[s=1:S, 1:num_arcs, 1:num_arcs+1], ...)
Line 130: @variable(model, Psi_tilde[s=1:S, 1:num_arcs, 1:num_arcs+1], ...)
Line 134: @variable(model, Pi_hat[s=1:S, 1:num_nodes-1, 1:num_arcs+1], ...)
Line 135: @variable(model, Pi_tilde[s=1:S, 1:num_nodes-1, 1:num_arcs+1], ...)
Line 138: @variable(model, Y_tilde[s=1:S, 1:num_arcs, 1:num_arcs+1], ...)
Line 141: @variable(model, Yts_tilde[s=1:S, 1, 1:num_arcs+1], ...)
```

Then the function `add_sparsity_constraints!()` (lines 413-504) **fixes non-adjacent entries to zero** by adding equality constraints:

```julia
# Lines 459-473: Arc-to-arc sparsity for Phi, Psi, Y
for s in 1:S, i in 1:num_arcs, j in 1:num_arcs
    if !network.arc_adjacency[i,j]
        @constraint(model, Phi_hat_L[s,i,j] == 0)      # Line 462
        @constraint(model, Phi_tilde_L[s,i,j] == 0)     # Line 464
        @constraint(model, Psi_hat_L[s,i,j] == 0)       # Line 466
        @constraint(model, Psi_tilde_L[s,i,j] == 0)     # Line 468
        @constraint(model, Y_tilde_L[s,i,j] == 0)       # Line 470
    end
end

# Lines 480-489: Node-to-arc sparsity for Pi
for s in 1:S, i in 1:num_nodes-1, j in 1:num_arcs
    if !network.node_arc_incidence[i,j]
        @constraint(model, Pi_hat_L[s,i,j] == 0)        # Line 483
        @constraint(model, Pi_tilde_L[s,i,j] == 0)      # Line 485
    end
end
```

**Problem**: This creates many unnecessary variables and then adds many "== 0" constraints. For a 5x5 grid with ~40 arcs, this means creating 40x40 = 1600 entries per matrix per scenario, then fixing ~80% of them to 0 with equality constraints. The SDP matrix `M_hat` and `M_tilde` remain `(|A|+1) x (|A|+1)` in size.

The SDP matrices are created at lines 210-211:
```
Line 210: @variable(model, Mhat[1:S, 1:num_arcs+1, 1:num_arcs+1])
Line 211: @variable(model, Mtilde[1:S, 1:num_arcs+1, 1:num_arcs+1])
```

And the PSD cone constraints at lines 233-234:
```
Line 233: @constraint(model, [s=1:S], Mhat[s,:,:] in PSDCone())
Line 234: @constraint(model, [s=1:S], Mtilde[s,:,:] in PSDCone())
```

### 2.2 Approach 2: Dual Side -- Selective Constraint Creation

**File**: `build_dualized_outer_subprob.jl`

In the dual formulation (outer subproblem), there are NO explicit LDR variables. Instead, the dual variables (M_hat, M_tilde, Z, beta, U, P, Gamma) are created at full dimension. The adjacency filtering happens at the **constraint creation** stage -- constraints are only created for adjacent arc pairs:

```julia
# Lines 217-221: Phi_hat_L constraint (only for adjacent arcs)
for i in 1:num_arcs, j in 1:num_arcs
    if network.arc_adjacency[i,j]              # <-- FILTER
        @constraint(model, lhs_L[i,j] == 0)
    end
end

# Lines 233-237: Psi_hat_L constraint (only for adjacent arcs)
for i in 1:num_arcs, j in 1:num_arcs
    if network.arc_adjacency[i,j]              # <-- FILTER
        @constraint(model, lhs_L[i,j] <= 0)
    end
end

# Lines 250-254: Phi_tilde_L constraint
for i in 1:num_arcs, j in 1:num_arcs
    if network.arc_adjacency[i,j]              # <-- FILTER
        @constraint(model, lhs_L[i,j] == 0)
    end
end

# Lines 265-270: Psi_tilde_L constraint
for i in 1:num_arcs, j in 1:num_arcs
    if network.arc_adjacency[i,j]              # <-- FILTER
        @constraint(model, lhs_L[i,j] <= 0.0)
    end
end

# Lines 288-292: Pi_hat_L constraint (only for incident node-arc pairs)
for i in 1:(num_nodes-1), j in 1:num_arcs
    if network.node_arc_incidence[i,j]         # <-- FILTER
        @constraint(model, [s=1:S], ... == 0.0)
    end
end

# Lines 298-302: Pi_tilde_L constraint
for i in 1:(num_nodes-1), j in 1:num_arcs
    if network.node_arc_incidence[i,j]         # <-- FILTER
        @constraint(model, [s=1:S], ... == 0.0)
    end
end

# Lines 308-312: Y_tilde_L constraint
for i in 1:num_arcs, j in 1:num_arcs
    if network.arc_adjacency[i,j]              # <-- FILTER
        @constraint(model, [s=1:S], ... == 0.0)
    end
end
```

The SDP matrices are still `(|A|+1) x (|A|+1)`:
```
Line 103: @variable(model, Mhat[s=1:S,1:num_arcs+1,1:num_arcs+1])
Line 104: @variable(model, Mtilde[s=1:S,1:num_arcs+1,1:num_arcs+1])
```

**The same pattern is replicated in**:
- `nested_benders.jl`, function `build_isp_leader()` (lines ~574-606 for leader, ~771-821 for follower)
- `nested_benders_trust_region.jl`, function `build_isp_leader()` (lines ~952-985 for leader, ~1149-1200 for follower)

## 3. Where Adjacency Data Comes From

**File**: `network_generator.jl`

The `GridNetworkData` struct (line 26) contains:
- `arc_adjacency::Matrix{Bool}` -- `|A| x |A|`, true if arcs i and j share a node (line 23)
- `node_arc_incidence::Matrix{Bool}` -- `(|V|-1) x |A|`, true if node i is incident to arc j (line 24)

These are computed by:
- `generate_arc_adjacency()` (lines 61-86): Two arcs are adjacent if they share ANY common node
- `generate_node_arc_incidence()` (lines 88-118): A node is incident to an arc if it's the arc's head or tail

## 4. SDP Matrix Dimension Analysis

Currently, the SDP matrices `M_hat[s]` and `M_tilde[s]` are `(|A|+1) x (|A|+1)`. This comes from the S-lemma reformulation of the worst-case expectation:

```
M_hat = | theta*I - D'*(Phi_L - v*Psi_L)       -(1/2)*[...]  |
        | -(1/2)*[...]'                     eta - ... - theta*eps^2 |
```

The top-left block `M_hat_11` is `|A| x |A|` because `Phi_L` and `Psi_L` are `|A| x |A|`. The "+1" dimension comes from the intercept/constant term.

**Why this matters for compact LDR**: If we make `Phi_L` sparse (only adjacent entries), then `D' * Phi_L` is also sparse. But the SDP constraint requires the **full** `(|A|+1) x (|A|+1)` matrix to be PSD, so the SDP dimension does NOT change just from sparsifying the LDR. However, by reducing the number of nonzero entries in `Phi_L`, we reduce:
1. The number of decision variables
2. The number of Big-M constraints (14j, 14k)
3. The number of dual constraints in the outer subproblem

## 5. Files and Exact Locations Requiring Modification

### 5.1 `build_full_model.jl` (Primal Full Model)

| Lines | Current Code | Proposed Change |
|---|---|---|
| 127-131 | Full `Phi_hat[s,1:num_arcs,1:num_arcs+1]` etc. | Create **compact** variables: for each arc k, only create columns for `adj_arcs(k)` plus intercept |
| 134-135 | Full `Pi_hat[s,1:num_nodes-1,1:num_arcs+1]` etc. | Create compact: for each node i, only columns for incident arcs |
| 138-141 | Full `Y_tilde`, `Yts_tilde` | Similar compaction |
| 145-150 | Slicing `_L` and `_0` parts | Need new indexing scheme for compact representation |
| 210-211 | `Mhat[1:S, 1:num_arcs+1, 1:num_arcs+1]` | **Potentially reducible** if we reformulate SDP with only active columns, but this requires deeper mathematical analysis |
| 214-231 | SDP matrix construction using `Phi_hat_L[s,:,:]` | Must use compact-to-full expansion or reformulate products |
| 250-264 | Big-M constraints over all `(i,j)` pairs | Only iterate over adjacent pairs |
| 295-348 | Dual constraints using matrix products like `N' * Pi_hat_L` | Must handle compact indexing |
| 413-504 | `add_sparsity_constraints!()` function | **Eliminate entirely** -- sparsity is built into the compact representation |

### 5.2 `build_dualized_outer_subprob.jl` (Dual Outer Subproblem)

| Lines | Current Code | Proposed Change |
|---|---|---|
| 103-104 | `Mhat[s,1:num_arcs+1,1:num_arcs+1]` | SDP dimension may remain same, but related dual variables can be compacted |
| 105-110 | `Uhat1..3`, `Utilde1..3` all `|A| x (|A|+1)` | Compact to only adjacent entries |
| 146-157 | `Phat1_Phi..Ptilde2_Yts` | Compact dimensions |
| 213-221 | `Phi_hat_L` constraint with `arc_adjacency` filter | Replace filter loop with iteration over compact indices |
| 233-237 | `Psi_hat_L` constraint with `arc_adjacency` filter | Same |
| 248-254 | `Phi_tilde_L` constraint | Same |
| 265-270 | `Psi_tilde_L` constraint | Same |
| 288-292 | `Pi_hat_L` constraint with `node_arc_incidence` filter | Same |
| 298-302 | `Pi_tilde_L` constraint | Same |
| 308-312 | `Y_tilde_L` constraint | Same |

### 5.3 `nested_benders.jl` (Nested Benders Inner Subproblems)

The functions `build_isp_leader()` and `build_isp_follower()` replicate the dual outer subproblem structure. Changes mirror Section 5.2 above.

**Leader** (`build_isp_leader`):
- Lines ~502: `Mhat` at `(|A|+1) x (|A|+1)`
- Lines ~503-505: `Uhat1..3` at full dimension
- Lines ~522-525: `Phat1_Phi..Phat2_Pi` at full dimension
- Lines ~570-606: Constraint loops with `arc_adjacency`/`node_arc_incidence` filter

**Follower** (`build_isp_follower`):
- Lines ~688: `Mtilde` at `(|A|+1) x (|A|+1)`
- Lines ~689-691: `Utilde1..3` at full dimension
- Lines ~715-722: `Ptilde1_Phi..Ptilde2_Yts` at full dimension
- Lines ~768-821: Constraint loops with adjacency/incidence filter

### 5.4 `nested_benders_trust_region.jl`

Same structure as `nested_benders.jl` with trust region additions. The leader inner subproblem and follower inner subproblem have the same adjacency-filtered constraint patterns.

**Leader**: Lines ~937-985
**Follower**: Lines ~1139-1200

### 5.5 `strict_benders.jl` (Strict Benders)

The outer master problem (`build_omp`) at lines 17-72 does NOT contain LDR variables. The outer subproblem is built via `build_dualized_outer_subprob.jl` (already covered).

The cut generation in `osp_optimize!()` (lines 74-135) and `strict_benders_optimize!()` (lines 145-263) reference solution values from the dual outer subproblem. The cut coefficients (`Uhat1, Utilde1, Uhat3, Utilde3, beta, Z`) would need to be interpreted under compact indexing.

### 5.6 `network_generator.jl`

No changes needed. The `arc_adjacency` and `node_arc_incidence` matrices are already computed and would be **consumed** by the new compact variable creation logic.

## 6. Proposed Compact LDR Approach

### 6.1 Core Idea

For each arc k, define `adj(k) = {l : arc_adjacency[k,l] == true}` (the set of arcs adjacent to arc k). Instead of creating `Phi_hat[s, k, 1:|A|+1]`, create `Phi_hat_compact[s, k, 1:|adj(k)|+1]`.

Similarly, for each node i, define `inc(i) = {j : node_arc_incidence[i,j] == true}`. Instead of `Pi_hat[s, i, 1:|A|+1]`, create `Pi_hat_compact[s, i, 1:|inc(i)|+1]`.

### 6.2 Data Structure

Create a mapping from compact indices to global arc indices:

```julia
# For each arc k, adj_map[k] = sorted list of adjacent arc indices
adj_map = [findall(network.arc_adjacency[k,:]) for k in 1:num_arcs]

# For each node i, inc_map[i] = sorted list of incident arc indices
inc_map = [findall(network.node_arc_incidence[i,:]) for i in 1:num_nodes-1]
```

### 6.3 Variable Creation

Since JuMP requires rectangular arrays, the compact approach would use one of:
1. **Dictionary-based variables**: `@variable(model, Phi_hat[s=1:S, (k,l) in adj_pairs])` where `adj_pairs = [(k,l) for k in 1:num_arcs for l in adj_map[k]]`
2. **Sparse matrix reconstruction**: Create variables only for nonzero positions, then assemble into sparse matrices when needed for constraint generation
3. **Per-row variable groups**: For each row k, create a small dense vector of length `|adj(k)|+1`

Option 1 (dictionary-indexed JuMP variables) is likely the cleanest.

### 6.4 SDP Matrix Impact

The SDP matrices `M_hat` and `M_tilde` are `(|A|+1) x (|A|+1)`. Their 11-block is defined as:

```
M_hat_11 = theta*I - D' * (Phi_hat_L - v * Psi_hat_L)
```

With compact LDR, `Phi_hat_L` is sparse. The product `D' * Phi_hat_L` is also sparse but still an `|A| x |A|` matrix. The SDP matrix dimension **does not shrink** directly. However:

- The **number of free variables** in M_hat_11 could be reduced if we note that `M_hat_11 = theta*I - (sparse matrix)`, making M_hat_11 a "diagonal + sparse" matrix. The solver may exploit this structure.
- A deeper reformulation could decompose the SDP using the sparsity pattern (chordal decomposition), but that is a separate research direction.

### 6.5 Variable Count Reduction Estimate

For a 3x3 grid (~17 arcs), the arc adjacency is ~50% dense. For a 5x5 grid (~40 arcs), it drops to ~20-30% dense.

Per scenario, current variable count for LDR matrices:
- `Phi_hat`: |A| * (|A|+1) = 40*41 = 1640
- `Psi_hat`: same
- `Phi_tilde`, `Psi_tilde`, `Y_tilde`: 3 * 1640 = 4920
- `Pi_hat`, `Pi_tilde`: 2 * (|V|-1) * (|A|+1) ~ 2 * 26 * 41 = 2132
- Total per scenario: ~10,332

With compact LDR at 25% density for the _L part:
- `Phi_hat_compact`: |A| * (0.25*|A| + 1) = 40 * 11 = 440
- Similar savings for other matrices
- Estimated total per scenario: ~3,000 (roughly 70% reduction)

The Big-M constraints (14j, 14k) iterate over `S * |A| * (|A|+1)` which is massive. With compaction, this drops proportionally.

## 7. Risks and Considerations

### 7.1 Mathematical Correctness
The compact LDR is mathematically equivalent to the full LDR with zero-fixing, provided:
- The "zero" entries are correctly excluded from all constraint formulations
- Matrix products (e.g., `N' * Pi_hat_L`, `I_0' * Phi_hat_L`) correctly reconstruct the sparse result

### 7.2 SDP Reformulation Complexity
The SDP matrix `M_hat` is defined in terms of `D' * (Phi_L - v*Psi_L)`. With compact variables, this product must be assembled from sparse data. This is straightforward but requires careful index bookkeeping.

### 7.3 Cut Coefficient Extraction (Benders)
In the Benders decomposition, cut coefficients are extracted from dual solutions. The compact representation changes which variables exist, so cut generation code must be updated to:
- Only extract coefficients for adjacent pairs
- Correctly reconstruct the cut expression in the master problem

### 7.4 Dual Outer Subproblem Variables
The dual variables `U_hat[s, k, l]`, `P_hat[s, k, l]` in the outer subproblem are dual to primal LDR constraints. If the primal constraint for `(k,l)` does not exist (non-adjacent), the dual variable should not exist either. Currently the code handles this by only creating the constraint for adjacent pairs, leaving the corresponding dual variable "free" (unconstrained). With compact LDR, these dual variables should also be compacted.

### 7.5 Testing Strategy
- Verify on a small instance (3x3 grid, S=1-2) that the compact formulation gives the same optimal objective as the full formulation with `add_sparsity_constraints!()`.
- Compare variable and constraint counts before/after.
- Benchmark solve time on the 5x5 grid instance that currently has numerical convergence issues.

### 7.6 Incremental vs. Big-Bang Approach
Given the number of files affected (4-5 files, each with multiple locations), a phased approach is recommended:
1. **Phase 1**: Refactor `build_full_model.jl` to use compact LDR; remove `add_sparsity_constraints!()`; validate
2. **Phase 2**: Refactor `build_dualized_outer_subprob.jl`; validate that strict Benders still works
3. **Phase 3**: Refactor `nested_benders.jl` inner subproblems (leader + follower)
4. **Phase 4**: Refactor `nested_benders_trust_region.jl`

## 8. Summary of All Code Locations

### build_full_model.jl
- **LDR variable creation**: Lines 127-141
- **LDR _L and _0 slicing**: Lines 145-150
- **SDP matrix creation**: Lines 210-211
- **SDP PSD constraints**: Lines 233-234
- **SDP block definitions** (M_hat_11, M_hat_12, M_hat_22): Lines 214-231
- **Big-M constraints (Psi <-> Phi <-> x)**: Lines 250-264
- **Dual feasibility constraints using matrix products**: Lines 295-348
- **Sparsity fix-to-zero function**: Lines 413-504 (REMOVE ENTIRELY)

### build_dualized_outer_subprob.jl
- **SDP matrix creation**: Lines 103-104
- **U variable creation (full dim)**: Lines 105-110
- **P variable creation (full dim)**: Lines 146-157
- **Phi_hat_L dual constraint (adjacency-filtered)**: Lines 217-221
- **Psi_hat_L dual constraint (adjacency-filtered)**: Lines 233-237
- **Phi_tilde_L dual constraint (adjacency-filtered)**: Lines 250-254
- **Psi_tilde_L dual constraint (adjacency-filtered)**: Lines 265-270
- **Pi_hat_L dual constraint (incidence-filtered)**: Lines 288-292
- **Pi_tilde_L dual constraint (incidence-filtered)**: Lines 298-302
- **Y_tilde_L dual constraint (adjacency-filtered)**: Lines 308-312

### nested_benders.jl
- **Leader ISP**: `build_isp_leader()` function
  - SDP matrix: Line ~502
  - U variables: Lines ~503-505
  - P variables: Lines ~522-525
  - Phi_hat_L constraint (filtered): Lines ~574-578
  - Psi_hat_L constraint (filtered): Lines ~590-594
  - Pi_hat_L constraint (filtered): Lines ~602-606
  - Phi_tilde_L constraint (filtered): Lines ~771-775
  - Psi_tilde_L constraint (filtered): Lines ~787-791
  - Pi_tilde_L constraint (filtered): Lines ~807-811
  - Y_tilde_L constraint (filtered): Lines ~817-821

- **Follower ISP**: `build_isp_follower()` function
  - Mtilde: Line ~688
  - Utilde: Lines ~689-691
  - Ptilde: Lines ~715-722
  - Same adjacency-filtered constraint pattern as leader

### nested_benders_trust_region.jl
- Same structure as `nested_benders.jl`, offset by ~370 lines
  - Leader: Lines ~937-985
  - Follower: Lines ~1139-1200

### strict_benders.jl
- **No LDR variables** in outer master problem
- Cut generation references dual solution values: Lines 95-128, 220-226
- Cut generation would need updated indexing for compact coefficients

### network_generator.jl
- **No changes needed** -- provides `arc_adjacency` and `node_arc_incidence` data

---

## 9. 1차 시도: fix() 접근법 (실패)

### 9.1 구현 내용

`fix(var, 0.0; force=true)`를 사용하여 비인접 LDR 항목을 고정하는 방식으로 4개 파일 생성:

| File | Function |
|------|----------|
| `compact_ldr_utils.jl` | `apply_primal_ldr_sparsity!()`, `apply_dual_ldr_sparsity!()` |
| `build_full_model_compact.jl` | `build_full_2DRNDP_model_compact()` |
| `build_dualized_outer_subprob_compact.jl` | `build_dualized_outer_subproblem_compact()` |
| `build_isp_compact.jl` | `build_isp_leader_compact()`, `build_isp_follower_compact()`, `initialize_isp_compact()` |

### 9.2 왜 실패인가

`fix()`는 **변수를 0으로 고정할 뿐, 변수 자체는 여전히 JuMP 모델에 존재**한다.
- JuMP 레벨: 변수 수 동일, 메모리 동일
- Solver presolve가 제거해줄 *수도* 있지만 보장 없음
- SDP 행렬 `Mhat`, `Mtilde`: 여전히 `(|A|+1) × (|A|+1)`
- 본질적으로 `add_sparsity_constraints!()` (== 0 제약조건) 과 같은 수준

**결론: `fix()` ≈ `add_sparsity_constraints!()`. 진정한 차원 축소 아님.**

### 9.3 실험 결과: Original vs fix() 비교 (compare_compact.jl)

3×3 grid, S=1,2,5, TR Nested Benders (inner TR only, outer TR off) 조건에서 실행.

| S | Original obj | Compact obj | Obj Diff | Orig Time | Compact Time | Speedup | Inner iters |
|---|---|---|---|---|---|---|---|
| 1 | 5.071268 | 5.071268 | **0.0** | 2.3s | 3.27s | 0.704x | [5,3,3,3,4,4,4,4,4,4,4] (동일) |
| 2 | 5.743457 | 5.743457 | **0.0** | 5.05s | 5.62s | 0.897x | [9,3,3,3,4,4,4,5,4,4,4] (동일) |
| 5 | 5.221763 | 5.221763 | **0.0** | 14.97s | 15.5s | 0.966x | [8,4,4,3,10,4,4,4,4,4,4,4] (동일) |

fix() 적용 변수 수: S=1 → 2,932개, S=2 → 5,864개, S=5 → 14,660개

**관찰:**
- 목적함수 값이 **모든 S에서 완전히 동일** → fix()가 수학적으로 정확함을 확인
- Iteration 패턴도 **모든 S에서 100% 동일** → solver가 동일한 경로로 수렴
- **시간이 일관되게 증가**: S=1에서 42% 증가, S=2에서 11%, S=5에서 3.5% → fix() 호출 오버헤드가 고정 비용이므로 S가 커질수록 비율은 줄어듦
- 성능 개선 효과 없음

**결론:** fix() 접근법은 정확성은 보장하지만 **성능 이득이 전혀 없고 오히려 overhead가 발생**한다.

**참고 (초기 실행 버그):** 최초 실행에서 for loop 내 `S = test_S`가 Julia soft scope 규칙에 의해 local 변수로 처리되어 global S가 warm-up 값(1)으로 유지됨. 이로 인해 S=2에서 수렴 실패처럼 보였으나, `global S = test_S`로 수정 후 정상 동작 확인.

### 9.4 교훈

- `fix()` 접근법은 코드 변경이 최소화되는 장점이 있으나, compact LDR의 핵심 목표인 **변수 수 자체의 실질적 감소**를 달성하지 못함
- 기존 algorithm 함수와 100% 호환되고 정확성은 확인되었으나, 성능 개선 효과 없음 (오히려 악화)
- vars dict 구조가 동일 → 기존 코드와 호환성은 확보됨 (이 부분은 2차 시도에서도 유지하면 좋음)

---

## 10. 2차 시도 계획: Dictionary-Indexed 진짜 Compact 변수

### 10.1 핵심 아이디어

비인접 항목의 **변수를 아예 생성하지 않는다**. JuMP의 dictionary-indexed variable 사용:

```julia
# 인접 쌍만 정의
arc_adj_pairs = [(i,j) for i in 1:num_arcs for j in 1:num_arcs if network.arc_adjacency[i,j]]
# intercept 열 포함
arc_full_pairs = vcat(arc_adj_pairs, [(i, num_arcs+1) for i in 1:num_arcs])

# 변수 생성: 인접 쌍에 대해서만
@variable(model, -ϕU <= Φhat[s=1:S, (i,j) in arc_full_pairs] <= ϕU)
```

접근 방식: `Φhat[(s, i, j)]` 으로 접근. non-adjacent `(i,j)`에 대한 변수는 **존재하지 않음**.

### 10.2 변경이 필요한 연산들

dictionary-indexed 변수는 행렬 슬라이싱 (`Φhat[s,:,:]`)이 불가능하므로, 행렬 연산을 수동으로 재구성해야 함.

#### (A) SDP 블록 구성: `D' * Φ_L`

현재 코드:
```julia
Mhat_11 .== ϑhat[s]*I - D_s'*(Φhat_L[s,:,:] - v*Ψhat_L[s,:,:])
```

D_s = diagm(ξ̄) 이므로 `(D_s' * Φ_L)[i,j] = ξ̄[i] * Φ_L[i,j]`. Compact 버전:
```julia
# Mhat_11[i,j] = ϑ*δ_{ij} - ξ̄[i]*(Φ_L[i,j] - v*Ψ_L[i,j])
for i in 1:num_arcs, j in 1:num_arcs
    if network.arc_adjacency[i,j]
        @constraint(model, Mhat_11[i,j] == ϑhat[s]*(i==j ? 1.0 : 0.0)
            - xi_bar[s][i] * (Φhat[(s,i,j)] - v*Ψhat[(s,i,j)]))
    else
        @constraint(model, Mhat_11[i,j] == ϑhat[s]*(i==j ? 1.0 : 0.0))
    end
end
```

**참고**: `arc_adjacency[i,i]` = true (자기 자신은 항상 인접) → 대각 항목은 항상 포함.

#### (B) SDP 12-블록: `(Φ_L - v*Ψ_L)*ξ̄ + D_s'*(Φ_0 - v*Ψ_0)`

`(Φ_L * ξ̄)[i] = Σ_j Φ_L[i,j] * ξ̄[j]` → 인접한 j에 대해서만 합산:
```julia
for i in 1:num_arcs
    adj_j = findall(network.arc_adjacency[i,:])
    term1 = sum((Φhat[(s,i,j)] - v*Ψhat[(s,i,j)]) * xi_bar[s][j] for j in adj_j)
    term2 = xi_bar[s][i] * (Φhat_0[(s,i)] - v*Ψhat_0[(s,i)])
    @constraint(model, Mhat_12[i] == -(1/2)*(term1 + term2))
end
```

#### (C) Big-M 제약조건

기존 `(i, j) in 1:num_arcs × 1:num_arcs+1` → compact `(i,j) in arc_full_pairs`:
```julia
for s in 1:S, (i,j) in arc_full_pairs
    @constraint(model, Ψhat[(s,i,j)] <= ϕU * x[i])
    @constraint(model, Ψhat[(s,i,j)] - Φhat[(s,i,j)] <= 0)
    @constraint(model, Φhat[(s,i,j)] - Ψhat[(s,i,j)] <= ϕU * (1 - x[i]))
end
```

#### (D) Dual 제약조건의 행렬곱: `N' * Π_L`, `I_0' * Φ_L`

`(N' * Π_L)[j, l]` = Σ_i N[i,j] * Π_L[i,l]`. Π_L은 node-arc 인접에 대해서만 존재:
```julia
# Q_hat[j,l] = (N' * Π_L + I_0' * Φ_L)[j,l]
for j in 1:num_arcs+1, l in 1:num_arcs
    NtPi = sum(N[i,j] * Πhat[(s,i,l)] for i in 1:num_nodes-1 if network.node_arc_incidence[i,l])
    if j <= num_arcs && network.arc_adjacency[j,l]
        I0tPhi = Φhat[(s,j,l)]
    elseif j <= num_arcs
        I0tPhi = 0.0
    else
        I0tPhi = 0.0
    end
    Q_hat[j,l] = NtPi + I0tPhi
end
```

#### (E) Dual 모델의 U, P 변수

Primal의 Big-M이 compact → dual의 U도 compact:
```julia
@variable(model, Uhat1[s=1:S, (i,j) in arc_full_pairs] >= 0)
```

Objective에서 `sum(Uhat1 .* diag_x_E)` → compact 인덱스로 합산:
```julia
obj_term1 = [-ϕU * sum(Uhat1[(s,i,j)] * x[i] for (i,j) in arc_full_pairs) for s=1:S]
obj_term2 = [-ϕU * sum(Uhat3[(s,i,j)] * (1-x[i]) for (i,j) in arc_full_pairs) for s=1:S]
```

### 10.3 변경 불가능한 부분: SDP 차원

SDP 행렬 `Mhat[s]`, `Mtilde[s]`는 여전히 `(|A|+1) × (|A|+1)`.

이유: S-lemma 기반 reformulation 구조상, worst-case expectation의 robust counterpart가 `(|A|+1) × (|A|+1)` PSD cone constraint를 요구. LDR 계수를 sparse하게 만들어도 `Mhat_11 = ϑI - D'(Φ_L - vΨ_L)` 에서 `D'Φ_L`이 sparse해지지만, `ϑI` 항이 full diagonal이므로 Mhat 자체는 dense.

**SDP 차원 축소를 위해서는**:
- Chordal decomposition (SDP의 sparsity pattern을 exploit)
- 또는 reformulation 자체를 바꾸는 연구 (별도 방향)

### 10.4 실질적 절감 효과 (SDP 제외)

5×5 grid 기준 (|A| ≈ 40, arc adjacency density ≈ 25%):

| 항목 | Full | Compact | 절감 |
|------|------|---------|------|
| Φ_L 변수 (per matrix, per scenario) | 40×40 = 1600 | ~400 | **75%** |
| Big-M 제약 (per scenario) | 6×40×41 = 9840 | 6×(400+40) = 2640 | **73%** |
| U dual 변수 (per matrix, per scenario) | 40×41 = 1640 | ~440 | **73%** |
| Π_L 변수 (per matrix, per scenario) | 26×40 = 1040 | ~130 | **87%** |
| **SDP Mhat** | **(41×41)** | **(41×41)** | **0%** |

### 10.5 구현 전략

1차 시도의 4개 파일을 **덮어쓰기**로 dictionary-indexed 방식으로 재구현.

helper 함수가 필요한 곳:
- `sparse_mat_vec(compact_var, s, adj_map, vec, num_rows)` — Φ_L * ξ̄ 계산
- `sparse_quadform(compact_var, s, adj_map, D, num_arcs)` — D' * Φ_L 계산
- `sparse_NtPi(Πhat, s, N, inc_map, num_arcs)` — N' * Π_L 계산

Phase:
1. `compact_ldr_utils.jl` — index set 생성 + sparse 행렬 연산 helper
2. `build_full_model_compact.jl` — primal model (SDP 블록 재구성 필요)
3. `build_dualized_outer_subprob_compact.jl` — dual model (U/P compact + objective 재구성)
4. `build_isp_compact.jl` — nested Benders ISP (Phase 3과 동일 패턴)
5. 검증: 소규모 인스턴스에서 원본과 동일한 목적함수 값 확인

---

## 11. 2차 시도: Dictionary-Indexed Compact 구현 (ISP만)

### 11.1 구현 범위

OMP, IMP에는 LDR 변수가 없으므로 **ISP (Inner SubProblem)만 compact화**하면 TR Nested Benders 비교가 가능하다. 원본 파일은 수정하지 않고, 별도 파일에 compact 버전을 구현한다.

### 11.2 구현 파일

| 파일 | 내용 |
|------|------|
| `compact_ldr_utils.jl` | 인덱스 셋 생성기 (arc_adj_pairs, arc_full_pairs, node_arc_inc_pairs, node_arc_full_pairs) + compact → dense 변환 헬퍼 |
| `build_isp_compact.jl` | 6개 함수: build_isp_leader_compact, build_isp_follower_compact, isp_leader_optimize_compact!, isp_follower_optimize_compact!, evaluate_master_opt_cut_compact, initialize_isp_compact |
| `compare_compact.jl` | @eval Main으로 4개 함수 교체 후 Original vs Compact 비교 (S=1,2,5) |

### 11.3 핵심 설계 결정

#### (A) Compact vs Dense 변수 분류

**Compact (dict-indexed, 인접 쌍에 대해서만 생성):**
- U (Uhat1~3, Utilde1~3): Big-M dual 변수 — `@variable(model, Uhat1[(i,j) in afp] >= 0)`
- P_Φ (Phat1_Φ, Phat2_Φ 등): Φ̂ 제약 LDR 희소성 dual
- P_Π (Phat1_Π, Phat2_Π 등): Π̂ 제약 LDR 희소성 dual (node-arc incidence 기반)
- P_Y (Ptilde1_Y, Ptilde2_Y): Ỹ 제약 LDR 희소성 dual (follower만)

**Dense (축소 불가):**
- M (Mhat, Mtilde): PSD cone, (na1 × na1), S-lemma 구조상 축소 불가
- Z, β, Γ: 블록 구조, LDR 희소성과 무관
- Ptilde_Yts: 1D per scenario, arc-pair 희소성 없음

#### (B) ISP의 S=1 특성

ISP는 `initialize_isp`에서 시나리오별로 1개씩 생성되므로 항상 S=1. 따라서 compact 변수에 s 인덱스가 불필요하다:
```julia
# Dense: s 인덱스 유지
@variable(model, Mhat[s=1:S, 1:na1, 1:na1])
# Compact: s 인덱스 없음
@variable(model, Uhat1[(i,j) in afp] >= 0)
```

#### (C) 제약조건 분리 전략

원본에서 `for i in 1:num_arcs, j in 1:num_arcs if arc_adjacency[i,j]` 루프를 사용하던 부분을 두 종류로 분리:
1. **_L 제약 (slope 열)**: `for (i,j) in aap` — 인접 쌍만
2. **_0 제약 (intercept 열)**: `for i in 1:num_arcs` — 모든 행

#### (D) evaluate_master_opt_cut의 compact → dense 변환

Outer loop의 cut 구성 코드가 dense 3D 배열 (`Uhat1[s,i,j] .* diag_x_E`)을 사용하므로, compact 값을 dense로 변환해야 한다:
```julia
Uhat1 = zeros(S, num_arcs, num_arcs+1)
for s in 1:S
    for (i,j) in afp
        Uhat1[s, i, j] = value(leader_instances[s][2][:Uhat1][(i,j)])
    end
end
```
비인접 항목은 0으로 유지되므로 수학적으로 동일한 cut이 생성된다.

#### (E) 함수 교체 방식

`@eval Main`을 사용하여 4개 함수를 runtime에 교체:
```julia
@eval Main initialize_isp = $initialize_isp_compact
@eval Main isp_leader_optimize! = $isp_leader_optimize_compact!
@eval Main isp_follower_optimize! = $isp_follower_optimize_compact!
@eval Main evaluate_master_opt_cut = $evaluate_master_opt_cut_compact
```
이전 fix() 시도에서는 `initialize_isp`만 교체했으나, compact 변수는 행렬 슬라이싱이 불가하므로 optimize/evaluate 함수도 반드시 교체해야 한다.

### 11.4 변수 수 절감 예상

| Grid | \|A\| | adj density | Full vars/matrix | Compact vars/matrix | 절감 |
|------|-------|-------------|------------------|---------------------|------|
| 3×3  | 18    | ~50%        | 18×19 = 342      | ~180                | ~47% |
| 5×5  | 40    | ~20%        | 40×41 = 1640     | ~370                | ~77% |

### 11.5 실험 결과

**(아직 실행 전 — 실행 후 여기에 결과 추가 예정)**

| S | Original obj | Compact obj | Obj Diff | Orig Time | Compact Time | Speedup |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 5 | | | | | | |
