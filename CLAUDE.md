# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Julia research codebase for **robust bilevel network interdiction games** under incomplete information on data uncertainty. Implements a 2-stage Distributionally Robust Network Design Problem (2DRNDP) where a leader chooses interdiction decisions and a follower solves max-flow under uncertain capacities.

## Running Code

This is a Julia project. There is no formal `Project.toml`; dependencies are managed manually. Run files directly in a Julia REPL:

```julia
include("solve_benders.jl")     # Main solver entry point (Benders decomposition)
include("solve_full_model.jl")  # Solve full model directly (for small instances)
include("test_maxflow.jl")      # Validate network flow computations
```

Key solver dependencies: **JuMP**, **Gurobi**, **Mosek**, **Pajarito**, **HiGHS**, **Hypatia**, **Plots**, **Infiltrator** (debugging), **Revise** (hot-reload).

## Architecture

### Problem formulation
- **Stage 1 (Leader)**: Binary interdiction decisions `x`, budget parameter `λ`, resource `h`, coupling `ψ0`
- **Stage 2 (Follower)**: Max-flow equilibrium under uncertainty set `U_s = {ξ : Rξ ≥ r}`
- Parameters: `ϕU` (interdiction bound), `λU` (budget upper), `γ` (interdiction budget), `w` (weight), `v` (effectiveness)

### Module dependency flow
```
network_generator.jl → build_uncertainty_set.jl → [algorithm choice] → plot_benders.jl
                                                    ├── strict_benders.jl
                                                    ├── nested_benders.jl
                                                    └── nested_benders_trust_region.jl
```

### Key modules

| File | Purpose |
|------|---------|
| `network_generator.jl` | `GridNetworkData` struct, grid network generation, capacity scenarios, node-arc incidence matrices |
| `build_uncertainty_set.jl` | Box-uncertainty sets, robust counterpart matrices `(R, r)` via `build_robust_counterpart_matrices()` |
| `build_full_model.jl` | Complete 2DRNDP model via `build_full_2DRNDP_model()`, uses Pajarito (outer approximation + conic) |
| `build_dualized_outer_subprob.jl` | Dual outer subproblem for bilevel decomposition via `build_dualized_outer_subproblem()` |
| `build_nominal_sp.jl` | Non-robust 2-stage stochastic program via `build_full_2SP_model()` (for comparison) |
| `strict_benders.jl` | Traditional Benders: `build_omp()`, `osp_optimize!()`, `strict_benders_optimize!()` |
| `nested_benders.jl` | Three-level decomposition (Outer → Inner Master → Inner Sub): `build_imp()`, `isp_leader_optimize!()`, `isp_follower_optimize!()`, `nested_benders_optimize!()` |
| `nested_benders_trust_region.jl` | Trust-region stabilized nested Benders with binary L∞-norm trust regions at both outer and inner levels: `tr_nested_benders_optimize!()` |
| `plot_benders.jl` | Convergence plotting (bounds, center changes, region expansions) |
| `compact_ldr_utils.jl` | Dictionary-indexed LDR index set generators (`arc_adj_pairs`, `arc_full_pairs`, etc.) and compact-to-dense converters |
| `build_isp_compact.jl` | Compact ISP builders using dict-indexed variables: `build_isp_leader_compact()`, `initialize_isp_compact()`, optimize and cut functions |
| `compare_compact.jl` | Benchmark script comparing original vs compact ISP on 3×3 grid, S=1,2,5 |

### Algorithm hierarchy
1. **Strict Benders** — basic two-level decomposition with optimality cuts
2. **Nested Benders** — three-level decomposition handling inner master + inner subproblems per scenario
3. **Trust-Region Nested Benders** — adds binary trust regions (`||x - x̂||₁ ≤ B_bin`) and reverse-region constraints for stabilized convergence

## Known Issues

- Numerical convergence gaps appear when scaling to larger instances (5×5 grids, S=5 scenarios) with the trust-region method.
- The project uses Korean commit messages and notes (`idea.txt`).

## Output Files

- `.jld2` — serialized JuMP model solutions
- `.jls` — algorithm iteration history/results
- `.cbf`, `.lp` — solver input format exports
