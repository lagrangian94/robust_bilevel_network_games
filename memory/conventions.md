# Sign Conventions & Technical Notes

## MOI Dual Sign Conventions (Min problem)
| Constraint type | MOI dual sign | Mapping to dual ISP variable |
|----------------|--------------|----------------------------|
| `<=` (LessThan) | ≤ 0 (nonpositive) | `Uhat = -dual(con) ≥ 0` |
| `>=` (GreaterThan) | ≥ 0 (nonnegative) | `β = dual(con) ≥ 0` |
| `==` (EqualTo) | free | `Z = dual(con)` (free) |

Verified empirically: dual ISP `value()` ≈ primal ISP `dual()` with above mapping.
Exact match not guaranteed due to dual degeneracy (multiple optimal dual solutions).

## Bug Fixes Applied
- `dim_R_cols = size(R[1], 2)` not `size(R[1], 1)` — R has shape (num_arcs+1, num_arcs)
- `α_sol = max.(value.(imp_vars[:α]), 0.0)` — IMP can return tiny negatives (~1e-7) causing DUAL_INFEASIBLE in ISP. 적용 위치:
  - `build_primal_isp.jl:646` (hybrid inner loop)
  - `nested_benders_trust_region.jl` — `tr_imp_optimize!` (dual inner loop), `tr_imp_optimize_partial!` (partial inner loop)

## nested_benders.jl vs nested_benders_trust_region.jl
`nested_benders_trust_region.jl` is strictly more general:
- `tr_nested_benders_optimize!(..., outer_tr=false, inner_tr=false)` = `nested_benders_optimize!`
- `nested_benders.jl` has no unique functions
