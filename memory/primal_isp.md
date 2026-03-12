# Primal ISP Implementation

## 3 ISP Modes (isp_mode parameter in tr_nested_benders_optimize!)

| Mode | Inner loop | Outer cut | Dual ISP needed? |
|------|-----------|-----------|-----------------|
| `:dual` (default) | dual ISP (`isp_leader_optimize!`) | `evaluate_master_opt_cut` (value of dual vars) | Yes |
| `:hybrid` | primal ISP (`primal_isp_leader_optimize!`) | `primal_evaluate_master_opt_cut` (re-solve dual ISP at converged α) | Yes |
| `:full_primal` | primal ISP | `evaluate_master_opt_cut_from_primal` (constraint shadow prices) | No |

## Parameter Location Reversal (Dual vs Primal ISP)
| | Dual ISP | Primal ISP |
|--|---------|-----------|
| α | constraint RHS (`set_normalized_rhs`) | objective coeff (`set_objective_coefficient`) |
| x,h,λ,ψ0 | objective (`set_objective_coefficient`) | constraint RHS (`set_normalized_rhs`) |
| μ extraction | `shadow_price(coupling_cons)` | `value(μhat)` directly |

## update_primal_isp_parameters! (replaces model rebuild)
Updates (x,h,λ,ψ0) via `set_normalized_rhs` instead of `initialize_primal_isp` each outer iter.

Constraints with parameters:
- **Leader**: Big-M1 (`ϕU*x`), Big-M3 (`ϕU*(1-x)`)
- **Follower**: Big-M1/3 (x), SOC eq block3 (`(λ-v*ψ0)*ξ̄`), SOC ineq block1 (`λ*d0`), SOC ineq block3 (`-h-(λ-v*ψ0)*ξ̄`)

## Outer Cut Extraction (full_primal mode)
Residual intercept approach: `intercept = subprob_obj - eval(cut_terms at x*,h*,λ*,ψ0*)`
- Avoids extracting P terms and variable bound duals
- Individual cut coefficients may differ from dual ISP (dual degeneracy) but cut is valid

## IPM Artifact: μ offset = ε (CRITICAL)
Mosek (IPM) returns analytic center for zero-cost variables. When α_k = 0, μhat_k has
zero objective coefficient → IPM inflates μhat_k by +ε (epsilon from uncertainty set).
Offset is ONLY on α_k = 0 components; α_k > 0 components have offset ≈ 0.

**Fix** (conditional, in `primal_isp_leader/follower_optimize!`):
```julia
ε = uncertainty_set[:epsilon]
for k in eachindex(subgradient)
    if α_sol[k] < 1e-8  # only correct zero-cost components
        subgradient[k] = max(subgradient[k] - ε, 0.0)
    end
end
intercept = obj_val - α_sol' * subgradient   # preserve cut tightness
```
- `obj_val` stays as `objective_value(model)` (unchanged)
- Blanket `max.(μ .- ε, 0)` over-corrects α_k > 0 components → degrades later iters

## Performance (5×5 grid, S=1, with conditional IPM fix)
| Method | Outer iter | Inner iter | Time |
|--------|-----------|-----------|------|
| Original (dual) | 60 | 242 | 61.9s |
| Hybrid (conditional fix) | 62 | 273 | 64.2s |
| Full Primal (conditional fix) | 93 | 432 | 79.6s |

All converge to same solution (obj gap < 1e-5).

## Known Issues (S>1)
- `con_bigM1_hat/tilde` indexed by (i,j) not (s,i,j) — overwritten for S>1
- `con_soc_eq_tilde`, `con_soc_ineq_tilde` overwritten each scenario in build loop
- Fix needed before running S>1 with full_primal mode
