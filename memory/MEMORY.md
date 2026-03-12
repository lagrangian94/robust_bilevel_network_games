# Project Memory

## Architecture Summary
- See [primal_isp.md](primal_isp.md) for primal ISP implementation details
- See [conventions.md](conventions.md) for sign conventions and known issues
- See [ipm_mu_offset.md](ipm_mu_offset.md) for IPM analytic center artifact (μ += ε bias) and fix

## Key Files
| File | Role |
|------|------|
| `nested_benders_trust_region.jl` | Main solver — `tr_nested_benders_optimize!` with `isp_mode` param |
| `build_primal_isp.jl` | Primal ISP builders, optimize, update, hybrid inner loop, outer cut extraction |
| `compare_benders.jl` | Benchmark script — includes hybrid/full_primal sections |
| `test_hybrid_benders.jl` | 3-way comparison test (original vs hybrid vs full_primal) |

## User Preferences
- Korean commit messages and comments
- Do NOT weaken error handling (no replacing errors with warnings)
- Prefers concise explanations with tables
