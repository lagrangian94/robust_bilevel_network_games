# Factor 3 Experiment Summary

## Capacity Factor Model

- **Function**: `generate_capacity_scenarios_factor_sparse` (additive factor model)
- **Formula**: `c_e^s = max(ε, c_bar + (1/k) Σ_j F_{ej} ξ_j^s)`
  - `F_{ej} ~ Uniform(-a, a)`, `a = 4.0`
  - `ξ_j^s ~ Exp(mean=1)`
  - `c_bar = 10.0` (baseline capacity)
  - `clip_eps = 0.1` (safety clip)
- **Factors**: `k = 5` (`num_factors=5`)
- **Scenarios**: `S = 20`, `seed = 42`
- **All arcs interdictable**: Yes (모든 네트워크에서 `interdictable_arcs = fill(true, ...)`)
- **λU**: 2.0, **v**: 1.0

## Network Settings

| Network | Nodes | Arcs | Intd Arcs | γ | w |
|---------|-------|------|-----------|---|---|
| polska | 12 | 36 | 36 (all) | 1 | varies |
| abilene | 12 | 30 | 30 (all) | 1 | varies |
| nobel_us | 14 | 42 | 42 (all) | 2 | varies |
| sioux_falls | 24 | 76 | 76 (all) | 2 | varies |

## Nominal vs Robust Solutions

Nominal: `build_full_2SP_model` (compact MILP) or Benders ε=0 (sioux_falls)
Robust: `true_dro_benders_optimize!` with ε=1.0

| Network | γ | x_nom (arcs) | Z₀_nom | x_rob (arcs) | Z₀_rob | DIFF? |
|---------|---|-------------|--------|-------------|--------|-------|
| polska | 1 | [18] | 21.183 | [33] | 22.446 | **DIFF** |
| abilene | 1 | [11] | 20.144 | [5] | 22.203 | **DIFF** |
| nobel_us | 2 | [7, 25] | 16.552 | [15, 20] | 16.656 | **DIFF** |
| sioux_falls | 2 | [72, 73] | ~15.81 | [35, 73] | 15.813 | **DIFF** |

### x* Solution Sources

| Network | x_nom source | x_rob source |
|---------|-------------|-------------|
| polska | `factor_3/logs/polska_all_intd_k5_gamma1.log` | same log |
| abilene | `factor_3/logs/abilene_all_intd_k5_gamma1.log` | same log |
| nobel_us | `factor_3/logs/nobel_us_all_intd_k5_gamma2.log` | same log |
| sioux_falls | `factor_3/logs/sioux_falls_all_intd_k5_gamma2.log` | same log |

## OOS Phase B Results

- **Method**: Asymmetric Dirichlet, shortcut (p_center direct)
- **M**: 200 outer samples
- **noise_scale**: 0.5
- **seed**: 42
- **gap = rob - nom** (maximization: gap > 0 → rob wins, follower gets more flow under rob)

| Network | γ | β | Nom Mean | Rob Mean | Gap Mean | Gap p5 | Gap p95 | Rob Win% |
|---------|---|---|----------|----------|----------|--------|---------|----------|
| polska | 1 | 0.1 | 21.1823 | 21.4192 | +0.2368 | --(neg) | 0.3947 | **100.0%** |
| polska | 1 | 0.3 | 21.1826 | 21.4193 | +0.2367 | -- | 0.3971 | **100.0%** |
| polska | 1 | 0.5 | 21.1827 | 21.4194 | +0.2366 | -- | 0.3972 | **100.0%** |
| polska | 1 | 1.0 | 21.1828 | 21.4194 | +0.2366 | -- | 0.3972 | **100.0%** |
| abilene | 1 | 0.1 | 20.1768 | 20.1928 | +0.0160 | -0.0376 | 0.0690 | **70.0%** |
| abilene | 1 | 0.3 | 20.1777 | 20.1935 | +0.0158 | -0.0387 | 0.0690 | **70.0%** |
| abilene | 1 | 0.5 | 20.1778 | 20.1936 | +0.0158 | -0.0387 | 0.0690 | **70.0%** |
| abilene | 1 | 1.0 | 20.1780 | 20.1937 | +0.0158 | -0.0387 | 0.0690 | **69.5%** |
| nobel_us | 2 | 0.1 | 16.5524 | 16.5526 | +0.0002 | -0.0000 | 0.0021 | 13.0% |
| nobel_us | 2 | 0.3 | 16.5526 | 16.5528 | +0.0002 | -0.0000 | 0.0021 | 14.0% |
| nobel_us | 2 | 0.5 | 16.5526 | 16.5528 | +0.0002 | -0.0000 | 0.0021 | 16.0% |
| nobel_us | 2 | 1.0 | 16.5526 | 16.5529 | +0.0002 | -0.0000 | 0.0021 | 15.0% |
| sioux_falls | 2 | 0.1 | 15.8131 | 15.8131 | +0.0000 | -0.0000 | 0.0000 | 38.5% |
| sioux_falls | 2 | 0.3 | 15.8131 | 15.8131 | +0.0000 | -0.0000 | 0.0000 | 39.0% |
| sioux_falls | 2 | 0.5 | 15.8131 | 15.8131 | +0.0000 | -0.0000 | 0.0000 | 42.0% |
| sioux_falls | 2 | 1.0 | 15.8131 | 15.8131 | +0.0000 | -0.0000 | 0.0000 | 41.0% |

### OOS Source Scripts

| Network | OOS script | Log file |
|---------|-----------|----------|
| polska, abilene, nobel_us | `factor_3/run_oos_eval.jl` | `factor_3/logs/oos_phase_b_eval.log` |
| sioux_falls | `factor_3/run_oos_eval_sf.jl` | `factor_3/logs/oos_phase_b_sioux_falls.log` |

## Key Observations

1. **Polska**: Rob wins 100% across all β. Strongest DIFF case. Gap ~0.24 (1.1% of nom mean).
2. **Abilene**: Rob wins 70%. Small but consistent gap ~0.016.
3. **Nobel_us**: x* differs but OOS performance nearly identical. Gap ~0.0002.
4. **Sioux_falls**: x* differs but OOS gap = 0.
5. **β insensitivity**: Gap is nearly constant across β values — suggests the distributional shift direction doesn't matter much, only that nom vs rob x* matters.
6. **In-sample vs OOS**: In-sample Z₀ gap (polska: 21.18 vs 22.45 = 6%) is larger than OOS gap (1.1%), as expected.

## Experiment History (settings tried before arriving at current)

| Setting | polska | abilene | nobel_us | sioux_falls |
|---------|--------|---------|----------|-------------|
| factor k=3, original intd, γ=3 | SAME | -- | -- | -- |
| factor k=3, original intd, γ=1 | SAME | -- | -- | -- |
| factor k=5, rand(0:10), γ=1 | DIFF | DIFF | SAME | SAME |
| factor k=7, γ=1 | -- | SAME | SAME | -- |
| additive c_bar=10, a=4, k=5, orig intd, γ=1 | SAME | DIFF | SAME | -- |
| additive, all intd, γ=1 | **DIFF** | **DIFF** | SAME | SAME |
| additive, all intd, γ=2 | -- | -- | **DIFF** | **DIFF** |
