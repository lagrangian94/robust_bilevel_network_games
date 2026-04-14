# Out-of-Sample Test Design: Implementation Specification

## 0. Context

This document specifies the out-of-sample (OOS) evaluation framework for our two-layer DRO network interdiction model. The codebase is in Julia; the existing solver infrastructure (Benders decomposition, network generation, uncertainty sets) is already implemented. This spec covers only the **OOS test layer** that wraps around the existing solver.

### Model Structure Recap

- **Leader** chooses binary interdiction `x ∈ X` (first stage, here-and-now)
- **Follower** observes `x`, then chooses capacity recovery `h` under **follower's belief** `P̃` (first stage for follower, here-and-now)
- **Nature** reveals scenario `ξ` according to **true distribution** `P*` (unknown to both)
- **Follower** then solves max-flow `y(x, h, ξ)` (second stage, wait-and-see)
- **Leader's objective**: evaluated under `P*` (also unknown to leader)

Leader's DRO uses TV-distance ambiguity sets centered at empirical `q̂ = (1/|K|)·1`:
- `P* ∈ B_{ε₁}(q̂)` for nature/OOS
- `P̃ ∈ B_{ε₂}(q̂)` for follower's belief

Under **symmetric ignorance** assumption: `ε₁ = ε₂ = ε`.

### Key References
- Sadana & Delage (2022): factor model, Dirichlet OOS protocol, grid network structure
- Lei et al. (2018): real-world networks, uniform capacity DGP
- Existing codebase: `network_generator.jl`, `build_uncertainty_set.jl`, Benders solvers

---

## 1. Capacity Scenario Generation

### 1.1 Factor Model (Sadana-style)

For **interdictable arcs only**. Non-interdictable arcs get constant capacity.

```julia
# Already implemented in network_generator.jl as:
# generate_capacity_scenarios_factor_model(num_arcs, num_scenarios; seed)
#
# Model: c^k = F * ξ^k
#   F ∈ R₊^{num_regular_arcs × k}, entries ~ Uniform(0,1)  [or Uniform(1,10)]
#   ξᵢ ~ Exponential(μᵢ), μ ~ Uniform(0,1)
#   k = 2 (number of latent factors, following Sadana)
#
# Non-interdictable arcs: capacity = sum of all regular arc capacities (dummy arc convention)
# In Sadana's original code: non-interdictable arcs get capacity = 100
```

**IMPORTANT**: The existing implementation uses `k=1` factor. Change to `k=2` to match Sadana. Verify that `F` is generated only for `num_regular_arcs = num_arcs - 1` (excluding dummy arc). This was already fixed per the docstring note.

### 1.2 Uniform i.i.d. Model (Lei et al.-style)

```julia
# Already implemented as:
# generate_capacity_scenarios_uniform_model(num_arcs, num_scenarios; seed)
#
# Each interdictable arc capacity ~ Uniform(1, 10) per scenario, i.i.d.
# No correlation structure between arcs.
```

### 1.3 Non-interdictable Arc Capacity

Following Sadana: non-interdictable arcs (first/last column arcs, source/sink arcs in grid; depends on network topology for real-world) get **constant capacity = 100** across all scenarios. The existing code uses `sum(regular_arc_capacities)` for the dummy arc — this is fine for the dummy arc but other non-interdictable arcs should be set to a large constant.

**Action item**: Verify that for grid networks, arcs in first/last columns and source/sink arcs are flagged as `interdictable = false` and receive constant capacity across scenarios.

---

## 2. Dirichlet Meta-Distribution for OOS

### 2.1 What Dirichlet Generates

The Dirichlet distribution generates **probability vectors over scenarios**, NOT arc capacities. Given `|K|` fixed capacity scenarios, `q ∈ Δ^{|K|}` assigns probability to each scenario.

```julia
using Distributions

"""
    sample_dirichlet(K::Int, β::Float64; n_samples::Int=1)

Sample probability vectors from Dirichlet(β·1_K).

# Arguments
- `K`: number of scenarios (= dimension of simplex)
- `β`: concentration parameter
  - β → 0: q concentrates on simplex vertices (one scenario dominates)
  - β = 1: q uniform on simplex
  - β → ∞: q → (1/K)·1 (uniform distribution over scenarios)
- `n_samples`: number of independent draws

# Returns
- Matrix of size (K × n_samples), each column is a probability vector
"""
function sample_dirichlet(K::Int, β::Float64; n_samples::Int=1)
    dir = Dirichlet(K, β)
    samples = zeros(K, n_samples)
    for i in 1:n_samples
        samples[:, i] = rand(dir)
    end
    return samples
end
```

### 2.2 Experimental Values of β

```julia
β_values = [0.1, 0.3, 0.5, 0.8]
```

- `β = 0.1`: very high distributional uncertainty, q can be far from uniform
- `β = 0.8`: moderate uncertainty, q stays relatively close to uniform

---

## 3. Ambiguity Radius Calibration

### 3.1 Procedure

For each `β`, calibrate `ε` so that a TV-distance ball of radius `ε` around `q̂ = (1/K)·1` contains 95% of Dirichlet draws.

```julia
"""
    calibrate_epsilon(K::Int, β::Float64; n_cal::Int=100, coverage::Float64=0.95)

Calibrate TV-distance radius ε for ambiguity set.

# Method
1. Draw n_cal samples from Dirichlet(β·1_K)
2. Compute L1 distance from each sample to uniform q̂ = (1/K)·1
3. Return the `coverage`-th quantile as ε

# Note
TV distance = (1/2) * L1 distance. We use L1 here because the TV-DRO
reformulation in our model uses L1 norm directly: ||q - q̂||₁ ≤ ε.
Adjust if your formulation uses TV = (1/2)||·||₁.
"""
function calibrate_epsilon(K::Int, β::Float64; n_cal::Int=100, coverage::Float64=0.95)
    q_hat = fill(1.0 / K, K)
    samples = sample_dirichlet(K, β; n_samples=n_cal)
    
    distances = [norm(samples[:, i] - q_hat, 1) for i in 1:n_cal]
    sort!(distances)
    
    idx = ceil(Int, coverage * n_cal)
    ε = distances[idx]
    
    return ε
end
```

### 3.2 Symmetric Ignorance → ε₁ = ε₂ = ε

**Modeling principle**: Leader has no additional information about `P*` vs `P̃`. Both are equally unknown deviations from `q̂`. Therefore both ambiguity radii are set to the same calibrated `ε`.

This is NOT a calibration result — it's a modeling assumption. The calibration only determines the absolute magnitude of the common `ε`.

---

## 4. Models to Compare

Three models, differing only in ambiguity radii:

```julia
models = Dict(
    :nominal      => (ε₁ = 0.0,  ε₂ = 0.0),   # no robustness
    :single_dro   => (ε₁ = ε,    ε₂ = 0.0),   # robust to OOS only; assumes P̃ = q̂
    :twolayer_dro  => (ε₁ = ε,    ε₂ = ε),     # robust to both OOS and follower belief
)
```

The key comparison: **single_dro vs twolayer_dro** gap = value of modeling follower belief uncertainty.

---

## 5. OOS Evaluation: Nested Design

### 5.1 Why Nested

- **Follower optimization** (solving for `h*` given `x*` and `q̃`): expensive (LP/MIP)
- **OOS evaluation** (computing expected flow given `x*`, `h*`, and `q_true`): cheap (weighted sum)

Nested design: few follower solves (outer loop), many OOS evaluations per follower solve (inner loop).

### 5.2 Procedure

```julia
"""
    oos_evaluate(x_star, network, capacity_scenarios, β, ε;
                 M=100, L=1000, seed=nothing)

Out-of-sample evaluation via nested Dirichlet sampling.

# Arguments
- `x_star`: leader's optimal interdiction decision (from solving the DRO model)
- `network`: GridNetworkData or RealWorldNetworkData
- `capacity_scenarios`: (|E| × |K|) matrix, fixed capacity scenarios
- `β`: Dirichlet concentration parameter
- `ε`: calibrated ambiguity radius (used only for reference, not in evaluation)
- `M`: number of follower belief draws (outer loop) — keep small (e.g., 100)
- `L`: number of OOS nature draws per follower (inner loop) — keep large (e.g., 1000)

# Returns
- `results`: Dict with keys :evals, :mean, :p95, :var_decomp
"""
function oos_evaluate(x_star, network, capacity_scenarios, β, ε;
                      M::Int=100, L::Int=1000, seed=nothing)
    !isnothing(seed) && Random.seed!(seed)
    
    K = size(capacity_scenarios, 2)  # number of scenarios
    
    # Storage
    Y_bar = zeros(M)          # E[flow | q̃⁽ʲ⁾]  (inner mean per outer sample)
    S2_j  = zeros(M)          # Var[flow | q̃⁽ʲ⁾] (inner variance per outer sample)
    all_evals = zeros(M, L)   # full evaluation matrix
    
    for j in 1:M
        # --- Outer loop: sample follower's belief ---
        q_tilde = rand(Dirichlet(K, β))
        
        # Solve follower's problem under q̃⁽ʲ⁾
        # h_star = solve_follower(x_star, q_tilde, network, capacity_scenarios, ...)
        # This calls your existing follower solver with q_tilde as the scenario weights.
        #
        # IMPORTANT: The follower solves:
        #   max_{h ∈ H}  Σ_k q̃_k · Q(h, x*, ξᵏ)
        # where Q(h, x, ξ) = max_y {d₀ᵀy : Ay ≤ b_x(ξ) - Bh}
        #
        # Implementation note: this requires solving a weighted-scenario version
        # of the follower's problem. If the existing solver assumes uniform weights,
        # it needs to be modified to accept arbitrary scenario weights q̃.
        
        h_star_j = solve_follower_weighted(x_star, q_tilde, network, capacity_scenarios)
        
        for ℓ in 1:L
            # --- Inner loop: sample nature's true distribution ---
            q_true = rand(Dirichlet(K, β))  # SAME Dirichlet, independent draw
            
            # Evaluate leader's objective under q_true
            # For each scenario k, compute flow(x*, h*⁽ʲ⁾, ξᵏ), then take weighted sum
            flow_per_scenario = compute_maxflow_per_scenario(
                x_star, h_star_j, network, capacity_scenarios
            )  # returns vector of length K
            
            eval_jl = dot(q_true, flow_per_scenario)
            all_evals[j, ℓ] = eval_jl
        end
        
        # Inner statistics
        Y_bar[j] = mean(all_evals[j, :])
        S2_j[j]  = var(all_evals[j, :])
    end
    
    # --- Aggregate statistics ---
    grand_mean = mean(Y_bar)
    
    # 95th percentile of ALL evaluations
    all_flat = vec(all_evals)
    p95 = quantile(all_flat, 0.95)
    
    # --- Variance decomposition (Law of Total Variance) ---
    # Total Var = Var_q̃[E[Y|q̃]] + E_q̃[Var[Y|q̃]]
    var_outer = var(Y_bar)            # Var_q̃[E_{q*}[flow | q̃]]  — follower belief effect
    var_inner_mean = mean(S2_j)       # E_q̃[Var_{q*}[flow | q̃]]  — nature effect
    total_var = var_outer + var_inner_mean
    follower_share = var_outer / total_var  # fraction of variance from follower belief
    
    return Dict(
        :evals       => all_evals,
        :mean        => grand_mean,
        :p95         => p95,
        :var_outer   => var_outer,       # Var from follower belief uncertainty
        :var_inner   => var_inner_mean,   # Var from nature uncertainty
        :total_var   => total_var,
        :follower_share => follower_share,
        :Y_bar       => Y_bar,           # per-outer-sample means (for boxplots)
    )
end
```

### 5.3 Helper: Max-Flow per Scenario

```julia
"""
    compute_maxflow_per_scenario(x, h, network, capacity_scenarios)

For fixed interdiction x and recovery h, compute max-flow for each scenario k.

# Returns
- Vector of length K: flow value under each scenario
"""
function compute_maxflow_per_scenario(x, h, network, capacity_scenarios)
    K = size(capacity_scenarios, 2)
    num_arcs = length(network.arcs) - 1
    flows = zeros(K)
    
    for k in 1:K
        # Effective capacity: ξ_e^k * (1 - v_e * x_e) + h_e
        # where v_e is interdiction effectiveness (default v=1 for full interdiction)
        ξ_k = capacity_scenarios[1:num_arcs, k]
        effective_cap = ξ_k .* (1.0 .- x[1:num_arcs]) .+ h[1:num_arcs]
        
        # Solve max-flow LP with effective capacities
        # Use your existing max-flow solver or JuMP LP
        flows[k] = solve_maxflow_lp(network, effective_cap)
    end
    
    return flows
end
```

### 5.4 Helper: Follower Problem with Weighted Scenarios

```julia
"""
    solve_follower_weighted(x, q, network, capacity_scenarios; w=..., v=1.0)

Solve follower's recovery problem under arbitrary scenario weights q.

    max_{h ∈ H}  Σ_k q_k · max_y {d₀ᵀy : Ay + Bh ≤ b_x(ξᵏ)}

where H = {h ≥ 0 : 1ᵀh ≤ w}.

# Arguments
- `x`: leader's interdiction (binary vector)
- `q`: scenario probability vector (length K)
- `network`: network data
- `capacity_scenarios`: (|E| × K) capacity matrix

# Returns
- `h_star`: optimal recovery vector
"""
function solve_follower_weighted(x, q, network, capacity_scenarios; w, v=1.0)
    K = size(capacity_scenarios, 2)
    num_arcs = length(network.arcs) - 1
    
    # Formulation:
    # max_{h, y_1,...,y_K}  Σ_k q_k · d₀ᵀ y_k
    # s.t.  A y_k ≤ b_x(ξᵏ) - B h,  ∀k    (flow constraints per scenario)
    #       1ᵀh ≤ w                          (recovery budget)
    #       h ≥ 0
    #       y_k ≥ 0,  ∀k
    #
    # This is a single LP (scenarios linked through shared h).
    # If K is small (e.g., 10), this is very tractable.
    
    # --- Implementation using JuMP ---
    # (Adapt to your existing model-building conventions)
    
    # TODO: implement using existing JuMP infrastructure
    # Return h_star
end
```

---

## 6. Main Experiment Loop

```julia
"""
    run_oos_experiment(;
        network_configs = [:grid_5x5, :grid_10x10],
        dgp = :factor_model,
        K = 10,
        β_values = [0.1, 0.3, 0.5, 0.8],
        γ_ratio = 0.3,
        ρ = 0.2,
        M = 100,
        L = 1000,
        seed = 42
    )

Main experiment driver.
"""
function run_oos_experiment(;
    network_configs = [:grid_5x5],
    dgp = :factor_model,
    K::Int = 10,
    β_values = [0.1, 0.3, 0.5, 0.8],
    γ_ratio = 0.3,
    ρ = 0.2,
    v = 1.0,
    M::Int = 100,
    L::Int = 1000,
    n_instances::Int = 10,
    seed::Int = 42
)
    results = Dict()
    
    for net_config in network_configs
        for instance_id in 1:n_instances
            instance_seed = seed + instance_id
            
            # --- Step 1: Generate network and capacity scenarios ---
            network = generate_network(net_config; seed=instance_seed)
            num_arcs = length(network.arcs) - 1
            
            if dgp == :factor_model
                cap, F = generate_capacity_scenarios_factor_model(
                    length(network.arcs), K; seed=instance_seed)
            elseif dgp == :uniform
                cap, F = generate_capacity_scenarios_uniform_model(
                    length(network.arcs), K; seed=instance_seed)
            end
            
            # --- Step 2: Set parameters ---
            num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
            γ = ceil(Int, γ_ratio * num_interdictable)
            interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
            c_bar = mean(cap[interdictable_idx, :])
            w = ρ * γ * c_bar
            
            for β in β_values
                # --- Step 3: Calibrate ε ---
                ε = calibrate_epsilon(K, β; n_cal=100, coverage=0.95)
                
                # --- Step 4: Solve each model ---
                model_results = Dict()
                
                for (model_name, (ε₁, ε₂)) in [
                    (:nominal,     (0.0, 0.0)),
                    (:single_dro,  (ε,   0.0)),
                    (:twolayer_dro, (ε,   ε  )),
                ]
                    # Solve leader problem with given (ε₁, ε₂)
                    # This calls your existing DRO solver.
                    # ε₁ controls the OOS ambiguity set
                    # ε₂ controls the follower belief ambiguity set
                    #
                    # For nominal (ε₁=ε₂=0): standard 2-stage SP with q̂ = uniform
                    # For single_dro (ε₂=0): DRO but follower uses q̂ exactly
                    # For twolayer_dro: full two-layer DRO (our model)
                    
                    x_star = solve_leader_dro(
                        network, cap, γ, w, v, ε₁, ε₂;
                        # ... pass solver options
                    )
                    
                    # --- Step 5: OOS Evaluation ---
                    oos = oos_evaluate(
                        x_star, network, cap, β, ε;
                        M=M, L=L, seed=instance_seed * 1000 + hash(model_name)
                    )
                    
                    model_results[model_name] = Dict(
                        :x_star => x_star,
                        :oos    => oos,
                        :ε      => ε,
                    )
                end
                
                key = (net_config, instance_id, β)
                results[key] = model_results
            end
        end
    end
    
    return results
end
```

---

## 7. Reporting

### 7.1 Primary Table: OOS Performance by Model × β

For each `(network, β)`, aggregate over instances:

| β | Model | Mean Flow | 95th Pctl | Follower Var Share |
|---|-------|-----------|-----------|-------------------|
| 0.1 | Nominal | ... | ... | ... |
| 0.1 | Single-DRO | ... | ... | ... |
| 0.1 | Two-layer DRO | ... | ... | ... |
| 0.3 | ... | ... | ... | ... |

### 7.2 Key Metric: Value of Follower Robustness (VFR)

```
VFR = (OOS_mean(single_dro) - OOS_mean(twolayer_dro)) / OOS_mean(twolayer_dro) × 100%
```

Interpretation: percentage improvement in OOS performance from modeling follower belief uncertainty.

### 7.3 Variance Decomposition Table

| β | Var(follower belief) | Var(nature) | Follower Share |
|---|---------------------|-------------|----------------|
| 0.1 | ... | ... | ...% |
| 0.3 | ... | ... | ...% |

If follower share is large → two-layer DRO is valuable.

### 7.4 Boxplots (à la Sadana Figure 3)

For each β: boxplot of `Y_bar` (per-outer-sample means) comparing Nominal vs Single-DRO vs Two-layer DRO across instances.

---

## 8. Instance Parameters Summary

| Parameter | Symbol | Values | Source |
|-----------|--------|--------|--------|
| Network size | m × n | 5×5, 10×10 | Sadana |
| Scenarios | \|K\| | 10 | Sadana (small to make DRO value visible) |
| Interdiction budget | γ | ⌊0.3 × \|interdictable arcs\|⌋ | Sadana: B = ⌊0.3m⌋ |
| Recovery budget ratio | ρ | 0.2 | Lei et al. |
| Interdiction effectiveness | v | 1.0 | Full interdiction |
| Dirichlet β | β | {0.1, 0.3, 0.5, 0.8} | Sadana |
| Calibration samples | n_cal | 100 | Sadana |
| Coverage level | - | 95% | Sadana |
| Outer OOS samples | M | 100 | Nested design |
| Inner OOS samples | L | 1000 | Nested design |
| Random instances | n_instances | 10 | Sadana |
| Factor model factors | k | 2 | Sadana |
| DGP types | - | Factor model, Uniform i.i.d. | Sadana / Lei |

---

## 9. Implementation Checklist

- [ ] **Verify factor model**: `k=2` factors, `F` generated only for interdictable arcs, non-interdictable arcs get constant capacity = 100
- [ ] **Implement `sample_dirichlet`**: wrapper around `Distributions.jl`
- [ ] **Implement `calibrate_epsilon`**: L1 distance, 95th percentile
- [ ] **Implement `solve_follower_weighted`**: follower LP with arbitrary scenario weights `q̃`
- [ ] **Implement `compute_maxflow_per_scenario`**: max-flow LP for each scenario given fixed `(x, h)`
- [ ] **Implement `oos_evaluate`**: nested sampling loop with variance decomposition
- [ ] **Modify leader solver interface**: accept `(ε₁, ε₂)` pair; `ε₂=0` should reduce to single-layer DRO
- [ ] **Implement `run_oos_experiment`**: main loop over networks × instances × β × models
- [ ] **Implement reporting**: tables, VFR metric, boxplots

---

## 10. File Structure Suggestion

```
oos_experiment/
├── oos_dirichlet.jl          # sample_dirichlet, calibrate_epsilon
├── oos_follower_weighted.jl  # solve_follower_weighted
├── oos_evaluate.jl           # oos_evaluate, compute_maxflow_per_scenario
├── oos_main.jl               # run_oos_experiment (main driver)
├── oos_report.jl             # tables, plots, VFR computation
└── oos_config.jl             # parameter defaults, network configs
```

All files should `include` or `using` the existing modules:
```julia
include("network_generator.jl")
include("build_uncertainty_set.jl")
# ... existing solver files as needed
```
