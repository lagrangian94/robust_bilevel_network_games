# Implementation Guide: Column-and-Constraint Generation (C&CG) Algorithm

## Background

We are replacing the current nested Benders (3-level: OMP → IMP → ISP) with a C&CG approach based on vertex optimality of α. The key insight: optimal α* lies at a vertex of the simplex Δ = {α ≥ 0 : Σα = w/S}, i.e., α* = (w/S)·e_j for some arc j. This eliminates the IMP entirely.

## Algorithm Overview

Two alternating phases:

1. **Benders phase**: For a restricted set J of active vertices, solve the OMP with scenario-wise multi-cuts via Benders decomposition until convergence.
2. **Pricing phase**: Given converged χ*, find the worst-case vertex outside J by solving max_{α ∈ Δ} Q_1(α, χ*). If it improves, add to J and return to Benders phase. Otherwise terminate.

## New File: `ccg_benders.jl`

### Dependencies
Same as existing code. Reuse:
- `network_generator.jl` — network data
- `build_uncertainty_set.jl` — R, r_dict, xi_bar
- `build_isp_compact.jl` OR the original ISP builders from `nested_benders_trust_region.jl` — ISP leader/follower models
- Solvers: Gurobi (OMP), Mosek (ISP conic subproblems)

### Data Structures

```julia
# Vertex: α^j = (w/S) * e_j, represented by arc index j
# For vertex j, scenario s: ISP instances are pre-built with α fixed

struct VertexData
    j::Int                          # arc index
    α::Vector{Float64}              # α vector (w/S * e_j)
    leader_instances::Dict{Int, Tuple{Model, Dict}}   # s => (model, vars)
    follower_instances::Dict{Int, Tuple{Model, Dict}}  # s => (model, vars)
end
```

### 1. OMP Construction: `build_omp_ccg`

Similar to existing `build_omp` but with vertex-scenario epigraph variables:

```julia
function build_omp_ccg(network, ϕU, λU, γ, w; optimizer=nothing)
    num_arcs = length(network.arcs) - 1
    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))
    
    # First-stage variables (same as existing)
    @variable(model, λ >= 0.001)
    @variable(model, x[1:num_arcs], Bin)
    @variable(model, h[1:num_arcs] >= 0)
    @variable(model, ψ0[1:num_arcs] >= 0)
    @variable(model, t_0)  # overall epigraph
    
    # Constraints on x, h, λ, ψ0 (same as existing build_omp)
    @constraint(model, resource_budget, sum(h) <= λ * w)
    @constraint(model, sum(x) <= γ)
    for i in 1:num_arcs
        if !network.interdictable_arcs[i]
            @constraint(model, x[i] == 0)
        end
    end
    # McCormick for ψ0
    for k in 1:num_arcs
        @constraint(model, ψ0[k] <= λU * x[k])
        @constraint(model, ψ0[k] <= λ)
        @constraint(model, ψ0[k] >= λ - λU * (1 - x[k]))
        @constraint(model, ψ0[k] >= 0)
    end
    
    @objective(model, Min, t_0)
    
    # NOTE: t_{j,s} variables and vertex constraints are added dynamically
    # when vertices are added to J. See add_vertex_to_omp!
    
    vars = Dict(:t_0 => t_0, :λ => λ, :x => x, :h => h, :ψ0 => ψ0,
                :vertex_vars => Dict{Int, Vector{VariableRef}}(),  # j => [t_{j,1}, ..., t_{j,S}]
                :vertex_constraints => Dict{Int, ConstraintRef}()) # j => (t_0 >= Σ_s t_{j,s})
    return model, vars
end
```

### 2. Adding a Vertex to OMP: `add_vertex_to_omp!`

When a new vertex j is discovered by pricing, add epigraph variables and the linking constraint:

```julia
function add_vertex_to_omp!(omp_model, omp_vars, j::Int, S::Int)
    # Create scenario epigraph variables for vertex j
    t_js = @variable(omp_model, [s=1:S], base_name="t_$(j)_s")
    
    # Linking constraint: t_0 >= Σ_s t_{j,s}
    con = @constraint(omp_model, omp_vars[:t_0] >= sum(t_js))
    
    omp_vars[:vertex_vars][j] = t_js
    omp_vars[:vertex_constraints][j] = con
    
    return t_js
end
```

### 3. ISP Instance Management for a Vertex: `build_vertex_isps`

For each vertex j, build ISP leader/follower instances for all scenarios. Reuse existing `build_isp_leader` / `build_isp_follower` (or compact versions).

```julia
function build_vertex_isps(j::Int, network, S, ϕU, λU, γ, w, v, uncertainty_set;
                           conic_optimizer=nothing, λ_sol, x_sol, h_sol, ψ0_sol)
    num_arcs = length(network.arcs) - 1
    α_j = zeros(num_arcs)
    α_j[j] = w / S  # vertex alpha
    
    # Reuse existing initialize_isp with fixed α
    leader_instances, follower_instances = initialize_isp(
        network, S, ϕU, λU, γ, w, v, uncertainty_set;
        conic_optimizer=conic_optimizer,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_j)
    
    return VertexData(j, α_j, leader_instances, follower_instances)
end
```

### 4. Evaluate a Vertex at Current χ: `evaluate_vertex!`

Solve all ISPs for vertex j at current (x_sol, h_sol, λ_sol, ψ0_sol). Return total objective and per-scenario cut info.

```julia
function evaluate_vertex!(vdata::VertexData, isp_data::Dict;
                          λ_sol, x_sol, h_sol, ψ0_sol)
    S = isp_data[:S]
    uncertainty_set = isp_data[:uncertainty_set]
    R = uncertainty_set[:R]
    r_dict = uncertainty_set[:r_dict]
    xi_bar = uncertainty_set[:xi_bar]
    epsilon = uncertainty_set[:epsilon]
    
    total_obj = 0.0
    cut_info_per_s = Dict{Int, Dict}()
    
    for s in 1:S
        U_s = Dict(:R => Dict(:1=>R[s]), :r_dict => Dict(:1=>r_dict[s]),
                   :xi_bar => Dict(:1=>xi_bar[s]), :epsilon => epsilon)
        
        (status_l, ci_l) = isp_leader_optimize!(
            vdata.leader_instances[s][1], vdata.leader_instances[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=vdata.α)
        
        (status_f, ci_f) = isp_follower_optimize!(
            vdata.follower_instances[s][1], vdata.follower_instances[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=vdata.α)
        
        total_obj += ci_l[:obj_val] + ci_f[:obj_val]
        cut_info_per_s[s] = Dict(:leader => ci_l, :follower => ci_f)
    end
    
    return total_obj, cut_info_per_s
end
```

### 5. Add Scenario-wise Benders Cut: `add_scenario_cuts_to_omp!`

For vertex j, scenario s, add a cut to t_{j,s} based on ISP dual info. This is per-scenario, so each cut bounds t_{j,s} individually.

```julia
function add_scenario_cuts_to_omp!(omp_model, omp_vars, vdata::VertexData,
                                    cut_info_per_s::Dict, isp_data::Dict, iter::Int)
    # For each scenario s, generate an optimality cut for t_{j,s}
    # The cut has the form: t_{j,s} >= affine_function(x, h, λ, ψ0)
    #
    # This is the same cut generation logic as in strict_benders_optimize!
    # but applied per-scenario instead of aggregated.
    # 
    # Key: use evaluate_master_opt_cut logic but decomposed by scenario.
    # Each scenario s gives:
    #   cut_s(χ) = intercept_l_s + intercept_f_s + gradient_l_s(χ) + gradient_f_s(χ)
    #
    # IMPORTANT: The cut coefficients come from ISP dual variables (U, β, Z, etc.)
    # evaluated at the current α^j. Since α^j is fixed for vertex j,
    # these are standard Benders cuts.
    
    j = vdata.j
    t_js = omp_vars[:vertex_vars][j]
    S = isp_data[:S]
    x, h, λ, ψ0 = omp_vars[:x], omp_vars[:h], omp_vars[:λ], omp_vars[:ψ0]
    num_arcs = length(x)
    
    # Reuse existing cut coefficient extraction logic
    # (adapt from strict_benders_optimize! or evaluate_master_opt_cut)
    # Generate one cut per scenario:
    for s in 1:S
        ci_l = cut_info_per_s[s][:leader]
        ci_f = cut_info_per_s[s][:follower]
        
        # Build affine expression in (x, h, λ, ψ0) for scenario s
        # This mirrors the cut construction in strict_benders.jl lines ~180-220
        # but for a single scenario instead of summing over all S.
        
        # ... (adapt existing cut construction logic here) ...
        # opt_cut_s = intercept_s + terms_in_x + terms_in_h + terms_in_λψ0
        
        # @constraint(omp_model, t_js[s] >= opt_cut_s)
    end
end
```

**NOTE to implementer**: The cut construction logic already exists in `strict_benders.jl` (around lines 180-220) and `evaluate_master_opt_cut`. The difference is that in the existing code, cuts are summed over all scenarios into a single expression. Here we keep them **separate per scenario**. The per-scenario components (Uhat1[s,:,:], Utilde1[s,:,:], etc.) are already computed individually in the existing code — just don't sum them.

### 6. Pricing Phase: `pricing_solve!`

Find the worst-case vertex at current χ*. This solves max_{α ∈ Δ} Q_1(α, χ*), which is exactly the existing inner Benders (IMP + ISP). Reuse `nested_benders_optimize!` or `tr_imp_optimize!`.

```julia
function pricing_solve!(network, S, ϕU, λU, γ, w, v, uncertainty_set, isp_data;
                        mip_optimizer, conic_optimizer,
                        λ_sol, x_sol, h_sol, ψ0_sol)
    # Build IMP (same as existing build_imp)
    imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
                                     mip_optimizer=mip_optimizer)
    
    # Initialize and run inner Benders (IMP ↔ ISP) to convergence
    # Reuse existing tr_imp_optimize! or the simpler imp loop from nested_benders.jl
    
    # ... (reuse existing inner Benders code) ...
    
    # Return: best vertex index j*, and Q_1(α^{j*}, χ*)
    α_sol = result[:α_sol]
    j_star = argmax(α_sol)  # vertex optimality: should be concentrated on one arc
    obj_val = result[:obj_val]
    
    return j_star, obj_val, α_sol
end
```

### 7. Main Algorithm: `ccg_benders_optimize!`

```julia
function ccg_benders_optimize!(network, ϕU, λU, γ, w, v, uncertainty_set;
                                mip_optimizer=Gurobi.Optimizer,
                                conic_optimizer=Mosek.Optimizer,
                                ε_benders=1e-4, ε_pricing=1e-4,
                                max_ccg_iter=50, max_benders_iter=200)
    
    S = length(uncertainty_set[:xi_bar])
    num_arcs = length(network.arcs) - 1
    isp_data = Dict(...)  # same as existing
    
    # ========== Initialization ==========
    # Build OMP (no vertices yet)
    omp_model, omp_vars = build_omp_ccg(network, ϕU, λU, γ, w; optimizer=mip_optimizer)
    
    # Initial vertex: solve pricing at an arbitrary χ_0 to get first vertex
    st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars)
    j_init, _, _ = pricing_solve!(...)
    
    # Active vertex set
    J = Set{Int}()
    vertex_data = Dict{Int, VertexData}()
    
    # Add initial vertex
    push!(J, j_init)
    add_vertex_to_omp!(omp_model, omp_vars, j_init, S)
    vertex_data[j_init] = build_vertex_isps(j_init, ...)
    
    # ========== Main C&CG Loop ==========
    upper_bound = Inf
    
    for ccg_iter in 1:max_ccg_iter
        @info "===== C&CG Iteration $ccg_iter, |J| = $(length(J)) ====="
        
        # -------- Benders Phase --------
        for benders_iter in 1:max_benders_iter
            # 1. Solve OMP
            optimize!(omp_model)
            x_sol = value.(omp_vars[:x])
            h_sol = value.(omp_vars[:h])
            λ_sol = value(omp_vars[:λ])
            ψ0_sol = value.(omp_vars[:ψ0])
            t_0_sol = value(omp_vars[:t_0])  # lower bound
            
            # 2. Evaluate ALL vertices in J (not just binding!)
            max_Q_j = -Inf
            for j in J
                obj_j, cut_info_j = evaluate_vertex!(vertex_data[j], isp_data;
                    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)
                
                max_Q_j = max(max_Q_j, obj_j)
                
                # 3. Add per-scenario cuts for vertex j
                add_scenario_cuts_to_omp!(omp_model, omp_vars, vertex_data[j],
                    cut_info_j, isp_data, benders_iter)
            end
            
            # 4. Update upper bound
            upper_bound = min(upper_bound, max_Q_j)
            gap = upper_bound - t_0_sol
            
            @info "  [Benders] Iter $benders_iter: LB=$t_0_sol, UB=$upper_bound, gap=$gap"
            
            if gap <= ε_benders
                @info "  Benders phase converged."
                break
            end
        end
        
        # -------- Pricing Phase --------
        j_new, Q_new, α_new = pricing_solve!(network, S, ϕU, λU, γ, w, v, uncertainty_set, isp_data;
            mip_optimizer=mip_optimizer, conic_optimizer=conic_optimizer,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)
        
        @info "  [Pricing] Best vertex: j=$j_new, Q=$Q_new"
        
        if j_new in J
            @info "  Worst-case vertex already in J. OPTIMAL."
            break
        elseif Q_new <= t_0_sol + ε_pricing
            @info "  No improving vertex found. ε-OPTIMAL."
            break
        else
            @info "  Adding vertex $j_new to J."
            push!(J, j_new)
            add_vertex_to_omp!(omp_model, omp_vars, j_new, S)
            vertex_data[j_new] = build_vertex_isps(j_new, ...)
        end
    end
    
    return Dict(
        :opt_sol => Dict(:x => x_sol, :h => h_sol, :λ => λ_sol, :ψ0 => ψ0_sol),
        :obj_val => upper_bound,
        :active_vertices => J,
        :vertex_data => vertex_data
    )
end
```

## Key Implementation Notes

### What to reuse from existing code
1. **OMP construction**: Adapt from `build_omp()` in `strict_benders.jl`. Same x, h, λ, ψ0 variables and constraints. Only the epigraph structure changes.
2. **ISP construction and solving**: Reuse `build_isp_leader`, `build_isp_follower`, `isp_leader_optimize!`, `isp_follower_optimize!` from `nested_benders_trust_region.jl` (or compact versions from `build_isp_compact.jl`).
3. **Cut coefficient extraction**: Reuse `evaluate_master_opt_cut` logic from `nested_benders_trust_region.jl` or `strict_benders.jl`. The only change: keep cuts **per-scenario** instead of summing across S.
4. **Pricing**: Reuse `build_imp` + inner Benders loop from `nested_benders.jl`. This is the existing IMP ↔ ISP loop.

### What is new
1. **OMP has vertex-indexed epigraph variables** `t_{j,s}` instead of a single `t_0` or `t_0_l + t_0_f`.
2. **Per-scenario cuts** within each vertex group (this was previously invalid without vertex grouping, now it's exact).
3. **C&CG outer loop** managing vertex set J.
4. **All vertices in J must be re-evaluated** when χ changes (not just binding ones).

### Critical correctness points
- Within each vertex-scenario, the cut is leader + follower combined into one cut for `t_{j,s}`.
- **Scenario multi-cut IS valid** here because α is fixed per vertex group.
- **Re-evaluate all j ∈ J** at each Benders iteration, not just binding vertices.
- **ISP parameter update**: When χ changes, ISP model parameters (RHS involving x, h, λ, ψ0) must be updated via `set_normalized_rhs` for all vertices in J.

### Testing strategy
1. First, test on small instance (3×3 grid, S=1) where full model solution is known.
2. Compare with existing `strict_benders_optimize!` — should give same optimal value.
3. Check that |J| at termination is small (typically 2-5 vertices).
4. Check that pricing correctly identifies vertex not in J.

### Expected file structure
```
ccg_benders.jl          # Main file: build_omp_ccg, add_vertex!, evaluate_vertex!, 
                        # add_scenario_cuts!, pricing_solve!, ccg_benders_optimize!
test_ccg_benders.jl     # Test script (similar to test_strict_benders.jl)
```
