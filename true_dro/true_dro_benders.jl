"""
true_dro_benders.jl — Outer Benders for True-DRO-Exact (true_dro_v5.md §9).

Single-level Benders:
  OMP (MILP, x∈X, t₀)  ↔  Bilinear subproblem (Gurobi NonConvex=2)

Each iteration:
  1. Solve OMP → x̄, t₀ (LB)
  2. Update subproblem objective with x̄, solve → Z₀(x̄)
     Update UB ← min(UB, Z₀(x̄))
  3. Compute cut from ρ values (§9.3) and add to OMP

Subproblem is built ONCE (constraints x-independent); only objective updates.
"""

using JuMP
using LinearAlgebra
using Printf


"""
    true_dro_benders_optimize!(td::TrueDROData;
        mip_optimizer, nlp_optimizer,
        max_iter=50, tol=1e-4, verbose=true,
        nonconvex_attr=nothing)

Run outer Benders.

# Arguments
- `mip_optimizer`: e.g., `Gurobi.Optimizer` (for OMP MILP)
- `nlp_optimizer`: e.g., `Gurobi.Optimizer` (for bilinear subproblem; needs NonConvex=2)
- `nonconvex_attr`: optional Pair to set on subproblem model after construction,
  e.g., `"NonConvex" => 2` for Gurobi. Defaults to `"NonConvex" => 2`.

Returns Dict with :status, :Z0, :x, :α, :lower_bound, :upper_bound, :iters, :history.
"""
function true_dro_benders_optimize!(td::TrueDROData;
        mip_optimizer, nlp_optimizer,
        max_iter=1e+3, tol=1e-4, verbose=true,
        sub_verbose=false,
        nonconvex_attr=("NonConvex" => 2))

    K = td.num_arcs

    # ---- Build OMP ----
    omp_model, omp_vars = build_true_dro_omp(td; optimizer=mip_optimizer)

    # ---- Build subproblem once with x_bar = 0 (objective will be updated each iter) ----
    x_init = zeros(K)
    sub_model, sub_vars = build_true_dro_subproblem(td, x_init; optimizer=nlp_optimizer, silent=!sub_verbose)
    if nonconvex_attr !== nothing
        try
            set_optimizer_attribute(sub_model, nonconvex_attr.first, nonconvex_attr.second)
        catch err
            @warn "Could not set $(nonconvex_attr.first) on subproblem: $err"
        end
    end

    history = Dict(
        :lower_bounds => Float64[],
        :upper_bounds => Float64[],
        :Z0_vals => Float64[],
    )

    lower_bound = -Inf
    upper_bound = Inf
    best_x = zeros(K)
    best_α = zeros(K)

    for iter in 1:max_iter
        if verbose
            @info "=== True-DRO Benders iter $iter ==="
        end

        # ---- Solve OMP ----
        optimize!(omp_model)
        st = termination_status(omp_model)
        if st != MOI.OPTIMAL
            error("True-DRO OMP not optimal: $st (iter=$iter)")
        end

        x_sol = [value(omp_vars[:x][k]) for k in 1:K]
        t0_val = objective_value(omp_model)
        lower_bound = t0_val

        if verbose
            x_int = round.(Int, x_sol)
            @printf("  OMP: t₀=%.6f, x=%s\n", t0_val, string(x_int))
        end

        # ---- Solve subproblem ----
        sub_info = solve_true_dro_subproblem!(sub_model, sub_vars, td, x_sol)
        Z0_val = sub_info[:Z0_val]

        if Z0_val < upper_bound
            upper_bound = Z0_val
            best_x = copy(x_sol)
            best_α = copy(sub_info[:α_val])
        end

        push!(history[:lower_bounds], lower_bound)
        push!(history[:upper_bounds], upper_bound)
        push!(history[:Z0_vals], Z0_val)

        gap = abs(upper_bound - lower_bound) / max(abs(upper_bound), 1e-10)
        if verbose
            α_str = join([@sprintf("%.3f", a) for a in sub_info[:α_val]], ",")
            @printf("  Sub: Z₀=%.6f, α=[%s]\n", Z0_val, α_str)
            @printf("  Iter %d: LB=%.6f  UB=%.6f  gap=%.2e\n",
                    iter, lower_bound, upper_bound, gap)
        end

        if gap <= tol
            if verbose
                @info "True-DRO Benders converged at iter $iter (gap=$gap)"
            end
            return Dict(
                :status => :Optimal,
                :Z0 => upper_bound,
                :x => best_x,
                :α => best_α,
                :lower_bound => lower_bound,
                :upper_bound => upper_bound,
                :iters => iter,
                :history => history,
            )
        end

        # ---- Add outer cut ----
        outer_cut = compute_true_dro_outer_cut(td, sub_info, x_sol)
        add_true_dro_optimality_cut!(omp_model, omp_vars, outer_cut, iter)

        if verbose
            @printf("  Cut: intercept=%.6f, π_x_range=[%.4f, %.4f]\n",
                    outer_cut[:intercept],
                    minimum(outer_cut[:π_x]), maximum(outer_cut[:π_x]))
        end
    end

    @warn "True-DRO Benders did not converge in $max_iter iterations"
    return Dict(
        :status => :MaxIter,
        :Z0 => upper_bound,
        :x => best_x,
        :α => best_α,
        :lower_bound => lower_bound,
        :upper_bound => upper_bound,
        :iters => max_iter,
        :history => history,
    )
end
