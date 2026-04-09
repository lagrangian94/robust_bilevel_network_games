"""
tv_nested_benders.jl — TV-DRO nested Benders decomposition.

Main algorithm: outer loop (OMP ↔ OSP) + inner loop (IMP ↔ ISP-L/ISP-F).

ISP constraints are independent of OMP variables (x,h,λ,ψ⁰) →
  build ISP-L/ISP-F ONCE, update objective only per outer iteration.
All subproblems are LP → HiGHS for LP, Gurobi for MILP.
"""

using JuMP
using LinearAlgebra
using Printf


"""
    tv_inner_loop!(tv, imp_model, imp_vars, isp_l_model, isp_l_vars,
                   isp_f_model, isp_f_vars;
                   max_inner_iter=100, inner_tol=1e-5, verbose=true)

Inner loop: IMP ↔ ISP-L + ISP-F.

Returns Dict with:
- :Z0_val: converged Z₀* = Z^L* + Z^F*
- :leader_cut_info: last ISP-L cut info
- :follower_cut_info: last ISP-F cut info (including β, σ̃ for outer cut)
- :α_sol: converged α
- :inner_iters: number of iterations
"""
function tv_inner_loop!(tv::TVData, imp_model, imp_vars,
                        isp_l_model, isp_l_vars,
                        isp_f_model, isp_f_vars;
                        max_inner_iter=1000, inner_tol=1e-5, verbose=true)
    K = tv.num_arcs
    lower_bound = -Inf
    upper_bound = Inf
    best_leader_cut = nothing
    best_follower_cut = nothing
    best_α = nothing

    # --- Seed cut: ISP at initial α (uniform) to make IMP bounded ---
    α_init = fill(tv.w / K, K)
    _, leader_cut_init = tv_isp_leader_optimize!(isp_l_model, isp_l_vars, tv, α_init)
    _, follower_cut_init = tv_isp_follower_optimize!(isp_f_model, isp_f_vars, tv, α_init)
    add_tv_inner_cut_leader!(imp_model, imp_vars, leader_cut_init, 0)
    add_tv_inner_cut_follower!(imp_model, imp_vars, follower_cut_init, 0)

    subprob_obj_init = leader_cut_init[:obj_val] + follower_cut_init[:obj_val]
    lower_bound = subprob_obj_init
    best_leader_cut = leader_cut_init
    best_follower_cut = follower_cut_init
    best_α = copy(α_init)

    if verbose
        @printf("    Seed cut at α_init: Z^L+Z^F = %.6f\n", subprob_obj_init)
    end

    for inner_iter in 1:max_inner_iter
        # --- Solve IMP ---
        optimize!(imp_model)
        st = termination_status(imp_model)
        if st != MOI.OPTIMAL
            error("TV IMP not optimal: $st (inner_iter=$inner_iter)")
        end

        α_sol = [value(imp_vars[:α][k]) for k in 1:K]
        imp_obj = objective_value(imp_model)
        upper_bound = imp_obj

        # --- Solve ISP-L ---
        _, leader_cut = tv_isp_leader_optimize!(isp_l_model, isp_l_vars, tv, α_sol)

        # --- Solve ISP-F ---
        _, follower_cut = tv_isp_follower_optimize!(isp_f_model, isp_f_vars, tv, α_sol)

        # --- Inner LB ---
        subprob_obj = leader_cut[:obj_val] + follower_cut[:obj_val]
        if subprob_obj > lower_bound
            lower_bound = subprob_obj
            best_leader_cut = leader_cut
            best_follower_cut = follower_cut
            best_α = copy(α_sol)
        end

        if verbose
            gap = abs(upper_bound - lower_bound) / max(abs(upper_bound), 1e-10)
            @printf("    Inner iter %3d: UB=%.6f  LB=%.6f  gap=%.2e\n",
                    inner_iter, upper_bound, lower_bound, gap)
        end

        # --- Convergence check ---
        gap = abs(upper_bound - lower_bound) / max(abs(upper_bound), 1e-10)
        if gap <= inner_tol
            if verbose
                @info "  Inner loop converged at iter $inner_iter (gap=$gap)"
            end
            return Dict(
                :Z0_val => subprob_obj,
                :leader_cut_info => leader_cut,
                :follower_cut_info => follower_cut,
                :α_sol => α_sol,
                :inner_iters => inner_iter,
            )
        end

        # --- Add cuts to IMP ---
        add_tv_inner_cut_leader!(imp_model, imp_vars, leader_cut, inner_iter)
        add_tv_inner_cut_follower!(imp_model, imp_vars, follower_cut, inner_iter)
    end

    @warn "Inner loop did not converge in $max_inner_iter iterations"
    return Dict(
        :Z0_val => lower_bound,
        :leader_cut_info => best_leader_cut,
        :follower_cut_info => best_follower_cut,
        :α_sol => best_α,
        :inner_iters => max_inner_iter,
    )
end


"""
    tv_nested_benders_optimize!(tv::TVData;
        lp_optimizer, mip_optimizer,
        max_outer_iter=50, max_inner_iter=100,
        outer_tol=1e-4, inner_tol=1e-5,
        verbose=true)

Main TV-DRO nested Benders algorithm.

ISP-L/ISP-F are built ONCE (constraints independent of OMP vars).
Each outer iteration: update ISP objectives, build fresh IMP, run inner loop.

Returns Dict with optimal solution and convergence info.
"""
function tv_nested_benders_optimize!(tv::TVData;
        lp_optimizer, mip_optimizer,
        max_outer_iter=50, max_inner_iter=100,
        outer_tol=1e-4, inner_tol=1e-5,
        verbose=true)

    K = tv.num_arcs
    S = tv.S

    # --- Build OMP ---
    omp_model, omp_vars = build_tv_omp(tv; optimizer=mip_optimizer)

    # --- Build ISP-L, ISP-F ONCE (constraints are x-independent) ---
    x_init = zeros(K)
    h_init = zeros(K)
    λ_init = 0.0
    ψ0_init = zeros(K)
    isp_l_model, isp_l_vars = build_tv_isp_leader(tv, x_init; optimizer=lp_optimizer)
    isp_f_model, isp_f_vars = build_tv_isp_follower(tv, x_init, h_init, λ_init, ψ0_init;
                                                      optimizer=lp_optimizer)

    # Iteration history
    history = Dict(
        :lower_bounds => Float64[],
        :upper_bounds => Float64[],
        :inner_iters_list => Int[],
    )

    lower_bound = -Inf
    upper_bound = Inf

    for outer_iter in 1:max_outer_iter
        if verbose
            @info "=== Outer iteration $outer_iter ==="
        end

        # --- Solve OMP ---
        optimize!(omp_model)
        st = termination_status(omp_model)
        if st != MOI.OPTIMAL
            error("TV OMP not optimal: $st (outer_iter=$outer_iter)")
        end

        x_sol = [value(omp_vars[:x][k]) for k in 1:K]
        h_sol = [value(omp_vars[:h][k]) for k in 1:K]
        λ_sol = value(omp_vars[:λ])
        ψ0_sol = [value(omp_vars[:ψ0][k]) for k in 1:K]
        t0_val = objective_value(omp_model)

        lower_bound = t0_val

        if verbose
            x_int = round.(Int, x_sol)
            @printf("  OMP: t₀=%.6f, λ=%.4f, x=%s\n", t0_val, λ_sol, string(x_int))
        end

        # --- Update ISP objectives only (no rebuild!) ---
        update_tv_isp_leader_objective!(isp_l_model, isp_l_vars, tv, x_sol)
        update_tv_isp_follower_objective!(isp_f_model, isp_f_vars, tv,
                                           x_sol, h_sol, λ_sol, ψ0_sol)

        # --- Build IMP (fresh each outer iter — cuts accumulate for this x) ---
        imp_model, imp_vars = build_tv_imp(tv; optimizer=lp_optimizer)

        # --- Inner loop ---
        inner_result = tv_inner_loop!(tv, imp_model, imp_vars,
                                       isp_l_model, isp_l_vars,
                                       isp_f_model, isp_f_vars;
                                       max_inner_iter=max_inner_iter,
                                       inner_tol=inner_tol,
                                       verbose=verbose)

        Z0_val = inner_result[:Z0_val]
        upper_bound = min(upper_bound, Z0_val)

        push!(history[:lower_bounds], lower_bound)
        push!(history[:upper_bounds], upper_bound)
        push!(history[:inner_iters_list], inner_result[:inner_iters])

        if verbose
            gap = abs(upper_bound - lower_bound) / max(abs(upper_bound), 1e-10)
            @printf("  Outer iter %d: LB=%.6f  UB=%.6f  gap=%.2e  inner_iters=%d\n",
                    outer_iter, lower_bound, upper_bound, gap, inner_result[:inner_iters])
        end

        # --- Convergence check ---
        gap = abs(upper_bound - lower_bound) / max(abs(upper_bound), 1e-10)
        if gap <= outer_tol
            if verbose
                @info "Outer loop converged at iter $outer_iter (gap=$gap)"
            end
            return Dict(
                :status => :Optimal,
                :Z0 => Z0_val,
                :x => x_sol,
                :h => h_sol,
                :λ => λ_sol,
                :ψ0 => ψ0_sol,
                :α => inner_result[:α_sol],
                :lower_bound => lower_bound,
                :upper_bound => upper_bound,
                :outer_iters => outer_iter,
                :history => history,
            )
        end

        # --- Add outer cut (no c parameter) ---
        outer_cut = compute_tv_outer_cut_coeffs(
            tv, inner_result[:leader_cut_info], inner_result[:follower_cut_info],
            x_sol, h_sol, λ_sol, ψ0_sol)
        add_tv_optimality_cut!(omp_model, omp_vars, outer_cut, outer_iter)

        if verbose
            @printf("  Added outer cut: intercept=%.6f, π_λ=%.4f, π_x_range=[%.4f, %.4f]\n",
                    outer_cut[:intercept], outer_cut[:π_λ],
                    minimum(outer_cut[:π_x]), maximum(outer_cut[:π_x]))
        end
    end

    @warn "Outer loop did not converge in $max_outer_iter iterations"
    return Dict(
        :status => :MaxIter,
        :Z0 => upper_bound,
        :lower_bound => lower_bound,
        :upper_bound => upper_bound,
        :outer_iters => max_outer_iter,
        :history => history,
    )
end
