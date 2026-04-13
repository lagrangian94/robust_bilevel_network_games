"""
true_dro_benders.jl — Outer Benders for True-DRO-Exact (true_dro_v5.md §9).

Single-level Benders:
  OMP (MILP, x∈X, t₀)  ↔  Bilinear subproblem (Gurobi NonConvex=2)

Each iteration:
  1. Solve OMP → x̄, t₀ (LB)
  2. Update subproblem objective with x̄, solve → Z₀(x̄)
     Update UB ← min(UB, Z₀(x̄))  [only if OPTIMAL]
  3. Compute cut from ρ values (§9.3) and add to OMP

Subproblem is built ONCE (constraints x-independent); only objective updates.

Adaptive time limit (sub_time_limit > 0):
  초반: time limit으로 빠르게 feasible incumbent → valid but weak cut.
  UB는 OPTIMAL일 때만 갱신. LB/UB stagnation 감지 시 time limit 해제.

Mini-Benders (§9.4, mini_benders=true):
  Bilinear solve → α* 고정 후, OMP ↔ ISP-L/F LP를 max_mini_benders_iter회 반복.
  α 고정이면 ISP-L ∥ ISP-F 독립 LP → 값싼 cut을 여러 개 축적.
"""

using JuMP
using LinearAlgebra
using Printf


"""
    true_dro_benders_optimize!(td::TrueDROData; ...)

Run outer Benders.

# Arguments
- `mip_optimizer`: OMP MILP solver
- `nlp_optimizer`: bilinear subproblem solver (needs NonConvex=2)
- `nonconvex_attr`: defaults to `"NonConvex" => 2`
- `sub_time_limit`: initial time limit (seconds) for bilinear subproblem.
  `nothing` = no limit. Incumbent에서 valid cut 생성, UB는 OPTIMAL만.
- `stagnation_window`: LB/UB 변화 없는 연속 iter 수 → time limit 해제 (default 3)
- `mini_benders`: LP-based inner loop with fixed α
- `lp_optimizer`: LP solver for mini-benders
- `max_mini_benders_iter`: mini-benders phase 반복 횟수 (default 5)

Returns Dict with :status, :Z0, :x, :α, :lower_bound, :upper_bound, :iters, :history.
"""
function true_dro_benders_optimize!(td::TrueDROData;
        mip_optimizer, nlp_optimizer,
        max_iter=1e+3, tol=1e-4, verbose=true,
        sub_verbose=false,
        nonconvex_attr=("NonConvex" => 2),
        sub_time_limit=nothing,
        stagnation_window::Int=3,
        mini_benders::Bool=false,
        lp_optimizer=nothing,
        max_mini_benders_iter::Int=5)

    K = td.num_arcs

    if mini_benders && lp_optimizer === nothing
        error("mini_benders=true requires lp_optimizer (e.g., HiGHS.Optimizer)")
    end

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

    # ---- Adaptive time limit state ----
    current_time_limit = sub_time_limit  # nothing = unlimited
    stagnation_count = 0
    prev_gap = Inf

    # ---- Mini-Benders: build ISP-L and ISP-F LP models once ----
    local isp_l_model, isp_l_vars, isp_f_model, isp_f_vars
    if mini_benders
        α_init = zeros(K)
        isp_l_model, isp_l_vars = build_true_dro_isp_leader(td, x_init, α_init; optimizer=lp_optimizer)
        isp_f_model, isp_f_vars = build_true_dro_isp_follower(td, x_init, α_init; optimizer=lp_optimizer)
        if verbose
            @info "Mini-Benders enabled: max_mini_iter=$max_mini_benders_iter"
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
    cut_count = 0

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

        # ---- Set subproblem time limit ----
        if current_time_limit !== nothing
            set_time_limit_sec(sub_model, current_time_limit)
        else
            set_time_limit_sec(sub_model, nothing)
        end

        # ---- Solve subproblem ----
        sub_info = solve_true_dro_subproblem!(sub_model, sub_vars, td, x_sol)
        Z0_val = sub_info[:Z0_val]
        is_exact = sub_info[:is_optimal]

        # UB는 OPTIMAL일 때만 갱신 (TIME_LIMIT incumbent는 Z₀_feas ≤ Z₀*)
        if is_exact && Z0_val < upper_bound
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
            exact_str = is_exact ? "" : " [TIME_LIMIT]"
            @printf("  Sub: Z₀=%.6f%s, α=[%s]\n", Z0_val, exact_str, α_str)
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

        # ---- Adaptive time limit: gap stagnation in near-convergence zone ----
        if current_time_limit !== nothing
            # gap이 1e-3 이하 진입 후에만 stagnation 체크
            if gap <= 1e-3
                gap_change = abs(prev_gap - gap)
                if gap_change < 1e-5
                    stagnation_count += 1
                else
                    stagnation_count = 0
                end
            end
            prev_gap = gap

            if stagnation_count >= stagnation_window
                current_time_limit = nothing
                if verbose
                    @printf("  → Time limit removed (gap stagnation %d iters in near-convergence zone)\n",
                            stagnation_count)
                end
            end
        end

        # ---- Add outer cut from bilinear solve ----
        outer_cut = compute_true_dro_outer_cut(td, sub_info, x_sol)
        cut_count += 1
        add_true_dro_optimality_cut!(omp_model, omp_vars, outer_cut, cut_count)

        if verbose
            @printf("  Cut: intercept=%.6f, π_x_range=[%.4f, %.4f]\n",
                    outer_cut[:intercept],
                    minimum(outer_cut[:π_x]), maximum(outer_cut[:π_x]))
        end

        # ---- Mini-Benders phase: fix α*, iterate OMP ↔ LP subproblem (§9.4) ----
        if mini_benders
            α_fixed = sub_info[:α_val]

            # Set α once for this phase
            update_isp_leader_alpha!(isp_l_model, isp_l_vars, td, α_fixed)
            update_isp_follower_alpha!(isp_f_model, isp_f_vars, td, α_fixed)

            prev_lb_mb = lower_bound
            stag_mb = 0

            for j in 1:max_mini_benders_iter
                # Re-solve OMP with accumulated cuts → new x̄
                optimize!(omp_model)
                st_mb = termination_status(omp_model)
                if st_mb != MOI.OPTIMAL
                    if verbose
                        @printf("  Mini-Benders[%d]: OMP %s, break\n", j, st_mb)
                    end
                    break
                end

                x_mb = [value(omp_vars[:x][k]) for k in 1:K]
                lb_mb = objective_value(omp_model)

                # LB stagnation check
                if lb_mb <= prev_lb_mb + 1e-6
                    stag_mb += 1
                else
                    stag_mb = 0
                end
                prev_lb_mb = lb_mb

                if stag_mb >= 3
                    if verbose
                        @printf("  Mini-Benders[%d]: LB stagnated, break\n", j)
                    end
                    break
                end

                # Solve ISP-L(α*, x̄_new) + ISP-F(α*, x̄_new) → cut
                update_isp_leader_objective!(isp_l_model, isp_l_vars, td, x_mb)
                l_info = solve_isp_leader!(isp_l_model, isp_l_vars, td)

                update_isp_follower_objective!(isp_f_model, isp_f_vars, td, x_mb)
                f_info = solve_isp_follower!(isp_f_model, isp_f_vars, td)

                Z0_mini = l_info[:obj_val] + f_info[:obj_val]

                mini_sub_info = Dict(
                    :Z0_val => Z0_mini,
                    :α_val => α_fixed,
                    :rho_hat_1_val => l_info[:rho_hat_1_val],
                    :rho_hat_3_val => l_info[:rho_hat_3_val],
                    :rho_tilde_1_val => f_info[:rho_tilde_1_val],
                    :rho_tilde_3_val => f_info[:rho_tilde_3_val],
                    :rho_psi0_1_val => f_info[:rho_psi0_1_val],
                    :rho_psi0_3_val => f_info[:rho_psi0_3_val],
                )

                mini_cut = compute_true_dro_outer_cut(td, mini_sub_info, x_mb)
                cut_count += 1
                add_true_dro_optimality_cut!(omp_model, omp_vars, mini_cut, cut_count)

                if verbose
                    @printf("  Mini-Benders[%d]: LB=%.6f, Z₀(α*)=%.6f, cut_intercept=%.6f\n",
                            j, lb_mb, Z0_mini, mini_cut[:intercept])
                end

                # Gap check (mini LB vs outer UB)
                mini_gap = abs(upper_bound - lb_mb) / max(abs(upper_bound), 1e-10)
                if mini_gap <= tol
                    if verbose
                        @printf("  Mini-Benders[%d]: gap=%.2e ≤ tol, break\n", j, mini_gap)
                    end
                    break
                end
            end

            # Update lower_bound from final OMP state after mini-benders
            optimize!(omp_model)
            if termination_status(omp_model) == MOI.OPTIMAL
                lower_bound = max(lower_bound, objective_value(omp_model))
            end
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
