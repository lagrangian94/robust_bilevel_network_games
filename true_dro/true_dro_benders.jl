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
- `inexact`: true이면 3회 중 2회는 OptimalityTarget=1 (local opt)으로 빠르게 풀고,
  3회째에만 global opt. Local opt에서도 valid cut 생성 (feasible point of max subproblem).
  UB는 global solve에서만 갱신.
- `strengthen_cuts`: `:none` (default), `:mw` (cut strengthening).
  `:mw` — outer bilinear: Sherali perturbation (x_pert로 추가 solve, constraint 변경 없음).
         mini-benders: MW (ISP-L/F 독립 LP Phase 2, joint Pareto-optimality 미보장).

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
        max_mini_benders_iter::Int=5,
        inexact::Bool=false,
        strengthen_cuts::Symbol=:none)

    K = td.num_arcs

    # lp_optimizer 미지정 시 nlp_optimizer (Gurobi) 사용
    if lp_optimizer === nothing
        lp_optimizer = nlp_optimizer
    end

    # ---- Inexact mode: local opt cycle ----
    const_inexact_cycle = 3  # every 3rd iter is global, others are local opt
    if inexact && verbose
        @info "Inexact mode: $(const_inexact_cycle-1)/$(const_inexact_cycle) iters use OptimalityTarget=1 (local opt)"
    end

    # ---- MW core point ----
    if strengthen_cuts == :mw
        n_interd = sum(td.interdictable_arcs)
        x_core = [(td.interdictable_arcs[k] ? td.gamma / n_interd : 0.0) for k in 1:K]
        if verbose
            @info "MW cut enabled: core point x_core (γ/$n_interd per interdictable arc)"
        end
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

        # ---- Inexact mode: set OptimalityTarget ----
        is_global_iter = true
        if inexact
            is_global_iter = (iter % const_inexact_cycle == 0)
            opt_target = is_global_iter ? 0 : 1
            set_optimizer_attribute(sub_model, "OptimalityTarget", opt_target)
        end

        # ---- Solve subproblem ----
        sub_info = solve_true_dro_subproblem!(sub_model, sub_vars, td, x_sol)
        Z0_val = sub_info[:Z0_val]
        is_exact = sub_info[:is_optimal]

        # UB는 OPTIMAL + global solve일 때만 갱신
        # (local opt of max subproblem ≤ global opt → UB 과소평가 위험)
        if is_exact && is_global_iter && Z0_val < upper_bound
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
            status_str = if inexact && !is_global_iter
                " [LOCAL]"
            elseif !is_exact
                " [TIME_LIMIT]"
            else
                ""
            end
            @printf("  Sub: Z₀=%.6f%s, α=[%s]\n", Z0_val, status_str, α_str)
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

        # ---- Sherali cut: perturbed bilinear solve at x_pert ----
        # Bilinear subproblem에 MW (optimality constraint 추가)하면 문제가 어려워져서
        # Sherali perturbation 사용: constraint 추가 없이 objective만 변경 → 동일 난이도.
        # (mini-benders에서는 LP이므로 MW 사용)
        #
        # ζ 선택 근거 (Sherali & Lunday 2011):
        #   - 원논문 Algorithm 5: RHS perturbation μ = 10⁻⁶ (best), 10⁻⁵~10⁻³도 시도
        #   - 우리 구조: objective perturbation (x_pert = (1-ζ)·x_sol + ζ·x_core)
        #     x가 objective에만 등장하므로 RHS perturbation이 아닌 objective perturbation
        #   - Bilinear B&B solver는 LP simplex보다 perturbation 감지 어려움 →
        #     RHS μ=10⁻⁶보다 큰 ζ=0.001 사용 (너무 크면 cut center 이탈, 너무 작으면 무효)
        if strengthen_cuts == :mw && is_exact
            ζ_sherali = 0.001
            x_pert = [(1.0 - ζ_sherali) * x_sol[k] + ζ_sherali * x_core[k] for k in 1:K]

            sherali_info = solve_true_dro_subproblem!(sub_model, sub_vars, td, x_pert)
            sherali_cut = compute_true_dro_outer_cut(td, sherali_info, x_pert)
            cut_count += 1
            add_true_dro_optimality_cut!(omp_model, omp_vars, sherali_cut, cut_count)

            if verbose
                @printf("  Sherali-Cut: intercept=%.6f, π_x_range=[%.4f, %.4f]\n",
                        sherali_cut[:intercept],
                        minimum(sherali_cut[:π_x]), maximum(sherali_cut[:π_x]))
            end
        end

        # ---- Mini-Benders phase: fix α*, iterate OMP ↔ LP subproblem (§9.4) ----
        #   Phase 1: α from bilinear solve → mini-benders cuts
        #   α-step:  fix (a*, d*) from last ISP-L/F → sub_model becomes LP → new α'
        #   Phase 2: α' → mini-benders cuts
        if mini_benders
            α_fixed = sub_info[:α_val]
            last_a_val = nothing  # from last ISP-L solve
            last_d_val = nothing  # from last ISP-F solve

            for phase in 1:2
                # Set α for this phase
                update_isp_leader_alpha!(isp_l_model, isp_l_vars, td, α_fixed)
                update_isp_follower_alpha!(isp_f_model, isp_f_vars, td, α_fixed)

                phase_tag = phase == 1 ? "Mini" : "Alt"
                prev_lb_mb = lower_bound
                stag_mb = 0

                for j in 1:max_mini_benders_iter
                    # Re-solve OMP with accumulated cuts → new x̄
                    optimize!(omp_model)
                    st_mb = termination_status(omp_model)
                    if st_mb != MOI.OPTIMAL
                        if verbose
                            @printf("  %s-Benders[%d]: OMP %s, break\n", phase_tag, j, st_mb)
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
                            @printf("  %s-Benders[%d]: LB stagnated, break\n", phase_tag, j)
                        end
                        break
                    end

                    # Solve ISP-L(α*, x̄_new) + ISP-F(α*, x̄_new) → cut
                    # Note: inexact mode의 local opt α나 alternating α가 특정 x̄에서
                    # ISP-F를 DUAL_INFEASIBLE (unbounded)로 만들 수 있음.
                    # 이 경우 mini-benders phase만 중단하고 outer loop은 계속 진행.
                    local l_info, f_info
                    try
                        update_isp_leader_objective!(isp_l_model, isp_l_vars, td, x_mb)
                        l_info = solve_isp_leader!(isp_l_model, isp_l_vars, td)

                        update_isp_follower_objective!(isp_f_model, isp_f_vars, td, x_mb)
                        f_info = solve_isp_follower!(isp_f_model, isp_f_vars, td)
                    catch e
                        if verbose
                            @printf("  %s-Benders[%d]: ISP failed (%s), break\n",
                                    phase_tag, j, sprint(showerror, e))
                        end
                        break
                    end

                    # 마지막 ISP solve의 a*, d* 저장 (α-step에 사용)
                    last_a_val = l_info[:a_val]
                    last_d_val = f_info[:d_val]

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

                    # ---- Cut 추가: base 또는 MW ----
                    # MW: ISP-L/F 독립 적용 (joint Pareto-optimality 미보장, valid cut 보장)
                    local cut_tag, final_cut
                    if strengthen_cuts == :mw
                        S = td.S
                        φ̂U = td.phi_hat_U
                        φ̃U = td.phi_tilde_U
                        λU = td.lambda_U

                        # ISP-L MW Phase 2
                        z_star_l = l_info[:obj_val]
                        orig_obj_l = sum(isp_l_vars[:σ_hat][s] for s in 1:S) -
                            φ̂U * sum(x_mb[k] * isp_l_vars[:ρ_hat_1][k, s] for k in 1:K, s in 1:S) -
                            φ̂U * sum((1.0 - x_mb[k]) * isp_l_vars[:ρ_hat_3][k, s] for k in 1:K, s in 1:S)
                        mw_con_l = @constraint(isp_l_model, orig_obj_l >= z_star_l - 1e-6)
                        update_isp_leader_objective!(isp_l_model, isp_l_vars, td, x_core)
                        optimize!(isp_l_model)
                        mw_l_ok = termination_status(isp_l_model) == MOI.OPTIMAL
                        mw_l_info = mw_l_ok ? Dict(
                            :obj_val => objective_value(isp_l_model),
                            :rho_hat_1_val => [value(isp_l_vars[:ρ_hat_1][k, s]) for k in 1:K, s in 1:S],
                            :rho_hat_3_val => [value(isp_l_vars[:ρ_hat_3][k, s]) for k in 1:K, s in 1:S],
                        ) : nothing
                        delete(isp_l_model, mw_con_l)
                        update_isp_leader_objective!(isp_l_model, isp_l_vars, td, x_mb)

                        # ISP-F MW Phase 2
                        z_star_f = f_info[:obj_val]
                        orig_obj_f = -φ̃U * sum(x_mb[k] * isp_f_vars[:ρ_tilde_1][k, s] for k in 1:K, s in 1:S) -
                            φ̃U * sum((1.0 - x_mb[k]) * isp_f_vars[:ρ_tilde_3][k, s] for k in 1:K, s in 1:S) -
                            λU * sum(x_mb[k] * isp_f_vars[:ρ_psi0_1][k] for k in 1:K) -
                            λU * sum((1.0 - x_mb[k]) * isp_f_vars[:ρ_psi0_3][k] for k in 1:K)
                        mw_con_f = @constraint(isp_f_model, orig_obj_f >= z_star_f - 1e-6)
                        update_isp_follower_objective!(isp_f_model, isp_f_vars, td, x_core)
                        optimize!(isp_f_model)
                        mw_f_ok = termination_status(isp_f_model) == MOI.OPTIMAL
                        mw_f_info = mw_f_ok ? Dict(
                            :obj_val => objective_value(isp_f_model),
                            :rho_tilde_1_val => [value(isp_f_vars[:ρ_tilde_1][k, s]) for k in 1:K, s in 1:S],
                            :rho_tilde_3_val => [value(isp_f_vars[:ρ_tilde_3][k, s]) for k in 1:K, s in 1:S],
                            :rho_psi0_1_val => [value(isp_f_vars[:ρ_psi0_1][k]) for k in 1:K],
                            :rho_psi0_3_val => [value(isp_f_vars[:ρ_psi0_3][k]) for k in 1:K],
                        ) : nothing
                        delete(isp_f_model, mw_con_f)
                        update_isp_follower_objective!(isp_f_model, isp_f_vars, td, x_mb)

                        if mw_l_ok && mw_f_ok
                            mw_mini_info = Dict(
                                :Z0_val => mw_l_info[:obj_val] + mw_f_info[:obj_val],
                                :α_val => α_fixed,
                                :rho_hat_1_val => mw_l_info[:rho_hat_1_val],
                                :rho_hat_3_val => mw_l_info[:rho_hat_3_val],
                                :rho_tilde_1_val => mw_f_info[:rho_tilde_1_val],
                                :rho_tilde_3_val => mw_f_info[:rho_tilde_3_val],
                                :rho_psi0_1_val => mw_f_info[:rho_psi0_1_val],
                                :rho_psi0_3_val => mw_f_info[:rho_psi0_3_val],
                            )
                            final_cut = compute_true_dro_outer_cut(td, mw_mini_info, x_core)
                            cut_tag = "mw"
                        else
                            # MW failed → fallback to base cut
                            final_cut = compute_true_dro_outer_cut(td, mini_sub_info, x_mb)
                            cut_tag = "base*"
                        end
                    else
                        final_cut = compute_true_dro_outer_cut(td, mini_sub_info, x_mb)
                        cut_tag = "base"
                    end

                    cut_count += 1
                    add_true_dro_optimality_cut!(omp_model, omp_vars, final_cut, cut_count)

                    if verbose
                        @printf("  %s-Benders[%d] (%s): LB=%.6f, Z₀(α*)=%.6f, intercept=%.6f\n",
                                phase_tag, j, cut_tag, lb_mb, Z0_mini, final_cut[:intercept])
                    end

                    # Gap check (mini LB vs outer UB)
                    mini_gap = abs(upper_bound - lb_mb) / max(abs(upper_bound), 1e-10)
                    if mini_gap <= tol
                        if verbose
                            @printf("  %s-Benders[%d]: gap=%.2e ≤ tol, break\n", phase_tag, j, mini_gap)
                        end
                        break
                    end
                end

                # ---- α-step: fix (a*, d*) in sub_model → LP over α → new α' ----
                if phase == 1 && last_a_val !== nothing
                    S = td.S

                    # Fix a, d → bilinear ζ=α·a becomes linear in α
                    for s in 1:S
                        fix(sub_vars[:a][s], last_a_val[s]; force=true)
                        fix(sub_vars[:d][s], last_d_val[s]; force=true)
                    end

                    # OMP x̄ for objective
                    optimize!(omp_model)
                    if termination_status(omp_model) != MOI.OPTIMAL
                        # Unfix and skip phase 2
                        for s in 1:S
                            unfix(sub_vars[:a][s])
                            unfix(sub_vars[:d][s])
                        end
                        break
                    end
                    x_alt = [value(omp_vars[:x][k]) for k in 1:K]

                    # Solve α-step (sub_model with a,d fixed → LP)
                    alt_info = solve_true_dro_subproblem!(sub_model, sub_vars, td, x_alt)
                    α_fixed = alt_info[:α_val]

                    # Cut from α-step
                    alt_cut = compute_true_dro_outer_cut(td, alt_info, x_alt)
                    cut_count += 1
                    add_true_dro_optimality_cut!(omp_model, omp_vars, alt_cut, cut_count)

                    if verbose
                        α_str = join([@sprintf("%.3f", a) for a in α_fixed], ",")
                        @printf("  α-step: Z₀=%.6f, α=[%s]\n", alt_info[:Z0_val], α_str)
                    end

                    # Unfix a, d for future bilinear solves
                    for s in 1:S
                        unfix(sub_vars[:a][s])
                        unfix(sub_vars[:d][s])
                    end
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
