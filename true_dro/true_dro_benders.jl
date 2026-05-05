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
using Statistics


"""
    true_dro_benders_optimize!(td::TrueDROData; ...)

Run outer Benders.

# Arguments
- `mip_optimizer`: OMP MILP solver
- `nlp_optimizer`: bilinear subproblem solver (needs NonConvex=2)
- `nonconvex_attr`: defaults to `"NonConvex" => 2`
- `sub_time_limit`: initial time limit (seconds) for bilinear subproblem.
  `nothing` = no limit. Incumbent에서 valid cut 생성, UB는 OPTIMAL만.
  Boost (same x + TIME_LIMIT 반복): time limit 해제 + MIPGap=0.5%, 수렴 tol도 0.5%로 완화.
- `mini_benders`: LP-based inner loop with fixed α
- `lp_optimizer`: LP solver for mini-benders
- `max_mini_benders_iter`: mini-benders phase 반복 횟수 (default 5)
- `inexact`: true이면 3회 중 2회는 OptimalityTarget=1 (local opt)으로 빠르게 풀고,
  3회째에만 global opt. Local opt에서도 valid cut 생성 (feasible point of max subproblem).
  UB는 global solve에서만 갱신.
- `strengthen_cuts`: `:none` (default), `:mw` (cut strengthening).
  `:mw` — outer bilinear: Sherali perturbation (x_pert로 추가 solve, constraint 변경 없음).
         mini-benders: MW (ISP-L/F 독립 LP Phase 2, joint Pareto-optimality 미보장).
- `valid_inequality`: `:none` (default), `:mincut`.
  `:mincut` — Phase 1 (all S, 1회) + Phase 2B (comp-min + α*, 매 iter) min-cut valid inequalities.

Returns Dict with :status, :Z0, :x, :α, :lower_bound, :upper_bound, :iters, :history.
"""
function true_dro_benders_optimize!(td::TrueDROData;
        mip_optimizer, nlp_optimizer,
        max_iter=1e+3, tol=1e-4, verbose=true,
        sub_verbose=false,
        nonconvex_attr=("NonConvex" => 2),
        sub_time_limit=nothing,
        mini_benders::Bool=false,
        lp_optimizer=nothing,
        max_mini_benders_iter::Int=5,
        inexact::Bool=false,
        strengthen_cuts::Symbol=:none,
        valid_inequality::Symbol=:none,
        add_objF_vi::Bool=false,
        phase2B_vi::Bool=false,
        source_sink_cut::Union{Nothing, Dict}=nothing)

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
    omp_model, omp_vars = build_true_dro_omp(td; optimizer=mip_optimizer, silent=true)

    # ---- Source/Sink connectivity cut ----
    if source_sink_cut !== nothing
        x = omp_vars[:x]
        if haskey(source_sink_cut, :source_arcs)
            src_arcs = source_sink_cut[:source_arcs]
            @constraint(omp_model, sum(x[a] for a in src_arcs) <= length(src_arcs) - 1)
            if verbose
                @info "Source-sink cut: Σx[source_arcs] ≤ $(length(src_arcs)-1) (arcs=$src_arcs)"
            end
        end
        if haskey(source_sink_cut, :sink_arcs)
            snk_arcs = source_sink_cut[:sink_arcs]
            @constraint(omp_model, sum(x[a] for a in snk_arcs) <= length(snk_arcs) - 1)
            if verbose
                @info "Source-sink cut: Σx[sink_arcs] ≤ $(length(snk_arcs)-1) (arcs=$snk_arcs)"
            end
        end
    end

    # ---- Min-cut valid inequality: Phase 1 (all S, 1회) ----
    local arc_topo
    if valid_inequality == :mincut
        add_phase1_mincut_vi!(omp_model, omp_vars, td)
        arc_topo = extract_arc_topology(td.Ny, td.nv1)
        if verbose
            @info "Min-cut VI enabled: Phase 1 (all S) added"
        end
    end

    # ---- Build subproblem: global model (no ρ bound) ----
    x_init = zeros(K)
    _use_nominal_compact = (td.eps_hat == 0.0 && td.eps_tilde == 0.0)
    _use_single_compact = (td.eps_tilde == 0.0)  # nominal도 포함
    if _use_nominal_compact
        _sub_builder = build_true_dro_subproblem_nominal
        verbose && @info "ε̂=ε̃=0 detected → using nominal compact subproblem (pure LP, no bilinear)"
    elseif _use_single_compact
        _sub_builder = build_true_dro_subproblem_single
        verbose && @info "ε̃=0 detected → using compact single-layer subproblem (no ζF bilinear)"
    else
        _sub_builder = build_true_dro_subproblem
    end
    sub_model, sub_vars = _sub_builder(td, x_init; optimizer=nlp_optimizer, silent=!sub_verbose,
                                       add_objF_vi=add_objF_vi)
    if nonconvex_attr !== nothing
        try
            set_optimizer_attribute(sub_model, nonconvex_attr.first, nonconvex_attr.second)
        catch err
            @warn "Could not set $(nonconvex_attr.first) on subproblem: $err"
        end
    end
    # set_optimizer_attribute(sub_model, "MIQCPMethod", 1)
    set_optimizer_attribute(sub_model, "LogFile", "global_miqcp.log")

    # ---- Build subproblem: local model (ρ ≤ M, local opt에서 ρ 폭발 방지) ----
    local sub_model_local, sub_vars_local
    const_rho_bound = 10.0
    if inexact
        # Local solve (OptimalityTarget=1)에는 VI 넣지 않음: Gurobi local opt가 VI 하에서
        # feasible point 탐색 실패 (ITERATION_LIMIT) 관측됨.
        sub_model_local, sub_vars_local = _sub_builder(td, x_init;
            optimizer=nlp_optimizer, silent=!sub_verbose, rho_upper_bound=const_rho_bound,
            add_objF_vi=false)
        if nonconvex_attr !== nothing
            try
                set_optimizer_attribute(sub_model_local, nonconvex_attr.first, nonconvex_attr.second)
            catch err
                @warn "Could not set $(nonconvex_attr.first) on local subproblem: $err"
            end
        end
        set_optimizer_attribute(sub_model_local, "OptimalityTarget", 1)
    end

    # ---- Adaptive time limit state ----
    current_time_limit = sub_time_limit  # nothing = unlimited
    is_boost = false                     # boost 상태 플래그
    const_boost_mipgap = _use_single_compact ? 1e-2 : 5e-3  # single-layer: 1%, else: 0.5%
    const_boost_time_limit = 3600.0  # boost 시 time limit (single/double 모두 3600s)
    prev_x_global = nothing      # 이전 global iter의 x_sol
    prev_t0_global = -Inf        # 이전 global iter의 t₀
    prev_global_was_timelimit = false  # 이전 global iter TIME_LIMIT 여부

    # ---- Mini-Benders: build ISP-L and ISP-F LP models once ----
    local isp_l_model, isp_l_vars, isp_f_model, isp_f_vars
    # α-step LP fallback: QCP fix() infeasible 오판 시 사용
    local astep_lp_model, astep_lp_vars
    if mini_benders
        α_init = zeros(K)
        isp_l_model, isp_l_vars = build_true_dro_isp_leader(td, x_init, α_init; optimizer=lp_optimizer)
        isp_f_model, isp_f_vars = build_true_dro_isp_follower(td, x_init, α_init; optimizer=lp_optimizer)
        # α-step LP: a,d 파라미터로 고정한 순수 LP (pre-build)
        a_dummy = fill(1.0 / td.S, td.S)
        d_dummy = fill(1.0 / td.S, td.S)
        astep_lp_model, astep_lp_vars = build_alpha_step_lp(td, x_init, a_dummy, d_dummy;
                                                              optimizer=lp_optimizer)
        if verbose
            @info "Mini-Benders enabled: max_mini_iter=$max_mini_benders_iter"
        end
    end

    history = Dict(
        :lower_bounds => Float64[],
        :upper_bounds => Float64[],
        :Z0_vals => Float64[],
        :wall_times => Float64[],
        :omp_times => Float64[],
        :sub_times => Float64[],
        :sub_is_exact => Bool[],
    )

    lower_bound = -Inf
    upper_bound = Inf
    best_x = zeros(K)
    best_α = zeros(K)
    cut_count = 0
    wall_start = time()

    for iter in 1:max_iter
        if verbose
            @printf("\n=== True-DRO Benders iter %d ===\n", iter)
            flush(stdout)
        end

        # ---- Solve OMP ----
        t_omp = @elapsed optimize!(omp_model)
        st = termination_status(omp_model)
        if st != MOI.OPTIMAL
            error("True-DRO OMP not optimal: $st (iter=$iter)")
        end

        x_sol = round.([value(omp_vars[:x][k]) for k in 1:K])
        t0_val = objective_value(omp_model)
        lower_bound = t0_val

        if verbose
            x_int = round.(Int, x_sol)
            @printf("  OMP: t₀=%.6f, UB=%.6f, gap=%.2e, x=%s (%.3fs)\n",
                    t0_val, upper_bound, abs(upper_bound - t0_val) / max(abs(upper_bound), 1e-10),
                    string(x_int), t_omp)
            flush(stdout)
        end

        # ---- Inexact mode: select model (global vs local with ρ bound) ----
        is_global_iter = true
        if inexact
            is_global_iter = (iter % const_inexact_cycle == 0)
        end
        cur_sub_model = is_global_iter ? sub_model : sub_model_local
        cur_sub_vars  = is_global_iter ? sub_vars  : sub_vars_local

        # ---- Adaptive boost 판정 (subproblem solve 전) ----
        # Boost: same x 반복 + prev TIME_LIMIT → time limit 해제 + MIPGap=0.5%
        # 비-boost: 기본 time limit + MIPGap=0 (global opt)
        is_boost = false
        if sub_time_limit !== nothing && is_global_iter
            t0_change = abs(t0_val - prev_t0_global) / max(abs(prev_t0_global), 1e-10)
            if prev_x_global !== nothing && x_sol == prev_x_global && prev_global_was_timelimit && t0_change < 1e-3
                is_boost = true
                if verbose
                    tl_str = const_boost_time_limit === nothing ? "off" : @sprintf("%.0fs", const_boost_time_limit)
                    @printf("  → Boost: time limit %s + MIPGap=%.1f%% (same x + prev TIME_LIMIT)\n",
                            tl_str, const_boost_mipgap * 100)
                    flush(stdout)
                end
            end
        end

        # ---- Set subproblem time limit & MIPGap ----
        if is_boost
            set_time_limit_sec(cur_sub_model, const_boost_time_limit)
            set_optimizer_attribute(cur_sub_model, "MIPGap", const_boost_mipgap)
            set_optimizer_attribute(cur_sub_model, "MIPFocus", 3)
            set_optimizer_attribute(cur_sub_model, "OutputFlag", 1)
            # Boost early stop: subproblem(max)의 incumbent ≥ t₀ 이면 outer gap 닫힘
            set_optimizer_attribute(cur_sub_model, "BestObjStop", t0_val * (1 + tol))
        else
            effective_time_limit = is_global_iter ? current_time_limit : sub_time_limit
            if effective_time_limit !== nothing
                set_time_limit_sec(cur_sub_model, effective_time_limit)
            else
                set_time_limit_sec(cur_sub_model, nothing)
            end
            set_optimizer_attribute(cur_sub_model, "MIPGap", 1e-4)
            set_optimizer_attribute(cur_sub_model, "MIPFocus", 0)
            set_optimizer_attribute(cur_sub_model, "OutputFlag", 0)
            set_optimizer_attribute(cur_sub_model, "BestObjStop", Inf)
        end

        # ---- Solve subproblem ----
        t_sub = @elapsed begin
            sub_info = solve_true_dro_subproblem!(cur_sub_model, cur_sub_vars, td, x_sol;
                                                is_global=is_global_iter)
        end
        Z0_val = sub_info[:Z0_val]
        is_exact = sub_info[:is_optimal]

        # UB 갱신 (global iter만)
        # - OPTIMAL (exact): Z₀ exact → UB, best_x, best_α 모두 갱신
        # - TIME_LIMIT / Boost: Z0_bound 상한 → UB + best_x, best_α 갱신
        if is_global_iter
            if is_exact && !is_boost && Z0_val < upper_bound
                upper_bound = Z0_val
                best_x = copy(x_sol)
                best_α = copy(sub_info[:α_val])
            elseif (!is_exact || is_boost) && sub_info[:Z0_bound] < upper_bound
                upper_bound = sub_info[:Z0_bound]
                best_x = copy(x_sol)
                best_α = copy(sub_info[:α_val])
            end
        end

        push!(history[:lower_bounds], lower_bound)
        push!(history[:upper_bounds], upper_bound)
        push!(history[:Z0_vals], Z0_val)
        push!(history[:wall_times], time() - wall_start)
        push!(history[:omp_times], t_omp)
        push!(history[:sub_times], t_sub)
        push!(history[:sub_is_exact], is_exact)

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
            @printf("  Sub: Z₀=%.6f%s (%.1fs), α=[%s]\n", Z0_val, status_str, t_sub, α_str)
            @printf("  Iter %d: LB=%.6f  UB=%.6f  gap=%.2e\n",
                    iter, lower_bound, upper_bound, gap)
            flush(stdout)
        end

        # Boost TIME_LIMIT 조기종료: incumbent Z0_val vs LB gap이 tol 이내면 near-optimal
        if is_boost && !is_exact
            boost_gap = abs(Z0_val - lower_bound) / max(abs(Z0_val), 1e-10)
            if boost_gap <= tol
                wall_elapsed = time() - wall_start
                if verbose
                    @printf("  Boost TIME_LIMIT early stop: Z0_val=%.6f, LB=%.6f, gap=%.2e ≤ tol\n",
                            Z0_val, lower_bound, boost_gap)
                    flush(stdout)
                end
                # incumbent은 valid UB (feasible point of max problem)
                if Z0_val < upper_bound
                    upper_bound = Z0_val
                    best_x = copy(x_sol)
                    best_α = copy(sub_info[:α_val])
                end
                gap = abs(upper_bound - lower_bound) / max(abs(upper_bound), 1e-10)
            end
        end

        effective_tol = (is_boost && gap > tol) ? const_boost_mipgap : tol
        if gap <= effective_tol
            wall_elapsed = time() - wall_start
            if verbose
                boost_tag = is_boost && effective_tol > tol ? " [boost-tol]" : ""
                @printf("True-DRO Benders converged at iter %d (gap=%.2e)%s\n", iter, gap, boost_tag)
                flush(stdout)
                @printf("  Wall time: %.2f sec\n", wall_elapsed)
                omp_ts = history[:omp_times]
                sub_exact_mask = history[:sub_is_exact]
                sub_exact_ts = history[:sub_times][sub_exact_mask]
                @printf("  OMP  time: mean=%.3fs, median=%.3fs, max=%.3fs\n",
                        mean(omp_ts), median(omp_ts), maximum(omp_ts))
                if !isempty(sub_exact_ts)
                    @printf("  Sub  time (exact): mean=%.3fs, median=%.3fs, q90=%.3fs, max=%.3fs (n=%d/%d)\n",
                            mean(sub_exact_ts), median(sub_exact_ts),
                            quantile(sub_exact_ts, 0.9), maximum(sub_exact_ts),
                            length(sub_exact_ts), length(history[:sub_times]))
                end
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
                :wall_time => wall_elapsed,
            )
        end

        # ---- Adaptive: prev state 업데이트 ----
        if is_global_iter
            prev_x_global = copy(x_sol)
            prev_t0_global = t0_val
            prev_global_was_timelimit = !is_exact
        end

        # ---- Add outer cut from bilinear solve ----
        outer_cut = compute_true_dro_outer_cut(td, sub_info, x_sol)
        cut_count += 1
        add_true_dro_optimality_cut!(omp_model, omp_vars, outer_cut, cut_count)

        if verbose
            @printf("  Cut: intercept=%.6f, π_x_range=[%.4f, %.4f]\n",
                    outer_cut[:intercept],
                    minimum(outer_cut[:π_x]), maximum(outer_cut[:π_x]))
            flush(stdout)
        end

        # ---- Min-cut valid inequality: Phase 2B (comp-min + α*) ----
        if phase2B_vi && valid_inequality == :mincut && is_exact
            add_phase2B_mincut_vi!(omp_model, omp_vars, td, sub_info[:α_val], iter;
                                   arc_topology=arc_topo)
            if verbose
                @printf("  Phase2B-VI added (α from iter %d)\n", iter)
            end
        end

        # ---- ρ diagnostic (sub_verbose) ----
        if sub_verbose
            ρ̂1 = sub_info[:rho_hat_1_val]; ρ̂3 = sub_info[:rho_hat_3_val]
            ρ̃1 = sub_info[:rho_tilde_1_val]; ρ̃3 = sub_info[:rho_tilde_3_val]
            ρ01 = sub_info[:rho_psi0_1_val]; ρ03 = sub_info[:rho_psi0_3_val]
            φ̂U = td.phi_hat_U; φ̃U = td.phi_tilde_U; λU = td.lambda_U
            # 각 component별 π_x 기여도
            π_L = [φ̂U * (-sum(ρ̂1[k,:]) + sum(ρ̂3[k,:])) for k in 1:K]
            π_F = [φ̃U * (-sum(ρ̃1[k,:]) + sum(ρ̃3[k,:])) for k in 1:K]
            π_ψ = [λU * (-ρ01[k] + ρ03[k]) for k in 1:K]
            @printf("  [ρ-diag] φ̂U=%.2f, φ̃U=%.2f, λU=%.2f\n", φ̂U, φ̃U, λU)
            @printf("    π_L range=[%.4f, %.4f], |π_L|_∞=%.4f\n",
                    minimum(π_L), maximum(π_L), maximum(abs.(π_L)))
            @printf("    π_F range=[%.4f, %.4f], |π_F|_∞=%.4f\n",
                    minimum(π_F), maximum(π_F), maximum(abs.(π_F)))
            @printf("    π_ψ range=[%.4f, %.4f], |π_ψ|_∞=%.4f\n",
                    minimum(π_ψ), maximum(π_ψ), maximum(abs.(π_ψ)))
            @printf("    ρ̂1: [%.4f, %.4f], ρ̂3: [%.4f, %.4f]\n",
                    minimum(ρ̂1), maximum(ρ̂1), minimum(ρ̂3), maximum(ρ̂3))
            @printf("    ρ̃1: [%.4f, %.4f], ρ̃3: [%.4f, %.4f]\n",
                    minimum(ρ̃1), maximum(ρ̃1), minimum(ρ̃3), maximum(ρ̃3))
            @printf("    ρ⁰1: [%.4f, %.4f], ρ⁰3: [%.4f, %.4f]\n",
                    minimum(ρ01), maximum(ρ01), minimum(ρ03), maximum(ρ03))
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
                    t_mini_iter_start = time()
                    # Re-solve OMP with accumulated cuts → new x̄
                    t_omp_mb = @elapsed optimize!(omp_model)
                    push!(history[:omp_times], t_omp_mb)
                    st_mb = termination_status(omp_model)
                    if st_mb != MOI.OPTIMAL
                        if verbose
                            @printf("  %s-Benders[%d]: OMP %s, break\n", phase_tag, j, st_mb)
                        end
                        break
                    end

                    x_mb = round.([value(omp_vars[:x][k]) for k in 1:K])
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
                    local t_isp_l, t_isp_f
                    try
                        update_isp_leader_objective!(isp_l_model, isp_l_vars, td, x_mb)
                        t_isp_l = @elapsed (l_info = solve_isp_leader!(isp_l_model, isp_l_vars, td))

                        update_isp_follower_objective!(isp_f_model, isp_f_vars, td, x_mb)
                        t_isp_f = @elapsed (f_info = solve_isp_follower!(isp_f_model, isp_f_vars, td))
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

                    # ---- φ̂, φ̃ 실측 진단 (DL-2, DF-6 dual = primal φ) ----
                    # Max 문제의 ≤ constraint dual → non-positive. φ = -dual(con).
                    if sub_verbose && j == 1 && phase == 1
                        φ̂_vals = [-dual(isp_l_vars[:DL2][k, s]) for k in 1:K, s in 1:td.S]
                        φ̃_vals = [-dual(isp_f_vars[:DF6][k, s]) for k in 1:K, s in 1:td.S]
                        @printf("  [φ-diag] φ̂ actual: [%.4f, %.4f], φ̂U=%.2f\n",
                                minimum(φ̂_vals), maximum(φ̂_vals), td.phi_hat_U)
                        @printf("  [φ-diag] φ̃ actual: [%.4f, %.4f], φ̃U=%.2f\n",
                                minimum(φ̃_vals), maximum(φ̃_vals), td.phi_tilde_U)
                    end

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
                    local t_mw_l, t_mw_f
                    t_mw_l = 0.0; t_mw_f = 0.0
                    if strengthen_cuts == :mw
                        S = td.S
                        φ̂U = td.phi_hat_U
                        φ̃U = td.phi_tilde_U
                        λU = td.lambda_U

                        # ISP-L MW Phase 2
                        t_mw_l = @elapsed begin
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
                        end  # @elapsed t_mw_l

                        # ISP-F MW Phase 2
                        t_mw_f = @elapsed begin
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
                        end  # @elapsed t_mw_f

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
                        t_mini_iter = time() - t_mini_iter_start
                        @printf("  %s-Benders[%d] (%s): LB=%.6f, Z₀(α*)=%.6f, intercept=%.6f (%.3fs) [OMP=%.3f ISP-L=%.3f ISP-F=%.3f MW-L=%.3f MW-F=%.3f]\n",
                                phase_tag, j, cut_tag, lb_mb, Z0_mini, final_cut[:intercept], t_mini_iter,
                                t_omp_mb, t_isp_l, t_isp_f, t_mw_l, t_mw_f)
                        flush(stdout)
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

                # ---- α-step: fix (a*, d*) → solve over α → new α' ----
                # 1차: QCP sub_model에 fix(a,d) 시도.
                # Gurobi NonConvex=2 + fix()가 infeasible 오판하면
                # 2차: 별도 LP model (quadratic 없음)로 fallback.
                if phase == 1 && last_a_val !== nothing
                    S = td.S

                    # Clamp (ISP LP numerical noise 방지)
                    a_clamped = _use_nominal_compact ? td.q_hat : [max(last_a_val[s], sub_vars[:a_min][s]) for s in 1:S]
                    d_clamped = _use_single_compact ? td.q_hat : [max(last_d_val[s], sub_vars[:d_min][s]) for s in 1:S]

                    # OMP x̄ for objective
                    t_omp_alt = @elapsed optimize!(omp_model)
                    push!(history[:omp_times], t_omp_alt)
                    if termination_status(omp_model) != MOI.OPTIMAL
                        break
                    end
                    x_alt = round.([value(omp_vars[:x][k]) for k in 1:K])

                    # Nominal compact: a,d 모두 고정 → fix 불필요, 바로 sub solve
                    # Single compact: a만 fix, d 없음
                    # Full: a,d 모두 fix
                    local alt_info
                    qcp_ok = true
                    try
                        if !_use_nominal_compact
                            for s in 1:S
                                fix(sub_vars[:a][s], a_clamped[s]; force=true)
                            end
                        end
                        if !_use_single_compact
                            for s in 1:S
                                fix(sub_vars[:d][s], d_clamped[s]; force=true)
                            end
                        end
                        alt_info = solve_true_dro_subproblem!(sub_model, sub_vars, td, x_alt)
                    catch e
                        qcp_ok = false
                        if verbose
                            @printf("  α-step QCP failed (%s), fallback to LP\n",
                                    sprint(showerror, e))
                        end
                    finally
                        # 반드시 unfix (QCP 성공/실패 무관)
                        if !_use_nominal_compact
                            unfix.(sub_vars[:a])
                            set_lower_bound.(sub_vars[:a], sub_vars[:a_min])
                            set_upper_bound.(sub_vars[:a], sub_vars[:a_max])
                        end
                        if !_use_single_compact
                            unfix.(sub_vars[:d])
                            set_lower_bound.(sub_vars[:d], sub_vars[:d_min])
                            set_upper_bound.(sub_vars[:d], sub_vars[:d_max])
                        end
                    end

                    # 2차: LP fallback
                    if !qcp_ok
                        try
                            alt_info = solve_alpha_step_lp!(astep_lp_model, astep_lp_vars,
                                                            td, x_alt, a_clamped, d_clamped)
                        catch e2
                            if verbose
                                @printf("  α-step LP also failed (%s), skip\n",
                                        sprint(showerror, e2))
                            end
                            break
                        end
                    end

                    α_fixed = alt_info[:α_val]

                    # Cut from α-step
                    alt_cut = compute_true_dro_outer_cut(td, alt_info, x_alt)
                    cut_count += 1
                    add_true_dro_optimality_cut!(omp_model, omp_vars, alt_cut, cut_count)

                    if verbose
                        α_str = join([@sprintf("%.3f", a) for a in α_fixed], ",")
                        @printf("  α-step: Z₀=%.6f, α=[%s]%s\n",
                                alt_info[:Z0_val], α_str, qcp_ok ? "" : " [LP-fallback]")
                    end
                end
            end

            # Update lower_bound from final OMP state after mini-benders
            t_omp_final = @elapsed optimize!(omp_model)
            push!(history[:omp_times], t_omp_final)
            if termination_status(omp_model) == MOI.OPTIMAL
                lower_bound = max(lower_bound, objective_value(omp_model))
            end
            if verbose
                @printf("  OMP re-solve: %.3fs\n", t_omp_final)
                flush(stdout)
            end
        end
    end

    wall_elapsed = time() - wall_start
    @warn "True-DRO Benders did not converge in $max_iter iterations"
    if verbose
        @printf("  Wall time: %.2f sec\n", wall_elapsed)
        omp_ts = history[:omp_times]
        sub_exact_mask = history[:sub_is_exact]
        sub_exact_ts = history[:sub_times][sub_exact_mask]
        @printf("  OMP  time: mean=%.3fs, median=%.3fs, max=%.3fs\n",
                mean(omp_ts), median(omp_ts), maximum(omp_ts))
        if !isempty(sub_exact_ts)
            @printf("  Sub  time (exact): mean=%.3fs, median=%.3fs, q90=%.3fs, max=%.3fs (n=%d/%d)\n",
                    mean(sub_exact_ts), median(sub_exact_ts),
                    quantile(sub_exact_ts, 0.9), maximum(sub_exact_ts),
                    length(sub_exact_ts), length(history[:sub_times]))
        end
    end
    return Dict(
        :status => :MaxIter,
        :Z0 => upper_bound,
        :x => best_x,
        :α => best_α,
        :lower_bound => lower_bound,
        :upper_bound => upper_bound,
        :iters => max_iter,
        :history => history,
        :wall_time => wall_elapsed,
    )
end
