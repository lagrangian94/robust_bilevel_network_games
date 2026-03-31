"""
Branch-and-Benders (Single Tree) — Gurobi lazy callback 기반 구현.

기존 `tr_nested_benders_optimize!`의 outer loop를 Gurobi single-tree B&B + lazy callback으로 대체.
Inner loop (`tr_imp_optimize!`)는 그대로 재사용.

## 구조
- `build_optimality_cut_expr`: `add_optimality_cuts!`에서 AffExpr 구성 로직만 분리 (constraint 없이)
- `evaluate_affexpr_at_callback`: AffExpr를 callback_value로 평가
- `extract_cut_coefficients_from_isp`: ISP solved 상태에서 outer cut coefficients 추출
- `branch_and_benders_optimize!`: 메인 함수 (Phase 1 LP warming + Phase 2 single tree)

## Callback: Two-Pass Filter
1. Pass 1 (cheap rejection): inexact IMP 또는 subgradient로 빠르게 rejection
2. α-history cheap cuts: 이전 callback의 α로 ISP-only evaluate
3. Pass 2 (exact): full IMP solve로 정확한 cut 생성. Incumbent acceptance는 여기서만.
"""

using JuMP
using LinearAlgebra
using SparseArrays
using Gurobi
using Mosek, MosekTools


"""
    build_optimality_cut_expr(omp_vars, cut_info, diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h_var, S)

`add_optimality_cuts!` (strict_benders.jl L152-163)의 AffExpr 구성 로직.
`@constraint` 없이 AffExpr만 반환. Callback에서 lazy constraint로 submit하기 위한 용도.
"""
function build_optimality_cut_expr(omp_vars, cut_info,
    diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h_var, S)

    opt_cut = AffExpr(0.0)
    for s in 1:S
        add_to_expression!(opt_cut, -ϕU, dot(cut_info[:Uhat1][s,:,:], diag_x_E))
        add_to_expression!(opt_cut, -ϕU, dot(cut_info[:Uhat3][s,:,:], E - diag_x_E))
        add_to_expression!(opt_cut, cut_info[:intercept_l][s])
        add_to_expression!(opt_cut, -ϕU, dot(cut_info[:Utilde1][s,:,:], diag_x_E))
        add_to_expression!(opt_cut, -ϕU, dot(cut_info[:Utilde3][s,:,:], E - diag_x_E))
        add_to_expression!(opt_cut, dot(cut_info[:Ztilde1_3][s,:,:], diag_λ_ψ * diagm(xi_bar[s])))
        add_to_expression!(opt_cut, dot(d0, cut_info[:βtilde1_1][s,:]), λ_var)
        add_to_expression!(opt_cut, -1.0, dot(cut_info[:βtilde1_3][s,:], h_var + diag_λ_ψ * xi_bar[s]))
        add_to_expression!(opt_cut, cut_info[:intercept_f][s])
    end
    return opt_cut
end


"""
    evaluate_affexpr_at_callback(expr::AffExpr, cb_data)

AffExpr를 `callback_value`로 평가하여 Float64 반환.
Callback 내부에서 violation check에 사용.
"""
function evaluate_affexpr_at_callback(expr::AffExpr, cb_data)
    result = expr.constant
    for (var, coef) in expr.terms
        result += coef * callback_value(cb_data, var)
    end
    return result
end


"""
    extract_cut_coefficients_from_isp(isp_leader_instances, isp_follower_instances, S)

`extract_outer_cut_from_current_isp` wrapper.
이미 solved 상태인 ISP에서 outer cut coefficients를 추출 (re-solve 없이).
"""
function extract_cut_coefficients_from_isp(isp_leader_instances::Dict, isp_follower_instances::Dict, S::Int)
    return extract_outer_cut_from_current_isp(isp_leader_instances, isp_follower_instances, S)
end


"""
    branch_and_benders_optimize!(omp_model, omp_vars, network, ϕU, λU, γ, w, v, uncertainty_set; ...)

Branch-and-Benders (single tree) 메인 함수.
Phase 1: LP warming으로 initial cuts 축적 (optional).
Phase 2: Gurobi single tree + lazy callback으로 B&B 수행.

Outer trust region은 B&B tree와 충돌하므로 제거. Inner trust region은 유지.
"""
function branch_and_benders_optimize!(
    omp_model::Model, omp_vars::Dict,
    network, ϕU, λU, γ, w, v, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer,
    conic_optimizer=Mosek.Optimizer,
    inner_tr=true,                  # IMP 내부 α trust region
    strengthen_cuts::Symbol=:none,  # :none | :mw | :sherali
    use_alpha_history::Bool=false,  # α-fixed cheap cuts
    max_alpha_history::Int=3,
    inexact_every_n::Int=1,         # 1=항상exact, 3=3번에1번exact
    use_subgradient::Bool=false,    # inexact phase에서 subgradient 사용
    subgrad_step_size::Float64=0.1,
    lp_warmup_iters::Int=1,         # Phase 1 LP warming iterations (≥1)
    mipnode_freq::Int=0,            # MIPNODE user cut frequency (0=off, N=every N-th call)
    tol::Float64=1e-4,
    πU=ϕU, yU=ϕU, ytsU=ϕU,
    parallel::Bool=false)

    lp_warmup_iters >= 1 || error("lp_warmup_iters must be ≥ 1 (Gurobi lazy callback requires at least one regular cut on epigraph)")

    # ====================================================================
    # 초기화 (기존 tr_nested_benders_optimize! 패턴)
    # ====================================================================
    S = length(uncertainty_set[:xi_bar])
    num_arcs = length(network.arcs) - 1
    E = ones(num_arcs, num_arcs + 1)
    d0 = zeros(num_arcs + 1); d0[end] = 1.0
    xi_bar = uncertainty_set[:xi_bar]

    x, h, λ_var, ψ0 = omp_vars[:x], omp_vars[:h], omp_vars[:λ], omp_vars[:ψ0]
    t_0 = omp_vars[:t_0]
    diag_x_E = Diagonal(x) * E
    diag_λ_ψ = Diagonal(λ_var * ones(num_arcs) - v .* ψ0)

    isp_data = Dict(
        :E => E, :network => network, :ϕU => ϕU,
        :πU => πU, :yU => yU, :ytsU => ytsU,
        :λU => λU, :γ => γ, :w => w, :v => v,
        :uncertainty_set => uncertainty_set, :d0 => d0, :S => S)

    # OMP 초기해 (IMP/ISP 구축용)
    st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars)

    # IMP 구축
    imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
        mip_optimizer=mip_optimizer)
    _, α_sol = initialize_imp(imp_model, imp_vars)

    # ISP 구축
    leader_instances, follower_instances = initialize_isp(
        network, S, ϕU, λU, γ, w, v, uncertainty_set;
        conic_optimizer=conic_optimizer,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol,
        πU=πU, yU=yU, ytsU=ytsU)

    # IMP cuts 상태 (callback 간 유지)
    imp_cuts = Dict{Symbol, Any}(:old_tr_constraints => nothing, :old_cuts => Dict())

    # MW core points (pre-compute)
    core_points = nothing
    if strengthen_cuts != :none
        interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
        core_points = generate_core_points(network, γ, λU, w, v;
            interdictable_idx=interdictable_idx, strategy=:interior)
    end

    # 결과 dict
    cut_pool = Vector{Dict{Symbol,Any}}()
    result = Dict(
        :cuts => Dict(),
        :cut_pool => cut_pool,
        :inner_iter => Int[],
        :lazy_cut_count => 0,
        :callback_count => 0,
        :pass1_rejections => 0,
        :usercut_count => 0,
        :mipnode_calls => 0,
    )

    time_start = time()

    # ====================================================================
    # Phase 1: LP Warming (optional — initial cut seeding)
    # ====================================================================
    if lp_warmup_iters > 0
        @info "Phase 1: LP warming ($lp_warmup_iters iterations)"
        undo_relax = relax_integrality(omp_model)

        for k in 1:lp_warmup_iters
            optimize!(omp_model)
            st_lp = MOI.get(omp_model, MOI.TerminationStatus())
            if st_lp ∉ (MOI.OPTIMAL, MOI.DUAL_INFEASIBLE)
                @warn "LP warmup: OMP status $st_lp at iter $k"
                break
            end
            x_lp = value.(x)
            h_lp, λ_lp, ψ0_lp = value.(h), value(λ_var), value.(ψ0)

            status_k, cut_info_k = tr_imp_optimize!(
                imp_model, imp_vars, leader_instances, follower_instances;
                isp_data=isp_data,
                λ_sol=λ_lp, x_sol=x_lp, h_sol=h_lp, ψ0_sol=ψ0_lp,
                outer_iter=k, imp_cuts=imp_cuts, inner_tr=inner_tr,
                parallel=parallel)

            if status_k == :OptimalityCut
                # evaluate_master_opt_cut으로 validated outer cut coefficients 추출
                outer_cut_info_k = evaluate_master_opt_cut(
                    leader_instances, follower_instances,
                    isp_data, cut_info_k, k; parallel=parallel)

                add_optimality_cuts!(omp_model, omp_vars, outer_cut_info_k,
                    diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h, S, k;
                    prefix="warmup", result_cuts=result[:cuts])
                push!(cut_pool, Dict{Symbol,Any}(
                    :type => :warmup, :iter => k, :cut_info => deepcopy(outer_cut_info_k),
                    :x_sol => copy(x_lp), :λ_sol => λ_lp, :h_sol => copy(h_lp), :ψ0_sol => copy(ψ0_lp),
                    :α_sol => haskey(cut_info_k, :α_sol) ? copy(cut_info_k[:α_sol]) : nothing,
                    :subprob_obj => cut_info_k[:obj_val]))

                # IMP cuts 업데이트
                if !get(cut_info_k, :subgradient, false)
                    imp_cuts[:old_cuts] = cut_info_k[:cuts]
                    if inner_tr && cut_info_k[:tr_constraints] !== nothing
                        imp_cuts[:old_tr_constraints] = cut_info_k[:tr_constraints]
                    end
                end
                @info "  Warmup iter $k: cut added (obj=$(round(cut_info_k[:obj_val], digits=6)))"
            end
        end

        # x를 다시 binary로 복원
        undo_relax()
        @info "Phase 1 complete: $(length(result[:cuts])) initial cuts seeded"
    end

    # ====================================================================
    # Phase 2: Branch-and-Benders (single tree)
    # ====================================================================
    @info "Phase 2: Branch-and-Benders (single tree)"

    # Closure 공유 상태
    cut_count = Ref(0)
    callback_count = Ref(0)
    usercut_count = Ref(0)
    mipnode_call_count = Ref(0)
    # outer_iter는 warmup 이후부터 연속 번호: tr_imp_optimize!에서 outer_iter>1일 때
    # old IMP cuts를 삭제하므로, 항상 >1이어야 warmup의 stale cuts가 삭제됨.
    imp_call_count = Ref(lp_warmup_iters + 1)
    ub_tracker = Ref(Inf)
    α_history = Vector{Vector{Float64}}()
    α_persistent = nothing  # subgradient heuristic α state

    function benders_lazy_callback(cb_data)
        status = callback_node_status(cb_data, omp_model)
        if status != MOI.CALLBACK_NODE_STATUS_INTEGER
            return
        end

        callback_count[] += 1
        cb_idx = callback_count[]

        # 현재 해 추출
        x_k = round.([callback_value(cb_data, x[i]) for i in 1:num_arcs])
        h_k = [callback_value(cb_data, h[i]) for i in 1:num_arcs]
        λ_k = callback_value(cb_data, λ_var)
        ψ0_k = [callback_value(cb_data, ψ0[i]) for i in 1:num_arcs]
        t_0_k = callback_value(cb_data, t_0)

        # ── Pass 1: Cheap rejection filter (inexact or subgradient) ──
        if inexact_every_n > 1 && (cb_idx % inexact_every_n != 0) && cb_idx > 1
            local Q_lower, pass1_outer_cut_info
            local pass1_inner_iter = 1
            local pass1_ok = true

            if use_subgradient
                # Subgradient heuristic
                if α_persistent === nothing
                    α_persistent = fill(isp_data[:w] / S / num_arcs, num_arcs)
                end
                _, sg_result = subgradient_alpha_step!(
                    α_persistent,
                    leader_instances, follower_instances, isp_data;
                    λ_sol=λ_k, x_sol=x_k, h_sol=h_k, ψ0_sol=ψ0_k,
                    step_size=subgrad_step_size, parallel=parallel)
                Q_lower = sg_result[:obj_val]
                pass1_outer_cut_info = sg_result[:outer_cut_info]
                α_persistent = sg_result[:α_new]  # update persistent state
            else
                # Inexact IMP
                imp_call_count[] += 1
                status_inex, inex_result = tr_imp_optimize!(
                    imp_model, imp_vars, leader_instances, follower_instances;
                    isp_data=isp_data,
                    λ_sol=λ_k, x_sol=x_k, h_sol=h_k, ψ0_sol=ψ0_k,
                    outer_iter=imp_call_count[],
                    imp_cuts=imp_cuts, inner_tr=inner_tr,
                    parallel=parallel, inexact=true)

                if status_inex != :OptimalityCut
                    @warn "Pass 1 inexact IMP failed, falling through to exact"
                    pass1_ok = false
                else
                    Q_lower = inex_result[:obj_val]
                    pass1_inner_iter = inex_result[:iter]
                    # IMP cuts 업데이트 (inexact도 cuts 생성)
                    if !get(inex_result, :subgradient, false)
                        imp_cuts[:old_cuts] = inex_result[:cuts]
                        if inner_tr && inex_result[:tr_constraints] !== nothing
                            imp_cuts[:old_tr_constraints] = inex_result[:tr_constraints]
                        end
                    end
                    pass1_outer_cut_info = extract_cut_coefficients_from_isp(
                        leader_instances, follower_instances, S)
                end
            end

            # Rejection check: t_0은 raw sum, Q_lower는 /S average
            if pass1_ok && t_0_k < Q_lower * S - tol * S
                # Valid cut from inexact/subgradient → reject incumbent
                opt_expr = build_optimality_cut_expr(
                    omp_vars, pass1_outer_cut_info,
                    diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h, S)
                con = @build_constraint(t_0 >= opt_expr)
                MOI.submit(omp_model, MOI.LazyConstraint(cb_data), con)
                cut_count[] += 1
                push!(cut_pool, Dict{Symbol,Any}(
                    :type => :pass1_lazy, :cb => cb_idx, :cut_info => deepcopy(pass1_outer_cut_info),
                    :x_sol => copy(x_k), :λ_sol => λ_k, :h_sol => copy(h_k), :ψ0_sol => copy(ψ0_k),
                    :Q => Q_lower))
                result[:pass1_rejections] += 1
                push!(result[:inner_iter], pass1_inner_iter)
                @info "  [CB $cb_idx] Pass 1 rejection: Q_lower=$(round(Q_lower, digits=6)), t_0/S=$(round(t_0_k/S, digits=6))"
                return
            end
            # Inexact couldn't reject (or failed) → fall through to exact
        end

        # ── α-history cheap cuts (optional: IMP 없이 ISP만 evaluate) ──
        if use_alpha_history && !isempty(α_history)
            R = uncertainty_set[:R]
            r_dict = uncertainty_set[:r_dict]
            epsilon = uncertainty_set[:epsilon]

            recent_αs = α_history[max(1, end-max_alpha_history+1):end]
            for α_old in recent_αs
                _, _ = solve_scenarios(S; parallel=parallel) do s
                    U_s = Dict(:R => Dict(:1=>R[s]),
                               :r_dict => Dict(:1=>r_dict[s]),
                               :xi_bar => Dict(:1=>xi_bar[s]),
                               :epsilon => epsilon)
                    isp_leader_optimize!(leader_instances[s][1], leader_instances[s][2];
                        isp_data=isp_data, uncertainty_set=U_s,
                        λ_sol=λ_k, x_sol=x_k, h_sol=h_k, ψ0_sol=ψ0_k, α_sol=α_old)
                    isp_follower_optimize!(follower_instances[s][1], follower_instances[s][2];
                        isp_data=isp_data, uncertainty_set=U_s,
                        λ_sol=λ_k, x_sol=x_k, h_sol=h_k, ψ0_sol=ψ0_k, α_sol=α_old)
                    return (true, nothing)
                end
                α_fixed_cut_info = extract_cut_coefficients_from_isp(
                    leader_instances, follower_instances, S)

                α_expr = build_optimality_cut_expr(
                    omp_vars, α_fixed_cut_info,
                    diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h, S)
                α_con = @build_constraint(t_0 >= α_expr)
                MOI.submit(omp_model, MOI.LazyConstraint(cb_data), α_con)
                cut_count[] += 1
                push!(cut_pool, Dict{Symbol,Any}(
                    :type => :α_history_lazy, :cb => cb_idx, :cut_info => deepcopy(α_fixed_cut_info),
                    :x_sol => copy(x_k), :λ_sol => λ_k, :h_sol => copy(h_k), :ψ0_sol => copy(ψ0_k),
                    :α_sol => copy(α_old)))

                # MW strengthening for α-history cut (기존 alpha_fixed_benders_phase! 패턴)
                if strengthen_cuts != :none && core_points !== nothing
                    α_hist_cut_info = Dict(:α_sol => α_old)
                    for (cp_idx, cp) in enumerate(core_points)
                        local str_info
                        if strengthen_cuts == :mw
                            str_info = evaluate_mw_opt_cut(
                                leader_instances, follower_instances,
                                isp_data, α_hist_cut_info, cut_count[];
                                x_sol=x_k, λ_sol=λ_k, h_sol=h_k, ψ0_sol=ψ0_k,
                                x_core=cp.x, λ_core=cp.λ,
                                h_core=cp.h, ψ0_core=cp.ψ0,
                                parallel=parallel)
                        elseif strengthen_cuts == :sherali
                            str_info = evaluate_sherali_opt_cut(
                                leader_instances, follower_instances,
                                isp_data, α_hist_cut_info, cut_count[];
                                x_sol=x_k, λ_sol=λ_k, h_sol=h_k, ψ0_sol=ψ0_k,
                                x_core=cp.x, λ_core=cp.λ,
                                h_core=cp.h, ψ0_core=cp.ψ0,
                                parallel=parallel)
                        end
                        str_expr = build_optimality_cut_expr(
                            omp_vars, str_info,
                            diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h, S)
                        str_con = @build_constraint(t_0 >= str_expr)
                        MOI.submit(omp_model, MOI.LazyConstraint(cb_data), str_con)
                        cut_count[] += 1
                        push!(cut_pool, Dict{Symbol,Any}(
                            :type => :α_history_mw_lazy, :cb => cb_idx, :cut_info => deepcopy(str_info),
                            :x_sol => copy(x_k), :λ_sol => λ_k, :h_sol => copy(h_k), :ψ0_sol => copy(ψ0_k),
                            :α_sol => copy(α_old), :core_point => cp_idx))
                    end
                end
            end
        end

        # ── Pass 2: Exact IMP solve ──
        imp_call_count[] += 1
        status_imp, cut_info = tr_imp_optimize!(
            imp_model, imp_vars, leader_instances, follower_instances;
            isp_data=isp_data,
            λ_sol=λ_k, x_sol=x_k, h_sol=h_k, ψ0_sol=ψ0_k,
            outer_iter=imp_call_count[],
            imp_cuts=imp_cuts, inner_tr=inner_tr,
            parallel=parallel, inexact=false)

        if status_imp != :OptimalityCut
            @warn "  [CB $cb_idx] Exact IMP did not converge"
            return
        end

        Q_k = cut_info[:obj_val]   # /S averaged
        ub_tracker[] = min(ub_tracker[], Q_k)
        push!(result[:inner_iter], cut_info[:iter])

        # IMP cuts 업데이트 (다음 callback에서 삭제용)
        imp_cuts[:old_cuts] = cut_info[:cuts]
        if inner_tr && cut_info[:tr_constraints] !== nothing
            imp_cuts[:old_tr_constraints] = cut_info[:tr_constraints]
        end

        # α_history 업데이트
        if use_alpha_history && haskey(cut_info, :α_sol)
            push!(α_history, copy(cut_info[:α_sol]))
            while length(α_history) > max_alpha_history
                popfirst!(α_history)
            end
        end

        # Subgradient recalibration
        if use_subgradient
            α_persistent = copy(cut_info[:α_sol])
        end

        # Violation check: t_0은 raw sum, Q_k는 /S average
        if t_0_k < Q_k * S - tol * S
            # Outer cut coefficients 추출 (validated)
            outer_cut_info = evaluate_master_opt_cut(
                leader_instances, follower_instances,
                isp_data, cut_info, cut_count[] + 1; parallel=parallel)

            opt_expr = build_optimality_cut_expr(
                omp_vars, outer_cut_info,
                diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h, S)
            con = @build_constraint(t_0 >= opt_expr)
            MOI.submit(omp_model, MOI.LazyConstraint(cb_data), con)
            cut_count[] += 1
            push!(cut_pool, Dict{Symbol,Any}(
                :type => :exact_lazy, :cb => cb_idx, :cut_info => deepcopy(outer_cut_info),
                :x_sol => copy(x_k), :λ_sol => λ_k, :h_sol => copy(h_k), :ψ0_sol => copy(ψ0_k),
                :α_sol => haskey(cut_info, :α_sol) ? copy(cut_info[:α_sol]) : nothing,
                :Q => Q_k, :inner_iter => cut_info[:iter]))

            @info "  [CB $cb_idx] Exact cut: Q=$(round(Q_k, digits=6)), t_0/S=$(round(t_0_k/S, digits=6)), inner_iter=$(cut_info[:iter])"

            # ── MW strengthening (optional) ──
            if strengthen_cuts != :none && core_points !== nothing
                for (cp_idx, cp) in enumerate(core_points)
                    local mw_info
                    if strengthen_cuts == :mw
                        mw_info = evaluate_mw_opt_cut(
                            leader_instances, follower_instances,
                            isp_data, cut_info, cut_count[];
                            x_sol=x_k, λ_sol=λ_k, h_sol=h_k, ψ0_sol=ψ0_k,
                            x_core=cp.x, λ_core=cp.λ,
                            h_core=cp.h, ψ0_core=cp.ψ0,
                            parallel=parallel)
                    elseif strengthen_cuts == :sherali
                        mw_info = evaluate_sherali_opt_cut(
                            leader_instances, follower_instances,
                            isp_data, cut_info, cut_count[];
                            x_sol=x_k, λ_sol=λ_k, h_sol=h_k, ψ0_sol=ψ0_k,
                            x_core=cp.x, λ_core=cp.λ,
                            h_core=cp.h, ψ0_core=cp.ψ0,
                            parallel=parallel)
                    end
                    mw_expr = build_optimality_cut_expr(
                        omp_vars, mw_info,
                        diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h, S)
                    mw_con = @build_constraint(t_0 >= mw_expr)
                    MOI.submit(omp_model, MOI.LazyConstraint(cb_data), mw_con)
                    cut_count[] += 1
                    push!(cut_pool, Dict{Symbol,Any}(
                        :type => :exact_mw_lazy, :cb => cb_idx, :cut_info => deepcopy(mw_info),
                        :x_sol => copy(x_k), :λ_sol => λ_k, :h_sol => copy(h_k), :ψ0_sol => copy(ψ0_k),
                        :core_point => cp_idx))
                end
            end
        else
            @info "  [CB $cb_idx] No violation: Q=$(round(Q_k, digits=6)), t_0/S=$(round(t_0_k/S, digits=6))"
        end
    end

    # ====================================================================
    # MIPNODE User Cut Callback (LP bound 개선용)
    # ====================================================================
    if mipnode_freq > 0
        # MIPNODE용 별도 IMP/ISP 인스턴스 (lazy callback과 공유 불가 — 동시 호출 방지)
        imp_model_uc, imp_vars_uc = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
            mip_optimizer=mip_optimizer)
        _, _ = initialize_imp(imp_model_uc, imp_vars_uc)

        leader_instances_uc, follower_instances_uc = initialize_isp(
            network, S, ϕU, λU, γ, w, v, uncertainty_set;
            conic_optimizer=conic_optimizer,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol,
            πU=πU, yU=yU, ytsU=ytsU)

        imp_cuts_uc = Dict{Symbol, Any}(:old_tr_constraints => nothing, :old_cuts => Dict())
        imp_call_count_uc = Ref(1)

        function benders_usercut_callback(cb_data)
            mipnode_call_count[] += 1
            # Frequency gating: mipnode_freq번 중 1번만 실행
            if mipnode_call_count[] % mipnode_freq != 1 && mipnode_call_count[] != 1
                return
            end

            # 현재 LP relaxation 해 추출 (fractional — rounding 하지 않음)
            x_frac = [callback_value(cb_data, x[i]) for i in 1:num_arcs]
            h_frac = [callback_value(cb_data, h[i]) for i in 1:num_arcs]
            λ_frac = callback_value(cb_data, λ_var)
            ψ0_frac = [callback_value(cb_data, ψ0[i]) for i in 1:num_arcs]
            t_0_frac = callback_value(cb_data, t_0)

            # IMP+ISP solve (fractional x로)
            imp_call_count_uc[] += 1
            status_uc, cut_info_uc = tr_imp_optimize!(
                imp_model_uc, imp_vars_uc, leader_instances_uc, follower_instances_uc;
                isp_data=isp_data,
                λ_sol=λ_frac, x_sol=x_frac, h_sol=h_frac, ψ0_sol=ψ0_frac,
                outer_iter=imp_call_count_uc[],
                imp_cuts=imp_cuts_uc, inner_tr=inner_tr,
                parallel=parallel, inexact=false)

            if status_uc != :OptimalityCut
                return
            end

            Q_frac = cut_info_uc[:obj_val]

            # IMP cuts 업데이트
            imp_cuts_uc[:old_cuts] = cut_info_uc[:cuts]
            if inner_tr && cut_info_uc[:tr_constraints] !== nothing
                imp_cuts_uc[:old_tr_constraints] = cut_info_uc[:tr_constraints]
            end

            # Violation check
            if t_0_frac < Q_frac * S - tol * S
                outer_cut_info_uc = evaluate_master_opt_cut(
                    leader_instances_uc, follower_instances_uc,
                    isp_data, cut_info_uc, usercut_count[] + 1; parallel=parallel)

                uc_expr = build_optimality_cut_expr(
                    omp_vars, outer_cut_info_uc,
                    diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h, S)
                uc_con = @build_constraint(t_0 >= uc_expr)
                MOI.submit(omp_model, MOI.UserCut(cb_data), uc_con)
                usercut_count[] += 1
                push!(cut_pool, Dict{Symbol,Any}(
                    :type => :mipnode_usercut, :mipnode => mipnode_call_count[],
                    :cut_info => deepcopy(outer_cut_info_uc),
                    :x_sol => copy(x_frac), :λ_sol => λ_frac, :h_sol => copy(h_frac), :ψ0_sol => copy(ψ0_frac),
                    :α_sol => haskey(cut_info_uc, :α_sol) ? copy(cut_info_uc[:α_sol]) : nothing,
                    :Q => Q_frac))
                @info "  [MIPNODE $(mipnode_call_count[])] User cut: Q=$(round(Q_frac, digits=6)), t_0/S=$(round(t_0_frac/S, digits=6))"
            end
        end
    end

    # ====================================================================
    # Gurobi 설정 + solve
    # ====================================================================
    set_attribute(omp_model, "LazyConstraints", 1)
    set_attribute(omp_model, "Threads", 1)       # Thread safety with Mosek ISP
    set_attribute(omp_model, "PreCrush", 1)      # Presolve가 변수 eliminate 방지
    set_attribute(omp_model, "OutputFlag", 1)    # Gurobi B&B 로그 출력
    MOI.set(omp_model, MOI.LazyConstraintCallback(), benders_lazy_callback)
    if mipnode_freq > 0
        MOI.set(omp_model, MOI.UserCutCallback(), benders_usercut_callback)
    end

    @info "Starting Gurobi B&B solve..."
    optimize!(omp_model)

    # ====================================================================
    # 결과 추출
    # ====================================================================
    result[:lazy_cut_count] = cut_count[]
    result[:callback_count] = callback_count[]
    result[:usercut_count] = usercut_count[]
    result[:mipnode_calls] = mipnode_call_count[]
    result[:upper_bound] = ub_tracker[]
    result[:solution_time] = time() - time_start

    st_final = MOI.get(omp_model, MOI.TerminationStatus())
    result[:termination_status] = st_final

    if st_final == MOI.OPTIMAL || st_final == MOI.ALMOST_OPTIMAL
        result[:opt_sol] = Dict(
            :x => round.(value.(x)), :h => value.(h),
            :λ => value(λ_var), :ψ0 => value.(ψ0))
        result[:obj_val] = value(t_0) / S
    end

    @info "Branch-and-Benders complete" status=st_final lazy_cuts=cut_count[] user_cuts=usercut_count[] callbacks=callback_count[] mipnode_calls=mipnode_call_count[] pass1_rejections=result[:pass1_rejections] time=round(result[:solution_time], digits=2)
    return result
end
