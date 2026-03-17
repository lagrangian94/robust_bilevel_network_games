using JuMP
using LinearAlgebra
using SparseArrays
using Infiltrator
using Pajarito
using Gurobi
using Mosek, MosekTools
using Hypatia, HiGHS

# Load network generator
includet("network_generator.jl")
includet("build_dualized_outer_subprob.jl")
includet("build_full_model.jl")
using .NetworkGenerator


function build_omp(network, ϕU, λU, γ, w; optimizer=nothing, multi_cut_lf=true, multi_cut_scenario=true, S=1)
    # Extract network dimensions
    num_arcs = length(network.arcs)-1 #dummy arc 제외
    # Create model
    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))
    if multi_cut_scenario && multi_cut_lf
        @variable(model, t_l[1:S] >= 0)
        @variable(model, t_f[1:S] >= 0)
        @objective(model, Min, (sum(t_l) + sum(t_f)) / S)
    elseif multi_cut_scenario
        @variable(model, t_s[1:S] >= 0)
        @objective(model, Min, sum(t_s) / S)
    elseif multi_cut_lf
        @variable(model, t_0_l >= 0)
        @variable(model, t_0_f >= 0)
        @objective(model, Min, (t_0_l + t_0_f) / S)
    else
        @variable(model, t_0 >= 0)
        @objective(model, Min, t_0 / S)
    end
    @variable(model, λ, lower_bound=0.0, upper_bound=λU)  # λ ≤ λU ≤ ϕU: LDR P-bound에서 π̃₀_sink ≤ ϕU이므로 eq(6d) Nts⊺π̃ ≥ λ 만족 필요
    @variable(model, x[1:num_arcs], Bin)
    @variable(model, h[1:num_arcs] >= 0)
    @variable(model, ψ0[1:num_arcs] >= 0)
    @constraint(model, resource_budget, sum(h) <= λ * w)
    # @constraint(model, [i=1:num_arcs], h[i] <= w * x[i])  # per-arc recovery cap: only interdicted arcs
    @constraint(model, sum(x) <= γ)
    # x must be binary, and only interdictable arcs can be selected
    for i in 1:num_arcs
        if !network.interdictable_arcs[i]
            @constraint(model, x[i] == 0)
            println("Arc $i is not interdictable")
        end
    end

    @constraint(model, λ>=0.001)
    # mccormick envelope constraints for ψ0
    for k in 1:num_arcs
        @constraint(model, ψ0[k] <= λU * x[k])
        @constraint(model, ψ0[k] <= λ)
        @constraint(model, ψ0[k] >= λ - λU * (1 - x[k]))
        @constraint(model, ψ0[k] >= 0)
    end
    vars = Dict(:λ => λ, :x => x, :h => h, :ψ0 => ψ0)
    if multi_cut_scenario && multi_cut_lf
        vars[:t_l] = t_l
        vars[:t_f] = t_f
        vars[:t_0] = sum(t_l) + sum(t_f)
    elseif multi_cut_scenario
        vars[:t_s] = t_s
        vars[:t_0] = sum(t_s)
    elseif multi_cut_lf
        vars[:t_0_l] = t_0_l
        vars[:t_0_f] = t_0_f
        vars[:t_0] = t_0_l + t_0_f
    else
        vars[:t_0] = t_0
    end
    return model, vars
end

function osp_optimize!(osp_model::Model, osp_vars::Dict, osp_data::Dict, λ_sol, x_sol, h_sol, ψ0_sol; multi_cut_lf=false)
    E = osp_data[:E]
    v = osp_data[:v]
    ϕU = osp_data[:ϕU]
    πU = get(osp_data, :πU, ϕU)
    yU = get(osp_data, :yU, ϕU)
    ytsU = get(osp_data, :ytsU, ϕU)
    S = osp_data[:S]
    d0 = osp_data[:d0]
    uncertainty_set = osp_data[:uncertainty_set]
    xi_bar = uncertainty_set[:xi_bar]
    num_arcs = length(xi_bar[1])


    Uhat1, Uhat3, Utilde1, Utilde3 = osp_vars[:Uhat1], osp_vars[:Uhat3], osp_vars[:Utilde1], osp_vars[:Utilde3]
    βhat1_1, βtilde1_1 = osp_vars[:βhat1_1], osp_vars[:βtilde1_1]
    βtilde1_3 = osp_vars[:βtilde1_3]
    Phat1_Φ, Phat1_Π = osp_vars[:Phat1_Φ], osp_vars[:Phat1_Π]
    Phat2_Φ, Phat2_Π = osp_vars[:Phat2_Φ], osp_vars[:Phat2_Π]
    Ptilde1_Φ, Ptilde1_Π = osp_vars[:Ptilde1_Φ], osp_vars[:Ptilde1_Π]
    Ptilde2_Φ, Ptilde2_Π = osp_vars[:Ptilde2_Φ], osp_vars[:Ptilde2_Π]
    Ptilde1_Y, Ptilde1_Yts = osp_vars[:Ptilde1_Y], osp_vars[:Ptilde1_Yts]
    Ptilde2_Y, Ptilde2_Yts = osp_vars[:Ptilde2_Y], osp_vars[:Ptilde2_Yts]
    Ztilde1_3 = osp_vars[:Ztilde1_3]
    diag_x_E = Diagonal(x_sol) * E  # diag(x)E
    diag_λ_ψ = Diagonal(λ_sol*ones(num_arcs) - v.*ψ0_sol)
    # s에 대해 summing이 필요하다면 sum over s 추가
    # matrix inner product: sum(M .* N)
    obj_term1 = [-ϕU * sum((Uhat1[s, :, :] + Utilde1[s, :, :]) .* diag_x_E) for s=1:S]
    obj_term2 = [-ϕU * sum((Uhat3[s, :, :] + Utilde3[s, :, :]) .* (E-diag_x_E)) for s=1:S]
    obj_term3 = [(d0')* βhat1_1[s,:] for s=1:S] #이거만 maximize하면 dual infeasible
    obj_term4 = [sum(Ztilde1_3[s, :, :] .* (diag_λ_ψ * diagm(xi_bar[s]))) for s=1:S]
    obj_term5 = [(λ_sol*d0')* βtilde1_1[s,:] for s=1:S] #이거만 maximize하면 dual infeasible
    obj_term6 = [-(h_sol + diag_λ_ψ * xi_bar[s])'* βtilde1_3[s,:] for s=1:S]

    obj_term_ub_hat = [-ϕU * sum(Phat1_Φ[s,:,:]) - πU * sum(Phat1_Π[s,:,:]) for s=1:S]
    obj_term_lb_hat = [-ϕU * sum(Phat2_Φ[s,:,:]) - πU * sum(Phat2_Π[s,:,:]) for s=1:S]
    obj_term_ub_tilde = [-ϕU * sum(Ptilde1_Φ[s,:,:]) - πU * sum(Ptilde1_Π[s,:,:]) - yU * sum(Ptilde1_Y[s,:,:]) - ytsU * sum(Ptilde1_Yts[s,:]) for s=1:S]
    obj_term_lb_tilde = [-ϕU * sum(Ptilde2_Φ[s,:,:]) - πU * sum(Ptilde2_Π[s,:,:]) - yU * sum(Ptilde2_Y[s,:,:]) - ytsU * sum(Ptilde2_Yts[s,:]) for s=1:S]
    @objective(osp_model, Max, (sum(obj_term1) + sum(obj_term2) + sum(obj_term3) + sum(obj_term4) + sum(obj_term5) + sum(obj_term6)
    + sum(obj_term_ub_hat) + sum(obj_term_lb_hat) + sum(obj_term_ub_tilde) + sum(obj_term_lb_tilde)) / S)

    optimize!(osp_model)
    st = MOI.get(osp_model, MOI.TerminationStatus())
    if (st == MOI.OPTIMAL) || (st == MOI.SLOW_PROGRESS)
        obj_val = objective_value(osp_model)
        cut_coeff = Dict(
            :Uhat1 => value.(Uhat1),
            :Utilde1 => value.(Utilde1),
            :Uhat3 => value.(Uhat3),
            :Utilde3 => value.(Utilde3),
            :βtilde1_1 => value.(βtilde1_1),
            :βtilde1_3 => value.(βtilde1_3),
            :Ztilde1_3 => value.(Ztilde1_3),
            :obj_val => obj_val,
            :α_sol => value.(osp_vars[:α]),
        )
        # Always compute separate intercepts (per-scenario vectors)
        cut_coeff[:intercept_l] = value.(obj_term3) .+ value.(obj_term_ub_hat) .+ value.(obj_term_lb_hat)
        cut_coeff[:intercept_f] = value.(obj_term_ub_tilde) .+ value.(obj_term_lb_tilde)
        cut_coeff[:intercept] = cut_coeff[:intercept_l] .+ cut_coeff[:intercept_f]
        return (:OptimalityCut, cut_coeff)
    else
        t_status = termination_status(osp_model)
        p_status = primal_status(osp_model)
        d_status = dual_status(osp_model)
        @warn "OSP failed: termination=$t_status, primal=$p_status, dual=$d_status"
        @infiltrate
        error("Subproblem is not optimal")
    end
end

"""
    add_optimality_cuts!(omp_model, omp_vars, cut_info, diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h_var, S, iter;
        multi_cut_lf=false, multi_cut_scenario=false, prefix="opt_cut", result_cuts=nothing)

Cut 조립 helper. cut_info에서 coefficient를 읽어 omp_model에 constraint를 추가.
4가지 모드 지원: (multi_cut_lf × multi_cut_scenario).
Returns combined AffExpr for tightness check.
"""
function add_optimality_cuts!(omp_model, omp_vars, cut_info, diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h_var, S, iter;
    multi_cut_lf=false, multi_cut_scenario=false, prefix="opt_cut", result_cuts=nothing)
    # Per-scenario cut terms (always compute leader/follower separately)
    leader_s = Vector{Any}(undef, S)
    follower_s = Vector{Any}(undef, S)
    for s in 1:S
        leader_s[s] = -ϕU * sum(cut_info[:Uhat1][s,:,:] .* diag_x_E) +
                      -ϕU * sum(cut_info[:Uhat3][s,:,:] .* (E - diag_x_E)) +
                      cut_info[:intercept_l][s]
        follower_s[s] = -ϕU * sum(cut_info[:Utilde1][s,:,:] .* diag_x_E) +
                        -ϕU * sum(cut_info[:Utilde3][s,:,:] .* (E - diag_x_E)) +
                        sum(cut_info[:Ztilde1_3][s,:,:] .* (diag_λ_ψ * diagm(xi_bar[s]))) +
                        (d0' * cut_info[:βtilde1_1][s,:]) * λ_var +
                        -(h_var + diag_λ_ψ * xi_bar[s])' * cut_info[:βtilde1_3][s,:] +
                        cut_info[:intercept_f][s]
    end

    # Add constraints based on mode
    if multi_cut_scenario && multi_cut_lf
        t_l, t_f = omp_vars[:t_l], omp_vars[:t_f]
        for s in 1:S
            c_l = @constraint(omp_model, t_l[s] >= leader_s[s])
            c_f = @constraint(omp_model, t_f[s] >= follower_s[s])
            set_name(c_l, "$(prefix)_$(iter)_l_s$(s)")
            set_name(c_f, "$(prefix)_$(iter)_f_s$(s)")
            if result_cuts !== nothing
                result_cuts["$(prefix)_$(iter)_l_s$(s)"] = c_l
                result_cuts["$(prefix)_$(iter)_f_s$(s)"] = c_f
            end
        end
    elseif multi_cut_scenario
        t_sv = omp_vars[:t_s]
        for s in 1:S
            c = @constraint(omp_model, t_sv[s] >= leader_s[s] + follower_s[s])
            set_name(c, "$(prefix)_$(iter)_s$(s)")
            if result_cuts !== nothing
                result_cuts["$(prefix)_$(iter)_s$(s)"] = c
            end
        end
    elseif multi_cut_lf
        opt_cut_l = sum(leader_s)
        opt_cut_f = sum(follower_s)
        c_l = @constraint(omp_model, omp_vars[:t_0_l] >= opt_cut_l)
        c_f = @constraint(omp_model, omp_vars[:t_0_f] >= opt_cut_f)
        set_name(c_l, "$(prefix)_$(iter)_l")
        set_name(c_f, "$(prefix)_$(iter)_f")
        if result_cuts !== nothing
            result_cuts["$(prefix)_$(iter)_l"] = c_l
            result_cuts["$(prefix)_$(iter)_f"] = c_f
        end
    else
        opt_cut = sum(leader_s) + sum(follower_s)
        c = @constraint(omp_model, omp_vars[:t_0] >= opt_cut)
        set_name(c, "$(prefix)_$(iter)")
        if result_cuts !== nothing
            result_cuts["$(prefix)_$(iter)"] = c
        end
    end

    # Return combined expression for tightness check
    return sum(leader_s) + sum(follower_s)
end

function initialize_omp(omp_model::Model, omp_vars::Dict)
    optimize!(omp_model) # 여기서 최적화를 하는 이유는 초기해를 뽑아서 subproblem을 build하기 위함임.
    st = MOI.get(omp_model, MOI.TerminationStatus())    
    @info "Initial status $st" # restricted master has a solution or is unbounded

    return st, value(omp_vars[:λ]), value.(omp_vars[:x]), value.(omp_vars[:h]), value.(omp_vars[:ψ0])
end

"""
    scenario_benders_optimize!(omp_model, omp_vars, network, ϕU, λU, γ, w, v, uncertainty_set; ...)

Scenario-decomposed Benders: OMP → S × OSP(s=1).
각 시나리오별 leader+follower+α를 독립 OSP에서 풀고, 시나리오 간 병렬화 가능.
Strict Benders와 Nested Benders의 중간 구조.

| | Strict | Scenario-Decomposed | Nested |
|---|---|---|---|
| 구조 | OMP → 1 OSP(S개) | OMP → S × OSP(1개) | OMP → IMP → S × (ISP_l + ISP_f) |
| α 공유 | 전체 공유 | 시나리오별 독립 | IMP에서 공유 |
| 병렬화 | 불가 | S개 병렬 | S개 병렬 (inner) |
"""
function scenario_benders_optimize!(omp_model::Model, omp_vars::Dict, network, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=nothing, multi_cut_lf=true, multi_cut_scenario=true,
    max_iter=1000, tol=1e-4, πU=ϕU, yU=ϕU, ytsU=ϕU, parallel=false, strengthen_cuts=:none)
    ### --------Begin Initialization--------
    st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars)
    x, h, λ, ψ0 = omp_vars[:x], omp_vars[:h], omp_vars[:λ], omp_vars[:ψ0]
    t_0 = omp_vars[:t_0]
    num_arcs = length(network.arcs) - 1
    S = length(uncertainty_set[:xi_bar])
    R_us = uncertainty_set[:R]
    r_dict_us = uncertainty_set[:r_dict]
    xi_bar = uncertainty_set[:xi_bar]
    epsilon = uncertainty_set[:epsilon]

    conic_opt = conic_optimizer !== nothing ? conic_optimizer : Mosek.Optimizer

    # Build S separate single-scenario OSP instances
    osp_instances = Vector{Tuple}(undef, S)
    for s in 1:S
        U_s = Dict(:R => Dict(1 => R_us[s]), :r_dict => Dict(1 => r_dict_us[s]),
                    :xi_bar => Dict(1 => xi_bar[s]), :epsilon => epsilon)
        osp_instances[s] = build_dualized_outer_subproblem(
            network, 1, ϕU, λU, γ, w, v, U_s, conic_opt,
            λ_sol, x_sol, h_sol, ψ0_sol; πU=πU, yU=yU, ytsU=ytsU)
    end

    # Common data from first instance
    E = osp_instances[1][3][:E]
    d0 = osp_instances[1][3][:d0]
    diag_x_E = Diagonal(x) * E
    diag_λ_ψ = Diagonal(λ*ones(num_arcs) - v.*ψ0)

    # ISP instances for cut strengthening (per-scenario α → ISP → MW/Sherali cuts)
    isp_leader_instances, isp_follower_instances = nothing, nothing
    isp_data_sd = nothing
    if strengthen_cuts != :none
        E_isp = ones(num_arcs, num_arcs + 1)
        d0_isp = zeros(num_arcs + 1); d0_isp[end] = 1.0
        α_dummy = zeros(num_arcs)
        isp_leader_instances, isp_follower_instances = initialize_isp(
            network, S, ϕU, λU, γ, w, v, uncertainty_set;
            conic_optimizer=conic_opt, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
            α_sol=α_dummy, πU=πU, yU=yU, ytsU=ytsU, scaling_S=1)
        isp_data_sd = Dict(:E => E_isp, :network => network, :ϕU => ϕU, :πU => πU,
            :yU => yU, :ytsU => ytsU, :λU => λU, :γ => γ, :w => w, :v => v,
            :uncertainty_set => uncertainty_set, :d0 => d0_isp, :S => S, :scaling_S => 1)
    end

    iter = 0
    past_obj = []
    past_subprob_obj = []
    past_upper_bound = []
    upper_bound = Inf
    result = Dict()
    result[:cuts] = Dict()
    result[:debug_α] = []
    result[:debug_intercept_l] = []
    result[:debug_intercept_f] = []
    result[:debug_coeff_norms] = []
    ### --------End Initialization--------
    time_start = time()
    while (st == MOI.DUAL_INFEASIBLE || st == MOI.OPTIMAL)
        iter += 1
        if iter > max_iter
            @warn "Maximum iterations ($max_iter) reached."
            break
        end
        @info "[Scenario-Decomposed] Iteration $iter"

        optimize!(omp_model)
        st = MOI.get(omp_model, MOI.TerminationStatus())

        x_sol = round.(value.(omp_vars[:x]))
        h_sol, λ_sol, ψ0_sol = value.(omp_vars[:h]), value(omp_vars[:λ]), value.(omp_vars[:ψ0])
        t_0_sol = value(omp_vars[:t_0]) / S

        # Solve S scenarios independently (each with S=1)
        scenario_results, all_ok = solve_scenarios(S; parallel=parallel) do s
            osp_m, osp_v, osp_d = osp_instances[s]
            (status_s, cut_info_s) = osp_optimize!(osp_m, osp_v, osp_d, λ_sol, x_sol, h_sol, ψ0_sol; multi_cut_lf=multi_cut_lf)
            return (status_s == :OptimalityCut, cut_info_s)
        end

        if !all_ok
            error("Some scenario subproblems failed")
        end

        # Assemble cut_info: stack [1,...] arrays into [S,...] shape
        cut_info = Dict{Symbol, Any}()
        for key in [:Uhat1, :Utilde1, :Uhat3, :Utilde3, :βtilde1_1, :βtilde1_3, :Ztilde1_3]
            cut_info[key] = cat([scenario_results[s][key] for s in 1:S]...; dims=1)
        end
        cut_info[:intercept_l] = [scenario_results[s][:intercept_l][1] for s in 1:S]
        cut_info[:intercept_f] = [scenario_results[s][:intercept_f][1] for s in 1:S]
        cut_info[:intercept] = cut_info[:intercept_l] .+ cut_info[:intercept_f]
        cut_info[:obj_val] = sum(scenario_results[s][:obj_val] for s in 1:S) / S
        cut_info[:α_sol] = [scenario_results[s][:α_sol] for s in 1:S]  # Vector of per-scenario α

        subprob_obj = cut_info[:obj_val]
        upper_bound = min(upper_bound, subprob_obj)

        # Debug logging
        push!(result[:debug_α], cut_info[:α_sol])
        push!(result[:debug_intercept_l], sum(cut_info[:intercept_l]))
        push!(result[:debug_intercept_f], sum(cut_info[:intercept_f]))
        push!(result[:debug_coeff_norms], Dict(
            :Uhat1 => norm(cut_info[:Uhat1]),
            :Utilde1 => norm(cut_info[:Utilde1]),
            :Uhat3 => norm(cut_info[:Uhat3]),
            :Utilde3 => norm(cut_info[:Utilde3]),
            :βtilde1_1 => norm(cut_info[:βtilde1_1]),
            :βtilde1_3 => norm(cut_info[:βtilde1_3]),
            :Ztilde1_3 => norm(cut_info[:Ztilde1_3]),
        ))

        gap = abs(upper_bound - t_0_sol) / max(abs(upper_bound), 1e-10)
        push!(past_obj, t_0_sol)
        push!(past_subprob_obj, subprob_obj)
        push!(past_upper_bound, upper_bound)

        # Convergence check
        if gap <= tol
            time_end = time()
            @info "Termination condition met"
            println("t_0_sol: ", t_0_sol, ", subprob_obj: ", subprob_obj)
            result[:past_obj] = past_obj
            result[:past_subprob_obj] = past_subprob_obj
            result[:past_upper_bound] = past_upper_bound
            result[:solution_time] = time_end - time_start
            result[:opt_sol] = Dict(:x=>x_sol, :h=>h_sol, :λ=>λ_sol, :ψ0=>ψ0_sol)
            return result
        end

        @info "Iter $iter: LB=$(round(t_0_sol, digits=4))  UB=$(round(upper_bound, digits=4))  gap=$(round(gap, digits=6))  ($(round(time()-time_start, digits=1))s)"

        opt_cut = add_optimality_cuts!(omp_model, omp_vars, cut_info, diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ, h, S, iter;
            multi_cut_lf=multi_cut_lf, multi_cut_scenario=multi_cut_scenario, prefix="opt_cut", result_cuts=result[:cuts])

        println("subproblem objective: ", subprob_obj)
        @info "Optimality cut added"

        # Tightness check
        y = Dict(
            [omp_vars[:x][k] => x_sol[k] for k in 1:num_arcs]...,
            [omp_vars[:h][k] => h_sol[k] for k in 1:num_arcs]...,
            omp_vars[:λ] => λ_sol,
            [omp_vars[:ψ0][k] => ψ0_sol[k] for k in 1:num_arcs]...
        )
        function evaluate_expr(expr::AffExpr, var_values::Dict)
            eval_result = expr.constant
            for (var, coef) in expr.terms
                if haskey(var_values, var)
                    eval_result += coef * var_values[var]
                else
                    error("Variable $var not found in var_values")
                end
            end
            return eval_result
        end
        if abs(subprob_obj * S - evaluate_expr(opt_cut, y)) > 1e-4
            println("something went wrong")
            @infiltrate
        end

        # ===== Cut Strengthening (per-scenario α) =====
        if strengthen_cuts != :none && isp_leader_instances !== nothing
            α_from_osp = cut_info[:α_sol]  # Vector of per-scenario α vectors
            osp_cut_as_info = Dict(:α_sol => α_from_osp, :obj_val => cut_info[:obj_val])

            if strengthen_cuts == :mw
                # MW: ISP solve → ISP cut + MW cut
                solve_scenarios(S; parallel=parallel) do s_isp
                    U_s = Dict(:R => Dict(1=>R_us[s_isp]), :r_dict => Dict(1=>r_dict_us[s_isp]),
                               :xi_bar => Dict(1=>xi_bar[s_isp]), :epsilon => epsilon)
                    α_s = α_from_osp[s_isp]
                    isp_leader_optimize!(isp_leader_instances[s_isp][1], isp_leader_instances[s_isp][2];
                        isp_data=isp_data_sd, uncertainty_set=U_s,
                        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_s)
                    isp_follower_optimize!(isp_follower_instances[s_isp][1], isp_follower_instances[s_isp][2];
                        isp_data=isp_data_sd, uncertainty_set=U_s,
                        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_s)
                    return (true, nothing)
                end

                # Step A: ISP-based cut
                isp_cut = evaluate_master_opt_cut(
                    isp_leader_instances, isp_follower_instances,
                    isp_data_sd, osp_cut_as_info, iter; multi_cut_lf=multi_cut_lf, parallel=parallel)

                add_optimality_cuts!(omp_model, omp_vars, isp_cut, diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ, h, S, iter;
                    multi_cut_lf=multi_cut_lf, multi_cut_scenario=multi_cut_scenario, prefix="isp_cut", result_cuts=result[:cuts])
                @info "  ISP-based cut added (per-scenario α)"

                # Step B: MW cuts from core points
                interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
                core_points = generate_core_points(network, γ, λU, w, v;
                    interdictable_idx=interdictable_idx, strategy=:interior)
                for (cp_idx, cp) in enumerate(core_points)
                    str_info = evaluate_mw_opt_cut(
                        isp_leader_instances, isp_follower_instances,
                        isp_data_sd, osp_cut_as_info, iter;
                        x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                        x_core=cp.x, λ_core=cp.λ, h_core=cp.h, ψ0_core=cp.ψ0,
                        multi_cut_lf=multi_cut_lf, parallel=parallel)
                    add_optimality_cuts!(omp_model, omp_vars, str_info, diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ, h, S, iter;
                        multi_cut_lf=multi_cut_lf, multi_cut_scenario=multi_cut_scenario, prefix="mw_cut_cp$(cp_idx)", result_cuts=result[:cuts])
                end
                @info "  $(length(core_points)) mw strengthening cuts added"

            elseif strengthen_cuts == :sherali
                interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
                core_points = generate_core_points(network, γ, λU, w, v;
                    interdictable_idx=interdictable_idx, strategy=:interior)
                for (cp_idx, cp) in enumerate(core_points)
                    str_info = evaluate_sherali_opt_cut(
                        isp_leader_instances, isp_follower_instances,
                        isp_data_sd, osp_cut_as_info, iter;
                        x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                        x_core=cp.x, λ_core=cp.λ, h_core=cp.h, ψ0_core=cp.ψ0,
                        multi_cut_lf=multi_cut_lf, parallel=parallel)
                    add_optimality_cuts!(omp_model, omp_vars, str_info, diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ, h, S, iter;
                        multi_cut_lf=multi_cut_lf, multi_cut_scenario=multi_cut_scenario, prefix="sherali_cut_cp$(cp_idx)", result_cuts=result[:cuts])
                end
                @info "  $(length(core_points)) sherali cuts added"
            end
        end
    end
    # max_iter reached or while condition false
    time_end = time()
    result[:past_obj] = past_obj
    result[:past_subprob_obj] = past_subprob_obj
    result[:past_upper_bound] = past_upper_bound
    result[:solution_time] = time_end - time_start
    return result
end

function strict_benders_optimize!(omp_model::Model, omp_vars::Dict, network, ϕU, λU, γ, w, uncertainty_set; optimizer=nothing, outer_tr=false, multi_cut_lf=true, multi_cut_scenario=true, max_iter=1000, tol=1e-4, πU=ϕU, yU=ϕU, ytsU=ϕU, strengthen_cuts=:none, conic_optimizer=nothing, parallel=false)
    ### --------Begin Initialization--------
    st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars)
    x, h, λ, ψ0 = omp_vars[:x], omp_vars[:h], omp_vars[:λ], omp_vars[:ψ0]
    t_0 = omp_vars[:t_0]  # always composite (sum of epigraph vars)
    num_arcs = length(network.arcs) - 1
    S_total = length(uncertainty_set[:xi_bar])  # scenario count for /S averaging
    osp_model, osp_vars, osp_data = build_dualized_outer_subproblem(network, S, ϕU, λU, γ, w, v, uncertainty_set, MosekTools.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol; πU=πU, yU=yU, ytsU=ytsU)

    diag_x_E = Diagonal(x) * osp_data[:E]  # diag(x)E
    diag_λ_ψ = Diagonal(λ*ones(num_arcs)-v.*ψ0)
    xi_bar = uncertainty_set[:xi_bar]
    iter = 0

    # ISP instances for cut strengthening (OSP α* → ISP coefficient extraction → MW cuts)
    isp_leader_instances, isp_follower_instances = nothing, nothing
    isp_data_strict = nothing
    if strengthen_cuts != :none
        _conic_opt = conic_optimizer !== nothing ? conic_optimizer : Mosek.Optimizer
        E_isp = ones(num_arcs, num_arcs + 1)
        d0_isp = zeros(num_arcs + 1); d0_isp[end] = 1.0
        α_dummy = zeros(num_arcs)  # initial α, will be overwritten
        isp_leader_instances, isp_follower_instances = initialize_isp(
            network, S, ϕU, λU, γ, w, v, uncertainty_set;
            conic_optimizer=_conic_opt, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
            α_sol=α_dummy, πU=πU, yU=yU, ytsU=ytsU)
        isp_data_strict = Dict(:E => E_isp, :network => network, :ϕU => ϕU, :πU => πU,
            :yU => yU, :ytsU => ytsU, :λU => λU, :γ => γ, :w => w, :v => v,
            :uncertainty_set => uncertainty_set, :d0 => d0_isp, :S => S)
    end

    past_obj = []
    past_subprob_obj = []
    past_upper_bound = []
    upper_bound = Inf
    result = Dict()
    result[:cuts] = Dict()
    # Debug logging arrays
    result[:debug_α] = []
    result[:debug_intercept_l] = []
    result[:debug_intercept_f] = []
    result[:debug_coeff_norms] = []

    ### --------Trust Region Initialization--------
    if outer_tr
        B_bin_sequence = [0.05, 0.5, 1.0]
        B_bin_stage = 1
        B_bin = B_bin_sequence[B_bin_stage] * sum(network.interdictable_arcs)
        B_con = nothing
        centers = Dict{Symbol, Any}(
            :x => copy(x_sol),
            :h => copy(h_sol),
            :λ => λ_sol,
            :ψ0 => copy(ψ0_sol)
        )
        β_relative = 1e-4
        tr_constraints = Dict{Symbol, Any}(
            :binary => nothing,
            :continuous => nothing
        )
        past_major_subprob_obj = []
        major_iter = []
        bin_B_steps = []
        past_local_lower_bound = []
        past_local_optimizer = []
        lower_bound = -Inf
        past_lower_bound = []
        result[:tr_info] = Dict()
    end
    ### --------End Initialization--------
    time_start = time()
    while (st == MOI.DUAL_INFEASIBLE || st == MOI.OPTIMAL)
        iter += 1
        if iter > max_iter
            @warn "Maximum iterations ($max_iter) reached."
            break
        end
        if outer_tr
            @info "Iteration $iter (B_bin=$B_bin, Stage=$(B_bin_stage)/$(length(B_bin_sequence)))"
        else
            @info "Iteration $iter"
        end
        optimize!(omp_model)
        st = MOI.get(omp_model, MOI.TerminationStatus())

        if outer_tr && st == MOI.INFEASIBLE
            @info "OMP infeasible (Converged): No search space left due to reverse regions"
            gap = 0.0
            # Fall through to gap check below
        else
            x_sol = round.(value.(omp_vars[:x]))  # binary rounding for numerical stability
            h_sol, λ_sol, ψ0_sol = value.(omp_vars[:h]), value(omp_vars[:λ]), value.(omp_vars[:ψ0])
            t_0_sol = value(omp_vars[:t_0]) / S_total  # average over scenarios

            (status, cut_info) = osp_optimize!(osp_model, osp_vars, osp_data, λ_sol, x_sol, h_sol, ψ0_sol; multi_cut_lf=multi_cut_lf)
            subprob_obj = cut_info[:obj_val]
            upper_bound = min(upper_bound, subprob_obj)

            # Debug logging
            push!(result[:debug_α], cut_info[:α_sol])
            push!(result[:debug_intercept_l], sum(cut_info[:intercept_l]))
            push!(result[:debug_intercept_f], sum(cut_info[:intercept_f]))
            push!(result[:debug_coeff_norms], Dict(
                :Uhat1 => norm(cut_info[:Uhat1]),
                :Utilde1 => norm(cut_info[:Utilde1]),
                :Uhat3 => norm(cut_info[:Uhat3]),
                :Utilde3 => norm(cut_info[:Utilde3]),
                :βtilde1_1 => norm(cut_info[:βtilde1_1]),
                :βtilde1_3 => norm(cut_info[:βtilde1_3]),
                :Ztilde1_3 => norm(cut_info[:Ztilde1_3]),
            ))

            if status != :OptimalityCut
                error("Subproblem returned unexpected status: $status")
            end

            if outer_tr
                lower_bound = max(lower_bound, t_0_sol)
                gap = abs(upper_bound - lower_bound) / max(abs(upper_bound), 1e-10)

                # Serious step test
                if iter == 1
                    push!(past_major_subprob_obj, subprob_obj)
                end
                tr_needs_update = false
                predicted_decrease = past_major_subprob_obj[end] - t_0_sol
                β_dynamic = max(1e-8, β_relative * predicted_decrease)
                improvement = past_major_subprob_obj[end] - subprob_obj
                is_serious_step = (improvement >= β_dynamic)
                if is_serious_step
                    centers[:x] = copy(x_sol)
                    centers[:h] = copy(h_sol)
                    centers[:λ] = λ_sol
                    centers[:ψ0] = copy(ψ0_sol)
                    push!(major_iter, iter)
                    push!(past_major_subprob_obj, subprob_obj)
                    tr_needs_update = true
                end
            else
                gap = abs(upper_bound - t_0_sol) / max(abs(upper_bound), 1e-10)
            end

            # Store history
            push!(past_obj, t_0_sol)
            push!(past_subprob_obj, subprob_obj)
            push!(past_upper_bound, upper_bound)
            if outer_tr
                push!(past_lower_bound, lower_bound)
            end
        end

        # Convergence check
        if gap <= tol
            if !outer_tr
                # No outer TR: simple convergence
                time_end = time()
                @info "Termination condition met"
                println("t_0_sol: ", t_0_sol, ", subprob_obj: ", subprob_obj)
                result[:past_obj] = past_obj
                result[:past_subprob_obj] = past_subprob_obj
                result[:past_upper_bound] = past_upper_bound
                result[:solution_time] = time_end - time_start
                result[:opt_sol] = Dict(:x=>x_sol, :h=>h_sol, :λ=>λ_sol, :ψ0=>ψ0_sol)
                return result
            end

            # Outer TR: local optimality reached
            if B_bin_stage <= length(B_bin_sequence) - 1
                # Expand trust region
                B_bin_stage += 1
                B_bin_old = B_bin
                B_bin = B_bin_sequence[B_bin_stage] * sum(network.interdictable_arcs)
                push!(bin_B_steps, iter)
                push!(past_local_lower_bound, lower_bound)
                push!(past_local_optimizer, Dict(:x=>copy(x_sol), :h=>copy(h_sol), :λ=>λ_sol, :ψ0=>copy(ψ0_sol)))
                @info "  ✓ Local optimal reached! Expanding B_bin to $B_bin"
                tr_constraints = update_outer_trust_region_constraints!(
                    omp_model, omp_vars, centers, B_bin, B_con, tr_constraints, network)
                lower_bound = -Inf
                _ = add_reverse_region_constraint!(omp_model, omp_vars[:x], centers[:x], B_bin_old, network)
            else
                # Global optimality
                time_end = time()
                @info "  ✓✓ GLOBAL OPTIMAL! (B_bin = full region)"
                min_idx = argmin(past_local_lower_bound)
                global_lower_bound = past_local_lower_bound[min_idx]
                iter_when_global_optimal = bin_B_steps[min_idx]
                global_upper_bound = past_upper_bound[iter_when_global_optimal]
                println("lower_bound: ", global_lower_bound, ", upper_bound: ", global_upper_bound)

                result[:past_obj] = past_obj
                result[:past_subprob_obj] = past_subprob_obj
                result[:past_upper_bound] = past_upper_bound
                result[:past_lower_bound] = past_lower_bound
                result[:past_local_lower_bound] = past_local_lower_bound
                result[:past_major_subprob_obj] = past_major_subprob_obj
                result[:solution_time] = time_end - time_start
                result[:tr_info][:final_B_bin_stage] = B_bin_stage
                result[:tr_info][:final_B_bin] = B_bin
                result[:tr_info][:major_iter] = major_iter
                result[:tr_info][:bin_B_steps] = bin_B_steps
                result[:opt_sol] = past_local_optimizer[min_idx]
                result[:iter_when_global_optimal] = iter_when_global_optimal
                return result
            end
        else
            # Gap still large → Add cut and continue
            @info "Iter $iter: LB=$(outer_tr ? round(lower_bound, digits=4) : round(t_0_sol, digits=4))  UB=$(round(upper_bound, digits=4))  gap=$(round(gap, digits=6))  (globalUB=$(round(upper_bound, digits=4)); $(round(time()-time_start, digits=1))s)"

            opt_cut = add_optimality_cuts!(omp_model, omp_vars, cut_info, diag_x_E, osp_data[:E], diag_λ_ψ, xi_bar, osp_data[:d0], ϕU, λ, h, S, iter;
                multi_cut_lf=multi_cut_lf, multi_cut_scenario=multi_cut_scenario, prefix="opt_cut", result_cuts=result[:cuts])

            println("subproblem objective: ", subprob_obj)
            @info "Optimality cut added"

            # Check tightness of the cut
            y = Dict(
                [omp_vars[:x][k] => x_sol[k] for k in 1:num_arcs]...,
                [omp_vars[:h][k] => h_sol[k] for k in 1:num_arcs]...,
                omp_vars[:λ] => λ_sol,
                [omp_vars[:ψ0][k] => ψ0_sol[k] for k in 1:num_arcs]...
            )
            function evaluate_expr(expr::AffExpr, var_values::Dict)
                eval_result = expr.constant
                for (var, coef) in expr.terms
                    if haskey(var_values, var)
                        eval_result += coef * var_values[var]
                    else
                        error("Variable $var not found in var_values")
                    end
                end
                return eval_result
            end
            if abs(subprob_obj * S_total - evaluate_expr(opt_cut, y)) > 1e-4  # opt_cut is raw (not /S), subprob_obj is /S
                println("something went wrong")
                @infiltrate
            end

            # ===== Strict → ISP + Cut Strengthening =====
            if strengthen_cuts != :none && isp_leader_instances !== nothing
                α_from_osp = cut_info[:α_sol]
                osp_cut_as_info = Dict(:α_sol => α_from_osp, :obj_val => cut_info[:obj_val])

                if strengthen_cuts == :mw
                    # MW: ISP solve (원본) → ISP cut + MW cut (2 solves)
                    R_us = uncertainty_set[:R]
                    r_dict_us = uncertainty_set[:r_dict]
                    xi_bar_us = uncertainty_set[:xi_bar]
                    epsilon_us = uncertainty_set[:epsilon]
                    solve_scenarios(S; parallel=parallel) do s_isp
                        U_s = Dict(:R => Dict(1=>R_us[s_isp]), :r_dict => Dict(1=>r_dict_us[s_isp]),
                                   :xi_bar => Dict(1=>xi_bar_us[s_isp]), :epsilon => epsilon_us)
                        isp_leader_optimize!(isp_leader_instances[s_isp][1], isp_leader_instances[s_isp][2];
                            isp_data=isp_data_strict, uncertainty_set=U_s,
                            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_from_osp)
                        isp_follower_optimize!(isp_follower_instances[s_isp][1], isp_follower_instances[s_isp][2];
                            isp_data=isp_data_strict, uncertainty_set=U_s,
                            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_from_osp)
                        return (true, nothing)
                    end

                    # Step A: ISP-based cut (Hybrid: OSP α → ISP coefficients)
                    isp_cut = evaluate_master_opt_cut(
                        isp_leader_instances, isp_follower_instances,
                        isp_data_strict, osp_cut_as_info, iter; multi_cut_lf=multi_cut_lf, parallel=parallel)

                    add_optimality_cuts!(omp_model, omp_vars, isp_cut, diag_x_E, osp_data[:E], diag_λ_ψ, xi_bar, osp_data[:d0], ϕU, λ, h, S, iter;
                        multi_cut_lf=multi_cut_lf, multi_cut_scenario=multi_cut_scenario, prefix="isp_cut", result_cuts=result[:cuts])
                    @info "  ISP-based cut added (Hybrid: OSP α → ISP coefficients)"

                    # Step B: MW cuts from core points
                    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
                    core_points = generate_core_points(network, γ, λU, w, v;
                        interdictable_idx=interdictable_idx, strategy=:interior)
                    for (cp_idx, cp) in enumerate(core_points)
                        str_info = evaluate_mw_opt_cut(
                            isp_leader_instances, isp_follower_instances,
                            isp_data_strict, osp_cut_as_info, iter;
                            x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                            x_core=cp.x, λ_core=cp.λ, h_core=cp.h, ψ0_core=cp.ψ0,
                            multi_cut_lf=multi_cut_lf, parallel=parallel)
                        add_optimality_cuts!(omp_model, omp_vars, str_info, diag_x_E, osp_data[:E], diag_λ_ψ, xi_bar, osp_data[:d0], ϕU, λ, h, S, iter;
                            multi_cut_lf=multi_cut_lf, multi_cut_scenario=multi_cut_scenario, prefix="mw_cut_cp$(cp_idx)", result_cuts=result[:cuts])
                    end
                    @info "  $(length(core_points)) mw strengthening cuts added"

                elseif strengthen_cuts == :sherali
                    # Sherali: perturbed solve 1회만 → cut 1개 (ISP 원본 solve 생략)
                    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
                    core_points = generate_core_points(network, γ, λU, w, v;
                        interdictable_idx=interdictable_idx, strategy=:interior)
                    for (cp_idx, cp) in enumerate(core_points)
                        str_info = evaluate_sherali_opt_cut(
                            isp_leader_instances, isp_follower_instances,
                            isp_data_strict, osp_cut_as_info, iter;
                            x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                            x_core=cp.x, λ_core=cp.λ, h_core=cp.h, ψ0_core=cp.ψ0,
                            multi_cut_lf=multi_cut_lf, parallel=parallel)
                        add_optimality_cuts!(omp_model, omp_vars, str_info, diag_x_E, osp_data[:E], diag_λ_ψ, xi_bar, osp_data[:d0], ϕU, λ, h, S, iter;
                            multi_cut_lf=multi_cut_lf, multi_cut_scenario=multi_cut_scenario, prefix="sherali_cut_cp$(cp_idx)", result_cuts=result[:cuts])
                    end
                    @info "  $(length(core_points)) sherali cuts added (direct, no ISP base solve)"
                end
            end

            # Update TR constraints if needed
            if outer_tr && tr_needs_update
                @info "Updating Trust Region"
                tr_constraints = update_outer_trust_region_constraints!(
                    omp_model, omp_vars, centers, B_bin, B_con, tr_constraints, network)
            end
        end
    end
    # max_iter reached or while condition false
    time_end = time()
    result[:past_obj] = past_obj
    result[:past_subprob_obj] = past_subprob_obj
    result[:past_upper_bound] = past_upper_bound
    result[:solution_time] = time_end - time_start
    if outer_tr
        result[:past_lower_bound] = past_lower_bound
        result[:opt_sol] = Dict(:x=>x_sol, :h=>h_sol, :λ=>λ_sol, :ψ0=>ψ0_sol)
    end
    return result
end