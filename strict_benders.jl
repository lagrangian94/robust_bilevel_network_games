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


function build_omp(network, ϕU, λU, γ, w; optimizer=nothing, multi_cut=false)
    # Extract network dimensions
    num_arcs = length(network.arcs)-1 #dummy arc 제외
    # Create model
    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))
    if multi_cut
        @variable(model, t_0_l >= 0)  # Objective epigraph variable
        @variable(model, t_0_f >= 0)  # Objective epigraph variable
        @objective(model, Min, t_0_l + t_0_f)
    else
        @variable(model, t_0 >= 0)  # Objective epigraph variable
        @objective(model, Min, t_0)
    end
    @variable(model, λ >= 0)  # Budget allocation parameter
    @variable(model, x[1:num_arcs], Bin)
    @variable(model, h[1:num_arcs] >= 0)
    @variable(model, ψ0[1:num_arcs] >= 0)
    @constraint(model, resource_budget, sum(h) <= λ * w)
    @constraint(model, sum(x) <= γ)
    # x must be binary, and only interdictable arcs can be selected
    for i in 1:num_arcs
        if !network.interdictable_arcs[i]
            @constraint(model, x[i] == 0)
            println("Arc $i is not interdictable")
        end
    end

    @constraint(model, λ>=0.01)
    # mccormick envelope constraints for ψ0
    for k in 1:num_arcs
        @constraint(model, ψ0[k] <= λU * x[k])
        @constraint(model, ψ0[k] <= λ)
        @constraint(model, ψ0[k] >= λ - λU * (1 - x[k]))
        @constraint(model, ψ0[k] >= 0)
    end
    if multi_cut
        vars = Dict(
            :t_0_l => t_0_l,
            :t_0_f => t_0_f,
            :λ => λ,
            :x => x,
            :h => h,
            :ψ0 => ψ0,
            :t_0 => t_0_l + t_0_f
        )
    else
        vars = Dict(
            :t_0 => t_0,
            :λ => λ,
            :x => x,
            :h => h,
            :ψ0 => ψ0
        )
    end
    return model, vars
end

function osp_optimize!(osp_model::Model, osp_vars::Dict, osp_data::Dict, λ_sol, x_sol, h_sol, ψ0_sol)
    E = osp_data[:E]
    v = osp_data[:v]
    ϕU = osp_data[:ϕU]
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

    obj_term_ub_hat = [-ϕU * sum(Phat1_Φ[s,:,:]) - ϕU * sum(Phat1_Π[s,:,:]) for s=1:S]
    obj_term_lb_hat = [-ϕU * sum(Phat2_Φ[s,:,:]) - ϕU * sum(Phat2_Π[s,:,:]) for s=1:S]
    obj_term_ub_tilde = [-ϕU * sum(Ptilde1_Φ[s,:,:]) - ϕU * sum(Ptilde1_Π[s,:,:]) - ϕU * sum(Ptilde1_Y[s,:,:]) - ϕU * sum(Ptilde1_Yts[s,:]) for s=1:S]
    obj_term_lb_tilde = [-ϕU * sum(Ptilde2_Φ[s,:,:]) - ϕU * sum(Ptilde2_Π[s,:,:]) - ϕU * sum(Ptilde2_Y[s,:,:]) - ϕU * sum(Ptilde2_Yts[s,:]) for s=1:S]
    @objective(osp_model, Max, sum(obj_term1) + sum(obj_term2) + sum(obj_term3) + sum(obj_term4) + sum(obj_term5) + sum(obj_term6)
    + sum(obj_term_ub_hat) + sum(obj_term_lb_hat) + sum(obj_term_ub_tilde) + sum(obj_term_lb_tilde))

    optimize!(osp_model)
    st = MOI.get(osp_model, MOI.TerminationStatus())
    if (st == MOI.OPTIMAL) || (st == MOI.SLOW_PROGRESS)
        obj_val = objective_value(osp_model)
        intercept = value.(obj_term3) .+ value.(obj_term_ub_hat) .+ value.(obj_term_lb_hat) .+ value.(obj_term_ub_tilde) .+ value.(obj_term_lb_tilde)
        cut_coeff = Dict(
            :Uhat1 => value.(Uhat1),
            :Utilde1 => value.(Utilde1),
            :Uhat3 => value.(Uhat3),
            :Utilde3 => value.(Utilde3),
            :βtilde1_1 => value.(βtilde1_1),
            :βtilde1_3 => value.(βtilde1_3),
            :Ztilde1_3 => value.(Ztilde1_3),
            :intercept => intercept,
            :obj_val => obj_val
        )
        return (:OptimalityCut, cut_coeff)
    else
        t_status = termination_status(osp_model)
        @infiltrate
        error("Subproblem is not optimal")
    end
end

function initialize_omp(omp_model::Model, omp_vars::Dict)
    optimize!(omp_model) # 여기서 최적화를 하는 이유는 초기해를 뽑아서 subproblem을 build하기 위함임.
    st = MOI.get(omp_model, MOI.TerminationStatus())    
    @info "Initial status $st" # restricted master has a solution or is unbounded

    return st, value(omp_vars[:λ]), value.(omp_vars[:x]), value.(omp_vars[:h]), value.(omp_vars[:ψ0])
end

function strict_benders_optimize!(omp_model::Model, omp_vars::Dict, network, ϕU, λU, γ, w, uncertainty_set; optimizer=nothing, outer_tr=false, max_iter=1000)
    ### --------Begin Initialization--------
    st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars)
    x, h, λ, ψ0, t_0 = omp_vars[:x], omp_vars[:h], omp_vars[:λ], omp_vars[:ψ0], omp_vars[:t_0]
    num_arcs = length(network.arcs) - 1
    osp_model, osp_vars, osp_data = build_dualized_outer_subproblem(network, S, ϕU, λU, γ, w, v, uncertainty_set, MosekTools.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol)

    diag_x_E = Diagonal(x) * osp_data[:E]  # diag(x)E
    diag_λ_ψ = Diagonal(λ*ones(num_arcs)-v.*ψ0)
    xi_bar = uncertainty_set[:xi_bar]
    iter = 0

    past_obj = []
    past_subprob_obj = []
    past_upper_bound = []
    upper_bound = Inf
    result = Dict()
    result[:cuts] = Dict()

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
            x_sol, h_sol, λ_sol, ψ0_sol = value.(omp_vars[:x]), value.(omp_vars[:h]), value(omp_vars[:λ]), value.(omp_vars[:ψ0])
            t_0_sol = value(omp_vars[:t_0])

            (status, cut_info) = osp_optimize!(osp_model, osp_vars, osp_data, λ_sol, x_sol, h_sol, ψ0_sol)
            subprob_obj = cut_info[:obj_val]
            upper_bound = min(upper_bound, subprob_obj)

            if status != :OptimalityCut
                error("Subproblem returned unexpected status: $status")
            end

            if outer_tr
                lower_bound = max(lower_bound, t_0_sol)
                gap = upper_bound - lower_bound

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
                gap = upper_bound - t_0_sol
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
        if gap <= 1e-4
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
            @info "Iter $iter: LB=$(outer_tr ? round(lower_bound, digits=4) : round(t_0_sol, digits=4))  UB=$(round(upper_bound, digits=4))  gap=$(round(gap, digits=6))"
            cut_1 =  -ϕU * [sum((cut_info[:Uhat1][s,:,:] + cut_info[:Utilde1][s,:,:]) .* diag_x_E) for s in 1:S]
            cut_2 =  -ϕU * [sum((cut_info[:Uhat3][s,:,:] + cut_info[:Utilde3][s,:,:]) .* (osp_data[:E] - diag_x_E)) for s in 1:S]
            cut_3 =  [sum(cut_info[:Ztilde1_3][s,:,:] .* (diag_λ_ψ * diagm(xi_bar[s]))) for s in 1:S]
            cut_4 =  [(osp_data[:d0]'*cut_info[:βtilde1_1][s,:]) * λ for s in 1:S]
            cut_5 =  -1* [(h + diag_λ_ψ * xi_bar[s])'* cut_info[:βtilde1_3][s,:] for s in 1:S]
            cut_intercept = cut_info[:intercept]
            opt_cut = sum(cut_1)+ sum(cut_2)+ sum(cut_3)+ sum(cut_4)+ sum(cut_5)+ sum(cut_intercept)

            cut_added = @constraint(omp_model, t_0 >= opt_cut)
            set_name(cut_added, "opt_cut_$iter")
            result[:cuts]["opt_cut_$iter"] = cut_added

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
            if abs(subprob_obj - evaluate_expr(opt_cut, y)) > 1e-4
                println("something went wrong")
                @infiltrate
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