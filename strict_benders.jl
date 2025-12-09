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


function build_omp(network, ϕU, λU, γ, w; optimizer=nothing)
    # Extract network dimensions
    num_arcs = length(network.arcs)-1 #dummy arc 제외
    # Create model
    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => false))
    @variable(model, t_0 >= 0)  # Objective epigraph variable
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
    @objective(model, Min, t_0)
    vars = Dict(
        :t_0 => t_0,
        :λ => λ,
        :x => x,
        :h => h,
        :ψ0 => ψ0
    )
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
    if st == MOI.OPTIMAL
        obj_val = objective_value(osp_model)
        constant = value.(obj_term3) .+ value.(obj_term_ub_hat) .+ value.(obj_term_lb_hat) .+ value.(obj_term_ub_tilde) .+ value.(obj_term_lb_tilde)
        cut_coeff = Dict(
            :Uhat1 => value.(Uhat1),
            :Utilde1 => value.(Utilde1),
            :Uhat3 => value.(Uhat3),
            :Utilde3 => value.(Utilde3),
            :βtilde1_1 => value.(βtilde1_1),
            :βtilde1_3 => value.(βtilde1_3),
            :Ztilde1_3 => value.(Ztilde1_3),
            :constant => constant,
            :obj_val => obj_val
        )
        return (:OptimalityCut, cut_coeff)
    else
        t_status = termination_status(osp_model)
        @infiltrate
        error("Subproblem is not optimal")
    end
end

function initialize_omp(omp_model::Model, omp_vars::Dict, network, ϕU, λU, γ, w, uncertainty_set; optimizer=nothing)
    optimize!(omp_model) # 여기서 최적화를 하는 이유는 초기해를 뽑아서 subproblem을 build하기 위함임.
    st = MOI.get(omp_model, MOI.TerminationStatus())    
    @info "Initial status $st" # restricted master has a solution or is unbounded

    return st, value(omp_vars[:λ]), value.(omp_vars[:x]), value.(omp_vars[:h]), value.(omp_vars[:ψ0])
end

function strict_benders_optimize!(omp_model::Model, omp_vars::Dict, network, ϕU, λU, γ, w, uncertainty_set; optimizer=nothing)
    ### --------Begin Initialization--------
    st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars, network, ϕU, λU, γ, w, uncertainty_set; optimizer=optimizer)
    x, h, λ, ψ0, t_0 = omp_vars[:x], omp_vars[:h], omp_vars[:λ], omp_vars[:ψ0], omp_vars[:t_0]
    num_arcs = length(network.arcs) - 1
    osp_model, osp_vars, osp_data = build_dualized_outer_subproblem(network, S, ϕU, λU, γ, w, v, uncertainty_set, MosekTools.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol)
    
    diag_x_E = Diagonal(x) * osp_data[:E]  # diag(x)E
    diag_λ_ψ = Diagonal(λ*ones(num_arcs)-v.*ψ0)
    xi_bar = uncertainty_set[:xi_bar]
    iter = 0

    past_obj = []
    subprob_obj = []
    result = Dict()
    result[:cuts] = Dict()
    ### --------End Initialization--------
    while (st == MOI.DUAL_INFEASIBLE || st == MOI.OPTIMAL)
        iter += 1
        @info "Iteration $iter"
        optimize!(omp_model)
        st = MOI.get(omp_model, MOI.TerminationStatus())
        x_sol, h_sol, λ_sol, ψ0_sol = value.(omp_vars[:x]), value.(omp_vars[:h]), value(omp_vars[:λ]), value.(omp_vars[:ψ0])
        t_0_sol = value(omp_vars[:t_0])

        (status, cut_info) =osp_optimize!(osp_model, osp_vars, osp_data, λ_sol, x_sol, h_sol, ψ0_sol)
        if status == :OptimalityCut
            if t_0_sol >= cut_info[:obj_val]-1e-4
                @info "Termination condition met"
                println("t_0_sol: ", t_0_sol, ", cut_info[:obj_val]: ", cut_info[:obj_val])
                push!(past_obj, t_0_sol)
                push!(subprob_obj, cut_info[:obj_val])
                result[:past_obj] = past_obj
                result[:subprob_obj] = subprob_obj
                """
                    Variable types: 36 continuous, 17 integer (17 binary)
                    Coefficient statistics:
                    Matrix range     [3e-10, 4e+03]
                    Objective range  [1e+00, 1e+00]
                    Bounds range     [0e+00, 0e+00]
                    RHS range        [2e+00, 6e+02]
                    Warning: Model contains large matrix coefficient range
                            Consider reformulating model or setting NumericFocus parameter
                            to avoid numerical issues.

                    MIP start from previous solve produced solution with objective 13.3585 (0.00s)
                    Loaded MIP start from previous solve with objective 13.3585

                    Presolve removed 69 rows and 37 columns
                    Presolve time: 0.00s
                    Presolved: 17 rows, 16 columns, 80 nonzeros
                    Variable types: 12 continuous, 4 integer (4 binary)

                    Root relaxation: interrupted, 0 iterations, 0.00 seconds (0.00 work units)

                    Explored 1 nodes (0 simplex iterations) in 0.00 seconds (0.00 work units)
                    Thread count was 24 (of 24 available processors)

                    Solution count 1: 13.3585

                    Optimal solution found (tolerance 1.00e-04)
                    Best objective 1.335846384616e+01, best bound 1.335824729951e+01, gap 0.0016%

                    User-callback calls 300, time in user-callback 0.00 sec
                """
                return result
            else
                push!(past_obj, t_0_sol)
                push!(subprob_obj, cut_info[:obj_val])
                cut_1 =  -ϕU * [sum((cut_info[:Uhat1][s,:,:] + cut_info[:Utilde1][s,:,:]) .* diag_x_E) for s in 1:S]
                cut_2 =  -ϕU * [sum((cut_info[:Uhat3][s,:,:] + cut_info[:Utilde3][s,:,:]) .* (osp_data[:E] - diag_x_E)) for s in 1:S]
                cut_3 =  [sum(cut_info[:Ztilde1_3][s,:,:] .* (diag_λ_ψ * diagm(xi_bar[s]))) for s in 1:S]
                cut_4 =  [(osp_data[:d0]'*cut_info[:βtilde1_1][s,:]) * λ for s in 1:S]
                cut_5 =  -1* [(h + diag_λ_ψ * xi_bar[s])'* cut_info[:βtilde1_3][s,:] for s in 1:S]
                cut_const = cut_info[:constant]
                opt_cut = sum(cut_1)+ sum(cut_2)+ sum(cut_3)+ sum(cut_4)+ sum(cut_5)+ sum(cut_const)
                
                cut_added = @constraint(omp_model, t_0 >= opt_cut)
                set_name(cut_added, "opt_cut_$iter")
                
                
                result[:cuts]["opt_cut_$iter"] = cut_added
                
                println("subproblem objective: ", cut_info[:obj_val])
                @info "Optimality cut added"

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
                if abs(cut_info[:obj_val] - evaluate_expr(opt_cut, y)) > 1e-4
                    println("something went wrong")
                    @infiltrate
                end
            end
        end
    end
end