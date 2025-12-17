"""
Adding trust-region methods to stabilize benders convergence.

"""



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
includet("strict_benders.jl")


using .NetworkGenerator
"""
Build the Inner Master and Inner Subproblem
"""


function update_trust_region_constraints!(
    model::Model, 
    vars::Dict, 
    centers::Dict,
    B_bin::Float64, 
    B_con::Union{Float64, Nothing},
    old_cons::Dict,
    network
)
    interdictable_arc_indices = findall(network.interdictable_arcs)
    x, h, λ, ψ0 = vars[:x], vars[:h], vars[:λ], vars[:ψ0]
    xhat, hhat, λhat, ψ0hat = centers[:x], centers[:h], centers[:λ], centers[:ψ0]

    # Remove old constraints if they exist
    if old_cons[:binary] !== nothing
        delete(model, old_cons[:binary])
    end
    if old_cons[:continuous] !== nothing
        delete(model, old_cons[:continuous])
    end
    
    # ---- Binary Trust Region (L1-norm) ----
    # ||x - x̂||₁ = Σ_{k: x̂_k=1} (1-x_k) + Σ_{k: x̂_k=0} x_k ≤ B_bin
    tr_binary_expr = @expression(model,
        sum((1 - x[k]) for k in interdictable_arc_indices if abs(xhat[k] - 1.0) < 1e-6) +
        sum(x[k] for k in interdictable_arc_indices if abs(xhat[k]) < 1e-6)
    )
    new_tr_binary = @constraint(model, tr_binary_expr <= B_bin)
    set_name(new_tr_binary, "TR_binary")
    
    # # ---- Continuous Trust Region (L2-norm) ----
    # # ||h - ĥ||₂² + (λ - λ̂)² + ||ψ0 - ψ̂0||₂² ≤ B_con²
    # new_tr_continuous = @constraint(model,
    #     sum((h[k] - ĥ[k])^2 for k in 1:num_arcs) +
    #     (λ - λ̂)^2 +
    #     sum((ψ0[k] - ψ̂0[k])^2 for k in 1:num_arcs)
    #     <= B_con^2
    # )
    # set_name(new_tr_continuous, "TR_continuous")
    new_tr_continuous = nothing

    new_cons = Dict(
        :binary => new_tr_binary,
        :continuous => new_tr_continuous
    )
    
    return new_cons
end

function add_reverse_region_constraint!(model, x, xhat, B_old, network)
    interdictable_arc_indices = findall(network.interdictable_arcs)
    
    reverse_expr = @expression(model,
        sum((1 - x[k]) for k in interdictable_arc_indices if abs(xhat[k] - 1.0) < 1e-6) +
        sum(x[k] for k in interdictable_arc_indices if abs(xhat[k]) < 1e-6)
    )
    
    reverse_con = @constraint(model, reverse_expr >= B_old + 1)
    set_name(reverse_con, "reverse_region")
    
    @info "  Added reverse region constraint: ||x - x̂_old||₁ ≥ $(B_old + 1)"
    
    return reverse_con
end

function build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_optimizer=nothing)
    num_arcs = length(network.arcs) - 1
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    S = length(xi_bar)
    flow_upper = sum(sum(xi_bar[s] for s in 1:S))
    model = Model(optimizer_with_attributes(mip_optimizer, MOI.Silent() => false))
    @variable(model, t_1_l[s=1:S], upper_bound= flow_upper)
    @variable(model, t_1_f[s=1:S], upper_bound= flow_upper)
    @variable(model, α[k=1:num_arcs] >= 0)
    @constraint(model, sum(α) == w*(1/S)) # full model에선 자연스럽게 inequality가 equality가 되지만 decomposed된 imp에선 그런다는 보장이 없으므로 명시적으로 equality 유지
    @objective(model, Max, sum(t_1_l) + sum(t_1_f))

    vars = Dict(
        :t_1_l => t_1_l,
        :t_1_f => t_1_f,
        :α => α
    )
    return model, vars
end

function isp_leader_optimize!(isp_leader_model::Model, isp_leader_vars::Dict; isp_data=nothing, uncertainty_set::Dict=nothing, λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing, α_sol=nothing)
    model, vars = isp_leader_model, isp_leader_vars
    E, ϕU, d0 = isp_data[:E], isp_data[:ϕU], isp_data[:d0]
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    diag_x_E = Diagonal(x_sol) * E  # diag(x)E
    true_S = isp_data[:S]
    
    S = 1
    ## update objective if necessary
    Uhat1, Uhat3, Phat1_Φ, Phat1_Π, Phat2_Φ, Phat2_Π = vars[:Uhat1], vars[:Uhat3], vars[:Phat1_Φ], vars[:Phat1_Π], vars[:Phat2_Φ], vars[:Phat2_Π]
    βhat1_1 = vars[:βhat1_1]
    obj_term1 = [-ϕU * sum(Uhat1[s, :, :] .* diag_x_E) for s=1:S]
    obj_term2 = [-ϕU * sum(Uhat3[s, :, :] .* (E-diag_x_E)) for s=1:S]
    obj_term3 = [(d0')* βhat1_1[s,:] for s=1:S] #이거만 maximize하면 dual infeasible

    obj_term_ub_hat = [-ϕU * sum(Phat1_Φ[s,:,:]) - ϕU * sum(Phat1_Π[s,:,:]) for s=1:S]
    obj_term_lb_hat = [-ϕU * sum(Phat2_Φ[s,:,:]) - ϕU * sum(Phat2_Π[s,:,:]) for s=1:S]
    @objective(model, Max, sum(obj_term1) + sum(obj_term2) + sum(obj_term3)
    + sum(obj_term_ub_hat) + sum(obj_term_lb_hat))

    ## update constraints
    coupling_cons = vec(model[:coupling_cons])
    set_normalized_rhs.(coupling_cons, α_sol)
    ## optimize model
    optimize!(model)
    st = MOI.get(model, MOI.TerminationStatus())
    if (st == MOI.OPTIMAL) || (st == MOI.SLOW_PROGRESS)
        ## obtain cuts
        μhat = shadow_price.(coupling_cons) # subgradient
        ηhat = shadow_price.(vec(model[:cons_dual_constant]))
        intercept, subgradient = (1/true_S)*sum(ηhat), μhat ##실제 S로 나눠주어야 함.
        dual_obj = intercept + α_sol'*subgradient
        #dual model의 목적함수를 shadow price로 query해서 evaluate한 뒤 strong duality 성립하는지 확인
        @assert abs(dual_obj - objective_value(model)) < 1e-4
        cut_coeff = Dict(:μhat=>μhat, :intercept=>intercept, :obj_val=>dual_obj)
        return (:OptimalityCut, cut_coeff)
    else
        t_status = termination_status(model)
        @infiltrate
        error("Subproblem is not optimal")
    end
end

function isp_follower_optimize!(isp_follower_model::Model, isp_follower_vars::Dict; isp_data=nothing, uncertainty_set::Dict=nothing, λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing, α_sol=nothing)
    model, vars = isp_follower_model, isp_follower_vars
    E, ϕU, d0 = isp_data[:E], isp_data[:ϕU], isp_data[:d0]
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    diag_x_E = Diagonal(x_sol) * E  # diag(x)E
    num_arcs = length(x_sol)
    diag_λ_ψ = Diagonal(λ_sol*ones(num_arcs)-v.*ψ0_sol)
    true_S = isp_data[:S]
    S = 1
    ## update objective if necessary
    Utilde1, Utilde3, Ztilde1_3, Ptilde1_Φ, Ptilde1_Π, Ptilde2_Φ, Ptilde2_Π, Ptilde1_Y, Ptilde1_Yts, Ptilde2_Y, Ptilde2_Yts = vars[:Utilde1], vars[:Utilde3], vars[:Ztilde1_3], vars[:Ptilde1_Φ], vars[:Ptilde1_Π], vars[:Ptilde2_Φ], vars[:Ptilde2_Π], vars[:Ptilde1_Y], vars[:Ptilde1_Yts], vars[:Ptilde2_Y], vars[:Ptilde2_Yts]
    βtilde1_1, βtilde1_3 = vars[:βtilde1_1], vars[:βtilde1_3]
    obj_term1 = [-ϕU * sum(Utilde1[s, :, :] .* diag_x_E) for s=1:S]
    obj_term2 = [-ϕU * sum(Utilde3[s, :, :] .* (E-diag_x_E)) for s=1:S]
    obj_term4 = [sum(Ztilde1_3[s, :, :] .* (diag_λ_ψ * diagm(xi_bar[s]))) for s=1:S]
    obj_term5 = [(λ_sol*d0')* βtilde1_1[s,:] for s=1:S] #이거만 maximize하면 dual infeasible
    obj_term6 = [-(h_sol + diag_λ_ψ * xi_bar[s])'* βtilde1_3[s,:] for s=1:S]

    obj_term_ub_tilde = [-ϕU * sum(Ptilde1_Φ[s,:,:]) - ϕU * sum(Ptilde1_Π[s,:,:]) - ϕU * sum(Ptilde1_Y[s,:,:]) - ϕU * sum(Ptilde1_Yts[s,:]) for s=1:S]
    obj_term_lb_tilde = [-ϕU * sum(Ptilde2_Φ[s,:,:]) - ϕU * sum(Ptilde2_Π[s,:,:]) - ϕU * sum(Ptilde2_Y[s,:,:]) - ϕU * sum(Ptilde2_Yts[s,:]) for s=1:S]
    @objective(model, Max, sum(obj_term1) + sum(obj_term2) + sum(obj_term4) + sum(obj_term5) + sum(obj_term6)
    + sum(obj_term_ub_tilde) + sum(obj_term_lb_tilde))


    ## update constraints
    coupling_cons = vec(model[:coupling_cons])
    set_normalized_rhs.(coupling_cons, α_sol)
    ## optimize model
    optimize!(model)
    st = MOI.get(model, MOI.TerminationStatus())
    if (st == MOI.OPTIMAL) || (st == MOI.SLOW_PROGRESS)
        ## obtain cuts
        μtilde = shadow_price.(coupling_cons) # subgradient
        # ηtilde = shadow_price.(vec(model[:cons_dual_constant]))
        # intercept = (1/true_S)*sum(ηtilde) ##실제 S로 나눠주어야 함.
        ηtilde_pos = shadow_price.(vec(model[:cons_dual_constant_pos]))
        ηtilde_neg = shadow_price.(vec(model[:cons_dual_constant_neg]))
        intercept = sum((1/true_S)*(ηtilde_pos-ηtilde_neg)) ## 이러면 ηtilde sign 반대로 나오는거 robust하게 대응 가능.
        subgradient = μtilde
        dual_obj = intercept + α_sol'*subgradient
        #dual model의 목적함수를 shadow price로 query해서 evaluate한 뒤 strong duality 성립하는지 확인
        if abs(dual_obj - objective_value(model)) > 1e-4
            @infiltrate
            # intercept = -1*intercept
            # @assert abs(intercept + α_sol'*subgradient - objective_value(model)) < 1e-4
        end
        cut_coeff = Dict(:μtilde=>μtilde, :intercept=>intercept, :obj_val=>dual_obj)
        return (:OptimalityCut, cut_coeff)
    else
        t_status = termination_status(model)
        @infiltrate
        error("Subproblem is not optimal")
    end
end

function imp_optimize!(imp_model::Model, imp_vars::Dict, isp_leader_instances::Dict, isp_follower_instances::Dict; isp_data=nothing, λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing, outer_iter=nothing, imp_cuts=nothing)
    st = MOI.get(imp_model, MOI.TerminationStatus())
    iter = 0
    uncertainty_set = isp_data[:uncertainty_set]
    past_obj = []
    past_subprob_obj = []
    past_lower_bound = []
    lower_bound = -Inf ## inner master problem은 Maximization이니까 feasible solution은 lower bound를 제공.
    result = Dict()
    result[:cuts] = Dict()
    ##
    ## 여기서 imp 초기화해야함.
    if outer_iter>1
        for (cut_name, cut) in imp_cuts[:old_cuts]
            delete(imp_model, cut)
        end
    end
    ##
    while (st == MOI.DUAL_INFEASIBLE || st == MOI.OPTIMAL)
        iter += 1
        @info "Inner Benders Iteration $iter"
        optimize!(imp_model)
        st = MOI.get(imp_model, MOI.TerminationStatus())
        α_sol = value.(imp_vars[:α])
        t_1_sol = sum(value.(imp_vars[:t_1_l])) + sum(value.(imp_vars[:t_1_f]))
        subprob_obj = 0
        dict_cut_info_l, dict_cut_info_f = Dict(), Dict()
        status = true
        for s in 1:S
            U_s = Dict(:R => Dict(:1=>R[s]), :r_dict => Dict(:1=>r_dict[s]), :xi_bar => Dict(:1=>xi_bar[s]), :epsilon => epsilon)
            (status_l, cut_info_l) =isp_leader_optimize!(isp_leader_instances[s][1], isp_leader_instances[s][2]; isp_data=isp_data, uncertainty_set=U_s, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
            (status_f, cut_info_f) =isp_follower_optimize!(isp_follower_instances[s][1], isp_follower_instances[s][2]; isp_data=isp_data, uncertainty_set=U_s, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
            status = status &&(status_l == :OptimalityCut) && (status_f == :OptimalityCut)
            dict_cut_info_l[s] = cut_info_l
            dict_cut_info_f[s] = cut_info_f
            subprob_obj += cut_info_l[:obj_val]+cut_info_f[:obj_val]
        end
        lower_bound = max(lower_bound, subprob_obj) ## inner master problem은 Maximization이니까 우린 항상 더 높은 값을 추구
        if status == true 
            if t_1_sol <= lower_bound+1e-4
                @info "Termination condition met"
                println("t_1_sol: ", t_1_sol, ", subprob_obj: ", subprob_obj)
                push!(past_obj, t_1_sol)
                push!(past_subprob_obj, subprob_obj)
                push!(past_lower_bound, lower_bound)
                result[:past_obj] = past_obj
                result[:past_subprob_obj] = past_subprob_obj
                result[:α_sol] = value.(imp_vars[:α])
                result[:obj_val] = objective_value(imp_model)
                result[:past_lower_bound] = past_lower_bound
                return (:OptimalityCut, result)
            else
                push!(past_obj, t_1_sol)
                push!(past_subprob_obj, subprob_obj)
                push!(past_lower_bound, lower_bound)
                subgradient_l = [dict_cut_info_l[s][:μhat] for s in 1:S]
                subgradient_f = [dict_cut_info_f[s][:μtilde] for s in 1:S]
                intercept_l = [dict_cut_info_l[s][:intercept] for s in 1:S]
                intercept_f = [dict_cut_info_f[s][:intercept] for s in 1:S]
                
                cut_added_l = @constraint(imp_model, [s=1:S], imp_vars[:t_1_l][s] <= intercept_l[s] + imp_vars[:α]'*subgradient_l[s])
                cut_added_f = @constraint(imp_model, [s=1:S], imp_vars[:t_1_f][s] <= intercept_f[s] + imp_vars[:α]'*subgradient_f[s])
                set_name.(cut_added_l, ["opt_cut_$(iter)_l_s$(s)" for s in 1:S])
                set_name.(cut_added_f, ["opt_cut_$(iter)_f_s$(s)" for s in 1:S])
                result[:cuts]["opt_cut_$(iter)_l"] = cut_added_l
                result[:cuts]["opt_cut_$(iter)_f"] = cut_added_f
                println("subproblem objective: ", subprob_obj)
                @info "Optimality cut added"

                """
                below evaluation is checking tightness of the cut
                """
                y = Dict(
                    [imp_vars[:α][k] => α_sol[k] for k in 1:length(α_sol)]...,
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
                opt_cut_val = sum(evaluate_expr(intercept_l[s] + imp_vars[:α]'*subgradient_l[s], y) for s in 1:S) + sum(evaluate_expr(intercept_f[s] + imp_vars[:α]'*subgradient_f[s], y) for s in 1:S)
                if abs(subprob_obj - opt_cut_val) > 1e-4
                    println("something went wrong")
                    @infiltrate
                end
            end
        end
    end
end

function initialize_imp(imp_model::Model, imp_vars::Dict)
    optimize!(imp_model)
    st = MOI.get(imp_model, MOI.TerminationStatus())
    α_sol = value.(imp_vars[:α])
    return st, α_sol
end

function initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set; conic_optimizer=nothing, λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing, α_sol=nothing)
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    leader_instances = Dict{Int, Tuple{Model, Dict}}()
    follower_instances = Dict{Int, Tuple{Model, Dict}}()
    for s in 1:S
        U_s = Dict(:R => Dict(:1=>R[s]), :r_dict => Dict(:1=>r_dict[s]), :xi_bar => Dict(:1=>xi_bar[s]), :epsilon => epsilon)
        leader_instances[s] = build_isp_leader(network, 1, ϕU, λU, γ, w, v, U_s, conic_optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, S)
        follower_instances[s] = build_isp_follower(network, 1, ϕU, λU, γ, w, v, U_s, conic_optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, S)
        
    end
    return leader_instances, follower_instances
end

function evaluate_master_opt_cut(isp_leader_instances::Dict, isp_follower_instances::Dict, isp_data::Dict, cut_info::Dict, iter::Int; multi_cut=false)
    S = isp_data[:S]
    α_sol = cut_info[:α_sol]
    status = true
    for s in 1:S
        model_l = isp_leader_instances[s][1]
        model_f = isp_follower_instances[s][1]
        set_normalized_rhs.(vec(model_l[:coupling_cons]), α_sol)
        optimize!(model_l)
        st_l = MOI.get(model_l, MOI.TerminationStatus())

        set_normalized_rhs.(vec(model_f[:coupling_cons]), α_sol)
        optimize!(model_f)
        st_f = MOI.get(model_f, MOI.TerminationStatus())

        status = status && (st_l == MOI.OPTIMAL) && (st_f == MOI.OPTIMAL)
        if status == false
            if (st_l == MOI.SLOW_PROGRESS) || (st_f == MOI.SLOW_PROGRESS)
                status = true
            else
                @infiltrate
            end
        end
    end
    Uhat1 = cat([value.(isp_leader_instances[s][2][:Uhat1]) for s in 1:S]...; dims=1)
    Utilde1 = cat([value.(isp_follower_instances[s][2][:Utilde1]) for s in 1:S]...; dims=1)
    Uhat3 = cat([value.(isp_leader_instances[s][2][:Uhat3]) for s in 1:S]...; dims=1)
    Utilde3 = cat([value.(isp_follower_instances[s][2][:Utilde3]) for s in 1:S]...; dims=1)
    Ztilde1_3 = cat([value.(isp_follower_instances[s][2][:Ztilde1_3]) for s in 1:S]...; dims=1)
    βtilde1_1 = cat([value.(isp_follower_instances[s][2][:βtilde1_1]) for s in 1:S]...; dims=1)
    βtilde1_3 = cat([value.(isp_follower_instances[s][2][:βtilde1_3]) for s in 1:S]...; dims=1)

    if multi_cut
        intercept_l = [value.(isp_leader_instances[s][2][:intercept]) for s in 1:S]
        intercept_f = [value.(isp_follower_instances[s][2][:intercept]) for s in 1:S]
        intercept = sum(intercept_l) + sum(intercept_f)
    else
        intercept = sum(value.(isp_leader_instances[s][2][:intercept]) for s in 1:S) + sum(value.(isp_follower_instances[s][2][:intercept]) for s in 1:S)
        intercept_l, intercept_f = nothing, nothing
    end
    leader_obj = sum(objective_value(isp_leader_instances[s][1]) for s in 1:S)
    follower_obj = sum(objective_value(isp_follower_instances[s][1]) for s in 1:S)
    println("summation of leader and follower objective: ", leader_obj+follower_obj, ", cut_info[:obj_val]: ", cut_info[:obj_val])
    println("Outer loop iteration: ", iter)
    @assert abs(leader_obj + follower_obj - cut_info[:obj_val]) < 1e-3
    return Dict(:Uhat1=>Uhat1, :Utilde1=>Utilde1, :Uhat3=>Uhat3, :Utilde3=>Utilde3, :Ztilde1_3=>Ztilde1_3
    ,:βtilde1_1=>βtilde1_1, :βtilde1_3=>βtilde1_3, :intercept=>intercept, :intercept_l=>intercept_l, :intercept_f=>intercept_f)
end


function nested_benders_optimize!(omp_model::Model, omp_vars::Dict, network, ϕU, λU, γ, w, uncertainty_set; mip_optimizer=nothing, conic_optimizer=nothing, multi_cut=false)
    ### -------- Trust Region 초기화 --------
    B_bin_sequence = [0.05, 0.5, 1.0]
    B_bin_stage = 1
    B_bin = B_bin_sequence[B_bin_stage] * sum(network.interdictable_arcs)
    B_con = nothing # 나중에 생각
    ## Stability Centers
    # Stability centers (will be initialized after first solve)
    centers = Dict{Symbol, Any}(
        :x => nothing,
        :h => nothing,
        :λ => nothing,
        :ψ0 => nothing # 근데 이건 굳이 해야하나? x*lambda인데
    )
    upper_bound = Inf # Will be updated after first subproblem solve
    ## Serious Step Parameters
    β_relative = 1e-4 # serious improvement threshold
    tr_constraints = Dict{Symbol, Any}(
        :binary => nothing,
        :continuous => nothing
    )
    ### --------Begin Outer Master problemInitialization--------
    st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars)
    x, h, λ, ψ0 = omp_vars[:x], omp_vars[:h], omp_vars[:λ], omp_vars[:ψ0]
     # Initialize stability centers with first solution
     centers[:x] = value.(x)
     centers[:h] = value.(h)
     centers[:λ] = value.(λ)
     centers[:ψ0] = value.(ψ0)
    if multi_cut
        t_0_l = omp_vars[:t_0_l]
        t_0_f = omp_vars[:t_0_f]
        t_0 = t_0_l + t_0_f
    else
        t_0 = omp_vars[:t_0]
    end

    num_arcs = length(network.arcs) - 1
    E = ones(num_arcs, num_arcs+1) # num_arcs × num_arcs+1 matrix of ones
    diag_x_E = Diagonal(x) * E  # diag(x)E
    diag_λ_ψ = Diagonal(λ*ones(num_arcs)-v.*ψ0)
    d0 = zeros(num_arcs + 1)
    d0[end] = 1.0
    xi_bar = uncertainty_set[:xi_bar]
    iter = 0
    past_obj = []
    past_major_subprob_obj = [] # major (serious) step에서 변화한 subproblem objective들만 모음
    past_minor_subprob_obj = [] # minor (null) step에서 구한 subproblem objective들 다 모음
    past_model_estimate = [] # 매 cutting plane omp의 objective 저장
    past_local_lower_bound = [] # reverse region에서 구한 local lower bound 저장
    past_upper_bound = []
    past_lower_bound = []
    past_local_optimizer = []
    major_iter = []
    bin_B_steps = [] # B_bin이 몇번째 outer loop에서 바꼈는지 체크
    # null_steps = []
    imp_cuts = Dict{Symbol, Any}()
    result = Dict()
    result[:cuts] = Dict()
    result[:tr_info] = Dict()
    upper_bound = Inf
    lower_bound = -Inf
    ### --------Begin Inner Master, Subproblem Initialization--------
    imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_optimizer=mip_optimizer)
    st, α_sol = initialize_imp(imp_model, imp_vars)
    leader_instances, follower_instances = initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set; conic_optimizer=conic_optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
    isp_data = Dict(:E => E, :network => network, :ϕU => ϕU, :λU => λU, :γ => γ, :w => w, :v => v, :uncertainty_set => uncertainty_set, :d0 => d0, :S=>S)
    ### --------End Initialization--------    
    while (st == MOI.DUAL_INFEASIBLE || st == MOI.OPTIMAL)
        iter += 1
        @info "Outer Iteration $iter (B_z=$B_z, Stage=$(B_z_stage+1)/$(length(B_z_sequence)))"
        # Outer Master Problem 풀기
        optimize!(omp_model)
        st = MOI.get(omp_model, MOI.TerminationStatus())
        x_sol, h_sol, λ_sol, ψ0_sol = value.(omp_vars[:x]), value.(omp_vars[:h]), value(omp_vars[:λ]), value.(omp_vars[:ψ0])
        model_estimate = value(t_0)
        lower_bound = max(lower_bound, model_estimate)
        # Outer Subproblem 풀기
        status, cut_info =imp_optimize!(imp_model, imp_vars, leader_instances, follower_instances; isp_data=isp_data, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, outer_iter=iter, imp_cuts=imp_cuts)
        if status != :OptimalityCut
            @warn "Outer Subproblem not optimal"
            @infiltrate
        end
        imp_cuts[:old_cuts] = cut_info[:cuts] ## 다음 iteration에서 지우기 위해 여기에 저장함
        subprob_obj = cut_info[:obj_val]
        upper_bound = min(upper_bound, subprob_obj)
        
        # Measure of progress
        if iter==1
            push!(past_major_subprob_obj, subprob_obj)
        end
        gap = upper_bound - lower_bound # gap between upper bound and lower bound
        # Serious Test
        tr_needs_update = false  # Flag for TR constraint update
        predicted_decrease = past_major_subprob_obj[end] - model_estimate # serious(major) step 가장 최근 subprob obj와 지금 구한 obj의 차이
        β_dynamic = max(1e-8, β_relative * predicted_decrease)  # 최소값 보장
        improvement = past_major_subprob_obj[end] - subprob_obj # decrease in the actual objective
        is_serious_step = (improvement >= β_dynamic) # decrease in the actual objective is at least some fraction of the decrease predicted by the model
        if is_serious_step
            # Serious Step: Move stability center
            centers[:x] = value.(x_sol)
            centers[:h] = value.(h_sol)
            centers[:λ] = value.(λ_sol)
            centers[:ψ0] = value.(ψ0_sol)
            push!(major_iter, iter)
            push!(past_major_subprob_obj, subprob_obj)
            # TR constraint needs an update with new center
            tr_needs_update = true
        end
        # 배열에 history 저장
        push!(past_lower_bound, lower_bound)
        push!(past_model_estimate, model_estimate)
        push!(past_minor_subprob_obj, subprob_obj)
        push!(past_upper_bound, upper_bound)
        if gap <= 1e-4
            # Local optimality 달성
            if B_bin_stage <= length(B_bin_sequence)-1
                # Trust region 확장
                B_bin_stage +=1
                B_bin_old = B_bin
                B_bin = B_bin_sequence[B_bin_stage] * sum(network.interdictable_arcs)
                push!(bin_B_steps, iter)
                push!(past_local_lower_bound, lower_bound)
                push!(past_local_optimizer, Dict(:x=>value.(x_sol), :h=>value.(h_sol), :λ=>value.(λ_sol), :ψ0=>value.(ψ0_sol)))
                @info "  ✓ Local optimal reached! Expanding B_bin to $B_bin"
                # TR constraint needs update (B_bin changed)
                tr_needs_update = true
                @info "Updating Trust Region"
                ## trust region radius를 확장
                tr_constraints = update_trust_region_constraints!(
                    omp_model, omp_vars, centers, B_bin, B_con, tr_constraints, network)
                lower_bound = -Inf # master problem 영역 확장했으니까 다시 초기화
                    # Reverse region constraint 추가 (선택사항)
                reverse_constraints = add_reverse_region_constraint!(omp_model, omp_vars[:x], centers[:x], B_bin_old, network)
                

                    # # Optional: Add reverse region constraint
                # if use_reverse_constraints
                #     reverse_con = add_reverse_region_constraint!(
                #         omp_model, x, centers[:x], B_z_old, network
                #     )
                #     push!(reverse_constraints, reverse_con)
                # end
            else
                # Global Optimality 달성
                @info "  ✓✓ GLOBAL OPTIMAL! (B_bin = full region)"
                # past_local_lower_bound 배열에서 최소값의 인덱스를 찾음
                min_idx = argmin(past_local_lower_bound)
                global_lower_bound = past_local_lower_bound[min_idx]
                iter_when_global_optimal = bin_B_steps[min_idx]
                global_upper_bound = past_upper_bound[iter_when_global_optimal]
                println("lower_bound: ", global_lower_bound, ", upper_bound: ", global_upper_bound)

                result[:past_lower_bound] = past_lower_bound
                result[:past_local_lower_bound] = past_local_lower_bound
                result[:past_minor_subprob_obj] = past_minor_subprob_obj
                result[:past_major_subprob_obj] = past_major_subprob_obj
                result[:past_upper_bound] = past_upper_bound
                result[:tr_info][:final_B_bin_stage] = B_bin_stage
                result[:tr_info][:final_B_bin] = B_bin
                # result[:tr_info][:final_B_con] = B_con
                result[:tr_info][:major_iter] = major_iter
                result[:tr_info][:bin_B_steps] = bin_B_steps
                # result[:tr_info][:null_steps] = null_steps
                result[:opt_sol] = past_local_optimizer[min_idx]
                result[:iter_when_global_optimal] = iter_when_global_optimal
                """
                Optimize a model with 740 rows, 143 columns and 75687 nonzeros
                Model fingerprint: 0x74f2eaf8
                Variable types: 96 continuous, 47 integer (47 binary)
                Coefficient statistics:
                Matrix range     [3e-11, 1e+03]
                Objective range  [1e+00, 1e+00]
                Bounds range     [0e+00, 0e+00]
                RHS range        [1e-02, 7e+02]
                Warning: Model contains large matrix coefficient range
                        Consider reformulating model or setting NumericFocus parameter
                        to avoid numerical issues.
                """
                return result
            end
        else
            # Gap still large → Add cut and continue
            outer_cut_info = evaluate_master_opt_cut(leader_instances, follower_instances, isp_data, cut_info, iter, multi_cut=multi_cut)
            if multi_cut
                cut_1_l =  -ϕU * [sum(outer_cut_info[:Uhat1][s,:,:] .* diag_x_E) for s in 1:S]
                cut_1_f =  -ϕU * [sum(outer_cut_info[:Utilde1][s,:,:] .* diag_x_E) for s in 1:S]
                cut_2_l =  -ϕU * [sum(outer_cut_info[:Uhat3][s,:,:] .* (isp_data[:E] - diag_x_E)) for s in 1:S]
                cut_2_f =  -ϕU * [sum(outer_cut_info[:Utilde3][s,:,:] .* (isp_data[:E] - diag_x_E)) for s in 1:S]
                cut_3_f =  [sum(outer_cut_info[:Ztilde1_3][s,:,:] .* (diag_λ_ψ * diagm(xi_bar[s]))) for s in 1:S]
                cut_4_f =  [(isp_data[:d0]'*outer_cut_info[:βtilde1_1][s,:]) * λ for s in 1:S]
                cut_5_f =  -1* [(h + diag_λ_ψ * xi_bar[s])'* outer_cut_info[:βtilde1_3][s,:] for s in 1:S]
                cut_intercept_l = outer_cut_info[:intercept_l]
                cut_intercept_f = outer_cut_info[:intercept_f]
                opt_cut_l = sum(cut_1_l)+ sum(cut_2_l) + sum(cut_intercept_l)
                opt_cut_f = sum(cut_1_f)+ sum(cut_2_f)+ sum(cut_3_f)+ sum(cut_4_f)+ sum(cut_5_f) + sum(cut_intercept_f)
                # 여기도 multi-cut 구현할 순 있음.
                ## 심지어 scenario별로 더 epigrph variable을 만들어서할수도있음.
                cut_added_l = @constraint(omp_model, t_0_l >= opt_cut_l)
                cut_added_f = @constraint(omp_model, t_0_f >= opt_cut_f)
                set_name(cut_added_l, "opt_cut_$(iter)_l")
                set_name(cut_added_f, "opt_cut_$(iter)_f")
                result[:cuts]["opt_cut_$(iter)_l"] = cut_added_l
                result[:cuts]["opt_cut_$(iter)_f"] = cut_added_f
            else
                cut_1 =  -ϕU * [sum((outer_cut_info[:Uhat1][s,:,:] + outer_cut_info[:Utilde1][s,:,:]) .* diag_x_E) for s in 1:S]
                cut_2 =  -ϕU * [sum((outer_cut_info[:Uhat3][s,:,:] + outer_cut_info[:Utilde3][s,:,:]) .* (isp_data[:E] - diag_x_E)) for s in 1:S]
                cut_3 =  [sum(outer_cut_info[:Ztilde1_3][s,:,:] .* (diag_λ_ψ * diagm(xi_bar[s]))) for s in 1:S]
                cut_4 =  [(isp_data[:d0]'*outer_cut_info[:βtilde1_1][s,:]) * λ for s in 1:S]
                cut_5 =  -1* [(h + diag_λ_ψ * xi_bar[s])'* outer_cut_info[:βtilde1_3][s,:] for s in 1:S]
                cut_intercept = outer_cut_info[:intercept]
                opt_cut = sum(cut_1)+ sum(cut_2)+ sum(cut_3)+ sum(cut_4)+ sum(cut_5)+ sum(cut_intercept)
                # 여기도 multi-cut 구현할 순 있음.
                cut_added = @constraint(omp_model, t_0 >= opt_cut)
                set_name(cut_added, "opt_cut_$iter")
                result[:cuts]["opt_cut_$iter"] = cut_added
            end
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
            if multi_cut
                opt_cut = opt_cut_l + opt_cut_f
            end
            if abs(subprob_obj - evaluate_expr(opt_cut, y)) > 1e-3
                println("something went wrong")
                @infiltrate
            end
            println("subproblem objective: ", subprob_obj)
            @info "Optimality cut added"

            # Update TR constraints if needed
            if tr_needs_update
                @info "Updating Trust Region"
                tr_constraints = update_trust_region_constraints!(
                    omp_model, omp_vars, centers, B_bin, B_con, tr_constraints, network)
            end
        end
    end
end



function build_isp_leader(network, S, ϕU, λU, γ, w, v, uncertainty_set, optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, true_S)
    # Extract network dimensions
    num_nodes = length(network.nodes)
    num_arcs = length(network.arcs)-1 #dummy arc 제외
    num_interdictable = sum(network.interdictable_arcs)


    
    # Node-arc incidence matrix (excluding source row)
    N = network.N
    N_y = N[:, 1:num_arcs]  # Regular arcs: (|V|-1) × |A|
    N_ts = N[:, end]         # Dummy arc: (|V|-1) × 1
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    # Dummy arc index (t,s)
    dummy_arc_idx = findfirst(arc -> arc == ("t", "s"), network.arcs)
    E = ones(num_arcs, num_arcs+1) # num_arcs × num_arcs+1 matrix of ones
    I_0 = [Matrix{Float64}(I, num_arcs, num_arcs) zeros(num_arcs)]
    # Create model
    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))
    d0 = zeros(num_arcs + 1)
    d0[end] = 1.0
    println("Building dualized outer subproblem...")
    println("  Nodes: $num_nodes, Arcs: $num_arcs, Scenarios: $S")
    println("  Interdictable arcs: $num_interdictable")
    println("  Dummy arc index: $dummy_arc_idx")
    println("  Parameters: ϕU = $ϕU, λU = $λU, γ = $γ, w = $w, v = $v")

    # =========================================================================
    # DECISION VARIABLES
    # =========================================================================
    λ, x, h, ψ0 = λ_sol, x_sol, h_sol, ψ0_sol
    # --- Scalar variables ---
    α = α_sol
    # --- Vector variables ---
    dim_Λhat1_rows = (num_arcs + 1) + (num_nodes - 1) + num_arcs ## equal to dim_Λhat1_rows in full model
    dim_Λhat2_rows = num_arcs ## equal to dim_Λhat2_rows in full model
    @variable(model, βhat1[s=1:S,1:dim_Λhat1_rows]>=0)
    @variable(model, βhat2[s=1:S,1:dim_Λhat2_rows]>=0)
    βhat1_1 = βhat1[:,1:num_arcs+1]
    block2_start = num_arcs+2
    block3_start = block2_start + num_nodes-1
    βhat1_2 = βhat1[:,block2_start:block3_start-1]
    βhat1_3 = βhat1[:,block3_start:end]
    block2_start, block3_start= -1, -1 ## 이후에 다시 쓰이는데 초기화
    @assert sum([size(βhat1_1,2), size(βhat1_2,2), size(βhat1_3,2)]) == dim_Λhat1_rows
    #βtilde1 block 분리
    block2_start = num_arcs+2
    block3_start = block2_start + num_nodes-1
    block4_start = block3_start + num_arcs
    block5_start = block4_start + num_nodes-1
    block6_start = block5_start + num_arcs
    block2_start, block3_start, block4_start, block5_start, block6_start= -1, -1, -1, -1, -1 ## 이후에 다시 쓰이는데 초기화
    # check if the blocks are correct (block들 column dimension 합이 dim_Λtilde1_rows와 같은지 확인)
    # --- Matrix variables ---
    @variable(model, Mhat[s=1:S,1:num_arcs+1,1:num_arcs+1])
    @variable(model, Uhat1[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    @variable(model, Uhat2[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    @variable(model, Uhat3[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    dim_R_cols = size(R[1],2)
    @variable(model, Zhat1[s=1:S,1:dim_Λhat1_rows,1:dim_R_cols])
    @variable(model, Zhat2[s=1:S,1:dim_Λhat2_rows,1:dim_R_cols])
    # Zhat1도 3개 블록으로 분리, sdp_build_full_model.jl 참고
    Zhat1_1 = Zhat1[:,1:num_arcs+1,:]
    block2_start = num_arcs+2
    block3_start = block2_start + num_nodes-1
    Zhat1_2 = Zhat1[:,block2_start:block3_start-1,:]
    Zhat1_3 = Zhat1[:,block3_start:end,:]
    block2_start, block3_start= -1, -1 ## 이후에 다시 쓰이는데 초기화
    # check if the blocks are correct (block들 column dimension 합이 dim_Λhat1_rows와 같은지 확인)
    @assert sum([size(Zhat1_1,2), size(Zhat1_2,2), size(Zhat1_3,2)]) == dim_Λhat1_rows
    # check if the blocks are correct (block들 column dimension 합이 dim_Λtilde1_rows와 같은지 확인)
    @variable(model, Γhat1[s=1:S, 1:dim_Λhat1_rows, 1:size(R[1],1)])
    @variable(model, Γhat2[s=1:S, 1:dim_Λhat2_rows, 1:size(R[1],1)])

    @variable(model, Phat1_Φ[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Phat1_Π[s=1:S, 1:num_nodes-1, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Phat2_Φ[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Phat2_Π[s=1:S, 1:num_nodes-1, 1:num_arcs+1], lower_bound=0.0)

    # =========================================================================
    # OBJECTIVE FUNCTION
    # =========================================================================
    diag_x_E = Diagonal(x) * E  # diag(x)E
    diag_λ_ψ = Diagonal(λ*ones(num_arcs)-v.*ψ0)
    # s에 대해 summing이 필요하다면 sum over s 추가
    # matrix inner product: sum(M .* N)
    obj_term1 = [-ϕU * sum(Uhat1[s, :, :] .* diag_x_E) for s=1:S]
    obj_term2 = [-ϕU * sum(Uhat3[s, :, :] .* (E-diag_x_E)) for s=1:S]
    obj_term3 = [(d0')* βhat1_1[s,:] for s=1:S] #이거만 maximize하면 dual infeasible
    obj_term_ub_hat = [-ϕU * sum(Phat1_Φ[s,:,:]) - ϕU * sum(Phat1_Π[s,:,:]) for s=1:S]
    obj_term_lb_hat = [-ϕU * sum(Phat2_Φ[s,:,:]) - ϕU * sum(Phat2_Π[s,:,:]) for s=1:S]
    @objective(model, Max, sum(obj_term1) + sum(obj_term2) + sum(obj_term3) 
    + sum(obj_term_ub_hat) + sum(obj_term_lb_hat))

    intercept = @expression(model, intercept, sum(obj_term3) + sum(obj_term_ub_hat) + sum(obj_term_lb_hat))

    # =========================================================================
    # CONSTRAINTS
    # =========================================================================

    # --- Semi-definite cone constraints ---
    @constraint(model, [s=1:S], Mhat[s,:,:] in PSDCone())
    # --- Second order cone constraints ---
    @constraint(model, [s=1:S, i=1:dim_Λhat1_rows], Γhat1[s, i, :] in SecondOrderCone())
    @constraint(model, [s=1:S, i=1:dim_Λhat2_rows], Γhat2[s, i, :] in SecondOrderCone())

    # Scalar constraints
    @constraint(model, cons_dual_constant[s=1:S], Mhat[s, num_arcs+1, num_arcs+1] <= 1/true_S)
    @constraint(model, [s=1:S], tr(Mhat[s, 1:num_arcs, 1:num_arcs]) - Mhat[s,end,end]*(epsilon^2) <= 0)
    # --- Matrix Constraints ---
    for s in 1:S
        D_s = diagm(xi_bar[s])
        # --- From Φhat ---
        Mhat_11 = Mhat[s, 1:num_arcs, 1:num_arcs]
        Mhat_12 = Mhat[s, 1:num_arcs, end]
        Mhat_22 = Mhat[s, end, end]
        Adj_L_Mhat_11 = -D_s*Mhat_11
        Adj_L_Mhat_12 = -Mhat_12*adjoint(xi_bar[s])

        Adj_0_Mhat_12 = -D_s * Mhat_12
        Adj_0_Mhat_22 = -xi_bar[s] * Mhat_22

        ## Φhat_L constraint
        lhs_L = Adj_L_Mhat_11+Adj_L_Mhat_12 + Uhat2[s,:,1:num_arcs] - Uhat3[s,:,1:num_arcs]
        -I_0*Zhat1_1[s,:,:] - Zhat1_3[s,:,:] + Zhat2[s,:,:] + Phat1_Φ[s,:,1:num_arcs] - Phat2_Φ[s,:,1:num_arcs]

        for i in 1:num_arcs, j in 1:num_arcs
            if network.arc_adjacency[i,j]
                @constraint(model, lhs_L[i,j] == 0)
            end
        end
        ## Φhat_0 constraint
        @constraint(model, Adj_0_Mhat_12+Adj_0_Mhat_22 + Uhat2[s,:,end] - Uhat3[s,:,end] + I_0*βhat1_1[s,:] + βhat1_3[s,:] - βhat2[s,:] + Phat1_Φ[s,:,end] - Phat2_Φ[s,:,end] .== 0)
        
        # --- From Ψhat
        Adj_L_Mhat_11 = v*D_s*Mhat_11 #if v=vector -> diagm(v)
        Adj_L_Mhat_12 = v*Mhat_12*adjoint(xi_bar[s])

        Adj_0_Mhat_12 = v*D_s * Mhat_12
        Adj_0_Mhat_22 = xi_bar[s] * Mhat_22 * v #if v=vector -> diagm(v)
        ## Ψhat_L constraint
        lhs_L = Adj_L_Mhat_11+Adj_L_Mhat_12 -Uhat1[s,:,1:num_arcs] - Uhat2[s,:,1:num_arcs] + Uhat3[s,:,1:num_arcs]
        for i in 1:num_arcs, j in 1:num_arcs
            if network.arc_adjacency[i,j]
                @constraint(model, lhs_L[i,j] <= 0)
            end
        end
        ## Ψhat_0 constraint
        @constraint(model, Adj_0_Mhat_12+Adj_0_Mhat_22 - Uhat1[s,:,end] - Uhat2[s,:,end] + Uhat3[s,:,end] .<= 0.0)
    end
    # --- From μhat ---
    @constraint(model, coupling_cons[s=1:S, k=1:num_arcs], βhat2[s,k] <= α[k])
    # --- From Πhat ---
    # --- Πhat_L constraint
    for i in 1:(num_nodes-1), j in 1:num_arcs
        if network.node_arc_incidence[i,j]
            @constraint(model, [s=1:S], (-N*Zhat1_1[s,:,:])[i,j]-Zhat1_2[s,i,j] + Phat1_Π[s,i,j] - Phat2_Π[s,i,j] == 0.0)
        end
    end

    # --- Πhat_0 constraint
    @constraint(model, [s=1:S], N*βhat1_1[s,:]+ βhat1_2[s,:] + Phat1_Π[s,:,end] - Phat2_Π[s,:,end] .== 0)
    # --- From Λhat1 ---
    @constraint(model, [s=1:S], Zhat1[s,:,:]*R[s]' + βhat1[s,:]*r_dict[s]' + Γhat1[s,:,:] .== 0.0)
    # --- From Λhat2 ---
    @constraint(model, [s=1:S], Zhat2[s,:,:]*R[s]' + βhat2[s,:]*r_dict[s]' + Γhat2[s,:,:] .== 0.0)

    vars = Dict(
        :Mhat => Mhat,
        :Zhat1 => Zhat1,
        :Zhat2 => Zhat2,
        :Γhat1 => Γhat1,
        :Γhat2 => Γhat2,
        :Phat1_Φ => Phat1_Φ,
        :Phat1_Π => Phat1_Π,
        :Phat2_Φ => Phat2_Φ,
        :Phat2_Π => Phat2_Π,
        :Uhat1 => Uhat1,
        :Uhat3 => Uhat3,
        :βhat1_1 => βhat1_1,
        :intercept => intercept,
    )


    return model, vars
end

function build_isp_follower(network, S, ϕU, λU, γ, w, v, uncertainty_set, optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, true_S)
    # Extract network dimensions
    num_nodes = length(network.nodes)
    num_arcs = length(network.arcs)-1 #dummy arc 제외
    num_interdictable = sum(network.interdictable_arcs)

    # Node-arc incidence matrix (excluding source row)
    N = network.N
    N_y = N[:, 1:num_arcs]  # Regular arcs: (|V|-1) × |A|
    N_ts = N[:, end]         # Dummy arc: (|V|-1) × 1
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    # Dummy arc index (t,s)
    dummy_arc_idx = findfirst(arc -> arc == ("t", "s"), network.arcs)
    E = ones(num_arcs, num_arcs+1) # num_arcs × num_arcs+1 matrix of ones
    I_0 = [Matrix{Float64}(I, num_arcs, num_arcs) zeros(num_arcs)]
    # Create model
    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))
    d0 = zeros(num_arcs + 1)
    d0[end] = 1.0
    println("Building dualized outer subproblem...")
    println("  Nodes: $num_nodes, Arcs: $num_arcs, Scenarios: $S")
    println("  Interdictable arcs: $num_interdictable")
    println("  Dummy arc index: $dummy_arc_idx")
    println("  Parameters: ϕU = $ϕU, λU = $λU, γ = $γ, w = $w, v = $v")

    # =========================================================================
    # DECISION VARIABLES
    # =========================================================================
    λ, x, h, ψ0 = λ_sol, x_sol, h_sol, ψ0_sol
    # --- Scalar variables ---
    α = α_sol
    # --- Vector variables ---
    dim_Λtilde1_rows = num_arcs+1 + (num_nodes - 1) + num_arcs + num_nodes-1 + num_arcs + num_arcs ## equal to dim_Λtilde1_rows in full model
    dim_Λtilde2_rows = num_arcs ## equal to dim_Λtilde2_rows in full model
    @variable(model, βtilde1[s=1:S,1:dim_Λtilde1_rows]>=0)
    @variable(model, βtilde2[s=1:S,1:dim_Λtilde2_rows]>=0)

    #βtilde1 block 분리
    block2_start = num_arcs+2
    block3_start = block2_start + num_nodes-1
    block4_start = block3_start + num_arcs
    block5_start = block4_start + num_nodes-1
    block6_start = block5_start + num_arcs
    βtilde1_1 = βtilde1[:,1:num_arcs+1] 
    βtilde1_2 = βtilde1[:,block2_start:block3_start-1]
    βtilde1_3 = βtilde1[:,block3_start:block4_start-1]
    βtilde1_4 = βtilde1[:,block4_start:block5_start-1]
    βtilde1_5 = βtilde1[:,block5_start:block6_start-1]
    βtilde1_6 = βtilde1[:,block6_start:end]
    block2_start, block3_start, block4_start, block5_start, block6_start= -1, -1, -1, -1, -1 ## 이후에 다시 쓰이는데 초기화
    # check if the blocks are correct (block들 column dimension 합이 dim_Λtilde1_rows와 같은지 확인)
    @assert sum([size(βtilde1_1,2), size(βtilde1_2,2), size(βtilde1_3,2), size(βtilde1_4,2), size(βtilde1_5,2), size(βtilde1_6,2)]) == dim_Λtilde1_rows
    # --- Matrix variables ---
    @variable(model, Mtilde[s=1:S,1:num_arcs+1,1:num_arcs+1])
    @variable(model, Utilde1[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    @variable(model, Utilde2[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    @variable(model, Utilde3[s=1:S,1:num_arcs, 1:num_arcs+1]>=0)
    dim_R_cols = size(R[1],2)
    @variable(model, Ztilde1[s=1:S,1:dim_Λtilde1_rows,1:dim_R_cols])
    @variable(model, Ztilde2[s=1:S,1:dim_Λtilde2_rows,1:dim_R_cols])

    # Zhat1도 3개 블록으로 분리, sdp_build_full_model.jl 참고
    # Ztilde1도 6개 블록으로 분리, sdp_build_full_model.jl 참고
    block2_start = num_arcs+2
    block3_start = block2_start + num_nodes-1
    block4_start = block3_start + num_arcs
    block5_start = block4_start + num_nodes-1
    block6_start = block5_start + num_arcs
    Ztilde1_1 = Ztilde1[:,1:num_arcs+1,:]
    Ztilde1_2 = Ztilde1[:,block2_start:block3_start-1,:]
    Ztilde1_3 = Ztilde1[:,block3_start:block4_start-1,:]
    Ztilde1_4 = Ztilde1[:,block4_start:block5_start-1,:]
    Ztilde1_5 = Ztilde1[:,block5_start:block6_start-1,:]
    Ztilde1_6 = Ztilde1[:,block6_start:end,:]
    block2_start, block3_start, block4_start, block5_start, block6_start= -1, -1, -1, -1, -1 ## 이후에 다시 쓰이는데 초기화
    # check if the blocks are correct (block들 column dimension 합이 dim_Λtilde1_rows와 같은지 확인)
    @assert sum([size(Ztilde1_1,2), size(Ztilde1_2,2), size(Ztilde1_3,2), size(Ztilde1_4,2), size(Ztilde1_5,2), size(Ztilde1_6,2)]) == dim_Λtilde1_rows
    @variable(model, Γtilde1[s=1:S, 1:dim_Λtilde1_rows, 1:size(R[1],1)])
    @variable(model, Γtilde2[s=1:S, 1:dim_Λtilde2_rows, 1:size(R[1],1)])

    @variable(model, Ptilde1_Φ[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Ptilde1_Π[s=1:S, 1:num_nodes-1, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Ptilde2_Φ[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Ptilde2_Π[s=1:S, 1:num_nodes-1, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Ptilde1_Y[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Ptilde2_Y[s=1:S, 1:num_arcs, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Ptilde1_Yts[s=1:S, 1:num_arcs+1], lower_bound=0.0)
    @variable(model, Ptilde2_Yts[s=1:S, 1:num_arcs+1], lower_bound=0.0)

    # =========================================================================
    # OBJECTIVE FUNCTION
    # =========================================================================
    diag_x_E = Diagonal(x) * E  # diag(x)E
    diag_λ_ψ = Diagonal(λ*ones(num_arcs)-v.*ψ0)
    # s에 대해 summing이 필요하다면 sum over s 추가
    # matrix inner product: sum(M .* N)
    obj_term1 = [-ϕU * sum(Utilde1[s, :, :] .* diag_x_E) for s=1:S]
    obj_term2 = [-ϕU * sum(Utilde3[s, :, :] .* (E-diag_x_E)) for s=1:S]
    obj_term4 = [sum(Ztilde1_3[s, :, :] .* (diag_λ_ψ * diagm(xi_bar[s]))) for s=1:S]
    obj_term5 = [(λ*d0')* βtilde1_1[s,:] for s=1:S] #이거만 maximize하면 dual infeasible
    obj_term6 = [-(h+diag_λ_ψ*xi_bar[s])'* βtilde1_3[s,:] for s=1:S]
    obj_term_ub_tilde = [-ϕU * sum(Ptilde1_Φ[s,:,:]) - ϕU * sum(Ptilde1_Π[s,:,:]) - ϕU * sum(Ptilde1_Y[s,:,:]) - ϕU * sum(Ptilde1_Yts[s,:]) for s=1:S]
    obj_term_lb_tilde = [-ϕU * sum(Ptilde2_Φ[s,:,:]) - ϕU * sum(Ptilde2_Π[s,:,:]) - ϕU * sum(Ptilde2_Y[s,:,:]) - ϕU * sum(Ptilde2_Yts[s,:]) for s=1:S]
    @objective(model, Max, sum(obj_term1) + sum(obj_term2) + sum(obj_term4) + sum(obj_term5) + sum(obj_term6)
    + sum(obj_term_ub_tilde) + sum(obj_term_lb_tilde))

    intercept = @expression(model, intercept, sum(obj_term_ub_tilde) + sum(obj_term_lb_tilde))
    # =========================================================================
    # CONSTRAINTS
    # =========================================================================
    # --- Semi-definite cone constraints ---
    @constraint(model, [s=1:S], Mtilde[s,:,:] in PSDCone())
    # --- Second order cone constraints ---
    @constraint(model, [s=1:S, i=1:dim_Λtilde1_rows], Γtilde1[s, i, :] in SecondOrderCone())
    @constraint(model, [s=1:S, i=1:dim_Λtilde2_rows], Γtilde2[s, i, :] in SecondOrderCone())

    # Scalar constraints
    # @constraint(model, cons_dual_constant[s=1:S], Mtilde[s, num_arcs+1, num_arcs+1] == 1/true_S)
    @constraint(model, cons_dual_constant_pos[s=1:S], Mtilde[s, num_arcs+1, num_arcs+1] <= 1/true_S)
    @constraint(model, cons_dual_constant_neg[s=1:S], -Mtilde[s, num_arcs+1, num_arcs+1] <= -1/true_S)
    @constraint(model, [s=1:S], tr(Mtilde[s, 1:num_arcs, 1:num_arcs]) - Mtilde[s,end,end]*(epsilon^2) <= 0)
    # --- Matrix Constraints ---
    for s in 1:S
        D_s = diagm(xi_bar[s])
        Mtilde_11 = Mtilde[s, 1:num_arcs, 1:num_arcs]
        Mtilde_12 = Mtilde[s, 1:num_arcs, end]
        Mtilde_22 = Mtilde[s, end, end]
        # --- From Φtilde ---
        Adj_L_Mtilde_11 = -D_s*Mtilde_11
        Adj_L_Mtilde_12 = -Mtilde_12*adjoint(xi_bar[s])

        Adj_0_Mtilde_12 = -D_s * Mtilde_12
        Adj_0_Mtilde_22 = -xi_bar[s] * Mtilde_22
        # --- Φtilde_L constraint
        lhs_L = Adj_L_Mtilde_11+Adj_L_Mtilde_12 + Utilde2[s,:,1:num_arcs] - Utilde3[s,:,1:num_arcs]
        -I_0*Ztilde1_1[s,:,:] - Ztilde1_5[s,:,:] + Ztilde2[s,:,:] + Ptilde1_Φ[s,:,1:num_arcs] - Ptilde2_Φ[s,:,1:num_arcs]
        for i in 1:num_arcs, j in 1:num_arcs
            if network.arc_adjacency[i,j]
                @constraint(model, lhs_L[i,j] == 0)
            end
        end
        # --- Φtilde_0 constraint
        @constraint(model, Adj_0_Mtilde_12+Adj_0_Mtilde_22 + Utilde2[s,:,end] - Utilde3[s,:,end] + I_0*βtilde1_1[s,:] + βtilde1_5[s,:] - βtilde2[s,:] + Ptilde1_Φ[s,:,end] - Ptilde2_Φ[s,:,end] .== 0)
        
        # --- From Ψtilde ---
        Adj_L_Mtilde_11 = v*D_s*Mtilde_11
        Adj_L_Mtilde_12 = v*(Mtilde_12*adjoint(xi_bar[s]))

        Adj_0_Mtilde_12 = v*D_s * Mtilde_12
        Adj_0_Mtilde_22 = v*xi_bar[s] * Mtilde_22
        # --- Ψtilde_L constraint
        lhs_L = Adj_L_Mtilde_11+Adj_L_Mtilde_12 - Utilde1[s,:,1:num_arcs] - Utilde2[s,:,1:num_arcs] + Utilde3[s,:,1:num_arcs]
        for i in 1:num_arcs, j in 1:num_arcs
            if network.arc_adjacency[i,j]
                @constraint(model, lhs_L[i,j] <= 0.0)
            end
        end
        # --- Ψtilde_0 constraint
        @constraint(model, Adj_0_Mtilde_12+Adj_0_Mtilde_22 - Utilde1[s,:,end] - Utilde2[s,:,end] + Utilde3[s,:,end] .<= 0.0)
        # --- From Ytilde_ts ---
        Adj_L_Mtilde_12 = Mtilde_12

        Adj_0_Mtilde_22 = Mtilde_22
        # --- Ytilde_ts_L constraint
        @constraint(model, adjoint(Adj_L_Mtilde_12) + N_ts' * Ztilde1_2[s,:,:] + Ptilde1_Yts[s,1:num_arcs]' - Ptilde2_Yts[s,1:num_arcs]' .== 0)
        # --- Ytilde_ts_0 constraint
        @constraint(model, Adj_0_Mtilde_22 - N_ts' * βtilde1_2[s,:] + Ptilde1_Yts[s,end]' - Ptilde2_Yts[s,end]' .== 0)
    end
    # --- From μtilde ---
    @constraint(model, coupling_cons[s=1:S, k=1:num_arcs], βtilde2[s,k] <= α[k])
    # --- From Πtilde ---
    # --- Πtilde_L constraint
    for i in 1:(num_nodes-1), j in 1:num_arcs
        if network.node_arc_incidence[i,j]
            @constraint(model, [s=1:S], (-N*Ztilde1_1[s,:,:])[i,j]-Ztilde1_4[s,i,j] + Ptilde1_Π[s,i,j] - Ptilde2_Π[s,i,j] == 0.0)
        end
    end
    
    # --- Πtilde_0 constraint
    @constraint(model, [s=1:S], N*βtilde1_1[s,:]+ βtilde1_4[s,:] + Ptilde1_Π[s,:,end] - Ptilde2_Π[s,:,end] .== 0)
    # --- From Ytilde ---
    # --- From Ytilde_L constraint
    for i in 1:num_arcs, j in 1:num_arcs
        if network.arc_adjacency[i,j]
            @constraint(model, [s=1:S], (N_y' * Ztilde1_2[s,:,:])[i,j]+Ztilde1_3[s,i,j]-Ztilde1_6[s,i,j] + Ptilde1_Y[s,i,j] - Ptilde2_Y[s,i,j] == 0.0)
        end
    end
    
    # --- Ytilde_0 constraint
    @constraint(model, [s=1:S], -N_y' * βtilde1_2[s,:]-βtilde1_3[s,:]+βtilde1_6[s,:]+ Ptilde1_Y[s,:,end] - Ptilde2_Y[s,:,end] .== 0)
    # --- From Λtilde1 ---
    @constraint(model, [s=1:S], Ztilde1[s,:,:]*R[s]' + βtilde1[s,:]*r_dict[s]' + Γtilde1[s,:,:] .== 0.0)
    # --- From Λtilde2 ---
    @constraint(model, [s=1:S], Ztilde2[s,:,:]*R[s]' + βtilde2[s,:]*r_dict[s]' + Γtilde2[s,:,:] .== 0.0)

    vars = Dict(
        :Mtilde => Mtilde,
        :Ztilde1 => Ztilde1,
        :Ztilde2 => Ztilde2,
        :Γtilde1 => Γtilde1,
        :Γtilde2 => Γtilde2,
        :Ptilde1_Φ => Ptilde1_Φ,
        :Ptilde1_Π => Ptilde1_Π,
        :Ptilde2_Φ => Ptilde2_Φ,
        :Ptilde2_Π => Ptilde2_Π,
        :Ptilde1_Y => Ptilde1_Y,
        :Ptilde1_Yts => Ptilde1_Yts,
        :Ptilde2_Y => Ptilde2_Y,
        :Ptilde2_Yts => Ptilde2_Yts,
        :Utilde1 => Utilde1,
        :Utilde3 => Utilde3,
        :βtilde1_1 => βtilde1_1,
        :βtilde1_3 => βtilde1_3,
        :Ztilde1_3 => Ztilde1_3,
        :intercept => intercept,
    )

    return model, vars
end