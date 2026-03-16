"""
Adding trust-region methods to stabilize benders convergence.

## 구조
- `tr_nested_benders_optimize!`: outer loop (OMP → IMP → cut 추가). `outer_tr`, `inner_tr` 키워드로
  outer/inner trust region을 독립적으로 on/off 가능.
- `tr_imp_optimize!`: inner loop (IMP → ISP leader/follower → cut 추가). `inner_tr` 키워드로 제어.

## Outer TR (binary L1-norm on x)
- B_bin을 단계적으로 확장 (0.05 → 0.5 → 1.0) × |interdictable arcs|
- 각 단계에서 local optimal 도달 → reverse region 추가 → 확장 → 최종적으로 global optimal
- Serious step test로 stability center 이동 여부 결정

## Inner TR (continuous L∞-norm on α)
- α에 box constraint: -B_conti ≤ α - α̂ ≤ B_conti (element-wise)
- B_conti를 serious/null step에 따라 확장/축소 (proximal bundle method 스타일)

## Known Issue: inner TR + lower_bound tracking의 조기 수렴
inner TR이 활성화된 경우, `lower_bound = max(lower_bound, subprob_obj)`가 부정확할 수 있다.

예시:
  iter 3: α₃에서 subprob_obj = 13.420 → lower_bound = 13.420
  iter 4: serious step으로 center 이동, 새 α₄에서 model_estimate = 13.420, subprob_obj = 13.418
          lower_bound = max(13.420, 13.418) = 13.420 (이전 값 유지)
          gap = 13.420 - 13.420 ≈ 0 → 수렴 판정!

문제: lower_bound=13.420은 α₃에서 구한 값이지만, 반환하는 α₄의 실제 값은 13.418.
TR로 center가 이동하면 이전 α에서의 subprob_obj가 현재 α의 quality를 반영하지 않음.

영향: inner loop이 약간 조기 수렴하여 loose한 cut을 생성할 수 있으나,
outer loop의 추가 iteration이 이를 보정하므로 전체 알고리즘의 correctness는 유지됨.

대응: result[:obj_val]은 objective_value(imp_model) (= model_estimate, upper bound)이 아닌
subprob_obj (현재 α에서의 실제 값)를 반환하여, outer loop의 upper bound tracking이 정확하도록 함.

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
includet("build_primal_isp.jl")


using .NetworkGenerator
"""
Build the Inner Master and Inner Subproblem
"""


function update_outer_trust_region_constraints!(
    model::Model, 
    vars::Dict, 
    centers::Dict,
    B_bin::Real,
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


function update_inner_trust_region_constraints!(
    model::Model, 
    vars::Dict, 
    centers::Dict,
    B_conti::Float64,
    old_cons::Dict,
    network
)
    α = vars[:α]
    αhat = centers[:α]

    # Remove old constraints if they exist
    if old_cons[:continuous] !== nothing
        delete(model, old_cons[:continuous][1])
        delete(model, old_cons[:continuous][2])
    end
    
    tr_conti_expr = @expression(model,
        α - αhat
    )
    new_tr_conti_left = @constraint(model, -B_conti .<= tr_conti_expr)
    new_tr_conti_right = @constraint(model, tr_conti_expr .<= B_conti)
    set_name.(new_tr_conti_left, "TR_conti_left")
    set_name.(new_tr_conti_right, "TR_conti_right")

    new_cons = Dict(
        :continuous => (new_tr_conti_left, new_tr_conti_right)
    )
    
    return new_cons
end


function build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_optimizer=nothing)
    num_arcs = length(network.arcs) - 1
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    S = length(xi_bar)
    flow_upper = sum(sum(xi_bar[s] for s in 1:S))
    model = Model(optimizer_with_attributes(mip_optimizer, MOI.Silent() => true))
    @variable(model, t_1_l[s=1:S], upper_bound= flow_upper)
    @variable(model, t_1_f[s=1:S], upper_bound= flow_upper)
    @variable(model, α[k=1:num_arcs] >= 0)
    @constraint(model, sum(α) <= w*(1/S)) # full model에선 자연스럽게 inequality가 equality가 되지만 decomposed된 imp에선 그런다는 보장이 없으므로 명시적으로 equality 유지
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
    πU = get(isp_data, :πU, ϕU)
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

    obj_term_ub_hat = [-ϕU * sum(Phat1_Φ[s,:,:]) - πU * sum(Phat1_Π[s,:,:]) for s=1:S]
    obj_term_lb_hat = [-ϕU * sum(Phat2_Φ[s,:,:]) - πU * sum(Phat2_Π[s,:,:]) for s=1:S]
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
    πU, yU, ytsU = get(isp_data, :πU, ϕU), get(isp_data, :yU, ϕU), get(isp_data, :ytsU, ϕU)
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

    obj_term_ub_tilde = [-ϕU * sum(Ptilde1_Φ[s,:,:]) - πU * sum(Ptilde1_Π[s,:,:]) - yU * sum(Ptilde1_Y[s,:,:]) - ytsU * sum(Ptilde1_Yts[s,:]) for s=1:S]
    obj_term_lb_tilde = [-ϕU * sum(Ptilde2_Φ[s,:,:]) - πU * sum(Ptilde2_Π[s,:,:]) - yU * sum(Ptilde2_Y[s,:,:]) - ytsU * sum(Ptilde2_Yts[s,:]) for s=1:S]
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
        # Mosek STALL 시 shadow price 부정확 → intercept 역산 보정 (see memory/intercept_fix.md)
        if abs(dual_obj - objective_value(model)) > 1e-4
            @warn "ISP follower duality gap $(round(abs(dual_obj - objective_value(model)), digits=6)) → intercept 역산 보정 (Mosek: $(MOI.get(model, MOI.RawStatusString())))"
            intercept = objective_value(model) - α_sol'*subgradient
            dual_obj = objective_value(model)
        end
        cut_coeff = Dict(:μtilde=>μtilde, :intercept=>intercept, :obj_val=>dual_obj)
        return (:OptimalityCut, cut_coeff)
    else
        t_status = termination_status(model)
        @infiltrate
        error("Subproblem is not optimal")
    end
end

function tr_imp_optimize!(imp_model::Model, imp_vars::Dict, isp_leader_instances::Dict, isp_follower_instances::Dict; isp_data=nothing, λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing, outer_iter=nothing, imp_cuts=nothing, inner_tr=true, tol=1e-4)
    st = MOI.get(imp_model, MOI.TerminationStatus())
    iter = 0
    uncertainty_set = isp_data[:uncertainty_set]
    past_obj = []
    past_subprob_obj = []
    past_major_subprob_obj = []
    past_lower_bound = []
    past_upper_bound = []
    major_iter = []
    lower_bound = -Inf ## inner master problem은 Maximization이니까 feasible solution은 lower bound를 제공.
    result = Dict()
    result[:cuts] = Dict()
    if inner_tr
        B_conti_max = isp_data[:w]/isp_data[:S]
        B_conti = B_conti_max * 0.01 # 초기값 어떻게?
        counter = 0
        β_relative = 1e-4 # serious improvement threshold
        ρ = 0.0
        centers = Dict(:α=>value.(imp_vars[:α]))
        tr_constraints = Dict(:continuous=>nothing)
    end
    ##
    ## 여기서 imp 초기화해야함.
    if outer_iter>1
        for (cut_name, cut) in imp_cuts[:old_cuts]
            delete(imp_model, cut)
        end
        if inner_tr && imp_cuts[:old_tr_constraints] !== nothing
            for tr_cons in imp_cuts[:old_tr_constraints]
                valid_cons = filter(c -> is_valid(imp_model, c), tr_cons)
                delete.(imp_model, valid_cons)
            end
        end
    end
    ##
    while (st == MOI.DUAL_INFEASIBLE || st == MOI.OPTIMAL)
        iter += 1
        @info "    [Inner] Iteration $iter"
        optimize!(imp_model)
        st = MOI.get(imp_model, MOI.TerminationStatus())
        α_sol = value.(imp_vars[:α])
        model_estimate = sum(value.(imp_vars[:t_1_l])) + sum(value.(imp_vars[:t_1_f]))
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
        gap = abs(model_estimate - lower_bound) / max(abs(model_estimate), 1e-10)
        if gap <= tol
            @info "Termination condition met"
            println("model_estimate: ", model_estimate, ", subprob_obj: ", subprob_obj)
            result[:past_obj] = past_obj
            result[:past_subprob_obj] = past_subprob_obj
            result[:α_sol] = α_sol
            result[:obj_val] = subprob_obj  # subproblem의 정확한 값 (IMP objective는 upper bound이므로 gap만큼 차이남)
            result[:past_lower_bound] = past_lower_bound
            result[:iter] = iter
            if inner_tr && tr_constraints[:continuous] !== nothing
                result[:tr_constraints] = tr_constraints[:continuous]
            else
                result[:tr_constraints] = nothing
            end
            # if round(result[:obj_val], digits=4) == 5.2495
            #     @infiltrate
            # end
            return (:OptimalityCut, result)
        else
            if inner_tr
                # Serious Test
                if iter==1
                    push!(past_major_subprob_obj, subprob_obj)
                end
                tr_needs_update = false # Flag for TR constraint update
                predicted_increase = model_estimate - past_major_subprob_obj[end]
                β_dynamic = max(1e-8, β_relative * predicted_increase)
                improvement = subprob_obj - past_major_subprob_obj[end]
                is_serious_step = (improvement >= β_dynamic)
                if is_serious_step
                    tr_needs_update = true
                    distance = norm(α_sol - centers[:α], Inf)  # l_infinity norm
                    centers[:α] = α_sol
                    push!(major_iter, iter)
                    push!(past_major_subprob_obj, subprob_obj)
                    if (improvement >= 0.5*β_dynamic) && (distance >= B_conti - 1e-6)
                        ## 매우 좋은 성능이고 trust region의 boundary에 닿았음
                        @info "Very good improvement: Expanding B_conti"
                        B_conti = min(B_conti_max, B_conti * 2.0)
                    else
                        ## 적당히 좋은 성능 - trust region radius는 유지
                        @info "Moderate improvement: Keeping B_conti"
                        B_conti = B_conti
                    end
                    tr_constraints = update_inner_trust_region_constraints!(imp_model, imp_vars, centers, B_conti, tr_constraints, network)
                else
                    @info "Poor improvement: Reducing B_conti"
                    ρ = min(1, B_conti) * improvement / β_dynamic
                    if ρ > 3.0
                        #즉시 감소 (매우 나쁜 model)
                        B_conti = B_conti / min(ρ,4)
                        counter = 0
                        tr_needs_update = true
                    elseif (1.0 < ρ) && (counter>=3)
                        #3번 누적 후 감소
                        B_conti = B_conti / min(ρ,4)
                        counter = 0
                        tr_needs_update = true
                    elseif (1.0 < ρ) && (counter<3)
                        #유지하지만 카운트 증가
                        counter += 1
                    elseif (0.0 < ρ) && (ρ <= 1.0)
                        #objective 감소했지만 예측보다 적게
                        counter += 1
                    else
                        #objective 증가했지만 불충분
                        B_conti = B_conti
                        counter = counter
                    end
                    if tr_needs_update
                        tr_constraints = update_inner_trust_region_constraints!(imp_model, imp_vars, centers, B_conti, tr_constraints, network)
                    end
                end
            end
            push!(past_obj, model_estimate)
            push!(past_subprob_obj, subprob_obj)
            push!(past_lower_bound, lower_bound)
            if status == false
                @warn "Subproblem is not optimal"
            end
        
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

function initialize_imp(imp_model::Model, imp_vars::Dict)
    optimize!(imp_model)
    st = MOI.get(imp_model, MOI.TerminationStatus())
    α_sol = value.(imp_vars[:α])
    return st, α_sol
end

function initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set; conic_optimizer=nothing, λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing, α_sol=nothing, πU=ϕU, yU=ϕU, ytsU=ϕU)
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    leader_instances = Dict{Int, Tuple{Model, Dict}}()
    follower_instances = Dict{Int, Tuple{Model, Dict}}()
    for s in 1:S
        U_s = Dict(:R => Dict(:1=>R[s]), :r_dict => Dict(:1=>r_dict[s]), :xi_bar => Dict(:1=>xi_bar[s]), :epsilon => epsilon)
        leader_instances[s] = build_isp_leader(network, 1, ϕU, λU, γ, w, v, U_s, conic_optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, S; πU=πU)
        follower_instances[s] = build_isp_follower(network, 1, ϕU, λU, γ, w, v, U_s, conic_optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, S; πU=πU, yU=yU, ytsU=ytsU)
        
    end
    return leader_instances, follower_instances
end

function evaluate_master_opt_cut(isp_leader_instances::Dict, isp_follower_instances::Dict, isp_data::Dict, cut_info::Dict, iter::Int; multi_cut=false)
    """
    α를 fix 시키고 outer subproblem의 값을 다시 정확하게 구하는 코드
    """
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


"""
    generate_core_points(network, γ, λU, w, v; interdictable_idx=nothing, strategy=:interior_and_arcs)

OMP의 feasible region 내부에 있는 core point들을 생성.
Magnanti-Wong cut의 objective로 사용됨.

Strategies:
- `:interior` — fractional x̄ᵢ = γ/|A_I|, λ̄ = λU/2
- `:arc_directed` — interdictable arc별 eᵢ (γ개 binary points)
- `:interior_and_arcs` — 둘 합산
"""
function generate_core_points(network, γ, λU, w, v;
    interdictable_idx=nothing, strategy=:interior_and_arcs)
    num_arcs = length(network.arcs) - 1
    if interdictable_idx === nothing
        interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    end
    num_interdictable = length(interdictable_idx)

    points = NamedTuple{(:x, :λ, :h, :ψ0), Tuple{Vector{Float64}, Float64, Vector{Float64}, Vector{Float64}}}[]

    if strategy == :interior || strategy == :interior_and_arcs
        # Interior point: fractional x
        x_bar = zeros(num_arcs)
        x_bar[interdictable_idx] .= γ / num_interdictable
        λ_bar = λU / 2
        h_bar = fill(λ_bar * w / num_arcs, num_arcs)
        # McCormick: ψ0 ∈ [max(λ - λU(1-x), 0), min(λU*x, λ)]
        ψ0_bar = [min(λU * x_bar[k], λ_bar, max(λ_bar - λU * (1 - x_bar[k]), 0.0)) for k in 1:num_arcs]
        push!(points, (x=x_bar, λ=λ_bar, h=h_bar, ψ0=ψ0_bar))
    end

    if strategy == :arc_directed || strategy == :interior_and_arcs
        # Arc-directed: single interdicted arc per core point
        for idx in interdictable_idx[1:min(γ, num_interdictable)]
            x_arc = zeros(num_arcs)
            x_arc[idx] = 1.0
            λ_arc = λU / 2
            h_arc = fill(λ_arc * w / num_arcs, num_arcs)
            ψ0_arc = [min(λU * x_arc[k], λ_arc, max(λ_arc - λU * (1 - x_arc[k]), 0.0)) for k in 1:num_arcs]
            push!(points, (x=x_arc, λ=λ_arc, h=h_arc, ψ0=ψ0_arc))
        end
    end
    @infiltrate
    return points
end


"""
    evaluate_mw_opt_cut(...)

Magnanti-Wong cut 생성: α*에서 ISP가 이미 solved 상태인 것을 전제.
z* = objective_value에 대해 optimality constraint를 추가한 뒤,
core point에서의 cut value를 최대화하여 Pareto-optimal cut을 생성.

반환: evaluate_master_opt_cut과 동일한 Dict.
"""
function evaluate_mw_opt_cut(
    isp_leader_instances, isp_follower_instances, isp_data, cut_info, iter;
    x_sol, λ_sol, h_sol, ψ0_sol,
    x_core, λ_core, h_core, ψ0_core,
    multi_cut=false)

    S = isp_data[:S]
    ϕU = isp_data[:ϕU]
    πU = isp_data[:πU]
    yU = isp_data[:yU]
    ytsU = isp_data[:ytsU]
    E = isp_data[:E]
    d0 = isp_data[:d0]
    v_param = isp_data[:v]
    α_sol = cut_info[:α_sol]
    num_arcs = length(x_sol)
    xi_bar = isp_data[:uncertainty_set][:xi_bar]

    # Pre-compute matrices
    diag_x_sol_E = Diagonal(x_sol) * E
    diag_x_core_E = Diagonal(x_core) * E
    diag_λ_ψ_sol = Diagonal(λ_sol * ones(num_arcs) - v_param .* ψ0_sol)
    diag_λ_ψ_core = Diagonal(λ_core * ones(num_arcs) - v_param .* ψ0_core)

    mw_cons = []  # track constraints for cleanup

    for s in 1:S
        xi_bar_s = xi_bar[s]

        # ===== Leader MW =====
        model_l = isp_leader_instances[s][1]
        vars_l = isp_leader_instances[s][2]
        Uhat1 = vars_l[:Uhat1]
        Uhat3 = vars_l[:Uhat3]
        Phat1_Φ = vars_l[:Phat1_Φ]
        Phat1_Π = vars_l[:Phat1_Π]
        Phat2_Φ = vars_l[:Phat2_Φ]
        Phat2_Π = vars_l[:Phat2_Π]
        βhat1_1 = vars_l[:βhat1_1]

        z_star_l = objective_value(model_l)

        # Reconstruct original objective (at x_sol) — mirrors isp_leader_optimize! lines 194-201
        orig_l_1 = -ϕU * sum(Uhat1[1, :, :] .* diag_x_sol_E)
        orig_l_2 = -ϕU * sum(Uhat3[1, :, :] .* (E - diag_x_sol_E))
        orig_l_3 = (d0') * βhat1_1[1, :]
        orig_l_ub = -ϕU * sum(Phat1_Φ[1, :, :]) - πU * sum(Phat1_Π[1, :, :])
        orig_l_lb = -ϕU * sum(Phat2_Φ[1, :, :]) - πU * sum(Phat2_Π[1, :, :])
        orig_obj_l = orig_l_1 + orig_l_2 + orig_l_3 + orig_l_ub + orig_l_lb

        # MW optimality constraint
        mw_con_l = @constraint(model_l, orig_obj_l >= z_star_l - 1e-6)
        push!(mw_cons, (model_l, mw_con_l))

        # Core objective (replace x_sol → x_core; intercept terms unchanged)
        core_l_1 = -ϕU * sum(Uhat1[1, :, :] .* diag_x_core_E)
        core_l_2 = -ϕU * sum(Uhat3[1, :, :] .* (E - diag_x_core_E))
        core_obj_l = core_l_1 + core_l_2 + orig_l_3 + orig_l_ub + orig_l_lb
        @objective(model_l, Max, core_obj_l)
        optimize!(model_l)

        st_l = termination_status(model_l)
        if !(st_l == MOI.OPTIMAL || st_l == MOI.SLOW_PROGRESS)
            @warn "MW leader solve failed for s=$s: $st_l, using original solution"
            # Cleanup and restore
            delete(model_l, mw_con_l)
            @objective(model_l, Max, orig_obj_l)
            optimize!(model_l)
            pop!(mw_cons)
            push!(mw_cons, nothing)
            continue
        end

        # ===== Follower MW =====
        model_f = isp_follower_instances[s][1]
        vars_f = isp_follower_instances[s][2]
        Utilde1 = vars_f[:Utilde1]
        Utilde3 = vars_f[:Utilde3]
        Ztilde1_3 = vars_f[:Ztilde1_3]
        Ptilde1_Φ = vars_f[:Ptilde1_Φ]
        Ptilde1_Π = vars_f[:Ptilde1_Π]
        Ptilde2_Φ = vars_f[:Ptilde2_Φ]
        Ptilde2_Π = vars_f[:Ptilde2_Π]
        Ptilde1_Y = vars_f[:Ptilde1_Y]
        Ptilde1_Yts = vars_f[:Ptilde1_Yts]
        Ptilde2_Y = vars_f[:Ptilde2_Y]
        Ptilde2_Yts = vars_f[:Ptilde2_Yts]
        βtilde1_1 = vars_f[:βtilde1_1]
        βtilde1_3 = vars_f[:βtilde1_3]

        z_star_f = objective_value(model_f)

        # Reconstruct original objective (at x_sol, λ_sol, h_sol, ψ0_sol) — mirrors isp_follower_optimize! lines 239-248
        orig_f_1 = -ϕU * sum(Utilde1[1, :, :] .* diag_x_sol_E)
        orig_f_2 = -ϕU * sum(Utilde3[1, :, :] .* (E - diag_x_sol_E))
        orig_f_4 = sum(Ztilde1_3[1, :, :] .* (diag_λ_ψ_sol * diagm(xi_bar_s)))
        orig_f_5 = (λ_sol * d0') * βtilde1_1[1, :]
        orig_f_6 = -(h_sol + diag_λ_ψ_sol * xi_bar_s)' * βtilde1_3[1, :]
        orig_f_ub = -ϕU * sum(Ptilde1_Φ[1, :, :]) - πU * sum(Ptilde1_Π[1, :, :]) - yU * sum(Ptilde1_Y[1, :, :]) - ytsU * sum(Ptilde1_Yts[1, :])
        orig_f_lb = -ϕU * sum(Ptilde2_Φ[1, :, :]) - πU * sum(Ptilde2_Π[1, :, :]) - yU * sum(Ptilde2_Y[1, :, :]) - ytsU * sum(Ptilde2_Yts[1, :])
        orig_obj_f = orig_f_1 + orig_f_2 + orig_f_4 + orig_f_5 + orig_f_6 + orig_f_ub + orig_f_lb

        # MW optimality constraint
        mw_con_f = @constraint(model_f, orig_obj_f >= z_star_f - 1e-6)
        push!(mw_cons, (model_f, mw_con_f))

        # Core objective (replace x_sol → x_core, λ_sol → λ_core, etc.; P-terms unchanged)
        core_f_1 = -ϕU * sum(Utilde1[1, :, :] .* diag_x_core_E)
        core_f_2 = -ϕU * sum(Utilde3[1, :, :] .* (E - diag_x_core_E))
        core_f_4 = sum(Ztilde1_3[1, :, :] .* (diag_λ_ψ_core * diagm(xi_bar_s)))
        core_f_5 = (λ_core * d0') * βtilde1_1[1, :]
        core_f_6 = -(h_core + diag_λ_ψ_core * xi_bar_s)' * βtilde1_3[1, :]
        core_obj_f = core_f_1 + core_f_2 + core_f_4 + core_f_5 + core_f_6 + orig_f_ub + orig_f_lb
        @objective(model_f, Max, core_obj_f)
        optimize!(model_f)

        st_f = termination_status(model_f)
        if !(st_f == MOI.OPTIMAL || st_f == MOI.SLOW_PROGRESS)
            @warn "MW follower solve failed for s=$s: $st_f"
        end
    end

    # ===== Extract coefficients (same as evaluate_master_opt_cut) =====
    Uhat1_out = cat([value.(isp_leader_instances[s][2][:Uhat1]) for s in 1:S]...; dims=1)
    Utilde1_out = cat([value.(isp_follower_instances[s][2][:Utilde1]) for s in 1:S]...; dims=1)
    Uhat3_out = cat([value.(isp_leader_instances[s][2][:Uhat3]) for s in 1:S]...; dims=1)
    Utilde3_out = cat([value.(isp_follower_instances[s][2][:Utilde3]) for s in 1:S]...; dims=1)
    Ztilde1_3_out = cat([value.(isp_follower_instances[s][2][:Ztilde1_3]) for s in 1:S]...; dims=1)
    βtilde1_1_out = cat([value.(isp_follower_instances[s][2][:βtilde1_1]) for s in 1:S]...; dims=1)
    βtilde1_3_out = cat([value.(isp_follower_instances[s][2][:βtilde1_3]) for s in 1:S]...; dims=1)

    if multi_cut
        intercept_l = [value.(isp_leader_instances[s][2][:intercept]) for s in 1:S]
        intercept_f = [value.(isp_follower_instances[s][2][:intercept]) for s in 1:S]
        intercept = sum(intercept_l) + sum(intercept_f)
    else
        intercept = sum(value.(isp_leader_instances[s][2][:intercept]) for s in 1:S) + sum(value.(isp_follower_instances[s][2][:intercept]) for s in 1:S)
        intercept_l, intercept_f = nothing, nothing
    end

    # ===== Cleanup: delete MW constraints, restore original objectives =====
    for item in mw_cons
        item === nothing && continue
        model, con = item
        delete(model, con)
    end
    # Restore original objectives and re-solve so ISPs are in correct state
    # (필수: 다음 MW core point 호출 시 objective_value가 원래 z*를 반환해야 함)
    for s in 1:S
        xi_bar_s = xi_bar[s]
        # Restore leader objective
        model_l = isp_leader_instances[s][1]
        vars_l = isp_leader_instances[s][2]
        Uhat1 = vars_l[:Uhat1]; Uhat3 = vars_l[:Uhat3]
        Phat1_Φ = vars_l[:Phat1_Φ]; Phat1_Π = vars_l[:Phat1_Π]
        Phat2_Φ = vars_l[:Phat2_Φ]; Phat2_Π = vars_l[:Phat2_Π]
        βhat1_1 = vars_l[:βhat1_1]
        @objective(model_l, Max,
            -ϕU * sum(Uhat1[1, :, :] .* diag_x_sol_E) +
            -ϕU * sum(Uhat3[1, :, :] .* (E - diag_x_sol_E)) +
            (d0') * βhat1_1[1, :] +
            -ϕU * sum(Phat1_Φ[1, :, :]) - πU * sum(Phat1_Π[1, :, :]) +
            -ϕU * sum(Phat2_Φ[1, :, :]) - πU * sum(Phat2_Π[1, :, :]))
        optimize!(model_l)
        # Restore follower objective
        model_f = isp_follower_instances[s][1]
        vars_f = isp_follower_instances[s][2]
        Utilde1 = vars_f[:Utilde1]; Utilde3 = vars_f[:Utilde3]; Ztilde1_3_v = vars_f[:Ztilde1_3]
        Ptilde1_Φ = vars_f[:Ptilde1_Φ]; Ptilde1_Π = vars_f[:Ptilde1_Π]
        Ptilde2_Φ = vars_f[:Ptilde2_Φ]; Ptilde2_Π = vars_f[:Ptilde2_Π]
        Ptilde1_Y = vars_f[:Ptilde1_Y]; Ptilde1_Yts = vars_f[:Ptilde1_Yts]
        Ptilde2_Y = vars_f[:Ptilde2_Y]; Ptilde2_Yts = vars_f[:Ptilde2_Yts]
        βtilde1_1_v = vars_f[:βtilde1_1]; βtilde1_3_v = vars_f[:βtilde1_3]
        @objective(model_f, Max,
            -ϕU * sum(Utilde1[1, :, :] .* diag_x_sol_E) +
            -ϕU * sum(Utilde3[1, :, :] .* (E - diag_x_sol_E)) +
            sum(Ztilde1_3_v[1, :, :] .* (diag_λ_ψ_sol * diagm(xi_bar_s))) +
            (λ_sol * d0') * βtilde1_1_v[1, :] +
            -(h_sol + diag_λ_ψ_sol * xi_bar_s)' * βtilde1_3_v[1, :] +
            -ϕU * sum(Ptilde1_Φ[1, :, :]) - πU * sum(Ptilde1_Π[1, :, :]) - yU * sum(Ptilde1_Y[1, :, :]) - ytsU * sum(Ptilde1_Yts[1, :]) +
            -ϕU * sum(Ptilde2_Φ[1, :, :]) - πU * sum(Ptilde2_Π[1, :, :]) - yU * sum(Ptilde2_Y[1, :, :]) - ytsU * sum(Ptilde2_Yts[1, :]))
        optimize!(model_f)
    end

    return Dict(:Uhat1=>Uhat1_out, :Utilde1=>Utilde1_out, :Uhat3=>Uhat3_out, :Utilde3=>Utilde3_out,
        :Ztilde1_3=>Ztilde1_3_out, :βtilde1_1=>βtilde1_1_out, :βtilde1_3=>βtilde1_3_out,
        :intercept=>intercept, :intercept_l=>intercept_l, :intercept_f=>intercept_f)
end


function tr_nested_benders_optimize!(omp_model::Model, omp_vars::Dict, network, ϕU, λU, γ, w, uncertainty_set; mip_optimizer=nothing, conic_optimizer=nothing, multi_cut=false, outer_tr=true, inner_tr=true, max_outer_iter=1000, isp_mode=:dual, tol=1e-4, πU=ϕU, yU=ϕU, ytsU=ϕU, strengthen_cuts=false)
    ### -------- Trust Region 초기화 --------
    if outer_tr
        num_interdictable = sum(network.interdictable_arcs)
        max_dist = min(Int(2γ), num_interdictable) # effective diameter: 2γ (binary vectors with sum ≤ γ)
        B_bin_sequence = unique([1, ceil(Int, max_dist/4), ceil(Int, max_dist/2), max_dist])
        B_bin_stage = 1
        B_bin = B_bin_sequence[B_bin_stage]
        B_con = nothing # 나중에 생각
        ## Stability Centers
        # Stability centers (will be initialized after first solve)
        centers = Dict{Symbol, Any}(
            :x => nothing,
            :h => nothing,
            :λ => nothing,
            :ψ0 => nothing # 근데 이건 굳이 해야하나? x*lambda인데
        )
        ## Serious Step Parameters
        β_relative = 1e-4 # serious improvement threshold
        tr_constraints = Dict{Symbol, Any}(
            :binary => nothing,
            :continuous => nothing
        )
    end
    upper_bound = Inf # Will be updated after first subproblem solve
    ### --------Begin Outer Master problemInitialization--------
    st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars)
    x, h, λ, ψ0 = omp_vars[:x], omp_vars[:h], omp_vars[:λ], omp_vars[:ψ0]
     # Initialize stability centers with first solution
     if outer_tr
         centers[:x] = value.(x)
         centers[:h] = value.(h)
         centers[:λ] = value.(λ)
         centers[:ψ0] = value.(ψ0)
     end
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
    imp_cuts = Dict{Symbol, Any}(:old_tr_constraints => nothing)
    result = Dict()
    result[:cuts] = Dict()
    result[:tr_info] = Dict()
    result[:inner_iter] = []
    # Debug logging arrays
    result[:debug_α] = []
    result[:debug_intercept_l] = []
    result[:debug_intercept_f] = []
    result[:debug_coeff_norms] = []
    upper_bound = Inf  # global UB (never reset)
    local_upper_bound = Inf  # local UB (reset per stage)
    lower_bound = -Inf
    ### --------Begin Inner Master, Subproblem Initialization--------
    imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_optimizer=mip_optimizer)
    st, α_sol = initialize_imp(imp_model, imp_vars)
    # Dual ISP instances (used for :dual and :hybrid modes)
    leader_instances, follower_instances = nothing, nothing
    if isp_mode != :full_primal
        leader_instances, follower_instances = initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set; conic_optimizer=conic_optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol, πU=πU, yU=yU, ytsU=ytsU)
    end
    # Primal ISP instances (used for :hybrid and :full_primal modes)
    primal_leader_instances, primal_follower_instances = nothing, nothing
    if isp_mode != :dual
        primal_leader_instances, primal_follower_instances = initialize_primal_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set; conic_optimizer=conic_optimizer, x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)
    end
    isp_data = Dict(:E => E, :network => network, :ϕU => ϕU, :πU => πU, :yU => yU, :ytsU => ytsU, :λU => λU, :γ => γ, :w => w, :v => v, :uncertainty_set => uncertainty_set, :d0 => d0, :S=>S)
    gap = Inf
    ### --------End Initialization--------
    time_start = time()    
    while (st == MOI.DUAL_INFEASIBLE || st == MOI.OPTIMAL)
        iter += 1
        if iter > max_outer_iter
            @warn "Maximum outer iterations ($max_outer_iter) reached. Gap = $gap"
            break
        end
        if outer_tr
            @info "[Outer-$isp_mode] Iteration $iter (B_bin=$B_bin, Stage=$(B_bin_stage)/$(length(B_bin_sequence)))"
        else
            @info "[Outer-$isp_mode] Iteration $iter"
        end
        # Outer Master Problem 풀기
        optimize!(omp_model)
        st = MOI.get(omp_model, MOI.TerminationStatus())
        if st == MOI.INFEASIBLE
            ## Reverse Region들 때문에 infeasible 난걸수도. 그러면 이건 에러가 아니라, 더이상 탐색할 영역이 없음을 의미.
            @info " Outer Master Problem infeasible (Converged): Due to Reverse Regions --- No search space left"
            gap = 0.0
        else
            x_sol = round.(value.(omp_vars[:x]))  # binary rounding for numerical stability
            h_sol, λ_sol, ψ0_sol = value.(omp_vars[:h]), value(omp_vars[:λ]), value.(omp_vars[:ψ0])
            model_estimate = value(t_0)
            lower_bound = max(lower_bound, model_estimate)
            # Update primal ISP parameters (x,h,λ,ψ0 are in constraint RHS → set_normalized_rhs)
            if isp_mode != :dual
                update_primal_isp_parameters!(primal_leader_instances, primal_follower_instances;
                    x_sol=x_sol, h_sol=h_sol, λ_sol=λ_sol, ψ0_sol=ψ0_sol, isp_data=isp_data)
            end
            # Outer Subproblem 풀기
            if isp_mode == :dual
                status, cut_info = tr_imp_optimize!(imp_model, imp_vars, leader_instances, follower_instances; isp_data=isp_data, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, outer_iter=iter, imp_cuts=imp_cuts, inner_tr=inner_tr, tol=tol)
            else
                status, cut_info = tr_imp_optimize_hybrid!(imp_model, imp_vars, primal_leader_instances, primal_follower_instances; isp_data=isp_data, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, outer_iter=iter, imp_cuts=imp_cuts, inner_tr=inner_tr, tol=tol)
            end
            if status != :OptimalityCut
                @warn "Outer Subproblem not optimal"
                @infiltrate
            end
            push!(result[:inner_iter], cut_info[:iter])
            imp_cuts[:old_cuts] = cut_info[:cuts] ## 다음 iteration에서 지우기 위해 여기에 저장함
            if inner_tr && cut_info[:tr_constraints] !== nothing
                imp_cuts[:old_tr_constraints] = cut_info[:tr_constraints]
            end
            subprob_obj = cut_info[:obj_val]
            upper_bound = min(upper_bound, subprob_obj)
            local_upper_bound = min(local_upper_bound, subprob_obj)

            # Measure of progress
            gap = abs(local_upper_bound - lower_bound) / max(abs(local_upper_bound), 1e-10)
            if outer_tr
                if iter==1
                    push!(past_major_subprob_obj, subprob_obj)
                end
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
            end
            @info "[Outer-$isp_mode] Iter $iter: localLB=$(round(lower_bound, digits=4))  localUB=$(round(local_upper_bound, digits=4))  localGap=$(round(gap, digits=6))  (globalUB=$(round(upper_bound, digits=4)))"
            # 배열에 history 저장
            push!(past_lower_bound, lower_bound)
            push!(past_model_estimate, model_estimate)
            push!(past_minor_subprob_obj, subprob_obj)
            push!(past_upper_bound, upper_bound)
        end
        # Pruning: localLB > globalUB → 이 영역은 global best보다 나을 수 없음
        pruned = outer_tr && (lower_bound > upper_bound + tol * max(abs(upper_bound), 1e-10))
        if pruned
            @info "  ✂ Pruned: localLB=$(round(lower_bound, digits=4)) > globalUB=$(round(upper_bound, digits=4)). Skipping to next stage."
        end
        if gap <= tol || pruned
            if !outer_tr
                # No outer TR: simple convergence
                time_end = time()
                result[:solution_time] = time_end - time_start
                @info "  ✓ OPTIMAL (no outer TR). Gap = $gap"
                result[:past_lower_bound] = past_lower_bound
                result[:past_minor_subprob_obj] = past_minor_subprob_obj
                result[:past_upper_bound] = past_upper_bound
                result[:opt_sol] = Dict(:x=>x_sol, :h=>h_sol, :λ=>λ_sol, :ψ0=>ψ0_sol)
                return result
            end
            # Local optimality 달성
            if B_bin_stage <= length(B_bin_sequence)-1
                # Trust region 확장
                B_bin_stage +=1
                B_bin_old = B_bin
                B_bin = B_bin_sequence[B_bin_stage]
                push!(bin_B_steps, iter)
                push!(past_local_lower_bound, lower_bound)
                push!(past_local_optimizer, Dict(:x=>value.(x_sol), :h=>value.(h_sol), :λ=>value.(λ_sol), :ψ0=>value.(ψ0_sol)))
                @info "  ✓ Local optimal reached! Expanding B_bin to $B_bin"
                # TR constraint needs update (B_bin changed)
                tr_needs_update = true
                @info "Updating Trust Region"
                ## trust region radius를 확장
                tr_constraints = update_outer_trust_region_constraints!(
                    omp_model, omp_vars, centers, B_bin, B_con, tr_constraints, network)
                lower_bound = -Inf # master problem 영역 확장했으니까 다시 초기화
                local_upper_bound = Inf # local UB도 리셋
                # Reverse region constraint 추가 (선택사항)
                """
                Reverse region을 넣으면 B radius를 확장해도 기존 local optimal 주변은 탐색하지 않음.
                그러면 B radius를 끝까지 확장한 이후에, 과거에 찾은 local optimal 중 가장 좋은 (작은) 값을 선택하면 그게 global optimal을 보장함.
                """
                _ = add_reverse_region_constraint!(omp_model, omp_vars[:x], centers[:x], B_bin_old, network)


                    # # Optional: Add reverse region constraint
                # if use_reverse_constraints
                #     reverse_con = add_reverse_region_constraint!(
                #         omp_model, x, centers[:x], B_z_old, network
                #     )
                #     push!(reverse_constraints, reverse_con)
                # end
            else
                # Global Optimality 달성
                time_end = time()
                result[:solution_time] = time_end - time_start
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
            if isp_mode == :full_primal
                outer_cut_info = evaluate_master_opt_cut_from_primal(
                    primal_leader_instances, primal_follower_instances,
                    isp_data, cut_info, iter;
                    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, multi_cut=multi_cut)
            elseif isp_mode == :hybrid
                outer_cut_info = primal_evaluate_master_opt_cut(
                    leader_instances, follower_instances,
                    isp_data, cut_info, iter;
                    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, multi_cut=multi_cut)
            else
                outer_cut_info = evaluate_master_opt_cut(leader_instances, follower_instances, isp_data, cut_info, iter, multi_cut=multi_cut)
            end
            # Debug logging
            push!(result[:debug_α], cut_info[:α_sol])
            if multi_cut
                push!(result[:debug_intercept_l], sum(outer_cut_info[:intercept_l]))
                push!(result[:debug_intercept_f], sum(outer_cut_info[:intercept_f]))
            else
                # intercept_l/f may be nothing in single-cut mode; use total intercept
                push!(result[:debug_intercept_l], NaN)
                push!(result[:debug_intercept_f], NaN)
            end
            push!(result[:debug_coeff_norms], Dict(
                :Uhat1 => norm(outer_cut_info[:Uhat1]),
                :Utilde1 => norm(outer_cut_info[:Utilde1]),
                :Uhat3 => norm(outer_cut_info[:Uhat3]),
                :Utilde3 => norm(outer_cut_info[:Utilde3]),
                :βtilde1_1 => norm(outer_cut_info[:βtilde1_1]),
                :βtilde1_3 => norm(outer_cut_info[:βtilde1_3]),
                :Ztilde1_3 => norm(outer_cut_info[:Ztilde1_3]),
            ))
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

            # ===== Magnanti-Wong Cut Strengthening =====
            if strengthen_cuts && leader_instances !== nothing
                interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
                core_points = generate_core_points(network, γ, λU, w, v;
                    interdictable_idx=interdictable_idx, strategy=:interior)
                for (cp_idx, cp) in enumerate(core_points)
                    mw_info = evaluate_mw_opt_cut(
                        leader_instances, follower_instances, isp_data, cut_info, iter;
                        x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                        x_core=cp.x, λ_core=cp.λ, h_core=cp.h, ψ0_core=cp.ψ0,
                        multi_cut=multi_cut)
                    # Add MW cut to OMP (same structure as regular cut)
                    if multi_cut
                        mw_1_l = -ϕU * [sum(mw_info[:Uhat1][s,:,:] .* diag_x_E) for s in 1:S]
                        mw_1_f = -ϕU * [sum(mw_info[:Utilde1][s,:,:] .* diag_x_E) for s in 1:S]
                        mw_2_l = -ϕU * [sum(mw_info[:Uhat3][s,:,:] .* (isp_data[:E] - diag_x_E)) for s in 1:S]
                        mw_2_f = -ϕU * [sum(mw_info[:Utilde3][s,:,:] .* (isp_data[:E] - diag_x_E)) for s in 1:S]
                        mw_3_f = [sum(mw_info[:Ztilde1_3][s,:,:] .* (diag_λ_ψ * diagm(xi_bar[s]))) for s in 1:S]
                        mw_4_f = [(isp_data[:d0]'*mw_info[:βtilde1_1][s,:]) * λ for s in 1:S]
                        mw_5_f = -1 * [(h + diag_λ_ψ * xi_bar[s])'* mw_info[:βtilde1_3][s,:] for s in 1:S]
                        mw_cut_l = sum(mw_1_l) + sum(mw_2_l) + sum(mw_info[:intercept_l])
                        mw_cut_f = sum(mw_1_f) + sum(mw_2_f) + sum(mw_3_f) + sum(mw_4_f) + sum(mw_5_f) + sum(mw_info[:intercept_f])
                        mw_added_l = @constraint(omp_model, t_0_l >= mw_cut_l)
                        mw_added_f = @constraint(omp_model, t_0_f >= mw_cut_f)
                        set_name(mw_added_l, "mw_cut_$(iter)_cp$(cp_idx)_l")
                        set_name(mw_added_f, "mw_cut_$(iter)_cp$(cp_idx)_f")
                        result[:cuts]["mw_cut_$(iter)_cp$(cp_idx)_l"] = mw_added_l
                        result[:cuts]["mw_cut_$(iter)_cp$(cp_idx)_f"] = mw_added_f
                    else
                        mw_1 = -ϕU * [sum((mw_info[:Uhat1][s,:,:] + mw_info[:Utilde1][s,:,:]) .* diag_x_E) for s in 1:S]
                        mw_2 = -ϕU * [sum((mw_info[:Uhat3][s,:,:] + mw_info[:Utilde3][s,:,:]) .* (isp_data[:E] - diag_x_E)) for s in 1:S]
                        mw_3 = [sum(mw_info[:Ztilde1_3][s,:,:] .* (diag_λ_ψ * diagm(xi_bar[s]))) for s in 1:S]
                        mw_4 = [(isp_data[:d0]'*mw_info[:βtilde1_1][s,:]) * λ for s in 1:S]
                        mw_5 = -1 * [(h + diag_λ_ψ * xi_bar[s])'* mw_info[:βtilde1_3][s,:] for s in 1:S]
                        mw_cut = sum(mw_1) + sum(mw_2) + sum(mw_3) + sum(mw_4) + sum(mw_5) + sum(mw_info[:intercept])
                        mw_added = @constraint(omp_model, t_0 >= mw_cut)
                        set_name(mw_added, "mw_cut_$(iter)_cp$(cp_idx)")
                        result[:cuts]["mw_cut_$(iter)_cp$(cp_idx)"] = mw_added
                    end
                end
                @info "  $(length(core_points)) MW strengthening cuts added"
            end

            # Update TR constraints if needed
            if outer_tr && tr_needs_update
                @info "Updating Trust Region"
                tr_constraints = update_outer_trust_region_constraints!(
                    omp_model, omp_vars, centers, B_bin, B_con, tr_constraints, network)
            end
        end
    end
    # max_outer_iter에 도달했거나 while 조건이 false가 된 경우
    result[:past_upper_bound] = past_upper_bound
    result[:past_lower_bound] = past_lower_bound
    result[:opt_sol] = Dict(:x=>x_sol, :h=>h_sol, :λ=>λ_sol, :ψ0=>ψ0_sol)
    return result
end


function tr_nested_benders_optimize_hybrid!(omp_model::Model, omp_vars::Dict, network, ϕU, λU, γ, w, uncertainty_set;
    mip_optimizer=nothing, conic_optimizer=nothing, multi_cut=false, outer_tr=true, inner_tr=true, max_outer_iter=1000, full_primal=false, tol=1e-4, πU=ϕU, yU=ϕU, ytsU=ϕU)

    # full_primal=true is NOT recommended.
    # Outer cut extraction via Mosek IPM shadow prices (evaluate_master_opt_cut_from_primal)
    # produces inaccurate cut coefficients due to conic dual degeneracy.
    # Unlike the inner cut μ offset (uniform +ε, correctable), outer cut shadow prices
    # differ non-uniformly (30-40%) from dual ISP variable values, making simple correction impossible.
    # This leads to invalid outer cuts → OMP selects extreme (x,h,λ,ψ0) → primal ISP infeasible.
    # Use full_primal=false (hybrid mode: primal ISP inner + dual ISP outer) instead.
    # See memory/ipm_mu_offset.md and debug_test/test_outer_cut_compare.jl for details.
    if full_primal
        error(
            "full_primal=true is disabled: outer cut extraction from primal ISP shadow prices " *
            "is unreliable due to IPM conic dual degeneracy (non-uniform 30-40% coefficient errors). " *
            "Use full_primal=false (hybrid mode: primal ISP inner loop + dual ISP outer cuts) instead."
        )
    end

    ### -------- Trust Region 초기화 --------
    if outer_tr
        num_interdictable = sum(network.interdictable_arcs)
        max_dist = min(Int(2γ), num_interdictable) # effective diameter: 2γ
        B_bin_sequence = unique([1, ceil(Int, max_dist/4), ceil(Int, max_dist/2), max_dist])
        B_bin_stage = 1
        B_bin = B_bin_sequence[B_bin_stage]
        B_con = nothing
        centers = Dict{Symbol, Any}(
            :x => nothing, :h => nothing, :λ => nothing, :ψ0 => nothing
        )
        β_relative = 1e-4
        tr_constraints = Dict{Symbol, Any}(:binary => nothing, :continuous => nothing)
    end
    upper_bound = Inf

    ### --------OMP Initialization--------
    st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars)
    x, h, λ, ψ0 = omp_vars[:x], omp_vars[:h], omp_vars[:λ], omp_vars[:ψ0]
    if outer_tr
        centers[:x] = value.(x)
        centers[:h] = value.(h)
        centers[:λ] = value.(λ)
        centers[:ψ0] = value.(ψ0)
    end
    if multi_cut
        t_0_l = omp_vars[:t_0_l]
        t_0_f = omp_vars[:t_0_f]
        t_0 = t_0_l + t_0_f
    else
        t_0 = omp_vars[:t_0]
    end

    num_arcs = length(network.arcs) - 1
    E = ones(num_arcs, num_arcs+1)
    diag_x_E = Diagonal(x) * E
    diag_λ_ψ = Diagonal(λ*ones(num_arcs)-v.*ψ0)
    d0 = zeros(num_arcs + 1)
    d0[end] = 1.0
    xi_bar_local = uncertainty_set[:xi_bar]
    iter = 0
    past_obj = []
    past_major_subprob_obj = []
    past_minor_subprob_obj = []
    past_model_estimate = []
    past_local_lower_bound = []
    past_upper_bound = []
    past_lower_bound = []
    past_local_optimizer = []
    major_iter = []
    bin_B_steps = []
    imp_cuts = Dict{Symbol, Any}(:old_tr_constraints => nothing)
    result = Dict()
    result[:cuts] = Dict()
    result[:tr_info] = Dict()
    result[:inner_iter] = []
    upper_bound = Inf  # global UB (never reset)
    local_upper_bound = Inf  # local UB (reset per stage)
    lower_bound = -Inf

    ### --------IMP + ISP Initialization--------
    imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_optimizer=mip_optimizer)
    st, α_sol = initialize_imp(imp_model, imp_vars)

    # Dual ISP instances (for outer cut generation — hybrid only)
    dual_leader_instances, dual_follower_instances = nothing, nothing
    if !full_primal
        dual_leader_instances, dual_follower_instances = initialize_isp(
            network, S, ϕU, λU, γ, w, v, uncertainty_set;
            conic_optimizer=conic_optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol,
            πU=πU, yU=yU, ytsU=ytsU)
    end

    # Primal ISP instances (for inner loop + full primal outer cuts)
    primal_leader_instances, primal_follower_instances = initialize_primal_isp(
        network, S, ϕU, λU, γ, w, v, uncertainty_set;
        conic_optimizer=conic_optimizer, x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)

    isp_data = Dict(:E => E, :network => network, :ϕU => ϕU, :πU => πU, :yU => yU, :ytsU => ytsU, :λU => λU, :γ => γ, :w => w, :v => v, :uncertainty_set => uncertainty_set, :d0 => d0, :S=>S)
    gap = Inf

    ### --------End Initialization--------
    time_start = time()
    while (st == MOI.DUAL_INFEASIBLE || st == MOI.OPTIMAL)
        iter += 1
        if iter > max_outer_iter
            @warn "Maximum outer iterations ($max_outer_iter) reached. Gap = $gap"
            break
        end
        if outer_tr
            @info "[Outer-$(full_primal ? "FullPrimal" : "Hybrid")] Iteration $iter (B_bin=$B_bin, Stage=$(B_bin_stage)/$(length(B_bin_sequence)))"
        else
            @info "[Outer-$(full_primal ? "FullPrimal" : "Hybrid")] Iteration $iter"
        end

        # Outer Master Problem
        optimize!(omp_model)
        st = MOI.get(omp_model, MOI.TerminationStatus())
        if st == MOI.INFEASIBLE
            @info " Outer Master Problem infeasible (Converged): No search space left"
            gap = 0.0
        else
            x_sol = round.(value.(omp_vars[:x]))  # binary rounding for numerical stability
            h_sol, λ_sol, ψ0_sol = value.(omp_vars[:h]), value(omp_vars[:λ]), value.(omp_vars[:ψ0])
            model_estimate = value(t_0)
            lower_bound = max(lower_bound, model_estimate)

            # Update primal ISP parameters (x,h,λ,ψ0 in constraint RHS → set_normalized_rhs)
            update_primal_isp_parameters!(primal_leader_instances, primal_follower_instances;
                x_sol=x_sol, h_sol=h_sol, λ_sol=λ_sol, ψ0_sol=ψ0_sol, isp_data=isp_data)

            # Hybrid inner loop (primal ISP)
            status, cut_info = tr_imp_optimize_hybrid!(imp_model, imp_vars,
                primal_leader_instances, primal_follower_instances;
                isp_data=isp_data, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                outer_iter=iter, imp_cuts=imp_cuts, inner_tr=inner_tr, tol=tol)

            if status != :OptimalityCut
                @warn "Outer Subproblem not optimal (hybrid)"
                @infiltrate
            end
            push!(result[:inner_iter], cut_info[:iter])
            imp_cuts[:old_cuts] = cut_info[:cuts]
            if inner_tr && cut_info[:tr_constraints] !== nothing
                imp_cuts[:old_tr_constraints] = cut_info[:tr_constraints]
            end
            subprob_obj = cut_info[:obj_val]
            upper_bound = min(upper_bound, subprob_obj)
            local_upper_bound = min(local_upper_bound, subprob_obj)

            gap = abs(local_upper_bound - lower_bound) / max(abs(local_upper_bound), 1e-10)
            if outer_tr
                if iter==1
                    push!(past_major_subprob_obj, subprob_obj)
                end
                tr_needs_update = false
                predicted_decrease = past_major_subprob_obj[end] - model_estimate
                β_dynamic = max(1e-8, β_relative * predicted_decrease)
                improvement = past_major_subprob_obj[end] - subprob_obj
                is_serious_step = (improvement >= β_dynamic)
                if is_serious_step
                    centers[:x] = value.(x_sol)
                    centers[:h] = value.(h_sol)
                    centers[:λ] = value.(λ_sol)
                    centers[:ψ0] = value.(ψ0_sol)
                    push!(major_iter, iter)
                    push!(past_major_subprob_obj, subprob_obj)
                    tr_needs_update = true
                end
            end
            @info "[Outer-$(full_primal ? "FullPrimal" : "Hybrid")] Iter $iter: localLB=$(round(lower_bound, digits=4))  localUB=$(round(local_upper_bound, digits=4))  localGap=$(round(gap, digits=6))  (globalUB=$(round(upper_bound, digits=4)))"
            push!(past_lower_bound, lower_bound)
            push!(past_model_estimate, model_estimate)
            push!(past_minor_subprob_obj, subprob_obj)
            push!(past_upper_bound, upper_bound)
        end
        # Pruning: localLB > globalUB → 이 영역은 global best보다 나을 수 없음
        pruned = outer_tr && (lower_bound > upper_bound + tol * max(abs(upper_bound), 1e-10))
        if pruned
            @info "  ✂ Pruned: localLB=$(round(lower_bound, digits=4)) > globalUB=$(round(upper_bound, digits=4)). Skipping to next stage."
        end
        if gap <= tol || pruned
            if !outer_tr
                time_end = time()
                result[:solution_time] = time_end - time_start
                @info "  ✓ OPTIMAL (no outer TR, hybrid). Gap = $gap"
                result[:past_lower_bound] = past_lower_bound
                result[:past_minor_subprob_obj] = past_minor_subprob_obj
                result[:past_upper_bound] = past_upper_bound
                result[:opt_sol] = Dict(:x=>x_sol, :h=>h_sol, :λ=>λ_sol, :ψ0=>ψ0_sol)
                return result
            end
            if B_bin_stage <= length(B_bin_sequence)-1
                B_bin_stage +=1
                B_bin_old = B_bin
                B_bin = B_bin_sequence[B_bin_stage]
                push!(bin_B_steps, iter)
                push!(past_local_lower_bound, lower_bound)
                push!(past_local_optimizer, Dict(:x=>value.(x_sol), :h=>value.(h_sol), :λ=>value.(λ_sol), :ψ0=>value.(ψ0_sol)))
                @info "  ✓ Local optimal reached! Expanding B_bin to $B_bin"
                tr_needs_update = true
                @info "Updating Trust Region"
                tr_constraints = update_outer_trust_region_constraints!(
                    omp_model, omp_vars, centers, B_bin, B_con, tr_constraints, network)
                lower_bound = -Inf
                local_upper_bound = Inf  # local UB도 리셋
                _ = add_reverse_region_constraint!(omp_model, omp_vars[:x], centers[:x], B_bin_old, network)
            else
                time_end = time()
                result[:solution_time] = time_end - time_start
                @info "  ✓✓ GLOBAL OPTIMAL (hybrid)! (B_bin = full region)"
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
                result[:tr_info][:major_iter] = major_iter
                result[:tr_info][:bin_B_steps] = bin_B_steps
                result[:opt_sol] = past_local_optimizer[min_idx]
                result[:iter_when_global_optimal] = iter_when_global_optimal
                return result
            end
        else
            # Gap still large → Generate outer cut
            if full_primal
                # Full primal: extract cut coefficients from primal ISP shadow prices
                outer_cut_info = evaluate_master_opt_cut_from_primal(
                    primal_leader_instances, primal_follower_instances,
                    isp_data, cut_info, iter;
                    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                    multi_cut=multi_cut)
            else
                # Hybrid: update dual ISP objectives, then extract cut coefficients
                outer_cut_info = primal_evaluate_master_opt_cut(
                    dual_leader_instances, dual_follower_instances,
                    isp_data, cut_info, iter;
                    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                    multi_cut=multi_cut)
            end

            if multi_cut
                cut_1_l =  -ϕU * [sum(outer_cut_info[:Uhat1][s,:,:] .* diag_x_E) for s in 1:S]
                cut_1_f =  -ϕU * [sum(outer_cut_info[:Utilde1][s,:,:] .* diag_x_E) for s in 1:S]
                cut_2_l =  -ϕU * [sum(outer_cut_info[:Uhat3][s,:,:] .* (isp_data[:E] - diag_x_E)) for s in 1:S]
                cut_2_f =  -ϕU * [sum(outer_cut_info[:Utilde3][s,:,:] .* (isp_data[:E] - diag_x_E)) for s in 1:S]
                cut_3_f =  [sum(outer_cut_info[:Ztilde1_3][s,:,:] .* (diag_λ_ψ * diagm(xi_bar_local[s]))) for s in 1:S]
                cut_4_f =  [(isp_data[:d0]'*outer_cut_info[:βtilde1_1][s,:]) * λ for s in 1:S]
                cut_5_f =  -1* [(h + diag_λ_ψ * xi_bar_local[s])'* outer_cut_info[:βtilde1_3][s,:] for s in 1:S]
                cut_intercept_l = outer_cut_info[:intercept_l]
                cut_intercept_f = outer_cut_info[:intercept_f]
                opt_cut_l = sum(cut_1_l)+ sum(cut_2_l) + sum(cut_intercept_l)
                opt_cut_f = sum(cut_1_f)+ sum(cut_2_f)+ sum(cut_3_f)+ sum(cut_4_f)+ sum(cut_5_f) + sum(cut_intercept_f)
                cut_added_l = @constraint(omp_model, t_0_l >= opt_cut_l)
                cut_added_f = @constraint(omp_model, t_0_f >= opt_cut_f)
                set_name(cut_added_l, "opt_cut_$(iter)_l")
                set_name(cut_added_f, "opt_cut_$(iter)_f")
                result[:cuts]["opt_cut_$(iter)_l"] = cut_added_l
                result[:cuts]["opt_cut_$(iter)_f"] = cut_added_f
            else
                cut_1 =  -ϕU * [sum((outer_cut_info[:Uhat1][s,:,:] + outer_cut_info[:Utilde1][s,:,:]) .* diag_x_E) for s in 1:S]
                cut_2 =  -ϕU * [sum((outer_cut_info[:Uhat3][s,:,:] + outer_cut_info[:Utilde3][s,:,:]) .* (isp_data[:E] - diag_x_E)) for s in 1:S]
                cut_3 =  [sum(outer_cut_info[:Ztilde1_3][s,:,:] .* (diag_λ_ψ * diagm(xi_bar_local[s]))) for s in 1:S]
                cut_4 =  [(isp_data[:d0]'*outer_cut_info[:βtilde1_1][s,:]) * λ for s in 1:S]
                cut_5 =  -1* [(h + diag_λ_ψ * xi_bar_local[s])'* outer_cut_info[:βtilde1_3][s,:] for s in 1:S]
                cut_intercept = outer_cut_info[:intercept]
                opt_cut = sum(cut_1)+ sum(cut_2)+ sum(cut_3)+ sum(cut_4)+ sum(cut_5)+ sum(cut_intercept)
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
                println("something went wrong (hybrid)")
                @infiltrate
            end
            println("subproblem objective (hybrid): ", subprob_obj)
            @info "Optimality cut added (hybrid)"

            if outer_tr && tr_needs_update
                @info "Updating Trust Region"
                tr_constraints = update_outer_trust_region_constraints!(
                    omp_model, omp_vars, centers, B_bin, B_con, tr_constraints, network)
            end
        end
    end
    # max_outer_iter reached or while condition became false
    result[:past_upper_bound] = past_upper_bound
    result[:past_lower_bound] = past_lower_bound
    result[:opt_sol] = Dict(:x=>x_sol, :h=>h_sol, :λ=>λ_sol, :ψ0=>ψ0_sol)
    return result
end


function build_isp_leader(network, S, ϕU, λU, γ, w, v, uncertainty_set, optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, true_S; πU=ϕU)
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
    obj_term_ub_hat = [-ϕU * sum(Phat1_Φ[s,:,:]) - πU * sum(Phat1_Π[s,:,:]) for s=1:S]
    obj_term_lb_hat = [-ϕU * sum(Phat2_Φ[s,:,:]) - πU * sum(Phat2_Π[s,:,:]) for s=1:S]
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

function build_isp_follower(network, S, ϕU, λU, γ, w, v, uncertainty_set, optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, true_S; πU=ϕU, yU=ϕU, ytsU=ϕU)
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
    obj_term_ub_tilde = [-ϕU * sum(Ptilde1_Φ[s,:,:]) - πU * sum(Ptilde1_Π[s,:,:]) - yU * sum(Ptilde1_Y[s,:,:]) - ytsU * sum(Ptilde1_Yts[s,:]) for s=1:S]
    obj_term_lb_tilde = [-ϕU * sum(Ptilde2_Φ[s,:,:]) - πU * sum(Ptilde2_Π[s,:,:]) - yU * sum(Ptilde2_Y[s,:,:]) - ytsU * sum(Ptilde2_Yts[s,:]) for s=1:S]
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