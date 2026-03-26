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
includet("parallel_utils.jl")
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
    @variable(model, α[k=1:num_arcs], lower_bound=0.0, upper_bound=w/S)
    @constraint(model, sum(α) == w*(1/S)) # full model에선 자연스럽게 inequality가 equality가 되지만 decomposed된 imp에선 그런다는 보장이 없으므로 명시적으로 equality 유지
    @objective(model, Max, (sum(t_1_l) + sum(t_1_f)) / S)

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
    scaling_S = get(isp_data, :scaling_S, isp_data[:S])

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
        intercept, subgradient = (1/scaling_S)*sum(ηhat), μhat ##실제 S로 나눠주어야 함.
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
    scaling_S = get(isp_data, :scaling_S, isp_data[:S])
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
        # intercept = (1/scaling_S)*sum(ηtilde) ##실제 S로 나눠주어야 함.
        ηtilde_pos = shadow_price.(vec(model[:cons_dual_constant_pos]))
        ηtilde_neg = shadow_price.(vec(model[:cons_dual_constant_neg]))
        intercept = sum((1/scaling_S)*(ηtilde_pos-ηtilde_neg)) ## 이러면 ηtilde sign 반대로 나오는거 robust하게 대응 가능.
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

function tr_imp_optimize!(imp_model::Model, imp_vars::Dict, isp_leader_instances::Dict, isp_follower_instances::Dict; isp_data=nothing, λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing, outer_iter=nothing, imp_cuts=nothing, inner_tr=true, tol=1e-6, parallel=false, inexact=false, inexact_tol=0.1, max_inexact_iter=5)
    st = MOI.get(imp_model, MOI.TerminationStatus())
    iter = 0
    uncertainty_set = isp_data[:uncertainty_set]
    R = uncertainty_set[:R]
    r_dict = uncertainty_set[:r_dict]
    xi_bar = uncertainty_set[:xi_bar]
    epsilon = uncertainty_set[:epsilon]
    S_total = isp_data[:S]  # scenario count for /S averaging
    S = S_total
    past_obj = []
    past_subprob_obj = []
    past_major_subprob_obj = []
    past_lower_bound = []
    past_upper_bound = []
    major_iter = []
    lower_bound = -Inf ## inner master problem은 Maximization이니까 feasible solution은 lower bound를 제공.
    effective_tol = inexact ? inexact_tol : tol
    effective_max_iter = inexact ? max_inexact_iter : typemax(Int)
    result = Dict()
    result[:cuts] = Dict()
    result[:inexact] = inexact  # 호출자에게 inexact 여부 전달
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
        model_estimate = (sum(value.(imp_vars[:t_1_l])) + sum(value.(imp_vars[:t_1_f]))) / S_total  # average over scenarios
        subprob_obj = 0
        dict_cut_info_l, dict_cut_info_f = Dict(), Dict()
        scenario_results, status = solve_scenarios(S; parallel=parallel) do s
            U_s = Dict(:R => Dict(:1=>R[s]), :r_dict => Dict(:1=>r_dict[s]), :xi_bar => Dict(:1=>xi_bar[s]), :epsilon => epsilon)
            (status_l, cut_info_l) =isp_leader_optimize!(isp_leader_instances[s][1], isp_leader_instances[s][2]; isp_data=isp_data, uncertainty_set=U_s, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
            (status_f, cut_info_f) =isp_follower_optimize!(isp_follower_instances[s][1], isp_follower_instances[s][2]; isp_data=isp_data, uncertainty_set=U_s, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
            ok = (status_l == :OptimalityCut) && (status_f == :OptimalityCut)
            return (ok, (cut_info_l, cut_info_f))
        end
        for s in 1:S
            dict_cut_info_l[s] = scenario_results[s][1]
            dict_cut_info_f[s] = scenario_results[s][2]
            subprob_obj += scenario_results[s][1][:obj_val] + scenario_results[s][2][:obj_val]
        end
        subprob_obj /= S_total  # average over scenarios
        # Best LB 갱신 시 outer cut coefficients 스냅샷 저장
        if subprob_obj > lower_bound + 1e-8
            result[:best_lb_outer_cut] = extract_outer_cut_from_current_isp(isp_leader_instances, isp_follower_instances, S)
            result[:best_lb_α] = copy(α_sol)
        end
        lower_bound = max(lower_bound, subprob_obj) ## inner master problem은 Maximization이니까 우린 항상 더 높은 값을 추구
        gap = abs(model_estimate - lower_bound) / max(abs(model_estimate), 1e-10)
        inner_converged = gap <= effective_tol || lower_bound > model_estimate - 1e-4 || iter >= effective_max_iter
        # Inexact: 수렴/iter cap 도달 시 즉시 반환 (B_conti 확장 없이)
        if inexact && inner_converged
            @info "Termination condition met (inexact, gap=$(round(gap, digits=4)), iter=$iter)"
            println("model_estimate: ", model_estimate, ", subprob_obj: ", subprob_obj)
            result[:past_obj] = past_obj
            result[:past_subprob_obj] = past_subprob_obj
            result[:α_sol] = α_sol
            result[:obj_val] = subprob_obj
            result[:past_lower_bound] = past_lower_bound
            result[:iter] = iter
            result[:tr_constraints] = (inner_tr && tr_constraints[:continuous] !== nothing) ? tr_constraints[:continuous] : nothing
            return (:OptimalityCut, result)
        end
        # inner_tr일 때: TR이 아직 최대가 아니면 확장하고 계속 탐색
        if inner_converged && inner_tr && B_conti < B_conti_max - 1e-8
            B_conti = min(B_conti_max, B_conti * 2.0)
            @info "Inner converged within TR: expanding B_conti to $(round(B_conti, digits=4))/$(round(B_conti_max, digits=4))"
            centers[:α] = α_sol
            push!(past_major_subprob_obj, subprob_obj)
            lower_bound = -Inf  # 확장 후 수렴 판정을 리셋해야 새 영역에서 cut 추가 가능
            tr_constraints = update_inner_trust_region_constraints!(imp_model, imp_vars, centers, B_conti, tr_constraints, network)
        elseif inner_converged
            @info "Termination condition met$(inexact ? " (inexact, gap=$(round(gap, digits=4)))" : "")"
            println("model_estimate: ", model_estimate, ", subprob_obj: ", subprob_obj)
            result[:past_obj] = past_obj
            result[:past_subprob_obj] = past_subprob_obj
            result[:α_sol] = α_sol
            result[:obj_val] = subprob_obj
            result[:past_lower_bound] = past_lower_bound
            result[:iter] = iter
            if inner_tr && tr_constraints[:continuous] !== nothing
                result[:tr_constraints] = tr_constraints[:continuous]
            else
                result[:tr_constraints] = nothing
            end
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
            if abs(subprob_obj * S_total - opt_cut_val) > 1e-4  # opt_cut_val is raw, subprob_obj is /S
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

function initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set; conic_optimizer=nothing, λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing, α_sol=nothing, πU=ϕU, yU=ϕU, ytsU=ϕU, scaling_S=S)
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    leader_instances = Dict{Int, Tuple{Model, Dict}}()
    follower_instances = Dict{Int, Tuple{Model, Dict}}()
    for s in 1:S
        U_s = Dict(:R => Dict(:1=>R[s]), :r_dict => Dict(:1=>r_dict[s]), :xi_bar => Dict(:1=>xi_bar[s]), :epsilon => epsilon)
        leader_instances[s] = build_isp_leader(network, 1, ϕU, λU, γ, w, v, U_s, conic_optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, scaling_S; πU=πU)
        follower_instances[s] = build_isp_follower(network, 1, ϕU, λU, γ, w, v, U_s, conic_optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, scaling_S; πU=πU, yU=yU, ytsU=ytsU)

    end
    return leader_instances, follower_instances
end

function evaluate_master_opt_cut(isp_leader_instances::Dict, isp_follower_instances::Dict, isp_data::Dict, cut_info::Dict, iter::Int; parallel=false)
    """
    α를 fix 시키고 outer subproblem의 값을 다시 정확하게 구하는 코드
    """
    S = isp_data[:S]
    α_sol = cut_info[:α_sol]
    _, status = solve_scenarios(S; parallel=parallel) do s
        α_s = α_sol isa Vector{<:AbstractVector} ? α_sol[s] : α_sol  # per-scenario α support
        model_l = isp_leader_instances[s][1]
        model_f = isp_follower_instances[s][1]
        set_normalized_rhs.(vec(model_l[:coupling_cons]), α_s)
        optimize!(model_l)
        st_l = MOI.get(model_l, MOI.TerminationStatus())

        set_normalized_rhs.(vec(model_f[:coupling_cons]), α_s)
        optimize!(model_f)
        st_f = MOI.get(model_f, MOI.TerminationStatus())

        ok = (st_l == MOI.OPTIMAL) && (st_f == MOI.OPTIMAL)
        if !ok
            if (st_l == MOI.SLOW_PROGRESS) || (st_f == MOI.SLOW_PROGRESS)
                ok = true
            elseif !parallel
                @infiltrate
            end
        end
        return (ok, nothing)
    end
    Uhat1 = cat([value.(isp_leader_instances[s][2][:Uhat1]) for s in 1:S]...; dims=1)
    Utilde1 = cat([value.(isp_follower_instances[s][2][:Utilde1]) for s in 1:S]...; dims=1)
    Uhat3 = cat([value.(isp_leader_instances[s][2][:Uhat3]) for s in 1:S]...; dims=1)
    Utilde3 = cat([value.(isp_follower_instances[s][2][:Utilde3]) for s in 1:S]...; dims=1)
    Ztilde1_3 = cat([value.(isp_follower_instances[s][2][:Ztilde1_3]) for s in 1:S]...; dims=1)
    βtilde1_1 = cat([value.(isp_follower_instances[s][2][:βtilde1_1]) for s in 1:S]...; dims=1)
    βtilde1_3 = cat([value.(isp_follower_instances[s][2][:βtilde1_3]) for s in 1:S]...; dims=1)

    intercept_l = [value.(isp_leader_instances[s][2][:intercept]) for s in 1:S]
    intercept_f = [value.(isp_follower_instances[s][2][:intercept]) for s in 1:S]
    intercept = sum(intercept_l) + sum(intercept_f)
    leader_obj = sum(objective_value(isp_leader_instances[s][1]) for s in 1:S)
    follower_obj = sum(objective_value(isp_follower_instances[s][1]) for s in 1:S)
    avg_obj = (leader_obj + follower_obj) / S  # average over scenarios
    println("avg of leader and follower objective: ", avg_obj, ", cut_info[:obj_val]: ", cut_info[:obj_val])
    println("Outer loop iteration: ", iter)
    @assert abs(avg_obj - cut_info[:obj_val]) < 1e-3
    return Dict(:Uhat1=>Uhat1, :Utilde1=>Utilde1, :Uhat3=>Uhat3, :Utilde3=>Utilde3, :Ztilde1_3=>Ztilde1_3
    ,:βtilde1_1=>βtilde1_1, :βtilde1_3=>βtilde1_3, :intercept=>intercept, :intercept_l=>intercept_l, :intercept_f=>intercept_f)
end


"""
    extract_outer_cut_from_current_isp(isp_leader_instances, isp_follower_instances, S)

현재 ISP solution에서 outer cut coefficients를 추출 (re-solve 없이 value.()만 호출).
Inner loop 중간 iteration에서 outer cut을 수집할 때 사용.
"""
function extract_outer_cut_from_current_isp(isp_leader_instances::Dict, isp_follower_instances::Dict, S::Int)
    Uhat1 = cat([value.(isp_leader_instances[s][2][:Uhat1]) for s in 1:S]...; dims=1)
    Utilde1 = cat([value.(isp_follower_instances[s][2][:Utilde1]) for s in 1:S]...; dims=1)
    Uhat3 = cat([value.(isp_leader_instances[s][2][:Uhat3]) for s in 1:S]...; dims=1)
    Utilde3 = cat([value.(isp_follower_instances[s][2][:Utilde3]) for s in 1:S]...; dims=1)
    Ztilde1_3 = cat([value.(isp_follower_instances[s][2][:Ztilde1_3]) for s in 1:S]...; dims=1)
    βtilde1_1 = cat([value.(isp_follower_instances[s][2][:βtilde1_1]) for s in 1:S]...; dims=1)
    βtilde1_3 = cat([value.(isp_follower_instances[s][2][:βtilde1_3]) for s in 1:S]...; dims=1)
    intercept_l = [value.(isp_leader_instances[s][2][:intercept]) for s in 1:S]
    intercept_f = [value.(isp_follower_instances[s][2][:intercept]) for s in 1:S]
    intercept = sum(intercept_l) + sum(intercept_f)
    return Dict(:Uhat1=>Uhat1, :Utilde1=>Utilde1, :Uhat3=>Uhat3, :Utilde3=>Utilde3,
                :Ztilde1_3=>Ztilde1_3, :βtilde1_1=>βtilde1_1, :βtilde1_3=>βtilde1_3,
                :intercept=>intercept, :intercept_l=>intercept_l, :intercept_f=>intercept_f)
end


# ===== Subgradient Heuristic: IMP 대체 =====

"""
    project_simplex(z, w)

z를 simplex {α ≥ 0, Σα = w}에 L2 projection.
"""
function project_simplex(z::Vector{Float64}, w::Float64)
    n = length(z)
    u = sort(z, rev=true)
    cssv = cumsum(u)
    rho = findlast(i -> u[i] > (cssv[i] - w) / i, 1:n)
    theta = (cssv[rho] - w) / rho
    return max.(z .- theta, 0.0)
end

"""
    subgradient_alpha_step!(α_current, isp_leader_instances, isp_follower_instances, isp_data;
        λ_sol, x_sol, h_sol, ψ0_sol, step_size=0.1, parallel=false)

IMP를 풀지 않고, 현재 α에서 ISP 1라운드(2S회)만 풀어서:
  1. outer cut coefficients 추출
  2. μ (supergradient) → projected subgradient step으로 α 업데이트
기존 tr_imp_optimize!의 inexact phase를 대체.

Returns: (:OptimalityCut, result) — tr_imp_optimize!과 동일한 인터페이스
"""
function subgradient_alpha_step!(α_current::Vector{Float64},
    isp_leader_instances::Dict, isp_follower_instances::Dict, isp_data::Dict;
    λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing,
    step_size::Float64=0.1, parallel=false)

    uncertainty_set = isp_data[:uncertainty_set]
    R = uncertainty_set[:R]
    r_dict = uncertainty_set[:r_dict]
    xi_bar = uncertainty_set[:xi_bar]
    epsilon = uncertainty_set[:epsilon]
    S = isp_data[:S]
    w = isp_data[:w]

    # Step 1: ISP 풀기 (2S회) — 현재 α에서
    α_sol = copy(α_current)
    scenario_results, status = solve_scenarios(S; parallel=parallel) do s
        U_s = Dict(:R => Dict(:1=>R[s]), :r_dict => Dict(:1=>r_dict[s]),
                    :xi_bar => Dict(:1=>xi_bar[s]), :epsilon => epsilon)
        (status_l, cut_info_l) = isp_leader_optimize!(
            isp_leader_instances[s][1], isp_leader_instances[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
        (status_f, cut_info_f) = isp_follower_optimize!(
            isp_follower_instances[s][1], isp_follower_instances[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
        ok = (status_l == :OptimalityCut) && (status_f == :OptimalityCut)
        return (ok, (cut_info_l, cut_info_f))
    end

    # Step 2: obj, μ 수집
    subprob_obj = 0.0
    μ_total = zeros(length(α_sol))
    for s in 1:S
        cut_l, cut_f = scenario_results[s]
        subprob_obj += cut_l[:obj_val] + cut_f[:obj_val]
        μ_total .+= cut_l[:μhat] .+ cut_f[:μtilde]
    end
    subprob_obj /= S  # average

    # Step 3: outer cut coefficients 추출 (re-solve 없이)
    outer_cut_info = extract_outer_cut_from_current_isp(isp_leader_instances, isp_follower_instances, S)

    # Step 4: α update — projected subgradient ascent
    α_new = project_simplex(α_current .+ step_size .* μ_total, w / S)

    # result 구성 (tr_imp_optimize! 호환)
    result = Dict(
        :α_sol => α_sol,        # ISP를 풀 때 사용한 α (outer cut용)
        :α_new => α_new,        # subgradient step 후 새 α (다음 iteration용)
        :obj_val => subprob_obj,
        :iter => 1,
        :cuts => Dict(),
        :inexact => true,
        :subgradient => true,   # subgradient heuristic 식별 플래그
        :outer_cut_info => outer_cut_info,  # ISP 이미 풀었으므로 re-solve 불필요
        :past_obj => [subprob_obj],
        :past_subprob_obj => [subprob_obj],
        :past_lower_bound => [subprob_obj],
        :tr_constraints => nothing,
        :μ_total => μ_total,
    )

    println("  [Subgrad] obj=$(round(subprob_obj, digits=6)), ‖μ‖=$(round(norm(μ_total), digits=4)), step=$step_size")
    return (:OptimalityCut, result)
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
    parallel=false)

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

    # Cache z_star before any modifications (re-solve 불필요하게 만듦)
    z_star_cache_l = [objective_value(isp_leader_instances[s][1]) for s in 1:S]
    z_star_cache_f = [objective_value(isp_follower_instances[s][1]) for s in 1:S]

    mw_results, _ = solve_scenarios(S; parallel=parallel) do s
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

        z_star_l = z_star_cache_l[s]

        # Reconstruct original objective (at x_sol) — mirrors isp_leader_optimize! lines 194-201
        orig_l_1 = -ϕU * sum(Uhat1[1, :, :] .* diag_x_sol_E)
        orig_l_2 = -ϕU * sum(Uhat3[1, :, :] .* (E - diag_x_sol_E))
        orig_l_3 = (d0') * βhat1_1[1, :]
        orig_l_ub = -ϕU * sum(Phat1_Φ[1, :, :]) - πU * sum(Phat1_Π[1, :, :])
        orig_l_lb = -ϕU * sum(Phat2_Φ[1, :, :]) - πU * sum(Phat2_Π[1, :, :])
        orig_obj_l = orig_l_1 + orig_l_2 + orig_l_3 + orig_l_ub + orig_l_lb

        # MW optimality constraint
        mw_con_l = @constraint(model_l, orig_obj_l >= z_star_l - 1e-6)

        # Core objective (replace x_sol → x_core; intercept terms unchanged)
        core_l_1 = -ϕU * sum(Uhat1[1, :, :] .* diag_x_core_E)
        core_l_2 = -ϕU * sum(Uhat3[1, :, :] .* (E - diag_x_core_E))
        core_obj_l = core_l_1 + core_l_2 + orig_l_3 + orig_l_ub + orig_l_lb
        @objective(model_l, Max, core_obj_l)
        optimize!(model_l)

        st_l = termination_status(model_l)
        if !(st_l == MOI.OPTIMAL || st_l == MOI.SLOW_PROGRESS)
            @warn "MW leader solve failed for s=$s: $st_l, using original solution"
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

        z_star_f = z_star_cache_f[s]

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

        return (true, (mw_con_l, model_l, mw_con_f, model_f))
    end

    # Collect mw_cons for cleanup
    mw_cons = []
    for s in 1:S
        (mw_con_l, model_l, mw_con_f, model_f) = mw_results[s]
        push!(mw_cons, (model_l, mw_con_l))
        push!(mw_cons, (model_f, mw_con_f))
    end

    # ===== Extract coefficients (same as evaluate_master_opt_cut) =====
    Uhat1_out = cat([value.(isp_leader_instances[s][2][:Uhat1]) for s in 1:S]...; dims=1)
    Utilde1_out = cat([value.(isp_follower_instances[s][2][:Utilde1]) for s in 1:S]...; dims=1)
    Uhat3_out = cat([value.(isp_leader_instances[s][2][:Uhat3]) for s in 1:S]...; dims=1)
    Utilde3_out = cat([value.(isp_follower_instances[s][2][:Utilde3]) for s in 1:S]...; dims=1)
    Ztilde1_3_out = cat([value.(isp_follower_instances[s][2][:Ztilde1_3]) for s in 1:S]...; dims=1)
    βtilde1_1_out = cat([value.(isp_follower_instances[s][2][:βtilde1_1]) for s in 1:S]...; dims=1)
    βtilde1_3_out = cat([value.(isp_follower_instances[s][2][:βtilde1_3]) for s in 1:S]...; dims=1)

    intercept_l = [value.(isp_leader_instances[s][2][:intercept]) for s in 1:S]
    intercept_f = [value.(isp_follower_instances[s][2][:intercept]) for s in 1:S]
    intercept = sum(intercept_l) + sum(intercept_f)

    # ===== Cleanup: delete MW constraints only (re-solve 불필요 — z_star는 캐싱됨) =====
    for item in mw_cons
        item === nothing && continue
        model, con = item
        delete(model, con)
    end

    return Dict(:Uhat1=>Uhat1_out, :Utilde1=>Utilde1_out, :Uhat3=>Uhat3_out, :Utilde3=>Utilde3_out,
        :Ztilde1_3=>Ztilde1_3_out, :βtilde1_1=>βtilde1_1_out, :βtilde1_3=>βtilde1_3_out,
        :intercept=>intercept, :intercept_l=>intercept_l, :intercept_f=>intercept_f)
end


"""
    evaluate_joint_mw_opt_cut(isp_data, cut_info, iter; ...)

Joint MW cut strengthening: leader+follower를 하나의 OSP 모델에서 동시 최적화.
Separate MW와 달리 z_l + z_f ≥ z*_total만 요구하므로 더 큰 자유도 → tighter cut.
OSP(build_dualized_outer_subproblem)를 재활용하여 α 고정 후 MW 수행.
"""
function evaluate_joint_mw_opt_cut(
    isp_data, cut_info, iter;
    x_sol, λ_sol, h_sol, ψ0_sol,
    x_core, λ_core, h_core, ψ0_core,
    conic_optimizer)

    S = isp_data[:S]
    ϕU = isp_data[:ϕU]
    πU = isp_data[:πU]
    yU = isp_data[:yU]
    ytsU = isp_data[:ytsU]
    λU = isp_data[:λU]
    γ_param = isp_data[:γ]
    w_param = isp_data[:w]
    v_param = isp_data[:v]
    network = isp_data[:network]
    uncertainty_set = isp_data[:uncertainty_set]
    E = isp_data[:E]
    d0 = isp_data[:d0]
    α_sol = cut_info[:α_sol]
    num_arcs = length(x_sol)
    xi_bar = uncertainty_set[:xi_bar]

    # Step 1: Build joint OSP model (fresh each call)
    osp_model, osp_vars, osp_data = build_dualized_outer_subproblem(
        network, S, ϕU, λU, γ_param, w_param, v_param, uncertainty_set, conic_optimizer,
        λ_sol, x_sol, h_sol, ψ0_sol;
        πU=πU, yU=yU, ytsU=ytsU, scaling_S=S)

    # Step 2: Fix α to IMP solution
    for k in 1:num_arcs
        fix(osp_vars[:α][k], α_sol[k]; force=true)
    end

    # Step 3: Solve at x_sol to get z*_total
    _, orig_cut_coeff = osp_optimize!(osp_model, osp_vars, osp_data, λ_sol, x_sol, h_sol, ψ0_sol)
    z_star_total = orig_cut_coeff[:obj_val]

    # Step 4: Add optimality constraint (original obj >= z*_total)
    # Reconstruct original objective expression (same as osp_optimize!)
    scaling_S_val = get(osp_data, :scaling_S, S)
    Uhat1 = osp_vars[:Uhat1]; Uhat3 = osp_vars[:Uhat3]
    Utilde1 = osp_vars[:Utilde1]; Utilde3 = osp_vars[:Utilde3]
    βhat1_1 = osp_vars[:βhat1_1]; βtilde1_1 = osp_vars[:βtilde1_1]; βtilde1_3 = osp_vars[:βtilde1_3]
    Ztilde1_3 = osp_vars[:Ztilde1_3]
    Phat1_Φ = osp_vars[:Phat1_Φ]; Phat1_Π = osp_vars[:Phat1_Π]
    Phat2_Φ = osp_vars[:Phat2_Φ]; Phat2_Π = osp_vars[:Phat2_Π]
    Ptilde1_Φ = osp_vars[:Ptilde1_Φ]; Ptilde1_Π = osp_vars[:Ptilde1_Π]
    Ptilde2_Φ = osp_vars[:Ptilde2_Φ]; Ptilde2_Π = osp_vars[:Ptilde2_Π]
    Ptilde1_Y = osp_vars[:Ptilde1_Y]; Ptilde1_Yts = osp_vars[:Ptilde1_Yts]
    Ptilde2_Y = osp_vars[:Ptilde2_Y]; Ptilde2_Yts = osp_vars[:Ptilde2_Yts]

    diag_x_sol_E = Diagonal(x_sol) * E
    diag_λ_ψ_sol = Diagonal(λ_sol * ones(num_arcs) - v_param .* ψ0_sol)

    orig_term1 = [-ϕU * sum((Uhat1[s,:,:] + Utilde1[s,:,:]) .* diag_x_sol_E) for s=1:S]
    orig_term2 = [-ϕU * sum((Uhat3[s,:,:] + Utilde3[s,:,:]) .* (E - diag_x_sol_E)) for s=1:S]
    orig_term3 = [(d0') * βhat1_1[s,:] for s=1:S]
    orig_term4 = [sum(Ztilde1_3[s,:,:] .* (diag_λ_ψ_sol * diagm(xi_bar[s]))) for s=1:S]
    orig_term5 = [(λ_sol * d0') * βtilde1_1[s,:] for s=1:S]
    orig_term6 = [-(h_sol + diag_λ_ψ_sol * xi_bar[s])' * βtilde1_3[s,:] for s=1:S]
    orig_term_ub_hat = [-ϕU * sum(Phat1_Φ[s,:,:]) - πU * sum(Phat1_Π[s,:,:]) for s=1:S]
    orig_term_lb_hat = [-ϕU * sum(Phat2_Φ[s,:,:]) - πU * sum(Phat2_Π[s,:,:]) for s=1:S]
    orig_term_ub_tilde = [-ϕU * sum(Ptilde1_Φ[s,:,:]) - πU * sum(Ptilde1_Π[s,:,:]) - yU * sum(Ptilde1_Y[s,:,:]) - ytsU * sum(Ptilde1_Yts[s,:]) for s=1:S]
    orig_term_lb_tilde = [-ϕU * sum(Ptilde2_Φ[s,:,:]) - πU * sum(Ptilde2_Π[s,:,:]) - yU * sum(Ptilde2_Y[s,:,:]) - ytsU * sum(Ptilde2_Yts[s,:]) for s=1:S]

    orig_obj_expr = (sum(orig_term1) + sum(orig_term2) + sum(orig_term3) + sum(orig_term4) +
        sum(orig_term5) + sum(orig_term6) + sum(orig_term_ub_hat) + sum(orig_term_lb_hat) +
        sum(orig_term_ub_tilde) + sum(orig_term_lb_tilde)) / scaling_S_val

    @constraint(osp_model, orig_obj_expr >= z_star_total - 1e-6)

    # Step 5: Set core-point objective
    diag_x_core_E = Diagonal(x_core) * E
    diag_λ_ψ_core = Diagonal(λ_core * ones(num_arcs) - v_param .* ψ0_core)

    core_term1 = [-ϕU * sum((Uhat1[s,:,:] + Utilde1[s,:,:]) .* diag_x_core_E) for s=1:S]
    core_term2 = [-ϕU * sum((Uhat3[s,:,:] + Utilde3[s,:,:]) .* (E - diag_x_core_E)) for s=1:S]
    core_term3 = orig_term3  # d0'*βhat1_1 — independent of master vars
    core_term4 = [sum(Ztilde1_3[s,:,:] .* (diag_λ_ψ_core * diagm(xi_bar[s]))) for s=1:S]
    core_term5 = [(λ_core * d0') * βtilde1_1[s,:] for s=1:S]
    core_term6 = [-(h_core + diag_λ_ψ_core * xi_bar[s])' * βtilde1_3[s,:] for s=1:S]

    @objective(osp_model, Max, (sum(core_term1) + sum(core_term2) + sum(core_term3) + sum(core_term4) +
        sum(core_term5) + sum(core_term6) + sum(orig_term_ub_hat) + sum(orig_term_lb_hat) +
        sum(orig_term_ub_tilde) + sum(orig_term_lb_tilde)) / scaling_S_val)

    # Step 6: Solve MW model
    optimize!(osp_model)
    st = termination_status(osp_model)
    if !(st == MOI.OPTIMAL || st == MOI.SLOW_PROGRESS)
        @warn "Joint MW OSP solve failed: $st, falling back to z*_total solution"
        # Return the original (non-strengthened) coefficients
        return Dict(:Uhat1=>orig_cut_coeff[:Uhat1], :Utilde1=>orig_cut_coeff[:Utilde1],
            :Uhat3=>orig_cut_coeff[:Uhat3], :Utilde3=>orig_cut_coeff[:Utilde3],
            :Ztilde1_3=>orig_cut_coeff[:Ztilde1_3],
            :βtilde1_1=>orig_cut_coeff[:βtilde1_1], :βtilde1_3=>orig_cut_coeff[:βtilde1_3],
            :intercept=>orig_cut_coeff[:intercept],
            :intercept_l=>orig_cut_coeff[:intercept_l], :intercept_f=>orig_cut_coeff[:intercept_f])
    end

    # Step 7: Extract coefficients (same format as osp_optimize! output)
    Uhat1_out = value.(Uhat1)
    Utilde1_out = value.(Utilde1)
    Uhat3_out = value.(Uhat3)
    Utilde3_out = value.(Utilde3)
    Ztilde1_3_out = value.(Ztilde1_3)
    βtilde1_1_out = value.(βtilde1_1)
    βtilde1_3_out = value.(βtilde1_3)

    intercept_l = value.(orig_term3) .+ value.(orig_term_ub_hat) .+ value.(orig_term_lb_hat)
    intercept_f = value.(orig_term_ub_tilde) .+ value.(orig_term_lb_tilde)
    intercept = sum(intercept_l) + sum(intercept_f)

    return Dict(:Uhat1=>Uhat1_out, :Utilde1=>Utilde1_out, :Uhat3=>Uhat3_out, :Utilde3=>Utilde3_out,
        :Ztilde1_3=>Ztilde1_3_out, :βtilde1_1=>βtilde1_1_out, :βtilde1_3=>βtilde1_3_out,
        :intercept=>intercept, :intercept_l=>intercept_l, :intercept_f=>intercept_f)
end


"""
    evaluate_sherali_opt_cut(...)

Sherali & Lunday (2011) ζ-perturbation cut 생성.
Master 변수를 ζ만큼 core point 방향으로 perturb한 뒤 ISP를 re-solve하여
ε₀-optimal maximal nondominated cut을 1회 solve로 추출.

MW 대비 장점: constraint 추가/삭제/cleanup 불필요, conic solve 1회 절약.
다음 iteration에서 ISP objective가 새 (x,λ,h,ψ0)로 갱신되므로 state 복원 불필요.

ζ 선택: additive perturbation x_pert = x_sol + ζ·x_core이므로, x_sol[i]=1인 arc에서
x_pert[i] > 1이 되면 ISP objective의 (E - diag(x)·E) 부호가 반전 → DUAL_INFEASIBLE.
ζ=1e-8은 solver가 수치적으로 무시하는 수준이라 안전. ζ ≥ 1e-7부터 unbounded 발생 확인됨.
상세: pareto_optimal_cuts.md 참조.

반환: evaluate_master_opt_cut과 동일한 Dict.
"""
function evaluate_sherali_opt_cut(
    isp_leader_instances, isp_follower_instances, isp_data, cut_info, iter;
    x_sol, λ_sol, h_sol, ψ0_sol,
    x_core, λ_core, h_core, ψ0_core,
    ζ=1e-8, parallel=false)  # ζ ≥ 1e-7 → DUAL_INFEASIBLE (see docstring)

    S = isp_data[:S]
    α_sol = cut_info[:α_sol]
    E = isp_data[:E]
    ϕU = isp_data[:ϕU]
    πU = get(isp_data, :πU, ϕU)
    yU = get(isp_data, :yU, ϕU)
    ytsU = get(isp_data, :ytsU, ϕU)
    d0 = isp_data[:d0]
    v_param = isp_data[:v]
    xi_bar = isp_data[:uncertainty_set][:xi_bar]
    num_arcs = length(x_sol)

    # 1. Perturb master variables
    x_pert = x_sol .+ ζ .* x_core
    λ_pert = λ_sol + ζ * λ_core
    h_pert = h_sol .+ ζ .* h_core
    ψ0_pert = ψ0_sol .+ ζ .* ψ0_core

    diag_x_pert_E = Diagonal(x_pert) * E
    diag_λ_ψ_pert = Diagonal(λ_pert * ones(num_arcs) - v_param .* ψ0_pert)

    # 2. Set perturbed objectives & solve ISP directly (bypass isp_leader/follower_optimize! assertions)
    solve_scenarios(S; parallel=parallel) do s
        α_s = α_sol isa Vector{<:AbstractVector} ? α_sol[s] : α_sol  # per-scenario α support
        xi_bar_s = xi_bar[s]

        # --- Leader: set objective with perturbed x ---
        model_l = isp_leader_instances[s][1]
        vars_l = isp_leader_instances[s][2]
        Uhat1_v = vars_l[:Uhat1]; Uhat3_v = vars_l[:Uhat3]
        Phat1_Φ = vars_l[:Phat1_Φ]; Phat1_Π = vars_l[:Phat1_Π]
        Phat2_Φ = vars_l[:Phat2_Φ]; Phat2_Π = vars_l[:Phat2_Π]
        βhat1_1_v = vars_l[:βhat1_1]

        @objective(model_l, Max,
            -ϕU * sum(Uhat1_v[1, :, :] .* diag_x_pert_E) +
            -ϕU * sum(Uhat3_v[1, :, :] .* (E - diag_x_pert_E)) +
            (d0') * βhat1_1_v[1, :] +
            -ϕU * sum(Phat1_Φ[1, :, :]) - πU * sum(Phat1_Π[1, :, :]) +
            -ϕU * sum(Phat2_Φ[1, :, :]) - πU * sum(Phat2_Π[1, :, :]))
        set_normalized_rhs.(vec(model_l[:coupling_cons]), α_s)
        optimize!(model_l)
        st_l = termination_status(model_l)
        if !(st_l == MOI.OPTIMAL || st_l == MOI.SLOW_PROGRESS)
            @warn "Sherali leader solve failed for s=$s: $st_l"
        end

        # --- Follower: set objective with perturbed x, λ, h, ψ0 ---
        model_f = isp_follower_instances[s][1]
        vars_f = isp_follower_instances[s][2]
        Utilde1_v = vars_f[:Utilde1]; Utilde3_v = vars_f[:Utilde3]; Ztilde1_3_v = vars_f[:Ztilde1_3]
        Ptilde1_Φ = vars_f[:Ptilde1_Φ]; Ptilde1_Π = vars_f[:Ptilde1_Π]
        Ptilde2_Φ = vars_f[:Ptilde2_Φ]; Ptilde2_Π = vars_f[:Ptilde2_Π]
        Ptilde1_Y = vars_f[:Ptilde1_Y]; Ptilde1_Yts = vars_f[:Ptilde1_Yts]
        Ptilde2_Y = vars_f[:Ptilde2_Y]; Ptilde2_Yts = vars_f[:Ptilde2_Yts]
        βtilde1_1_v = vars_f[:βtilde1_1]; βtilde1_3_v = vars_f[:βtilde1_3]

        @objective(model_f, Max,
            -ϕU * sum(Utilde1_v[1, :, :] .* diag_x_pert_E) +
            -ϕU * sum(Utilde3_v[1, :, :] .* (E - diag_x_pert_E)) +
            sum(Ztilde1_3_v[1, :, :] .* (diag_λ_ψ_pert * diagm(xi_bar_s))) +
            (λ_pert * d0') * βtilde1_1_v[1, :] +
            -(h_pert + diag_λ_ψ_pert * xi_bar_s)' * βtilde1_3_v[1, :] +
            -ϕU * sum(Ptilde1_Φ[1, :, :]) - πU * sum(Ptilde1_Π[1, :, :]) - yU * sum(Ptilde1_Y[1, :, :]) - ytsU * sum(Ptilde1_Yts[1, :]) +
            -ϕU * sum(Ptilde2_Φ[1, :, :]) - πU * sum(Ptilde2_Π[1, :, :]) - yU * sum(Ptilde2_Y[1, :, :]) - ytsU * sum(Ptilde2_Yts[1, :]))
        set_normalized_rhs.(vec(model_f[:coupling_cons]), α_s)
        optimize!(model_f)
        st_f = termination_status(model_f)
        if !(st_f == MOI.OPTIMAL || st_f == MOI.SLOW_PROGRESS)
            @warn "Sherali follower solve failed for s=$s: $st_f"
        end
        return (true, nothing)
    end

    # 3. Extract coefficients (primal variable values — same as evaluate_master_opt_cut)
    Uhat1 = cat([value.(isp_leader_instances[s][2][:Uhat1]) for s in 1:S]...; dims=1)
    Utilde1 = cat([value.(isp_follower_instances[s][2][:Utilde1]) for s in 1:S]...; dims=1)
    Uhat3 = cat([value.(isp_leader_instances[s][2][:Uhat3]) for s in 1:S]...; dims=1)
    Utilde3 = cat([value.(isp_follower_instances[s][2][:Utilde3]) for s in 1:S]...; dims=1)
    Ztilde1_3 = cat([value.(isp_follower_instances[s][2][:Ztilde1_3]) for s in 1:S]...; dims=1)
    βtilde1_1 = cat([value.(isp_follower_instances[s][2][:βtilde1_1]) for s in 1:S]...; dims=1)
    βtilde1_3 = cat([value.(isp_follower_instances[s][2][:βtilde1_3]) for s in 1:S]...; dims=1)

    intercept_l = [value.(isp_leader_instances[s][2][:intercept]) for s in 1:S]
    intercept_f = [value.(isp_follower_instances[s][2][:intercept]) for s in 1:S]
    intercept = sum(intercept_l) + sum(intercept_f)

    # Logging: perturbed obj + cut value at original point
    pert_obj = sum(objective_value(isp_leader_instances[s][1]) for s in 1:S) +
               sum(objective_value(isp_follower_instances[s][1]) for s in 1:S)
    # Evaluate Sherali cut at original (x_sol, λ_sol, h_sol, ψ0_sol)
    diag_x_sol_E = Diagonal(x_sol) * E
    diag_λ_ψ_sol = Diagonal(λ_sol * ones(num_arcs) - v_param .* ψ0_sol)
    cut_val_at_x = 0.0
    for s in 1:S
        cut_val_at_x += -ϕU * sum(Uhat1[s,:,:] .* diag_x_sol_E)
        cut_val_at_x += -ϕU * sum(Uhat3[s,:,:] .* (E - diag_x_sol_E))
        cut_val_at_x += -ϕU * sum(Utilde1[s,:,:] .* diag_x_sol_E)
        cut_val_at_x += -ϕU * sum(Utilde3[s,:,:] .* (E - diag_x_sol_E))
        cut_val_at_x += sum(Ztilde1_3[s,:,:] .* (diag_λ_ψ_sol * diagm(xi_bar[s])))
        cut_val_at_x += (d0' * βtilde1_1[s,:]) * λ_sol
        cut_val_at_x += -(h_sol + diag_λ_ψ_sol * xi_bar[s])' * βtilde1_3[s,:]
    end
    cut_val_at_x += intercept
    slack = cut_val_at_x - cut_info[:obj_val]
    if slack > 1e-3
        @error "Sherali cut INVALID: cut@x̄ > z*(x̄) + 1e-3 (slack=$slack)"
        @assert false "Invalid Sherali cut: cut@x̄=$(cut_val_at_x) > z*=$(cut_info[:obj_val])"
    elseif slack > -1e-2
        quality = "(near-optimal)"
    elseif slack > -1e-1
        quality = "(slightly loose)"
    else
        quality = "(LOOSE — weak strengthening)"
    end
    @info "  Sherali cut: ζ=$ζ, cut@x̄=$(round(cut_val_at_x, digits=4)), orig_obj=$(round(cut_info[:obj_val], digits=4)), slack=$(round(slack, digits=6)) $quality"

    return Dict(:Uhat1=>Uhat1, :Utilde1=>Utilde1, :Uhat3=>Uhat3, :Utilde3=>Utilde3,
        :Ztilde1_3=>Ztilde1_3, :βtilde1_1=>βtilde1_1, :βtilde1_3=>βtilde1_3,
        :intercept=>intercept, :intercept_l=>intercept_l, :intercept_f=>intercept_f)
end


"""
    alpha_fixed_benders_phase!(omp_model, omp_vars, α_sol_fixed,
        leader_instances, follower_instances, isp_data; kwargs...)

α를 고정시킨 상태에서 mini-Benders loop을 돌려 OMP에 추가 cuts를 생성.
tr_imp_optimize!에서 찾은 optimal α에 대해, 여러 χ 방향의 valid cuts를 한꺼번에 수집하여
lower bound tightening을 가속시킴.

Validity: OMP는 min_χ t_0 s.t. t_0 ≥ max_α Q(α, χ) 이므로,
고정 α에서의 cut t_0 ≥ Q(α_fixed, χ)는 항상 valid.
"""
function alpha_fixed_benders_phase!(
    omp_model, omp_vars, α_sol_fixed,
    leader_instances, follower_instances,
    isp_data;
    max_iter=5,
    strengthen_cuts=:none,
    conic_optimizer=nothing,
    outer_iter=0,
    result_cuts=nothing,
    parallel=false)

    S = isp_data[:S]
    ϕU = isp_data[:ϕU]
    network = isp_data[:network]
    v_param = isp_data[:v]
    num_arcs = length(network.arcs) - 1
    E = isp_data[:E]
    d0 = isp_data[:d0]
    uncertainty_set = isp_data[:uncertainty_set]
    xi_bar = uncertainty_set[:xi_bar]
    R = uncertainty_set[:R]
    r_dict = uncertainty_set[:r_dict]
    S_total = S

    x = omp_vars[:x]
    h = omp_vars[:h]
    λ_var = omp_vars[:λ]
    ψ0 = omp_vars[:ψ0]
    t_0 = omp_vars[:t_0]

    # JuMP expression matrices for cut construction
    diag_x_E = Diagonal(x) * E
    diag_λ_ψ = Diagonal(λ_var * ones(num_arcs) - v_param .* ψ0)

    # Core points for MW strengthening (generate once)
    core_points = nothing
    if strengthen_cuts != :none
        interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
        core_points = generate_core_points(network, isp_data[:γ], isp_data[:λU], isp_data[:w], v_param;
            interdictable_idx=interdictable_idx, strategy=:interior)
    end

    mini_cuts_added = 0
    mini_lb_history = Float64[]

    for j in 1:max_iter
        # 1. Re-solve OMP with accumulated cuts
        optimize!(omp_model)
        st = MOI.get(omp_model, MOI.TerminationStatus())
        if st != MOI.OPTIMAL
            @info "  [Mini-Benders] OMP status $st at iter $j, stopping."
            break
        end

        x_sol_j = round.(value.(x))
        h_sol_j = value.(h)
        λ_sol_j = value(λ_var)
        ψ0_sol_j = value.(ψ0)
        mini_lb = value(t_0) / S_total
        push!(mini_lb_history, mini_lb)

        # Check LB stagnation (2-iter window, relative improvement)
        if length(mini_lb_history) >= 3 && abs(mini_lb_history[end] - mini_lb_history[end-2]) / max(abs(mini_lb_history[end-2]), 1e-10) < 1e-4
            @info "  [Mini-Benders] LB stagnated at iter $j (LB=$(round(mini_lb, digits=6)), history=$(round.(mini_lb_history, digits=6))), stopping."
            break
        end

        # 2. Evaluate ISP at (α_fixed, χ_new) — updates objective + coupling + solves
        _, _ = solve_scenarios(S; parallel=parallel) do s
            U_s = Dict(:R => Dict(:1=>R[s]), :r_dict => Dict(:1=>r_dict[s]),
                       :xi_bar => Dict(:1=>xi_bar[s]), :epsilon => uncertainty_set[:epsilon])
            (status_l, _) = isp_leader_optimize!(leader_instances[s][1], leader_instances[s][2];
                isp_data=isp_data, uncertainty_set=U_s,
                λ_sol=λ_sol_j, x_sol=x_sol_j, h_sol=h_sol_j, ψ0_sol=ψ0_sol_j, α_sol=α_sol_fixed)
            (status_f, _) = isp_follower_optimize!(follower_instances[s][1], follower_instances[s][2];
                isp_data=isp_data, uncertainty_set=U_s,
                λ_sol=λ_sol_j, x_sol=x_sol_j, h_sol=h_sol_j, ψ0_sol=ψ0_sol_j, α_sol=α_sol_fixed)
            ok = (status_l == :OptimalityCut) && (status_f == :OptimalityCut)
            return (ok, nothing)
        end

        # 3. Extract cut coefficients (same pattern as evaluate_master_opt_cut)
        Uhat1 = cat([value.(leader_instances[s][2][:Uhat1]) for s in 1:S]...; dims=1)
        Utilde1 = cat([value.(follower_instances[s][2][:Utilde1]) for s in 1:S]...; dims=1)
        Uhat3 = cat([value.(leader_instances[s][2][:Uhat3]) for s in 1:S]...; dims=1)
        Utilde3 = cat([value.(follower_instances[s][2][:Utilde3]) for s in 1:S]...; dims=1)
        Ztilde1_3 = cat([value.(follower_instances[s][2][:Ztilde1_3]) for s in 1:S]...; dims=1)
        βtilde1_1 = cat([value.(follower_instances[s][2][:βtilde1_1]) for s in 1:S]...; dims=1)
        βtilde1_3 = cat([value.(follower_instances[s][2][:βtilde1_3]) for s in 1:S]...; dims=1)
        intercept_l = [value.(leader_instances[s][2][:intercept]) for s in 1:S]
        intercept_f = [value.(follower_instances[s][2][:intercept]) for s in 1:S]
        intercept = sum(intercept_l) + sum(intercept_f)

        mini_ub = (sum(objective_value(leader_instances[s][1]) for s in 1:S) +
                   sum(objective_value(follower_instances[s][1]) for s in 1:S)) / S

        outer_cut_info = Dict(
            :Uhat1=>Uhat1, :Utilde1=>Utilde1, :Uhat3=>Uhat3, :Utilde3=>Utilde3,
            :Ztilde1_3=>Ztilde1_3, :βtilde1_1=>βtilde1_1, :βtilde1_3=>βtilde1_3,
            :intercept=>intercept, :intercept_l=>intercept_l, :intercept_f=>intercept_f)

        # 4. Add cut to OMP
        add_optimality_cuts!(omp_model, omp_vars, outer_cut_info, diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h, S, outer_iter;
            prefix="mini_bd_$(j)", result_cuts=result_cuts)
        mini_cuts_added += 1

        # 5. MW strengthening with α-fixed
        if strengthen_cuts != :none && core_points !== nothing
            mini_cut_info = Dict(:α_sol => α_sol_fixed)
            for (cp_idx, cp) in enumerate(core_points)
                if strengthen_cuts == :mw
                    str_info = evaluate_mw_opt_cut(
                        leader_instances, follower_instances, isp_data, mini_cut_info, outer_iter;
                        x_sol=x_sol_j, λ_sol=λ_sol_j, h_sol=h_sol_j, ψ0_sol=ψ0_sol_j,
                        x_core=cp.x, λ_core=cp.λ, h_core=cp.h, ψ0_core=cp.ψ0,
                        parallel=parallel)
                elseif strengthen_cuts == :mw_joint
                    str_info = evaluate_joint_mw_opt_cut(
                        isp_data, mini_cut_info, outer_iter;
                        x_sol=x_sol_j, λ_sol=λ_sol_j, h_sol=h_sol_j, ψ0_sol=ψ0_sol_j,
                        x_core=cp.x, λ_core=cp.λ, h_core=cp.h, ψ0_core=cp.ψ0,
                        conic_optimizer=conic_optimizer)
                elseif strengthen_cuts == :sherali
                    str_info = evaluate_sherali_opt_cut(
                        leader_instances, follower_instances, isp_data, mini_cut_info, outer_iter;
                        x_sol=x_sol_j, λ_sol=λ_sol_j, h_sol=h_sol_j, ψ0_sol=ψ0_sol_j,
                        x_core=cp.x, λ_core=cp.λ, h_core=cp.h, ψ0_core=cp.ψ0,
                        parallel=parallel)
                end
                str_label = strengthen_cuts in (:mw, :mw_joint) ? "mw" : "sherali"
                add_optimality_cuts!(omp_model, omp_vars, str_info, diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h, S, outer_iter;
                    prefix="mini_bd_$(j)_$(str_label)_cp$(cp_idx)", result_cuts=result_cuts)
                mini_cuts_added += 1
            end
        end

        mini_gap = abs(mini_ub - mini_lb) / max(abs(mini_ub), 1e-10)
        @info "  [Mini-Benders] iter $j: LB=$(round(mini_lb, digits=6)), α-fixed UB=$(round(mini_ub, digits=6)), gap=$(round(mini_gap, digits=8))"

        if mini_gap <= 1e-4
            @info "  [Mini-Benders] Converged at iter $j"
            break
        end
    end

    return mini_cuts_added
end


function tr_nested_benders_optimize!(omp_model::Model, omp_vars::Dict, network, ϕU, λU, γ, w, uncertainty_set; mip_optimizer=nothing, conic_optimizer=nothing, outer_tr=true, inner_tr=true, max_outer_iter=1000, isp_mode=:dual, tol=1e-4, πU=ϕU, yU=ϕU, ytsU=ϕU, strengthen_cuts=:none, parallel=false, mini_benders=false, max_mini_benders_iter=5)
    ### -------- Trust Region 초기화 --------
    if outer_tr
        num_interdictable = sum(network.interdictable_arcs)
        max_dist = min(Int(2γ), num_interdictable) # effective diameter: 2γ (binary vectors with sum ≤ γ)
        B_bin_sequence = unique([1, ceil(Int, max_dist/4), ceil(Int, max_dist/2), max_dist])
        B_bin_stage = 1
        B_bin = B_bin_sequence[B_bin_stage]
        B_con = nothing # 나중에 생각
        # Mini-Benders 활성화 stage 설정: 마지막 N개 stage에서만 실행
        # 1 = 마지막 stage만, 2 = 마지막 2개, 0 = 전체 stage
        mini_benders_last_n = 0
        n_stages = length(B_bin_sequence)
        # Inexact inner solve 설정: N번에 1번만 exact (UB 업데이트)
        inexact_every_n = 3  # 3번에 1번 exact (1=항상 exact, 0=항상 inexact)
        use_subgradient = true  # true: inexact phase에서 IMP 대신 subgradient heuristic 사용
        subgrad_step_size = 0.1  # subgradient step size (γ₀)
        use_best_lb_cut = true  # inner loop 중 best LB 달성 시 intermediate cut 추가
        α_persistent = nothing  # subgradient heuristic용 α state (across outer iters 누적)
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
    t_0 = omp_vars[:t_0]  # always composite (sum of epigraph vars)

    num_arcs = length(network.arcs) - 1
    E = ones(num_arcs, num_arcs+1) # num_arcs × num_arcs+1 matrix of ones
    diag_x_E = Diagonal(x) * E  # diag(x)E
    diag_λ_ψ = Diagonal(λ*ones(num_arcs)-v.*ψ0)
    d0 = zeros(num_arcs + 1)
    d0[end] = 1.0
    xi_bar = uncertainty_set[:xi_bar]
    S_total = length(xi_bar)  # scenario count for /S averaging
    iter = 0
    past_obj = []
    past_major_subprob_obj = [] # major (serious) step에서 변화한 subproblem objective들만 모음
    past_minor_subprob_obj = [] # minor (null) step에서 구한 subproblem objective들 다 모음
    past_model_estimate = [] # 매 cutting plane omp의 objective 저장
    past_local_lower_bound = [] # reverse region에서 구한 local lower bound 저장
    past_upper_bound = []
    past_lower_bound = []
    past_local_optimizer = []
    past_local_center = []
    major_iter = []
    bin_B_steps = [] # B_bin이 몇번째 outer loop에서 바꼈는지 체크
    stage_just_changed = false  # stage 전환 후 첫 iteration에서 v̂ 초기화용
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
            model_estimate = value(t_0) / S_total  # average over scenarios
            lower_bound = max(lower_bound, model_estimate)
            # # ===== DEBUG: per-iteration detail =====
            # if outer_tr
            #     println("  ┌─ DEBUG iter $iter ─────────────────────────────")
            #     println("  │ x_sol arcs = $(findall(x_sol .> 0.5)),  λ=$(round(λ_sol, digits=6)),  t_0_raw=$(round(value(t_0), digits=6))")
            #     println("  │ model_estimate (t_0/S) = $(round(model_estimate, digits=6))")
            #     if centers[:x] !== nothing
            #         println("  │ center arcs = $(findall(round.(centers[:x]) .> 0.5)),  ||x-center||₁ = $(Int(sum(abs.(x_sol .- round.(centers[:x])))))")
            #     end
            # end
            # Update primal ISP parameters (x,h,λ,ψ0 are in constraint RHS → set_normalized_rhs)
            if isp_mode != :dual
                update_primal_isp_parameters!(primal_leader_instances, primal_follower_instances;
                    x_sol=x_sol, h_sol=h_sol, λ_sol=λ_sol, ψ0_sol=ψ0_sol, isp_data=isp_data)
            end
            # Outer Subproblem 풀기 (inexact: N번에 1번만 exact)
            # 첫 iter과 stage 전환 직후는 항상 exact (v̂ 초기화 필요)
            use_inexact = inexact_every_n > 1 && (iter % inexact_every_n != 0) && iter > 1 && !stage_just_changed
            if use_inexact && use_subgradient && isp_mode == :dual
                # Subgradient heuristic: IMP 없이 ISP 1라운드 + α projection
                if α_persistent === nothing
                    α_persistent = fill(isp_data[:w] / isp_data[:S] / (length(network.arcs)-1), length(network.arcs)-1)
                end
                status, cut_info = subgradient_alpha_step!(α_persistent,
                    leader_instances, follower_instances, isp_data;
                    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                    step_size=subgrad_step_size, parallel=parallel)
                α_persistent = cut_info[:α_new]  # 다음 iteration용 α 누적
            elseif isp_mode == :dual
                status, cut_info = tr_imp_optimize!(imp_model, imp_vars, leader_instances, follower_instances; isp_data=isp_data, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, outer_iter=iter, imp_cuts=imp_cuts, inner_tr=inner_tr, parallel=parallel, inexact=use_inexact)
            else
                status, cut_info = tr_imp_optimize_hybrid!(imp_model, imp_vars, primal_leader_instances, primal_follower_instances; isp_data=isp_data, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, outer_iter=iter, imp_cuts=imp_cuts, inner_tr=inner_tr, parallel=parallel)
            end
            if status != :OptimalityCut
                @warn "Outer Subproblem not optimal"
                @infiltrate
            end
            push!(result[:inner_iter], cut_info[:iter])
            subprob_obj = cut_info[:obj_val]
            is_inexact = get(cut_info, :inexact, false)
            is_subgradient = get(cut_info, :subgradient, false)
            if !is_subgradient
                imp_cuts[:old_cuts] = cut_info[:cuts] ## 다음 iteration에서 지우기 위해 여기에 저장함
            end
            if inner_tr && !is_subgradient && cut_info[:tr_constraints] !== nothing
                imp_cuts[:old_tr_constraints] = cut_info[:tr_constraints]
            end
            if !is_inexact
                # Exact solve만 UB 업데이트 (inexact α는 suboptimal → valid UB 아님)
                upper_bound = min(upper_bound, subprob_obj)
                local_upper_bound = min(local_upper_bound, subprob_obj)
                # Exact IMP 수렴 후 α_persistent 재보정 (subgradient drift 교정)
                if use_subgradient
                    α_persistent = copy(cut_info[:α_sol])
                end
            end

            # Measure of progress
            gap = abs(local_upper_bound - lower_bound) / max(abs(local_upper_bound), 1e-10)
            if outer_tr
                # ===== DEBUG: subprob + bounds =====
                println("  │ subprob_obj = $(round(subprob_obj, digits=6)),  inner_iters = $(cut_info[:iter])$(is_inexact ? " (inexact)" : "")")
                println("  │ LB = $(round(lower_bound, digits=6)),  localUB = $(round(local_upper_bound, digits=6)),  globalUB = $(round(upper_bound, digits=6)),  gap = $(round(gap, digits=8))")

                is_last_stage = (B_bin_stage == length(B_bin_sequence))
                if !is_inexact && !is_last_stage
                    # Exact solve에서만 v̂ 초기화 및 SS 판정 (마지막 stage 제외: B_bin이 전체 도메인 포함)
                    if iter==1 || stage_just_changed
                        # 첫 iteration 또는 stage 전환 직후: v̂를 현재 subprob_obj로 초기화
                        # (논문 Algorithm 3의 v̂^1 = ∞ 대신, iter==1과 동일하게 첫 해로 설정)
                        push!(past_major_subprob_obj, subprob_obj)
                        stage_just_changed = false
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
                        # 논문 Algorithm 3: center가 바뀌면 새 TR에서 LB 재계산 필요
                        lower_bound = -Inf
                    end
                    # ===== DEBUG: serious/null step =====
                    println("  │ past_major_subprob_obj[end] = $(round(past_major_subprob_obj[end], digits=6)),  improvement = $(round(improvement, digits=6)),  β_dynamic = $(round(β_dynamic, digits=8))")
                    println("  │ $(is_serious_step ? "★ SERIOUS STEP" : "  null step")")
                else
                    # Inexact solve or last stage: SS skip, only add cuts
                    tr_needs_update = false
                    is_serious_step = false
                    if is_inexact
                        println("  │ (inexact solve — SS skip)")
                    elseif is_last_stage
                        println("  │ (last stage — SS skip)")
                    end
                end
            end
            if outer_tr
                @info "[Outer-$isp_mode] Stage $(B_bin_stage)/$(length(B_bin_sequence)) Iter $iter: localLB=$(round(lower_bound, digits=4))  localUB=$(round(local_upper_bound, digits=4))  localGap=$(round(gap, digits=6))  (globalUB=$(round(upper_bound, digits=4)); $(round(time()-time_start, digits=1))s)"
            else
                @info "[Outer-$isp_mode] Iter $iter: LB=$(round(lower_bound, digits=4))  UB=$(round(upper_bound, digits=4))  Gap=$(round(gap, digits=6))  ($(round(time()-time_start, digits=1))s)"
            end
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
        # 논문 Algorithm 3: convergence check는 SS 이전 (Step 2 < Step 3)
        # SS와 convergence 동시 발생 방지: SS 발생 시 새 center 주변 재탐색 필요
        is_ss_this_iter = outer_tr && @isdefined(is_serious_step) && is_serious_step
        # 이전 stage는 loose tol (1e-3), 마지막 stage만 tight tol
        local_tol = (outer_tr && B_bin_stage < length(B_bin_sequence)) ? 1e-3 : tol
        converged = (gap <= local_tol || lower_bound > local_upper_bound - 1e-4) && !is_ss_this_iter
        if outer_tr && (converged || pruned)
            println("  │ ▶ CONVERGED/PRUNED: gap=$(round(gap, digits=8)), LB=$(round(lower_bound, digits=6)) vs localUB=$(round(local_upper_bound, digits=6))")
            println("  └──────────────────────────────────────────────")
        end
        if converged || pruned
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
                push!(past_local_center, copy(centers[:x]))
                @info "  ✓ Local optimal reached! Expanding B_bin to $B_bin"
                println("  ★★ STAGE $(B_bin_stage-1) → $(B_bin_stage) 전환 ★★")
                println("  localLB = $lower_bound,  localUB = $local_upper_bound,  globalUB = $upper_bound")
                println("  center = $(findall(round.(centers[:x]) .> 0.5))")
                println("  past_model_estimate = $(round.(past_model_estimate, digits=4))")
                println("  past_minor_subprob_obj = $(round.(past_minor_subprob_obj, digits=4))")
                # TR constraint needs update (B_bin changed)
                tr_needs_update = true
                @info "Updating Trust Region"
                ## trust region radius를 확장
                tr_constraints = update_outer_trust_region_constraints!(
                    omp_model, omp_vars, centers, B_bin, B_con, tr_constraints, network)
                lower_bound = -Inf # master problem 영역 확장했으니까 다시 초기화
                local_upper_bound = Inf # local UB도 리셋
                stage_just_changed = true  # 다음 iteration에서 v̂ 초기화
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
                # 마지막 stage 결과도 push
                push!(bin_B_steps, iter)
                push!(past_local_lower_bound, lower_bound)
                push!(past_local_optimizer, Dict(:x=>value.(x_sol), :h=>value.(h_sol), :λ=>value.(λ_sol), :ψ0=>value.(ψ0_sol)))
                push!(past_local_center, copy(centers[:x]))
                # past_local_lower_bound 배열에서 최소값의 인덱스를 찾음
                min_idx = argmin(past_local_lower_bound)
                global_lower_bound = past_local_lower_bound[min_idx]
                iter_when_global_optimal = bin_B_steps[min_idx]
                global_upper_bound = past_upper_bound[iter_when_global_optimal]
                println("lower_bound: ", global_lower_bound, ", upper_bound: ", global_upper_bound)

                result[:past_lower_bound] = past_lower_bound
                result[:past_local_lower_bound] = past_local_lower_bound
                result[:past_local_center] = past_local_center
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
            if is_subgradient
                # Subgradient heuristic: ISP 이미 풀었으므로 re-solve 불필요
                outer_cut_info = cut_info[:outer_cut_info]
            elseif isp_mode == :full_primal
                outer_cut_info = evaluate_master_opt_cut_from_primal(
                    primal_leader_instances, primal_follower_instances,
                    isp_data, cut_info, iter;
                    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)
            elseif isp_mode == :hybrid
                outer_cut_info = primal_evaluate_master_opt_cut(
                    leader_instances, follower_instances,
                    isp_data, cut_info, iter;
                    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)
            else
                outer_cut_info = evaluate_master_opt_cut(leader_instances, follower_instances, isp_data, cut_info, iter, parallel=parallel)
            end
            # Debug logging
            push!(result[:debug_α], cut_info[:α_sol])
            push!(result[:debug_intercept_l], sum(outer_cut_info[:intercept_l]))
            push!(result[:debug_intercept_f], sum(outer_cut_info[:intercept_f]))
            push!(result[:debug_coeff_norms], Dict(
                :Uhat1 => norm(outer_cut_info[:Uhat1]),
                :Utilde1 => norm(outer_cut_info[:Utilde1]),
                :Uhat3 => norm(outer_cut_info[:Uhat3]),
                :Utilde3 => norm(outer_cut_info[:Utilde3]),
                :βtilde1_1 => norm(outer_cut_info[:βtilde1_1]),
                :βtilde1_3 => norm(outer_cut_info[:βtilde1_3]),
                :Ztilde1_3 => norm(outer_cut_info[:Ztilde1_3]),
            ))
            opt_cut = add_optimality_cuts!(omp_model, omp_vars, outer_cut_info, diag_x_E, isp_data[:E], diag_λ_ψ, xi_bar, isp_data[:d0], ϕU, λ, h, S, iter;
                prefix="opt_cut", result_cuts=result[:cuts])
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
            opt_cut_val_at_xsol = evaluate_expr(opt_cut, y)
            if abs(subprob_obj * S_total - opt_cut_val_at_xsol) > 1e-3  # opt_cut is raw, subprob_obj is /S
                println("something went wrong")
                @infiltrate
            end
            if outer_tr
                println("  │ opt_cut at x_sol = $(round(opt_cut_val_at_xsol, digits=6)) (should ≈ subprob*S = $(round(subprob_obj * S_total, digits=6)))")
            end
            println("subproblem objective: ", subprob_obj)
            @info "Optimality cut added"

            # ===== Best-LB intermediate cut (inner loop 중 best LB 달성 시의 cut) =====
            if use_best_lb_cut && haskey(cut_info, :best_lb_outer_cut) && cut_info[:best_lb_outer_cut] !== nothing
                best_lb_cut_info = cut_info[:best_lb_outer_cut]
                best_lb_cut = add_optimality_cuts!(omp_model, omp_vars, best_lb_cut_info, diag_x_E, isp_data[:E], diag_λ_ψ, xi_bar, isp_data[:d0], ϕU, λ, h, S, iter;
                    prefix="best_lb_cut", result_cuts=result[:cuts])
                best_lb_cut_val = evaluate_expr(best_lb_cut, y)
                println("  │ [Best-LB cut] val at x_sol = $(round(best_lb_cut_val, digits=6))")
            end

            # ===== Cut Strengthening (:mw or :sherali) =====
            if strengthen_cuts != :none && leader_instances !== nothing
                interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
                core_points = generate_core_points(network, γ, λU, w, v;
                    interdictable_idx=interdictable_idx, strategy=:interior)
                for (cp_idx, cp) in enumerate(core_points)
                    if strengthen_cuts == :mw
                        str_info = evaluate_mw_opt_cut(
                            leader_instances, follower_instances, isp_data, cut_info, iter;
                            x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                            x_core=cp.x, λ_core=cp.λ, h_core=cp.h, ψ0_core=cp.ψ0,
                            parallel=parallel)
                    elseif strengthen_cuts == :mw_joint
                        str_info = evaluate_joint_mw_opt_cut(
                            isp_data, cut_info, iter;
                            x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                            x_core=cp.x, λ_core=cp.λ, h_core=cp.h, ψ0_core=cp.ψ0,
                            conic_optimizer=conic_optimizer)
                    elseif strengthen_cuts == :sherali
                        str_info = evaluate_sherali_opt_cut(
                            leader_instances, follower_instances, isp_data, cut_info, iter;
                            x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                            x_core=cp.x, λ_core=cp.λ, h_core=cp.h, ψ0_core=cp.ψ0,
                            parallel=parallel)
                    end
                    str_label = strengthen_cuts in (:mw, :mw_joint) ? "mw" : "sherali"
                    str_cut = add_optimality_cuts!(omp_model, omp_vars, str_info, diag_x_E, isp_data[:E], diag_λ_ψ, xi_bar, isp_data[:d0], ϕU, λ, h, S, iter;
                        prefix="$(str_label)_cut_cp$(cp_idx)", result_cuts=result[:cuts])
                    # MW/Sherali cut validation: x_sol에서 평가하여 subprob_obj*S 초과 여부 확인
                    str_cut_val = evaluate_expr(str_cut, y)
                    if outer_tr
                        println("  │ $(str_label)_cut_cp$(cp_idx) at x_sol = $(round(str_cut_val, digits=6))  (diff from opt = $(round(str_cut_val - opt_cut_val_at_xsol, digits=6)))")
                    end
                    # if str_cut_val > subprob_obj * S_total + 1e-3
                    #     @warn "  ⚠ INVALID $(str_label) cut cp$(cp_idx) at iter $iter: cut_val=$(round(str_cut_val, digits=6)) > subprob*S=$(round(subprob_obj * S_total, digits=6))  (excess=$(round(str_cut_val - subprob_obj * S_total, digits=6)))"
                    #     @infiltrate
                    # end
                end
                @info "  $(length(core_points)) $(strengthen_cuts) strengthening cuts added"
            end

            # Update TR constraints if needed
            if outer_tr && tr_needs_update
                @info "Updating Trust Region"
                tr_constraints = update_outer_trust_region_constraints!(
                    omp_model, omp_vars, centers, B_bin, B_con, tr_constraints, network)
                println("  │ TR updated: center=$(findall(round.(centers[:x]) .> 0.5)), B_bin=$B_bin")
            end

            # ===== Mini-Benders phase (α-fixed, exact solve에서만) =====
            # inexact_every_n > 1: exact iteration마다 항상 mini-benders (stage 무관)
            # inexact_every_n = 1: mini_benders_last_n 기준으로 stage 제한
            mini_benders_stage_ok = inexact_every_n > 1 ||
                !outer_tr || mini_benders_last_n == 0 || B_bin_stage > n_stages - mini_benders_last_n
            mini_benders_active = mini_benders && !is_inexact && leader_instances !== nothing && mini_benders_stage_ok
            if mini_benders_active
                n_mini = alpha_fixed_benders_phase!(
                    omp_model, omp_vars, cut_info[:α_sol],
                    leader_instances, follower_instances,
                    isp_data;
                    max_iter=max_mini_benders_iter,
                    strengthen_cuts=strengthen_cuts,
                    conic_optimizer=conic_optimizer,
                    outer_iter=iter,
                    result_cuts=result[:cuts],
                    parallel=parallel)
                if outer_tr
                    println("  │ [Mini-Benders] $n_mini extra cuts added (α-fixed, max_iter=$max_mini_benders_iter)")
                else
                    @info "[Mini-Benders] $n_mini extra cuts added (α-fixed)"
                end
            end

            if outer_tr
                println("  └──────────────────────────────────────────────")
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
    mip_optimizer=nothing, conic_optimizer=nothing, outer_tr=true, inner_tr=true, max_outer_iter=1000, full_primal=false, tol=1e-4, πU=ϕU, yU=ϕU, ytsU=ϕU, strengthen_cuts=:none)

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
    t_0 = omp_vars[:t_0]  # always composite (sum of epigraph vars)

    num_arcs = length(network.arcs) - 1
    E = ones(num_arcs, num_arcs+1)
    diag_x_E = Diagonal(x) * E
    diag_λ_ψ = Diagonal(λ*ones(num_arcs)-v.*ψ0)
    d0 = zeros(num_arcs + 1)
    d0[end] = 1.0
    xi_bar_local = uncertainty_set[:xi_bar]
    S_total = length(xi_bar_local)  # scenario count for /S averaging
    iter = 0
    past_obj = []
    past_major_subprob_obj = []
    past_minor_subprob_obj = []
    past_model_estimate = []
    past_local_lower_bound = []
    past_upper_bound = []
    past_lower_bound = []
    past_local_optimizer = []
    past_local_center = []
    major_iter = []
    bin_B_steps = []
    stage_just_changed = false  # stage 전환 후 첫 iteration에서 v̂ 초기화용
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
            model_estimate = value(t_0) / S_total  # average over scenarios
            lower_bound = max(lower_bound, model_estimate)

            # Update primal ISP parameters (x,h,λ,ψ0 in constraint RHS → set_normalized_rhs)
            update_primal_isp_parameters!(primal_leader_instances, primal_follower_instances;
                x_sol=x_sol, h_sol=h_sol, λ_sol=λ_sol, ψ0_sol=ψ0_sol, isp_data=isp_data)

            # Hybrid inner loop (primal ISP)
            status, cut_info = tr_imp_optimize_hybrid!(imp_model, imp_vars,
                primal_leader_instances, primal_follower_instances;
                isp_data=isp_data, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                outer_iter=iter, imp_cuts=imp_cuts, inner_tr=inner_tr, tol=tol, parallel=parallel)

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
                if iter==1 || stage_just_changed
                    push!(past_major_subprob_obj, subprob_obj)
                    stage_just_changed = false
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
                    # 논문 Algorithm 3: center가 바뀌면 새 TR에서 LB 재계산 필요
                    lower_bound = -Inf
                end
            end
            if outer_tr
                @info "[Outer-$(full_primal ? "FullPrimal" : "Hybrid")] Stage $(B_bin_stage)/$(length(B_bin_sequence)) Iter $iter: localLB=$(round(lower_bound, digits=4))  localUB=$(round(local_upper_bound, digits=4))  localGap=$(round(gap, digits=6))  (globalUB=$(round(upper_bound, digits=4)); $(round(time()-time_start, digits=1))s)"
            else
                @info "[Outer-$(full_primal ? "FullPrimal" : "Hybrid")] Iter $iter: LB=$(round(lower_bound, digits=4))  UB=$(round(upper_bound, digits=4))  Gap=$(round(gap, digits=6))  ($(round(time()-time_start, digits=1))s)"
            end
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
        # 논문 Algorithm 3: SS와 convergence 동시 발생 방지
        is_ss_this_iter = outer_tr && @isdefined(is_serious_step) && is_serious_step
        local_tol = (outer_tr && B_bin_stage < length(B_bin_sequence)) ? 1e-3 : tol
        converged = (gap <= local_tol || lower_bound > local_upper_bound - 1e-4) && !is_ss_this_iter
        if converged || pruned
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
                push!(past_local_center, copy(centers[:x]))
                @info "  ✓ Local optimal reached! Expanding B_bin to $B_bin"
                tr_needs_update = true
                @info "Updating Trust Region"
                tr_constraints = update_outer_trust_region_constraints!(
                    omp_model, omp_vars, centers, B_bin, B_con, tr_constraints, network)
                lower_bound = -Inf
                local_upper_bound = Inf  # local UB도 리셋
                stage_just_changed = true  # 다음 iteration에서 v̂ 초기화
                _ = add_reverse_region_constraint!(omp_model, omp_vars[:x], centers[:x], B_bin_old, network)
            else
                time_end = time()
                result[:solution_time] = time_end - time_start
                @info "  ✓✓ GLOBAL OPTIMAL (hybrid)! (B_bin = full region)"
                # 마지막 stage 결과도 push
                push!(bin_B_steps, iter)
                push!(past_local_lower_bound, lower_bound)
                push!(past_local_optimizer, Dict(:x=>value.(x_sol), :h=>value.(h_sol), :λ=>value.(λ_sol), :ψ0=>value.(ψ0_sol)))
                push!(past_local_center, copy(centers[:x]))
                min_idx = argmin(past_local_lower_bound)
                global_lower_bound = past_local_lower_bound[min_idx]
                iter_when_global_optimal = bin_B_steps[min_idx]
                global_upper_bound = past_upper_bound[iter_when_global_optimal]
                println("lower_bound: ", global_lower_bound, ", upper_bound: ", global_upper_bound)

                result[:past_lower_bound] = past_lower_bound
                result[:past_local_lower_bound] = past_local_lower_bound
                result[:past_local_center] = past_local_center
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
                outer_cut_info = evaluate_master_opt_cut_from_primal(
                    primal_leader_instances, primal_follower_instances,
                    isp_data, cut_info, iter;
                    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)
            else
                outer_cut_info = primal_evaluate_master_opt_cut(
                    dual_leader_instances, dual_follower_instances,
                    isp_data, cut_info, iter;
                    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)
            end

            opt_cut = add_optimality_cuts!(omp_model, omp_vars, outer_cut_info, diag_x_E, isp_data[:E], diag_λ_ψ, xi_bar_local, isp_data[:d0], ϕU, λ, h, S, iter;
                prefix="opt_cut", result_cuts=result[:cuts])

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
            if abs(subprob_obj * S_total - evaluate_expr(opt_cut, y)) > 1e-3  # opt_cut is raw, subprob_obj is /S
                println("something went wrong (hybrid)")
                @infiltrate
            end
            println("subproblem objective (hybrid): ", subprob_obj)
            @info "Optimality cut added (hybrid)"

            # ===== Cut Strengthening (hybrid) =====
            if strengthen_cuts != :none && dual_leader_instances !== nothing
                interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
                core_points = generate_core_points(network, γ, λU, w, v;
                    interdictable_idx=interdictable_idx, strategy=:interior)
                for (cp_idx, cp) in enumerate(core_points)
                    if strengthen_cuts == :mw
                        str_info = evaluate_mw_opt_cut(
                            dual_leader_instances, dual_follower_instances, isp_data, cut_info, iter;
                            x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                            x_core=cp.x, λ_core=cp.λ, h_core=cp.h, ψ0_core=cp.ψ0,
                            parallel=parallel)
                    elseif strengthen_cuts == :mw_joint
                        str_info = evaluate_joint_mw_opt_cut(
                            isp_data, cut_info, iter;
                            x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                            x_core=cp.x, λ_core=cp.λ, h_core=cp.h, ψ0_core=cp.ψ0,
                            conic_optimizer=conic_optimizer)
                    elseif strengthen_cuts == :sherali
                        str_info = evaluate_sherali_opt_cut(
                            dual_leader_instances, dual_follower_instances, isp_data, cut_info, iter;
                            x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                            x_core=cp.x, λ_core=cp.λ, h_core=cp.h, ψ0_core=cp.ψ0,
                            parallel=parallel)
                    end
                    str_label = strengthen_cuts in (:mw, :mw_joint) ? "mw" : "sherali"
                    add_optimality_cuts!(omp_model, omp_vars, str_info, diag_x_E, isp_data[:E], diag_λ_ψ, xi_bar_local, isp_data[:d0], ϕU, λ, h, S, iter;
                        prefix="$(str_label)_cut_cp$(cp_idx)", result_cuts=result[:cuts])
                end
                @info "  $(length(core_points)) $(strengthen_cuts) strengthening cuts added (hybrid)"
            end

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
        # NOTE: unary - 파싱 버그 수정 — 연산자를 첫째 줄 끝에 배치
        lhs_L = Adj_L_Mhat_11+Adj_L_Mhat_12 + Uhat2[s,:,1:num_arcs] - Uhat3[s,:,1:num_arcs] -
            I_0*Zhat1_1[s,:,:] - Zhat1_3[s,:,:] + Zhat2[s,:,:] + Phat1_Φ[s,:,1:num_arcs] - Phat2_Φ[s,:,1:num_arcs]

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
        # NOTE: unary - 파싱 버그 수정 — 연산자를 첫째 줄 끝에 배치
        lhs_L = Adj_L_Mtilde_11+Adj_L_Mtilde_12 + Utilde2[s,:,1:num_arcs] - Utilde3[s,:,1:num_arcs] -
            I_0*Ztilde1_1[s,:,:] - Ztilde1_5[s,:,:] + Ztilde2[s,:,:] + Ptilde1_Φ[s,:,1:num_arcs] - Ptilde2_Φ[s,:,1:num_arcs]
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