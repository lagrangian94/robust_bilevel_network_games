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
includet("sdp_build_dualized_outer_subproblem.jl")
using .NetworkGenerator


function build_rmp(network, ϕU, γ, w, uncertainty_set; optimizer=nothing)
    # Extract network dimensions
    num_nodes = length(network.nodes)
    num_arcs = length(network.arcs)-1 #dummy arc 제외
    num_interdictable = sum(network.interdictable_arcs)
    
    # Node-arc incidence matrix (excluding source row)
    N = network.N
    R, r_dict, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:epsilon]
    # Dummy arc index (t,s)
    dummy_arc_idx = findfirst(arc -> arc == ("t", "s"), network.arcs)
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
    # mccormick envelope constraints for ψ0
    λU = ϕU # 100  # Upper bound on λ (should be set based on problem)
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

function subprob_optimize!(osp_model::Model, osp_vars::Dict, osp_data::Dict, λ_sol, x_sol, h_sol, ψ0_sol)
    E = osp_data[:E]
    v = osp_data[:v]
    ϕU = osp_data[:ϕU]
    S = osp_data[:S]
    d0 = osp_data[:d0]

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
    diag_λ_ψ = Diagonal(λ_sol .-v.*ψ0_sol)
    # s에 대해 summing이 필요하다면 sum over s 추가
    # matrix inner product: sum(M .* N)
    obj_term1 = [-ϕU * sum((Uhat1[s, :, :] + Utilde1[s, :, :]) .* diag_x_E) for s=1:S]
    obj_term2 = [-ϕU * sum((Uhat3[s, :, :] + Utilde3[s, :, :]) .* (E-diag_x_E)) for s=1:S]
    obj_term3 = [(d0')* βhat1_1[s,:] for s=1:S] #이거만 maximize하면 dual infeasible
    obj_term4 = [sum(Ztilde1_3[s, :, :] .* diag_λ_ψ) for s=1:S]
    obj_term5 = [(λ_sol*d0')* βtilde1_1[s,:] for s=1:S] #이거만 maximize하면 dual infeasible
    obj_term6 = [-h_sol'* βtilde1_3[s,:] for s=1:S]

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
        x_coeff1 = [-ϕU * value.(Uhat1[s, :, :] + Utilde1[s, :, :]) for s=1:S]
        x_coeff2 = [-ϕU * value.(Uhat3[s, :, :] + Utilde3[s, :, :]) for s=1:S]
        λ_coeff1 = [value.(Ztilde1_3[s, :, :] ) for s=1:S]
        λ_coeff2 = [d0'* value.(βtilde1_1[s,:]) for s=1:S]
        h_coeff = [-value.(βtilde1_3[s,:]) for s=1:S]
        constant = value.(obj_term3) .+ value.(obj_term_ub_hat) .+ value.(obj_term_lb_hat) .+ value.(obj_term_ub_tilde) .+ value.(obj_term_lb_tilde)
        return (:OptimalityCut, Dict(:x_coeff1 => x_coeff1, :x_coeff2 => x_coeff2, :λ_coeff1 => λ_coeff1, :λ_coeff2 => λ_coeff2, :h_coeff => h_coeff,
         :constant => constant, :obj_val => obj_val))
    else
        @infiltrate
        error("Subproblem is not optimal")
    end
end

function benders_optimize!(rmp_model::Model, rmp_vars::Dict, network, ϕU, γ, w, uncertainty_set; optimizer=nothing)
    # Solve restricted master problem
    optimize!(rmp_model)
    st = MOI.get(rmp_model, MOI.TerminationStatus())
    # restricted master has a solution or is unbounded
    λ_sol, x_sol, h_sol, ψ0_sol = value(rmp_vars[:λ]), value.(rmp_vars[:x]), value.(rmp_vars[:h]), value.(rmp_vars[:ψ0])
    x, h, λ, ψ0, t_0 = rmp_vars[:x], rmp_vars[:h], rmp_vars[:λ], rmp_vars[:ψ0], rmp_vars[:t_0]
    nopt_cons, nfeas_cons = (0,0)
    cuts = Dict(:x_coeff1 => [], :x_coeff2 => [], :λ_coeff1 => [], :λ_coeff2 => [], :h_coeff => [], :constant => [])
    @info "Initial status $st"
    osp_model, osp_vars, osp_data = build_dualized_outer_subproblem(network, S, ϕU, γ, w, v, uncertainty_set, λ_sol, x_sol, h_sol, ψ0_sol, optimizer=Mosek.Optimizer)
    diag_x_E = Diagonal(x) * osp_data[:E]  # diag(x)E
    diag_λ_ψ = Diagonal(λ .-v.*ψ0)
    iter = 0
    while (st == MOI.DUAL_INFEASIBLE || st == MOI.OPTIMAL)
        iter += 1
        @info "Iteration $iter"
        optimize!(rmp_model)
        st = MOI.get(rmp_model, MOI.TerminationStatus())
        x_sol, h_sol, λ_sol, ψ0_sol = value.(rmp_vars[:x]), value.(rmp_vars[:h]), value(rmp_vars[:λ]), value.(rmp_vars[:ψ0])
        t_0_sol = value(rmp_vars[:t_0])
        (status, cut_info) =subprob_optimize!(osp_model, osp_vars, osp_data, λ_sol, x_sol, h_sol, ψ0_sol)
        if status == :OptimalityCut
            @info "Optimality cut added"
            if t_0_sol >= cut_info[:obj_val]
                break
            else
                nopt_cons +=1
                cut_1 =  [sum(cut_info[:x_coeff1][s] .* diag_x_E) for s in 1:S]
                cut_2 =  [sum(cut_info[:x_coeff2][s] .* (osp_data[:E] - diag_x_E)) for s in 1:S]
                cut_3 =  [sum(cut_info[:λ_coeff1][s] .* diag_λ_ψ) for s in 1:S]
                cut_4 =  [cut_info[:λ_coeff2][s] * λ for s in 1:S]
                cut_5 =  [cut_info[:h_coeff][s]'* h for s in 1:S]
                cut_const = [cut_info[:constant][s] for s in 1:S]
                @constraint(rmp_model, t_0 >= sum(cut_1)+ sum(cut_2)+ sum(cut_3)+ sum(cut_4)+ sum(cut_5)+ sum(cut_const))
            end

        end
        # push!(cuts[:x_coeff1], cut_info[:x_coeff1])
        # push!(cuts[:x_coeff2], cut_info[:x_coeff2])
        # push!(cuts[:λ_coeff1], cut_info[:λ_coeff1])
        # push!(cuts[:λ_coeff2], cut_info[:λ_coeff2])
        # push!(cuts[:h_coeff], cut_info[:h_coeff])
        # push!(cuts[:constant], cut_info[:constant])

    end
end