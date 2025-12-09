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
function imp_optimize!(network, S, ϕU, λU, γ, w, v, uncertainty_set, optimizer, λ_sol, x_sol, h_sol, ψ0_sol)
    
    imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set, optimizer, λ_sol, x_sol, h_sol, ψ0_sol)
    optimize!(imp_model)
    @infiltrate
    α_sol = value.(imp_vars[:α])
    st = MOI.get(imp_model, MOI.TerminationStatus())
    result = Dict()
    result[:imp_obj] = []

    while (st == MOI.DUAL_INFEASIBLE || st == MOI.OPTIMAL)
        iter += 1
        @info "Innermost Benders Iteration $iter"
        optimize!(omp_model)
        st = MOI.get(omp_model, MOI.TerminationStatus())
        α_sol = value.(imp_vars[:α])
        t_1_sol = value(imp_vars[:t_1])

        (status, cut_info) =isp_optimize!(network, S, ϕU, λU, γ, w, v, uncertainty_set, optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, t_1_sol)
    end
end

function build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set,optimizer, λ, x, h, ψ0)
    num_arcs = length(network.arcs) - 1
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    S = length(xi_bar)
    flow_upper = sum(sum(xi_bar[s] for s in 1:S))
    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => false))
    @variable(model, t_1[s=1:S], upper_bound= flow_upper)
    @variable(model, α[k=1:num_arcs] >= 0)
    @constraint(model, sum(α) <= w*(1/S))
    @objective(model, Max, sum(t_1))

    vars = Dict(
        :t_1 => t_1,
        :α => α
    )
    return model, vars
end


function isp_optimize!(network, S, ϕU, λU, γ, w, v, uncertainty_set, optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, t_1_sol)
    isp_model, isp_vars = build_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set, optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, t_1_sol)
end


function nested_benders_optimize!(network, S, ϕU, λU, γ, w, v, uncertainty_set, optimizer, λ_sol, x_sol, h_sol, ψ0_sol)
    ### --------Begin Initialization--------
    st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars, network, ϕU, λU, γ, w, uncertainty_set; optimizer=optimizer)
    x, h, λ, ψ0, t_0 = omp_vars[:x], omp_vars[:h], omp_vars[:λ], omp_vars[:ψ0], omp_vars[:t_0]
    num_arcs = length(network.arcs) - 1
    diag_x_E = Diagonal(x) * osp_data[:E]  # diag(x)E
    diag_λ_ψ = Diagonal(λ*ones(num_arcs)-v.*ψ0)
    xi_bar = uncertainty_set[:xi_bar]
    iter = 0


    imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set, optimizer, λ_sol, x_sol, h_sol, ψ0_sol)
    isp_model, isp_vars = build_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set, optimizer, λ_sol, x_sol, h_sol, ψ0_sol, α_sol, t_1_sol)
    
    past_obj = []
    subprob_obj = []
    result[:cuts] = Dict()
    ### --------End Initialization--------
    while (st == MOI.DUAL_INFEASIBLE || st == MOI.OPTIMAL)
        iter += 1
    end