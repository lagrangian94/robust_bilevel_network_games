"""
Debug: ytsU 민감도 테스트.
Full model의 obj가 ytsU bound에 의존하는지 확인.
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using LinearAlgebra
using Pajarito
using Revise

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("build_full_model.jl")
includet("strict_benders.jl")

using .NetworkGenerator: generate_capacity_scenarios_uniform_model
using .NetworkGenerator: generate_polska_network, print_realworld_network_summary

# ===== Setup =====
S = 2
seed = 42
epsilon = 0.5
γ_ratio = 0.10
ρ = 0.2
v = 1.0
network = generate_polska_network()
print_realworld_network_summary(network)
num_arcs = length(network.arcs) - 1
ϕU = 1/epsilon
λU = ϕU
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, γ_ratio * num_interdictable)

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = round(ρ * γ * c_bar, digits=4)

capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

source_arc_idx = findall(a -> a[1] == "s", network.arcs[1:num_arcs])
max_flow_ub = maximum(sum(capacities[source_arc_idx, s] for s in 1:S) / S)
max_cap = maximum(capacity_scenarios_regular)
πU = ϕU
yU = min(max_cap, ϕU)
ytsU_default = min(max_flow_ub, ϕU)

println("="^80)
println("Parameters: S=$S, γ=$γ, ϕU=$ϕU, λU=$λU, w=$(round(w, digits=4)), v=$v")
println("LDR bounds: πU=$πU, yU=$yU, ytsU_default=$ytsU_default")
println("="^80)

# ===== Benders baseline =====
x_opt = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
λ_opt = 0.005347647652838779
h_opt = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.010963146805888255, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.010784131903911209, 0.0, 0.0, 0.0]
ψ0_opt = [λ_opt * x_opt[k] for k in 1:num_arcs]
println("\nBASELINE: OSP at x_opt")
osp_model, osp_vars, osp_data = build_dualized_outer_subproblem(
    network, S, ϕU, λU, γ, w, v, uncertainty_set, Mosek.Optimizer,
    λ_opt, x_opt, h_opt, ψ0_opt; πU=πU, yU=yU, ytsU=ytsU_default)
(osp_status, osp_coeff) = osp_optimize!(osp_model, osp_vars, osp_data,
    λ_opt, x_opt, h_opt, ψ0_opt)
osp_obj = osp_coeff[:obj_val]
println("  OSP obj: $osp_obj")

# ===== ytsU sensitivity =====
println("\n" * "="^80)
println("ytsU SENSITIVITY TEST")
println("="^80)
println("  ytsU_default = $ytsU_default")
println()

results = []
for ytsU_test in [0.0]
    model_t, vars_t = build_full_2DRNDP_model(network, S, ϕU, λU, γ, w, v, uncertainty_set,
        mip_solver=Gurobi.Optimizer, conic_solver=Mosek.Optimizer,
        πU=πU, yU=yU, ytsU=ytsU_test)
    add_sparsity_constraints!(model_t, vars_t, network, S)
    optimize!(model_t)
    st_t = termination_status(model_t)
    if st_t == MOI.OPTIMAL || st_t == MOI.ALMOST_OPTIMAL
        obj_t = objective_value(model_t)
        yts0_vals = [value(vars_t[:Yts_tilde][s, 1, num_arcs+1]) for s in 1:S]
        eta_t = value.(vars_t[:ηtilde])
        nu_t = value(vars_t[:nu])
        lambda_t = value(vars_t[:λ])
        ytsL_norms = [norm(value.(vars_t[:Yts_tilde][s, 1, 1:num_arcs])) for s in 1:S]
        push!(results, (ytsU_test, obj_t, eta_t, yts0_vals, nu_t, lambda_t, ytsL_norms))
        marker = ytsU_test ≈ ytsU_default ? " ← default" : ""
        println("  ytsU=$(lpad(round(ytsU_test, digits=2), 6)): obj=$(lpad(round(obj_t, digits=4), 10)), " *
                "ηtilde=$(round.(eta_t, digits=4)), Yts_0=$(round.(yts0_vals, digits=4)), " *
                "‖YtsL‖=$(round.(ytsL_norms, digits=6)), " *
                "nu=$(round(nu_t, digits=4)), λ=$(round(lambda_t, digits=4))$marker")
    else
        println("  ytsU=$(lpad(round(ytsU_test, digits=2), 6)): $st_t")
    end
end

# ===== Summary =====
println("\n" * "="^80)
println("SUMMARY")
println("="^80)
println("  Benders OSP at x_opt: $(round(osp_obj, digits=6))")
println()
println("  ytsU  |  Full model obj  |  Yts_0 hits bound?")
println("  ------|------------------|-------------------")
for (ytsU_test, obj_t, eta_t, yts0_vals, nu_t, lambda_t) in results
    hits_ub = any(abs.(yts0_vals) .>= ytsU_test - 1e-4)
    marker = ytsU_test ≈ ytsU_default ? " ← default" : ""
    println("  $(lpad(round(ytsU_test, digits=4), 5)) | $(lpad(round(obj_t, digits=6), 16)) | $(hits_ub ? "YES" : "no")$marker")
end
println("="^80)
