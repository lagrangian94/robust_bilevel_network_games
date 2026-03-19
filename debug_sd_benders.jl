"""
Debug script: x_opt에서 ISP vs Joint OSP 값 직접 비교.
Nested Benders의 inner loop (IMP → ISP_l + ISP_f)가 joint OSP와 같은 값을 주는지 확인.
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using LinearAlgebra
using Infiltrator
using Revise
using Logging

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("build_full_model.jl")
includet("parallel_utils.jl")
includet("strict_benders.jl")
includet("nested_benders_trust_region.jl")

using .NetworkGenerator: generate_capacity_scenarios_uniform_model
using .NetworkGenerator: generate_polska_network, print_realworld_network_summary

# ===== Setup (global scope) =====
S = 2
seed = 42
epsilon = 0.5
γ_ratio = 0.10
ρ = 0.2
v = 1.0
multi_cut_lf = true
network = generate_polska_network()
print_realworld_network_summary(network)
num_arcs = length(network.arcs) - 1
strengthen_cuts = :mw
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
ytsU = min(max_flow_ub, ϕU)

println("="^80)
println("Parameters: S=$S, γ=$γ, ϕU=$ϕU, λU=$λU, w=$(round(w, digits=4)), v=$v")
println("LDR bounds: πU=$πU, yU=$yU, ytsU=$ytsU")
println("="^80)

# ===== sd optimal solution (하드코딩) =====
x_opt = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
λ_opt = 0.005347647652838779
h_opt = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.010963146805888255, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.010784131903911209, 0.0, 0.0, 0.0]
ψ0_opt = [λ_opt * x_opt[k] for k in 1:num_arcs]

# ===== 1. Joint OSP at x_opt (ground truth) =====
println("\n" * "="^80)
println("STEP 1: Joint OSP at x_opt (ground truth)")
println("="^80)

osp_model, osp_vars, osp_data = build_dualized_outer_subproblem(
    network, S, ϕU, λU, γ, w, v, uncertainty_set, Mosek.Optimizer,
    λ_opt, x_opt, h_opt, ψ0_opt; πU=πU, yU=yU, ytsU=ytsU)

(osp_status, osp_coeff) = osp_optimize!(osp_model, osp_vars, osp_data,
    λ_opt, x_opt, h_opt, ψ0_opt)
osp_obj = osp_coeff[:obj_val]
α_osp = osp_coeff[:α_sol]
println("  Joint OSP obj: $osp_obj")
println("  α_osp sum: $(sum(α_osp))")

# ===== 2. ISP_l + ISP_f at x_opt, same α =====
println("\n" * "="^80)
println("STEP 2: ISP_l + ISP_f at x_opt with α from joint OSP")
println("="^80)

E = ones(num_arcs, num_arcs+1)
d0 = zeros(num_arcs + 1); d0[end] = 1.0
isp_data = Dict(:E => E, :network => network, :ϕU => ϕU, :πU => πU, :yU => yU, :ytsU => ytsU, :λU => λU, :γ => γ, :w => w, :v => v, :uncertainty_set => uncertainty_set, :d0 => d0, :S=>S)

leader_instances, follower_instances = initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, λ_sol=λ_opt, x_sol=x_opt, h_sol=h_opt, ψ0_sol=ψ0_opt, α_sol=α_osp,
    πU=πU, yU=yU, ytsU=ytsU)

total_isp = 0.0
for s in 1:S
    U_s = Dict(:R => Dict(:1=>R[s]), :r_dict => Dict(:1=>r_dict[s]), :xi_bar => Dict(:1=>xi_bar[s]), :epsilon => epsilon)
    (st_l, ci_l) = isp_leader_optimize!(leader_instances[s][1], leader_instances[s][2];
        isp_data=isp_data, uncertainty_set=U_s, λ_sol=λ_opt, x_sol=x_opt, h_sol=h_opt, ψ0_sol=ψ0_opt, α_sol=α_osp)
    (st_f, ci_f) = isp_follower_optimize!(follower_instances[s][1], follower_instances[s][2];
        isp_data=isp_data, uncertainty_set=U_s, λ_sol=λ_opt, x_sol=x_opt, h_sol=h_opt, ψ0_sol=ψ0_opt, α_sol=α_osp)
    isp_l_obj = ci_l[:obj_val]
    isp_f_obj = ci_f[:obj_val]
    println("  s=$s: ISP_l=$(round(isp_l_obj, digits=6)), ISP_f=$(round(isp_f_obj, digits=6)), sum=$(round(isp_l_obj+isp_f_obj, digits=6))")
    global total_isp += isp_l_obj + isp_f_obj
end
avg_isp = total_isp / S
println("  Total ISP avg (should == Joint OSP): $(round(avg_isp, digits=6))")
println("  Joint OSP:                           $(round(osp_obj, digits=6))")
println("  DIFF: $(round(avg_isp - osp_obj, digits=6))")

# ===== 3. IMP inner loop at x_opt (full nested benders inner loop) =====
println("\n" * "="^80)
println("STEP 3: Full IMP inner loop at x_opt")
println("="^80)

imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_optimizer=Gurobi.Optimizer)
st_imp, _ = initialize_imp(imp_model, imp_vars)

# Re-initialize ISP for inner loop
leader_instances2, follower_instances2 = initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, λ_sol=λ_opt, x_sol=x_opt, h_sol=h_opt, ψ0_sol=ψ0_opt, α_sol=α_osp,
    πU=πU, yU=yU, ytsU=ytsU)

imp_cuts = Dict{Symbol, Any}(:old_tr_constraints => nothing)
status, cut_info = tr_imp_optimize!(imp_model, imp_vars, leader_instances2, follower_instances2;
    isp_data=isp_data, λ_sol=λ_opt, x_sol=x_opt, h_sol=h_opt, ψ0_sol=ψ0_opt,
    outer_iter=1, imp_cuts=imp_cuts, inner_tr=true, tol=1e-4, parallel=true)

inner_obj = cut_info[:obj_val]
inner_α = cut_info[:α_sol]
println("  IMP converged obj: $(round(inner_obj, digits=6))")
println("  IMP α sum: $(sum(inner_α))")
println("  Joint OSP obj:     $(round(osp_obj, digits=6))")
println("  DIFF: $(round(inner_obj - osp_obj, digits=6))")

# ===== 4. ISP at IMP's converged α (cross-check) =====
println("\n" * "="^80)
println("STEP 4: ISP at IMP's α vs Joint OSP at IMP's α")
println("="^80)

# Joint OSP at IMP's α
osp_model2, osp_vars2, osp_data2 = build_dualized_outer_subproblem(
    network, S, ϕU, λU, γ, w, v, uncertainty_set, Mosek.Optimizer,
    λ_opt, x_opt, h_opt, ψ0_opt; πU=πU, yU=yU, ytsU=ytsU)
# fix α in joint OSP
for k in 1:num_arcs
    fix(osp_vars2[:α][k], inner_α[k]; force=true)
end
optimize!(osp_model2)
joint_at_imp_α = objective_value(osp_model2)
println("  Joint OSP at IMP α: $(round(joint_at_imp_α, digits=6))")
println("  IMP obj:            $(round(inner_obj, digits=6))")
println("  DIFF:               $(round(inner_obj - joint_at_imp_α, digits=6))")

println("\n" * "="^80)
println("DONE")
println("="^80)

# ===== 5. Full Model FREE (Pajarito) =====
println("\n" * "="^80)
println("STEP 5: Full Model FREE (all variables optimized)")
println("="^80)

using Pajarito

model_free, vars_free = build_full_2DRNDP_model(network, S, ϕU, λU, γ, w, v, uncertainty_set,
    mip_solver=Gurobi.Optimizer, conic_solver=Mosek.Optimizer,
    πU=πU, yU=yU, ytsU=ytsU)
add_sparsity_constraints!(model_free, vars_free, network, S)

optimize!(model_free)
fm_free_status = termination_status(model_free)
println("  Status: $fm_free_status")
fm_free_obj = NaN
if fm_free_status == MOI.OPTIMAL || fm_free_status == MOI.ALMOST_OPTIMAL
    fm_free_obj = objective_value(model_free)
    println("  Full model obj (free): $(round(fm_free_obj, digits=6))")
    println("  x: $(value.(vars_free[:x]))")
    println("  λ: $(value(vars_free[:λ]))")
    println("  ηhat: $(value.(vars_free[:ηhat]))")
    println("  ηtilde: $(value.(vars_free[:ηtilde]))")
    println("  nu: $(value(vars_free[:nu]))")
    println("  Yts_tilde_0: $(value.(vars_free[:Yts_tilde][:,1,end]))")
end

# ===== 6. Cross-check: OSP at full model's free solution =====
println("\n" * "="^80)
println("STEP 6: OSP at full model's free optimal")
println("="^80)

if fm_free_status == MOI.OPTIMAL || fm_free_status == MOI.ALMOST_OPTIMAL
    λ_free = value(vars_free[:λ])
    x_free = value.(vars_free[:x])
    h_free = value.(vars_free[:h])
    ψ0_free = value.(vars_free[:ψ0])

    osp_model_c, osp_vars_c, osp_data_c = build_dualized_outer_subproblem(
        network, S, ϕU, λU, γ, w, v, uncertainty_set, Mosek.Optimizer,
        λ_free, x_free, h_free, ψ0_free; πU=πU, yU=yU, ytsU=ytsU)
    (osp_status_c, osp_coeff_c) = osp_optimize!(osp_model_c, osp_vars_c, osp_data_c,
        λ_free, x_free, h_free, ψ0_free)
    println("  OSP obj at full model's solution: $(osp_coeff_c[:obj_val])")
    println("  Full model obj (free):            $(round(fm_free_obj, digits=6))")
    println("  DIFF: $(round(osp_coeff_c[:obj_val] - fm_free_obj, digits=6))")
end

@infiltrate
