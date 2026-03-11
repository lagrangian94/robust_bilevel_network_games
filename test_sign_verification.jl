"""
test_sign_verification.jl — Quick test comparing dual ISP variable values with primal ISP constraint duals.
Verifies that the sign conventions in evaluate_master_opt_cut_from_primal are correct.
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using LinearAlgebra
using Revise

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("strict_benders.jl")
includet("nested_benders_trust_region.jl")
includet("build_primal_isp.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model

S = 1; ϕU = 10.0; λU = 10.0; γ = 2.0; w = 1.0; v = 1.0; seed = 42; epsilon = 0.5
network = generate_grid_network(3, 3, seed=seed)
capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacities[1:end-1, :], epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)
num_arcs = length(network.arcs) - 1

# Get initial point from OMP
omp_model, omp_vars = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
optimize!(omp_model)
x_sol = value.(omp_vars[:x]); h_sol = value.(omp_vars[:h])
λ_sol = value(omp_vars[:λ]); ψ0_sol = value.(omp_vars[:ψ0])

# Get α from IMP
imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_optimizer=Gurobi.Optimizer)
optimize!(imp_model)
α_sol = max.(value.(imp_vars[:α]), 0.0)

E = ones(num_arcs, num_arcs+1)
d0 = zeros(num_arcs + 1); d0[end] = 1.0
isp_data = Dict(:E=>E, :ϕU=>ϕU, :d0=>d0, :S=>S, :w=>w, :v=>v, :uncertainty_set=>uncertainty_set, :network=>network, :λU=>λU, :γ=>γ)

# Build both dual ISP and primal ISP
dual_li, dual_fi = initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
primal_li, primal_fi = initialize_primal_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)

println("R[1] shape: ", size(R[1]))
println("dim_R_cols (correct): ", size(R[1], 2))

for s in 1:S
    U_s = Dict(:R => Dict(1=>R[s]), :r_dict => Dict(1=>r_dict[s]),
                :xi_bar => Dict(1=>xi_bar[s]), :epsilon => epsilon)

    # Solve dual ISP
    isp_leader_optimize!(dual_li[s][1], dual_li[s][2];
        isp_data=isp_data, uncertainty_set=U_s,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
    isp_follower_optimize!(dual_fi[s][1], dual_fi[s][2];
        isp_data=isp_data, uncertainty_set=U_s,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)

    # Solve primal ISP
    primal_isp_leader_optimize!(primal_li[s][1], primal_li[s][2]; isp_data=isp_data, α_sol=α_sol)
    primal_isp_follower_optimize!(primal_fi[s][1], primal_fi[s][2]; isp_data=isp_data, α_sol=α_sol)

    # === LEADER: Compare Uhat1, Uhat3 ===
    dual_Uhat1 = value.(dual_li[s][2][:Uhat1])[s,:,:]
    primal_Uhat1 = -dual.(primal_li[s][2][:con_bigM1_hat])  # -dual for <=

    dual_Uhat3 = value.(dual_li[s][2][:Uhat3])[s,:,:]
    primal_Uhat3 = -dual.(primal_li[s][2][:con_bigM3_hat])

    println("\n=== Scenario $s ===")
    println("LEADER Uhat1:")
    println("  dual ISP max |val|:   ", maximum(abs.(dual_Uhat1)))
    println("  primal ISP max |val|: ", maximum(abs.(primal_Uhat1)))
    println("  max gap:              ", maximum(abs.(dual_Uhat1 - primal_Uhat1)))
    println("  min(dual):  ", minimum(dual_Uhat1), "  min(primal): ", minimum(primal_Uhat1))

    println("LEADER Uhat3:")
    println("  dual ISP max |val|:   ", maximum(abs.(dual_Uhat3)))
    println("  primal ISP max |val|: ", maximum(abs.(primal_Uhat3)))
    println("  max gap:              ", maximum(abs.(dual_Uhat3 - primal_Uhat3)))

    # === FOLLOWER: Compare Utilde1, Utilde3, Ztilde1_3, βtilde1_1, βtilde1_3 ===
    dual_Utilde1 = value.(dual_fi[s][2][:Utilde1])[s,:,:]
    primal_Utilde1 = -dual.(primal_fi[s][2][:con_bigM1_tilde])

    dual_Utilde3 = value.(dual_fi[s][2][:Utilde3])[s,:,:]
    primal_Utilde3 = -dual.(primal_fi[s][2][:con_bigM3_tilde])

    println("FOLLOWER Utilde1:")
    println("  max gap:              ", maximum(abs.(dual_Utilde1 - primal_Utilde1)))

    println("FOLLOWER Utilde3:")
    println("  max gap:              ", maximum(abs.(dual_Utilde3 - primal_Utilde3)))

    # Ztilde1_3
    b3s = primal_fi[s][2][:block3_start_idx]
    b3e = primal_fi[s][2][:block3_end_idx]
    b1sz = primal_fi[s][2][:block1_size]

    dual_Ztilde1_3 = value.(dual_fi[s][2][:Ztilde1_3])[s,:,:]
    eq_duals = dual.(primal_fi[s][2][:con_soc_eq_tilde])
    primal_Ztilde1_3 = eq_duals[b3s:b3e, :]

    println("Ztilde1_3:")
    println("  dual ISP shape: ", size(dual_Ztilde1_3))
    println("  primal ISP shape: ", size(primal_Ztilde1_3))
    if size(dual_Ztilde1_3) == size(primal_Ztilde1_3)
        println("  max gap:              ", maximum(abs.(dual_Ztilde1_3 - primal_Ztilde1_3)))
    else
        println("  *** SHAPE MISMATCH ***")
    end

    # βtilde1_1
    ineq_duals = dual.(primal_fi[s][2][:con_soc_ineq_tilde])
    dual_βtilde1_1 = value.(dual_fi[s][2][:βtilde1_1])[s,:]
    primal_βtilde1_1 = ineq_duals[1:b1sz]

    println("βtilde1_1:")
    println("  dual ISP:   ", round.(dual_βtilde1_1, digits=6))
    println("  primal ISP: ", round.(primal_βtilde1_1, digits=6))
    println("  max gap:    ", maximum(abs.(dual_βtilde1_1 - primal_βtilde1_1)))

    # βtilde1_3
    dual_βtilde1_3 = value.(dual_fi[s][2][:βtilde1_3])[s,:]
    primal_βtilde1_3 = ineq_duals[b3s:b3e]

    println("βtilde1_3:")
    println("  dual ISP:   ", round.(dual_βtilde1_3, digits=6))
    println("  primal ISP: ", round.(primal_βtilde1_3, digits=6))
    println("  max gap:    ", maximum(abs.(dual_βtilde1_3 - primal_βtilde1_3)))

    # Objective comparison
    println("\nObjective comparison:")
    println("  Dual leader obj:   ", objective_value(dual_li[s][1]))
    println("  Primal leader obj: ", objective_value(primal_li[s][1]))
    println("  Dual follower obj:   ", objective_value(dual_fi[s][1]))
    println("  Primal follower obj: ", objective_value(primal_fi[s][1]))
end

println("\n" * "="^60)
println("SIGN VERIFICATION COMPLETE")
println("="^60)
