"""
test_lambda_zero_xnz.jl — Test primal ISP with λ=0 and nonzero x
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using LinearAlgebra
using Revise

includet("../network_generator.jl")
includet("../build_uncertainty_set.jl")
includet("../strict_benders.jl")
includet("../nested_benders_trust_region.jl")
includet("../build_primal_isp.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model

S = 1; ϕU = 10.0; λU = 10.0; γ = 2.0; w = 1.0; v = 1.0; seed = 42; epsilon = 0.5
network = generate_grid_network(3, 3, seed=seed)
capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

num_arcs = length(network.arcs) - 1
E = ones(num_arcs, num_arcs+1)
d0 = zeros(num_arcs + 1); d0[end] = 1.0
isp_data = Dict(:E => E, :network => network, :ϕU => ϕU, :λU => λU, :γ => γ,
    :w => w, :v => v, :uncertainty_set => uncertainty_set, :d0 => d0, :S => S)
R_unc = uncertainty_set[:R]; r_dict_unc = uncertainty_set[:r_dict]; xi_bar_unc = uncertainty_set[:xi_bar]

α_sol = ones(num_arcs)

# Test: x nonzero, λ=0
# With McCormick: ψ0 ≤ λ = 0, so ψ0 = 0
# Also ψ0 ≥ λ - λU(1-x) = -λU(1-x) when λ=0, so ψ0 ≥ 0 (already)

println("Interdictable arcs: ", findall(network.interdictable_arcs[1:num_arcs]))

# Set x[i]=1 for first 2 interdictable arcs
x_sol = zeros(num_arcs)
interdictable = findall(network.interdictable_arcs[1:num_arcs])
for idx in interdictable[1:min(2, length(interdictable))]
    x_sol[idx] = 1.0
end
h_sol = zeros(num_arcs)
ψ0_sol = zeros(num_arcs)

for λ_test in [1.0, 0.1, 0.01, 0.001, 1e-6, 0.0]
    # With λ > 0: ψ0 should be min(λ, λU*x) for active arcs
    ψ0_test = copy(ψ0_sol)
    for i in 1:num_arcs
        ψ0_test[i] = min(λ_test, λU * x_sol[i])
    end

    println("\n--- λ=$λ_test, x=$(round.(x_sol, digits=1)), ψ0_max=$(maximum(ψ0_test)) ---")

    # Dual ISP
    dl, df = initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
        conic_optimizer=Mosek.Optimizer, λ_sol=λ_test, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_test, α_sol=α_sol)
    U_s = Dict(:R => Dict(:1=>R_unc[1]), :r_dict => Dict(:1=>r_dict_unc[1]),
                :xi_bar => Dict(:1=>xi_bar_unc[1]), :epsilon => epsilon)
    (st_l, ci_l) = isp_leader_optimize!(dl[1][1], dl[1][2];
        isp_data=isp_data, uncertainty_set=U_s,
        λ_sol=λ_test, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_test, α_sol=α_sol)
    (st_f, ci_f) = isp_follower_optimize!(df[1][1], df[1][2];
        isp_data=isp_data, uncertainty_set=U_s,
        λ_sol=λ_test, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_test, α_sol=α_sol)
    println("  Dual: leader=$st_l ($(round(ci_l[:obj_val],digits=4))), follower=$st_f ($(round(ci_f[:obj_val],digits=4)))")

    # Primal ISP
    pl, pf = initialize_primal_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
        conic_optimizer=Mosek.Optimizer, x_sol=x_sol, λ_sol=λ_test, h_sol=h_sol, ψ0_sol=ψ0_test)
    μhat_p = pl[1][2][:μhat]
    for k in 1:num_arcs
        set_objective_coefficient(pl[1][1], μhat_p[1, k], α_sol[k])
    end
    optimize!(pl[1][1])
    st_pl = termination_status(pl[1][1])

    μtilde_p = pf[1][2][:μtilde]
    for k in 1:num_arcs
        set_objective_coefficient(pf[1][1], μtilde_p[1, k], α_sol[k])
    end
    optimize!(pf[1][1])
    st_pf = termination_status(pf[1][1])

    println("  Primal: leader=$st_pl, follower=$st_pf")
    if st_pl == MOI.OPTIMAL
        println("    leader obj=$(round(objective_value(pl[1][1]), digits=4))")
    end
    if st_pf == MOI.OPTIMAL
        println("    follower obj=$(round(objective_value(pf[1][1]), digits=4))")
    end
end

# Also test with h nonzero
println("\n\n=== Now with h nonzero ===")
h_sol2 = zeros(num_arcs)
h_sol2[interdictable[1]] = 5.0
h_sol2[interdictable[2]] = 3.0

for λ_test in [0.01, 0.0]
    ψ0_test = zeros(num_arcs)
    for i in 1:num_arcs
        ψ0_test[i] = min(λ_test, λU * x_sol[i])
    end

    println("\n--- λ=$λ_test, h_nz=$(count(x->x>0, h_sol2)) ---")

    dl, df = initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
        conic_optimizer=Mosek.Optimizer, λ_sol=λ_test, x_sol=x_sol, h_sol=h_sol2, ψ0_sol=ψ0_test, α_sol=α_sol)
    U_s = Dict(:R => Dict(:1=>R_unc[1]), :r_dict => Dict(:1=>r_dict_unc[1]),
                :xi_bar => Dict(:1=>xi_bar_unc[1]), :epsilon => epsilon)
    (st_f, ci_f) = isp_follower_optimize!(df[1][1], df[1][2];
        isp_data=isp_data, uncertainty_set=U_s,
        λ_sol=λ_test, x_sol=x_sol, h_sol=h_sol2, ψ0_sol=ψ0_test, α_sol=α_sol)
    println("  Dual follower: $st_f ($(round(ci_f[:obj_val],digits=4)))")

    pf2 = initialize_primal_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
        conic_optimizer=Mosek.Optimizer, x_sol=x_sol, λ_sol=λ_test, h_sol=h_sol2, ψ0_sol=ψ0_test)[2]
    μtilde_p = pf2[1][2][:μtilde]
    for k in 1:num_arcs
        set_objective_coefficient(pf2[1][1], μtilde_p[1, k], α_sol[k])
    end
    optimize!(pf2[1][1])
    st_pf = termination_status(pf2[1][1])
    println("  Primal follower: $st_pf", st_pf == MOI.OPTIMAL ? " ($(round(objective_value(pf2[1][1]),digits=4)))" : "")
end
