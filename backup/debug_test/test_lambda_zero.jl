"""
test_lambda_zero.jl — Check if dual/primal ISP work at λ=0
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

# Test with various λ values
x_sol = zeros(num_arcs)
h_sol = zeros(num_arcs)
ψ0_sol = zeros(num_arcs)
α_sol = ones(num_arcs)  # all nonzero

for λ_test in [1.0, 0.1, 0.01, 0.001, 0.0]
    println("\n" * "="^60)
    println("Testing λ = $λ_test")
    println("="^60)

    # --- Dual ISP ---
    dl, df = initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
        conic_optimizer=Mosek.Optimizer, λ_sol=λ_test, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)

    U_s = Dict(:R => Dict(:1=>R_unc[1]), :r_dict => Dict(:1=>r_dict_unc[1]),
                :xi_bar => Dict(:1=>xi_bar_unc[1]), :epsilon => epsilon)

    (st_l, ci_l) = isp_leader_optimize!(dl[1][1], dl[1][2];
        isp_data=isp_data, uncertainty_set=U_s,
        λ_sol=λ_test, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
    println("  Dual ISP leader: $st_l, obj=$(round(ci_l[:obj_val], digits=4))")

    (st_f, ci_f) = isp_follower_optimize!(df[1][1], df[1][2];
        isp_data=isp_data, uncertainty_set=U_s,
        λ_sol=λ_test, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
    println("  Dual ISP follower: $st_f, obj=$(round(ci_f[:obj_val], digits=4))")

    # --- Primal ISP ---
    try
        pl, pf = initialize_primal_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
            conic_optimizer=Mosek.Optimizer, x_sol=x_sol, λ_sol=λ_test, h_sol=h_sol, ψ0_sol=ψ0_sol)

        μhat_p = pl[1][2][:μhat]
        for k in 1:num_arcs
            set_objective_coefficient(pl[1][1], μhat_p[1, k], α_sol[k])
        end
        optimize!(pl[1][1])
        st_pl = termination_status(pl[1][1])
        println("  Primal ISP leader: $st_pl", st_pl == MOI.OPTIMAL ? ", obj=$(round(objective_value(pl[1][1]), digits=4))" : "")

        μtilde_p = pf[1][2][:μtilde]
        for k in 1:num_arcs
            set_objective_coefficient(pf[1][1], μtilde_p[1, k], α_sol[k])
        end
        optimize!(pf[1][1])
        st_pf = termination_status(pf[1][1])
        println("  Primal ISP follower: $st_pf", st_pf == MOI.OPTIMAL ? ", obj=$(round(objective_value(pf[1][1]), digits=4))" : "")
    catch e
        println("  Primal ISP error: $e")
    end
end
