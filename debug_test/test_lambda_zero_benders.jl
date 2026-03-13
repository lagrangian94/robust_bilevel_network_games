"""
test_lambda_zero_benders.jl — Run Benders with λ≥0 (no 0.001 lower bound)
Reproduce primal ISP infeasibility
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

# 3×3 grid (17 arcs, matching the error)
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

# Build OMP WITHOUT λ≥0.001
println("Building OMP (λ≥0, no lower bound)...")
omp_model, omp_vars = build_omp(network, ϕU, λU, γ, w;
    optimizer=Gurobi.Optimizer, multi_cut=true)
# NOTE: build_omp line 44 should be commented out (λ>=0.001)

st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars)

imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer)
st_imp, α_init = initialize_imp(imp_model, imp_vars)

println("Initial: λ=$(λ_sol), x=$(round.(x_sol, digits=2))")

# Run outer iterations
for outer_iter in 1:30
    global st, λ_sol, x_sol, h_sol, ψ0_sol

    optimize!(omp_model)
    st = MOI.get(omp_model, MOI.TerminationStatus())
    if st == MOI.INFEASIBLE
        println("OMP infeasible at outer $outer_iter")
        break
    end
    x_sol = value.(omp_vars[:x])
    h_sol = value.(omp_vars[:h])
    λ_sol = value(omp_vars[:λ])
    ψ0_sol = value.(omp_vars[:ψ0])

    println("\n--- Outer $outer_iter: λ=$(round(λ_sol, digits=6)), x_nz=$(count(x->x>0.5, x_sol)) ---")

    # Try primal ISP
    pl, pf = initialize_primal_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
        conic_optimizer=Mosek.Optimizer, x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)

    # Also try dual ISP
    dl, df = initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
        conic_optimizer=Mosek.Optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_init)

    U_s = Dict(:R => Dict(:1=>R_unc[1]), :r_dict => Dict(:1=>r_dict_unc[1]),
                :xi_bar => Dict(:1=>xi_bar_unc[1]), :epsilon => epsilon)

    # Quick inner solve
    optimize!(imp_model)
    α_sol = value.(imp_vars[:α])

    # Dual ISP
    (st_l, ci_l) = isp_leader_optimize!(dl[1][1], dl[1][2];
        isp_data=isp_data, uncertainty_set=U_s,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
    (st_f, ci_f) = isp_follower_optimize!(df[1][1], df[1][2];
        isp_data=isp_data, uncertainty_set=U_s,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
    println("  Dual ISP: leader=$st_l follower=$st_f")

    # Primal ISP
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

    println("  Primal ISP: leader=$st_pl follower=$st_pf")

    if st_pf != MOI.OPTIMAL
        println("  *** PRIMAL ISP FOLLOWER FAILED at λ=$(λ_sol) ***")
        println("  x_sol = ", round.(x_sol, digits=4))
        println("  h_sol = ", round.(h_sol, digits=4))
        println("  ψ0_sol = ", round.(ψ0_sol, digits=4))
        println("  α_sol = ", round.(α_sol, digits=4))
        break
    end

    # Add outer cut (simplified)
    subprob_obj = ci_l[:obj_val] + ci_f[:obj_val]
    @constraint(omp_model, omp_vars[:t_0_l] + omp_vars[:t_0_f] >= subprob_obj)

    # Add inner cut
    cl = @constraint(imp_model, [s=1:S], imp_vars[:t_1_l][s] <= ci_l[:intercept] + imp_vars[:α]'*ci_l[:μhat])
    cf = @constraint(imp_model, [s=1:S], imp_vars[:t_1_f][s] <= ci_f[:intercept] + imp_vars[:α]'*ci_f[:μtilde])
end
