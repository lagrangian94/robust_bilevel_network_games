# diag_fix_d_eq_a.jl — Fix d=a and compare Z₀ to check multiple optima
# If Z₀(d=a) = Z₀(free), then d=a is also optimal (multiple optima)

using JuMP, Gurobi, Printf, LinearAlgebra

include("../../network_generator.jl")
NG = NetworkGenerator
include("../true_dro_data.jl")
include("../true_dro_build_omp.jl")
include("../true_dro_build_subproblem.jl")
include("../true_dro_build_isp_leader.jl")
include("../true_dro_build_isp_follower.jl")
include("../true_dro_benders.jl")

net = NG.generate_grid_network(3, 3; seed=42)
num_arcs = length(net.arcs) - 1
S = 3; K = num_arcs

caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S;
    interdictable_arcs=net.interdictable_arcs, seed=42, num_factors=3)
intd_idx = findall(net.interdictable_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
q_hat = [0.5, 0.3, 0.2]
gamma = 2

x_bar = zeros(K)
for a in intd_idx[1:gamma]; x_bar[a] = 1.0; end

configs = [
    (0.1, 0.1),
    (0.1, 0.3),
    (0.2, 0.2),
    (0.3, 0.3),
    (0.5, 0.5),
]

println("=" ^ 80)
println("Comparing Z₀(free) vs Z₀(d=a) — 3x3 grid, S=3")
println("=" ^ 80)
@printf("%-12s | %-12s | %-12s | %-12s | %-8s\n",
    "(ε̂, ε̃)", "Z₀(free)", "Z₀(d=a)", "gap", "feasible?")
println("-" ^ 65)

for (ε_hat, ε_tilde) in configs
    # --- Free solve ---
    td = make_true_dro_data(net, caps, q_hat, ε_hat, ε_tilde; w=w, lambda_U=10.0, gamma=gamma)
    sub_model, sub_vars = build_true_dro_subproblem(td, x_bar;
        optimizer=Gurobi.Optimizer, rho_upper_bound=10.0)
    set_optimizer_attribute(sub_model, "NonConvex", 2)
    set_optimizer_attribute(sub_model, "TimeLimit", 600.0)
    set_optimizer_attribute(sub_model, "MIPGap", 1e-8)
    set_optimizer_attribute(sub_model, "OutputFlag", 0)
    optimize!(sub_model)
    Z0_free = objective_value(sub_model)

    # --- Fix d = a ---
    td2 = make_true_dro_data(net, caps, q_hat, ε_hat, ε_tilde; w=w, lambda_U=10.0, gamma=gamma)
    sub_model2, sub_vars2 = build_true_dro_subproblem(td2, x_bar;
        optimizer=Gurobi.Optimizer, rho_upper_bound=10.0)
    set_optimizer_attribute(sub_model2, "NonConvex", 2)
    set_optimizer_attribute(sub_model2, "TimeLimit", 600.0)
    set_optimizer_attribute(sub_model2, "MIPGap", 1e-8)
    set_optimizer_attribute(sub_model2, "OutputFlag", 0)

    # Add constraint d[s] = a[s] for all s
    for s in 1:S
        @constraint(sub_model2, sub_vars2[:d][s] == sub_vars2[:a][s])
    end
    optimize!(sub_model2)

    st2 = termination_status(sub_model2)
    if st2 == MOI.OPTIMAL || st2 == MOI.TIME_LIMIT
        Z0_fixed = objective_value(sub_model2)
        gap = Z0_free - Z0_fixed
        @printf("(%.1f, %.1f)   | %12.6f | %12.6f | %+12.6f | %s\n",
            ε_hat, ε_tilde, Z0_free, Z0_fixed, gap, st2)
    else
        @printf("(%.1f, %.1f)   | %12.6f | %12s | %12s | %s\n",
            ε_hat, ε_tilde, Z0_free, "INFEASIBLE", "—", st2)
    end
    flush(stdout)
end

println("\nDone.")
