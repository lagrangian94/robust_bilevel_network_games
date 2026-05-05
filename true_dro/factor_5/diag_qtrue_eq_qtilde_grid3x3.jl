# diag_qtrue_eq_qtilde_grid3x3.jl — Small instance for OPTIMAL verification
# 3x3 grid, S=3, should solve to global optimality quickly

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

# Non-uniform q̂
q_hat = [0.5, 0.3, 0.2]

# Use a reasonable x (interdict 2 arcs from interdictable set)
gamma = 2
println("Interdictable arcs: $intd_idx ($(length(intd_idx)) total)")
println("Using first 2 interdictable arcs: $(intd_idx[1:gamma])")
x_bar = zeros(K)
for a in intd_idx[1:gamma]; x_bar[a] = 1.0; end

println("\n" * "=" ^ 80)
println("3x3 Grid, S=$S, γ=$gamma, q̂=$q_hat")
println("x = arcs $(findall(x_bar .> 0.5))")
println("=" ^ 80)

# Test: ε̂ ≤ ε̃
configs_leq = [
    (0.1, 0.1),
    (0.1, 0.3),
    (0.2, 0.2),
    (0.2, 0.5),
    (0.3, 0.3),
    (0.3, 1.0),
    (0.5, 0.5),
    (0.5, 1.0),
]

println("\n--- Case: ε̂ ≤ ε̃ (expecting q* = d*) ---\n")
@printf("%-12s | %-8s | %-12s | %-10s | %-10s | %-10s\n",
    "(ε̂, ε̃)", "status", "Z₀", "TV(a*,d*)", "TV(a*,q̂)", "TV(d*,q̂)")
println("-" ^ 75)

for (ε_hat, ε_tilde) in configs_leq
    td = make_true_dro_data(net, caps, q_hat, ε_hat, ε_tilde; w=w, lambda_U=10.0, gamma=gamma)
    sub_model, sub_vars = build_true_dro_subproblem(td, x_bar;
        optimizer=Gurobi.Optimizer, rho_upper_bound=10.0)
    set_optimizer_attribute(sub_model, "NonConvex", 2)
    set_optimizer_attribute(sub_model, "TimeLimit", 600.0)
    set_optimizer_attribute(sub_model, "MIPGap", 1e-8)
    set_optimizer_attribute(sub_model, "OutputFlag", 0)
    optimize!(sub_model)

    st = termination_status(sub_model)
    Z0 = objective_value(sub_model)
    a_val = [value(sub_vars[:a][s]) for s in 1:S]
    d_val = [value(sub_vars[:d][s]) for s in 1:S]

    tv_ad = 0.5 * sum(abs.(a_val .- d_val))
    tv_a_qhat = 0.5 * sum(abs.(a_val .- q_hat))
    tv_d_qhat = 0.5 * sum(abs.(d_val .- q_hat))

    mark = tv_ad < 1e-4 ? "✓" : "✗"
    @printf("(%.1f, %.1f)   | %-8s | %12.6f | %10.6f%s | %10.6f | %10.6f\n",
        ε_hat, ε_tilde, st, Z0, tv_ad, mark, tv_a_qhat, tv_d_qhat)

    if tv_ad > 1e-4
        @printf("    a* = %s\n", string(round.(a_val; digits=5)))
        @printf("    d* = %s\n", string(round.(d_val; digits=5)))
    end
    flush(stdout)
end

# Counter-check: ε̂ > ε̃
configs_gt = [(0.3, 0.1), (0.5, 0.2), (1.0, 0.1)]

println("\n--- Case: ε̂ > ε̃ (expecting q* ≠ d*) ---\n")
@printf("%-12s | %-8s | %-12s | %-10s | %-10s | %-10s\n",
    "(ε̂, ε̃)", "status", "Z₀", "TV(a*,d*)", "TV(a*,q̂)", "TV(d*,q̂)")
println("-" ^ 75)

for (ε_hat, ε_tilde) in configs_gt
    td = make_true_dro_data(net, caps, q_hat, ε_hat, ε_tilde; w=w, lambda_U=10.0, gamma=gamma)
    sub_model, sub_vars = build_true_dro_subproblem(td, x_bar;
        optimizer=Gurobi.Optimizer, rho_upper_bound=10.0)
    set_optimizer_attribute(sub_model, "NonConvex", 2)
    set_optimizer_attribute(sub_model, "TimeLimit", 600.0)
    set_optimizer_attribute(sub_model, "MIPGap", 1e-8)
    set_optimizer_attribute(sub_model, "OutputFlag", 0)
    optimize!(sub_model)

    st = termination_status(sub_model)
    Z0 = objective_value(sub_model)
    a_val = [value(sub_vars[:a][s]) for s in 1:S]
    d_val = [value(sub_vars[:d][s]) for s in 1:S]

    tv_ad = 0.5 * sum(abs.(a_val .- d_val))
    tv_a_qhat = 0.5 * sum(abs.(a_val .- q_hat))
    tv_d_qhat = 0.5 * sum(abs.(d_val .- q_hat))

    mark = tv_ad < 1e-4 ? "✓" : "✗"
    @printf("(%.1f, %.1f)   | %-8s | %12.6f | %10.6f%s | %10.6f | %10.6f\n",
        ε_hat, ε_tilde, st, Z0, tv_ad, mark, tv_a_qhat, tv_d_qhat)

    if tv_ad > 1e-4
        @printf("    a* = %s\n", string(round.(a_val; digits=5)))
        @printf("    d* = %s\n", string(round.(d_val; digits=5)))
    end
    flush(stdout)
end

println("\nDone.")
