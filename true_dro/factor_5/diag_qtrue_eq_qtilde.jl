# diag_qtrue_eq_qtilde.jl — Verify q*(leader) = d*(follower) when ε̂ ≤ ε̃
# User's mathematical finding: in double-layer DRO, zero-sum structure forces
# the leader's worst-case q_true and follower's worst-case q_tilde to coincide.

using JuMP, Gurobi, Printf, LinearAlgebra

include("../../network_generator.jl")
NG = NetworkGenerator
include("../true_dro_data.jl")
include("../true_dro_build_omp.jl")
include("../true_dro_build_subproblem.jl")
include("../true_dro_build_isp_leader.jl")
include("../true_dro_build_isp_follower.jl")
include("../true_dro_benders.jl")
include("qhat_samples.jl")

net = NG.generate_polska_network()
intd_arcs = fill(true, length(net.arcs))
net = NG.RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
    net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
num_arcs = length(net.arcs) - 1; S = 20; K = num_arcs

caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S;
    interdictable_arcs=intd_arcs, seed=42, num_factors=5)
intd_idx = findall(intd_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
q_hat = QHAT_SAMPLES["sample8"] ./ sum(QHAT_SAMPLES["sample8"])

# Use optimal x from NOM
x_bar = zeros(K); x_bar[3] = 1.0; x_bar[6] = 1.0

# Test configurations: (ε̂, ε̃) with ε̂ ≤ ε̃
configs = [
    (0.1, 0.1),
    (0.1, 0.3),
    (0.1, 1.0),
    (0.3, 0.3),
    (0.3, 1.0),
    (0.5, 0.5),
    (0.5, 1.0),
]

println("=" ^ 80)
println("Verifying q*(leader) = d*(follower) in double-layer DRO (ε̂ ≤ ε̃)")
println("Network: Polska, x=[3,6], q̂=sample8")
println("=" ^ 80)

for (ε_hat, ε_tilde) in configs
    td = make_true_dro_data(net, caps, q_hat, ε_hat, ε_tilde; w=w, lambda_U=10.0, gamma=2)
    sub_model, sub_vars = build_true_dro_subproblem(td, x_bar;
        optimizer=Gurobi.Optimizer, rho_upper_bound=10.0)
    set_optimizer_attribute(sub_model, "NonConvex", 2)
    set_optimizer_attribute(sub_model, "TimeLimit", 300.0)
    set_optimizer_attribute(sub_model, "MIPGap", 1e-6)
    set_optimizer_attribute(sub_model, "OutputFlag", 0)
    optimize!(sub_model)

    st = termination_status(sub_model)
    Z0 = objective_value(sub_model)

    a_val = [value(sub_vars[:a][s]) for s in 1:S]  # leader's q*
    d_val = [value(sub_vars[:d][s]) for s in 1:S]  # follower's d*

    # TV distance between a* and d*
    tv_ad = 0.5 * sum(abs.(a_val .- d_val))
    # TV distance from q̂
    tv_a_qhat = 0.5 * sum(abs.(a_val .- q_hat))
    tv_d_qhat = 0.5 * sum(abs.(d_val .- q_hat))

    @printf("\n--- (ε̂=%.2f, ε̃=%.2f) | status=%s | Z₀=%.6f ---\n", ε_hat, ε_tilde, st, Z0)
    @printf("  TV(a*, d*)   = %.8f  %s\n", tv_ad, tv_ad < 1e-4 ? "✓ EQUAL" : "✗ DIFFERENT")
    @printf("  TV(a*, q̂)   = %.6f  (budget ε̂=%.2f)\n", tv_a_qhat, ε_hat)
    @printf("  TV(d*, q̂)   = %.6f  (budget ε̃=%.2f)\n", tv_d_qhat, ε_tilde)

    # Print first few entries for visual comparison
    @printf("  a* (first 5): ")
    for s in 1:min(5, S); @printf("%.5f ", a_val[s]); end
    println()
    @printf("  d* (first 5): ")
    for s in 1:min(5, S); @printf("%.5f ", d_val[s]); end
    println()
    @printf("  q̂  (first 5): ")
    for s in 1:min(5, S); @printf("%.5f ", q_hat[s]); end
    println()

    flush(stdout)
end

# Also test ε̂ > ε̃ (should NOT hold)
println("\n" * "=" ^ 80)
println("Counter-check: ε̂ > ε̃ (should NOT have q* = d*)")
println("=" ^ 80)

for (ε_hat, ε_tilde) in [(0.3, 0.1), (1.0, 0.1)]
    td = make_true_dro_data(net, caps, q_hat, ε_hat, ε_tilde; w=w, lambda_U=10.0, gamma=2)
    sub_model, sub_vars = build_true_dro_subproblem(td, x_bar;
        optimizer=Gurobi.Optimizer, rho_upper_bound=10.0)
    set_optimizer_attribute(sub_model, "NonConvex", 2)
    set_optimizer_attribute(sub_model, "TimeLimit", 300.0)
    set_optimizer_attribute(sub_model, "MIPGap", 1e-6)
    set_optimizer_attribute(sub_model, "OutputFlag", 0)
    optimize!(sub_model)

    st = termination_status(sub_model)
    Z0 = objective_value(sub_model)

    a_val = [value(sub_vars[:a][s]) for s in 1:S]
    d_val = [value(sub_vars[:d][s]) for s in 1:S]

    tv_ad = 0.5 * sum(abs.(a_val .- d_val))
    tv_a_qhat = 0.5 * sum(abs.(a_val .- q_hat))
    tv_d_qhat = 0.5 * sum(abs.(d_val .- q_hat))

    @printf("\n--- (ε̂=%.2f, ε̃=%.2f) | status=%s | Z₀=%.6f ---\n", ε_hat, ε_tilde, st, Z0)
    @printf("  TV(a*, d*)   = %.8f  %s\n", tv_ad, tv_ad < 1e-4 ? "✓ EQUAL" : "✗ DIFFERENT")
    @printf("  TV(a*, q̂)   = %.6f  (budget ε̂=%.2f)\n", tv_a_qhat, ε_hat)
    @printf("  TV(d*, q̂)   = %.6f  (budget ε̃=%.2f)\n", tv_d_qhat, ε_tilde)

    @printf("  a* (first 5): ")
    for s in 1:min(5, S); @printf("%.5f ", a_val[s]); end
    println()
    @printf("  d* (first 5): ")
    for s in 1:min(5, S); @printf("%.5f ", d_val[s]); end
    println()

    flush(stdout)
end

println("\nDone.")
