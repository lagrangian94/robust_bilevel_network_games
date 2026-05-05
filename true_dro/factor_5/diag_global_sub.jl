# diag_global_sub.jl — Global subproblem solve for single_F, sample8

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

x_bar = zeros(K); x_bar[3] = 1.0; x_bar[6] = 1.0

for ε_tilde in [0.0, 0.1, 0.3]
    td = make_true_dro_data(net, caps, q_hat, 0.0, ε_tilde; w=w, lambda_U=10.0, gamma=2)

    # Global solve (NonConvex=2, long time limit)
    sub_model, sub_vars = build_true_dro_subproblem(td, x_bar;
        optimizer=Gurobi.Optimizer, rho_upper_bound=10.0)
    set_optimizer_attribute(sub_model, "NonConvex", 2)
    set_optimizer_attribute(sub_model, "TimeLimit", 300.0)
    set_optimizer_attribute(sub_model, "MIPGap", 1e-6)
    optimize!(sub_model)

    st = termination_status(sub_model)
    Z0 = objective_value(sub_model)
    α_val = max.([value(sub_vars[:α][k]) for k in 1:K], 0.0)
    d_val = [value(sub_vars[:d][s]) for s in 1:S]

    @printf("\n=== ε̃=%.1f, status=%s ===\n", ε_tilde, st)
    @printf("Z₀ = %.6f\n", Z0)
    if st != MOI.OPTIMAL
        @printf("Best bound = %.6f\n", objective_bound(sub_model))
    end
    @printf("TV(d*, q̂) = %.6f\n", 0.5 * sum(abs.(d_val .- q_hat)))

    nz = findall(α_val .> 1e-4)
    @printf("α nonzero: ")
    for k in nz
        @printf("%d(%.3f) ", k, α_val[k])
    end
    println()
    @printf("Σα = %.4f\n", sum(α_val))

    # interdicted arcs detail
    @printf("  arc 3 (INTD): α=%.4f\n", α_val[3])
    @printf("  arc 6 (INTD): α=%.4f\n", α_val[6])
end
