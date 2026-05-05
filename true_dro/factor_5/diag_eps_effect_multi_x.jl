# diag_eps_effect_multi_x.jl — Check if Z₀(x) changes with ε̃ for various x

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

# Test multiple x configurations (from Benders iterations)
x_configs = [
    ([3, 6],   "optimal"),
    ([17, 34], "iter3-UB"),
    ([13, 34], "iter1"),
    ([14, 34], "iter2"),
    ([5, 6],   "iter13"),
    ([22, 33], "iter15"),
    ([9, 22],  "iter16"),
    ([14, 22], "iter9"),
]

eps_vals = [0.0, 1.0]

println("=" ^ 70)
println("Comparing Z₀(x) for ε̃=0.0 vs ε̃=1.0 across multiple x")
println("=" ^ 70)

@printf("%-15s | %-12s | %-12s | %-12s\n", "x", "Z₀(ε̃=0)", "Z₀(ε̃=1)", "ΔZ₀")
println("-" ^ 60)

for (arcs, label) in x_configs
    x_bar = zeros(K)
    for a in arcs; x_bar[a] = 1.0; end

    z_vals = Float64[]
    for ε_tilde in eps_vals
        td = make_true_dro_data(net, caps, q_hat, 0.0, ε_tilde; w=w, lambda_U=10.0, gamma=2)
        sub_model, sub_vars = build_true_dro_subproblem(td, x_bar;
            optimizer=Gurobi.Optimizer, rho_upper_bound=10.0)
        set_optimizer_attribute(sub_model, "NonConvex", 2)
        set_optimizer_attribute(sub_model, "TimeLimit", 300.0)
        set_optimizer_attribute(sub_model, "MIPGap", 1e-6)
        set_optimizer_attribute(sub_model, "OutputFlag", 0)
        optimize!(sub_model)
        push!(z_vals, objective_value(sub_model))
    end

    delta = z_vals[2] - z_vals[1]
    @printf("%-15s | %12.6f | %12.6f | %+12.6f\n", "$arcs ($label)", z_vals[1], z_vals[2], delta)
    flush(stdout)
end

println("\nDone.")
