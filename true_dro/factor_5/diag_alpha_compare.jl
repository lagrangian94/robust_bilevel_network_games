# diag_alpha_compare.jl — Compare α* across ε̃ values for single_F

using JuMP, Gurobi, Printf, LinearAlgebra

include("../../network_generator.jl")
NG = NetworkGenerator
include("../true_dro_data.jl")
include("../true_dro_build_omp.jl")
include("../true_dro_build_subproblem.jl")
include("../true_dro_build_isp_leader.jl")
include("../true_dro_build_isp_follower.jl")
include("../true_dro_benders.jl")

net = NG.generate_polska_network()
intd_arcs = fill(true, length(net.arcs))
net = NG.RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
    net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
num_arcs = length(net.arcs) - 1; S = 20; K = num_arcs

caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S;
    interdictable_arcs=intd_arcs, seed=42, num_factors=5)
intd_idx = findall(intd_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
q_hat = fill(1.0/S, S)
x_bar = zeros(K); x_bar[3] = 1.0; x_bar[6] = 1.0

alphas = Dict{Float64, Vector{Float64}}()

for ε_tilde in [0.0, 0.1, 0.3]
    td = make_true_dro_data(net, caps, q_hat, 0.0, ε_tilde; w=w, lambda_U=10.0, gamma=2)
    sub_model, sub_vars = build_true_dro_subproblem(td, x_bar;
        optimizer=Gurobi.Optimizer, rho_upper_bound=10.0)
    set_optimizer_attribute(sub_model, "NonConvex", 2)
    optimize!(sub_model)

    α_val = max.([value(sub_vars[:α][k]) for k in 1:K], 0.0)
    d_val = [value(sub_vars[:d][s]) for s in 1:S]
    alphas[ε_tilde] = α_val

    @printf("\n=== ε̃ = %.1f: Z₀=%.6f ===\n", ε_tilde, objective_value(sub_model))
    nz = findall(α_val .> 1e-4)
    @printf("α nonzero: ")
    for k in nz; @printf("%d(%.3f) ", k, α_val[k]); end
    println()
    @printf("Σα = %.4f, TV(d*,q̂)=%.6f\n", sum(α_val), 0.5*sum(abs.(d_val .- q_hat)))
end

println("\n=== α diff: ε̃=0.1 vs 0.0 ===")
d01 = alphas[0.1] .- alphas[0.0]
for k in findall(abs.(d01) .> 0.01)
    @printf("  arc %d: %+.4f (%.3f → %.3f)\n", k, d01[k], alphas[0.0][k], alphas[0.1][k])
end
@printf("  max|Δα| = %.6f\n", maximum(abs.(d01)))

println("=== α diff: ε̃=0.3 vs 0.0 ===")
d03 = alphas[0.3] .- alphas[0.0]
for k in findall(abs.(d03) .> 0.01)
    @printf("  arc %d: %+.4f (%.3f → %.3f)\n", k, d03[k], alphas[0.0][k], alphas[0.3][k])
end
@printf("  max|Δα| = %.6f\n", maximum(abs.(d03)))

# ── Per-scenario max-flow comparison ──
include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")

println("\n=== Per-scenario max-flow from α* ===")
println("Using α* as h (recovery), computing maxflow per scenario")

for ε_tilde in [0.0, 0.1, 0.3]
    α_val = alphas[ε_tilde]
    # h* = α* (recovery allocation)
    flows = compute_maxflow_per_scenario(net, x_bar, α_val, 1.0, caps)
    eq = dot(q_hat, flows)
    @printf("\nε̃=%.1f: E_q̂[flow]=%.6f\n", ε_tilde, eq)
    for s in 1:S
        @printf("  s=%2d: flow=%.4f\n", s, flows[s])
    end
end

# Direct diff
println("\n=== Flow diff: ε̃=0.1 vs 0.0 ===")
f0 = compute_maxflow_per_scenario(net, x_bar, alphas[0.0], 1.0, caps)
f1 = compute_maxflow_per_scenario(net, x_bar, alphas[0.1], 1.0, caps)
f3 = compute_maxflow_per_scenario(net, x_bar, alphas[0.3], 1.0, caps)
for s in 1:S
    d1 = f1[s] - f0[s]
    d3 = f3[s] - f0[s]
    if abs(d1) > 1e-4 || abs(d3) > 1e-4
        @printf("  s=%2d: nom=%.4f, ε̃01=%.4f(%+.4f), ε̃03=%.4f(%+.4f)\n",
                s, f0[s], f1[s], d1, f3[s], d3)
    end
end
@printf("E_q̂: nom=%.6f, ε̃01=%.6f(%+.6f), ε̃03=%.6f(%+.6f)\n",
        dot(q_hat,f0), dot(q_hat,f1), dot(q_hat,f1)-dot(q_hat,f0),
        dot(q_hat,f3), dot(q_hat,f3)-dot(q_hat,f0))
