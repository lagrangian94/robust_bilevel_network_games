using JuMP, Gurobi, Printf, LinearAlgebra
include("../network_generator.jl"); NG = NetworkGenerator
include("true_dro_data.jl"); include("true_dro_build_omp.jl"); include("true_dro_build_subproblem.jl")

net = NG.generate_grid_network(3, 3; seed=42)
num_arcs = length(net.arcs) - 1; S = 3; γ = 2
caps, _ = NG.generate_capacity_scenarios_uniform_model(length(net.arcs), S;
    interdictable_arcs=net.interdictable_arcs, seed=42)
intd_idx = findall(net.interdictable_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
q_hat = fill(1.0/S, S)

x_bar = zeros(num_arcs); x_bar[3] = 1.0; x_bar[8] = 1.0

for λU in [2.0, 10.0, 50.0]
    td = make_true_dro_data(net, caps, q_hat, 0.2, 0.2; w=w, lambda_U=λU, gamma=γ, beta=0.95)
    
    # free
    m1, v1 = build_true_dro_subproblem(td, x_bar; optimizer=Gurobi.Optimizer, silent=true)
    set_optimizer_attribute(m1, "NonConvex", 2); JuMP.optimize!(m1)
    Z_free = objective_value(m1)
    a_free = [value(v1[:a][s]) for s in 1:S]
    d_free = [value(v1[:d][s]) for s in 1:S]
    α_free = [value(v1[:α][k]) for k in 1:num_arcs]
    
    # tied
    m2, v2 = build_true_dro_subproblem(td, x_bar; optimizer=Gurobi.Optimizer, silent=true)
    set_optimizer_attribute(m2, "NonConvex", 2)
    for s in 1:S; @constraint(m2, v2[:a][s] == v2[:d][s]); end
    JuMP.optimize!(m2)
    Z_tied = objective_value(m2)
    α_tied = [value(v2[:α][k]) for k in 1:num_arcs]
    
    gap = Z_free - Z_tied
    α_diff = sum(abs.(α_free .- α_tied))
    @printf("\nλU=%.0f: Z_free=%.6f  Z_tied=%.6f  gap=%.6f  ||α_diff||₁=%.6f  ||a-d||₁(free)=%.6f\n",
        λU, Z_free, Z_tied, gap, α_diff, sum(abs.(a_free .- d_free)))
end
