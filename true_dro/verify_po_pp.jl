using JuMP, Gurobi, Printf, LinearAlgebra

include("../network_generator.jl")
using .NetworkGenerator
include("true_dro_data.jl")
include("true_dro_build_subproblem.jl")

network = generate_grid_network(5, 5; seed=42)
num_arcs = length(network.arcs) - 1
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, 0.10 * num_interdictable)
S = 20
capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
    interdictable_arcs=network.interdictable_arcs, seed=42)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
w = round(maximum(capacities[interdictable_idx, :]); digits=4)
q_hat = fill(1.0/S, S)
λU = 2.0

x_neut = Float64.([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
x_rob  = Float64.([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])

eps_list = [0.5, 0.75, 0.875, 0.8828]

@printf("%-8s  %12s  %12s  %12s  %8s\n", "eps", "f_eps_neut", "f_eps_rob", "PO-PP", "result")
println("-"^62)

for ε in eps_list
    td = make_true_dro_data(network, capacities, q_hat, ε, ε; w=w, lambda_U=λU, gamma=γ)
    mdl, vars = build_true_dro_subproblem(td, zeros(num_arcs);
        optimizer=optimizer_with_attributes(Gurobi.Optimizer, MOI.Silent() => true, "NonConvex" => 2),
        rho_upper_bound=10.0)
    set_optimizer_attribute(mdl, "TimeLimit", 3600.0)
    set_optimizer_attribute(mdl, "MIPGap", 0.005)

    sn = solve_true_dro_subproblem!(mdl, vars, td, x_neut)
    f_neut = sn[:Z0_val]
    opt_n = sn[:is_optimal]

    sr = solve_true_dro_subproblem!(mdl, vars, td, x_rob)
    f_rob = sr[:Z0_val]
    opt_r = sr[:is_optimal]

    po_pp = f_neut - f_rob
    sign_str = po_pp >= 0 ? ">=0 WRONG" : "<0 ok"
    tl_str = (opt_n && opt_r) ? "" : " [TL]"
    @printf("%-8.4f  %12.6f  %12.6f  %12.6f  %s%s\n", ε, f_neut, f_rob, po_pp, sign_str, tl_str)
    flush(stdout)
end
