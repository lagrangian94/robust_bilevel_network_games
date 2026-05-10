"""
test_nominal_lp_check.jl — nominal compact subproblem이 LP로 인식되는지 확인
Gurobi 로그에서 model type (LP vs QCP vs MIQCP) 확인
"""

using JuMP, Gurobi, Printf, Random

include("../network_generator.jl")
NG = NetworkGenerator

include("true_dro_data.jl")
include("true_dro_build_subproblem.jl")

# ── small grid3x3 ──
net = NG.generate_grid_network(3, 3; seed=42)
S = 3
K = length(net.arcs) - 1
intd_arcs = net.interdictable_arcs

caps, _ = NG.generate_capacity_scenarios_uniform_model(length(net.arcs), S;
    interdictable_arcs=intd_arcs, seed=42)
intd_idx = findall(intd_arcs[1:K])
w = round(maximum(caps[intd_idx, :]); digits=4)
q_hat = fill(1.0/S, S)

Random.seed!(42)
v_rand = zeros(K, S)
for k in 1:K, s in 1:S
    v_rand[k, s] = intd_arcs[k] ? (rand() < 0.75 ? 1.0 : 0.0) : 0.0
end

td = make_true_dro_data(net, caps, q_hat, 0.0, 0.0;
    w=w, lambda_U=10.0, gamma=ceil(Int, K * 0.1), v_scenarios=v_rand)

x_init = zeros(K)

# ── Build nominal compact subproblem ──
println("=" ^ 60)
println("Building nominal compact subproblem (ε̂=0, ε̃=0)")
println("=" ^ 60)

sub_model, sub_vars = build_true_dro_subproblem_nominal(td, x_init;
    optimizer=Gurobi.Optimizer, silent=false)

# Do NOT set NonConvex=2
println("\n--- Solving WITHOUT NonConvex=2 ---")
optimize!(sub_model)
println("Termination: ", termination_status(sub_model))
println("Objective:   ", objective_value(sub_model))

# ── Compare: with NonConvex=2 ──
println("\n" * "=" ^ 60)
println("Building same model WITH NonConvex=2")
println("=" ^ 60)

sub_model2, _ = build_true_dro_subproblem_nominal(td, x_init;
    optimizer=Gurobi.Optimizer, silent=false)
set_optimizer_attribute(sub_model2, "NonConvex", 2)

println("\n--- Solving WITH NonConvex=2 ---")
optimize!(sub_model2)
println("Termination: ", termination_status(sub_model2))
println("Objective:   ", objective_value(sub_model2))

println("\n" * "=" ^ 60)
println("Done. Compare Gurobi log output above for model type (LP vs MIQCP).")
println("=" ^ 60)
