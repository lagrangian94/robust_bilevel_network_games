# test_eps_equal.jl — ε̂=ε̃=0.2 에서 saturation 검사
# γ=2, x=[9,11], grid3x3, S=10, β=0.95
# 비교: ε̃=0.1 (기존) vs ε̃=0.1 (동일 ball)

using JuMP, Gurobi, Printf, LinearAlgebra, Random
include("../network_generator.jl"); NG = NetworkGenerator
include("true_dro_data.jl"); include("true_dro_build_omp.jl"); include("true_dro_build_subproblem.jl")

net = NG.generate_grid_network(3, 3; seed=42)
K = length(net.arcs) - 1; S = 10
caps, _ = NG.generate_capacity_scenarios_uniform_model(length(net.arcs), S;
    interdictable_arcs=net.interdictable_arcs, seed=42)
# caps, _ = NG.generate_capacity_scenarios_factor_model(length(net.arcs), S;
#     interdictable_arcs=net.interdictable_arcs, seed=42)
intd_idx = findall(net.interdictable_arcs[1:K])
w = round(maximum(caps[intd_idx, :]); digits=4)
q_hat = fill(1.0/S, S); λU = 10.0

# 모든 arc interdictable
intd_all = fill(true, length(net.arcs))
net = NG.GridNetworkData(net.m, net.n, net.nodes, net.arcs, net.N,
    intd_all, net.arc_directions, net.arc_adjacency, net.node_arc_incidence)

# v ~ Bernoulli(0.75)
Random.seed!(42)
v_rand = zeros(K, S)
for k in 1:K, s in 1:S
    v_rand[k, s] = rand() < 0.75 ? 1.0 : 0.0
end

γ = 2; x_list = [9,11]
x_bar = zeros(K); for k in x_list; x_bar[k] = 1.0; end


eps_hat = 0.1
for (label, eps_tilde) in [
    ("ε̂=0.2, ε̃=0.5 (기존, ε̂<ε̃)", eps_hat+0.1),
    ("ε̂=0.2, ε̃=0.2 (동일 ball)",   eps_hat),
]

td = make_true_dro_data(net, caps, q_hat, 0.1, eps_tilde;
    w=w, lambda_U=λU, gamma=γ, beta=0.75, v_scenarios=v_rand)

println("\n" * "="^80)
println(label)
@printf("γ=%d, x=%s, w=%.1f, λU=%.1f, β=0.95, ε̂=%.2f, ε̃=%.2f\n",
    γ, x_list, w, λU, eps_hat, eps_tilde)
println("="^80)

# (1) free
println("\n--- FREE (a≠d) ---")
m1, v1 = build_true_dro_subproblem(td, x_bar; optimizer=Gurobi.Optimizer, silent=false)
set_optimizer_attribute(m1, "NonConvex", 2)
JuMP.optimize!(m1)
Z_free = objective_value(m1)
a_f = [value(v1[:a][s]) for s in 1:S]
d_f = [value(v1[:d][s]) for s in 1:S]
α_f = [value(v1[:α][k]) for k in 1:K]
@printf("  Z_free = %.6f\n", Z_free)
@printf("  a = %s\n", [round(v; digits=4) for v in a_f])
@printf("  d = %s\n", [round(v; digits=4) for v in d_f])
@printf("  ||a-d||₁ = %.6f\n", sum(abs.(a_f .- d_f)))
α_nz = findall(abs.(α_f) .> 1e-6)
@printf("  α: [%s]  (sum=%.4f)\n",
    join(["k$(k)=$(round(α_f[k];digits=2))" for k in α_nz], ", "), sum(α_f))

# (2) tied (a=d)
println("\n--- TIED (a=d), MIPGap=0.8% ---")
m2, v2 = build_true_dro_subproblem(td, x_bar; optimizer=Gurobi.Optimizer, silent=false)
set_optimizer_attribute(m2, "NonConvex", 2)
set_optimizer_attribute(m2, "MIPGap", 0.008)
@constraint(m2, tied[s=1:S], v2[:a][s] == v2[:d][s])
JuMP.optimize!(m2)
Z_tied = objective_value(m2)
best_bd = objective_bound(m2)
a_t = [value(v2[:a][s]) for s in 1:S]
α_t = [value(v2[:α][k]) for k in 1:K]
@printf("  Z_tied (incumbent) = %.6f\n", Z_tied)
@printf("  best bound         = %.6f\n", best_bd)
@printf("  MIP gap            = %.4f%%\n", 100.0*abs(Z_tied-best_bd)/max(abs(Z_tied),1e-10))
@printf("  a=d = %s\n", [round(v; digits=4) for v in a_t])
α_nz_t = findall(abs.(α_t) .> 1e-6)
@printf("  α: [%s]  (sum=%.4f)\n",
    join(["k$(k)=$(round(α_t[k];digits=2))" for k in α_nz_t], ", "), sum(α_t))

# (3) comparison
println("\n--- COMPARISON ---")
@printf("  Z_free       = %.6f\n", Z_free)
@printf("  Z_tied (inc) = %.6f\n", Z_tied)
@printf("  Z_tied (bd)  = %.6f\n", best_bd)
@printf("  gap (free - tied_inc) = %.6f\n", Z_free - Z_tied)
if best_bd < Z_free - 0.1
    println("  → SATURATION BROKEN (a≠d matters)")
else
    println("  → saturation holds (or gap < 0.1)")
end

end  # for loop
