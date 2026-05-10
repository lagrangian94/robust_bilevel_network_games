"""
test_alpine_compare.jl — Alpine.jl vs Gurobi NonConvex=2 비교
  polska network, S=10, q̂=uniform, x 고정
  동일 subproblem을 Alpine과 Gurobi로 각각 풀어서 비교.

Usage:
  julia test_alpine_compare.jl
"""

using JuMP, Gurobi, Alpine, Printf, LinearAlgebra, Random
using Ipopt

include("../network_generator.jl")
NG = NetworkGenerator

include("true_dro_data.jl")
include("true_dro_build_subproblem.jl")

# ── network setup (polska, same as test_network.jl) ──
net = NG.generate_polska_network()
intd_arcs = fill(true, length(net.arcs))
net = NG.RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
    net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)

num_arcs = length(net.arcs) - 1
γ = ceil(Int, num_arcs * 0.1)
S = 10
λU = 10.0

caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S;
    interdictable_arcs=intd_arcs, seed=42, num_factors=5)
# caps, _ = NG.generate_capacity_scenarios_uniform_model(length(net.arcs), S;
#     interdictable_arcs=intd_arcs, seed=42)
intd_idx = findall(intd_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
q_hat = fill(1.0/S, S)

# v_scenarios
Random.seed!(42)
v_rand = zeros(num_arcs, S)
for k in 1:num_arcs, s in 1:S
    v_rand[k, s] = intd_arcs[k] ? (rand() < 0.75 ? 1.0 : 0.0) : 0.0
end

β_risk = 0.4
ε_hat = 0.2
ε_tilde = 0.2

@printf("polska: arcs=%d, intd=%d, γ=%d, w=%.4f, λU=%.1f, β=%.2f\n",
        num_arcs, length(intd_idx), γ, w, λU, β_risk)
@printf("ε̂=%.2f, ε̃=%.2f, S=%d, q̂=uniform\n", ε_hat, ε_tilde, S)

td = make_true_dro_data(net, caps, q_hat, ε_hat, ε_tilde;
    w=w, lambda_U=λU, gamma=γ, beta=β_risk, v_scenarios=v_rand)

# ── fixed x ──
x_bar = Float64[0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
@printf("x = %s\n\n", findall(x_bar .> 0))

# ══════════════════════════════════════════════════════
# 1. Gurobi NonConvex=2
# ══════════════════════════════════════════════════════
println("=" ^ 60)
println("Solver 1: Gurobi NonConvex=2")
println("=" ^ 60)

sub_model_g, sub_vars_g = build_true_dro_subproblem(td, x_bar;
    optimizer=Gurobi.Optimizer, silent=false)
set_optimizer_attribute(sub_model_g, "NonConvex", 2)
set_optimizer_attribute(sub_model_g, "LogToConsole", 1)
set_optimizer_attribute(sub_model_g, "TimeLimit", 120.0)

t_gurobi = @elapsed optimize!(sub_model_g)
st_g = termination_status(sub_model_g)
Z_gurobi = st_g in (MOI.OPTIMAL, MOI.TIME_LIMIT) ? objective_value(sub_model_g) : NaN
Z_gurobi_bound = st_g in (MOI.OPTIMAL, MOI.TIME_LIMIT) ? objective_bound(sub_model_g) : NaN

@printf("\n[Gurobi] status=%s, Z₀=%.6f, bound=%.6f, time=%.1fs\n",
        st_g, Z_gurobi, Z_gurobi_bound, t_gurobi)
if st_g in (MOI.OPTIMAL, MOI.TIME_LIMIT)
    α_g = [value(sub_vars_g[:α][k]) for k in 1:num_arcs]
    α_nz = findall(abs.(α_g) .> 1e-4)
    @printf("  α nonzero: %s\n", ["k$(k)=$(round(α_g[k];digits=2))" for k in α_nz])
end
flush(stdout)

# ══════════════════════════════════════════════════════
# 2. Alpine (Gurobi MIP + Gurobi NLP)
# ══════════════════════════════════════════════════════
println("\n" * "=" ^ 60)
println("Solver 2: Alpine (Gurobi MIP + Gurobi NLP)")
println("=" ^ 60)

alpine_opt = optimizer_with_attributes(
    Alpine.Optimizer,
    # "nlp_solver" => optimizer_with_attributes(
    #     Gurobi.Optimizer,
    #     MOI.Silent() => true,
    #     "NonConvex" => 2,
    #     "OptimalityTarget" => 1,
    # ),
    "nlp_solver" => optimizer_with_attributes(
        Ipopt.Optimizer,
        MOI.Silent() => true,
    ),
    "mip_solver" => optimizer_with_attributes(
        Gurobi.Optimizer,
        MOI.Silent() => true,
    ),
    "log_level" => 100,
    # "time_limit" => 120.0,
    "max_iter" => 99,
    "presolve_bt" => true,
)

sub_model_a, sub_vars_a = build_true_dro_subproblem(td, x_bar;
    optimizer=alpine_opt, silent=false)

t_alpine = @elapsed optimize!(sub_model_a)
st_a = termination_status(sub_model_a)
Z_alpine = st_a in (MOI.OPTIMAL, MOI.TIME_LIMIT, MOI.LOCALLY_SOLVED) ? objective_value(sub_model_a) : NaN
Z_alpine_bound = st_a in (MOI.OPTIMAL, MOI.TIME_LIMIT, MOI.LOCALLY_SOLVED) ? objective_bound(sub_model_a) : NaN

@printf("\n[Alpine] status=%s, Z₀=%.6f, bound=%.6f, time=%.1fs\n",
        st_a, Z_alpine, Z_alpine_bound, t_alpine)
if st_a in (MOI.OPTIMAL, MOI.TIME_LIMIT, MOI.LOCALLY_SOLVED)
    α_a = [value(sub_vars_a[:α][k]) for k in 1:num_arcs]
    α_nz_a = findall(abs.(α_a) .> 1e-4)
    @printf("  α nonzero: %s\n", ["k$(k)=$(round(α_a[k];digits=2))" for k in α_nz_a])
end
flush(stdout)

# ══════════════════════════════════════════════════════
# 비교 요약
# ══════════════════════════════════════════════════════
println("\n" * "=" ^ 60)
println("Summary")
println("=" ^ 60)
@printf("  %-12s  %12s  %12s  %8s\n", "Solver", "Z₀ (inc)", "Bound", "Time")
@printf("  %-12s  %12.6f  %12.6f  %7.1fs\n", "Gurobi", Z_gurobi, Z_gurobi_bound, t_gurobi)
@printf("  %-12s  %12.6f  %12.6f  %7.1fs\n", "Alpine", Z_alpine, Z_alpine_bound, t_alpine)
@printf("  Gap (Z₀): %.6f\n", abs(Z_gurobi - Z_alpine))
