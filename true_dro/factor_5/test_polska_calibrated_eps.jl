"""
test_polska_calibrated_eps.jl — polska, all-intd, γ=1, λU=10.0, ε=0.148148 (calibrated)
Nominal: Benders ε=0, Robust: Benders ε=0.148148
"""

using JuMP, Gurobi, Printf, Dates, Serialization, LinearAlgebra

include("../../network_generator.jl")
using .NetworkGenerator

include("../true_dro_data.jl")
include("../true_dro_build_omp.jl")
include("../true_dro_build_subproblem.jl")
include("../true_dro_build_isp_leader.jl")
include("../true_dro_build_isp_follower.jl")
include("../true_dro_benders.jl")
include("../true_dro_mincut_vi.jl")

net = generate_polska_network()
num_arcs = length(net.arcs) - 1

all_intd = fill(true, length(net.arcs))
net_mod = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
    net.N, all_intd, net.arc_adjacency, net.node_arc_incidence)

caps, _ = generate_capacity_scenarios_factor_sparse(length(net_mod.arcs), 20;
    interdictable_arcs=all_intd, seed=42, num_factors=5)
intd_idx = findall(all_intd[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
γ = 1
S = 20
λU = 10.0
ε_cal = 0.148148
q_hat = fill(1.0/S, S)

@printf("polska ALL intd k=5: arcs=%d, intd=%d, γ=%d, λU=%.1f, ε=%.6f, w=%.4f\n",
        num_arcs, length(intd_idx), γ, λU, ε_cal, w)
flush(stdout)

# Nominal (Benders ε=0)
println("\n" * "="^60)
println("Nominal (Benders ε=0, λU=10.0)")
println("="^60)
flush(stdout)

td_0 = make_true_dro_data(net_mod, caps, q_hat, 0.0, 0.0; w=w, lambda_U=λU, gamma=γ)
t0 = time()
res_nom = true_dro_benders_optimize!(td_0;
    mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
    max_iter=500, tol=1e-4, verbose=true, sub_time_limit=30.0,
    mini_benders=true, max_mini_benders_iter=5,
    strengthen_cuts=:mw, valid_inequality=:mincut,
    inexact=true, nonconvex_attr=("NonConvex" => 2))
wt_nom = time() - t0

x_nom_int = round.(Int, res_nom[:x])
@printf("\nNominal: Z₀=%.6f, iters=%d, time=%.2fs\n", res_nom[:Z0], res_nom[:iters], wt_nom)
println("x_nom arcs = $(findall(x_nom_int .> 0))")
flush(stdout)

# Calibrated DRO (ε=0.148148)
println("\n" * "="^60)
@printf("Calibrated DRO (ε=%.6f, λU=%.1f)\n", ε_cal, λU)
println("="^60)
flush(stdout)

td_cal = make_true_dro_data(net_mod, caps, q_hat, ε_cal, ε_cal; w=w, lambda_U=λU, gamma=γ)
t0 = time()
res_cal = true_dro_benders_optimize!(td_cal;
    mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
    max_iter=500, tol=1e-4, verbose=true, sub_time_limit=30.0,
    mini_benders=true, max_mini_benders_iter=5,
    strengthen_cuts=:mw, valid_inequality=:mincut,
    inexact=true, nonconvex_attr=("NonConvex" => 2))
wt_cal = time() - t0
x_cal = round.(Int, res_cal[:x])

@printf("\nCalibrated: Z₀=%.6f, iters=%d, time=%.1fs\n", res_cal[:Z0], res_cal[:iters], wt_cal)
println("x_cal arcs = $(findall(x_cal .> 0))")
flush(stdout)

println("\n" * "="^60)
println("x_nom arcs: $(findall(x_nom_int .> 0))")
println("x_cal arcs: $(findall(x_cal .> 0))")
println("SAME? $(x_nom_int == x_cal)")
println("="^60)
