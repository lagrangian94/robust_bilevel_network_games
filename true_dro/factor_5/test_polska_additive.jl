"""
test_polska_sparse.jl — polska, sparse factor (70% zero), k=5, γ=1
Nominal SP (build_full_2SP_model) vs Robust DRO (Benders ε=1.0)
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

_nominal_sp_src = read(joinpath(@__DIR__, "..", "..", "build_nominal_sp.jl"), String)
_func_start = findfirst("function build_full_2SP_model", _nominal_sp_src)
_func_str = _nominal_sp_src[first(_func_start):end]
include_string(Main, _func_str)

net = generate_polska_network()
num_arcs = length(net.arcs) - 1
caps, F_mat = generate_capacity_scenarios_factor_sparse(length(net.arcs), 20;
    interdictable_arcs=net.interdictable_arcs, seed=42, num_factors=5)
intd_idx = findall(net.interdictable_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
γ = 1
S = 20
λU = 2.0
q_hat = fill(1.0/S, S)

@printf("polska sparse k=5: arcs=%d, intd=%d, γ=%d, w=%.4f\n", num_arcs, length(intd_idx), γ, w)
flush(stdout)

# Nominal SP
println("\n" * "="^60)
println("Nominal SP (MILP)")
println("="^60)
flush(stdout)

xi_bar_vecs = [caps[1:num_arcs, s] for s in 1:S]
uncertainty_set = Dict(:xi_bar => xi_bar_vecs, :R => zeros(0, 0),
                       :r_dict_hat => Dict(), :epsilon_hat => 0.0)
model_nom, vars_nom = build_full_2SP_model(net, S, λU, λU, γ, w, 1.0, uncertainty_set)
set_optimizer_attribute(model_nom, "OutputFlag", 1)

t0 = time()
optimize!(model_nom)
wt_nom = time() - t0

x_nom = Float64.(round.(Int, value.(vars_nom[:x])))
Z0_nom = objective_value(model_nom)
x_nom_int = round.(Int, x_nom)

@printf("\nNominal SP: Z₀=%.6f, time=%.2fs\n", Z0_nom, wt_nom)
println("x_nom arcs = $(findall(x_nom_int .> 0))")
flush(stdout)

# Robust DRO
println("\n" * "="^60)
println("Robust DRO (ε=1.0)")
println("="^60)
flush(stdout)

td_1 = make_true_dro_data(net, caps, q_hat, 1.0, 1.0; w=w, lambda_U=λU, gamma=γ)
t0 = time()
res_rob = true_dro_benders_optimize!(td_1;
    mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
    max_iter=500, tol=1e-4, verbose=true, sub_time_limit=30.0,
    mini_benders=true, max_mini_benders_iter=5,
    strengthen_cuts=:mw, valid_inequality=:mincut,
    inexact=true, nonconvex_attr=("NonConvex" => 2))
wt_rob = time() - t0
x_rob = round.(Int, res_rob[:x])

@printf("\nRobust: Z₀=%.6f, iters=%d, time=%.1fs\n", res_rob[:Z0], res_rob[:iters], wt_rob)
println("x_rob arcs = $(findall(x_rob .> 0))")
flush(stdout)

println("\n" * "="^60)
println("x_nom arcs: $(findall(x_nom_int .> 0))")
println("x_rob arcs: $(findall(x_rob .> 0))")
println("SAME? $(x_nom_int == x_rob)")
println("="^60)
