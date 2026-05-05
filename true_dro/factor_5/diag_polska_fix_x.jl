"""
diag_polska_fix_x.jl — x_nom을 robust(ε=1.0)에 고정해서 subproblem Z₀ 비교
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
net = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
    net.N, all_intd, net.arc_adjacency, net.node_arc_incidence)

caps, _ = generate_capacity_scenarios_factor_additive(length(net.arcs), 20;
    interdictable_arcs=all_intd, seed=42, num_factors=5)
intd_idx = findall(all_intd[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
γ = 1
S = 20
λU = 10.0
q_hat = fill(1.0/S, S)

@printf("polska k=5: arcs=%d, intd=%d, γ=%d, λU=%.1f, w=%.4f\n", num_arcs, length(intd_idx), γ, λU, w)

# ── 1) Nominal solve ──
println("\n=== Nominal (ε=0) ===")
td_0 = make_true_dro_data(net, caps, q_hat, 0.0, 0.0; w=w, lambda_U=λU, gamma=γ)
res_nom = true_dro_benders_optimize!(td_0;
    mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
    max_iter=500, tol=1e-4, verbose=false, sub_time_limit=30.0,
    mini_benders=true, max_mini_benders_iter=5,
    strengthen_cuts=:mw, valid_inequality=:mincut,
    inexact=true, nonconvex_attr=("NonConvex" => 2))
x_nom = round.(Int, res_nom[:x])
@printf("Nominal: Z₀=%.6f, x=%s\n", res_nom[:Z0], findall(x_nom .> 0))

# ── 2) Robust solve ──
println("\n=== Robust (ε=1.0) ===")
td_1 = make_true_dro_data(net, caps, q_hat, 1.0, 1.0; w=w, lambda_U=λU, gamma=γ)
res_rob = true_dro_benders_optimize!(td_1;
    mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
    max_iter=500, tol=1e-4, verbose=false, sub_time_limit=30.0,
    mini_benders=true, max_mini_benders_iter=5,
    strengthen_cuts=:mw, valid_inequality=:mincut,
    inexact=true, nonconvex_attr=("NonConvex" => 2))
x_rob = round.(Int, res_rob[:x])
@printf("Robust:  Z₀=%.6f, x=%s\n", res_rob[:Z0], findall(x_rob .> 0))

# ── 3) x_nom을 robust subproblem에 고정 ──
println("\n=== Robust subproblem with x_nom fixed ===")
td_fix = make_true_dro_data(net, caps, q_hat, 1.0, 1.0; w=w, lambda_U=λU, gamma=γ)
function build_and_solve_sub(td, x_vec)
    m, v = build_true_dro_subproblem(td, Float64.(x_vec); optimizer=Gurobi.Optimizer)
    set_optimizer_attribute(m, "NonConvex", 2)
    set_optimizer_attribute(m, "TimeLimit", 60.0)
    return solve_true_dro_subproblem!(m, v, td, Float64.(x_vec); is_global=true)
end

sub_res = build_and_solve_sub(td_fix, x_nom)
@printf("Robust sub(x_nom): Z₀=%.6f\n", sub_res[:Z0_val])

# ── 4) x_rob를 robust subproblem에 고정 ──
println("\n=== Robust subproblem with x_rob fixed ===")
sub_res2 = build_and_solve_sub(td_fix, x_rob)
@printf("Robust sub(x_rob): Z₀=%.6f\n", sub_res2[:Z0_val])

# ── 5) capacity 통계 ──
println("\n=== Capacity statistics ===")
cap_intd = caps[intd_idx, :]
@printf("cap range: [%.4f, %.4f]\n", minimum(cap_intd), maximum(cap_intd))
@printf("cap mean:  %.4f, std: %.4f\n", mean(cap_intd), std(cap_intd))
@printf("cap per-arc std (mean): %.4f\n", mean(std(cap_intd, dims=2)))

println("\n=== Summary ===")
@printf("Z₀(nom)          = %.6f\n", res_nom[:Z0])
@printf("Z₀(rob)          = %.6f\n", res_rob[:Z0])
@printf("Z₀(rob|x_nom)    = %.6f\n", sub_res[:Z0_val])
@printf("Z₀(rob|x_rob)    = %.6f\n", sub_res2[:Z0_val])
@printf("Δ(rob - nom)      = %.6f\n", res_rob[:Z0] - res_nom[:Z0])
@printf("Δ(rob|x_nom - rob|x_rob) = %.6f\n", sub_res[:Z0_val] - sub_res2[:Z0_val])
flush(stdout)
