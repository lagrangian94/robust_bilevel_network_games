"""
test_outer_cut_compare.jl — Compare outer cut coefficients from dual ISP (variable values)
vs primal ISP (shadow prices) at the same (x,h,λ,ψ0,α).
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using LinearAlgebra
using Revise

includet("../network_generator.jl")
includet("../build_uncertainty_set.jl")
includet("../strict_benders.jl")
includet("../nested_benders_trust_region.jl")
includet("../build_primal_isp.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model

S = 1; ϕU = 2.0; λU = 10.0; γ = 2.0; w = 1.0; v = 1.0; seed = 42; epsilon = 0.5
network = generate_grid_network(3, 3, seed=seed)
capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

num_arcs = length(network.arcs) - 1
E = ones(num_arcs, num_arcs+1)
d0 = zeros(num_arcs + 1); d0[end] = 1.0
isp_data = Dict(:E => E, :network => network, :ϕU => ϕU, :λU => λU, :γ => γ,
    :w => w, :v => v, :uncertainty_set => uncertainty_set, :d0 => d0, :S => S)

# Fixed point: x with 1 interdictable arc active, λ=0
x_sol = zeros(num_arcs)
interdictable = findall(network.interdictable_arcs[1:num_arcs])
x_sol[interdictable[1]] = 1.0
h_sol = zeros(num_arcs)
λ_sol = 0.0
ψ0_sol = zeros(num_arcs)
α_sol = ones(num_arcs)

println("Test point: x_nz=$(count(x->x>0.5, x_sol)), λ=$λ_sol")
println("Interdictable arcs: $interdictable, active: $(interdictable[1])")

# === Dual ISP: extract variable values ===
dl, df = initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)

U_s_dict = Dict(:R => Dict(:1=>uncertainty_set[:R][1]), :r_dict => Dict(:1=>uncertainty_set[:r_dict][1]),
            :xi_bar => Dict(:1=>uncertainty_set[:xi_bar][1]), :epsilon => epsilon)

(st_l, _) = isp_leader_optimize!(dl[1][1], dl[1][2];
    isp_data=isp_data, uncertainty_set=U_s_dict,
    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
(st_f, _) = isp_follower_optimize!(df[1][1], df[1][2];
    isp_data=isp_data, uncertainty_set=U_s_dict,
    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)

# Dual ISP variable values
Uhat1_dual = value.(dl[1][2][:Uhat1])
Uhat3_dual = value.(dl[1][2][:Uhat3])
Utilde1_dual = value.(df[1][2][:Utilde1])
Utilde3_dual = value.(df[1][2][:Utilde3])
Ztilde1_3_dual = value.(df[1][2][:Ztilde1_3])
βtilde1_1_dual = value.(df[1][2][:βtilde1_1])
βtilde1_3_dual = value.(df[1][2][:βtilde1_3])
intercept_l_dual = value.(dl[1][2][:intercept])
intercept_f_dual = value.(df[1][2][:intercept])

println("\n=== Dual ISP (variable values) ===")
println("  leader obj = ", round(objective_value(dl[1][1]), digits=4))
println("  follower obj = ", round(objective_value(df[1][1]), digits=4))
println("  Uhat1 norm = ", round(norm(Uhat1_dual), digits=6))
println("  Uhat3 norm = ", round(norm(Uhat3_dual), digits=6))
println("  Utilde1 norm = ", round(norm(Utilde1_dual), digits=6))
println("  Utilde3 norm = ", round(norm(Utilde3_dual), digits=6))
println("  βtilde1_1 = ", round.(βtilde1_1_dual, digits=4))
println("  βtilde1_3 = ", round.(βtilde1_3_dual, digits=4))
println("  intercept_l = ", round(intercept_l_dual, digits=4))
println("  intercept_f = ", round(intercept_f_dual, digits=4))

# === Primal ISP: extract shadow prices ===
pl, pf = initialize_primal_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)

# Set α and solve
μhat_p = pl[1][2][:μhat]
for k in 1:num_arcs
    set_objective_coefficient(pl[1][1], μhat_p[1, k], α_sol[k])
end
optimize!(pl[1][1])

μtilde_p = pf[1][2][:μtilde]
for k in 1:num_arcs
    set_objective_coefficient(pf[1][1], μtilde_p[1, k], α_sol[k])
end
optimize!(pf[1][1])

# Primal ISP shadow prices
vars_l = pl[1][2]
vars_f = pf[1][2]
Uhat1_primal = -dual.(vars_l[:con_bigM1_hat])
Uhat3_primal = -dual.(vars_l[:con_bigM3_hat])
Utilde1_primal = -dual.(vars_f[:con_bigM1_tilde])
Utilde3_primal = -dual.(vars_f[:con_bigM3_tilde])

b3s = vars_f[:block3_start_idx]
b3e = vars_f[:block3_end_idx]
b1sz = vars_f[:block1_size]
eq_duals = dual.(vars_f[:con_soc_eq_tilde])
ineq_duals = dual.(vars_f[:con_soc_ineq_tilde])
Ztilde1_3_primal = eq_duals[b3s:b3e, :]
βtilde1_1_primal = ineq_duals[1:b1sz]
βtilde1_3_primal = ineq_duals[b3s:b3e]

println("\n=== Primal ISP (shadow prices) ===")
println("  leader obj = ", round(objective_value(pl[1][1]), digits=4))
println("  follower obj = ", round(objective_value(pf[1][1]), digits=4))
println("  Uhat1 norm = ", round(norm(Uhat1_primal), digits=6))
println("  Uhat3 norm = ", round(norm(Uhat3_primal), digits=6))
println("  Utilde1 norm = ", round(norm(Utilde1_primal), digits=6))
println("  Utilde3 norm = ", round(norm(Utilde3_primal), digits=6))
println("  βtilde1_1 = ", round.(βtilde1_1_primal, digits=4))
println("  βtilde1_3 = ", round.(βtilde1_3_primal, digits=4))

# === Differences ===
println("\n=== Differences (primal - dual) ===")
println("  Uhat1 diff norm = ", round(norm(Uhat1_primal - Uhat1_dual[1:1,:,:]), digits=6))
println("  Uhat3 diff norm = ", round(norm(Uhat3_primal - Uhat3_dual[1:1,:,:]), digits=6))
println("  Utilde1 diff norm = ", round(norm(Utilde1_primal - Utilde1_dual[1:1,:,:]), digits=6))
println("  Utilde3 diff norm = ", round(norm(Utilde3_primal - Utilde3_dual[1:1,:,:]), digits=6))
println("  Ztilde1_3 diff norm = ", round(norm(Ztilde1_3_primal - Ztilde1_3_dual[1:1,:,:]), digits=6))
println("  βtilde1_1 diff = ", round.(βtilde1_1_primal - βtilde1_1_dual[1:1,:], digits=4))
println("  βtilde1_3 diff = ", round.(βtilde1_3_primal - βtilde1_3_dual[1:1,:], digits=4))

# === Cut evaluation at a different point ===
# Evaluate both cuts at (x'=0, λ'=5, h'=0, ψ0'=0) — see how much they differ
x_test = zeros(num_arcs)
λ_test = 5.0
h_test = zeros(num_arcs)
ψ0_test = zeros(num_arcs)

diag_x_E_test = Diagonal(x_test) * E
diag_λ_ψ_test = Diagonal(λ_test * ones(num_arcs) - v .* ψ0_test)

function eval_cut(Uhat1, Uhat3, Utilde1, Utilde3, Ztilde1_3, βtilde1_1, βtilde1_3, int_l, int_f, x_e, λψ, h_v, λ_v)
    ct1_l = -ϕU * sum(Uhat1 .* x_e)
    ct2_l = -ϕU * sum(Uhat3 .* (E - x_e))
    ct1_f = -ϕU * sum(Utilde1 .* x_e)
    ct2_f = -ϕU * sum(Utilde3 .* (E - x_e))
    ct3 = sum(Ztilde1_3 .* (λψ * diagm(xi_bar[1])))
    ct4 = (d0' * vec(βtilde1_1)) * λ_v
    ct5 = -(h_v + λψ * xi_bar[1])' * vec(βtilde1_3)
    return (ct1_l + ct2_l + int_l) + (ct1_f + ct2_f + ct3 + ct4 + ct5 + int_f)
end

# Need intercept for primal — compute residually
diag_x_E_star = Diagonal(x_sol) * E
diag_λ_ψ_star = Diagonal(λ_sol * ones(num_arcs) - v .* ψ0_sol)
ct1_l_s = -ϕU * sum(Uhat1_primal .* diag_x_E_star)
ct2_l_s = -ϕU * sum(Uhat3_primal .* (E - diag_x_E_star))
int_l_primal = objective_value(pl[1][1]) - (ct1_l_s + ct2_l_s)

ct1_f_s = -ϕU * sum(Utilde1_primal .* diag_x_E_star)
ct2_f_s = -ϕU * sum(Utilde3_primal .* (E - diag_x_E_star))
ct3_s = sum(Ztilde1_3_primal .* (diag_λ_ψ_star * diagm(xi_bar[1])))
ct4_s = (d0' * βtilde1_1_primal) * λ_sol
ct5_s = -(h_sol + diag_λ_ψ_star * xi_bar[1])' * βtilde1_3_primal
int_f_primal = objective_value(pf[1][1]) - (ct1_f_s + ct2_f_s + ct3_s + ct4_s + ct5_s)

println("\n=== Cut intercepts ===")
println("  Dual:   int_l=$(round(intercept_l_dual, digits=4)), int_f=$(round(intercept_f_dual, digits=4))")
println("  Primal: int_l=$(round(int_l_primal, digits=4)), int_f=$(round(int_f_primal, digits=4))")

cut_dual = eval_cut(Uhat1_dual[1,:,:], Uhat3_dual[1,:,:], Utilde1_dual[1,:,:], Utilde3_dual[1,:,:],
    Ztilde1_3_dual[1,:,:], βtilde1_1_dual[1,:], βtilde1_3_dual[1,:],
    intercept_l_dual, intercept_f_dual, diag_x_E_test, diag_λ_ψ_test, h_test, λ_test)

cut_primal = eval_cut(Uhat1_primal, Uhat3_primal, Utilde1_primal, Utilde3_primal,
    Ztilde1_3_primal, βtilde1_1_primal, βtilde1_3_primal,
    int_l_primal, int_f_primal, diag_x_E_test, diag_λ_ψ_test, h_test, λ_test)

println("\n=== Cut values at test point (x=0, λ=5) ===")
println("  Dual cut value:   ", round(cut_dual, digits=4))
println("  Primal cut value: ", round(cut_primal, digits=4))
println("  Difference:       ", round(cut_primal - cut_dual, digits=4))

# Also at λ=10
diag_λ_ψ_10 = Diagonal(10.0 * ones(num_arcs))
cut_dual_10 = eval_cut(Uhat1_dual[1,:,:], Uhat3_dual[1,:,:], Utilde1_dual[1,:,:], Utilde3_dual[1,:,:],
    Ztilde1_3_dual[1,:,:], βtilde1_1_dual[1,:], βtilde1_3_dual[1,:],
    intercept_l_dual, intercept_f_dual, diag_x_E_test, diag_λ_ψ_10, h_test, 10.0)
cut_primal_10 = eval_cut(Uhat1_primal, Uhat3_primal, Utilde1_primal, Utilde3_primal,
    Ztilde1_3_primal, βtilde1_1_primal, βtilde1_3_primal,
    int_l_primal, int_f_primal, diag_x_E_test, diag_λ_ψ_10, h_test, 10.0)

println("\n=== Cut values at test point (x=0, λ=10) ===")
println("  Dual cut value:   ", round(cut_dual_10, digits=4))
println("  Primal cut value: ", round(cut_primal_10, digits=4))
println("  Difference:       ", round(cut_primal_10 - cut_dual_10, digits=4))
