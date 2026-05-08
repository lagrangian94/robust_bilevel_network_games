using JuMP, Gurobi, Printf, LinearAlgebra
include("../network_generator.jl"); NG = NetworkGenerator
include("true_dro_data.jl"); include("true_dro_build_omp.jl"); include("true_dro_build_subproblem.jl")

net = NG.generate_grid_network(3, 3; seed=42)
num_arcs = length(net.arcs) - 1; S = 3
caps, _ = NG.generate_capacity_scenarios_uniform_model(length(net.arcs), S;
    interdictable_arcs=net.interdictable_arcs, seed=42)
intd_idx = findall(net.interdictable_arcs[1:num_arcs])
w_default = round(maximum(caps[intd_idx, :]); digits=4)
q_hat = fill(1.0/S, S)
K = num_arcs
λU = 50.0

function solve_gap(td, x_bar)
    m1, v1 = build_true_dro_subproblem(td, x_bar; optimizer=Gurobi.Optimizer, silent=true)
    set_optimizer_attribute(m1, "NonConvex", 2); JuMP.optimize!(m1)
    Z_free = objective_value(m1)
    
    m2, v2 = build_true_dro_subproblem(td, x_bar; optimizer=Gurobi.Optimizer, silent=true)
    set_optimizer_attribute(m2, "NonConvex", 2)
    for s in 1:td.S; @constraint(m2, v2[:a][s] == v2[:d][s]); end
    JuMP.optimize!(m2)
    Z_tied = objective_value(m2)
    
    α_f = [value(v1[:α][k]) for k in 1:K]
    α_t = [value(v2[:α][k]) for k in 1:K]
    return Z_free, Z_tied, α_f, α_t
end

# x 고정
x_bar = zeros(num_arcs); x_bar[3] = 1.0; x_bar[8] = 1.0

println("="^90)
println("Sweep w (α budget), γ=2 fixed, λU=$λU, β=0.95, x=[3,8]")
println("="^90)
@printf("%8s | %10s %10s %10s %10s | %10s\n",
    "w", "Z(free)", "Z(tied)", "gap", "gap%", "||Δα||₁")
println("-"^70)
for w_mult in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    w = w_default * w_mult
    td = make_true_dro_data(net, caps, q_hat, 0.2, 0.2; w=w, lambda_U=λU, gamma=2, beta=0.95)
    Zf, Zt, αf, αt = solve_gap(td, x_bar)
    gap = Zf - Zt
    @printf("%8.2f | %10.4f %10.4f %10.6f %10.4f%% | %10.4f\n",
        w, Zf, Zt, gap, gap/max(abs(Zt),1e-10)*100, sum(abs.(αf.-αt)))
end

println("\n" * "="^90)
println("Sweep γ (interdiction budget), w=$w_default fixed, λU=$λU, β=0.95")
println("="^90)
@printf("%5s | %20s | %10s %10s %10s | %10s\n",
    "γ", "x (arcs)", "Z(free)", "Z(tied)", "gap", "||Δα||₁")
println("-"^85)
for γ in [1, 2, 3, 4, 5]
    td = make_true_dro_data(net, caps, q_hat, 0.2, 0.2; w=w_default, lambda_U=λU, gamma=γ, beta=0.95)
    # x: first γ interdictable arcs
    x_bar_g = zeros(num_arcs)
    intd = findall(net.interdictable_arcs[1:num_arcs])
    for i in 1:min(γ, length(intd)); x_bar_g[intd[i]] = 1.0; end
    x_arcs = findall(x_bar_g .> 0.5)
    
    Zf, Zt, αf, αt = solve_gap(td, x_bar_g)
    gap = Zf - Zt
    @printf("%5d | %20s | %10.4f %10.4f %10.6f | %10.4f\n",
        γ, x_arcs, Zf, Zt, gap, sum(abs.(αf.-αt)))
end

println("\n" * "="^90)
println("Sweep β (CVaR level), w=$w_default, γ=2, λU=$λU, x=[3,8]")
println("="^90)
x_bar2 = zeros(num_arcs); x_bar2[3] = 1.0; x_bar2[8] = 1.0
@printf("%8s | %10s %10s %10s | %10s\n",
    "β", "Z(free)", "Z(tied)", "gap", "||Δα||₁")
println("-"^60)
for β in [0.0, 0.3, 0.5, 0.8, 0.95, 0.99]
    td = make_true_dro_data(net, caps, q_hat, 0.2, 0.2; w=w_default, lambda_U=λU, gamma=2, beta=β)
    Zf, Zt, αf, αt = solve_gap(td, x_bar2)
    gap = Zf - Zt
    @printf("%8.2f | %10.4f %10.4f %10.6f | %10.4f\n",
        β, Zf, Zt, gap, sum(abs.(αf.-αt)))
end
