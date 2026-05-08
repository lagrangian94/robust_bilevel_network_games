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
λU = 50.0

td = make_true_dro_data(net, caps, q_hat, 0.2, 0.2; w=w, lambda_U=λU, gamma=γ, beta=0.95)
x_bar = zeros(num_arcs); x_bar[3] = 1.0; x_bar[8] = 1.0
K = num_arcs

# ── max-flow LP for scenario s, given x and α ──
function maxflow_scenario(td, x_bar, α_val, s)
    K = td.num_arcs; m = td.nv1
    Ny = td.Ny; Nts = td.Nts
    ξ = td.xi_bar; v = td.v

    model = Model(Gurobi.Optimizer); set_silent(model)
    # primal max-flow: max σ s.t. N_y·y + N_ts·σ = 0, 0 ≤ y_k ≤ cap_k
    @variable(model, y[1:K] >= 0)
    @variable(model, σ)
    eff_cap = [ξ[k,s] * (1.0 - v[k,s] * x_bar[k]) + α_val[k] for k in 1:K]
    for k in 1:K
        set_upper_bound(y[k], max(0.0, eff_cap[k]))
    end
    @constraint(model, [j=1:m], sum(Ny[j,k]*y[k] for k in 1:K) + Nts[j]*σ == 0)
    @objective(model, Max, σ)
    optimize!(model)
    return objective_value(model), eff_cap
end

# ── Solve free and tied ──
function solve_sub(td, x_bar; force_a_eq_d=false)
    m, v = build_true_dro_subproblem(td, x_bar; optimizer=Gurobi.Optimizer, silent=true)
    set_optimizer_attribute(m, "NonConvex", 2)
    if force_a_eq_d
        for s in 1:td.S; @constraint(m, v[:a][s] == v[:d][s]); end
    end
    JuMP.optimize!(m)
    return (Z0=objective_value(m),
            a=[value(v[:a][s]) for s in 1:S],
            d=[value(v[:d][s]) for s in 1:S],
            r=[value(v[:r][s]) for s in 1:S],
            α=[value(v[:α][k]) for k in 1:K])
end

free = solve_sub(td, x_bar; force_a_eq_d=false)
tied = solve_sub(td, x_bar; force_a_eq_d=true)

println("="^80)
@printf("λU=%.0f, x=[3,8], β=0.95\n", λU)
@printf("Z₀(free)=%.6f  Z₀(tied)=%.6f\n\n", free.Z0, tied.Z0)

# ── per-scenario max-flow ──
println("--- Per-scenario max-flow ---")
@printf("%3s | %10s %10s | %10s %10s | %8s %8s\n",
    "s", "MF(free)", "MF(tied)", "a(free)", "a(tied)", "r(free)", "r(tied)")
println("-"^80)

mf_free = Float64[]; mf_tied = Float64[]
for s in 1:S
    f_val, _ = maxflow_scenario(td, x_bar, free.α, s)
    t_val, _ = maxflow_scenario(td, x_bar, tied.α, s)
    push!(mf_free, f_val); push!(mf_tied, t_val)
    @printf("%3d | %10.4f %10.4f | %10.6f %10.6f | %8.4f %8.4f\n",
        s, f_val, t_val, free.a[s], tied.a[s], free.r[s], tied.r[s])
end

# ── weighted cost ──
println("\n--- Weighted costs ---")
ea_free = sum(free.a[s] * mf_free[s] for s in 1:S)
ea_tied = sum(tied.a[s] * mf_tied[s] for s in 1:S)
er_free = sum(free.r[s] * mf_free[s] for s in 1:S)
er_tied = sum(tied.r[s] * mf_tied[s] for s in 1:S)
ed_free = sum(free.d[s] * mf_free[s] for s in 1:S)
ed_tied = sum(tied.d[s] * mf_tied[s] for s in 1:S)
@printf("  E_a[MF] (leader expect):  free=%.4f  tied=%.4f\n", ea_free, ea_tied)
@printf("  E_r[MF] (leader CVaR):    free=%.4f  tied=%.4f\n", er_free, er_tied)
@printf("  E_d[MF] (follower expect): free=%.4f  tied=%.4f\n", ed_free, ed_tied)

# ── effective capacity 비교 (nonzero diff만) ──
println("\n--- Effective capacity diff (free vs tied), top arcs ---")
@printf("%3s %3s | %10s %10s %10s | %8s\n", "k", "s", "cap(free)", "cap(tied)", "diff", "ξ_raw")
for s in 1:S
    for k in 1:K
        cf = td.xi_bar[k,s]*(1-td.v[k,s]*x_bar[k]) + free.α[k]
        ct = td.xi_bar[k,s]*(1-td.v[k,s]*x_bar[k]) + tied.α[k]
        if abs(cf - ct) > 0.01
            @printf("%3d %3d | %10.4f %10.4f %10.4f | %8.4f\n", k, s, cf, ct, cf-ct, td.xi_bar[k,s])
        end
    end
end
