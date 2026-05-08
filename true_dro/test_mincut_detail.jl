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

# α values from previous run
α_free = zeros(K)
α_free[1]=0.25; α_free[2]=0.00; α_free[9]=3.4652; α_free[10]=2.9367
α_free[12]=0.3133; α_free[13]=0.7848; α_free[15]=0.00; α_free[16]=0.25

α_tied = zeros(K)
α_tied[1]=0.0379; α_tied[2]=0.2121; α_tied[9]=3.2139; α_tied[10]=3.0361
α_tied[12]=0.2139; α_tied[13]=1.0360; α_tied[15]=0.038; α_tied[16]=0.212

# Network topology
println("--- Network arcs ---")
for k in 1:K
    @printf("  k=%2d: %s → %s\n", k, net.arcs[k][1], net.arcs[k][2])
end

# Scenario 1 detail (r=1 → this is what matters)
s = 1
println("\n" * "="^80)
@printf("Scenario %d (r=1.0, the only one that matters)\n", s)
println("="^80)

println("\n--- Effective capacity per arc ---")
@printf("%3s  %6s → %-6s | %6s %6s | %10s %10s %10s | %5s\n",
    "k", "from", "to", "ξ_raw", "v·x", "cap(free)", "cap(tied)", "diff", "intd?")
println("-"^85)
for k in 1:K
    ξ_raw = td.xi_bar[k,s]
    vx = td.v[k,s] * x_bar[k]
    cf = ξ_raw * (1 - vx) + α_free[k]
    ct = ξ_raw * (1 - vx) + α_tied[k]
    intd = x_bar[k] > 0.5 ? "★" : ""
    @printf("%3d  %6s → %-6s | %6.2f %6.2f | %10.4f %10.4f %10.4f | %5s\n",
        k, net.arcs[k][1], net.arcs[k][2], ξ_raw, vx, cf, ct, cf-ct, intd)
end

# Max-flow + dual (min-cut) for scenario 1
function maxflow_with_mincut(td, x_bar, α_val, s)
    K = td.num_arcs; m = td.nv1
    Ny = td.Ny; Nts = td.Nts; ξ = td.xi_bar; v = td.v

    model = Model(Gurobi.Optimizer); set_silent(model)
    @variable(model, y[1:K] >= 0)
    @variable(model, σ)
    eff_cap = [ξ[k,s]*(1-v[k,s]*x_bar[k]) + α_val[k] for k in 1:K]
    cap_con = []
    for k in 1:K
        c = @constraint(model, y[k] <= max(0.0, eff_cap[k]))
        push!(cap_con, c)
    end
    @constraint(model, flow[j=1:m], sum(Ny[j,k]*y[k] for k in 1:K) + Nts[j]*σ == 0)
    @objective(model, Max, σ)
    optimize!(model)

    mf = objective_value(model)
    y_val = [value(y[k]) for k in 1:K]
    slack = [eff_cap[k] - y_val[k] for k in 1:K]
    # min-cut: arcs where capacity constraint is tight (slack ≈ 0)
    return mf, y_val, eff_cap, slack
end

mf_f, y_f, cap_f, slack_f = maxflow_with_mincut(td, x_bar, α_free, s)
mf_t, y_t, cap_t, slack_t = maxflow_with_mincut(td, x_bar, α_tied, s)

@printf("\nMax-flow(free)=%.4f  Max-flow(tied)=%.4f\n", mf_f, mf_t)

println("\n--- Flow & slack (scenario 1) ---")
@printf("%3s  %6s→%-6s | %8s %8s %8s | %8s %8s %8s | %s\n",
    "k", "from", "to", "cap(f)", "flow(f)", "slack(f)", "cap(t)", "flow(t)", "slack(t)", "cut?")
println("-"^100)
for k in 1:K
    cut_f = slack_f[k] < 1e-4 ? "★" : ""
    cut_t = slack_t[k] < 1e-4 ? "☆" : ""
    cut = cut_f * cut_t
    if abs(y_f[k]) > 1e-6 || abs(y_t[k]) > 1e-6 || !isempty(cut)
        @printf("%3d  %6s→%-6s | %8.4f %8.4f %8.4f | %8.4f %8.4f %8.4f | %s\n",
            k, net.arcs[k][1], net.arcs[k][2],
            cap_f[k], y_f[k], slack_f[k], cap_t[k], y_t[k], slack_t[k], cut)
    end
end
println("★=free min-cut arc, ☆=tied min-cut arc")

# Min-cut capacity sum
cut_arcs_f = findall(slack_f .< 1e-4)
cut_arcs_t = findall(slack_t .< 1e-4)
@printf("\nMin-cut arcs (free): %s  → Σcap=%.4f\n", cut_arcs_f, sum(cap_f[k] for k in cut_arcs_f))
@printf("Min-cut arcs (tied): %s  → Σcap=%.4f\n", cut_arcs_t, sum(cap_t[k] for k in cut_arcs_t))
