"""
diag_subproblem_d.jl — x=[3,6], x=[6,34] 고정 후 subproblem의 d(follower q̃) 추출
  Polska, factor_additive, γ=2, S=20, λU=10.0
  ε̂=ε̃ = 0.1, 0.5, 1.0 에서 d가 q̂에서 얼마나 어디로 움직이는지 분석.
"""

using JuMP, Gurobi, Printf, Dates, LinearAlgebra, Statistics

include("../../network_generator.jl")
using .NetworkGenerator

include("../true_dro_data.jl")
include("../true_dro_build_subproblem.jl")

# ── network setup ──
net = generate_polska_network()
num_arcs = length(net.arcs) - 1

all_intd = fill(true, length(net.arcs))
net = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
    net.N, all_intd, net.arc_adjacency, net.node_arc_incidence)

S = 20
caps, _ = generate_capacity_scenarios_factor_additive(length(net.arcs), S;
    interdictable_arcs=all_intd, seed=42, num_factors=5)
intd_idx = findall(all_intd[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
γ = 2
λU = 10.0
q_hat = fill(1.0/S, S)

@printf("polska factor_additive: arcs=%d, w=%.4f\n\n", num_arcs, w)

# ── solve subproblem for fixed x, extract a, d, α ──
function solve_sub_fixed_x(x_arcs, ε; time_limit=120.0)
    x_vec = zeros(num_arcs)
    for a in x_arcs; x_vec[a] = 1.0; end

    td = make_true_dro_data(net, caps, q_hat, ε, ε; w=w, lambda_U=λU, gamma=γ)
    m, v = build_true_dro_subproblem(td, x_vec; optimizer=Gurobi.Optimizer, silent=true)
    set_optimizer_attribute(m, "NonConvex", 2)
    set_optimizer_attribute(m, "TimeLimit", time_limit)

    optimize!(m)
    status = termination_status(m)
    Z0 = objective_value(m)

    a_val = value.(v[:a])
    d_val = value.(v[:d])
    α_val = value.(v[:α])
    σ_hat_val = value.(v[:σ_hat])
    σ_tilde_val = value.(v[:σ_tilde])

    return Dict(:Z0 => Z0, :status => status,
                :a => a_val, :d => d_val, :α => α_val,
                :σ_hat => σ_hat_val, :σ_tilde => σ_tilde_val)
end

# ── main ──
x_configs = [
    ([3, 6],  "x_nom [3,6]"),
    ([6, 34], "x_rob [6,34]"),
]
eps_list = [0.1, 0.5, 1.0]

for ε in eps_list
    println("=" ^ 70)
    @printf("ε̂ = ε̃ = %.1f   (TV ball radius, max TV(q̃,q̂) = %.1f)\n", ε, ε)
    println("=" ^ 70)

    for (x_arcs, label) in x_configs
        @printf("\n--- %s ---\n", label)
        flush(stdout)

        res = solve_sub_fixed_x(x_arcs, ε)
        @printf("  Z₀ = %.6f  (%s)\n", res[:Z0], res[:status])

        # α
        α_nz = findall(res[:α] .> 1e-6)
        @printf("  α non-zero: ")
        for k in α_nz
            @printf("%d(%.4f) ", k, res[:α][k])
        end
        println()

        # a (leader q̂ perturbation)
        a_diff = res[:a] .- q_hat
        @printf("  Leader a: TV(a,q̂) = %.4f\n", 0.5 * sum(abs.(a_diff)))
        a_sorted = sortperm(a_diff)
        @printf("    most ↓: s=%d(Δ=%+.5f) s=%d(Δ=%+.5f) s=%d(Δ=%+.5f)\n",
                a_sorted[1], a_diff[a_sorted[1]],
                a_sorted[2], a_diff[a_sorted[2]],
                a_sorted[3], a_diff[a_sorted[3]])
        @printf("    most ↑: s=%d(Δ=%+.5f) s=%d(Δ=%+.5f) s=%d(Δ=%+.5f)\n",
                a_sorted[end], a_diff[a_sorted[end]],
                a_sorted[end-1], a_diff[a_sorted[end-1]],
                a_sorted[end-2], a_diff[a_sorted[end-2]])

        # d (follower q̃ perturbation)
        d_diff = res[:d] .- q_hat
        @printf("  Follower d: TV(d,q̂) = %.4f\n", 0.5 * sum(abs.(d_diff)))
        d_sorted = sortperm(d_diff)
        @printf("    most ↓: s=%d(Δ=%+.5f) s=%d(Δ=%+.5f) s=%d(Δ=%+.5f)\n",
                d_sorted[1], d_diff[d_sorted[1]],
                d_sorted[2], d_diff[d_sorted[2]],
                d_sorted[3], d_diff[d_sorted[3]])
        @printf("    most ↑: s=%d(Δ=%+.5f) s=%d(Δ=%+.5f) s=%d(Δ=%+.5f)\n",
                d_sorted[end], d_diff[d_sorted[end]],
                d_sorted[end-1], d_diff[d_sorted[end-1]],
                d_sorted[end-2], d_diff[d_sorted[end-2]])

        # full table
        println("    s   q̂      a(leader)  Δa       d(follwr)  Δd       σ̂       σ̃")
        for s in 1:S
            @printf("    %2d  %.4f  %.5f  %+.5f  %.5f  %+.5f  %.4f  %.4f\n",
                    s, q_hat[s], res[:a][s], a_diff[s], res[:d][s], d_diff[s],
                    res[:σ_hat][s], res[:σ_tilde][s])
        end

        flush(stdout)
    end
    println()
end
