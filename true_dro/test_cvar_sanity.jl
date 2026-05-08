"""
test_cvar_sanity.jl — β=0 sanity test for CVaR-Leader extension.
grid3x3, S=3, ε̂=0.3, ε̃=0.0 (single layer).
Runs Benders twice: once without beta kwarg (default), once with beta=0.0.
Verifies Z₀ and x* are identical.
"""

using JuMP, Gurobi, Printf, LinearAlgebra

include("../network_generator.jl")
NG = NetworkGenerator

include("true_dro_data.jl")
include("true_dro_build_omp.jl")
include("true_dro_build_subproblem.jl")
include("true_dro_build_isp_leader.jl")
include("true_dro_build_isp_follower.jl")
include("true_dro_benders.jl")
include("true_dro_mincut_vi.jl")

# ── small instance ──
net = NG.generate_grid_network(3, 3; seed=42)
num_arcs = length(net.arcs) - 1
S = 3
γ = 2
caps, _ = NG.generate_capacity_scenarios_uniform_model(length(net.arcs), S;
    interdictable_arcs=net.interdictable_arcs, seed=42)
intd_idx = findall(net.interdictable_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
q_hat = fill(1.0/S, S)

@printf("grid3x3: arcs=%d, intd=%d, γ=%d, S=%d, w=%.4f\n", num_arcs, length(intd_idx), γ, S, w)

# ── Run 1: default (no beta kwarg) ──
println("\n" * "="^50)
println("Run 1: default (no beta kwarg)")
println("="^50)
td1 = make_true_dro_data(net, caps, q_hat, 0.3, 0.0; w=w, lambda_U=2.0, gamma=γ)
@printf("td1.beta = %.1f\n", td1.beta)

res1 = true_dro_benders_optimize!(td1;
    mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer,
    max_iter=200, tol=1e-4, verbose=true,
    nonconvex_attr=("NonConvex" => 2),
    valid_inequality=:mincut)

x1 = round.(Int, res1[:x])
Z1 = res1[:Z0]

# ── Run 2: explicit beta=0.0 ──
println("\n" * "="^50)
println("Run 2: explicit beta=0.0")
println("="^50)
td2 = make_true_dro_data(net, caps, q_hat, 0.3, 0.0; w=w, lambda_U=2.0, gamma=γ, beta=0.0)
@printf("td2.beta = %.1f\n", td2.beta)

res2 = true_dro_benders_optimize!(td2;
    mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer,
    max_iter=200, tol=1e-4, verbose=true,
    nonconvex_attr=("NonConvex" => 2),
    valid_inequality=:mincut)

x2 = round.(Int, res2[:x])
Z2 = res2[:Z0]

# ── Compare ──
println("\n" * "="^50)
println("COMPARISON")
println("="^50)
@printf("Z₀ (Run1): %.6f  (iters=%d)\n", Z1, res1[:iters])
@printf("Z₀ (Run2): %.6f  (iters=%d)\n", Z2, res2[:iters])
@printf("Z₀ diff:   %.2e\n", abs(Z1 - Z2))
println("x1 = $(findall(x1 .> 0))")
println("x2 = $(findall(x2 .> 0))")
println("x SAME? $(x1 == x2)")

gap = abs(Z1 - Z2) / max(abs(Z1), 1e-10)
if gap < 1e-3 && x1 == x2
    println("\n✓ β=0 SANITY TEST PASSED")
else
    println("\n✗ β=0 SANITY TEST FAILED (gap=$gap)")
end
