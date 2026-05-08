"""
test_cvar_mw.jl — mini-benders + MW cuts 검증.
  (A) β=0 + MW: 기존과 동일한 Z₀ 나오는지
  (B) β=0.3 + MW: 수렴 확인
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

net = NG.generate_grid_network(3, 3; seed=42)
num_arcs = length(net.arcs) - 1
S = 3; γ = 2
caps, _ = NG.generate_capacity_scenarios_uniform_model(length(net.arcs), S;
    interdictable_arcs=net.interdictable_arcs, seed=42)
intd_idx = findall(net.interdictable_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
q_hat = fill(1.0/S, S)

@printf("grid3x3: arcs=%d, intd=%d, γ=%d, S=%d, w=%.4f\n\n", num_arcs, length(intd_idx), γ, S, w)

# ── (A) β=0 + mini-benders + MW ──
println("="^60)
println("(A) β=0 + mini-benders + MW")
println("="^60)

td_a = make_true_dro_data(net, caps, q_hat, 0.3, 0.0; w=w, lambda_U=2.0, gamma=γ, beta=0.0)
res_a = true_dro_benders_optimize!(td_a;
    mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
    max_iter=200, tol=1e-4, verbose=true,
    nonconvex_attr=("NonConvex" => 2),
    mini_benders=true, max_mini_benders_iter=5,
    strengthen_cuts=:mw,
    valid_inequality=:mincut)

xa = round.(Int, res_a[:x])
@printf("\n(A) Z₀=%.6f, iters=%d, x=%s, status=%s\n", res_a[:Z0], res_a[:iters], string(findall(xa .> 0)), res_a[:status])

# ── (B) β=0.3 + mini-benders + MW ──
println("\n" * "="^60)
println("(B) β=0.3 + mini-benders + MW")
println("="^60)

td_b = make_true_dro_data(net, caps, q_hat, 0.3, 0.0; w=w, lambda_U=2.0, gamma=γ, beta=0.3)
res_b = true_dro_benders_optimize!(td_b;
    mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
    max_iter=200, tol=1e-4, verbose=true,
    nonconvex_attr=("NonConvex" => 2),
    mini_benders=true, max_mini_benders_iter=5,
    strengthen_cuts=:mw,
    valid_inequality=:mincut)

xb = round.(Int, res_b[:x])
@printf("\n(B) Z₀=%.6f, iters=%d, x=%s, status=%s\n", res_b[:Z0], res_b[:iters], string(findall(xb .> 0)), res_b[:status])

# ── Summary ──
println("\n" * "="^60)
println("SUMMARY")
println("="^60)
@printf("  (A) β=0  + MW: Z₀=%.6f  iters=%d  status=%s\n", res_a[:Z0], res_a[:iters], res_a[:status])
@printf("  (B) β=0.3+ MW: Z₀=%.6f  iters=%d  status=%s\n", res_b[:Z0], res_b[:iters], res_b[:status])

# regression check: (A) should match previous test4 (Z₀=12.233333)
ref_z0 = 12.233333
gap_a = abs(res_a[:Z0] - ref_z0) / max(abs(ref_z0), 1e-10)
@printf("  (A) vs ref (%.6f): gap=%.2e %s\n", ref_z0, gap_a, gap_a < 1e-3 ? "✓" : "✗")

# β>0 should be more conservative
@printf("  β=0.3 > β=0? %.6f > %.6f → %s\n", res_b[:Z0], res_a[:Z0],
        res_b[:Z0] >= res_a[:Z0] - 1e-6 ? "✓" : "✗")
