"""
test_cvar_all.jl — CVaR-Leader 4종 테스트.
  1. β>0 smoke test (grid3x3, S=3, ε̂=0.3, ε̃=0, β=0.3)
  2. ε̂=ε̃=0 + β=0 nominal compact path
  3. ε̂>0, ε̃>0 + β=0 full double-layer regression
  4. mini-benders + β=0 (ISP-L r_val, α-step)
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

# ── common setup ──
net = NG.generate_grid_network(3, 3; seed=42)
num_arcs = length(net.arcs) - 1
S = 3
γ = 2
caps, _ = NG.generate_capacity_scenarios_uniform_model(length(net.arcs), S;
    interdictable_arcs=net.interdictable_arcs, seed=42)
intd_idx = findall(net.interdictable_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
q_hat = fill(1.0/S, S)

@printf("grid3x3: arcs=%d, intd=%d, γ=%d, S=%d, w=%.4f\n\n", num_arcs, length(intd_idx), γ, S, w)

results = Dict{String, Any}()

# ====================================================================
# Test 1: β>0 smoke test (ε̂=0.3, ε̃=0, β=0.3)
# ====================================================================
println("="^60)
println("TEST 1: β>0 smoke (ε̂=0.3, ε̃=0, β=0.3)")
println("="^60)

td1 = make_true_dro_data(net, caps, q_hat, 0.3, 0.0; w=w, lambda_U=2.0, gamma=γ, beta=0.3)
@printf("  beta=%.1f, eps_hat=%.1f, eps_tilde=%.1f\n", td1.beta, td1.eps_hat, td1.eps_tilde)

try
    res1 = true_dro_benders_optimize!(td1;
        mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer,
        max_iter=300, tol=1e-4, verbose=true,
        nonconvex_attr=("NonConvex" => 2),
        valid_inequality=:mincut)
    results["test1"] = res1
    x1 = round.(Int, res1[:x])
    @printf("\nTEST 1 RESULT: Z₀=%.6f, iters=%d, x=%s, status=%s\n",
            res1[:Z0], res1[:iters], string(findall(x1 .> 0)), res1[:status])
    println("TEST 1: ✓ PASSED")
catch e
    println("TEST 1: ✗ FAILED — $(sprint(showerror, e))")
    results["test1"] = nothing
end
flush(stdout)

# ====================================================================
# Test 2: ε̂=ε̃=0 + β=0 (nominal compact)
# ====================================================================
println("\n" * "="^60)
println("TEST 2: nominal compact (ε̂=0, ε̃=0, β=0)")
println("="^60)

td2 = make_true_dro_data(net, caps, q_hat, 0.0, 0.0; w=w, lambda_U=2.0, gamma=γ, beta=0.0)
@printf("  beta=%.1f, eps_hat=%.1f, eps_tilde=%.1f\n", td2.beta, td2.eps_hat, td2.eps_tilde)

try
    res2 = true_dro_benders_optimize!(td2;
        mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer,
        max_iter=200, tol=1e-4, verbose=true,
        nonconvex_attr=("NonConvex" => 2))
    results["test2"] = res2
    x2 = round.(Int, res2[:x])
    @printf("\nTEST 2 RESULT: Z₀=%.6f, iters=%d, x=%s, status=%s\n",
            res2[:Z0], res2[:iters], string(findall(x2 .> 0)), res2[:status])
    println("TEST 2: ✓ PASSED")
catch e
    println("TEST 2: ✗ FAILED — $(sprint(showerror, e))")
    results["test2"] = nothing
end
flush(stdout)

# ====================================================================
# Test 3: ε̂>0, ε̃>0 + β=0 (full double-layer)
# ====================================================================
println("\n" * "="^60)
println("TEST 3: full double-layer (ε̂=0.3, ε̃=0.3, β=0)")
println("="^60)

td3 = make_true_dro_data(net, caps, q_hat, 0.3, 0.3; w=w, lambda_U=2.0, gamma=γ, beta=0.0)
@printf("  beta=%.1f, eps_hat=%.1f, eps_tilde=%.1f\n", td3.beta, td3.eps_hat, td3.eps_tilde)

try
    res3 = true_dro_benders_optimize!(td3;
        mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer,
        max_iter=300, tol=1e-4, verbose=true,
        nonconvex_attr=("NonConvex" => 2),
        valid_inequality=:mincut)
    results["test3"] = res3
    x3 = round.(Int, res3[:x])
    @printf("\nTEST 3 RESULT: Z₀=%.6f, iters=%d, x=%s, status=%s\n",
            res3[:Z0], res3[:iters], string(findall(x3 .> 0)), res3[:status])
    println("TEST 3: ✓ PASSED")
catch e
    println("TEST 3: ✗ FAILED — $(sprint(showerror, e))")
    results["test3"] = nothing
end
flush(stdout)

# ====================================================================
# Test 4: mini-benders + β=0 (ε̂=0.3, ε̃=0)
# ====================================================================
println("\n" * "="^60)
println("TEST 4: mini-benders (ε̂=0.3, ε̃=0, β=0)")
println("="^60)

td4 = make_true_dro_data(net, caps, q_hat, 0.3, 0.0; w=w, lambda_U=2.0, gamma=γ, beta=0.0)
@printf("  beta=%.1f, eps_hat=%.1f, eps_tilde=%.1f\n", td4.beta, td4.eps_hat, td4.eps_tilde)

try
    res4 = true_dro_benders_optimize!(td4;
        mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer,
        lp_optimizer=Gurobi.Optimizer,
        max_iter=200, tol=1e-4, verbose=true,
        nonconvex_attr=("NonConvex" => 2),
        mini_benders=true, max_mini_benders_iter=5,
        valid_inequality=:mincut)
    results["test4"] = res4
    x4 = round.(Int, res4[:x])
    @printf("\nTEST 4 RESULT: Z₀=%.6f, iters=%d, x=%s, status=%s\n",
            res4[:Z0], res4[:iters], string(findall(x4 .> 0)), res4[:status])
    println("TEST 4: ✓ PASSED")
catch e
    println("TEST 4: ✗ FAILED — $(sprint(showerror, e))")
    results["test4"] = nothing
end
flush(stdout)

# ====================================================================
# Summary
# ====================================================================
println("\n" * "="^60)
println("SUMMARY")
println("="^60)
for (name, label) in [("test1","β>0 smoke"), ("test2","nominal compact"),
                       ("test3","full double"), ("test4","mini-benders")]
    r = results[name]
    if r === nothing
        @printf("  %-20s  ✗ FAILED\n", label)
    else
        @printf("  %-20s  ✓ Z₀=%.6f  iters=%d  status=%s\n", label, r[:Z0], r[:iters], r[:status])
    end
end
