"""
tv_run.jl — TV-DRO entry point + verification.

1. Full model vs Benders comparison (3×3 grid, S=2,3)
2. ε→0 nominal convergence check
"""

using JuMP
using HiGHS
using Gurobi
using Printf
using LinearAlgebra

# Load parent module
include("../network_generator.jl")
using .NetworkGenerator

# Load TV-DRO modules
include("tv_data.jl")
include("tv_build_isp_leader.jl")
include("tv_build_isp_follower.jl")
include("tv_build_imp.jl")
include("tv_build_omp.jl")
include("tv_build_full_model.jl")
include("tv_nested_benders.jl")


"""
    run_tv_verification(; m=3, n=3, S=2, seed=42,
                          eps_hat=0.1, eps_tilde=0.1,
                          gamma=2, w=1.0, lambda_U=10.0)

Full model vs Benders 비교.
"""
function run_tv_verification(; m=3, n=3, S=2, seed=42,
                               eps_hat=0.1, eps_tilde=0.1,
                               gamma=2, w=1.0, lambda_U=10.0)
    println("=" ^ 60)
    println("TV-DRO Verification: $(m)×$(n) grid, S=$S, ε̂=$eps_hat, ε̃=$eps_tilde")
    println("=" ^ 60)

    # Generate network
    network = generate_grid_network(m, n; seed=seed)
    num_arcs = length(network.arcs) - 1
    println("Network: $(network.m)×$(network.n), |A|=$num_arcs")

    # Generate scenarios: returns (num_arcs+1 × S matrix, F)
    num_arcs_with_dummy = length(network.arcs)
    scenarios, _ = generate_capacity_scenarios_uniform_model(num_arcs_with_dummy, S; seed=seed)

    # Nominal probabilities
    q_hat = fill(1.0 / S, S)

    # Create TVData
    tv = make_tv_data(network, scenarios, q_hat, eps_hat, eps_tilde;
                      w=w, lambda_U=lambda_U, gamma=gamma)

    # ============================================================
    # 1. Full model
    # ============================================================
    println("\n--- Full model (direct solve) ---")
    full_model, full_vars = build_full_tv_model(tv; optimizer=Gurobi.Optimizer)
    optimize!(full_model)
    full_st = termination_status(full_model)
    if full_st == MOI.OPTIMAL
        full_obj = objective_value(full_model)
        x_full = round.(Int, [value(full_vars[:x][k]) for k in 1:num_arcs])
        λ_full = value(full_vars[:λ])
        @printf("Full model: obj=%.6f, λ=%.4f, x=%s\n", full_obj, λ_full, string(x_full))
    else
        println("Full model status: $full_st")
        full_obj = NaN
    end

    # ============================================================
    # 2. Benders decomposition
    # ============================================================
    println("\n--- Nested Benders ---")
    result = tv_nested_benders_optimize!(tv;
        lp_optimizer=HiGHS.Optimizer,
        mip_optimizer=Gurobi.Optimizer,
        max_outer_iter=50,
        max_inner_iter=100,
        outer_tol=1e-4,
        inner_tol=1e-5,
        verbose=true)

    benders_obj = result[:Z0]
    @printf("Benders: Z₀=%.6f, status=%s\n", benders_obj, result[:status])
    if haskey(result, :x)
        x_benders = round.(Int, result[:x])
        @printf("Benders: λ=%.4f, x=%s\n", result[:λ], string(x_benders))
    end

    # ============================================================
    # 3. Comparison
    # ============================================================
    if !isnan(full_obj)
        gap = abs(full_obj - benders_obj) / max(abs(full_obj), 1e-10)
        @printf("\nFull=%.6f  Benders=%.6f  gap=%.2e\n", full_obj, benders_obj, gap)
        if gap < 1e-3
            println("✓ PASS: Full model ≈ Benders")
        else
            println("✗ FAIL: Full model ≠ Benders (gap=$gap)")
        end
    end

    return (full_obj=full_obj, benders_obj=benders_obj, result=result)
end


"""
    run_tv_epsilon_test(; m=3, n=3, S=2, seed=42)

ε→0 에서 nominal SP 수렴 확인.
"""
function run_tv_epsilon_test(; m=3, n=3, S=2, seed=42,
                               gamma=2, w=1.0, lambda_U=10.0)
    println("=" ^ 60)
    println("TV-DRO ε→0 convergence test")
    println("=" ^ 60)

    network = generate_grid_network(m, n; seed=seed)
    num_arcs = length(network.arcs) - 1
    num_arcs_with_dummy = length(network.arcs)
    scenarios, _ = generate_capacity_scenarios_uniform_model(num_arcs_with_dummy, S; seed=seed)
    q_hat = fill(1.0 / S, S)

    epsilons = [0.5, 0.2, 0.1, 0.05, 0.01, 0.001]
    results = []

    for ε in epsilons
        tv = make_tv_data(network, scenarios, q_hat, ε, ε;
                          w=w, lambda_U=lambda_U, gamma=gamma)
        full_model, full_vars = build_full_tv_model(tv; optimizer=Gurobi.Optimizer)
        optimize!(full_model)
        obj = termination_status(full_model) == MOI.OPTIMAL ? objective_value(full_model) : NaN
        push!(results, (ε=ε, obj=obj))
        @printf("ε=%.4f → obj=%.6f\n", ε, obj)
    end

    println("\nAs ε→0, objective should converge to nominal SP value.")
    return results
end


# ============================================================
# Run if executed directly
# ============================================================
if abspath(PROGRAM_FILE) == @__FILE__
    run_tv_verification()
    println()
    run_tv_epsilon_test()
end
