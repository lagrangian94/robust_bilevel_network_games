"""
test_piece_f_across_x.jl — 다양한 x에 대해 bilinear subproblem의 obj_F가 항상 0인지 검증.
"""

using Revise
using JuMP
using Gurobi
using Printf
using LinearAlgebra
using Random
using Infiltrator

include("../network_generator.jl")
using .NetworkGenerator

include("true_dro_data.jl")
include("true_dro_build_omp.jl")
include("true_dro_build_subproblem.jl")

function setup_abilene_gamma3()
    network = generate_abilene_network()
    num_arcs = length(network.arcs) - 1
    S = 10
    γ_ratio = 0.10
    ρ = 0.2
    seed = 42
    ε_hat = 0.1
    ε_tilde = ε_hat

    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = ceil(Int, γ_ratio * num_interdictable)

    capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
        interdictable_arcs=network.interdictable_arcs, seed=seed)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(ρ * γ * c_bar; digits=4)
    λU = 2.0
    q_hat = fill(1.0 / S, S)

    td = make_true_dro_data(network, capacities, q_hat, ε_hat, ε_tilde;
                            w=w, lambda_U=λU, gamma=γ)
    return network, td, γ, interdictable_idx
end

network, td, γ, interdictable_idx = setup_abilene_gamma3()
K = td.num_arcs
S = td.S
φ̃U = td.phi_tilde_U
λU = td.lambda_U

println("=" ^ 70)
println("ABILENE |A|=$K, S=$S, γ=$γ, φ̃U=$φ̃U, λU=$λU")
println("|interdictable|=$(length(interdictable_idx))")
println("=" ^ 70)

# --- 다양한 x 생성 (interdictable arcs에서 γ개 선택) ---
function random_x(interdictable_idx, γ, K, rng)
    selected = shuffle(rng, interdictable_idx)[1:γ]
    x = zeros(K)
    x[selected] .= 1.0
    return x
end

# Subproblem build
model, vars = build_true_dro_subproblem(td, zeros(K); optimizer=Gurobi.Optimizer, silent=true)

function compute_obj_F(vars, x_sol, td)
    K = td.num_arcs
    S = td.S
    φ̃U = td.phi_tilde_U
    λU = td.lambda_U
    ρ̃1 = [value(vars[:ρ_tilde_1][k, s]) for k in 1:K, s in 1:S]
    ρ̃3 = [value(vars[:ρ_tilde_3][k, s]) for k in 1:K, s in 1:S]
    ρ⁰1 = [value(vars[:ρ_psi0_1][k]) for k in 1:K]
    ρ⁰3 = [value(vars[:ρ_psi0_3][k]) for k in 1:K]

    obj_F = -φ̃U * sum(x_sol[k] * ρ̃1[k, s] for k in 1:K, s in 1:S) -
             φ̃U * sum((1.0 - x_sol[k]) * ρ̃3[k, s] for k in 1:K, s in 1:S) -
             λU * sum(x_sol[k] * ρ⁰1[k] for k in 1:K) -
             λU * sum((1.0 - x_sol[k]) * ρ⁰3[k] for k in 1:K)
    return obj_F
end

function compute_obj_L(vars, x_sol, td)
    K = td.num_arcs
    S = td.S
    φ̂U = td.phi_hat_U
    σ̂ = [value(vars[:σ_hat][s]) for s in 1:S]
    ρ̂1 = [value(vars[:ρ_hat_1][k, s]) for k in 1:K, s in 1:S]
    ρ̂3 = [value(vars[:ρ_hat_3][k, s]) for k in 1:K, s in 1:S]

    obj_L = sum(σ̂[s] for s in 1:S) -
             φ̂U * sum(x_sol[k] * ρ̂1[k, s] for k in 1:K, s in 1:S) -
             φ̂U * sum((1.0 - x_sol[k]) * ρ̂3[k, s] for k in 1:K, s in 1:S)
    return obj_L
end

# Benders cut coefficients: Z(x) = const + Σ_k π_x[k]·x_k
# π_L[k] = φ̂U · (-Σ_s ρ̂1[k,s] + Σ_s ρ̂3[k,s])   ← Leader piece
# π_F[k] = φ̃U · (-Σ_s ρ̃1[k,s] + Σ_s ρ̃3[k,s])   ← Follower (ρ̃ part)
# π_ψ[k] = λU  · (-ρ⁰1[k]      + ρ⁰3[k])          ← Follower (ρ⁰ part)
function compute_pi_components(vars, td)
    K = td.num_arcs
    S = td.S
    φ̂U = td.phi_hat_U; φ̃U = td.phi_tilde_U; λU = td.lambda_U
    ρ̂1 = [value(vars[:ρ_hat_1][k, s]) for k in 1:K, s in 1:S]
    ρ̂3 = [value(vars[:ρ_hat_3][k, s]) for k in 1:K, s in 1:S]
    ρ̃1 = [value(vars[:ρ_tilde_1][k, s]) for k in 1:K, s in 1:S]
    ρ̃3 = [value(vars[:ρ_tilde_3][k, s]) for k in 1:K, s in 1:S]
    ρ⁰1 = [value(vars[:ρ_psi0_1][k]) for k in 1:K]
    ρ⁰3 = [value(vars[:ρ_psi0_3][k]) for k in 1:K]
    π_L = [φ̂U * (-sum(ρ̂1[k, :]) + sum(ρ̂3[k, :])) for k in 1:K]
    π_F = [φ̃U * (-sum(ρ̃1[k, :]) + sum(ρ̃3[k, :])) for k in 1:K]
    π_ψ = [λU  * (-ρ⁰1[k]        + ρ⁰3[k])        for k in 1:K]
    return π_L, π_F, π_ψ
end

# --- 10개 random x + 몇 가지 corner cases ---
rng = MersenneTwister(42)

x_tests = Vector{Vector{Float64}}()
labels = String[]

# All-zero x (interdict nothing, but need γ constraint... 그냥 전부 0)
push!(x_tests, zeros(K))
push!(labels, "x=0 (no interdiction)")

# Random x들
for i in 1:8
    push!(x_tests, random_x(interdictable_idx, γ, K, rng))
    push!(labels, "random #$i")
end

# γ=3 로그의 x*
push!(x_tests, Float64[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
push!(labels, "log γ=3 x*")

println("\n" * "-" ^ 100)
@printf("%-25s %10s %10s %10s %10s %10s %10s %10s\n",
        "Case", "Z₀", "obj_L", "obj_F", "|π_L|∞", "|π_F|∞", "|π_ψ|∞", "|α|∞")
println("-" ^ 100)

results = []
for (lbl, x_try) in zip(labels, x_tests)
    try
        update_true_dro_subproblem_objective!(model, vars, td, x_try)
        set_time_limit_sec(model, 60.0)
        optimize!(model)
        st = termination_status(model)
        if st == MOI.OPTIMAL || (st == MOI.TIME_LIMIT && has_values(model))
            Z0 = objective_value(model)
            obj_F = compute_obj_F(vars, x_try, td)
            obj_L = compute_obj_L(vars, x_try, td)
            π_L, π_F, π_ψ = compute_pi_components(vars, td)
            α_vals = [value(vars[:α][k]) for k in 1:K]
            α_max = maximum(abs.(α_vals))
            piL_max = maximum(abs.(π_L))
            piF_max = maximum(abs.(π_F))
            piψ_max = maximum(abs.(π_ψ))
            @printf("%-25s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n",
                    lbl, Z0, obj_L, obj_F, piL_max, piF_max, piψ_max, α_max)
            push!(results, (lbl, Z0, obj_L, obj_F, π_L, π_F, π_ψ, α_max))
        else
            @printf("%-25s  %s\n", lbl, st)
        end
    catch e
        @printf("%-25s  ERROR: %s\n", lbl, e)
    end
end

println("-" ^ 100)
println("\nSummary (cut coefficients from Follower piece):")
println("-" ^ 100)
@printf("%-25s %15s %15s %15s\n", "Case", "obj_F", "|π_F|∞ (ρ̃)", "|π_ψ|∞ (ρ⁰)")
println("-" ^ 100)
for (lbl, Z0, obj_L, obj_F, π_L, π_F, π_ψ, α_max) in results
    @printf("  %-25s  %12.6f   %12.4f    %12.4f\n",
            lbl, obj_F, maximum(abs.(π_F)), maximum(abs.(π_ψ)))
end

# 구체적으로 한 예시의 nonzero cut coefficient 출력
println("\n" * "=" ^ 70)
println("Example: log γ=3 x* — per-arc π_F, π_ψ values")
println("=" ^ 70)
if length(results) > 0
    lbl, Z0, obj_L, obj_F, π_L, π_F, π_ψ, α_max = results[end]
    println("Case: $lbl")
    @printf("  obj_F = %.6f   (Piece-F 목적함수 값)\n", obj_F)
    @printf("  Σ_k π_F[k] = %.6f,  Σ_k π_ψ[k] = %.6f\n", sum(π_F), sum(π_ψ))
    println("\n  Nonzero π_F, π_ψ (follower piece cut coefficients):")
    @printf("  %5s %12s %12s %12s\n", "arc", "π_L[k]", "π_F[k]", "π_ψ[k]")
    for k in 1:K
        if abs(π_L[k]) > 1e-4 || abs(π_F[k]) > 1e-4 || abs(π_ψ[k]) > 1e-4
            @printf("  %5d %12.4f %12.4f %12.4f\n", k, π_L[k], π_F[k], π_ψ[k])
        end
    end
end
