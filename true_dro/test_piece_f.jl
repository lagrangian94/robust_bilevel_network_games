"""
test_piece_f.jl — Piece-F recovery 검증.
γ=3 로그(log_abilene_S10_20260415_2118152)의 x*, α*로 Piece-F LP 풀기.
"""

using Revise
using JuMP
using Gurobi
using Printf
using LinearAlgebra
using Infiltrator

include("../network_generator.jl")
using .NetworkGenerator

include("true_dro_data.jl")
include("true_dro_build_omp.jl")
include("true_dro_build_subproblem.jl")
include("true_dro_build_isp_follower.jl")
include("true_dro_recover.jl")

# --- γ=3 케이스 재현 (compute_interdict_budget 수정 전) ---
function setup_abilene_gamma3()
    network = generate_abilene_network()
    print_realworld_network_summary(network)
    num_arcs = length(network.arcs) - 1

    S = 10
    γ_ratio = 0.10
    ρ = 0.2
    seed = 42
    ε_hat = 0.1
    ε_tilde = ε_hat

    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = ceil(Int, γ_ratio * num_interdictable)  # γ=3 (수정 전)

    capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
        interdictable_arcs=network.interdictable_arcs, seed=seed)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(ρ * γ * c_bar; digits=4)
    λU = 2.0
    q_hat = fill(1.0 / S, S)

    println("  |A|=$num_arcs, S=$S, γ=$γ, w=$(round(w, digits=4)), λU=$λU")

    td = make_true_dro_data(network, capacities, q_hat, ε_hat, ε_tilde;
                            w=w, lambda_U=λU, gamma=γ)
    return network, td
end

network, td = setup_abilene_gamma3()
K = td.num_arcs

# --- 로그에서 가져온 x*, α* (γ=3 케이스) ---
x_sol = Float64[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
α_sol = Float64[0, 0, 0, 0, 0, 0, 0, 2.8758, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8049, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

println("\n" * "=" ^ 60)
println("Test 1: build_primal_piece_F (현재 코드)")
println("=" ^ 60)
println("  x* = $(round.(Int, x_sol))")
println("  α* = $(round.(α_sol, digits=4))")

model, vars = build_primal_piece_F(td, x_sol, α_sol; optimizer=Gurobi.Optimizer)

# silent 해제해서 solver 로그 보기
unset_silent(model)
optimize!(model)

st = termination_status(model)
println("\n  Status: $st")
println("  Piece-F obj = $(objective_value(model))")

# h, λ, ψ⁰ 출력
h_val = [value(vars[:h][k]) for k in 1:K]
λ_val = value(vars[:λ])
ψ0_val = [value(vars[:ψ0][k]) for k in 1:K]
y_ts_val = [value(vars[:y_ts][s]) for s in 1:td.S]

@printf("  λ*   = %.6f\n", λ_val)
@printf("  Σh*  = %.6f  (budget: λw = %.6f)\n", sum(h_val), λ_val * td.w)

println("\n  h* (nonzero arcs):")
for k in 1:K
    if abs(h_val[k]) > 1e-6
        @printf("    arc %2d: h=%.6f\n", k, h_val[k])
    end
end

println("  ψ⁰* (nonzero arcs):")
for k in 1:K
    if abs(ψ0_val[k]) > 1e-6
        @printf("    arc %2d: ψ⁰=%.6f  (x=%d)\n", k, ψ0_val[k], round(Int, x_sol[k]))
    end
end

println("\n  ỹ_ts per scenario:")
for s in 1:td.S
    @printf("    s=%d: ỹ_ts=%.6f\n", s, y_ts_val[s])
end

# --- Test 2: 로그에서 나온 h, λ 고정 ---
println("\n" * "=" ^ 60)
println("Test 2: h=7.3614 (arc8), λ=2.0 고정")
println("=" ^ 60)

model2, vars2 = build_primal_piece_F(td, x_sol, α_sol; optimizer=Gurobi.Optimizer)

# λ=2.0 고정
fix(vars2[:λ], 2.0; force=true)

# h: arc 8에만 7.3614, 나머지 0
for k in 1:K
    fix(vars2[:h][k], 0.0; force=true)
end
fix(vars2[:h][8], 7.3614; force=true)

# ψ⁰: x=1인 arc(8,11,20)에 ψ⁰=λ=2.0, 나머지 0
for k in 1:K
    fix(vars2[:ψ0][k], 0.0; force=true)
end
fix(vars2[:ψ0][8], 2.0; force=true)
fix(vars2[:ψ0][11], 2.0; force=true)
fix(vars2[:ψ0][20], 2.0; force=true)

unset_silent(model2)
optimize!(model2)
st2 = termination_status(model2)
println("\n  Status: $st2")
if st2 == MOI.OPTIMAL
    @printf("  Piece-F obj (h,λ fixed from log) = %.6f\n", objective_value(model2))
    y_ts2 = [value(vars2[:y_ts][s]) for s in 1:td.S]
    println("\n  ỹ_ts per scenario:")
    for s in 1:td.S
        @printf("    s=%d: ỹ_ts=%.6f\n", s, y_ts2[s])
    end
end

# --- Test 3: ISP-F dual (§7.2) 직접 풀기 ---
println("\n" * "=" ^ 60)
println("Test 3: ISP-F dual (same α*, x*)")
println("=" ^ 60)

ispf_model, ispf_vars = build_true_dro_isp_follower(td, x_sol, α_sol; optimizer=Gurobi.Optimizer)
unset_silent(ispf_model)
optimize!(ispf_model)

st3 = termination_status(ispf_model)
println("\n  Status: $st3")
if st3 == MOI.OPTIMAL
    @printf("  ISP-F dual obj = %.6f\n", objective_value(ispf_model))
    println("\n  → Primal Piece-F = 0  vs  Dual ISP-F = $(round(objective_value(ispf_model), digits=6))")
    println("  → Strong duality 위반 여부: $(abs(objective_value(ispf_model)) > 1e-6 ? "YES ⚠️" : "no")")
end
