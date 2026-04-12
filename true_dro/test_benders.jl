"""
test_benders.jl — True-DRO Benders 검증.

Interactive input으로 network config, S 선택.
compare_benders.jl의 setup_instance 패턴 참고.
"""

using Revise
using JuMP
using Gurobi
using HiGHS
using Printf
using LinearAlgebra
using Infiltrator

# ---- Load modules ----
if !@isdefined(NetworkGenerator)
    include("../network_generator.jl")
end
using .NetworkGenerator

# True-DRO
includet("true_dro_data.jl")
includet("true_dro_build_omp.jl")
includet("true_dro_build_subproblem.jl")
includet("true_dro_benders.jl")

# TV-DRO (for sandwich comparison)
includet("../TV_DRO/tv_data.jl")
includet("../TV_DRO/tv_build_isp_leader.jl")
includet("../TV_DRO/tv_build_isp_follower.jl")
includet("../TV_DRO/tv_build_imp.jl")
includet("../TV_DRO/tv_build_omp.jl")
includet("../TV_DRO/tv_build_full_model.jl")
includet("../TV_DRO/tv_nested_benders.jl")


# ===== Network Instance Configs =====
network_configs = Dict(
    :grid_3x3   => Dict(:type => :grid, :m => 3, :n => 3),
    :grid_5x5   => Dict(:type => :grid, :m => 5, :n => 5),
    :sioux_falls => Dict(:type => :real_world, :generator => generate_sioux_falls_network),
    :nobel_us   => Dict(:type => :real_world, :generator => generate_nobel_us_network),
    :abilene    => Dict(:type => :real_world, :generator => generate_abilene_network),
    :polska     => Dict(:type => :real_world, :generator => generate_polska_network),
)


"""
    setup_true_dro_instance(config_key; S, γ_ratio, ρ, v, seed, epsilon_hat, epsilon_tilde)

네트워크 + 파라미터 → TrueDROData 생성. compare_benders.jl의 setup_instance 패턴.
"""
function setup_true_dro_instance(config_key::Symbol;
    S=10, γ_ratio=0.10, ρ=0.2, v=1.0, seed=42,
    epsilon_hat=0.5, epsilon_tilde=epsilon_hat)

    config = network_configs[config_key]

    # --- Network 생성 ---
    if config[:type] == :grid
        network = generate_grid_network(config[:m], config[:n]; seed=seed)
        print_network_summary(network)
    elseif config[:type] == :real_world
        network = config[:generator]()
        print_realworld_network_summary(network)
    end

    num_arcs = length(network.arcs) - 1

    # --- Parameters (compare_benders.jl 기준) ---
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = ceil(Int, γ_ratio * num_interdictable)

    capacities, _ = generate_capacity_scenarios_uniform_model(length(network.arcs), S; seed=seed)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(ρ * γ * c_bar; digits=4)

    # λU: leader's λ upper bound
    λU = 1.0 / epsilon_hat

    q_hat = fill(1.0 / S, S)

    println("  |A|=$num_arcs, S=$S, γ=$γ, w=$(round(w, digits=4)), λU=$λU")
    println("  ε̂=$epsilon_hat, ε̃=$epsilon_tilde")

    td = make_true_dro_data(network, capacities, q_hat, epsilon_hat, epsilon_tilde;
                            w=w, lambda_U=λU, gamma=γ)

    return network, td
end


# ===== Interactive Instance 선택 =====
println("=" ^ 70)
println("True-DRO Benders Test")
println("=" ^ 70)
println("  1. Grid network")
println("  2. Sioux Falls (24 nodes, 76 arcs)")
println("  3. Nobel US")
println("  4. Abilene")
println("  5. Polska")
print("선택 (1-5): "); net_choice = parse(Int, readline())

if net_choice == 1
    print("Grid rows (m): "); m = parse(Int, readline())
    print("Grid cols (n): "); n = parse(Int, readline())
    network_configs[Symbol("grid_$(m)x$(n)")] = Dict(:type => :grid, :m => m, :n => n)
    instance_key = Symbol("grid_$(m)x$(n)")
elseif net_choice == 2
    instance_key = :sioux_falls
elseif net_choice == 3
    instance_key = :nobel_us
elseif net_choice == 4
    instance_key = :abilene
elseif net_choice == 5
    instance_key = :polska
else
    error("잘못된 선택: $net_choice")
end

print("시나리오 수 S: "); S = parse(Int, readline())
print("ε̂ (leader TV radius) [0.1]: "); ε_hat_str = strip(readline())
ε_hat = isempty(ε_hat_str) ? 0.1 : parse(Float64, ε_hat_str)
print("ε̃ (follower TV radius) [ε̂]: "); ε_tilde_str = strip(readline())
ε_tilde = isempty(ε_tilde_str) ? ε_hat : parse(Float64, ε_tilde_str)

println("\n" * "=" ^ 70)
println("INSTANCE: $instance_key (S=$S, ε̂=$ε_hat, ε̃=$ε_tilde)")
println("=" ^ 70)

network, td = setup_true_dro_instance(instance_key; S=S,
    epsilon_hat=ε_hat, epsilon_tilde=ε_tilde)


# ===== 1. True-DRO Benders =====
println("\n" * "=" ^ 70)
println("1. True-DRO Benders (bilinear subproblem via Gurobi NonConvex=2)")
println("=" ^ 70)

result = true_dro_benders_optimize!(td;
    mip_optimizer=Gurobi.Optimizer,
    nlp_optimizer=Gurobi.Optimizer,
    max_iter=1000,
    tol=1e-4,
    verbose=true,
    sub_verbose=true)

gap = abs(result[:upper_bound] - result[:lower_bound]) /
      max(abs(result[:upper_bound]), 1e-10)

println("\n" * "-" ^ 40)
@printf("True-DRO: status=%s, Z₀=%.6f, iters=%d\n", result[:status], result[:Z0], result[:iters])
@printf("  LB=%.6f, UB=%.6f, gap=%.2e\n", result[:lower_bound], result[:upper_bound], gap)
x_int = round.(Int, result[:x])
println("  x* = $x_int")
α_str = join([@sprintf("%.3f", a) for a in result[:α]], ", ")
println("  α* = [$α_str]")

@infiltrate

# ===== 2. Sandwich: TV-DRO (V^Dir) =====
println("\n" * "=" ^ 70)
println("2. Sandwich comparison: V*(true) ≤ V^Dir(TV-DRO)")
println("=" ^ 70)

tv = make_tv_data(network, td.xi_bar, td.q_hat, td.eps_hat, td.eps_tilde;
                  w=td.w, lambda_U=td.lambda_U, gamma=td.gamma)

# TV-DRO full model (direct route)
full_model, full_vars = build_full_tv_model(tv; optimizer=Gurobi.Optimizer)
optimize!(full_model)

V_true = result[:Z0]
if termination_status(full_model) == MOI.OPTIMAL
    V_dir = objective_value(full_model)
    @printf("  V*(true)  = %.6f\n", V_true)
    @printf("  V^Dir(TV) = %.6f\n", V_dir)
    @printf("  Gap       = %.6f (%.2f%%)\n", V_dir - V_true,
            100 * (V_dir - V_true) / max(abs(V_true), 1e-10))
    if V_true <= V_dir + 1e-4
        println("  ✓ V* ≤ V^Dir")
    else
        println("  ✗ V* > V^Dir — violation!")
    end
else
    println("  TV-DRO full model: $(termination_status(full_model))")
end
