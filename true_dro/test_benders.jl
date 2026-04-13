"""
test_benders.jl — True-DRO Benders 검증.

Interactive input으로 network config, S 선택.
compare_benders.jl의 setup_instance 패턴 참고.
"""

"""
5x5, S=10, 0.1, 0.1, Sub time =30, mini-bd=true
User-callback calls 243, time in user-callback 0.00 sec
  Sub: Z₀=11.500237 [LOCAL], α=[0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.297,0.720,2.316,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]
  Iter 637: LB=8.133596  UB=8.133597  gap=1.46e-07
[ Info: True-DRO Benders converged at iter 637 (gap=1.455851768957056e-7)

----------------------------------------
True-DRO: status=Optimal, Z₀=8.133597, iters=637
  LB=8.133596, UB=8.133597, gap=1.46e-07
  x* = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  α* = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.112, 1.175, 0.000, 0.000, 1.047, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
============================================================
Primal Recovery: x* → α* → (h*, λ*, ψ⁰*)
============================================================

lambdaU=2.0 했을 시,
  Sub: Z₀=8.133597, α=[0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,1.074,1.222,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,1.038,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]
  Iter 198: LB=8.133598  UB=8.133597  gap=8.90e-08
[ Info: True-DRO Benders converged at iter 198 (gap=8.904142162382194e-8)

----------------------------------------
True-DRO: status=Optimal, Z₀=8.133597, iters=198
  LB=8.133598, UB=8.133597, gap=8.90e-08
  x* = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  α* = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.074, 1.222, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.038, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
============================================================

lambdaU=2.0, omp VI (phase A) 했을 시,
  Sub: Z₀=8.133599, α=[0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,1.494,0.000,0.000,0.000,0.704,0.000,0.000,0.000,1.135,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]
  Iter 102: LB=8.133598  UB=8.133599  gap=1.67e-07
[ Info: True-DRO Benders converged at iter 102 (gap=1.669338904071469e-7)

----------------------------------------
True-DRO: status=Optimal, Z₀=8.133599, iters=102
  LB=8.133598, UB=8.133599, gap=1.67e-07
  x* = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  α* = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.494, 0.000, 0.000, 0.000, 0.704, 0.000, 0.000, 0.000, 1.135, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
  Wall time: 362.06 sec


LambdaU=2.0 ,omp VI (phase A + 2B) 했을 시,
  Sub: Z₀=8.133600, α=[0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,1.891,0.000,0.000,0.000,0.865,0.000,0.000,0.000,0.578,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]
  Iter 105: LB=8.133598  UB=8.133600  gap=2.32e-07
[ Info: True-DRO Benders converged at iter 105 (gap=2.3234024756858074e-7)

----------------------------------------
True-DRO: status=Optimal, Z₀=8.133600, iters=105
  LB=8.133598, UB=8.133600, gap=2.32e-07
  x* = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  α* = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.891, 0.000, 0.000, 0.000, 0.865, 0.000, 0.000, 0.000, 0.578, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
  Wall time: 564.02 sec
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
includet("true_dro_build_isp_leader.jl")
includet("true_dro_build_isp_follower.jl")
includet("true_dro_benders.jl")
includet("true_dro_mincut_vi.jl")
includet("true_dro_recover.jl")

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
    λU = 2.0

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
print("Sub time limit (sec, 0=none) [30]: "); tl_str = strip(readline())
sub_tl = isempty(tl_str) ? 30.0 : parse(Float64, tl_str)
sub_tl = sub_tl <= 0 ? nothing : sub_tl
print("Mini-Benders? (y/n) [n]: "); mb_str = strip(readline())
use_mini_benders = lowercase(mb_str) == "y"
max_mb_iter = 5
print("Strengthen cuts? (none/mw) [none]: "); sc_str = strip(readline())
strengthen_cuts = isempty(sc_str) || sc_str == "none" ? :none : Symbol(sc_str)
print("Valid inequality? (none/mincut) [none]: "); vi_str = strip(readline())
vi_sym = isempty(vi_str) || vi_str == "none" ? :none : Symbol(vi_str)

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
    inexact=true,
    max_iter=1000,
    tol=1e-4,
    verbose=true,
    sub_verbose=true,
    sub_time_limit=sub_tl,
    mini_benders=use_mini_benders,
    lp_optimizer=(use_mini_benders ? Gurobi.Optimizer : nothing),
    max_mini_benders_iter=max_mb_iter,
    strengthen_cuts=strengthen_cuts,
    valid_inequality=vi_sym)

gap = abs(result[:upper_bound] - result[:lower_bound]) /
      max(abs(result[:upper_bound]), 1e-10)

println("\n" * "-" ^ 40)
wt = get(result, :wall_time, NaN)
@printf("True-DRO: status=%s, Z₀=%.6f, iters=%d, time=%.2fs\n", result[:status], result[:Z0], result[:iters], wt)
@printf("  LB=%.6f, UB=%.6f, gap=%.2e\n", result[:lower_bound], result[:upper_bound], gap)
x_int = round.(Int, result[:x])
println("  x* = $x_int")
α_str = join([@sprintf("%.3f", a) for a in result[:α]], ", ")
println("  α* = [$α_str]")

# ===== Primal Recovery =====
rec = recover_and_print(td, result; optimizer=HiGHS.Optimizer)

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


# ===== 3. ε→0 nominal convergence =====
println("\n" * "=" ^ 70)
println("3. ε→0 nominal convergence")
println("=" ^ 70)

epsilons = [0.3, 0.1, 0.05, 0.01, 0.001]
eps_results = []

for ε in epsilons
    network_eps, td_eps = setup_true_dro_instance(instance_key; S=S,
        epsilon_hat=ε, epsilon_tilde=ε)
    r = true_dro_benders_optimize!(td_eps;
        mip_optimizer=Gurobi.Optimizer,
        nlp_optimizer=Gurobi.Optimizer,
        max_iter=1000, tol=1e-4, verbose=false,
        sub_time_limit=sub_tl,
        mini_benders=use_mini_benders,
        lp_optimizer=(use_mini_benders ? Gurobi.Optimizer : nothing),
        max_mini_benders_iter=max_mb_iter)
    push!(eps_results, (ε=ε, Z0=r[:Z0], status=r[:status], iters=r[:iters]))
    @printf("  ε=%.4f → Z₀=%.6f  (%s, %d iters)\n", ε, r[:Z0], r[:status], r[:iters])
end

# ε 작아질수록 Z₀ monotone non-increasing (tighter ambiguity → smaller worst-case)
objs = [r.Z0 for r in eps_results]
is_mono = all(objs[i] >= objs[i+1] - 1e-4 for i in 1:length(objs)-1)
if is_mono
    println("  ✓ Z₀ monotone non-increasing as ε→0")
else
    println("  ✗ Z₀ not monotone — check!")
end
