"""
compare_objF_vi.jl — obj_F ≥ 0 VI의 효과 비교.

두 번의 Benders solve:
  Run A: baseline (add_objF_vi=false)
  Run B: VI 활성화 (add_objF_vi=true)

둘 다 동일 instance/seed/파라미터. Iter 수, wall time, Z₀ 비교.

Usage:
  julia compare_objF_vi.jl <instance> <S> [<γ>]
예:
  julia compare_objF_vi.jl abilene 10
  julia compare_objF_vi.jl grid_3x3 5
"""

using Revise
using JuMP
using Gurobi
using Printf
using LinearAlgebra
using Dates
using Infiltrator

if !@isdefined(NetworkGenerator)
    include("../network_generator.jl")
end
using .NetworkGenerator

includet("true_dro_data.jl")
includet("true_dro_build_omp.jl")
includet("true_dro_build_subproblem.jl")
includet("true_dro_build_isp_leader.jl")
includet("true_dro_build_isp_follower.jl")
includet("true_dro_benders.jl")
includet("true_dro_mincut_vi.jl")

# ---- Instance setup ----
const NETWORK_CONFIGS = Dict(
    :grid_3x3   => Dict(:type => :grid, :m => 3, :n => 3),
    :grid_5x5   => Dict(:type => :grid, :m => 5, :n => 5),
    :sioux_falls => Dict(:type => :real_world, :generator => generate_sioux_falls_network),
    :nobel_us   => Dict(:type => :real_world, :generator => generate_nobel_us_network),
    :abilene    => Dict(:type => :real_world, :generator => generate_abilene_network),
    :polska     => Dict(:type => :real_world, :generator => generate_polska_network),
)

function setup_instance(config_key::Symbol; S=10, γ_override=nothing, γ_ratio=0.10,
                        ρ=0.2, seed=42, ε_hat=0.1, ε_tilde=0.1)
    config = NETWORK_CONFIGS[config_key]
    if config[:type] == :grid
        network = generate_grid_network(config[:m], config[:n]; seed=seed)
    else
        network = config[:generator]()
    end
    num_arcs = length(network.arcs) - 1
    num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
    γ = γ_override === nothing ? (
            config_key in (:sioux_falls, :abilene) ? 2 :
            ceil(Int, γ_ratio * num_interdictable)
        ) : γ_override

    capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
        interdictable_arcs=network.interdictable_arcs, seed=seed)

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(ρ * γ * c_bar; digits=4)
    λU = 2.0
    q_hat = fill(1.0 / S, S)

    td = make_true_dro_data(network, capacities, q_hat, ε_hat, ε_tilde;
                            w=w, lambda_U=λU, gamma=γ)
    @printf("  |A|=%d, S=%d, γ=%d, w=%.4f, λU=%.1f, ε̂=%.2f, ε̃=%.2f\n",
            num_arcs, S, γ, w, λU, ε_hat, ε_tilde)
    return network, td
end

# ---- CLI args ----
instance = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :abilene
S_val    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
γ_ovr    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : nothing

println("=" ^ 70)
println("obj_F ≥ 0 VI 비교 — $instance, S=$S_val" * (γ_ovr !== nothing ? ", γ=$γ_ovr" : ""))
println("=" ^ 70)

_, td = setup_instance(instance; S=S_val, γ_override=γ_ovr)

# ---- 공통 Benders 옵션 ----
common_kwargs = (
    mip_optimizer = Gurobi.Optimizer,
    nlp_optimizer = Gurobi.Optimizer,
    inexact = true,
    max_iter = 1000,
    tol = 1e-4,
    verbose = false,
    sub_verbose = false,
    sub_time_limit = 30.0,
    mini_benders = false,
    strengthen_cuts = :none,
    valid_inequality = :none,
)

# ===== Run A: baseline =====
println("\n" * "-" ^ 70)
println("Run A: BASELINE (add_objF_vi=false)")
println("-" ^ 70)
flush(stdout)

t_A = time()
result_A = true_dro_benders_optimize!(td; common_kwargs..., add_objF_vi=false)
wall_A = time() - t_A

gap_A = abs(result_A[:upper_bound] - result_A[:lower_bound]) /
        max(abs(result_A[:upper_bound]), 1e-10)

@printf("  status=%s, Z₀=%.6f, iters=%d, wall=%.2fs\n",
        result_A[:status], result_A[:Z0], result_A[:iters], wall_A)
@printf("  LB=%.6f, UB=%.6f, gap=%.2e\n",
        result_A[:lower_bound], result_A[:upper_bound], gap_A)

# ===== Run B: with VI =====
println("\n" * "-" ^ 70)
println("Run B: WITH obj_F ≥ 0 VI (add_objF_vi=true)")
println("-" ^ 70)
flush(stdout)

t_B = time()
result_B = true_dro_benders_optimize!(td; common_kwargs..., add_objF_vi=true)
wall_B = time() - t_B

gap_B = abs(result_B[:upper_bound] - result_B[:lower_bound]) /
        max(abs(result_B[:upper_bound]), 1e-10)

@printf("  status=%s, Z₀=%.6f, iters=%d, wall=%.2fs\n",
        result_B[:status], result_B[:Z0], result_B[:iters], wall_B)
@printf("  LB=%.6f, UB=%.6f, gap=%.2e\n",
        result_B[:lower_bound], result_B[:upper_bound], gap_B)

# ===== Summary =====
println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)
@printf("%-25s %12s %12s\n", "Metric", "Baseline", "With VI")
println("-" ^ 50)
@printf("%-25s %12s %12s\n", "status",
        string(result_A[:status]), string(result_B[:status]))
@printf("%-25s %12.6f %12.6f\n", "Z₀",        result_A[:Z0],          result_B[:Z0])
@printf("%-25s %12d %12d\n",     "iters",     result_A[:iters],       result_B[:iters])
@printf("%-25s %12.2f %12.2f\n", "wall time (s)", wall_A,              wall_B)
@printf("%-25s %12.2e %12.2e\n", "gap",       gap_A,                  gap_B)
@printf("%-25s %12.6f %12.6f\n", "LB",        result_A[:lower_bound], result_B[:lower_bound])
@printf("%-25s %12.6f %12.6f\n", "UB",        result_A[:upper_bound], result_B[:upper_bound])

println("-" ^ 50)
if result_A[:iters] > 0 && result_B[:iters] > 0
    iter_ratio = result_B[:iters] / result_A[:iters]
    time_ratio = wall_B / max(wall_A, 1e-6)
    @printf("  iter ratio  B/A = %.2fx  (< 1.0 means VI is faster)\n", iter_ratio)
    @printf("  time ratio  B/A = %.2fx\n", time_ratio)
    if abs(result_A[:Z0] - result_B[:Z0]) > 1e-3
        @printf("  ⚠️  Z₀ 차이 = %.6f — VI가 optimal cut-off 했을 가능성\n",
                abs(result_A[:Z0] - result_B[:Z0]))
    else
        println("  ✓ Z₀ 일치 (VI validity 확인)")
    end
end
