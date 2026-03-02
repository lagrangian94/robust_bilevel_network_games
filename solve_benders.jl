using JuMP
using Gurobi
using HiGHS
using LinearAlgebra
using Infiltrator
using Plots
using Serialization
# Load modules
using Revise
includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("strict_benders.jl")
includet("plot_benders.jl")
using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_factor_model, generate_capacity_scenarios_uniform_model, print_network_summary

println("="^80)
println("TESTING STRICT BENDERS MODEL CONSTRUCTION")
println("="^80)

# Model parameters
S = 2  # Number of scenarios
ϕU = 10.0  # Upper bound on interdiction effectiveness
λU = 10.0  # Upper bound on λ
γ = 2.0  # Interdiction budget
w = 1.0  # Budget weight
v = 1.0  # Interdiction effectiveness parameter (NOT the decision variable ν!)
seed = 42

epsilon = 0.5  # Robustness parameter

# ===== JIT Warm-up =====
#
# Julia는 JIT(Just-In-Time) 컴파일러를 사용한다. 함수가 처음 호출될 때 Julia는
# 해당 함수의 인자 타입에 맞는 네이티브 머신코드를 생성(컴파일)한다.
# 이 과정은 한 번만 발생하며, 이후 동일 타입으로 호출하면 이미 컴파일된 코드를 재사용한다.
#
# 문제: 첫 번째로 실행되는 알고리즘이 JIT 컴파일 시간을 떠안게 되어,
# 실제 알고리즘 실행 시간보다 훨씬 느리게 측정된다.
# 예) Strict Benders가 첫 번째로 실행되면 23초, warm-up 후엔 2.8초.
#
# 해결: 실제 측정 전에 작은 인스턴스(3x3)로 모든 코드 경로를 한 번씩 실행하여
# JIT 컴파일을 완료시킨다. warm-up 실행의 결과는 버린다.
# 이후 실제 인스턴스 측정에서는 순수 알고리즘 실행 시간만 측정된다.
# 추가로 각 측정 전 GC.gc()를 호출하여 가비지 컬렉션이 측정 중에 개입하는 것을 방지한다.
#
println("\n" * "="^80)
println("JIT WARM-UP (3x3 grid, S=1, results discarded)")
println("="^80)

nested_benders = false
multi_cut = nested_benders ? true : false

warmup_S = 1
actual_S = S  # 실제 S를 보존
S = warmup_S  # solver 내부에서 전역 S, R, r_dict, xi_bar, epsilon을 참조하므로 임시로 변경

network = generate_grid_network(3, 3, seed=seed)
warm_cap, _ = generate_capacity_scenarios_uniform_model(length(network.arcs), warmup_S, seed=seed)
R, r_dict, xi_bar = build_robust_counterpart_matrices(warm_cap[1:end-1, :], epsilon)
warm_uset = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

wm1, wv1 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=false)
strict_benders_optimize!(wm1, wv1, network, ϕU, λU, γ, w, warm_uset; optimizer=Gurobi.Optimizer)
if nested_benders
    includet("nested_benders_trust_region.jl")
    wm2, wv2 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
    tr_nested_benders_optimize!(wm2, wv2, network, ϕU, λU, γ, w, warm_uset; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer, multi_cut=true)
end

S = actual_S  # 실제 S 복원
println("Warm-up complete.\n")

# ===== Generate Network & Uncertainty Set =====
println("\n[1] Generating 3×3 grid network...")
network = generate_grid_network(3, 3, seed=seed)
print_network_summary(network)
# ===== Use Factor Model =====
# capacities, F = generate_capacity_scenarios(length(network.arcs), network.interdictable_arcs, S, seed=120)
# ===== Use Uniform Model =====
capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)

# Build uncertainty set
# ===== BUILD ROBUST COUNTERPART MATRICES R AND r =====
println("\n" * "="^80)
println("BUILD ROBUST COUNTERPART MATRICES (R, r)")
println("="^80)

# Remove dummy arc from capacity scenarios (|A| = regular arcs only)
capacity_scenarios_regular = capacities[1:end-1, :]  # Remove last row (dummy arc)

println("\n[3] Building R and r matrices...")
println("Number of regular arcs |A|: $(size(capacity_scenarios_regular, 1))")
println("Number of scenarios S: $S")
println("Robustness parameter ε: $epsilon")

R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)


println("\n[2] Building model...")
println("  Parameters:")
println("    S (scenarios) = $S")
println("    ϕU (interdiction bound) = $ϕU")
println("    γ (interdiction budget) = $γ")
println("    w (budget weight) = $w")
println("    v (interdiction effectiveness param) = $v")
println("  Note: v is a parameter in COP matrix [Φ - v*W]")
println("        ν (nu) is a decision variable in objective t + w*ν")
# Build model (without optimizer for initial testing)

GC.gc()
model, vars = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=multi_cut)
if nested_benders
    # ## trust region nested benders
    includet("nested_benders_trust_region.jl")
    result = tr_nested_benders_optimize!(model, vars, network, ϕU, λU, γ, w, uncertainty_set; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer, multi_cut=multi_cut)
    plot_tr_nested_benders_convergence(result)
    serialize("tr_nested_benders_result.jls", result)
    println("Time taken: $(result[:solution_time]) seconds")
    ## basic nested benders 
    # includet("nested_benders.jl")
    # result = nested_benders_optimize!(model, vars, network, ϕU, λU, γ, w, uncertainty_set; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer, multi_cut=multi_cut)
    # plot_nested_benders_convergence(result)
    # serialize("nested_benders_result.jls", result)
    # println("Time taken: $(result[:solution_time]) seconds")
else
    time_start = time()
    result = strict_benders_optimize!(model, vars, network, ϕU, λU, γ, w, uncertainty_set; optimizer=Gurobi.Optimizer)
    time_end = time()
    println("Time taken: $(time_end - time_start) seconds")
    plot_benders_convergence(result)
end
    


using Serialization
tr_result = deserialize("tr_nested_benders_result.jls")
basic_result = deserialize("nested_benders_result.jls")
compare_inner_iter(tr_result, basic_result)