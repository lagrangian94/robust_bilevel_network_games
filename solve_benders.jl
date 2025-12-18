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
S = 20  # Number of scenarios
ϕU = 10.0  # Upper bound on interdiction effectiveness
λU = 10.0  # Upper bound on λ
γ = 2.0  # Interdiction budget
w = 1.0  # Budget weight
v = 1.0  # Interdiction effectiveness parameter (NOT the decision variable ν!)
seed = 42

# Generate a small test network
println("\n[1] Generating 3×3 grid network...")
network = generate_grid_network(5, 5, seed=seed)
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
epsilon = 0.5  # Robustness parameter

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


multi_cut = true
nested_benders = true
if !nested_benders
    multi_cut = false
end

model, vars = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=multi_cut)
if nested_benders
    ## trust region nested benders
    includet("nested_benders_trust_region.jl")
    result = tr_nested_benders_optimize!(model, vars, network, ϕU, λU, γ, w, uncertainty_set; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer, multi_cut=multi_cut)
    plot_tr_nested_benders_convergence(result)
    serialize("tr_nested_benders_result.jls", result)
    println("Time taken: $(result[:solution_time]) seconds")
    ## basic nested benders 
    includet("nested_benders.jl")
    model, vars = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=multi_cut)
    result = nested_benders_optimize!(model, vars, network, ϕU, λU, γ, w, uncertainty_set; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer, multi_cut=multi_cut)
    plot_nested_benders_convergence(result)
    serialize("nested_benders_result.jls", result)
    println("Time taken: $(result[:solution_time]) seconds")
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