using JuMP
using Gurobi, Mosek, MosekTools
using HiGHS, Hypatia
using LinearAlgebra
using Infiltrator
# Load modules
using Revise
includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("build_full_model.jl")
includet("build_dualized_outer_subprob.jl")
includet("build_nominal_sp.jl")
using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_factor_model, generate_capacity_scenarios_uniform_model, print_network_summary

println("="^80)
println("TESTING FULL 2DRNDP MODEL CONSTRUCTION")
println("="^80)
function show_nonzero(var; tol=1e-8)
    indices = findall(x -> abs(x) > tol, var)
    
    if isempty(indices)
        println("All values ≈ 0")
        return
    end
    
    println("Non-zero values ($(length(indices)) entries):")
    for idx in indices
        # CartesianIndex를 튜플로 변환
        pos = Tuple(idx)
        println("  $pos => $(var[idx])")
    end
end
# Model parameters
S = 50# Number of scenarios
ϕU = 10.0  # Upper bound on interdiction effectiveness
λU = 10.0  # Upper bound on λ
γ = 2.0  # Interdiction budget
w = 1.0  # Budget weight
v = 1.0  # Interdiction effectiveness parameter (NOT the decision variable ν!)
seed = 42
mip_solver = Gurobi.Optimizer
conic_solver = Mosek.Optimizer

# Generate a small test network
println("\n[1] Generating 3×3 grid network...")
network = generate_grid_network(3, 3, seed=seed)
print_network_summary(network)

# ===== Use Factor Model =====
# capacities, F = generate_capacity_scenarios(length(network.arcs), network.interdictable_arcs, S, seed=120)
# ===== Use Uniform Model =====
capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)

# S=1
# capacities = capacities[:,2]
# @infiltrate
# Build uncertainty set
# ===== BUILD ROBUST COUNTERPART MATRICES R AND r =====
println("\n" * "="^80)
println("BUILD ROBUST COUNTERPART MATRICES (R, r)")
println("="^80)

# Remove dummy arc from capacity scenarios (|A| = regular arcs only)
capacity_scenarios_regular = capacities[1:end-1, :]  # Remove last row (dummy arc)
epsilon = 0.5  # Robustness parameter
# @infiltrate
println("\n[3] Building R and r matrices...")
println("Number of regular arcs |A|: $(size(capacity_scenarios_regular, 1))")
println("Number of scenarios S: $S")
println("Robustness parameter ε: $epsilon")

R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)
solve_full_model = true
# model, vars = build_full_2SP_model(network, S, ϕU, λU, γ, w, v,uncertainty_set)
# optimize!(model)
# @infiltrate
if solve_full_model
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
    model, vars = build_full_2DRNDP_model(network, S, ϕU, λU, γ, w, v,uncertainty_set, mip_solver=mip_solver, conic_solver=conic_solver)
    add_sparsity_constraints!(model, vars, network, S)
    @constraint(model, vars[:λ]>=0.01)
    # @constraint(model, vars[:x].== [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # @constraint(model, vars[:h][9].== 0.01)
    # model, vars = build_full_2SP_model(network, S, ϕU, λU, γ, w, v,uncertainty_set)
    println("\n[3] Model structure verification:")
    println("  Scalar variables:")
    println("    nu: $(vars[:nu])")
    println("    λ: $(vars[:λ])")

    println("\n  Vector variables:")
    println("    x: $(length(vars[:x])) binary variables")
    println("    h: $(length(vars[:h])) continuous variables")
    println("    ψ0: $(length(vars[:ψ0])) auxiliary variables")

    println("\n  Scenario-indexed scalars:")
    println("    ηhat: $(length(vars[:ηhat])) variables (S scenarios)")
    println("    ηtilde: $(length(vars[:ηtilde])) variables (S scenarios)")

    println("\n  Matrix variables per scenario:")
    num_arcs = length(network.arcs)-1
    num_nodes = length(network.nodes)
    println("    Φhat: $(S) × $(num_arcs) × $(num_arcs) = $(S * num_arcs * num_arcs) variables")
    println("    Ψhat: $(S) × $(num_arcs) × $(num_arcs) = $(S * num_arcs * num_arcs) variables")
    println("    Φtilde: $(S) × $(num_arcs) × $(num_arcs) = $(S * num_arcs * num_arcs) variables")
    println("    Ψtilde: $(S) × $(num_arcs) × $(num_arcs) = $(S * num_arcs * num_arcs) variables")
    println("    Πhat: $(S) × $(num_nodes-1) × $(num_arcs) = $(S * (num_nodes-1) * num_arcs) variables")
    println("    Πtilde: $(S) × $(num_nodes-1) × $(num_arcs) = $(S * (num_nodes-1) * num_arcs) variables")
    println("    Y: $(S) × $(num_arcs) × $(num_arcs) = $(S * num_arcs * num_arcs) variables")
    println("    Yts: $(S) × $(num_arcs) = $(S * num_arcs) variables")

    println("\n[4] Checking specific constraints:")

    # Check constraint 14b (resource budget)
    budget_con = constraint_by_name(model, "resource_budget")
    println("  (14b) Resource budget constraint: ", budget_con !== nothing ? "✓" : "✗")

    # Check constraint 14c (total cost)
    cost_con = constraint_by_name(model, "total_cost")
    println("  (14c) Total cost constraint: ", cost_con !== nothing ? "✓" : "✗")


    # Check budget constraint for dual variables (14l)
    println("  (14l) Dual budget constraints: $(num_arcs) constraints (one per arc)")

    # Check linearization constraints (14q)
    println("  (14q) Linearization constraints: $(4 * num_arcs) constraints")

    println("\n[5] Model statistics:")
    println("  Total variables: $(num_variables(model))")
    println("  Binary variables: $(sum(is_binary(vars[:x][i]) for i in 1:num_arcs))")
    println("  Total constraints: $(sum(num_constraints(model, F, S) for (F, S) in list_of_constraint_types(model)))")

    println("\n" * "="^80)
    println("MODEL CONSTRUCTION TEST COMPLETE")
    println("="^80)
    optimize!(model)
    # Save objective value and optimal solution after solving

    # Check if the model has a feasible solution
    t_status = termination_status(model)
    p_status = primal_status(model)

    if t_status == MOI.OPTIMAL || t_status == MOI.FEASIBLE_POINT
        obj_value = objective_value(model)
        println("\nOptimal objective value: ", obj_value)

        # Save solution for selected variables
        sol = Dict()
        sol[:objective_value] = obj_value

        # Extract primal variables (add more if needed)
        sol[:x] = value.(vars[:x])
        sol[:h] = value.(vars[:h])
        sol[:nu] = value(vars[:nu])
        sol[:λ] = value(vars[:λ])
        sol[:ψ0] = value.(vars[:ψ0])
        # Save scenario values (examples)
        sol[:ηhat] = value.(vars[:ηhat])
        sol[:ηtilde] = value.(vars[:ηtilde])
        sol[:Yts_tilde] = value.(vars[:Yts_tilde])
        sol[:Ytilde] = value.(vars[:Ytilde])
        sol[:Πhat] = value.(vars[:Πhat])
        sol[:Πtilde] = value.(vars[:Πtilde])
        sol[:Φhat] = value.(vars[:Φhat])
        sol[:Φtilde] = value.(vars[:Φtilde])
        sol[:Ψhat] = value.(vars[:Ψhat])
        sol[:Ψtilde] = value.(vars[:Ψtilde])
        sol[:μhat] = value.(vars[:μhat])
        sol[:μtilde] = value.(vars[:μtilde])
        # Optionally save any other variables you want (e.g., dual variables)
        # ...
        println("interdicted arcs: ", [i for i in 1:length(sol[:x]) if sol[:x][i] == 1.0])
        recovered_arcs = [(i, sol[:h][i]) for i in 1:length(sol[:h]) if sol[:h][i] != 0.0]
        println("recovered arcs (index, amount): ", recovered_arcs)
        # Save to .jld2 file (assumes JLD2 is loaded elsewhere if not, replace with Serialization or plain text)
        try
            @info "Saving solution to 'full_model_solution.jld2'..."
            using JLD2
            JLD2.save("full_model_solution.jld2", "sol", sol)
        # Save the optimal objective value separately as a plain text file
        try
            open("full_model_objective_value.txt", "w") do io
                println(io, obj_value)
            end
        catch err_txt
            @warn "Could not save optimal objective value to txt: $err_txt"
        end
        # Save also in the JLD2 solution file (already included as sol[:objective_value])
        catch err
            @warn "Could not save using JLD2, writing to JSON instead."
            try
                using JSON
                open("full_model_solution.json", "w") do io
                    JSON.print(io, sol)
                end
            catch err2
                @error "Could not save solution to file: $err2"
            end
        end

    else
        println("\nNo feasible/optimal solution found. Status: $t_status")
    end
    @infiltrate
else
    # Load previously saved solution for x, λ, h (JLD2 preferred, JSON as fallback)
    local sol = nothing
    if isfile("full_model_solution.jld2")
        try
            using JLD2
            sol = JLD2.load("full_model_solution.jld2", "sol")
            println("Loaded solution from full_model_solution.jld2")
        catch err
            @warn "Could not load JLD2 solution: $err"
        end
    elseif isfile("full_model_solution.json")
        try
            using JSON
            sol_temp = nothing
            open("full_model_solution.json", "r") do io
                sol_temp = JSON.parse(IOBuffer(read(io, String)))
            end
            sol = sol_temp
            println("Loaded solution from full_model_solution.json")
        catch err
            @warn "Could not load JSON solution file: $err"
        end
    else
        error("No previously saved solution could be loaded for x, λ, h.")
    end

    if sol === nothing || !isa(sol, Dict) || isempty(sol) || !haskey(sol, :x) || !haskey(sol, :λ) || !haskey(sol, :h)
        error("Loaded solution is empty or missing required keys (x, λ, h) — cannot proceed.")
    end

    # Convert to correct types if necessary
    x_sol = Array{Float64}(sol[:x])
    λ_sol = sol[:λ]
    h_sol = Array{Float64}(sol[:h])
    ψ0_sol = Array{Float64}(sol[:ψ0])
    println("Loaded x: ", x_sol)
    println("Loaded λ: ", λ_sol)
    println("Loaded h: ", h_sol)

    # Build continuous conic subproblem
    model, vars = build_full_2DRNDP_model(network, S, ϕU, λU, γ, w, v,uncertainty_set, conic_solver=conic_solver, x_fixed=x_sol, λ_fixed=λ_sol, h_fixed=h_sol, ψ0_fixed=ψ0_sol)
    add_sparsity_constraints!(model, vars, network, S)
    optimize!(model)
    # objective value 파일에서 읽기 (간단하게)
    obj_val = isfile("full_model_objective_value.txt") ? parse(Float64, readline(open("full_model_objective_value.txt"))) : nothing
    new_obj = termination_status(model) == MOI.OPTIMAL ? objective_value(model) : nothing
    if obj_val !== nothing && new_obj !== nothing
        println("Stored objective value: ", obj_val)
        println("Newly solved objective value: ", new_obj)
        if isapprox(obj_val, new_obj; atol=1e-4, rtol=1e-6)
            println("✓ Objective values match within tolerance.")
        else
            println("✗ Objective mismatch! Difference: ", abs(obj_val - new_obj))
        end
    elseif obj_val === nothing
        println("No stored objective value found in file for comparison.")
    else
        println("Model did not solve to optimality, cannot compare objective values.")
    end
    # using Dualization
    # dual_model = dualize(model; dual_names = DualNames("dual_var_", "dual_con_"))
    # Convert DenseAxisArray to regular array for findall
    Yts_tilde_array = Array(sol[:Yts_tilde])
    inds = findall(>(0.01), Yts_tilde_array)
    vals = Yts_tilde_array[inds]
    println("Indices where Yts_tilde > 0.01: ", inds)
    println("Values: ", vals)
    # Print elements of sol[:Φtilde] > 0.01, including their indices
    # Build the dualized outer subproblem using the loaded x, λ, h solution values
    # Assumes you have access to the following objects: network, S, ϕU, γ, w, v, uncertainty_set

    dual_model, dual_vars, _ = build_dualized_outer_subproblem(network, S, ϕU, λU, γ, w, v, uncertainty_set, MosekTools.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol)
    optimize!(dual_model)
    # Compare the optimal objective values of primal (obj_val, loaded) and dual (dual_model)
    dual_obj_val = termination_status(dual_model) == MOI.OPTIMAL ? objective_value(dual_model) : nothing
    if obj_val !== nothing && dual_obj_val !== nothing
        println("Primal model stored objective value: ", obj_val)
        println("Dual model objective value:          ", dual_obj_val)
        obj_diff = abs(obj_val - dual_obj_val)
        if isapprox(obj_val, dual_obj_val; atol=1e-4, rtol=1e-6)
            @infiltrate
            println("✓ Duality gap is vanished within tolerance.")
        else
            @infiltrate
            println("✗ Duality gap exists! Duality gap: ", obj_diff)
        end
    else
        @infiltrate
        println("Could not compare primal (stored) and dual objectives: one or both objectives missing or not solved optimally. Duality gap cannot be calculated.")
    end
    println("Dualized outer subproblem built.")
    @infiltrate
end