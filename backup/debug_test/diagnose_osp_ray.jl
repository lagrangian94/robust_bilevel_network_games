"""
Diagnostic script: Extract Mosek's infeasibility certificate (ray) from the
DUAL_INFEASIBLE OSP to identify which variables drive the unboundedness.

When primal_status == INFEASIBILITY_CERTIFICATE, value.(var) gives the ray direction.
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using LinearAlgebra
using Infiltrator
using Revise

includet("../network_generator.jl")
includet("../build_uncertainty_set.jl")
# Use TEST copy — includes are commented out in test copies
includet("test_build_dualized_outer_subprob.jl")
includet("test_strict_benders.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

# ===== Same parameters as compare_benders.jl =====
S = 1
λU = 10.0
γ_ratio = 0.10
ρ = 0.2
v = 1.0
seed = 42
epsilon = 0.5
ϕU = 1/epsilon  # = 2.0 (the problematic value)

# ===== Generate Network =====
network = generate_grid_network(4, 4, seed=seed)
print_network_summary(network)

num_arcs = length(network.arcs) - 1
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, γ_ratio * num_interdictable)

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = round(ρ * γ * c_bar, digits=4)

capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

# Override γ, w (same as compare_benders.jl)
γ = 2.0
w = 1.0

println("\n" * "="^80)
println("DIAGNOSTIC: OSP Ray Extraction")
println("  ϕU = $ϕU, γ = $γ, w = $w")
println("="^80)

# ===== Step 1: Get initial OMP solution =====
model_omp, vars_omp = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=false)

# Run one Benders iteration to get a non-trivial solution
optimize!(model_omp)
x_sol_init = value.(vars_omp[:x])
h_sol_init = value.(vars_omp[:h])
λ_sol_init = value(vars_omp[:λ])
ψ0_sol_init = value.(vars_omp[:ψ0])

println("\nInitial OMP solution (iter 0): x = $x_sol_init, λ = $λ_sol_init")

# Build OSP with initial solution, solve, add cut, re-solve OMP
osp_model_init, osp_vars_init, osp_data_init = build_dualized_outer_subproblem(
    network, S, ϕU, λU, γ, w, v, uncertainty_set,
    Mosek.Optimizer, λ_sol_init, x_sol_init, h_sol_init, ψ0_sol_init)
optimize!(osp_model_init)
t_status_init = termination_status(osp_model_init)
println("Iter 0 OSP status: $t_status_init")

if t_status_init == MOI.OPTIMAL
    # Extract cut and add to OMP
    obj_val_init = objective_value(osp_model_init)
    println("  OSP obj = $obj_val_init → adding optimality cut to OMP")

    x = vars_omp[:x]
    h = vars_omp[:h]
    λ_var = vars_omp[:λ]
    ψ0 = vars_omp[:ψ0]
    t_0 = vars_omp[:t_0]
    E_mat = osp_data_init[:E]
    d0_vec = osp_data_init[:d0]
    xi_bar_val = uncertainty_set[:xi_bar]

    # Get cut coefficients
    Uhat1_v = value.(osp_vars_init[:Uhat1])
    Utilde1_v = value.(osp_vars_init[:Utilde1])
    Uhat3_v = value.(osp_vars_init[:Uhat3])
    Utilde3_v = value.(osp_vars_init[:Utilde3])
    βtilde1_1_v = value.(osp_vars_init[:βtilde1_1])
    βtilde1_3_v = value.(osp_vars_init[:βtilde1_3])
    Ztilde1_3_v = value.(osp_vars_init[:Ztilde1_3])
    βhat1_1_v = value.(osp_vars_init[:βhat1_1])
    Phat1_Φ_v = value.(osp_vars_init[:Phat1_Φ])
    Phat1_Π_v = value.(osp_vars_init[:Phat1_Π])
    Phat2_Φ_v = value.(osp_vars_init[:Phat2_Φ])
    Phat2_Π_v = value.(osp_vars_init[:Phat2_Π])
    Ptilde1_Φ_v = value.(osp_vars_init[:Ptilde1_Φ])
    Ptilde1_Π_v = value.(osp_vars_init[:Ptilde1_Π])
    Ptilde2_Φ_v = value.(osp_vars_init[:Ptilde2_Φ])
    Ptilde2_Π_v = value.(osp_vars_init[:Ptilde2_Π])
    Ptilde1_Y_v = value.(osp_vars_init[:Ptilde1_Y])
    Ptilde2_Y_v = value.(osp_vars_init[:Ptilde2_Y])
    Ptilde1_Yts_v = value.(osp_vars_init[:Ptilde1_Yts])
    Ptilde2_Yts_v = value.(osp_vars_init[:Ptilde2_Yts])

    diag_x_E = Diagonal(x) * E_mat
    diag_λ_ψ = Diagonal(λ_var*ones(num_arcs) - v.*ψ0)

    intercept_v = [(d0_vec' * βhat1_1_v[s,:]) for s in 1:S]
    intercept_v .+= [-ϕU*sum(Phat1_Φ_v[s,:,:]) - ϕU*sum(Phat1_Π_v[s,:,:]) for s in 1:S]
    intercept_v .+= [-ϕU*sum(Phat2_Φ_v[s,:,:]) - ϕU*sum(Phat2_Π_v[s,:,:]) for s in 1:S]
    intercept_v .+= [-ϕU*sum(Ptilde1_Φ_v[s,:,:]) - ϕU*sum(Ptilde1_Π_v[s,:,:]) - ϕU*sum(Ptilde1_Y_v[s,:,:]) - ϕU*sum(Ptilde1_Yts_v[s,:]) for s in 1:S]
    intercept_v .+= [-ϕU*sum(Ptilde2_Φ_v[s,:,:]) - ϕU*sum(Ptilde2_Π_v[s,:,:]) - ϕU*sum(Ptilde2_Y_v[s,:,:]) - ϕU*sum(Ptilde2_Yts_v[s,:]) for s in 1:S]

    cut_1 = -ϕU * [sum((Uhat1_v[s,:,:] + Utilde1_v[s,:,:]) .* diag_x_E) for s in 1:S]
    cut_2 = -ϕU * [sum((Uhat3_v[s,:,:] + Utilde3_v[s,:,:]) .* (E_mat - diag_x_E)) for s in 1:S]
    cut_3 = [sum(Ztilde1_3_v[s,:,:] .* (diag_λ_ψ * diagm(xi_bar_val[s]))) for s in 1:S]
    cut_4 = [(d0_vec' * βtilde1_1_v[s,:]) * λ_var for s in 1:S]
    cut_5 = -1 * [(h + diag_λ_ψ * xi_bar_val[s])' * βtilde1_3_v[s,:] for s in 1:S]
    opt_cut = sum(cut_1) + sum(cut_2) + sum(cut_3) + sum(cut_4) + sum(cut_5) + sum(intercept_v)
    @constraint(model_omp, t_0 >= opt_cut)

    # Re-solve OMP with cut
    optimize!(model_omp)
    x_sol = value.(vars_omp[:x])
    h_sol = value.(vars_omp[:h])
    λ_sol = value(vars_omp[:λ])
    ψ0_sol = value.(vars_omp[:ψ0])

    println("\nAfter 1 cut — OMP solution:")
    println("  λ = $λ_sol")
    println("  x nonzero: $(findall(x_sol .> 0.5))")
    println("  h[nonzero] = $(findall(h_sol .> 1e-6))")
    println("  ψ0[nonzero] = $(findall(ψ0_sol .> 1e-6))")
else
    # OSP already failed on first try — use initial solution
    x_sol = x_sol_init
    h_sol = h_sol_init
    λ_sol = λ_sol_init
    ψ0_sol = ψ0_sol_init
    println("  OSP failed on initial solution!")
end

# ===== Step 2: Build OSP =====
osp_model, osp_vars, osp_data = build_dualized_outer_subproblem(
    network, S, ϕU, λU, γ, w, v, uncertainty_set,
    Mosek.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol)

# ===== Step 3: Solve and extract ray =====
optimize!(osp_model)
t_status = termination_status(osp_model)
p_status = primal_status(osp_model)
d_status = dual_status(osp_model)

println("\nOSP Status:")
println("  Termination: $t_status")
println("  Primal: $p_status")
println("  Dual: $d_status")

if p_status == MOI.INFEASIBILITY_CERTIFICATE
    println("\n" * "="^80)
    println("INFEASIBILITY CERTIFICATE (RAY) ANALYSIS")
    println("="^80)

    # Extract ray values for all variable groups
    ray_vars = Dict{String, Any}()

    # α
    α_ray = value.(osp_model[:α])
    ray_vars["α"] = α_ray

    # M variables (PSD)
    Mhat_ray = value.(osp_model[:Mhat])
    Mtilde_ray = value.(osp_model[:Mtilde])
    ray_vars["Mhat"] = Mhat_ray
    ray_vars["Mtilde"] = Mtilde_ray

    # U variables
    Uhat1_ray = value.(osp_model[:Uhat1])
    Uhat2_ray = value.(osp_model[:Uhat2])
    Uhat3_ray = value.(osp_model[:Uhat3])
    Utilde1_ray = value.(osp_model[:Utilde1])
    Utilde2_ray = value.(osp_model[:Utilde2])
    Utilde3_ray = value.(osp_model[:Utilde3])
    ray_vars["Uhat1"] = Uhat1_ray
    ray_vars["Uhat2"] = Uhat2_ray
    ray_vars["Uhat3"] = Uhat3_ray
    ray_vars["Utilde1"] = Utilde1_ray
    ray_vars["Utilde2"] = Utilde2_ray
    ray_vars["Utilde3"] = Utilde3_ray

    # β variables
    βhat1_ray = value.(osp_model[:βhat1])
    βhat2_ray = value.(osp_model[:βhat2])
    βtilde1_ray = value.(osp_model[:βtilde1])
    βtilde2_ray = value.(osp_model[:βtilde2])
    ray_vars["βhat1"] = βhat1_ray
    ray_vars["βhat2"] = βhat2_ray
    ray_vars["βtilde1"] = βtilde1_ray
    ray_vars["βtilde2"] = βtilde2_ray

    # Z variables (free)
    Zhat1_ray = value.(osp_model[:Zhat1])
    Zhat2_ray = value.(osp_model[:Zhat2])
    Ztilde1_ray = value.(osp_model[:Ztilde1])
    Ztilde2_ray = value.(osp_model[:Ztilde2])
    ray_vars["Zhat1"] = Zhat1_ray
    ray_vars["Zhat2"] = Zhat2_ray
    ray_vars["Ztilde1"] = Ztilde1_ray
    ray_vars["Ztilde2"] = Ztilde2_ray

    # Γ variables (SOC)
    Γhat1_ray = value.(osp_model[:Γhat1])
    Γhat2_ray = value.(osp_model[:Γhat2])
    Γtilde1_ray = value.(osp_model[:Γtilde1])
    Γtilde2_ray = value.(osp_model[:Γtilde2])
    ray_vars["Γhat1"] = Γhat1_ray
    ray_vars["Γhat2"] = Γhat2_ray
    ray_vars["Γtilde1"] = Γtilde1_ray
    ray_vars["Γtilde2"] = Γtilde2_ray

    # P variables
    P_names = ["Phat1_Φ", "Phat2_Φ", "Phat1_Π", "Phat2_Π",
               "Ptilde1_Φ", "Ptilde2_Φ", "Ptilde1_Π", "Ptilde2_Π",
               "Ptilde1_Y", "Ptilde2_Y", "Ptilde1_Yts", "Ptilde2_Yts"]
    for pname in P_names
        ray_vars[pname] = value.(osp_model[Symbol(pname)])
    end

    # ===== Print summary: which variable groups have nonzero ray components =====
    println("\nRay magnitude by variable group (L∞ norm):")
    println("  " * rpad("Variable", 20) * rpad("Max |ray|", 15) * "Sum |ray|")
    println("  " * "-"^50)

    sorted_vars = sort(collect(ray_vars), by=x->maximum(abs.(x.second)), rev=true)
    for (name, vals) in sorted_vars
        max_val = maximum(abs.(vals))
        sum_val = sum(abs.(vals))
        if max_val > 1e-8
            println("  " * rpad(name, 20) * rpad(round(max_val, digits=6), 15) * "$(round(sum_val, digits=6))")
        end
    end

    # ===== Compute objective ray contribution by term =====
    println("\n\nObjective ray contribution by term:")
    E = osp_data[:E]
    d0 = osp_data[:d0]
    xi_bar_val = uncertainty_set[:xi_bar]
    diag_x_E = Diagonal(x_sol) * E
    diag_λ_ψ = Diagonal(λ_sol*ones(num_arcs) - v.*ψ0_sol)

    for s in 1:S
        println("\n  Scenario $s:")

        t1 = -ϕU * sum((Uhat1_ray[s,:,:] + Utilde1_ray[s,:,:]) .* diag_x_E)
        t2 = -ϕU * sum((Uhat3_ray[s,:,:] + Utilde3_ray[s,:,:]) .* (E - diag_x_E))

        βhat1_1_ray = βhat1_ray[s, 1:num_arcs+1]
        t3 = d0' * βhat1_1_ray

        # Ztilde1_3 block
        block3_start_tilde = num_arcs+2 + (length(network.nodes)-1) + 1  # skip block1, block2
        # Actually need to compute block indices properly
        num_nodes = length(network.nodes)
        zt3_start = num_arcs+1 + (num_nodes-1) + 1
        zt3_end = zt3_start + num_arcs - 1
        Ztilde1_3_ray = Ztilde1_ray[s, zt3_start:zt3_end, :]
        t4 = sum(Ztilde1_3_ray .* (diag_λ_ψ * diagm(xi_bar_val[s])))

        βtilde1_1_ray = βtilde1_ray[s, 1:num_arcs+1]
        t5 = λ_sol * d0' * βtilde1_1_ray

        bt3_start = num_arcs+1 + (num_nodes-1) + 1
        bt3_end = bt3_start + num_arcs - 1
        βtilde1_3_ray = βtilde1_ray[s, bt3_start:bt3_end]
        t6 = -(h_sol + diag_λ_ψ * xi_bar_val[s])' * βtilde1_3_ray

        # P terms
        tp_ub_hat = -ϕU * sum(ray_vars["Phat1_Φ"][s,:,:]) - ϕU * sum(ray_vars["Phat1_Π"][s,:,:])
        tp_lb_hat = -ϕU * sum(ray_vars["Phat2_Φ"][s,:,:]) - ϕU * sum(ray_vars["Phat2_Π"][s,:,:])
        tp_ub_tilde = -ϕU * sum(ray_vars["Ptilde1_Φ"][s,:,:]) - ϕU * sum(ray_vars["Ptilde1_Π"][s,:,:]) - ϕU * sum(ray_vars["Ptilde1_Y"][s,:,:]) - ϕU * sum(ray_vars["Ptilde1_Yts"][s,:])
        tp_lb_tilde = -ϕU * sum(ray_vars["Ptilde2_Φ"][s,:,:]) - ϕU * sum(ray_vars["Ptilde2_Π"][s,:,:]) - ϕU * sum(ray_vars["Ptilde2_Y"][s,:,:]) - ϕU * sum(ray_vars["Ptilde2_Yts"][s,:])

        total = t1 + t2 + t3 + t4 + t5 + t6 + tp_ub_hat + tp_lb_hat + tp_ub_tilde + tp_lb_tilde

        terms = [
            ("U1+Ut1 (×-ϕU×x)", t1),
            ("U3+Ut3 (×-ϕU×(1-x))", t2),
            ("βhat1_1 (d0)", t3),
            ("Ztilde1_3 (λ-ψ)", t4),
            ("βtilde1_1 (λ×d0)", t5),
            ("βtilde1_3 (-(h+...))", t6),
            ("P_ub_hat (×-ϕU)", tp_ub_hat),
            ("P_lb_hat (×-ϕU)", tp_lb_hat),
            ("P_ub_tilde (×-ϕU)", tp_ub_tilde),
            ("P_lb_tilde (×-ϕU)", tp_lb_tilde),
        ]

        for (label, val) in terms
            if abs(val) > 1e-8
                println("    " * rpad(label, 30) * "$(round(val, digits=8))")
            end
        end
        println("    " * "-"^45)
        println("    " * rpad("TOTAL", 30) * "$(round(total, digits=8))")
    end

    # ===== Check what happens with ϕU=10 =====
    println("\n\n" * "="^80)
    println("COMPARISON: Same ray with ϕU=10 objective coefficients")
    println("="^80)
    ϕU_large = 10.0
    for s in 1:S
        t1 = -ϕU_large * sum((Uhat1_ray[s,:,:] + Utilde1_ray[s,:,:]) .* diag_x_E)
        t2 = -ϕU_large * sum((Uhat3_ray[s,:,:] + Utilde3_ray[s,:,:]) .* (E - diag_x_E))

        βhat1_1_ray = βhat1_ray[s, 1:num_arcs+1]
        t3 = d0' * βhat1_1_ray

        num_nodes = length(network.nodes)
        zt3_start = num_arcs+1 + (num_nodes-1) + 1
        zt3_end = zt3_start + num_arcs - 1
        Ztilde1_3_ray = Ztilde1_ray[s, zt3_start:zt3_end, :]
        t4 = sum(Ztilde1_3_ray .* (diag_λ_ψ * diagm(xi_bar_val[s])))

        βtilde1_1_ray = βtilde1_ray[s, 1:num_arcs+1]
        t5 = λ_sol * d0' * βtilde1_1_ray

        bt3_start = num_arcs+1 + (num_nodes-1) + 1
        bt3_end = bt3_start + num_arcs - 1
        βtilde1_3_ray = βtilde1_ray[s, bt3_start:bt3_end]
        t6 = -(h_sol + diag_λ_ψ * xi_bar_val[s])' * βtilde1_3_ray

        tp_ub_hat = -ϕU_large * sum(ray_vars["Phat1_Φ"][s,:,:]) - ϕU_large * sum(ray_vars["Phat1_Π"][s,:,:])
        tp_lb_hat = -ϕU_large * sum(ray_vars["Phat2_Φ"][s,:,:]) - ϕU_large * sum(ray_vars["Phat2_Π"][s,:,:])
        tp_ub_tilde = -ϕU_large * sum(ray_vars["Ptilde1_Φ"][s,:,:]) - ϕU_large * sum(ray_vars["Ptilde1_Π"][s,:,:]) - ϕU_large * sum(ray_vars["Ptilde1_Y"][s,:,:]) - ϕU_large * sum(ray_vars["Ptilde1_Yts"][s,:])
        tp_lb_tilde = -ϕU_large * sum(ray_vars["Ptilde2_Φ"][s,:,:]) - ϕU_large * sum(ray_vars["Ptilde2_Π"][s,:,:]) - ϕU_large * sum(ray_vars["Ptilde2_Y"][s,:,:]) - ϕU_large * sum(ray_vars["Ptilde2_Yts"][s,:])

        total = t1 + t2 + t3 + t4 + t5 + t6 + tp_ub_hat + tp_lb_hat + tp_ub_tilde + tp_lb_tilde

        println("  Scenario $s: TOTAL = $(round(total, digits=8))")
        println("    β terms (unchanged): $(round(t3+t4+t5+t6, digits=8))")
        println("    U+P penalty (×ϕU): $(round(t1+t2+tp_ub_hat+tp_lb_hat+tp_ub_tilde+tp_lb_tilde, digits=8))")
    end

    # ===== Find critical ϕU =====
    for s in 1:S
        βhat1_1_ray = βhat1_ray[s, 1:num_arcs+1]
        t3 = d0' * βhat1_1_ray

        num_nodes = length(network.nodes)
        zt3_start = num_arcs+1 + (num_nodes-1) + 1
        zt3_end = zt3_start + num_arcs - 1
        Ztilde1_3_ray = Ztilde1_ray[s, zt3_start:zt3_end, :]
        t4 = sum(Ztilde1_3_ray .* (diag_λ_ψ * diagm(xi_bar_val[s])))

        βtilde1_1_ray = βtilde1_ray[s, 1:num_arcs+1]
        t5 = λ_sol * d0' * βtilde1_1_ray

        bt3_start = num_arcs+1 + (num_nodes-1) + 1
        bt3_end = bt3_start + num_arcs - 1
        βtilde1_3_ray = βtilde1_ray[s, bt3_start:bt3_end]
        t6 = -(h_sol + diag_λ_ψ * xi_bar_val[s])' * βtilde1_3_ray

        β_contribution = t3 + t4 + t5 + t6  # ϕU-independent terms

        # U+P terms with ϕU factored out
        U_sum = sum((Uhat1_ray[s,:,:] + Utilde1_ray[s,:,:]) .* diag_x_E) +
                sum((Uhat3_ray[s,:,:] + Utilde3_ray[s,:,:]) .* (E - diag_x_E))
        P_sum = sum(ray_vars["Phat1_Φ"][s,:,:]) + sum(ray_vars["Phat1_Π"][s,:,:]) +
                sum(ray_vars["Phat2_Φ"][s,:,:]) + sum(ray_vars["Phat2_Π"][s,:,:]) +
                sum(ray_vars["Ptilde1_Φ"][s,:,:]) + sum(ray_vars["Ptilde1_Π"][s,:,:]) +
                sum(ray_vars["Ptilde2_Φ"][s,:,:]) + sum(ray_vars["Ptilde2_Π"][s,:,:]) +
                sum(ray_vars["Ptilde1_Y"][s,:,:]) + sum(ray_vars["Ptilde1_Yts"][s,:]) +
                sum(ray_vars["Ptilde2_Y"][s,:,:]) + sum(ray_vars["Ptilde2_Yts"][s,:])

        UP_coefficient = U_sum + P_sum  # total multiplied by -ϕU

        if UP_coefficient > 1e-10
            critical_ϕU = β_contribution / UP_coefficient
            println("\n  Critical ϕU* (s=$s) = $(round(critical_ϕU, digits=6))")
            println("    β_contribution = $(round(β_contribution, digits=8))")
            println("    U+P coefficient = $(round(UP_coefficient, digits=8))")
            println("    For ϕU < ϕU*, the ray has positive objective → UNBOUNDED")
        else
            println("\n  U+P coefficient ≤ 0 (s=$s), problem is structural")
        end
    end

    # ===== N_ts structure =====
    println("\n\n" * "="^80)
    println("NETWORK STRUCTURE: N_ts (dummy arc column)")
    println("="^80)
    N = network.N
    N_ts = N[:, end]
    num_nodes_val = length(network.nodes)
    println("  N_ts nonzero entries:")
    for i in 1:(num_nodes_val-1)
        if abs(N_ts[i]) > 1e-10
            println("    Node $(network.nodes[i+1]) (row $i): N_ts = $(N_ts[i])")
        end
    end
    println("  Total nonzero: $(sum(abs.(N_ts) .> 1e-10))")
    println("  Sum of positive N_ts: $(sum(max.(N_ts, 0)))")
    println("  Sum of negative N_ts: $(sum(min.(N_ts, 0)))")

    # ===== βhat1_1 block detailed analysis =====
    println("\n\n" * "="^80)
    println("βhat1_1 RAY DETAIL (the block with +d0 objective coefficient)")
    println("="^80)
    for s in 1:S
        β1_1 = βhat1_ray[s, 1:num_arcs+1]
        println("  s=$s: βhat1_1[end] = $(β1_1[end])")
        nonzero_β = findall(abs.(β1_1) .> 1e-8)
        println("  s=$s: nonzero indices: $nonzero_β")
        for idx in nonzero_β
            println("    βhat1_1[$idx] = $(round(β1_1[idx], digits=8))")
        end
    end

    println("\n\n" * "="^80)
    println("βtilde1_1 RAY DETAIL (the block with +λ×d0 objective coefficient)")
    println("="^80)
    for s in 1:S
        β1_1 = βtilde1_ray[s, 1:num_arcs+1]
        println("  s=$s: βtilde1_1[end] = $(β1_1[end])")
        nonzero_β = findall(abs.(β1_1) .> 1e-8)
        println("  s=$s: nonzero indices: $nonzero_β")
        for idx in nonzero_β
            println("    βtilde1_1[$idx] = $(round(β1_1[idx], digits=8))")
        end
    end

    # ===== Check Ztilde1_3 contribution =====
    println("\n\n" * "="^80)
    println("Ztilde1_3 RAY DETAIL (obj_term4: ϕU-independent, data-dependent)")
    println("="^80)
    for s in 1:S
        num_nodes_s = length(network.nodes)
        zt3_s = num_arcs+1 + (num_nodes_s-1) + 1
        zt3_e = zt3_s + num_arcs - 1
        Zt3 = Ztilde1_ray[s, zt3_s:zt3_e, :]
        max_zt3 = maximum(abs.(Zt3))
        println("  s=$s: max|Ztilde1_3| = $(round(max_zt3, digits=8))")
        if max_zt3 > 1e-8
            # Find largest entries
            for i in 1:num_arcs, j in 1:size(Zt3, 2)
                if abs(Zt3[i,j]) > max_zt3 * 0.1
                    println("    Zt3[$i,$j] = $(round(Zt3[i,j], digits=8))")
                end
            end
        end
    end

    println("\n" * "="^80)
    println("Use @infiltrate below to interactively inspect ray_vars")
    println("="^80)
    @infiltrate
else
    println("\nNo infeasibility certificate available.")
    if t_status == MOI.OPTIMAL
        println("  OSP solved optimally! obj = $(objective_value(osp_model))")
        println("  (Try with a different initial solution or parameters)")
    end
end
