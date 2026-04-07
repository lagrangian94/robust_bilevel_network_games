"""
Diagnostic v2: Run Benders loop until DUAL_INFEASIBLE, then extract ray.
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using LinearAlgebra
using Infiltrator
using Revise

includet("../network_generator.jl")
includet("../build_uncertainty_set.jl")
includet("test_build_dualized_outer_subprob.jl")
includet("test_strict_benders.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

S = 1; λU = 10.0; γ_ratio = 0.10; ρ = 0.2; v = 1.0; seed = 42; epsilon = 0.5
ϕU = 1/epsilon  # = 2.0

network = generate_grid_network(4, 4, seed=seed)
num_arcs = length(network.arcs) - 1
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ_computed = ceil(Int, γ_ratio * num_interdictable)

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w_computed = ρ * γ_computed * c_bar

capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

γ = 2.0; w = 1.0

println("ϕU=$ϕU, γ=$γ, w=$w, num_arcs=$num_arcs")

# Build OMP
model_omp, vars_omp = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=false)
x_var = vars_omp[:x]; h_var = vars_omp[:h]; λ_var = vars_omp[:λ]; ψ0_var = vars_omp[:ψ0]; t_0 = vars_omp[:t_0]

E_mat = ones(num_arcs, num_arcs+1)
d0_vec = zeros(num_arcs+1); d0_vec[end] = 1.0
I_0 = [Matrix{Float64}(I, num_arcs, num_arcs) zeros(num_arcs)]

max_iter = 50
global osp_model = nothing
global osp_vars = nothing
global osp_data = nothing
global x_sol = nothing
global h_sol = nothing
global λ_sol = nothing
global ψ0_sol = nothing

for iter in 1:max_iter
    optimize!(model_omp)
    omp_st = termination_status(model_omp)
    if omp_st != MOI.OPTIMAL && omp_st != MOI.DUAL_INFEASIBLE
        println("Iter $iter: OMP status = $omp_st, stopping")
        break
    end

    global x_sol = value.(x_var); global h_sol = value.(h_var)
    global λ_sol = value(λ_var); global ψ0_sol = value.(ψ0_var)
    t0_sol = value(t_0)
    println("\nIter $iter: OMP obj=$t0_sol, λ=$λ_sol, x_nonzero=$(findall(x_sol .> 0.5))")

    # Build fresh OSP each iteration (parameters change)
    global osp_model, osp_vars, osp_data = build_dualized_outer_subproblem(
        network, S, ϕU, λU, γ, w, v, uncertainty_set,
        Mosek.Optimizer, λ_sol, x_sol, h_sol, ψ0_sol)
    set_silent(osp_model)
    optimize!(osp_model)

    t_status = termination_status(osp_model)
    p_status = primal_status(osp_model)
    println("  OSP: termination=$t_status, primal=$p_status")

    if t_status == MOI.DUAL_INFEASIBLE
        println("\n" * "="^80)
        println("DUAL_INFEASIBLE at iteration $iter!")
        println("  x = $x_sol")
        println("  λ = $λ_sol, h = $h_sol")
        println("  ψ0 = $ψ0_sol")
        println("="^80)

        if p_status == MOI.INFEASIBILITY_CERTIFICATE
            # ===== EXTRACT RAY =====
            ray = Dict{String, Any}()
            for vname in [:α, :Mhat, :Mtilde,
                          :Uhat1, :Uhat2, :Uhat3, :Utilde1, :Utilde2, :Utilde3,
                          :βhat1, :βhat2, :βtilde1, :βtilde2,
                          :Zhat1, :Zhat2, :Ztilde1, :Ztilde2,
                          :Γhat1, :Γhat2, :Γtilde1, :Γtilde2,
                          :Phat1_Φ, :Phat2_Φ, :Phat1_Π, :Phat2_Π,
                          :Ptilde1_Φ, :Ptilde2_Φ, :Ptilde1_Π, :Ptilde2_Π,
                          :Ptilde1_Y, :Ptilde2_Y, :Ptilde1_Yts, :Ptilde2_Yts]
                ray[string(vname)] = value.(osp_model[vname])
            end

            # Ray magnitudes
            println("\nRay magnitude by variable (top 15):")
            sorted = sort(collect(ray), by=x->maximum(abs.(x.second)), rev=true)
            for (i, (name, vals)) in enumerate(sorted)
                max_v = maximum(abs.(vals))
                if max_v > 1e-10 && i <= 15
                    println("  $(rpad(name,20)) max|ray|=$(round(max_v, digits=8))  sum|ray|=$(round(sum(abs.(vals)), digits=6))")
                end
            end

            # Objective decomposition
            diag_x_E = Diagonal(x_sol) * E_mat
            diag_λ_ψ = Diagonal(λ_sol*ones(num_arcs) - v.*ψ0_sol)
            xi_bar_val = uncertainty_set[:xi_bar]
            num_nodes = length(network.nodes)

            for s in 1:S
                println("\nObjective ray decomposition (s=$s):")
                U1r = ray["Uhat1"]; U3r = ray["Uhat3"]; Ut1r = ray["Utilde1"]; Ut3r = ray["Utilde3"]
                t1 = -ϕU * sum((U1r[s,:,:] + Ut1r[s,:,:]) .* diag_x_E)
                t2 = -ϕU * sum((U3r[s,:,:] + Ut3r[s,:,:]) .* (E_mat - diag_x_E))
                t3 = d0_vec' * ray["βhat1"][s, 1:num_arcs+1]

                zt3_s = num_arcs+1 + (num_nodes-1) + 1
                zt3_e = zt3_s + num_arcs - 1
                t4 = sum(ray["Ztilde1"][s, zt3_s:zt3_e, :] .* (diag_λ_ψ * diagm(xi_bar_val[s])))
                t5 = λ_sol * d0_vec' * ray["βtilde1"][s, 1:num_arcs+1]

                bt3_s = num_arcs+1 + (num_nodes-1) + 1; bt3_e = bt3_s + num_arcs - 1
                t6 = -(h_sol + diag_λ_ψ * xi_bar_val[s])' * ray["βtilde1"][s, bt3_s:bt3_e]

                tp1 = -ϕU * (sum(ray["Phat1_Φ"][s,:,:]) + sum(ray["Phat1_Π"][s,:,:]))
                tp2 = -ϕU * (sum(ray["Phat2_Φ"][s,:,:]) + sum(ray["Phat2_Π"][s,:,:]))
                tp3 = -ϕU * (sum(ray["Ptilde1_Φ"][s,:,:]) + sum(ray["Ptilde1_Π"][s,:,:]) + sum(ray["Ptilde1_Y"][s,:,:]) + sum(ray["Ptilde1_Yts"][s,:]))
                tp4 = -ϕU * (sum(ray["Ptilde2_Φ"][s,:,:]) + sum(ray["Ptilde2_Π"][s,:,:]) + sum(ray["Ptilde2_Y"][s,:,:]) + sum(ray["Ptilde2_Yts"][s,:]))

                terms = [
                    ("U1+Ut1 (×-ϕU×x)", t1), ("U3+Ut3 (×-ϕU×(1-x))", t2),
                    ("βhat1_1·d0", t3), ("Zt3·(λ-ψ)·ξ", t4),
                    ("βt1_1·λ·d0", t5), ("βt1_3·(-(h+...))", t6),
                    ("P_ub_hat", tp1), ("P_lb_hat", tp2),
                    ("P_ub_tilde", tp3), ("P_lb_tilde", tp4),
                ]

                total = sum(x[2] for x in terms)
                for (label, val) in terms
                    abs(val) > 1e-10 && println("  $(rpad(label,28)) $(round(val, digits=8))")
                end
                println("  $(rpad("TOTAL",28)) $(round(total, digits=8))")

                # Critical ϕU
                β_terms = t3 + t4 + t5 + t6
                UP_raw = (sum((U1r[s,:,:] + Ut1r[s,:,:]) .* diag_x_E) +
                          sum((U3r[s,:,:] + Ut3r[s,:,:]) .* (E_mat - diag_x_E)) +
                          sum(ray["Phat1_Φ"][s,:,:]) + sum(ray["Phat1_Π"][s,:,:]) +
                          sum(ray["Phat2_Φ"][s,:,:]) + sum(ray["Phat2_Π"][s,:,:]) +
                          sum(ray["Ptilde1_Φ"][s,:,:]) + sum(ray["Ptilde1_Π"][s,:,:]) +
                          sum(ray["Ptilde2_Φ"][s,:,:]) + sum(ray["Ptilde2_Π"][s,:,:]) +
                          sum(ray["Ptilde1_Y"][s,:,:]) + sum(ray["Ptilde1_Yts"][s,:]) +
                          sum(ray["Ptilde2_Y"][s,:,:]) + sum(ray["Ptilde2_Yts"][s,:]))
                if UP_raw > 1e-10
                    crit = β_terms / UP_raw
                    println("\n  Critical ϕU* = $(round(crit, digits=6))")
                    println("  β_contribution = $(round(β_terms, digits=8))")
                    println("  U+P coefficient = $(round(UP_raw, digits=8))")
                end
            end

            # N_ts structure
            N_ts = network.N[:, end]
            println("\nN_ts (dummy arc column):")
            for i in 1:(num_nodes-1)
                abs(N_ts[i]) > 1e-10 && println("  $(network.nodes[i+1]) (row $i): $(N_ts[i])")
            end

            # β detail
            println("\nβhat1_1 ray (d0 block):")
            for s in 1:S
                b = ray["βhat1"][s, 1:num_arcs+1]
                nz = findall(abs.(b) .> 1e-8)
                for idx in nz; println("  βhat1_1[$idx] = $(round(b[idx], digits=8))"); end
            end
            println("\nβtilde1_1 ray (λ·d0 block):")
            for s in 1:S
                b = ray["βtilde1"][s, 1:num_arcs+1]
                nz = findall(abs.(b) .> 1e-8)
                for idx in nz; println("  βtilde1_1[$idx] = $(round(b[idx], digits=8))"); end
            end

            @infiltrate
        else
            println("  No infeasibility certificate available (primal=$p_status)")
        end
        break
    end

    # OSP optimal — add cut to OMP
    obj_val = objective_value(osp_model)
    println("  OSP obj = $(round(obj_val, digits=6))")

    diag_x_E = Diagonal(x_var) * E_mat
    diag_λ_ψ = Diagonal(λ_var*ones(num_arcs) - v.*ψ0_var)
    xi_bar_val = uncertainty_set[:xi_bar]

    Uhat1_v = value.(osp_vars[:Uhat1]); Utilde1_v = value.(osp_vars[:Utilde1])
    Uhat3_v = value.(osp_vars[:Uhat3]); Utilde3_v = value.(osp_vars[:Utilde3])
    βtilde1_1_v = value.(osp_vars[:βtilde1_1]); βtilde1_3_v = value.(osp_vars[:βtilde1_3])
    Ztilde1_3_v = value.(osp_vars[:Ztilde1_3])
    βhat1_1_v = value.(osp_vars[:βhat1_1])

    # intercept (x-independent parts)
    Phat1_Φ_v = value.(osp_vars[:Phat1_Φ]); Phat1_Π_v = value.(osp_vars[:Phat1_Π])
    Phat2_Φ_v = value.(osp_vars[:Phat2_Φ]); Phat2_Π_v = value.(osp_vars[:Phat2_Π])
    Ptilde1_Φ_v = value.(osp_vars[:Ptilde1_Φ]); Ptilde1_Π_v = value.(osp_vars[:Ptilde1_Π])
    Ptilde2_Φ_v = value.(osp_vars[:Ptilde2_Φ]); Ptilde2_Π_v = value.(osp_vars[:Ptilde2_Π])
    Ptilde1_Y_v = value.(osp_vars[:Ptilde1_Y]); Ptilde2_Y_v = value.(osp_vars[:Ptilde2_Y])
    Ptilde1_Yts_v = value.(osp_vars[:Ptilde1_Yts]); Ptilde2_Yts_v = value.(osp_vars[:Ptilde2_Yts])

    intercept = [(d0_vec' * βhat1_1_v[s,:]) for s in 1:S]
    intercept .+= [-ϕU*sum(Phat1_Φ_v[s,:,:]) - ϕU*sum(Phat1_Π_v[s,:,:]) for s in 1:S]
    intercept .+= [-ϕU*sum(Phat2_Φ_v[s,:,:]) - ϕU*sum(Phat2_Π_v[s,:,:]) for s in 1:S]
    intercept .+= [-ϕU*sum(Ptilde1_Φ_v[s,:,:]) - ϕU*sum(Ptilde1_Π_v[s,:,:]) - ϕU*sum(Ptilde1_Y_v[s,:,:]) - ϕU*sum(Ptilde1_Yts_v[s,:]) for s in 1:S]
    intercept .+= [-ϕU*sum(Ptilde2_Φ_v[s,:,:]) - ϕU*sum(Ptilde2_Π_v[s,:,:]) - ϕU*sum(Ptilde2_Y_v[s,:,:]) - ϕU*sum(Ptilde2_Yts_v[s,:]) for s in 1:S]

    cut_1 = -ϕU * [sum((Uhat1_v[s,:,:] + Utilde1_v[s,:,:]) .* diag_x_E) for s in 1:S]
    cut_2 = -ϕU * [sum((Uhat3_v[s,:,:] + Utilde3_v[s,:,:]) .* (E_mat - diag_x_E)) for s in 1:S]
    cut_3 = [sum(Ztilde1_3_v[s,:,:] .* (diag_λ_ψ * diagm(xi_bar_val[s]))) for s in 1:S]
    cut_4 = [(d0_vec' * βtilde1_1_v[s,:]) * λ_var for s in 1:S]
    cut_5 = -1 * [(h_var + diag_λ_ψ * xi_bar_val[s])' * βtilde1_3_v[s,:] for s in 1:S]

    opt_cut = sum(cut_1) + sum(cut_2) + sum(cut_3) + sum(cut_4) + sum(cut_5) + sum(intercept)
    @constraint(model_omp, t_0 >= opt_cut)
    println("  Cut added. Cut value at current sol = $(round(obj_val, digits=6))")
end
