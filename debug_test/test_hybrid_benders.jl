"""
test_hybrid_benders.jl — Compare hybrid nested Benders (primal ISP inner loop + dual ISP outer cuts)
against the original tr_nested_benders_optimize! (dual ISP only).

Test plan:
1. 3×3 grid, S=1
2. Run original tr_nested_benders_optimize! → save result
3. Run hybrid version (same setup) → save result
4. Compare: final objective, iteration counts, convergence pattern
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using Hypatia
using LinearAlgebra
using Infiltrator
using Revise

includet("../network_generator.jl")
includet("../build_uncertainty_set.jl")
includet("../strict_benders.jl")
includet("../nested_benders_trust_region.jl")
includet("../build_primal_isp.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

println("="^80)
println("HYBRID NESTED BENDERS TEST")
println("="^80)

# ===== Parameters =====
# 옛날 파라미터.
# γ = 2.0
# w = 1.0
# v = 1.0

S = 10
λU = 10.0
γ_ratio = 0.10  # Interdiction budget as fraction of interdictable arcs: γ = ceil(γ_ratio * |A_I|)
                 # Sensitivity: γ_ratio ∈ {0.03, 0.05, 0.10}
ρ = 0.2  # Recovery power ratio: w = ρ·γ·c̄, follower's max recovery = ρ × expected interdiction damage
         # Sensitivity: ρ ∈ {0.05, 0.1, 0.2, 0.3}
v = 1.0
seed = 42
epsilon = 0.5
ϕU = 1/epsilon # valid upper bound?
# ===== Generate Network & Uncertainty Set =====
println("\n[1] Generating 5×5 grid network...")
network = generate_grid_network(4, 4, seed=seed)
print_network_summary(network)

# Compute γ from network size
num_arcs = length(network.arcs) - 1
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, γ_ratio * num_interdictable)
println("  Interdiction budget: γ = ceil($γ_ratio × $num_interdictable) = $γ")

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)

# Compute w = ρ · γ · c̄ (mean capacity of interdictable arcs)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = ρ * γ * c_bar
println("  Recovery budget: w = ρ·γ·c̄ = $ρ × $γ × $(round(c_bar, digits=2)) = $(round(w, digits=4))")


# γ = 2.0
# w = 1.0

capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

# =====================================================================
# TEST 1: Run original tr_nested_benders_optimize!
# =====================================================================
println("\n" * "="^80)
println("TEST 1: Original tr_nested_benders_optimize! (dual ISP only)")
println("="^80)

omp_model_orig, omp_vars_orig = build_omp(network, ϕU, λU, γ, w;
    optimizer=Gurobi.Optimizer, multi_cut=true)

result_orig = tr_nested_benders_optimize!(omp_model_orig, omp_vars_orig, network,
    ϕU, λU, γ, w, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    multi_cut=true, outer_tr=true, inner_tr=true)

println("\nOriginal result:")
println("  Solution time: $(result_orig[:solution_time]) s")
println("  Inner iterations per outer: $(result_orig[:inner_iter])")
if haskey(result_orig, :opt_sol)
    println("  x* = $(result_orig[:opt_sol][:x])")
    println("  λ* = $(result_orig[:opt_sol][:λ])")
    println("  h* = $(result_orig[:opt_sol][:h]/result_orig[:opt_sol][:λ]) (recovered)")
end

# =====================================================================
# TEST 2: Run hybrid nested Benders
# =====================================================================
println("\n" * "="^80)
println("TEST 2: Hybrid nested Benders (primal ISP inner + dual ISP outer cuts)")
println("="^80)

omp_model_hyb, omp_vars_hyb = build_omp(network, ϕU, λU, γ, w;
    optimizer=Gurobi.Optimizer, multi_cut=true)

# We reuse the outer loop structure from tr_nested_benders_optimize!
# but replace tr_imp_optimize! with tr_imp_optimize_hybrid!
# and evaluate_master_opt_cut with primal_evaluate_master_opt_cut

function tr_nested_benders_optimize_hybrid!(omp_model::Model, omp_vars::Dict, network, ϕU, λU, γ, w, uncertainty_set;
    mip_optimizer=nothing, conic_optimizer=nothing, multi_cut=false, outer_tr=true, inner_tr=true, max_outer_iter=1000, full_primal=false)

    # full_primal=true is NOT recommended.
    # Outer cut extraction via Mosek IPM shadow prices (evaluate_master_opt_cut_from_primal)
    # produces inaccurate cut coefficients due to conic dual degeneracy.
    # Unlike the inner cut μ offset (uniform +ε, correctable), outer cut shadow prices
    # differ non-uniformly (30-40%) from dual ISP variable values, making simple correction impossible.
    # This leads to invalid outer cuts → OMP selects extreme (x,h,λ,ψ0) → primal ISP infeasible.
    # Use full_primal=false (hybrid mode: primal ISP inner + dual ISP outer) instead.
    # See memory/ipm_mu_offset.md and debug_test/test_outer_cut_compare.jl for details.
    if full_primal
        error(
            "full_primal=true is disabled: outer cut extraction from primal ISP shadow prices " *
            "is unreliable due to IPM conic dual degeneracy (non-uniform 30-40% coefficient errors). " *
            "Use full_primal=false (hybrid mode: primal ISP inner loop + dual ISP outer cuts) instead."
        )
    end

    ### -------- Trust Region 초기화 --------
    if outer_tr
        B_bin_sequence = [0.05, 0.5, 1.0]
        B_bin_stage = 1
        B_bin = B_bin_sequence[B_bin_stage] * sum(network.interdictable_arcs)
        B_con = nothing
        centers = Dict{Symbol, Any}(
            :x => nothing, :h => nothing, :λ => nothing, :ψ0 => nothing
        )
        β_relative = 1e-4
        tr_constraints = Dict{Symbol, Any}(:binary => nothing, :continuous => nothing)
    end
    upper_bound = Inf

    ### --------OMP Initialization--------
    st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars)
    x, h, λ, ψ0 = omp_vars[:x], omp_vars[:h], omp_vars[:λ], omp_vars[:ψ0]
    if outer_tr
        centers[:x] = value.(x)
        centers[:h] = value.(h)
        centers[:λ] = value.(λ)
        centers[:ψ0] = value.(ψ0)
    end
    if multi_cut
        t_0_l = omp_vars[:t_0_l]
        t_0_f = omp_vars[:t_0_f]
        t_0 = t_0_l + t_0_f
    else
        t_0 = omp_vars[:t_0]
    end

    num_arcs = length(network.arcs) - 1
    E = ones(num_arcs, num_arcs+1)
    diag_x_E = Diagonal(x) * E
    diag_λ_ψ = Diagonal(λ*ones(num_arcs)-v.*ψ0)
    d0 = zeros(num_arcs + 1)
    d0[end] = 1.0
    xi_bar_local = uncertainty_set[:xi_bar]
    iter = 0
    past_obj = []
    past_major_subprob_obj = []
    past_minor_subprob_obj = []
    past_model_estimate = []
    past_local_lower_bound = []
    past_upper_bound = []
    past_lower_bound = []
    past_local_optimizer = []
    major_iter = []
    bin_B_steps = []
    imp_cuts = Dict{Symbol, Any}(:old_tr_constraints => nothing)
    result = Dict()
    result[:cuts] = Dict()
    result[:tr_info] = Dict()
    result[:inner_iter] = []
    upper_bound = Inf
    lower_bound = -Inf

    ### --------IMP + ISP Initialization--------
    imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_optimizer=mip_optimizer)
    st, α_sol = initialize_imp(imp_model, imp_vars)

    # Dual ISP instances (for outer cut generation — hybrid only)
    dual_leader_instances, dual_follower_instances = nothing, nothing
    if !full_primal
        dual_leader_instances, dual_follower_instances = initialize_isp(
            network, S, ϕU, λU, γ, w, v, uncertainty_set;
            conic_optimizer=conic_optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
    end

    # Primal ISP instances (for inner loop + full primal outer cuts)
    primal_leader_instances, primal_follower_instances = initialize_primal_isp(
        network, S, ϕU, λU, γ, w, v, uncertainty_set;
        conic_optimizer=conic_optimizer, x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)

    isp_data = Dict(:E => E, :network => network, :ϕU => ϕU, :λU => λU, :γ => γ, :w => w, :v => v, :uncertainty_set => uncertainty_set, :d0 => d0, :S=>S)
    gap = Inf

    ### --------End Initialization--------
    time_start = time()
    while (st == MOI.DUAL_INFEASIBLE || st == MOI.OPTIMAL)
        iter += 1
        if iter > max_outer_iter
            @warn "Maximum outer iterations ($max_outer_iter) reached. Gap = $gap"
            break
        end
        if outer_tr
            @info "[Outer-$(full_primal ? "FullPrimal" : "Hybrid")] Iteration $iter (B_bin=$B_bin, Stage=$(B_bin_stage+1)/$(length(B_bin_sequence)))"
        else
            @info "[Outer-$(full_primal ? "FullPrimal" : "Hybrid")] Iteration $iter"
        end

        # Outer Master Problem
        optimize!(omp_model)
        st = MOI.get(omp_model, MOI.TerminationStatus())
        if st == MOI.INFEASIBLE
            @info " Outer Master Problem infeasible (Converged): No search space left"
            gap = 0.0
        else
            x_sol, h_sol, λ_sol, ψ0_sol = value.(omp_vars[:x]), value.(omp_vars[:h]), value(omp_vars[:λ]), value.(omp_vars[:ψ0])
            model_estimate = value(t_0)
            lower_bound = max(lower_bound, model_estimate)

            # Update primal ISP parameters (x,h,λ,ψ0 in constraint RHS → set_normalized_rhs)
            update_primal_isp_parameters!(primal_leader_instances, primal_follower_instances;
                x_sol=x_sol, h_sol=h_sol, λ_sol=λ_sol, ψ0_sol=ψ0_sol, isp_data=isp_data)

            # Hybrid inner loop (primal ISP)
            status, cut_info = tr_imp_optimize_hybrid!(imp_model, imp_vars,
                primal_leader_instances, primal_follower_instances;
                isp_data=isp_data, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                outer_iter=iter, imp_cuts=imp_cuts, inner_tr=inner_tr)

            if status != :OptimalityCut
                @warn "Outer Subproblem not optimal (hybrid)"
                @infiltrate
            end
            push!(result[:inner_iter], cut_info[:iter])
            imp_cuts[:old_cuts] = cut_info[:cuts]
            if inner_tr && cut_info[:tr_constraints] !== nothing
                imp_cuts[:old_tr_constraints] = cut_info[:tr_constraints]
            end
            subprob_obj = cut_info[:obj_val]
            upper_bound = min(upper_bound, subprob_obj)

            gap = upper_bound - lower_bound
            if outer_tr
                if iter==1
                    push!(past_major_subprob_obj, subprob_obj)
                end
                tr_needs_update = false
                predicted_decrease = past_major_subprob_obj[end] - model_estimate
                β_dynamic = max(1e-8, β_relative * predicted_decrease)
                improvement = past_major_subprob_obj[end] - subprob_obj
                is_serious_step = (improvement >= β_dynamic)
                if is_serious_step
                    centers[:x] = value.(x_sol)
                    centers[:h] = value.(h_sol)
                    centers[:λ] = value.(λ_sol)
                    centers[:ψ0] = value.(ψ0_sol)
                    push!(major_iter, iter)
                    push!(past_major_subprob_obj, subprob_obj)
                    tr_needs_update = true
                end
            end
            @info "[Outer-$(full_primal ? "FullPrimal" : "Hybrid")] Iter $iter: LB=$(round(lower_bound, digits=4))  UB=$(round(upper_bound, digits=4))  gap=$(round(gap, digits=6))"
            push!(past_lower_bound, lower_bound)
            push!(past_model_estimate, model_estimate)
            push!(past_minor_subprob_obj, subprob_obj)
            push!(past_upper_bound, upper_bound)
        end
        if gap <= 1e-4
            if !outer_tr
                time_end = time()
                result[:solution_time] = time_end - time_start
                @info "  ✓ OPTIMAL (no outer TR, hybrid). Gap = $gap"
                result[:past_lower_bound] = past_lower_bound
                result[:past_minor_subprob_obj] = past_minor_subprob_obj
                result[:past_upper_bound] = past_upper_bound
                result[:opt_sol] = Dict(:x=>x_sol, :h=>h_sol, :λ=>λ_sol, :ψ0=>ψ0_sol)
                return result
            end
            if B_bin_stage <= length(B_bin_sequence)-1
                B_bin_stage +=1
                B_bin_old = B_bin
                B_bin = B_bin_sequence[B_bin_stage] * sum(network.interdictable_arcs)
                push!(bin_B_steps, iter)
                push!(past_local_lower_bound, lower_bound)
                push!(past_local_optimizer, Dict(:x=>value.(x_sol), :h=>value.(h_sol), :λ=>value.(λ_sol), :ψ0=>value.(ψ0_sol)))
                @info "  ✓ Local optimal reached! Expanding B_bin to $B_bin"
                tr_needs_update = true
                @info "Updating Trust Region"
                tr_constraints = update_outer_trust_region_constraints!(
                    omp_model, omp_vars, centers, B_bin, B_con, tr_constraints, network)
                lower_bound = -Inf
                _ = add_reverse_region_constraint!(omp_model, omp_vars[:x], centers[:x], B_bin_old, network)
            else
                time_end = time()
                result[:solution_time] = time_end - time_start
                @info "  ✓✓ GLOBAL OPTIMAL (hybrid)! (B_bin = full region)"
                min_idx = argmin(past_local_lower_bound)
                global_lower_bound = past_local_lower_bound[min_idx]
                iter_when_global_optimal = bin_B_steps[min_idx]
                global_upper_bound = past_upper_bound[iter_when_global_optimal]
                println("lower_bound: ", global_lower_bound, ", upper_bound: ", global_upper_bound)

                result[:past_lower_bound] = past_lower_bound
                result[:past_local_lower_bound] = past_local_lower_bound
                result[:past_minor_subprob_obj] = past_minor_subprob_obj
                result[:past_major_subprob_obj] = past_major_subprob_obj
                result[:past_upper_bound] = past_upper_bound
                result[:tr_info][:final_B_bin_stage] = B_bin_stage
                result[:tr_info][:final_B_bin] = B_bin
                result[:tr_info][:major_iter] = major_iter
                result[:tr_info][:bin_B_steps] = bin_B_steps
                result[:opt_sol] = past_local_optimizer[min_idx]
                result[:iter_when_global_optimal] = iter_when_global_optimal
                return result
            end
        else
            # Gap still large → Generate outer cut
            if full_primal
                # Full primal: extract cut coefficients from primal ISP shadow prices
                outer_cut_info = evaluate_master_opt_cut_from_primal(
                    primal_leader_instances, primal_follower_instances,
                    isp_data, cut_info, iter;
                    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                    multi_cut=multi_cut)
            else
                # Hybrid: update dual ISP objectives, then extract cut coefficients
                outer_cut_info = primal_evaluate_master_opt_cut(
                    dual_leader_instances, dual_follower_instances,
                    isp_data, cut_info, iter;
                    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                    multi_cut=multi_cut)
            end

            if multi_cut
                cut_1_l =  -ϕU * [sum(outer_cut_info[:Uhat1][s,:,:] .* diag_x_E) for s in 1:S]
                cut_1_f =  -ϕU * [sum(outer_cut_info[:Utilde1][s,:,:] .* diag_x_E) for s in 1:S]
                cut_2_l =  -ϕU * [sum(outer_cut_info[:Uhat3][s,:,:] .* (isp_data[:E] - diag_x_E)) for s in 1:S]
                cut_2_f =  -ϕU * [sum(outer_cut_info[:Utilde3][s,:,:] .* (isp_data[:E] - diag_x_E)) for s in 1:S]
                cut_3_f =  [sum(outer_cut_info[:Ztilde1_3][s,:,:] .* (diag_λ_ψ * diagm(xi_bar_local[s]))) for s in 1:S]
                cut_4_f =  [(isp_data[:d0]'*outer_cut_info[:βtilde1_1][s,:]) * λ for s in 1:S]
                cut_5_f =  -1* [(h + diag_λ_ψ * xi_bar_local[s])'* outer_cut_info[:βtilde1_3][s,:] for s in 1:S]
                cut_intercept_l = outer_cut_info[:intercept_l]
                cut_intercept_f = outer_cut_info[:intercept_f]
                opt_cut_l = sum(cut_1_l)+ sum(cut_2_l) + sum(cut_intercept_l)
                opt_cut_f = sum(cut_1_f)+ sum(cut_2_f)+ sum(cut_3_f)+ sum(cut_4_f)+ sum(cut_5_f) + sum(cut_intercept_f)
                cut_added_l = @constraint(omp_model, t_0_l >= opt_cut_l)
                cut_added_f = @constraint(omp_model, t_0_f >= opt_cut_f)
                set_name(cut_added_l, "opt_cut_$(iter)_l")
                set_name(cut_added_f, "opt_cut_$(iter)_f")
                result[:cuts]["opt_cut_$(iter)_l"] = cut_added_l
                result[:cuts]["opt_cut_$(iter)_f"] = cut_added_f
            else
                cut_1 =  -ϕU * [sum((outer_cut_info[:Uhat1][s,:,:] + outer_cut_info[:Utilde1][s,:,:]) .* diag_x_E) for s in 1:S]
                cut_2 =  -ϕU * [sum((outer_cut_info[:Uhat3][s,:,:] + outer_cut_info[:Utilde3][s,:,:]) .* (isp_data[:E] - diag_x_E)) for s in 1:S]
                cut_3 =  [sum(outer_cut_info[:Ztilde1_3][s,:,:] .* (diag_λ_ψ * diagm(xi_bar_local[s]))) for s in 1:S]
                cut_4 =  [(isp_data[:d0]'*outer_cut_info[:βtilde1_1][s,:]) * λ for s in 1:S]
                cut_5 =  -1* [(h + diag_λ_ψ * xi_bar_local[s])'* outer_cut_info[:βtilde1_3][s,:] for s in 1:S]
                cut_intercept = outer_cut_info[:intercept]
                opt_cut = sum(cut_1)+ sum(cut_2)+ sum(cut_3)+ sum(cut_4)+ sum(cut_5)+ sum(cut_intercept)
                cut_added = @constraint(omp_model, t_0 >= opt_cut)
                set_name(cut_added, "opt_cut_$iter")
                result[:cuts]["opt_cut_$iter"] = cut_added
            end

            y = Dict(
                [omp_vars[:x][k] => x_sol[k] for k in 1:num_arcs]...,
                [omp_vars[:h][k] => h_sol[k] for k in 1:num_arcs]...,
                omp_vars[:λ] => λ_sol,
                [omp_vars[:ψ0][k] => ψ0_sol[k] for k in 1:num_arcs]...
            )
            function evaluate_expr(expr::AffExpr, var_values::Dict)
                eval_result = expr.constant
                for (var, coef) in expr.terms
                    if haskey(var_values, var)
                        eval_result += coef * var_values[var]
                    else
                        error("Variable $var not found in var_values")
                    end
                end
                return eval_result
            end
            if multi_cut
                opt_cut = opt_cut_l + opt_cut_f
            end
            if abs(subprob_obj - evaluate_expr(opt_cut, y)) > 1e-3
                println("something went wrong (hybrid)")
                @infiltrate
            end
            println("subproblem objective (hybrid): ", subprob_obj)
            @info "Optimality cut added (hybrid)"

            if outer_tr && tr_needs_update
                @info "Updating Trust Region"
                tr_constraints = update_outer_trust_region_constraints!(
                    omp_model, omp_vars, centers, B_bin, B_con, tr_constraints, network)
            end
        end
    end
    # max_outer_iter reached or while condition became false
    result[:past_upper_bound] = past_upper_bound
    result[:past_lower_bound] = past_lower_bound
    result[:opt_sol] = Dict(:x=>x_sol, :h=>h_sol, :λ=>λ_sol, :ψ0=>ψ0_sol)
    return result
end

# Run hybrid
result_hyb = tr_nested_benders_optimize_hybrid!(omp_model_hyb, omp_vars_hyb, network,
    ϕU, λU, γ, w, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    multi_cut=true, outer_tr=true, inner_tr=true)

println("\nHybrid result:")
println("  Solution time: $(result_hyb[:solution_time]) s")
println("  Inner iterations per outer: $(result_hyb[:inner_iter])")
if haskey(result_hyb, :opt_sol)
    println("  x* = $(result_hyb[:opt_sol][:x])")
    println("  λ* = $(result_hyb[:opt_sol][:λ])")
    println("  h* = $(result_hyb[:opt_sol][:h]/result_hyb[:opt_sol][:λ]) (recovered)")
end

# =====================================================================
# COMPARISON: Original vs Hybrid
# =====================================================================
println("\n" * "="^80)
println("COMPARISON: Original vs Hybrid")
println("="^80)

# Compare optimal solutions
if haskey(result_orig, :opt_sol) && haskey(result_hyb, :opt_sol)
    x_gap = maximum(abs.(result_orig[:opt_sol][:x] - result_hyb[:opt_sol][:x]))
    λ_gap = abs(result_orig[:opt_sol][:λ] - result_hyb[:opt_sol][:λ])
    println("  x* gap (max): $x_gap  $(x_gap < 1e-3 ? "✓" : "✗")")
    println("  λ* gap: $λ_gap  $(λ_gap < 1e-3 ? "✓" : "✗")")
end

# Compare bounds
if haskey(result_orig, :past_local_lower_bound) && haskey(result_hyb, :past_local_lower_bound)
    orig_obj = minimum(result_orig[:past_local_lower_bound])
    hyb_obj = minimum(result_hyb[:past_local_lower_bound])
    obj_gap = abs(orig_obj - hyb_obj)
    println("  Optimal objective (orig): $orig_obj")
    println("  Optimal objective (hybrid): $hyb_obj")
    println("  Objective gap: $obj_gap  $(obj_gap < 1e-3 ? "✓" : "✗")")
end

# Compare iteration counts
println("  Total outer iterations (orig): $(length(result_orig[:inner_iter]))")
println("  Total outer iterations (hybrid): $(length(result_hyb[:inner_iter]))")
println("  Total inner iterations (orig): $(sum(result_orig[:inner_iter]))")
println("  Total inner iterations (hybrid): $(sum(result_hyb[:inner_iter]))")
println("  Time (orig): $(result_orig[:solution_time]) s")
println("  Time (hybrid): $(result_hyb[:solution_time]) s")

# =====================================================================
# COMPARISON: Original vs Hybrid
# =====================================================================
println("\n" * "="^80)
println("COMPARISON: Original vs Hybrid")
println("="^80)

# Compare optimal solutions
for (name, res) in [("Original", result_orig), ("Hybrid", result_hyb)]
    if haskey(res, :opt_sol)
        println("  $name: x*=$(res[:opt_sol][:x]), λ*=$(res[:opt_sol][:λ])")
    end
end

# Compare objectives
for (name, res) in [("Original", result_orig), ("Hybrid", result_hyb)]
    if haskey(res, :past_local_lower_bound)
        obj = minimum(res[:past_local_lower_bound])
        println("  $name objective: $obj")
    end
end

# Compare iteration counts and times
for (name, res) in [("Original", result_orig), ("Hybrid", result_hyb)]
    println("  $name: outer=$(length(res[:inner_iter])), inner=$(sum(res[:inner_iter])), time=$(res[:solution_time])s")
end

# Objective gap
if haskey(result_orig, :past_local_lower_bound) && haskey(result_hyb, :past_local_lower_bound)
    orig_obj = minimum(result_orig[:past_local_lower_bound])
    hyb_obj = minimum(result_hyb[:past_local_lower_bound])
    gap = abs(orig_obj - hyb_obj)
    println("\n  Orig vs Hybrid objective gap: $gap  $(gap < 1e-3 ? "✓" : "✗")")
end

# NOTE: full_primal=true is disabled.
# Outer cut extraction via Mosek IPM shadow prices (evaluate_master_opt_cut_from_primal)
# produces non-uniformly inaccurate coefficients (30-40% error vs dual ISP variable values)
# due to conic dual degeneracy. This causes invalid outer cuts → OMP extreme solutions → ISP infeasible.
# Unlike inner cut μ offset (uniform +ε, correctable), this cannot be simply fixed.
# See memory/ipm_mu_offset.md and debug_test/test_outer_cut_compare.jl.

println("\n" * "="^80)
println("TEST COMPLETE")
println("="^80)
