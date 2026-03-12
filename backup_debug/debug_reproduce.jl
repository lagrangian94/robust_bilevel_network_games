"""
Reproduce DUAL_INFEASIBLE: inner loop without TR, just IMP + primal ISP cuts.
Trust region 없이 순수하게 α를 바꿔가며 primal ISP가 언제 실패하는지 확인.
실패한 α를 dual ISP로도 테스트.
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using Hypatia
using LinearAlgebra
using Serialization
using Revise

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("strict_benders.jl")
includet("nested_benders_trust_region.jl")
includet("build_primal_isp.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model

S = 1; ϕU = 10.0; λU = 10.0; γ = 2.0; w = 1.0; v = 1.0; seed = 42; epsilon = 0.5
network = generate_grid_network(3, 3, seed=seed)
capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacities[1:end-1, :], epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)
num_arcs = length(network.arcs) - 1

# OMP solve → initial (x,h,λ,ψ0)
omp_model, omp_vars = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
optimize!(omp_model)
x_sol = value.(omp_vars[:x]); h_sol = value.(omp_vars[:h])
λ_sol = value(omp_vars[:λ]); ψ0_sol = value.(omp_vars[:ψ0])
println("x_sol = ", x_sol)
println("λ_sol = ", λ_sol)

E = ones(num_arcs, num_arcs+1); d0 = zeros(num_arcs+1); d0[end] = 1.0
isp_data = Dict(:E=>E, :ϕU=>ϕU, :d0=>d0, :S=>S, :w=>w, :v=>v,
    :uncertainty_set=>uncertainty_set, :network=>network, :λU=>λU, :γ=>γ)

# IMP
imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_optimizer=Gurobi.Optimizer)
optimize!(imp_model)

# Primal ISP
primal_li, primal_fi = initialize_primal_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)

# Dual ISP (for comparison)
dual_li, dual_fi = initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=zeros(num_arcs))

# Inner loop — no TR, just cuts
println("\n" * "="^60)
println("Inner loop reproduction (no TR)")
println("="^60)

α_fail = nothing
for iter in 1:20
    optimize!(imp_model)
    α_sol = value.(imp_vars[:α])
    println("\n--- Inner iter $iter ---")
    println("  α = ", round.(α_sol, digits=8))
    println("  sum(α) = ", sum(α_sol))

    subprob_obj = 0.0
    failed = false
    for s in 1:S
        # Primal ISP leader
        model_l = primal_li[s][1]
        vars_l = primal_li[s][2]
        μhat = vars_l[:μhat]
        for k in 1:num_arcs
            set_objective_coefficient(model_l, μhat[1, k], α_sol[k])
        end
        optimize!(model_l)
        st_l = MOI.get(model_l, MOI.TerminationStatus())
        println("  Primal leader s=$s: ", termination_status(model_l))

        if !(st_l == MOI.OPTIMAL || st_l == MOI.SLOW_PROGRESS)
            println("  *** PRIMAL LEADER FAILED ***")
            α_fail = copy(α_sol)
            failed = true
            break
        end

        # Primal ISP follower
        model_f = primal_fi[s][1]
        vars_f = primal_fi[s][2]
        μtilde = vars_f[:μtilde]
        for k in 1:num_arcs
            set_objective_coefficient(model_f, μtilde[1, k], α_sol[k])
        end
        optimize!(model_f)
        st_f = MOI.get(model_f, MOI.TerminationStatus())
        println("  Primal follower s=$s: ", termination_status(model_f))

        if !(st_f == MOI.OPTIMAL || st_f == MOI.SLOW_PROGRESS)
            println("  *** PRIMAL FOLLOWER FAILED ***")
            α_fail = copy(α_sol)
            failed = true
            break
        end

        # Extract cuts and add to IMP
        μhat_val = collect(value.(μhat[1, :]))
        ηhat_val = value(vars_l[:ηhat][1])
        intercept_l = (1/S) * ηhat_val
        μtilde_val = collect(value.(μtilde[1, :]))
        ηtilde_val = value(vars_f[:ηtilde][1])
        intercept_f = (1/S) * ηtilde_val

        subprob_obj += (intercept_l + α_sol' * μhat_val) + (intercept_f + α_sol' * μtilde_val)

        @constraint(imp_model, imp_vars[:t_1_l][s] <= intercept_l + imp_vars[:α]' * μhat_val)
        @constraint(imp_model, imp_vars[:t_1_f][s] <= intercept_f + imp_vars[:α]' * μtilde_val)
    end

    if failed
        break
    end
    println("  subprob_obj = ", subprob_obj)
end

# ===== 실패한 α로 dual ISP 테스트 =====
if α_fail !== nothing
    println("\n" * "="^60)
    println("Testing failing α with DUAL ISP")
    println("="^60)
    println("α_fail = ", α_fail)

    for s in 1:S
        U_s = Dict(:R => Dict(:1=>R[s]), :r_dict => Dict(:1=>r_dict[s]),
                    :xi_bar => Dict(:1=>xi_bar[s]), :epsilon => epsilon)
        (status_dl, cut_dl) = isp_leader_optimize!(
            dual_li[s][1], dual_li[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_fail)
        println("  Dual leader s=$s: status=$status_dl, obj=$(cut_dl[:obj_val])")

        (status_df, cut_df) = isp_follower_optimize!(
            dual_fi[s][1], dual_fi[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_fail)
        println("  Dual follower s=$s: status=$status_df, obj=$(cut_df[:obj_val])")
    end

    # Fresh primal ISP build with failing α
    println("\n--- Fresh primal ISP build with failing α ---")
    primal_li2, primal_fi2 = initialize_primal_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
        conic_optimizer=Mosek.Optimizer, x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)
    for s in 1:S
        (st_pl, ci_pl) = primal_isp_leader_optimize!(primal_li2[s][1], primal_li2[s][2];
            isp_data=isp_data, α_sol=α_fail)
        println("  Fresh primal leader s=$s: status=$st_pl")
        if st_pl == :OptimalityCut
            println("    obj=$(ci_pl[:obj_val])")
        end
    end
else
    println("\nNo failure in 20 iterations!")
end
