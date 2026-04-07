# Direct test: reproduce DUAL_INFEASIBLE with known failing solution and extract ray.
# From Benders iteration 21: x=[10,17], λ=2.3813
using JuMP, Gurobi, Mosek, MosekTools, LinearAlgebra, Revise

includet("../network_generator.jl")
includet("../build_uncertainty_set.jl")
includet("test_build_dualized_outer_subprob.jl")
includet("test_strict_benders.jl")
using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model

S=1; λU=10.0; v=1.0; seed=42; epsilon=0.5; ϕU=1/epsilon
network = generate_grid_network(4, 4, seed=seed)
num_arcs = length(network.arcs) - 1
num_nodes = length(network.nodes)
capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R=>R, :r_dict=>r_dict, :xi_bar=>xi_bar, :epsilon=>epsilon)
γ = 2.0; w = 1.0

# Failing solution from Benders iter 21
x_sol = zeros(num_arcs); x_sol[10] = 1.0; x_sol[17] = 1.0
λ_sol = 2.3813
h_sol = zeros(num_arcs)
ψ0_sol = zeros(num_arcs); ψ0_sol[10] = λ_sol; ψ0_sol[17] = λ_sol

println("x_nonzero = $(findall(x_sol .> 0.5)), λ = $λ_sol")
println("Arc 10: $(network.arcs[10]), Arc 17: $(network.arcs[17])")

osp_model, osp_vars, osp_data = build_dualized_outer_subproblem(
    network, S, ϕU, λU, γ, w, v, uncertainty_set, Mosek.Optimizer,
    λ_sol, x_sol, h_sol, ψ0_sol)
set_silent(osp_model)
optimize!(osp_model)

t_st = termination_status(osp_model)
p_st = primal_status(osp_model)
println("\nStatus: $t_st / $p_st")

if p_st != MOI.INFEASIBILITY_CERTIFICATE
    println("No infeasibility certificate. OSP status = $t_st")
    if t_st == MOI.OPTIMAL
        println("obj = $(objective_value(osp_model))")
    end
    # Try with even larger λ
    for test_λ in [3.0, 4.0, 5.0, 7.0, 10.0]
        x_t = zeros(num_arcs); x_t[10]=1.0; x_t[17]=1.0
        ψ0_t = zeros(num_arcs); ψ0_t[10]=test_λ; ψ0_t[17]=test_λ
        m2, v2, d2 = build_dualized_outer_subproblem(
            network, S, ϕU, λU, γ, w, v, uncertainty_set, Mosek.Optimizer,
            test_λ, x_t, zeros(num_arcs), ψ0_t)
        set_silent(m2)
        optimize!(m2)
        println("  λ=$test_λ: $(termination_status(m2)) / $(primal_status(m2))")
    end
else
    # ===== RAY EXTRACTION =====
    E_mat = ones(num_arcs, num_arcs+1)
    d0 = zeros(num_arcs+1); d0[end] = 1.0
    diag_x_E = Diagonal(x_sol) * E_mat
    diag_λ_ψ = Diagonal(λ_sol*ones(num_arcs) - v.*ψ0_sol)

    ray = Dict{String,Any}()
    for vn in [:α,:Mhat,:Mtilde,:Uhat1,:Uhat2,:Uhat3,:Utilde1,:Utilde2,:Utilde3,
               :βhat1,:βhat2,:βtilde1,:βtilde2,:Zhat1,:Zhat2,:Ztilde1,:Ztilde2,
               :Γhat1,:Γhat2,:Γtilde1,:Γtilde2,
               :Phat1_Φ,:Phat2_Φ,:Phat1_Π,:Phat2_Π,
               :Ptilde1_Φ,:Ptilde2_Φ,:Ptilde1_Π,:Ptilde2_Π,
               :Ptilde1_Y,:Ptilde2_Y,:Ptilde1_Yts,:Ptilde2_Yts]
        ray[string(vn)] = value.(osp_model[vn])
    end

    println("\n=== Ray magnitudes (top 15) ===")
    sorted = sort(collect(ray), by=x->maximum(abs.(x.second)), rev=true)
    for (i,(name,vals)) in enumerate(sorted)
        mx = maximum(abs.(vals))
        if mx > 1e-10 && i <= 15
            println("  $(rpad(name,20)) max=$(round(mx,digits=8))  sum=$(round(sum(abs.(vals)),digits=6))")
        end
    end

    println("\n=== Objective decomposition ===")
    for s in 1:S
        U1r=ray["Uhat1"]; U3r=ray["Uhat3"]; Ut1r=ray["Utilde1"]; Ut3r=ray["Utilde3"]
        t1 = -ϕU * sum((U1r[s,:,:] + Ut1r[s,:,:]) .* diag_x_E)
        t2 = -ϕU * sum((U3r[s,:,:] + Ut3r[s,:,:]) .* (E_mat - diag_x_E))
        t3 = dot(d0, ray["βhat1"][s, 1:num_arcs+1])

        zt3s = num_arcs+1+(num_nodes-1)+1; zt3e = zt3s+num_arcs-1
        t4 = sum(ray["Ztilde1"][s, zt3s:zt3e, :] .* (diag_λ_ψ * diagm(xi_bar[s])))
        t5 = λ_sol * dot(d0, ray["βtilde1"][s, 1:num_arcs+1])

        bt3s = num_arcs+1+(num_nodes-1)+1; bt3e = bt3s+num_arcs-1
        t6 = -dot(h_sol + diag_λ_ψ * xi_bar[s], ray["βtilde1"][s, bt3s:bt3e])

        tp1 = -ϕU*(sum(ray["Phat1_Φ"][s,:,:])+sum(ray["Phat1_Π"][s,:,:]))
        tp2 = -ϕU*(sum(ray["Phat2_Φ"][s,:,:])+sum(ray["Phat2_Π"][s,:,:]))
        tp3 = -ϕU*(sum(ray["Ptilde1_Φ"][s,:,:])+sum(ray["Ptilde1_Π"][s,:,:])+sum(ray["Ptilde1_Y"][s,:,:])+sum(ray["Ptilde1_Yts"][s,:]))
        tp4 = -ϕU*(sum(ray["Ptilde2_Φ"][s,:,:])+sum(ray["Ptilde2_Π"][s,:,:])+sum(ray["Ptilde2_Y"][s,:,:])+sum(ray["Ptilde2_Yts"][s,:]))

        terms = [
            ("U1+Ut1(×-ϕU×x)", t1), ("U3+Ut3(×-ϕU×(1-x))", t2),
            ("βh1_1·d0", t3), ("Zt3·(λ-ψ)·ξ", t4),
            ("βt1_1·λ·d0", t5), ("βt1_3·-(h+...)", t6),
            ("P_ub_hat", tp1), ("P_lb_hat", tp2),
            ("P_ub_tilde", tp3), ("P_lb_tilde", tp4),
        ]
        total = sum(x[2] for x in terms)
        for (label, val) in terms
            if abs(val) > 1e-10
                println("  $(rpad(label,28)) $(round(val, digits=8))")
            end
        end
        println("  $(rpad("TOTAL",28)) $(round(total, digits=8))")

        # Critical ϕU
        β_terms = t3 + t4 + t5 + t6
        UP_raw = (sum((U1r[s,:,:]+Ut1r[s,:,:]).*diag_x_E) +
                  sum((U3r[s,:,:]+Ut3r[s,:,:]).*(E_mat-diag_x_E)) +
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
            println("  UP_coefficient = $(round(UP_raw, digits=8))")
        else
            println("\n  UP_raw ≤ 0: purely structural unboundedness!")
            println("  β_contribution = $(round(β_terms, digits=8))")
            println("  UP_raw = $(round(UP_raw, digits=8))")
        end
    end

    # β detail
    println("\n=== βhat1_1 ray entries ===")
    for s in 1:S
        b = ray["βhat1"][s, 1:num_arcs+1]
        nz = findall(abs.(b) .> 1e-8)
        println("  nonzero indices: $nz")
        for i in nz; println("    [$i] = $(round(b[i], digits=8))"); end
    end
    println("\n=== βtilde1_1 ray entries ===")
    for s in 1:S
        b = ray["βtilde1"][s, 1:num_arcs+1]
        nz = findall(abs.(b) .> 1e-8)
        println("  nonzero indices: $nz")
        for i in nz; println("    [$i] = $(round(b[i], digits=8))"); end
    end

    # N_ts
    N_ts = network.N[:, end]
    println("\n=== N_ts (dummy arc column) ===")
    for i in 1:(num_nodes-1)
        if abs(N_ts[i]) > 1e-10
            println("  $(network.nodes[i+1]) row=$i: $(N_ts[i])")
        end
    end

    # Mhat/Mtilde ray (should be ~0 for true ray)
    println("\n=== M variables in ray (should be ≈0) ===")
    println("  max|Mhat_ray| = $(maximum(abs.(ray["Mhat"])))")
    println("  max|Mtilde_ray| = $(maximum(abs.(ray["Mtilde"])))")
    println("  max|α_ray| = $(maximum(abs.(ray["α"])))")
    println("  max|βhat2_ray| = $(maximum(abs.(ray["βhat2"])))")
    println("  max|βtilde2_ray| = $(maximum(abs.(ray["βtilde2"])))")
end
