"""
test_offset_by_alpha.jl — Measure IPM μ offset as a function of α density.

For a fixed (x, h, λ, ψ0), construct α vectors with varying number of nonzero
components and compare μ from primal ISP vs dual ISP.
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using LinearAlgebra
using Statistics
using Revise

includet("../network_generator.jl")
includet("../build_uncertainty_set.jl")
includet("../strict_benders.jl")
includet("../nested_benders_trust_region.jl")
includet("../build_primal_isp.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

# ===== Parameters =====
S = 1
ϕU = 10.0
λU = 10.0
γ = 2.0
w = 1.0
v = 1.0
seed = 42
epsilon = 0.5

# ===== Generate Network =====
println("Generating 5×5 grid network...")
network = generate_grid_network(5, 5, seed=seed)

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

num_arcs = length(network.arcs) - 1
E = ones(num_arcs, num_arcs+1)
d0 = zeros(num_arcs + 1)
d0[end] = 1.0

isp_data = Dict(:E => E, :network => network, :ϕU => ϕU, :λU => λU, :γ => γ,
    :w => w, :v => v, :uncertainty_set => uncertainty_set, :d0 => d0, :S => S)

R_unc = uncertainty_set[:R]
r_dict_unc = uncertainty_set[:r_dict]
xi_bar_unc = uncertainty_set[:xi_bar]

# ===== Step 1: Get a reasonable (x, h, λ, ψ0) from OMP =====
println("\nSolving OMP to get leader decisions...")
omp_model, omp_vars = build_omp(network, ϕU, λU, γ, w;
    optimizer=Gurobi.Optimizer, multi_cut=true)
st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars)

# Run a few Benders iterations to get a nontrivial x
imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer)
st_imp, α_init = initialize_imp(imp_model, imp_vars)

dl, df = initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_init)

t_0_l = omp_vars[:t_0_l]
t_0_f = omp_vars[:t_0_f]

# Run 10 simplified outer iterations to get diverse x solutions
for outer_iter in 1:10
    global st, λ_sol, x_sol, h_sol, ψ0_sol, imp_model, imp_vars, dl, df

    optimize!(omp_model)
    st = MOI.get(omp_model, MOI.TerminationStatus())
    if st == MOI.INFEASIBLE
        break
    end
    x_sol = value.(omp_vars[:x])
    h_sol = value.(omp_vars[:h])
    λ_sol = value(omp_vars[:λ])
    ψ0_sol = value.(omp_vars[:ψ0])

    # Quick inner solve
    optimize!(imp_model)
    α_sol = value.(imp_vars[:α])

    subprob_obj = 0.0
    for s in 1:S
        U_s = Dict(:R => Dict(:1=>R_unc[s]), :r_dict => Dict(:1=>r_dict_unc[s]),
                    :xi_bar => Dict(:1=>xi_bar_unc[s]), :epsilon => epsilon)
        (_, cut_info_l) = isp_leader_optimize!(dl[s][1], dl[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
        (_, cut_info_f) = isp_follower_optimize!(df[s][1], df[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
        subprob_obj += cut_info_l[:obj_val] + cut_info_f[:obj_val]

        # Add inner cut
        cl = @constraint(imp_model, [ss=1:S], imp_vars[:t_1_l][ss] <= cut_info_l[:intercept] + imp_vars[:α]'*cut_info_l[:μhat])
        cf = @constraint(imp_model, [ss=1:S], imp_vars[:t_1_f][ss] <= cut_info_f[:intercept] + imp_vars[:α]'*cut_info_f[:μtilde])
    end

    @constraint(omp_model, t_0_l + t_0_f >= subprob_obj)
    println("  Outer $outer_iter: x_nonzero=$(count(x->x>0.5, x_sol)), λ=$(round(λ_sol, digits=3)), obj=$(round(subprob_obj, digits=4))")
end

println("\nFinal x = ", round.(x_sol, digits=1))
println("Final λ = ", round(λ_sol, digits=4))

# ===== Step 2: Construct α vectors with varying density, measure offset =====
println("\n" * "="^80)
println("Step 2: Measure μ offset vs α density")
println("="^80)

# Build primal and dual ISP for the final (x,h,λ,ψ0)
pl, pf = initialize_primal_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)

dl2, df2 = initialize_isp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_init)

U_s = Dict(:R => Dict(:1=>R_unc[1]), :r_dict => Dict(:1=>r_dict_unc[1]),
            :xi_bar => Dict(:1=>xi_bar_unc[1]), :epsilon => epsilon)

# Construct α vectors with 1, 5, 10, 15, 20, 25 nonzero components
densities = [1, 3, 5, 10, 15, 20, 25]

println("\n#nonzero | mean_off_l (α>0) | mean_off_l (α=0) | mean_off_f (α>0) | mean_off_f (α=0) | max_off_l | min_off_l")

for n_nz in densities
    # Construct α: first n_nz arcs = 1.0, rest = 0.0
    α_test = zeros(num_arcs)
    α_test[1:min(n_nz, num_arcs)] .= 1.0

    # Dual ISP
    (_, ci_l_d) = isp_leader_optimize!(dl2[1][1], dl2[1][2];
        isp_data=isp_data, uncertainty_set=U_s,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_test)
    (_, ci_f_d) = isp_follower_optimize!(df2[1][1], df2[1][2];
        isp_data=isp_data, uncertainty_set=U_s,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_test)

    # Primal ISP — RAW μ (no ε correction)
    μhat_p = pl[1][2][:μhat]
    for k in 1:num_arcs
        set_objective_coefficient(pl[1][1], μhat_p[1, k], α_test[k])
    end
    optimize!(pl[1][1])
    μ_leader_primal = collect(value.(μhat_p[1, :]))

    μtilde_p = pf[1][2][:μtilde]
    for k in 1:num_arcs
        set_objective_coefficient(pf[1][1], μtilde_p[1, k], α_test[k])
    end
    optimize!(pf[1][1])
    μ_follower_primal = collect(value.(μtilde_p[1, :]))

    μ_leader_dual = ci_l_d[:μhat]
    μ_follower_dual = ci_f_d[:μtilde]

    offset_l = μ_leader_primal .- μ_leader_dual
    offset_f = μ_follower_primal .- μ_follower_dual

    mask_pos = α_test .> 0.01
    mask_zero = α_test .<= 0.01

    mean_off_l_pos = sum(mask_pos) > 0 ? mean(offset_l[mask_pos]) : NaN
    mean_off_l_zero = sum(mask_zero) > 0 ? mean(offset_l[mask_zero]) : NaN
    mean_off_f_pos = sum(mask_pos) > 0 ? mean(offset_f[mask_pos]) : NaN
    mean_off_f_zero = sum(mask_zero) > 0 ? mean(offset_f[mask_zero]) : NaN

    println(
        "$(lpad(n_nz, 8)) | " *
        "$(lpad(round(mean_off_l_pos, digits=5), 17)) | " *
        "$(lpad(round(mean_off_l_zero, digits=5), 16)) | " *
        "$(lpad(round(mean_off_f_pos, digits=5), 16)) | " *
        "$(lpad(round(mean_off_f_zero, digits=5), 16)) | " *
        "$(lpad(round(maximum(offset_l), digits=5), 9)) | " *
        "$(lpad(round(minimum(offset_l), digits=5), 9))"
    )

    # Also print detailed per-component offset for first density
    if n_nz == 1 || n_nz == 25
        println("  Detailed offset_l: ", round.(offset_l, digits=4))
    end
end

# ===== Step 3: Random α values (not just 0/1) =====
println("\n" * "="^80)
println("Step 3: Random α values (continuous)")
println("="^80)

using Random
rng = MersenneTwister(42)

println("\n#nonzero | mean_off_l (α>0) | mean_off_l (α=0) | std_off_l (α>0) | std_off_l (α=0)")

for n_nz in [1, 5, 10, 20, 47]
    # Random α with n_nz nonzero components
    α_test = zeros(num_arcs)
    perm = randperm(rng, num_arcs)
    for idx in perm[1:min(n_nz, num_arcs)]
        α_test[idx] = rand(rng) * 2.0  # random in [0, 2]
    end

    # Dual ISP
    (_, ci_l_d) = isp_leader_optimize!(dl2[1][1], dl2[1][2];
        isp_data=isp_data, uncertainty_set=U_s,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_test)

    # Primal ISP — RAW
    μhat_p = pl[1][2][:μhat]
    for k in 1:num_arcs
        set_objective_coefficient(pl[1][1], μhat_p[1, k], α_test[k])
    end
    optimize!(pl[1][1])
    μ_leader_primal = collect(value.(μhat_p[1, :]))
    μ_leader_dual = ci_l_d[:μhat]

    offset_l = μ_leader_primal .- μ_leader_dual

    mask_pos = α_test .> 0.01
    mask_zero = α_test .<= 0.01

    mean_off_l_pos = sum(mask_pos) > 0 ? mean(offset_l[mask_pos]) : NaN
    mean_off_l_zero = sum(mask_zero) > 0 ? mean(offset_l[mask_zero]) : NaN
    std_off_l_pos = sum(mask_pos) > 1 ? std(offset_l[mask_pos]) : NaN
    std_off_l_zero = sum(mask_zero) > 1 ? std(offset_l[mask_zero]) : NaN

    println(
        "$(lpad(n_nz, 8)) | " *
        "$(lpad(round(mean_off_l_pos, digits=5), 17)) | " *
        "$(lpad(round(mean_off_l_zero, digits=5), 16)) | " *
        "$(lpad(round(std_off_l_pos, digits=5), 15)) | " *
        "$(lpad(round(std_off_l_zero, digits=5), 15))"
    )
end
