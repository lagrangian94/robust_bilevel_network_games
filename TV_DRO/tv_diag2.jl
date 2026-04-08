# Inner loop convergence diagnostic — 1 outer iteration
include("tv_verify.jl")

network, tv = setup_instance(; m=3, n=3, S=2, seed=42, eps_hat=0.2, eps_tilde=0.2)
K = tv.num_arcs

# Simulate first OMP solve (t₀≥0, no cuts → t₀=0, rest unconstrained)
omp_model, omp_vars = build_tv_omp(tv; optimizer=Gurobi.Optimizer)
optimize!(omp_model)
x1 = [value(omp_vars[:x][k]) for k in 1:K]
h1 = [value(omp_vars[:h][k]) for k in 1:K]
λ1 = value(omp_vars[:λ])
ψ1 = [value(omp_vars[:ψ0][k]) for k in 1:K]
@printf("OMP iter1: t₀=%.4f  λ=%.4f  Σh=%.4f  x=%s\n",
    objective_value(omp_model), λ1, sum(h1), string(round.(Int, x1)))

c1 = compute_c(tv, x1)
r1 = compute_r(tv, h1, λ1, c1)
@printf("  c range: [%.2f, %.2f],  r range: [%.2f, %.2f]\n",
    minimum(c1), maximum(c1), minimum(r1), maximum(r1))

# OSP primal at this point
Z_osp = solve_osp_primal(tv, x1, h1, λ1, ψ1)
@printf("  OSP primal: %.6f\n", Z_osp)

# Inner loop with verbose — first 10 iters only
imp_m, imp_v = build_tv_imp(tv; optimizer=HiGHS.Optimizer)
isp_l, isp_lv = build_tv_isp_leader(tv, c1; optimizer=HiGHS.Optimizer)
isp_f, isp_fv = build_tv_isp_follower(tv, c1, r1, λ1; optimizer=HiGHS.Optimizer)

inner = tv_inner_loop!(tv, imp_m, imp_v, isp_l, isp_lv, isp_f, isp_fv;
    max_inner_iter=15, inner_tol=1e-6, verbose=true)
@printf("  Inner result: Z₀=%.6f  iters=%d\n", inner[:Z0_val], inner[:inner_iters])

# Now test at full model's solution
println("\n--- At full model solution (λ=10) ---")
x_f = zeros(K); x_f[9]=1.0; x_f[10]=1.0  # from full model
λ_f = 10.0; ψ_f = λ_f .* x_f
h_f = zeros(K)  # start with zero

c_f = compute_c(tv, x_f)
r_f = compute_r(tv, h_f, λ_f, c_f)
@printf("  c range: [%.2f, %.2f],  r range: [%.2f, %.2f]\n",
    minimum(c_f), maximum(c_f), minimum(r_f), maximum(r_f))

Z_osp_f = solve_osp_primal(tv, x_f, h_f, λ_f, ψ_f)
@printf("  OSP primal: %.6f\n", Z_osp_f)

imp_m2, imp_v2 = build_tv_imp(tv; optimizer=HiGHS.Optimizer)
isp_l2, isp_l2v = build_tv_isp_leader(tv, c_f; optimizer=HiGHS.Optimizer)
isp_f2, isp_f2v = build_tv_isp_follower(tv, c_f, r_f, λ_f; optimizer=HiGHS.Optimizer)

inner2 = tv_inner_loop!(tv, imp_m2, imp_v2, isp_l2, isp_l2v, isp_f2, isp_f2v;
    max_inner_iter=15, inner_tol=1e-6, verbose=true)
@printf("  Inner result: Z₀=%.6f  iters=%d\n", inner2[:Z0_val], inner2[:inner_iters])
