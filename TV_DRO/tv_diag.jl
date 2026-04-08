# Quick diagnostic: full vs benders key numbers only
include("tv_verify.jl")

network, tv = setup_instance(; m=3, n=3, S=2, seed=42, eps_hat=0.2, eps_tilde=0.2)
K = tv.num_arcs

# Full model
full_model, full_vars = build_full_tv_model(tv; optimizer=Gurobi.Optimizer)
optimize!(full_model)
full_obj = objective_value(full_model)
x_full = round.(Int, [value(full_vars[:x][k]) for k in 1:K])
λ_full = value(full_vars[:λ])
t_full = value(full_vars[:t])
ν_full = value(full_vars[:ν])
println("=== Full model ===")
@printf("obj=%.6f  t=%.6f  ν=%.6f  w*ν=%.6f\n", full_obj, t_full, ν_full, tv.w*ν_full)
@printf("λ=%.4f  x=%s\n", λ_full, string(x_full))

# Benders (verbose=false, capture result)
println("\n=== Benders ===")
result = tv_nested_benders_optimize!(tv;
    lp_optimizer=HiGHS.Optimizer,
    mip_optimizer=Gurobi.Optimizer,
    max_outer_iter=30, max_inner_iter=200,
    outer_tol=1e-4, inner_tol=1e-5, verbose=false)

@printf("Z₀=%.6f  status=%s  outer_iters=%d\n", result[:Z0], result[:status], result[:outer_iters])
if haskey(result, :x)
    x_bd = round.(Int, result[:x])
    @printf("λ=%.4f  x=%s\n", result[:λ], string(x_bd))
    @printf("LB=%.6f  UB=%.6f\n", result[:lower_bound], result[:upper_bound])
end

gap = abs(full_obj - result[:Z0]) / max(abs(full_obj), 1e-10)
@printf("\nFull=%.6f  Benders=%.6f  gap=%.2e\n", full_obj, result[:Z0], gap)

# Also check: at full model's x, what does OSP give?
println("\n=== OSP at full model's (x,h,λ,ψ⁰) ===")
h_full = [value(full_vars[:h][k]) for k in 1:K]
ψ0_full = [value(full_vars[:ψ0][k]) for k in 1:K]
Z_osp = solve_osp_primal(tv, Float64.(x_full), h_full, λ_full, ψ0_full)
@printf("Z_OSP = %.6f  (should ≈ full_obj = %.6f)\n", Z_osp, full_obj)
