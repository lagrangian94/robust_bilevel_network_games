root = dirname(@__DIR__)   # 거점별 한도 pooled (d=15 seed 1 S=3) 의 λᵁ 진단 — 문서 §10.19 (5)
using JuMP, Gurobi, Printf
const GRB_ENV = Gurobi.Env()
GRB() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 2); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
for pool in (:store, :total)
    nd1 = nz_with_delta(make_location_instance(; coords=:random, n_loc=15, nA=2, nB=3, NB=3, S=3, seed=1, bA=750.0, bB=750.0, qB=750.0,
        dlo=50.0, dhi=150.0, reservation=:pooled, wres=300.0, pool_limit=pool, quota=300.0, eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
    x̄ = [1.0, 1.0, 0.0]
    α° = [300.0, 300.0, 265.4, 142.9, 0.0]; αs = [265.4, 300.0, 300.0, 142.9, 0.0]
    pool == :total && (α° = α° .* (300 / sum(α°)); αs = αs .* (300 / sum(αs)))
    for λ in (1000.0, 1e4, 1e5), θ in (50.0, 500.0)
        nd = nz_with_bounds(nd1; thetaU=θ, lambdaU=λ)
        hU, _ = nz_presolve_hU(nd, x̄); nd = nz_with_hU(nd, hU)
        F = nz_build_fixed_alpha(nd, x̄; optimizer=GRB)
        @printf("pool=%s λᵁ=%8.0f θᵁ=%5.0f  G(α°)=%10.4f  G(α*)=%10.4f\n", pool, λ, θ, nz_eval_alpha!(F, nd, α°), nz_eval_alpha!(F, nd, αs))
    end
end
