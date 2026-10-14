root = dirname(@__DIR__)   # 거점별 한도 pooled (d=15 seed 1 S=3) 의 λᵁ 진단 — 문서 §10.19 (5)
using JuMP, Gurobi, Printf
const GRB_ENV = Gurobi.Env()
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
nd1 = nz_with_delta(make_location_instance(; coords=:random, n_loc=15, nA=2, nB=3, NB=3, S=3, seed=1, bA=750.0, bB=750.0, qB=750.0,
    dlo=50.0, dhi=150.0, reservation=:pooled, wres=300.0, pool_limit=:store, quota=300.0, eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
vf = Dict([1, 2] => (-2081.5, -2081.01), [3] => (-1357.5, -1355.26), [1, 3] => (-2081.5, -2080.88))
for λ in (1500.0, 1e4), xi in ([1, 2], [3], [1, 3])
    nd = nz_with_bounds(nd1; thetaU=nz_theta_circuit_exact(nd1)[1], lambdaU=λ)
    x̄ = zeros(nd.nx); x̄[xi] .= 1
    t = @elapsed r = nz_alpha_bnb_dr(nd, x̄; nworkers=4, time_limit=300.0, rel_gap=1e-4, verbose=false, ipopt_time=30.0)
    lo, hi = vf[xi]
    ok = r[:LB] <= hi + 1e-4 * abs(hi) && r[:UB] >= lo - 1e-4 * abs(lo)
    @printf("λᵁ=%6.0f x=%-7s B&B [%.4f, %.4f] %s %.1fs | 독립 평가 [%.2f, %.2f] → %s\n", λ, string(xi), r[:LB], r[:UB],
            r[:is_exact] ? "Optimal" : "TimeLim", t, lo, hi, ok ? "겹침" : "** 어긋남 **")
    flush(stdout)
end
