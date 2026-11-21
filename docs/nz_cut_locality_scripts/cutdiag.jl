root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
include(joinpath(root, "nonzero_sum", "nz_root_tighten.jl"))
nd = make_loc_tkp(; g=1, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
for mw in (false, true)
res = nz_belief_menu_benders(nd; optimizer=GRB, tol=5e-3, oracle=:alpha_bnb, bnb=:dr, oracle_gap=1e-3, nworkers=4, time_limit=600,
        local_first=true, local_time=60.0, verbose=true, max_iter=4, mw=mw)
println("mw=$mw")
for c in res[:cuts]
    @printf("kind=%s it=%d x=%s int=%.4e  slope min %.4e max %.4e  |slope|>1e4: %d\n", c.kind, c.iter, string(findall(c.x .> 0.5)), c.intercept,
            minimum(c.slope), maximum(c.slope), count(abs.(c.slope) .> 1e4))
    println("   slope = ", round.(c.slope; digits=1))
end
end
println("piLU max ", maximum(nd.piLU), "  piFU max ", maximum(nd.piFU), " varpiU max ", maximum(filter(isfinite, nd.varpiU)), " lambdaU ", nd.lambdaU, " thetaU ", nd.thetaU)
