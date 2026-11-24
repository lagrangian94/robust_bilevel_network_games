root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Serialization
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_belief_ccg.jl"))
env = Gurobi.Env()
lp_opt = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
for f in ARGS
    d = deserialize(f)
    nd = make_loc_tkp(; g=d.g, S=d.S, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2 / 3, beta=0.4, lambdaU=d.lam)
    ev = BCCGEval(nd, env); _bccg_setx!(ev, nd, d.x)
    F = nz_build_fixed_alpha(nd, d.x; optimizer=lp_opt)
    for (nm, α) in (("원래", d.α), ("소수 1", round.(d.α; digits=1)), ("소수 2", round.(d.α; digits=2)))
        a1 = [(v = clamp(α[i], 0.0, nd.hU[i]); v < 1e-6 * max(1.0, nd.hU[i]) ? 0.0 : v) for i in eachindex(α)]
        v3, dd = _bccg_value!(ev, nd, a1)
        how = "직접"
        if dd === nothing
            a1 = nz_feasible_alpha(nd, d.x, a1; lp_optimizer=lp_opt, tol=1e-3); v3, dd = _bccg_value!(ev, nd, a1); how = "투영"
        end
        @printf("%s %-6s: 벌점 Ω %.4f, 정확값 %.4f (%s, belief %s)\n", f, nm, nz_eval_alpha!(F, nd, α), v3, how, dd === nothing ? "없음" : "있음")
    end
end
