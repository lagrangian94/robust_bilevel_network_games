# 고정 x 에서 V*_{λᵁ}(x) 의 λᵁ 민감도 (Coupled-DR r=0.2, S=10). 값이 같으면 그 λᵁ 로도 정확 벌점
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl")); include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
cases = [(2, [(7, 1)]), (11, [(8, 1)]), (6, [(3, 3), (7, 2)]), (1, Tuple{Int,Int}[])]
for (g, sel) in cases, lam in (1.0, 10.0, 100.0)
    nd = make_loc_tkp(; g=g, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, lambdaU=lam)
    nA, K = nd.meta[:nA], nd.meta[:K]
    x̄ = zeros(nd.nx); for (i, k) in sel; x̄[(i - nA - 1) * K + k] = 1; end
    t = @elapsed (res = nz_alpha_bnb_dr(nd, x̄; nworkers=6, time_limit=600.0, rel_gap=1e-4))
    @printf("LAM g=%d x=%s λᵁ=%5.0f  LB %.4f UB %.4f %s (%.0fs)\n", g, loc_tkp_xstr(nd, x̄), lam, res[:LB], res[:UB], res[:is_exact] ? "exact" : "TL", t); flush(stdout)
end
