root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Serialization
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_belief_ccg.jl"))
env = Gurobi.Env(); opt = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); o)
for (g, S, xs) in ((1, 10, "{9:200;10:300;15:300}"), (5, 10, "{4:200;7:200;12:300}"), (1, 20, "{9:200;10:300;15:200}"), (5, 10, "{10:300;12:300}"), (1, 10, "{6:400;10:200;15:200}"))
    nd = make_loc_tkp(; g=g, S=S, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2 / 3, beta=0.4, lambdaU=1e4)
    K = nd.meta[:K]; tiers = nd.meta[:tiers]; nA = nd.meta[:nA]
    x̄ = zeros(nd.nx)
    for p in split(strip(xs, ['{', '}']), ";"; keepempty=false)
        i, t = parse.(Int, split(p, ":")); k = findfirst(==(Float64(t)), tiers); x̄[(i - nA - 1) * K + k] = 1.0
    end
    hU0, nf = nz_presolve_hU(nd, x̄)
    # 스크립트 HUCAP
    hs = copy(hU0); sx = nd.meta[:site_x]; hidx = nd.meta[:hidx]
    for i in nA+1:nd.meta[:nst]
        b = sum(tiers[(t - 1) % K + 1] * x̄[t] for t in sx[i]; init=0.0); k = hidx(i, 1); (b > 0.5 && b < hs[k]) && (hs[k] = b)
    end
    t1 = time(); hf, nt = nz_presolve_hU_feas(nz_with_hU(nd, hU0), x̄; optimizer=opt); t = time() - t1
    d = [(i, hU0[i], hs[i], round(hf[i]; digits=4)) for i in eachindex(hf) if hU0[i] > 0 && (hf[i] < hU0[i] || hs[i] < hU0[i])]
    @printf("g%d S%d %s: 닫힌 거점 고정 %d, LP 로 줄인 좌표 %d (%.2fs), (좌표, hU, 스크립트, LP) = %s, 최대 차이 %.2e\n", g, S, xs, nf, nt, t, string(d), maximum(abs.(hf .- hs)))
end
