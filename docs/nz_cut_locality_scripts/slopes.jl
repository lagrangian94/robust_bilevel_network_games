root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
nd = make_loc_tkp(; g=1, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
Oloc = build_nz_omega(nd; optimizer=GRB); set_optimizer_attribute(Oloc.model, "OptimalityTarget", 1)
v = Oloc.v; XE = v[:XE]; θ, λU = nd.thetaU, nd.lambdaU
for xs in ([Int[]], [[6, 12, 15]])[1:2], xl in xs
    x̄ = zeros(nd.nx); x̄[xl] .= 1
    rl = _nz_local_solve!(Oloc, nd, x̄; time_limit=60.0)
    G = zeros(4, nd.nx)
    for (jj, e) in enumerate(XE), s in 1:nd.S
        i, k = e.i, e.k
        G[1, i] += -θ * e.g[s] * value(v[:ϖ][k, s])
        G[2, i] += -nd.piLU[k] * (value(v[:ρ̂1][jj, s]) - value(v[:ρ̂3][jj, s]))
        G[3, i] += -λU * nd.piFU[k] * (value(v[:ρ̃1][jj, s]) - value(v[:ρ̃3][jj, s]))
    end
    for i in 1:nd.nx; G[4, i] += -λU * (value(v[:ρ01][i]) - value(v[:ρ03][i])); end
    @printf("x̄=%s Ω_loc=%.2f  λ=%s\n", string(xl), rl[:Fval], haskey(v, :λ) ? string(value(v[:λ])) : "-")
    for (g, nm) in enumerate(("θgϖ (리더 recourse)", "π̂ᵁρ̂ (리더 McC)", "λᵁπ_Fᵁρ̃ (follower primal McC)", "λᵁρ⁰ (follower VF McC)"))
        @printf("  %-34s min %.3e max %.3e\n", nm, minimum(G[g, :]), maximum(G[g, :]))
    end
    # 거점 4 (x 10,11,12) 만 자세히
    println("  site x 10:12 → ", round.(G[:, 10:12]; sigdigits=3))
    # ρ01 크기와 Σ g π̃ 비교
    ρ01 = [value(v[:ρ01][i]) for i in 1:nd.nx]
    gp = zeros(nd.nx); for (jj, e) in enumerate(XE), s in 1:nd.S; gp[e.i] += e.g[s] * value(v[:π̃][e.k, s]); end
    println("  ρ01[10:12] = ", round.(ρ01[10:12]; sigdigits=3), "  Σ g π̃ [10:12] = ", round.(gp[10:12]; sigdigits=3))
    println("  π̃ 용량 행 (거점 4, s=1..3) / d_s = ", [round(value(v[:π̃][XE[10].k, s]) / max(value(v[:d][s]), 1e-9); sigdigits=4) for s in 1:3], "  piFU = ", nd.piFU[XE[10].k])
end
