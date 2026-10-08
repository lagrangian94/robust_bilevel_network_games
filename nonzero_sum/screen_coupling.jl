"""
screen_coupling.jl — belief 결합 δ 가 값을 실제로 바꾸는 인스턴스 탐색 (KKT 독립 평가기만 사용, 빠름).

각 설정·x 에서
  R    = V*_rect(ε̂, ε̃)            (δ = Inf)
  Rp   = V*_rect(ε̂', ε̃')          (유효 반경, δ = SCR_DELTA)
  Vd   = V*_δ                      (δ = SCR_DELTA)
  V0   = V*_0                      (common prior, δ = 0)
을 비교. "결합 고유 효과" = Rp − Vd > 0 (유효 반경으로 설명되지 않는 감소).

환경변수: SCR_KIND = zs | loc,  SCR_DELTA = 0.1,  SCR_TL = 120
실행: julia nonzero_sum/screen_coupling.jl
"""

root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
envf(k, d) = get(ENV, k, d)
const δS = parse(Float64, envf("SCR_DELTA", "0.1"))
const TL = parse(Float64, envf("SCR_TL", "120"))

all_x(nd) = [Float64.(collect(bits)) for bits in Iterators.product(fill(0:1, nd.nx)...)
             if sum(bits) <= nd.gamma && all(nd.x_allowed[i] || bits[i] == 0 for i in 1:nd.nx)]

function with_eps(nd, ε̂, ε̃, β)
    NZData(nd.name, nd.A, nd.c, nd.ell, nd.u, nd.hrow, nd.hcoef, nd.xrow, nd.g, nd.W, nd.wvec, nd.c0, nd.hU,
           nd.nx, nd.x_allowed, nd.gamma, nd.qx, nd.S, nd.q_hat, ε̂, ε̃, β,
           nd.thetaU, nd.piLU, nd.piFU, nd.lambdaU, nd.varpiU, copy(nd.meta))
end
function kv(nd, x)
    r = try nz_kkt_value(nd, x; optimizer=GRB, time_limit=TL) catch; return NaN end
    return r[:status] == MOI.OPTIMAL ? r[:value] : NaN
end

function screen(nd0, label; xs=all_x(nd0), ε_pairs=[(0.2, 0.2), (0.1, 0.3), (0.05, 0.3), (0.0, 0.3)], betas=[0.4, 0.7])
    for (ε̂, ε̃) in ε_pairs, β in betas
        nd = with_eps(nd0, ε̂, ε̃, β)
        ndp = with_eps(nd0, min(ε̂, ε̃ + δS), min(ε̃, ε̂ + δS), β)
        best = (0.0, nothing); bestc = (0.0, nothing); nx_eff = 0
        t = @elapsed for x in xs
            R = kv(nd, x); Rp = kv(ndp, x)
            Vd = kv(nz_with_delta(nd, δS), x); V0 = kv(nz_with_delta(nd, 0.0), x)
            sc = max(1.0, abs(R))
            (R - V0) / sc > 1e-5 && (nx_eff += 1)
            (R - Vd) / sc > best[1] && (best = ((R - Vd) / sc, (x=findall(x .> 0.5), R=R, Rp=Rp, Vd=Vd, V0=V0)))
            (Rp - Vd) / sc > bestc[1] && (bestc = ((Rp - Vd) / sc, (x=findall(x .> 0.5), R=R, Rp=Rp, Vd=Vd, V0=V0)))
        end
        @printf("%-34s ε̂=%.2f ε̃=%.2f β=%.1f | x with R>V0: %2d/%d | max rel(R−Vδ)=%.3e %s | max rel(Rp−Vδ)=%.3e %s (%.0fs)\n",
                label, ε̂, ε̃, β, nx_eff, length(xs), best[1], best[2] === nothing ? "" : string(best[2]),
                bestc[1], bestc[2] === nothing ? "" : string(bestc[2]), t)
        flush(stdout)
    end
end

kind = envf("SCR_KIND", "zs")
if kind == "zs"
    include(joinpath(root, "network_generator.jl"))
    using .NetworkGenerator
    include(joinpath(root, "true_dro", "true_dro_data.jl"))
    include(joinpath(root, "nonzero_sum", "nz_paper_instances.jl"))
    # 원고 생성 방식 (factor_additive, w = 0.5γ median, v ~ Bern(0.75)) 의 작은 grid. 이전 판 (uniform 용량, w = 1) 은
    # 회복 예산이 너무 작아 follower 효과가 없었다.
    for (g, S, seed) in [("grid3x3", 5, 1), ("grid3x3", 5, 2), ("grid3x3", 5, 3), ("grid3x4", 5, 1), ("grid3x4", 5, 2),
                         ("grid3x3", 5, 4)]
        nd, _ = make_paper_maxflow(g; S=S, seed=seed)
        screen(nd, @sprintf("%s S=%d seed=%d K=%d", g, S, seed, nd.nx); ε_pairs=[(0.2, 0.2), (0.0, 0.3)])
    end
else
    insts = [(:sgb128, :pair, 1), (:random, :pair, 4), (:random, :pooled, 2), (:random, :pooled, 4), (:random, :pooled, 6)]
    skip = parse(Int, envf("SCR_SKIP", "0"))
    for (coords, res, seed) in insts[skip+1:end]
        nd = make_location_instance(; coords=coords, reservation=res, S=3, seed=seed)
        screen(nd, "$(nd.name)"; betas=[0.4], ε_pairs=[(0.0, 0.3), (0.1, 0.3), (0.2, 0.2)])
    end
end
