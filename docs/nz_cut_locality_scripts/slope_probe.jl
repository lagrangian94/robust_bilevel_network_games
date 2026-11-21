# x̂ 에서 restricted ISP (α 고정 Ω LP) 의 cut 기울기: 기본 해 / MW / 최적면 안의 기울기 범위 / F̂ (α 고정 LP 를 분수 x 로) 의 방향 미분
# (2026-11-21, Benders 기울기 진단). 저장소 코드는 읽기만 한다.
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl")); include(joinpath(root, "nonzero_sum", "nz_belief_ccg.jl"))
const ENV1 = Gurobi.Env()
GRB1() = (o = Gurobi.Optimizer(ENV1); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
nd = make_loc_tkp(; g=1, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
K = nd.meta[:K]; nB = nd.meta[:nB]; S = nd.S; XE = nz_xentries(nd); θ, λU = nd.thetaU, nd.lambdaU
xi(site, k) = (site - 1) * K + k
x̂ = zeros(nd.nx); x̂[xi(1, 2)] = 1; x̂[xi(8, 1)] = 1; x̂[xi(13, 2)] = 1        # 알려진 최적 (점포 3:300, 10:200, 15:300)
@printf("x̂ = %s, θᵁ = %.3f, λᵁ = %.1f\n", loc_tkp_xstr(nd, x̂), θ, λU)

# 최악 belief 의 반응 α̂: 싼 탐색 (두 성분 질량 이동 후보) 으로
ev = BCCGEval(nd, ENV1); _bccg_setx!(ev, nd, x̂)
qh = nd.q_hat; bx = nd.meta[:box]
best = (-Inf, nothing)
for i in 1:S, j in 1:S
    i == j && continue
    mv = min(qh[i] - bx.dl[i], bx.du[j] - qh[j], nd.eps_tilde); mv <= 1e-9 && continue
    d = copy(qh); d[i] -= mv; d[j] += mv
    v, dd, r, α = _bccg_g!(ev, nd, d)
    v > best[1] && (global best = (v, α))
end
α̂ = best[2]; @printf("싼 탐색 최고값 (리더 항) %.3f\n", best[1])

Fx = nz_build_fixed_alpha(nd, x̂; optimizer=GRB1)
z = nz_eval_alpha!(Fx, nd, α̂); @printf("restricted ISP 값 g(x̂) = %.3f\n", z)
cut0 = nz_cut_from_solution(Fx, nd, x̂)
mwc = _nz_mw_cut!(Fx, nd, x̂, z, nz_core_point(nd))
nz_set_objective!(Fx, nd, x̂); optimize!(Fx.model)

v = Fx.v
function fexpr(i)
    ex = AffExpr(0.0)
    for (jj, e) in enumerate(XE)
        e.i == i || continue
        k = e.k
        for s in 1:S
            add_to_expression!(ex, -θ * e.g[s], v[:ϖ][k, s])
            add_to_expression!(ex, -nd.piLU[k], v[:ρ̂1][jj, s]); add_to_expression!(ex, nd.piLU[k], v[:ρ̂3][jj, s])
            add_to_expression!(ex, -λU * nd.piFU[k], v[:ρ̃1][jj, s]); add_to_expression!(ex, λU * nd.piFU[k], v[:ρ̃3][jj, s])
        end
    end
    add_to_expression!(ex, -λU, v[:ρ01][i]); add_to_expression!(ex, λU, v[:ρ03][i])
    return ex
end
# F̂ 방향 미분: α 고정 LP 를 x̂ + t eᵢ 에서
function fhat(xx)
    nz_set_objective!(Fx, nd, xx); optimize!(Fx.model)
    termination_status(Fx.model) == MOI.OPTIMAL || return NaN
    return objective_value(Fx.model)
end
probe = [(4, 2), (6, 2), (11, 2), (2, 1), (5, 3), (12, 2)]
@printf("\n%-10s %10s %10s %12s %12s %12s %12s %12s\n", "거점:크기", "기본 cut", "MW cut", "최적면 min", "최적면 max", "F̂ t=1e-3", "F̂ t=0.5", "F̂ t=1")
for (st, k) in probe
    i = xi(st, k)
    d3 = (fhat(x̂ .+ 1e-3 .* (1:nd.nx .== i)) - z) / 1e-3
    d5 = (fhat(x̂ .+ 0.5 .* (1:nd.nx .== i)) - z) / 0.5
    d1 = (fhat(x̂ .+ 1.0 .* (1:nd.nx .== i)) - z) / 1.0
    nz_set_objective!(Fx, nd, x̂)
    con = @constraint(Fx.model, objective_function(Fx.model) >= z - 1e-7 * max(1.0, abs(z)))
    fe = fexpr(i)
    @objective(Fx.model, Min, fe); optimize!(Fx.model); fmin = termination_status(Fx.model) == MOI.OPTIMAL ? objective_value(Fx.model) : NaN
    @objective(Fx.model, Max, fe); optimize!(Fx.model); fmax = termination_status(Fx.model) == MOI.OPTIMAL ? objective_value(Fx.model) : NaN
    delete(Fx.model, con); nz_set_objective!(Fx, nd, x̂)
    @printf("%-10s %10.1f %10.1f %12.1f %12.1f %12.1f %12.1f %12.1f\n", string(st + 2, ":", Int(nd.meta[:tiers][k])),
            cut0[:slope][i], mwc === nothing ? NaN : mwc[:slope][i], fmin, fmax, d3, d5, d1)
end
println("\n(기울기 단위: 리더 단위 / x 한 단위. 참고: 한 곳 값 ≈ −5·min(용량, 300), 크기 300 이면 약 −1,500)")
