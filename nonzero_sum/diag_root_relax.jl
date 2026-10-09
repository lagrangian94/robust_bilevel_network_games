"""
diag_root_relax.jl — 고정 x 에서 Ω 의 루트 McCormick 완화 상한이 참값 V*(x) 보다 얼마나 큰지, 그 차이가 어느 bilinear 에서
오는지 분해 (α-B&B 상한이 느슨한 원인 진단).

Ω 의 bilinear: ζL = α r (리더 recourse rhs), ζF = α d (follower recourse rhs), ζW = α ϖ (리더 pessimistic 복사본의
follower 최적성 dual, 목적에 −θ hcoef ζW). zero-sum 은 θ = 0 이라 ζW 가 없다.

변형
  base   : 세 곱 모두 McCormick (α ∈ [0, hᵁ], r ∈ [0, rmax], d ∈ [dmin, dmax], ϖ ∈ [0, ϖᵁ rmax])
  W_exact: ζW 만 정확 (ζW = α ϖ, Gurobi NonConvex) 나머지는 McCormick → ζW 기여 측정
  LF_exact: ζL, ζF 만 정확, ζW 는 McCormick
각 변형의 상한 − V*. 완화 해에서 θ Σ hcoef (α ϖ − ζW) (ζW 의 McCormick 이득) 도 출력.

환경변수: DR_X = "1,2"  DR_QUOTA = 150  DR_DELTA = 0.1  DR_EPS = 0.3  DR_TL = 600
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
TL = parse(Float64, envf("DR_TL", "600"))
nd = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=parse(Float64, envf("DR_QUOTA", "150")),
        wres=300.0, S=3, seed=1, eps_hat=parse(Float64, envf("DR_EPS", "0.3")), eps_tilde=parse(Float64, envf("DR_EPS", "0.3")),
        beta=0.4), pd(envf("DR_DELTA", "0.1")))
x̄ = zeros(nd.nx); x̄[parse.(Int, split(envf("DR_X", "1,2"), ","))] .= 1
Vstar = nz_kkt_value(nd, x̄; optimizer=GRB, time_limit=300.0)[:value]
@printf("%s  x=%s  θᵁ=%.1f λᵁ=%.1f  V*(x) (KKT) = %.4f\n", nd.name, string(findall(x̄ .> 0.5)), nd.thetaU, nd.lambdaU, Vstar)

mc!(m, z, u, v, ul, uu, vl, vu) = (@constraint(m, z >= ul * v + vl * u - ul * vl); @constraint(m, z >= uu * v + vu * u - uu * vu);
                                    @constraint(m, z <= uu * v + vl * u - uu * vl); @constraint(m, z <= ul * v + vu * u - ul * vu))
function run(variant)
    O = build_nz_omega(nd; optimizer=GRB, mode=:relax)
    m = O.model; v = O.v
    set_optimizer_attribute(m, "NonConvex", 2); set_time_limit_sec(m, TL)
    set_optimizer_attribute(m, "Threads", 4)        # 밤샘 벤치마크와 코어를 나눠 씀
    S, nh = nd.S, nz_nh(nd); Hr = v[:Hr]
    α, r, d, ϖ, ζL, ζF, ζW = v[:α], v[:r], v[:d], v[:ϖ], v[:ζL], v[:ζF], v[:ζW]
    for i in 1:nh, s in 1:S
        if variant == :LF_exact
            @constraint(m, ζL[i, s] == α[i] * r[s]); @constraint(m, ζF[i, s] == α[i] * d[s])
        else
            mc!(m, ζL[i, s], α[i], r[s], 0.0, nd.hU[i], 0.0, upper_bound(r[s]))
            mc!(m, ζF[i, s], α[i], d[s], 0.0, nd.hU[i], lower_bound(d[s]), upper_bound(d[s]))
        end
    end
    for (jj, k) in enumerate(Hr), s in 1:S
        i = nd.hrow[k]
        if variant == :W_exact
            @constraint(m, ζW[jj, s] == α[i] * ϖ[k, s])
        else
            mc!(m, ζW[jj, s], α[i], ϖ[k, s], 0.0, nd.hU[i], 0.0, upper_bound(ϖ[k, s]))
        end
    end
    nz_set_objective!(O, nd, x̄)
    t = @elapsed optimize!(m)
    st = termination_status(m)
    bd = st == MOI.OPTIMAL ? objective_value(m) : objective_bound(m)
    gainW = nd.thetaU * sum(nd.hcoef[k] * (value(α[nd.hrow[k]]) * value(ϖ[k, s]) - value(ζW[jj, s]))
                            for (jj, k) in enumerate(Hr), s in 1:S)
    @printf("  %-9s 상한 %12.2f  (상한 − V* = %10.2f)  상태 %-12s %6.1fs | 완화 해에서 θΣhcoef(αϖ − ζW) = %10.2f\n",
            string(variant), bd, bd - Vstar, string(st), t, gainW)
    flush(stdout)
end
for variant in (:base, :W_exact, :LF_exact); run(variant); end
