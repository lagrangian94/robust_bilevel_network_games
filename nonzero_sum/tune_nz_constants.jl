"""
tune_nz_constants.jl — θᵁ, λᵁ 를 인스턴스에 맞게 조이고 Ω = V* 를 인증.

  1. KKT 로 모든 x 의 V*(x) 와 최적 α 를 구함 (big-M 없음).
  2. θ 진단: (x, α, s) 에서 필요한 θ (2단계 LP 의 최적성 제약 dual) 의 최댓값 θ̂.
     α 는 각 x 의 KKT 최적 α + H 의 무작위 점 NZ_NRAND 개.  → θᵁ 후보 = NZ_THETA_MULT · θ̂ (circuit 상계 v/해상도 이하로)
  3. λ 스캔: 고른 θᵁ 에서 λᵁ 를 내려가며 어려운 x 들에서 Ω incumbent > V* (과대평가) 가 생기는 경계를 찾음.
  4. 인증: (θᵁ, λᵁ) 로 모든 x 에서 Ω 를 풀고 incumbent·상한을 V* 와 비교 (상한 ≤ V* + tol 이면 exact 인증).
환경변수: run_nz_benders.jl 과 같은 인스턴스 인자, NZ_NRAND=20, NZ_THETA_MULT=2, NZ_LAMBDAS="20,10,5,2,1", NZ_LAMBDA_MULT=4,
         NZ_TL=300
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))

envf(k, d) = get(ENV, k, d)
xs_str(x) = string(findall(x .> 0.5))

function main()
    kw = Dict{Symbol,Any}(:coords => Symbol(envf("NZ_COORDS", "sgb128")), :reservation => Symbol(envf("NZ_RES", "pair")),
        :quota => parse(Float64, envf("NZ_QUOTA", "100")), :wres => parse(Float64, envf("NZ_WRES", "300")),
        :S => parse(Int, envf("NZ_S", "3")), :seed => parse(Int, envf("NZ_SEED", "1")))
    nd0 = make_location_instance(; kw...)
    tl = parse(Float64, envf("NZ_TL", "300"))
    X = [Float64.(collect(b)) for b in Iterators.product(fill(0:1, nd0.nx)...) if sum(b) <= nd0.gamma]
    @printf("%s  (circuit θᵁ = %.3f, 기본 λᵁ = %.1f, max π̂ᵁ = %.1f, max π_Fᵁ = %.1f)\n", nd0.name, nd0.thetaU, nd0.lambdaU,
            maximum(nd0.piLU), maximum(nd0.piFU)); flush(stdout)

    # ---- 1. KKT ----
    kkt = Dict{Vector{Float64},Any}()
    for x in X
        kkt[x] = nz_kkt_value(nd0, x; optimizer=GRB, time_limit=tl)
    end
    println("1. KKT V*(x): ", join(["$(xs_str(x))=$(round(kkt[x][:value]; digits=4))" for x in X], ", ")); flush(stdout)

    # ---- 2. θ 진단 ----
    rng = MersenneTwister(0)
    nr = parse(Int, envf("NZ_NRAND", "20"))
    θhat = 0.0; arg = nothing
    for x in X
        αs = [kkt[x][:α]]
        for _ in 1:nr
            a = rand(rng, length(nd0.c0)) .* nd0.hU
            push!(αs, _clip_H(a, nd0))
        end
        for α in αs, s in 1:nd0.S
            dg = nz_theta_diag(nd0, x, α, s; lp_optimizer=GRB)
            dg[:theta_need] > θhat && (θhat = dg[:theta_need]; arg = (xs_str(x), s))
        end
    end
    θU = min(parse(Float64, envf("NZ_THETA_MULT", "2")) * θhat, nd0.thetaU)
    θU = max(θU, 1e-6)
    @printf("2. θ 진단: 최대 필요 θ = %.4f (x=%s, s=%d), circuit 상계 %.3f  → θᵁ = %.4f\n", θhat, arg..., nd0.thetaU, θU)
    flush(stdout)

    # ---- 3. λ 스캔 (과대평가는 incumbent 만으로 드러남) ----
    hard = sort(X; by=x -> -sum(x))[1:min(6, length(X))]
    push!(hard, zeros(nd0.nx))
    λb = 0.0
    for λ in parse.(Float64, split(envf("NZ_LAMBDAS", "20,10,5,2,1"), ","))
        nd = nz_with_bounds(nd0; thetaU=θU, lambdaU=λ)
        O = build_nz_omega(nd; optimizer=GRB)
        worst = -Inf; tmax = 0.0
        for x in hard
            t = @elapsed (r = nz_solve!(O, nd, x; time_limit=min(tl, 120.0)))
            worst = max(worst, (r[:Fval] - kkt[x][:value]) / max(1, abs(kkt[x][:value]))); tmax = max(tmax, t)
        end
        @printf("3. λᵁ = %-6g  max(Ω inc − V*)/|V*| = %+.2e  (최대 %.1fs)%s\n", λ, worst, tmax, worst > 1e-5 ? "  ← 과대평가" : "")
        flush(stdout)
        worst > 1e-5 && (λb = max(λb, λ))
    end
    λU = λb > 0 ? parse(Float64, envf("NZ_LAMBDA_MULT", "4")) * λb : parse(Float64, split(envf("NZ_LAMBDAS", "20,10,5,2,1"), ",")[end])
    @printf("   과대평가가 생긴 최대 λ = %g → λᵁ = %g\n", λb, λU); flush(stdout)

    # ---- 4. 인증 ----
    nd = nz_with_bounds(nd0; thetaU=θU, lambdaU=λU)
    O = build_nz_omega(nd; optimizer=GRB)
    ncert = 0
    for x in X
        t = @elapsed (r = nz_solve!(O, nd, x; time_limit=tl))
        V = kkt[x][:value]; tol = 1e-5 * max(1, abs(V))
        cert = r[:bound] <= V + tol
        ncert += cert
        @printf("4. x=%-14s V*=%11.4f  Ω inc=%11.4f bd=%11.4f  %6.1fs %-10s %s\n", xs_str(x), V, r[:Fval], r[:bound], t,
                r[:status] == MOI.OPTIMAL ? "" : string(r[:status]), cert ? "인증" : (r[:Fval] > V + tol ? "과대평가!" : "미인증"))
        flush(stdout)
    end
    @printf("인증 %d / %d  (θᵁ = %.4f, λᵁ = %g)\n", ncert, length(X), θU, λU)
end

"H = {h ≥ 0 : Wh ≤ w} 로 비례 축소"
function _clip_H(a, nd)
    b = clamp.(a, 0.0, nd.hU)
    for l in 1:size(nd.W, 1)
        t = dot(nd.W[l, :], b)
        t > nd.wvec[l] && (b .*= nd.wvec[l] / t)
    end
    return b
end

main()
