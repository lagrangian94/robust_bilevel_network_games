"""
test_nz_coupling.jl — 종속 ambiguity set (belief 결합 𝒟_δ = {(p,p̃) ∈ D̂×D̃ : d_TV(p,p̃) ≤ δ}) 검증.

각 x 와 δ 에서 V*_δ(x) 를 독립적으로 세 번 계산해 비교한다.
  Ω     : 일반 Ω (nz_omega.jl, Gurobi NonConvex) — 정리 (minimax) 의 𝔅 → 𝔅_δ 판, θᵁ·λᵁ·πᵁ 사용
  KKT   : big-M 없는 독립 평가 (nz_kkt_eval.jl) — 결합 제약만 추가. Ω 와 같으면 exact penalty (λᵁ) 가 결합에서도 유효
  old Ω : (zero-sum 만) 원고 코드 build_true_dro_subproblem(delta_couple=δ)
그리고 구조 성질을 확인한다.
  (P1) δ ≥ ε̂ + ε̃ 이면 rectangular (δ = Inf) 와 같다
  (P2) V*_δ(x) 는 δ 에 대해 비감소
  (P3) V*_δ(x) ≤ V*_rect(x; ε̂', ε̃'),  ε̃' = min(ε̃, ε̂+δ), ε̂' = min(ε̂, ε̃+δ)   (유효 반경 상계)

환경변수
  CPL_PART = zs | loc | all (기본 all)
  CPL_DELTAS = "Inf,0.4,0.2,0.1,0.05,0"   CPL_TL = 600 (Ω·KKT 시간 제한)
  CPL_ZS_NX = 0 (zero-sum 에서 평가할 x 수, 0 = 전부)
  location: NZ_COORDS=random NZ_RES=pooled NZ_SEED=4 NZ_S=3 (기본: 검증용 무작위 pooled seed 4)
실행: julia nonzero_sum/test_nz_coupling.jl
"""

root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)

include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))

envf(k, d) = get(ENV, k, d)
parse_delta(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
const DELTAS = parse_delta.(split(envf("CPL_DELTAS", "Inf,0.4,0.2,0.1,0.05,0"), ","))
const TL = parse(Float64, envf("CPL_TL", "600"))
const PART = envf("CPL_PART", "all")
const GAP = parse(Float64, envf("CPL_GAP", "1e-5"))

all_x(nd) = [Float64.(collect(bits)) for bits in Iterators.product(fill(0:1, nd.nx)...)
             if sum(bits) <= nd.gamma && all(nd.x_allowed[i] || bits[i] == 0 for i in 1:nd.nx)]

"x 마다 δ 별 (Ω, KKT [, old]) 표. 반환 Dict(x => Dict(δ => NamedTuple))"
function run_table(nd0, xs; old=nothing)
    out = Dict{Vector{Float64},Dict{Float64,Any}}()
    Os = Dict(δ => build_nz_omega(nz_with_delta(nd0, δ); optimizer=GRB) for δ in DELTAS)
    worst = 0.0; nbad = Ref(0)
    for x in xs
        out[x] = Dict{Float64,Any}()
        for δ in DELTAS
            nd = nz_with_delta(nd0, δ)
            t1 = @elapsed (res = nz_solve!(Os[δ], nd, x; time_limit=TL, gap=GAP))
            t2 = @elapsed (kk = nz_kkt_value(nd, x; optimizer=GRB, time_limit=TL))
            ov = NaN
            if old !== nothing
                om, ovars, td = old
                omdl, ov_ = build_true_dro_subproblem(td, x; optimizer=GRB, silent=true, delta_couple=δ)
                set_optimizer_attribute(omdl, "NonConvex", 2); set_optimizer_attribute(omdl, "MIPGap", 1e-6)
                set_time_limit_sec(omdl, TL)
                info = solve_true_dro_subproblem!(omdl, ov_, td, x)
                ov = info[:Z0_val]
            end
            # 결합 확인: KKT 해의 (a, d) 가 결합을 만족하는가
            tvad = 0.5 * sum(abs.(kk[:a] .- kk[:d]))
            diff = res[:Fval] - kk[:value]
            sc = max(1.0, abs(kk[:value]))
            # Ω 가 시간 제한이면 incumbent ≤ V* ≤ bound 만 확인 (KKT 가 OPTIMAL 일 때)
            contain = kk[:status] == MOI.OPTIMAL ?
                (res[:Fval] <= kk[:value] + 1e-5 * sc && kk[:value] <= res[:bound] + 1e-5 * sc) : true
            contain || (nbad[] += 1; println("  !! KKT 가 Ω 의 [incumbent, bound] 밖"))
            res[:status] == MOI.OPTIMAL && kk[:status] == MOI.OPTIMAL && (worst = max(worst, abs(diff) / sc))
            out[x][δ] = (Ω=res[:Fval], Ωbd=res[:bound], Ωst=res[:status], kkt=kk[:value], kktbd=kk[:bound],
                         kktst=kk[:status], old=ov, tvad=tvad, tΩ=t1, tK=t2)
            @printf("  x=%-12s δ=%-5s Ω=%12.4f [%s %.1fs]  KKT=%12.4f [%s %.1fs]%s  Ω−KKT=%+.2e  TV(a,d)=%.4f\n",
                    string(findall(x .> 0.5)), string(δ), res[:Fval], res[:status], t1, kk[:value], kk[:status], t2,
                    old === nothing ? "" : @sprintf("  old=%12.4f (Ω−old %+.1e)", ov, res[:Fval] - ov),
                    diff, tvad)
            flush(stdout)
        end
    end
    @printf("  최대 상대 |Ω − KKT| (둘 다 OPTIMAL) = %.2e,  [incumbent, bound] 포함 위반 %d 건\n", worst, nbad[])
    return out
end

"구조 성질 (P1)-(P3)"
function check_props(nd0, tab; label="")
    ε̂, ε̃ = nd0.eps_hat, nd0.eps_tilde
    ds = sort(collect(DELTAS))
    np1 = np2 = 0; bad1 = bad2 = 0
    for (x, row) in tab
        sc = max(1.0, abs(row[Inf].kkt))
        tol = 1e-5 * sc
        for δ in ds
            if isfinite(δ) && δ >= ε̂ + ε̃
                np1 += 1; abs(row[δ].kkt - row[Inf].kkt) > tol && (bad1 += 1;
                    @printf("  (P1) 위반 x=%s δ=%g: %.6f vs %.6f\n", string(findall(x .> 0.5)), δ, row[δ].kkt, row[Inf].kkt))
            end
        end
        for k in 1:length(ds)-1
            np2 += 1
            row[ds[k]].kkt > row[ds[k+1]].kkt + tol && (bad2 += 1;
                @printf("  (P2) 위반 x=%s δ=%g→%g: %.6f > %.6f\n", string(findall(x .> 0.5)), ds[k], ds[k+1],
                        row[ds[k]].kkt, row[ds[k+1]].kkt))
        end
    end
    @printf("  %s(P1) δ ≥ ε̂+ε̃ ⇒ rect: %d/%d 일치   (P2) δ 단조: %d/%d\n", label, np1 - bad1, np1, np2 - bad2, np2)
    # δ 별 rect 대비 감소량 요약
    for δ in ds
        isfinite(δ) || continue
        red = [row[Inf].kkt - row[δ].kkt for (x, row) in tab]
        nz = count(>(1e-5 * max(1.0, maximum(abs(row[Inf].kkt) for (x, row) in tab))), red)
        @printf("    δ=%-5g: V*_rect − V*_δ  최대 %.4f, 평균 %.4f, 감소한 x %d/%d\n", δ, maximum(red), sum(red) / length(red),
                nz, length(red))
    end
end

"(P3) 유효 반경 상계: V*_δ ≤ V*_rect(ε̂', ε̃')"
function check_effective_radius(nd0, tab, xs)
    ε̂, ε̃ = nd0.eps_hat, nd0.eps_tilde
    nbad = 0; ntot = 0
    for δ in DELTAS
        isfinite(δ) || continue
        ε̂p, ε̃p = min(ε̂, ε̃ + δ), min(ε̃, ε̂ + δ)
        (ε̂p == ε̂ && ε̃p == ε̃) && continue
        ndr = NZData(nd0.name, nd0.A, nd0.c, nd0.ell, nd0.u, nd0.hrow, nd0.hcoef, nd0.xrow, nd0.g, nd0.W, nd0.wvec, nd0.c0,
                     nd0.hU, nd0.nx, nd0.x_allowed, nd0.gamma, nd0.qx, nd0.S, nd0.q_hat, ε̂p, ε̃p, nd0.beta,
                     nd0.thetaU, nd0.piLU, nd0.piFU, nd0.lambdaU, nd0.varpiU, Dict{Symbol,Any}())
        for x in xs
            kk = nz_kkt_value(ndr, x; optimizer=GRB, time_limit=TL)
            ntot += 1
            v = tab[x][δ].kkt
            ok = v <= kk[:value] + 1e-5 * max(1.0, abs(v))
            ok || (nbad += 1)
            @printf("  (P3) x=%-12s δ=%-5g V*_δ=%12.4f ≤ V*_rect(%.2f,%.2f)=%12.4f  %s\n", string(findall(x .> 0.5)), δ, v,
                    ε̂p, ε̃p, kk[:value], ok ? "ok" : "위반")
        end
    end
    @printf("  (P3) %d/%d 성립\n", ntot - nbad, ntot)
end

# ===================================================================== zero-sum
if PART in ("zs", "all")
    include(joinpath(root, "network_generator.jl"))
    using .NetworkGenerator
    include(joinpath(root, "true_dro", "true_dro_data.jl"))
    include(joinpath(root, "true_dro", "true_dro_build_subproblem.jl"))
    println("="^90, "\n(ZS) zero-sum max-flow grid 3×3, S=3, β=0.4, ε̂=ε̃=0.2, λᵁ=10 — δ = ", DELTAS, "\n", "="^90)
    net = generate_grid_network(3, 3; seed=42)
    caps, _ = generate_capacity_scenarios_uniform_model(length(net.arcs), 3; seed=42)
    td = make_true_dro_data(net, caps, fill(1/3, 3), 0.2, 0.2; w=1.0, lambda_U=10.0, gamma=2, beta=0.4)
    ndz = nz_from_true_dro(td)
    xs = all_x(ndz)
    nxz = parse(Int, envf("CPL_ZS_NX", "0"))
    if nxz > 0 && nxz < length(xs)
        rng = MersenneTwister(0); xs = vcat([zeros(ndz.nx)], shuffle(rng, xs[2:end])[1:nxz-1])
    end
    @printf("  K=%d, 평가 x %d 개\n", ndz.nx, length(xs)); flush(stdout)
    tabz = run_table(ndz, xs; old=(nothing, nothing, td))
    check_props(ndz, tabz; label="ZS ")
    check_effective_radius(ndz, tabz, xs)
end

# ===================================================================== location
if PART in ("loc", "all")
    nd = make_location_instance(; coords=Symbol(envf("NZ_COORDS", "random")), reservation=Symbol(envf("NZ_RES", "pooled")),
                                S=parse(Int, envf("NZ_S", "3")), seed=parse(Int, envf("NZ_SEED", "4")))
    println("="^90, "\n(LOC) ", nd.name, "  θᵁ=", nd.thetaU, " λᵁ=", nd.lambdaU, " — δ = ", DELTAS, "\n", "="^90)
    xs = all_x(nd)
    tabl = run_table(nd, xs)
    check_props(nd, tabl; label="LOC ")
    check_effective_radius(nd, tabl, xs)
    # 리더 최적해 (전수): min_x qxᵀx + V*_δ(x)
    for δ in DELTAS
        vals = [(dot(nd.qx, x) + tabl[x][δ].kkt, findall(x .> 0.5)) for x in xs]
        sort!(vals; by=first)
        @printf("  δ=%-5s x*=%-10s 값=%.4f  (2등 %s %.4f)\n", string(δ), string(vals[1][2]), vals[1][1],
                string(vals[2][2]), vals[2][1])
    end
end
