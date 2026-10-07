"""
test_nz_alpha_bnb.jl — 비제로섬 α-B&B 검증 + λᵁ 민감도 (Gurobi NonConvex 와 비교).

  각 (λᵁ, x) 에서  α-B&B [LB, UB] / 시간,  Gurobi Ω [incumbent, bound] / 시간,  KKT V*(x) 를 비교.
  정확성: α-B&B 의 [LB, UB] 가 V* 를 포함해야 하고 (λᵁ ≥ 50), LB 는 실현 가능해 값이라 ≤ Ω* 여야 한다.

환경변수: NZ_LAMBDAS="1000,100,50", NZ_XS="3;4;3,4;1,3" (B 후보 번호, 빈 문자열 = ∅), NZ_TL=300, NZ_GAP=1e-4,
         NZ_SKIP_GRB=1 이면 Gurobi 생략, NZ_WORKERS=12.
실행: julia -t 14,1 nonzero_sum/test_nz_alpha_bnb.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))

function main()
    nd0 = make_location_instance(; S=parse(Int, get(ENV, "NZ_S", "3")), seed=parse(Int, get(ENV, "NZ_SEED", "4")),
                                 wres=parse(Float64, get(ENV, "NZ_WRES", "100")))
    lams = parse.(Float64, split(get(ENV, "NZ_LAMBDAS", "1000,100,50"), ","))
    xs = map(split(get(ENV, "NZ_XS", "3;4;3,4;1,3"), ";")) do str
        x = zeros(nd0.nx)
        isempty(strip(str)) || (x[parse.(Int, split(str, ","))] .= 1.0)
        x
    end
    tl = parse(Float64, get(ENV, "NZ_TL", "300"))
    gap = parse(Float64, get(ENV, "NZ_GAP", "1e-4"))
    nw = parse(Int, get(ENV, "NZ_WORKERS", "12"))
    skip_grb = get(ENV, "NZ_SKIP_GRB", "0") == "1"
    # NZ_LOCAL_WLS=1: 학술 WLS (동시 세션 2) 용 — worker 1 개가 테스트 공용 Env 를 쓰고 Ipopt 휴리스틱은 끔
    local_wls = get(ENV, "NZ_LOCAL_WLS", "0") == "1"
    local_wls && (nw = 1)
    bnb_kw = local_wls ? (envs=[GRB_ENV], heuristic=false) : NamedTuple()
    @printf("%s, workers=%d, 시간 제한 %.0fs, gap %.0e\n", nd0.name, nw, tl, gap)
    kkt = Dict(x => nz_kkt_value(nd0, x; optimizer=GRB)[:value] for x in xs)
    @printf("%-6s %-8s %10s | %11s %11s %7s %6s %6s | %11s %11s %7s | %s\n", "λᵁ", "x", "KKT V*",
            "BnB LB", "BnB UB", "BnB s", "nodes", "ipopt", "Grb inc", "Grb bd", "Grb s", "검사")
    flush(stdout)
    for λ in lams
        nd = nz_with_bounds(nd0; lambdaU=λ)
        O = skip_grb ? nothing : build_nz_omega(nd; optimizer=GRB)
        for x in xs
            r = nz_alpha_bnb(nd, x; nworkers=nw, time_limit=tl, rel_gap=gap, verbose=false, bnb_kw...)
            V = kkt[x]; t = 1e-6 * max(1, abs(V))
            ok = r[:LB] <= V + 10t && r[:UB] >= V - 10t      # [LB, UB] 가 V* 를 포함
            g = skip_grb ? nothing : (tg = @elapsed (gr = nz_solve!(O, nd, x; time_limit=tl, gap=gap)); (gr, tg))
            @printf("%-6g %-8s %10.4f | %11.4f %11.4f %7.1f %6d %6d | %11s %11s %7s | %s\n", λ,
                    string(findall(x .> 0.5)), V, r[:LB], r[:UB], r[:time], r[:nodes], r[:ipopt_calls],
                    g === nothing ? "-" : @sprintf("%.4f", g[1][:Fval]), g === nothing ? "-" : @sprintf("%.4f", g[1][:bound]),
                    g === nothing ? "-" : @sprintf("%.1f", g[2]), ok ? "ok" : "포함 안 됨!")
            flush(stdout)
        end
    end
end
main()
