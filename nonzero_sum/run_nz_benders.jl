"""
run_nz_benders.jl — location 인스턴스에서 Benders (belief-menu / 표준) 를 한 번 돌리는 실행 스크립트.

환경변수 (기본값):
  NZ_COORDS=sgb128 | random    NZ_RES=pair | pooled    NZ_QUOTA=100   NZ_WRES=300   NZ_S=3   NZ_SEED=1
  NZ_LAMBDA (기본: 인스턴스 기본값)   NZ_DISTRES (기본: 인스턴스 기본값)
  NZ_METHODS=menu,std    NZ_ORACLE=alpha_bnb | gurobi    NZ_TARGET=1 (belief-menu α-B&B 목표값 조기 종료)
  NZ_TOL=1e-4   NZ_ORACLE_TIME=600   NZ_BOOST=2400   NZ_WORKERS=12   NZ_LOCAL_WLS=1 (학술 WLS: worker 1, 휴리스틱 끔)
실행: julia -t 14,1 nonzero_sum/run_nz_benders.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))

envf(k, d) = get(ENV, k, d)
function main()
    kw = Dict{Symbol,Any}(:coords => Symbol(envf("NZ_COORDS", "sgb128")), :reservation => Symbol(envf("NZ_RES", "pair")),
        :quota => parse(Float64, envf("NZ_QUOTA", "100")), :wres => parse(Float64, envf("NZ_WRES", "300")),
        :S => parse(Int, envf("NZ_S", "3")), :seed => parse(Int, envf("NZ_SEED", "1")))
    haskey(ENV, "NZ_LAMBDA") && (kw[:lambdaU] = parse(Float64, ENV["NZ_LAMBDA"]))
    haskey(ENV, "NZ_DISTRES") && (kw[:dist_res] = parse(Float64, ENV["NZ_DISTRES"]))
    nd = make_location_instance(; kw...)
    oracle = Symbol(envf("NZ_ORACLE", "alpha_bnb"))
    nw = parse(Int, envf("NZ_WORKERS", "12"))
    local_wls = envf("NZ_LOCAL_WLS", "0") == "1"
    local_wls && (nw = 1)
    bkw = local_wls ? (envs=[GRB_ENV], heuristic=false) : NamedTuple()
    common = (optimizer=GRB, tol=parse(Float64, envf("NZ_TOL", "1e-4")), oracle_time=parse(Float64, envf("NZ_ORACLE_TIME", "600")),
              boost_time=parse(Float64, envf("NZ_BOOST", "2400")), oracle=oracle, nworkers=nw, bnb_kw=bkw)
    @printf("%s  θᵁ=%.2f λᵁ=%.1f  oracle=%s workers=%d\n", nd.name, nd.thetaU, nd.lambdaU, oracle, nw)
    isempty(nd.meta[:cities]) || println("  점포 (A 먼저): ", join(nd.meta[:cities][1:nd.meta[:nst]], " / "),
                                         "\n  고객: ", join(nd.meta[:cities][nd.meta[:nst]+1:end], " / "))
    flush(stdout)
    for meth in split(envf("NZ_METHODS", "menu,std"), ",")
        if meth == "menu"
            ts = oracle == :alpha_bnb && envf("NZ_TARGET", "1") == "1"
            r = nz_belief_menu_benders(nd; common..., target_stop=ts)
            @printf("belief-menu: %s x*=%s LB=%.4f UB=%.4f iter=%d oracle=%d menu=%d wall=%.1fs\n", r[:status],
                    string(findall(r[:x] .> 0.5)), r[:LB], r[:UB], r[:iters], r[:oracle_calls], r[:menu_size], r[:wall])
        else
            r = nz_standard_benders(nd; common...)
            @printf("standard: %s x*=%s LB=%.4f UB=%.4f iter=%d oracle=%d wall=%.1fs\n", r[:status],
                    string(findall(r[:x] .> 0.5)), r[:LB], r[:UB], r[:iters], r[:oracle_calls], r[:wall])
        end
        flush(stdout)
    end
end
main()
