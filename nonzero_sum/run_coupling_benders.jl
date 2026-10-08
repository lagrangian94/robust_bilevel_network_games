"""
run_coupling_benders.jl — 종속 ambiguity set (δ 결합) 에서 Benders (belief-menu, α-B&B oracle) 를 δ 별로 실행.

환경변수
  CB_KIND = loc | maxflow
    loc    : NZ_COORDS (random) NZ_RES (pooled) NZ_QUOTA (100, 원고 w_i) NZ_WRES (300) NZ_SEED (4) NZ_S (3) NZ_EPSH NZ_EPST (0.2) NZ_BETA (0.4)
    maxflow: CB_NET (abilene) CB_BETA (0.7) CB_EPSH CB_EPST (0.2)  — 원고 실험 인스턴스 (nz_paper_instances.jl)
  CB_DELTAS = "Inf,0.1"   CB_METHOD = menu | std   CB_ORACLE = alpha_bnb | gurobi
  CB_TOL = 1e-4  CB_ORACLE_TIME = 600  CB_BOOST = 2400  CB_MAXITER = 300
  CB_CHECK = 1 : 끝난 뒤 x* 에서 KKT 독립 평가로 V*_δ(x*) 확인 (CB_CHECK_TL = 1800)
  CB_LOCAL_WLS = 0 (기본: CB_WORKERS=12, 휴리스틱 켬) | 1 이면 worker 1, 휴리스틱 끔 (학술 WLS 가 worker 를 제한하는 다른 PC 용)
  CB_MW = 1
실행: julia -t 14,1 nonzero_sum/run_coupling_benders.jl
실행 (worker 제한 PC): CB_LOCAL_WLS=1 julia -t 3,1 nonzero_sum/run_coupling_benders.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
envf(k, d) = get(ENV, k, d)
pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)

kind = envf("CB_KIND", "loc")
if kind == "maxflow"
    include(joinpath(root, "network_generator.jl"))
    using .NetworkGenerator
    include(joinpath(root, "true_dro", "true_dro_data.jl"))
    include(joinpath(root, "nonzero_sum", "nz_paper_instances.jl"))
    nd0, _ = make_paper_maxflow(envf("CB_NET", "abilene"); beta=parse(Float64, envf("CB_BETA", "0.7")),
        eps_hat=parse(Float64, envf("CB_EPSH", "0.2")), eps_tilde=parse(Float64, envf("CB_EPST", "0.2")))
else
    nd0 = make_location_instance(; coords=Symbol(envf("NZ_COORDS", "random")), reservation=Symbol(envf("NZ_RES", "pooled")),
        quota=parse(Float64, envf("NZ_QUOTA", "100")), wres=parse(Float64, envf("NZ_WRES", "300")),
        S=parse(Int, envf("NZ_S", "3")), seed=parse(Int, envf("NZ_SEED", "4")),
        eps_hat=parse(Float64, envf("NZ_EPSH", "0.2")), eps_tilde=parse(Float64, envf("NZ_EPST", "0.2")),
        beta=parse(Float64, envf("NZ_BETA", "0.4")))
end
@printf("%s  K/nx=%d S=%d ε̂=%.2f ε̃=%.2f β=%.2f θᵁ=%.1f λᵁ=%.1f\n", nd0.name, nd0.nx, nd0.S, nd0.eps_hat, nd0.eps_tilde,
        nd0.beta, nd0.thetaU, nd0.lambdaU)
flush(stdout)

oracle = Symbol(envf("CB_ORACLE", "alpha_bnb"))
local_wls = envf("CB_LOCAL_WLS", "0") == "1"        # 학술 WLS (worker 2 이상이면 Error 10009): worker 1, Env 재사용, 휴리스틱 끔
nw = local_wls ? 1 : parse(Int, envf("CB_WORKERS", "12"))
common = (optimizer=GRB, tol=parse(Float64, envf("CB_TOL", "1e-4")), oracle_time=parse(Float64, envf("CB_ORACLE_TIME", "600")),
          boost_time=parse(Float64, envf("CB_BOOST", "2400")), oracle=oracle, nworkers=nw,
          bnb_kw=local_wls ? (envs=[GRB_ENV], heuristic=false) : NamedTuple(), mw=envf("CB_MW", "1") == "1",
          max_iter=parse(Int, envf("CB_MAXITER", "300")))
@printf("oracle=%s workers=%d local_wls=%s mw=%s\n", oracle, nw, local_wls, common.mw)
summary = []
for δ in pd.(split(envf("CB_DELTAS", "Inf,0.1"), ","))
    nd = nz_with_delta(nd0, δ)
    println("="^80, "\nδ = ", δ, "\n", "="^80); flush(stdout)
    r = envf("CB_METHOD", "menu") == "menu" ?
        nz_belief_menu_benders(nd; common..., target_stop=(oracle == :alpha_bnb)) :
        nz_standard_benders(nd; common...)
    xs = findall(r[:x] .> 0.5)
    kv = NaN
    if envf("CB_CHECK", "1") == "1"
        kk = nz_kkt_value(nd, r[:x]; optimizer=GRB, time_limit=parse(Float64, envf("CB_CHECK_TL", "1800")))
        kv = dot(nd.qx, r[:x]) + kk[:value]
        @printf("  KKT 확인: qᵀx* + V*_δ(x*) = %.6f  (UB %.6f, 차 %+.2e, KKT %s)\n", kv, r[:UB], r[:UB] - kv, kk[:status])
    end
    @printf("δ=%-5s %s x*=%s LB=%.6f UB=%.6f iter=%d oracle=%d menu=%s wall=%.1fs\n", string(δ), r[:status], string(xs),
            r[:LB], r[:UB], r[:iters], r[:oracle_calls], string(get(r, :menu_size, "-")), r[:wall])
    push!(summary, (δ=δ, status=r[:status], x=xs, LB=r[:LB], UB=r[:UB], kkt=kv, iters=r[:iters], wall=r[:wall]))
    flush(stdout)
end
println("\n", "="^80, "\n요약")
for s in summary
    @printf("  δ=%-5s %-8s x*=%-12s LB=%.6f UB=%.6f KKT(x*)=%.6f iter=%d wall=%.0fs\n", string(s.δ), string(s.status),
            string(s.x), s.LB, s.UB, s.kkt, s.iters, s.wall)
end
