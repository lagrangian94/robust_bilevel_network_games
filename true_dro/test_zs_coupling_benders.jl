#=
test_zs_coupling_benders.jl — 원래 zero-sum 코드의 Benders 두 가지에 결합 (δ) 을 넣은 것 검증.

  인스턴스: 원고 생성 방식 grid3x4 (S=5, seed 1, β=0.7, ε̂=0.1, ε̃=0.3) — δ=0.05 에서 결합이 binding
            (KKT: x=∅ 45.68 → 43.60, x=[7,13] 45.68 → 42.01, nonzero_sum/find_zs_binding.jl)
  (1) true_dro_benders_optimize!  (원고 실험 구성: inexact, mini-Benders → 결합이면 α 고정 joint LP, MW, min-cut VI,
                                   α-B&B boost)
  (2) belief_menu_benders_optimize! (기본 구성: α-B&B oracle, local_first, menu MW, min-cut VI)
  기준: (3) 일반 코드 (nz, 결합 검증 완료) 의 belief-menu Benders 최적값, 그리고 각 x* 에서 KKT 독립 평가 (전수 열거는 안 함).
  이 PC (학술 WLS): α-B&B worker 1 + 휴리스틱 (Env 2 개 재사용). 실행: julia -t 4,1 true_dro/test_zs_coupling_benders.jl
  환경변수: ZB_DELTAS = "0.05,Inf"
=#
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env(); const HEUR_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "network_generator.jl")); using .NetworkGenerator
include(joinpath(root, "true_dro", "true_dro_data.jl"))
include(joinpath(root, "true_dro", "true_dro_build_omp.jl"))
include(joinpath(root, "true_dro", "true_dro_build_subproblem.jl"))
include(joinpath(root, "true_dro", "true_dro_build_isp_leader.jl"))
include(joinpath(root, "true_dro", "true_dro_build_isp_follower.jl"))
include(joinpath(root, "true_dro", "true_dro_benders.jl"))
include(joinpath(root, "true_dro", "belief_menu_benders.jl"))
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_paper_instances.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl"))      # 기준 (3) 용 (nz_alpha_bnb 도 함께 include)

nd0, td = make_paper_maxflow("grid3x4"; S=5, seed=1, beta=0.7, eps_hat=0.1, eps_tilde=0.3)
println("grid3x4 S=5 seed=1 β=0.7 ε̂=0.1 ε̃=0.3  K=", td.num_arcs, " interdictable=", count(td.interdictable_arcs[1:td.num_arcs]),
        " γ=", td.gamma, " w=", td.w)
all_x(nd) = [Float64.(collect(b)) for b in Iterators.product(fill(0:1, nd.nx)...)
             if sum(b) <= nd.gamma && all(nd.x_allowed[i] || b[i] == 0 for i in 1:nd.nx)]
pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)

for δ in pd.(split(get(ENV, "ZB_DELTAS", "0.05,Inf"), ","))
    println("="^90, "
δ = ", δ, "
", "="^90); flush(stdout)
    nd = nz_with_delta(nd0, δ)
    res = []
    # ---- (1) 원래 Benders (원고 실험 구성) ----
    t1 = @elapsed r1 = true_dro_benders_optimize!(td; mip_optimizer=GRB, nlp_optimizer=GRB, lp_optimizer=GRB,
        max_iter=1000, tol=5e-3, verbose=false, sub_time_limit=15.0, mini_benders=true, max_mini_benders_iter=5,
        strengthen_cuts=:mw, valid_inequality=:mincut, inexact=true, nonconvex_attr=("NonConvex" => 2),
        boost_nworkers=1, boost_envs=[GRB_ENV, HEUR_ENV], delta_couple=δ, wall_time_limit=1800.0)
    push!(res, ("(1) true_dro_benders", r1[:status], Float64.(r1[:x] .> 0.5), r1[:lower_bound], r1[:upper_bound], t1))
    @printf("  %s
", res[end]); flush(stdout)
    # ---- (2) belief-menu Benders (기본 구성) ----
    t2 = @elapsed r2 = belief_menu_benders_optimize!(td; mip_optimizer=GRB, nlp_optimizer=GRB, lp_optimizer=GRB,
        nworkers=1, bnb_envs=[GRB_ENV, HEUR_ENV], verbose=false, delta_couple=δ, wall_time_limit=1800.0)
    push!(res, ("(2) belief_menu", r2[:status], Float64.(r2[:x] .> 0.5), r2[:lower_bound], r2[:upper_bound], t2))
    @printf("  %s
", res[end]); flush(stdout)
    # ---- (3) 기준: 일반 코드 (nz, 검증 완료) belief-menu Benders 기본 구성 ----
    t3 = @elapsed r3 = nz_belief_menu_benders(nd; optimizer=GRB, nworkers=1, bnb_kw=(envs=[GRB_ENV, HEUR_ENV],),
        verbose=false, time_limit=1800.0)
    push!(res, ("(3) nz 기준", r3[:status], Float64.(r3[:x] .> 0.5), r3[:LB], r3[:UB], t3))
    @printf("  %s
", res[end]); flush(stdout)
    # ---- KKT 독립 평가: 각 x* 에서만 ----
    for xx in unique([r[3] for r in res])
        k = nz_kkt_value(nd, xx; optimizer=GRB, time_limit=600.0)
        @printf("  KKT V*_δ(%s) = %.6f [%s]
", string(findall(xx .> 0.5)), k[:value], k[:status]); flush(stdout)
    end
    println("  요약:")
    for (nm, st, xx, lb, ub, t) in res
        @printf("    %-22s %-9s x*=%-10s LB=%.6f UB=%.6f %.0fs
", nm, string(st), string(findall(xx .> 0.5)), lb, ub, t)
    end
    flush(stdout)
end
