"""
run_coupling_variants.jl — 결합 (δ) 아래에서 Benders 강화 요소 비교·검증.

변형 (CV_VARIANTS, 쉼표 구분): <menu|std>_<mw|nomw>_<bnb|grb>
  menu = belief cut (belief-menu), std = 표준 (restricted ISP cut = α 고정 LP, 또는 Gurobi Ω 해 cut)
  mw = Magnanti–Wong 강화, bnb = α-B&B (RLT) oracle, grb = Gurobi 전역 bilinear oracle
cut 유효성: 모든 변형에서 기록된 cut 을, 방문된 x 중 일부 (CV_NCHECK 개, 방문 빈도 순) 에서 KKT 정답 V*_δ(x) 와 비교
  (location 만. max-flow 는 KKT 가 너무 느려 생략하고 변형 간 최종값 일치만 봄).

환경변수
  CB_KIND = loc | maxflow  (loc: NZ_COORDS=sgb128 NZ_RES=pair NZ_SEED=1 NZ_S=3 NZ_EPSH NZ_EPST=0.2 NZ_BETA=0.4,
                            maxflow: CB_NET=abilene CB_BETA=0.7 CB_EPSH CB_EPST=0.2)
  CV_DELTA = 0.1   CV_VARIANTS = menu_mw_bnb,std_mw_bnb,menu_nomw_bnb,menu_mw_grb
  CV_TL = 1800 (변형별 전체 시간)  CV_ORACLE_TIME = 600  CV_BOOST = 3600  CV_TOL = 1e-4  (모두 nz_benders 기본값)
  CV_WORKERS = 1 (이 PC 학술 WLS: worker 1 + 휴리스틱. 실험 머신에서는 12)
  CV_CHECK = 1  CV_NCHECK = 6
실행: julia -t 4,1 nonzero_sum/run_coupling_variants.jl   (worker 12 면 -t 14,1). 프로세스는 하나만.
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
const HEUR_ENV = Gurobi.Env()   # α-B&B Ipopt 휴리스틱 전용 (기본 구성: 휴리스틱 켬)
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
    nd0, _ = make_paper_maxflow(envf("CB_NET", "abilene"); S=parse(Int, envf("CB_S", "10")), seed=parse(Int, envf("CB_SEED", "42")), beta=parse(Float64, envf("CB_BETA", "0.7")),
        eps_hat=parse(Float64, envf("CB_EPSH", "0.2")), eps_tilde=parse(Float64, envf("CB_EPST", "0.2")))
else
    nd0 = make_location_instance(; coords=Symbol(envf("NZ_COORDS", "sgb128")), reservation=Symbol(envf("NZ_RES", "pair")),
        S=parse(Int, envf("NZ_S", "3")), seed=parse(Int, envf("NZ_SEED", "1")),
        eps_hat=parse(Float64, envf("NZ_EPSH", "0.2")), eps_tilde=parse(Float64, envf("NZ_EPST", "0.2")),
        beta=parse(Float64, envf("NZ_BETA", "0.4")))
end
δ = pd(envf("CV_DELTA", "0.1"))
nd = nz_with_delta(nd0, δ)
# 튜닝된 기본 구성 그대로. 이 PC (학술 WLS) 에서는 worker 1 + 휴리스틱 = Env 2 개. 실험 머신은 CV_WORKERS=12.
nw = parse(Int, envf("CV_WORKERS", "1"))
bnbkw = nw == 1 ? (envs=[GRB_ENV, HEUR_ENV],) : NamedTuple()
@printf("%s  nx=%d S=%d ε̂=%.2f ε̃=%.2f β=%.2f δ=%s  workers=%d\n", nd.name, nd.nx, nd.S, nd.eps_hat, nd.eps_tilde, nd.beta,
        string(δ), nw)
flush(stdout)

results = []
for v in split(envf("CV_VARIANTS", "menu_mw_bnb,std_mw_bnb,menu_nomw_bnb,menu_mw_grb"), ",")
    meth, mws, orc = split(strip(v), "_")
    oracle = orc == "bnb" ? :alpha_bnb : :gurobi
    # tol·oracle_time·boost_time·oracle_gap 은 nz_benders.jl 기본값 (1e-4, 600, 3600, 1e-5). 바꿀 때만 환경변수.
    kw = (optimizer=GRB, tol=parse(Float64, envf("CV_TOL", "1e-4")), oracle_time=parse(Float64, envf("CV_ORACLE_TIME", "600")),
          boost_time=parse(Float64, envf("CV_BOOST", "3600")), oracle=oracle, nworkers=nw,
          bnb_kw=bnbkw, mw=(mws == "mw"),
          max_iter=100000, time_limit=parse(Float64, envf("CV_TL", "1800")), verbose=true)
    println("="^90, "\n변형 ", v, "\n", "="^90); flush(stdout)
    r = try
        meth == "menu" ? nz_belief_menu_benders(nd; kw..., target_stop=(oracle == :alpha_bnb)) : nz_standard_benders(nd; kw...)
    catch e
        ee = e isa TaskFailedException ? e.task.exception : e          # 중첩 task 오류를 꺼내서 보여줌
        println("!! 변형 ", v, " 실패: ", first(sprint(showerror, ee), 600)); flush(stdout); continue
    end
    kinds = Dict(k => count(c -> c.kind == k, r[:cuts]) for k in (:belief, :restricted, :omega))
    nmw = count(c -> c.mw, r[:cuts])
    @printf("→ %s %s x*=%s LB=%.6f UB=%.6f iter=%d oracle=%d cuts belief/restricted/omega=%d/%d/%d (MW %d) wall=%.0fs\n",
            v, r[:status], string(findall(r[:x] .> 0.5)), r[:LB], r[:UB], r[:iters], r[:oracle_calls],
            kinds[:belief], kinds[:restricted], kinds[:omega], nmw, r[:wall])
    haskey(r, :exact_log) && !isempty(r[:exact_log]) &&
        @printf("  belief exactness: max |Ω(x̄) − LP(x̄)| = %.2e  (음수 gap 은 oracle 이 목표값 조기 종료한 incumbent)\n",
                maximum(abs(e.gap) for e in r[:exact_log] if e.gap >= 0; init=0.0))
    isdefined(Main, :_NZ_NUMERR) && @printf("  α-B&B 노드 LP 수치 오류 누적: %d 회 (재시도로도 실패 %d 회)
", _NZ_NUMERR[], _NZ_NUMERR_FAIL[])
    push!(results, (v=v, r=r))
    flush(stdout)
end

# ---------------- cut 유효성 (location) ----------------
if envf("CV_CHECK", kind == "loc" ? "1" : "0") == "1"
    visits = Dict{Vector{Float64},Int}()
    for (_, r) in results, c in r[:cuts]
        visits[c.x] = get(visits, c.x, 0) + 1
    end
    for (_, r) in results; visits[r[:x]] = get(visits, r[:x], 0) + 1000; end   # 최종해는 항상 포함
    cand = first.(sort(collect(visits); by=last, rev=true))
    xs = cand[1:min(parse(Int, envf("CV_NCHECK", "6")), length(cand))]
    println("="^90, "\ncut 유효성: 확인 x = ", [findall(x .> 0.5) for x in xs])
    ref = Dict{Vector{Float64},Float64}()
    for x in xs
        t = @elapsed k = nz_kkt_value(nd, x; optimizer=GRB, time_limit=600.0)
        ref[x] = k[:status] == MOI.OPTIMAL ? k[:value] : k[:bound]          # 시간 제한이면 상한으로 (보수적 확인)
        @printf("  KKT V*_δ(%s) = %.6f [%s %.0fs]\n", string(findall(x .> 0.5)), ref[x], k[:status], t)
    end
    for (v, r) in results
        worst = -Inf; wk = nothing; ntot = 0
        for c in r[:cuts], x in xs
            val = c.intercept + dot(c.slope, x)
            viol = (val - ref[x]) / max(1.0, abs(ref[x]))
            ntot += 1
            viol > worst && (worst = viol; wk = (c.kind, c.mw, findall(x .> 0.5)))
        end
        @printf("  %-16s cut %4d 개 × x %d: max (cut(x) − V*(x))/|V*| = %+.2e  %s  %s\n", v, length(r[:cuts]), length(xs),
                worst, worst <= 1e-6 ? "유효" : "!! 위반", string(wk))
    end
end

println("="^90, "\n요약 (δ = ", δ, ")")
for (v, r) in results
    @printf("  %-16s %-9s x*=%-10s LB=%12.4f UB=%12.4f gap=%.2e iter=%4d oracle=%3d wall=%6.0fs\n", v, string(r[:status]),
            string(findall(r[:x] .> 0.5)), r[:LB], r[:UB], abs(r[:UB] - r[:LB]) / max(1, abs(r[:UB])), r[:iters],
            r[:oracle_calls], r[:wall])
end
