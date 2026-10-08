"""
test_coupling_alpha_bnb.jl — 결합이 값을 바꾸는 점에서 α-B&B (Ω_δ 위의 전역 해법) 와 Gurobi 전역 Ω 가
KKT 독립 평가값을 [LB, UB] 로 감싸는지 확인 (Theorem minimax + Lemma certificate (iii) 의 수치 확인).

환경변수: NZ_COORDS (sgb128) NZ_RES (pair) NZ_SEED (1) NZ_S (3) NZ_EPSH NZ_EPST (0.2) NZ_BETA (0.4)
          TA_XS = "1,4;3,4"  TA_DELTAS = "Inf,0.1,0"  TA_TL = 1800  TA_GUROBI = 1
실행: julia -t 4,1 nonzero_sum/test_coupling_alpha_bnb.jl  (worker 1 + 휴리스틱; 프로세스는 하나만)
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
const HEUR_ENV = Gurobi.Env()   # α-B&B Ipopt 휴리스틱 전용 (기본 구성: 휴리스틱 켬)
include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
envf(k, d) = get(ENV, k, d)
pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
TL = parse(Float64, envf("TA_TL", "1800"))
nd0 = make_location_instance(; coords=Symbol(envf("NZ_COORDS", "sgb128")), reservation=Symbol(envf("NZ_RES", "pair")),
    S=parse(Int, envf("NZ_S", "3")), seed=parse(Int, envf("NZ_SEED", "1")),
    eps_hat=parse(Float64, envf("NZ_EPSH", "0.2")), eps_tilde=parse(Float64, envf("NZ_EPST", "0.2")),
    beta=parse(Float64, envf("NZ_BETA", "0.4")))
println(nd0.name, @sprintf("  ε̂=%.2f ε̃=%.2f β=%.2f λᵁ=%.1f θᵁ=%.2f", nd0.eps_hat, nd0.eps_tilde, nd0.beta, nd0.lambdaU, nd0.thetaU))
xs = map(t -> (x = zeros(nd0.nx); isempty(strip(t)) || (x[parse.(Int, split(t, ","))] .= 1.0); x),
         split(envf("TA_XS", "1,4;3,4"), ";"))
for x in xs, δ in pd.(split(envf("TA_DELTAS", "Inf,0.1,0"), ","))
    nd = nz_with_delta(nd0, δ)
    tk = @elapsed kk = nz_kkt_value(nd, x; optimizer=GRB, time_limit=TL)
    # α-B&B 는 튜닝된 기본 구성 그대로 (휴리스틱 켬, rel_gap 등 기본값). 이 PC 는 학술 WLS 라 worker 만 1 개
    # (worker 1 + 휴리스틱 1 = Env 2 개). 실험 머신에서는 TA_WORKERS=12 (Env 풀 기본).
    nw = parse(Int, envf("TA_WORKERS", "1"))
    bkw = nw == 1 ? (envs=[GRB_ENV, HEUR_ENV],) : NamedTuple()
    ta = @elapsed ab = nz_alpha_bnb(nd, x; nworkers=nw, time_limit=TL, verbose=false, bkw...)
    sc = max(1.0, abs(kk[:value]))
    ok = ab[:LB] <= kk[:value] + 1e-5 * sc && kk[:value] <= ab[:UB] + 1e-5 * sc
    @printf("  x=%-8s δ=%-4s KKT=%.6f [%s %.0fs]  α-B&B [LB %.6f, UB %.6f] nodes=%d %.0fs  포함=%s\n",
            string(findall(x .> 0.5)), string(δ), kk[:value], kk[:status], tk, ab[:LB], ab[:UB], ab[:nodes], ta, ok)
    if envf("TA_GUROBI", "1") == "1"
        O = build_nz_omega(nd; optimizer=GRB)
        tg = @elapsed r = nz_solve!(O, nd, x; time_limit=TL)   # gap 기본값 (1e-5)
        okg = r[:Fval] <= kk[:value] + 1e-5 * sc && kk[:value] <= r[:bound] + 1e-5 * sc
        @printf("  %-23s Gurobi Ω [inc %.6f, bd %.6f] %s %.0fs  포함=%s\n", "", r[:Fval], r[:bound], r[:status], tg, okg)
    end
    flush(stdout)
end
