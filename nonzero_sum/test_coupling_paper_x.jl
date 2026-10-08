"""
test_coupling_paper_x.jl — 원고 max-flow 인스턴스 (Abilene 등) 의 고정 x 에서 V*_δ(x) 를 세 방법으로.
  Ω (일반, Gurobi NonConvex), KKT (big-M 없는 독립 평가), old Ω (원고 build_true_dro_subproblem, delta_couple)
환경변수: CP_NET=abilene CP_BETA=0.7 CP_EPS=0.2 CP_EPSH (기본 CP_EPS) CP_EPST (기본 CP_EPS)
          CP_DELTAS="Inf,0.1"  CP_XS="5,27;26,27"  CP_TL=600  CP_OLD=1  CP_KKT=1
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "network_generator.jl"))
using .NetworkGenerator
include(joinpath(root, "true_dro", "true_dro_data.jl"))
include(joinpath(root, "true_dro", "true_dro_build_subproblem.jl"))
include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_paper_instances.jl"))
envf(k, d) = get(ENV, k, d)
pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
ε = parse(Float64, envf("CP_EPS", "0.2"))
ε̂ = parse(Float64, envf("CP_EPSH", string(ε))); ε̃ = parse(Float64, envf("CP_EPST", string(ε)))
β = parse(Float64, envf("CP_BETA", "0.7"))
net = envf("CP_NET", "abilene")
TL = parse(Float64, envf("CP_TL", "600"))
nd0, td = make_paper_maxflow(net; eps_hat=ε̂, eps_tilde=ε̃, beta=β)
@printf("%s K=%d S=%d w=%.4f ε̂=%.2f ε̃=%.2f β=%.2f card=%s\n", net, nd0.nx, nd0.S, nd0.meta[:w], ε̂, ε̃, β, string(nd0.meta[:omp_card]))
flush(stdout)
xs = map(split(envf("CP_XS", "5,27;26,27"), ";")) do t
    x = zeros(nd0.nx); isempty(strip(t)) || (x[parse.(Int, split(t, ","))] .= 1.0); x
end
for x in xs, δ in pd.(split(envf("CP_DELTAS", "Inf,0.1"), ","))
    nd = nz_with_delta(nd0, δ)
    O = build_nz_omega(nd; optimizer=GRB)
    t1 = @elapsed r = nz_solve!(O, nd, x; time_limit=TL, gap=1e-6)
    kv = NaN; t2 = 0.0
    if envf("CP_KKT", "1") == "1"
        t2 = @elapsed kk = nz_kkt_value(nd, x; optimizer=GRB, time_limit=TL)
        kv = kk[:value]
    end
    ov = NaN; t3 = 0.0
    if envf("CP_OLD", "1") == "1"
        om, ovv = build_true_dro_subproblem(td, x; optimizer=GRB, silent=true, delta_couple=δ)
        set_optimizer_attribute(om, "NonConvex", 2); set_optimizer_attribute(om, "MIPGap", 1e-6); set_time_limit_sec(om, TL)
        t3 = @elapsed info = solve_true_dro_subproblem!(om, ovv, td, x)
        ov = info[:Z0_val]
    end
    @printf("  x=%-10s δ=%-5s Ω=%.6f [%s bd %.6f %.0fs]  KKT=%.6f (%.0fs)  old=%.6f (%.0fs)\n", string(findall(x .> 0.5)),
            string(δ), r[:Fval], r[:status], r[:bound], t1, kv, t2, ov, t3)
    flush(stdout)
end
