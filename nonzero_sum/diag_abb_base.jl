"""
diag_abb_base.jl — 기본 인스턴스 x = {1,2} 에서 α-B&B 상한이 느슨한 원인 진단: quota × δ 별로 α-B&B 를 같은 시간 돌려 LB/UB/노드 비교.
환경변수: DA_TL = 600   DA_QUOTAS = "100,150"   DA_DELTAS = "Inf,0.1"   DA_EPS = 0.3   DA_X = "1,2"
          DA_LAMBDAS = "100" (λᵁ 값들, 쉼표)   DA_SOLVERS = "abb" | "abb,gurobi"  (gurobi = Ω 를 Gurobi NonConvex 로 같은 시간)   DA_VERBOSE = 0 (1 이면 α-B&B 진행 로그, 60 s 마다)
실행: julia -t 14,1 nonzero_sum/diag_abb_base.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
TL = parse(Float64, envf("DA_TL", "600")); ε = parse(Float64, envf("DA_EPS", "0.3"))
xi = parse.(Int, split(envf("DA_X", "1,2"), ","))
for quota in pd.(split(envf("DA_QUOTAS", "100,150"), ",")), δ in pd.(split(envf("DA_DELTAS", "Inf,0.1"), ",")),
    λ in pd.(split(envf("DA_LAMBDAS", "100"), ","))
    nd = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=quota, wres=300.0, S=3, seed=1,
                                              eps_hat=ε, eps_tilde=ε, beta=0.4, lambdaU=λ), δ)
    x̄ = zeros(nd.nx); x̄[xi] .= 1
    k = nz_kkt_value(nd, x̄; optimizer=GRB, time_limit=300.0)
    solvers = split(envf("DA_SOLVERS", "abb"), ",")
    if "abb" in solvers
        t = @elapsed r = nz_alpha_bnb(nd, x̄; nworkers=12, time_limit=TL, verbose=envf("DA_VERBOSE", "0") == "1", log_every=60.0)
        @printf("RESULT quota=%g δ=%s x=%s KKT V*=%.4f | α-B&B LB=%.4f UB=%.4f gap=%.2e nodes=%s numerr=%s (%.0fs) | θᵁ=%.1f λᵁ=%.1f\n",
                quota, string(δ), string(xi), k[:value], r[:LB], r[:UB], (r[:UB] - r[:LB]) / max(1, abs(r[:LB])),
                string(get(r, :nodes, "-")), string(get(r, :numerr, "-")), t, nd.thetaU, nd.lambdaU)
        flush(stdout)
    end
    if "gurobi" in solvers
        O = build_nz_omega(nd; optimizer=GRB, silent=envf("DA_VERBOSE", "0") != "1")
        envf("DA_VERBOSE", "0") == "1" && set_optimizer_attribute(O.model, "DisplayInterval", 60)
        t = @elapsed g = nz_solve!(O, nd, x̄; time_limit=TL)
        @printf("RESULT quota=%g δ=%s x=%s KKT V*=%.4f | Gurobi Ω LB=%.4f UB=%.4f gap=%.2e status=%s (%.0fs)\n",
                quota, string(δ), string(xi), k[:value], g[:Fval], g[:bound], (g[:bound] - g[:Fval]) / max(1, abs(g[:Fval])),
                string(g[:status]), t)
        flush(stdout)
    end
end
