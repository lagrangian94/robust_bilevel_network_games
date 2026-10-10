"""
h2h_oracle.jl — 같은 oracle 문제 (기본 인스턴스, x = {1,2}, θᵁ = circuit, WD) 에서 α-B&B vs Gurobi 전역 (Ω NonConvex) 의 정면 비교.
같은 시간·같은 스레드 수 (α-B&B worker 12 개 / Gurobi Threads 12).
환경변수: HH_S = "50,200"  HH_TL = 600  HH_SOLVERS = "gurobi,abb"  HH_ABB = "base" (base | bardx | dynbarc, 쉼표로 여러 개)  HH_EPS = 0.3  HH_DELTA = 0.1
실행: julia -t 14,1 nonzero_sum/h2h_oracle.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
TL = parse(Float64, envf("HH_TL", "600")); ε = parse(Float64, envf("HH_EPS", "0.3")); δ = pd(envf("HH_DELTA", "0.1"))
solvers = split(envf("HH_SOLVERS", "gurobi,abb"), ",")
abbkw = Dict("base" => NamedTuple(), "bardx" => (lp_method=2, lp_presolve=-1, defer_rows=true, lp_crossover=0),
             "dynbarc" => (dyn_threads=true, lp_method=2, lp_presolve=-1))
abbvars = split(envf("HH_ABB", "base"), ",")
for S in parse.(Int, split(envf("HH_S", "50,200"), ","))
    nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=1,
                                               eps_hat=ε, eps_tilde=ε, beta=0.4), δ)
    nd = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
    x̄ = zeros(nd.nx); x̄[[1, 2]] .= 1
    if "gurobi" in solvers
        O = build_nz_omega(nd; optimizer=GRB)
        set_optimizer_attribute(O.model, "Threads", 12)
        t = @elapsed g = nz_solve!(O, nd, x̄; time_limit=TL, gap=1e-3)
        @printf("RESULT S=%d gurobi        LB=%.4f UB=%.4f gap=%.3e status=%s (%.0fs)\n",
                S, g[:Fval], g[:bound], (g[:bound] - g[:Fval]) / max(1, abs(g[:Fval])), string(g[:status]), t)
        flush(stdout)
    end
    if "abb" in solvers
        for v in abbvars
            t = @elapsed r = nz_alpha_bnb(nd, x̄; nworkers=12, time_limit=TL, rel_gap=1e-3, verbose=false, abbkw[v]...)
            @printf("RESULT S=%d abb-%-8s LB=%.4f UB=%.4f gap=%.3e nodes=%d (%.0fs)\n",
                    S, v, r[:LB], r[:UB], (r[:UB] - r[:LB]) / max(1, abs(r[:LB])), r[:nodes], t)
            flush(stdout)
        end
    end
end
