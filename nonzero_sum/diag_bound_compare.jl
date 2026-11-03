"""
diag_bound_compare.jl — 고정된 x̄ 에서 V*(x̄) 의 상계를 세 방법으로 같은 시간·스레드에서 비교 (2026-10-22, BD-KKT 가 왜 느린지).

  KKT   : monolithic KKT 모형 (nz_kkt_value, BD-KKT 의 전역 단계와 같음, 시작해 q̂) + Gurobi
  Ω-GRB : 우리 재정식 Ω (build_nz_omega, θᵁ·λᵁ) + Gurobi NonConvex (BD-G 의 전역 단계)
  Ω-BnB : 우리 재정식 Ω + 우리 spatial B&B (nz_alpha_bnb_dr, Ours 의 전역 단계)
모두 V*(x̄) 의 상계 (λᵁ = 2000 은 이 인스턴스들에서 인증됨). 목표값 조기 종료 없이 시간 한도까지 (또는 gap 1e-3 까지) 푼다.

환경변수: DB_T = 300 (방법당 초)  DB_THREADS = 2 (Gurobi 스레드 = B&B worker 수)  DB_CASES = "L1:1:1,L1:2:3,L2:2:2" (설정:시드:열린 점포)
실행: julia -t 4,1 nonzero_sum/diag_bound_compare.jl   (B&B worker + 2 스레드)
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
envf(k, d) = get(ENV, k, d)
T = parse(Float64, envf("DB_T", "300")); NT = parse(Int, envf("DB_THREADS", "2"))
GRB() = Gurobi.Optimizer(GRB_ENV)
GRBT() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), NT); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
include(joinpath(root, "nonzero_sum", "nz_root_tighten.jl"))

const SETS = Dict("L1" => (15, 2, 3, 3), "L2" => (15, 3, 2, 2), "L3" => (20, 2, 3, 3), "L4" => (20, 3, 2, 2), "L5" => (25, 2, 3, 3))
function mk(L, seed)
    nloc, nA, nB, NB = SETS[L]; capv = 150.0 * (nloc - nA - nB) / nA
    nd = nz_with_delta(make_location_instance(; coords=:random, n_loc=nloc, nA=nA, nB=nB, NB=NB, S=10, seed=seed,
        bA=capv, bB=capv, qB=750.0, dlo=50.0, dhi=150.0, p=0.3, f=0.1, reservation=:pooled, pool_limit=:store, quota=300.0,
        eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
    return nz_with_bounds(nd; thetaU=nz_theta_circuit_exact(nd)[1], lambdaU=2000.0)
end
cases = [(String(a), parse(Int, b), parse.(Int, split(c, "+"))) for (a, b, c) in (split(s, ":") for s in split(envf("DB_CASES", "L1:1:1,L1:2:3,L2:2:2"), ","))]
@printf("상계 비교: 방법당 %.0f s, 스레드 %d\n", T, NT); flush(stdout)
for (L, seed, xo) in cases
    nd = mk(L, seed); x̄ = zeros(nd.nx); x̄[xo] .= 1.0
    @printf("\n=== %s seed %d x=%s (n_h=%d)\n", L, seed, string(xo), nz_nh(nd)); flush(stdout)
    # 1. Ω + 우리 B&B (참값 기준도 여기서)
    t = @elapsed (rb = Main.nz_alpha_bnb_dr(nd, x̄; nworkers=NT, time_limit=T, rel_gap=1e-3, verbose=false, ipopt_time=60.0))
    @printf("  Ω-BnB : 하계 %12.4f  상계 %12.4f  (루트 상계 %12.4f)  %6.1f s  노드 %d\n", rb[:LB], rb[:UB], rb[:root_UB], t, rb[:nodes]); flush(stdout)
    # 2. Ω + Gurobi
    O = build_nz_omega(nd; optimizer=GRBT); nz_set_objective!(O, nd, x̄)
    set_time_limit_sec(O.model, T); set_optimizer_attribute(O.model, "MIPGap", 1e-3)
    t = @elapsed optimize!(O.model)
    @printf("  Ω-GRB : 하계 %12.4f  상계 %12.4f  %6.1f s  %s\n", has_values(O.model) ? objective_value(O.model) : -Inf,
            objective_bound(O.model), t, string(termination_status(O.model))); flush(stdout)
    # 3. KKT + Gurobi (시작해 q̂)
    t = @elapsed (k = nz_kkt_value(nd, x̄; optimizer=GRBT, time_limit=T, gap=1e-3, allow_no_incumbent=true,
                                   start_beliefs=[copy(nd.q_hat)], lp_optimizer=GRBT))
    @printf("  KKT   : 하계 %12.4f  상계 %12.4f  %6.1f s  %s (시작값 %.4f)\n", k[:value], k[:bound], t, string(k[:status]), k[:start_val])
    flush(stdout)
end
