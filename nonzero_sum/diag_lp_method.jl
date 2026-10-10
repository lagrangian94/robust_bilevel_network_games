"""
diag_lp_method.jl — α-B&B 노드 완화 LP 의 Gurobi 풀이 방법·스레드 비교 (S=200 루트 LP 가 57 s 인데 worker 는 1 스레드).

1. 루트: 분리 수렴까지 (기본: dual simplex 1 스레드) 후 최종 행 집합의 LP 를 복사한 새 모델에서 cold start 로
   Method (0 primal, 1 dual, 2 barrier, 2 + crossover 끔, 3 concurrent) × Threads (1, LM_THREADS) × Presolve (끔, 자동) 시간.
2. 자식 노드 (warm start): 중점 분할로 하강하며 노드마다 같은 모델에서 dual simplex / primal simplex 1 스레드 시간.
환경변수: LM_S = "200"  LM_THREADS = 12  LM_DEPTH = 4
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf
const GRB_ENV = Gurobi.Env()
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
envf(k, d) = get(ENV, k, d)
nth = parse(Int, envf("LM_THREADS", "12")); depth = parse(Int, envf("LM_DEPTH", "4"))
mk1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
scratch = mktempdir()

for S in parse.(Int, split(envf("LM_S", "200"), ","))
    nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=1,
                                               eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
    nd = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
    x̄ = zeros(nd.nx); x̄[[1, 2]] .= 1
    hUp, _ = nz_presolve_hU(nd, x̄); nd = nz_with_hU(nd, hUp)
    nh = nz_nh(nd)
    P = _nz_build_sep(nd, x̄; optimizer=mk1)
    l = zeros(nh); u = copy(nd.hU)
    _nz_sep_set_box!(P, l, u)
    tr = @elapsed z0 = _nz_sep_solve!(P; maxrounds=30)
    m = P.O.model
    @printf("\nS=%d: 루트 분리 (dual simplex 1 스레드, 라운드 포함) %.1fs, 상한 %.4f, 변수 %d, 행 %d\n",
            S, tr, z0, num_variables(m), num_constraints(m; count_variable_in_set_constraints=false))
    for (meth, xo, name) in ((1, -1, "dual"), (0, -1, "primal"), (2, -1, "barrier"), (2, 0, "barrier 교차 끔"), (3, -1, "concurrent")),
        th in (1, nth), pre in (0, -1)
        meth == 3 && th == 1 && continue
        g, _ = copy_model(m); set_optimizer(g, () -> Gurobi.Optimizer(GRB_ENV)); set_silent(g)
        set_optimizer_attribute(g, "Threads", th); set_optimizer_attribute(g, "Method", meth); set_optimizer_attribute(g, "Presolve", pre)
        xo >= 0 && set_optimizer_attribute(g, "Crossover", xo)
        set_time_limit_sec(g, 600.0)
        optimize!(g); t = solve_time(g)
        st = termination_status(g)
        zz = st == MOI.OPTIMAL ? objective_value(g) : NaN
        @printf("  cold %-16s Threads %2d Presolve %2d: %7.1fs  %s  값 %.4f (차 %.1e)\n", name, th, pre, t, st, zz, abs(zz - z0) / max(1, abs(z0)))
        flush(stdout)
    end
    # 자식 노드 warm start: 같은 모델, dual vs primal (분리 없이 같은 행 집합)
    for meth in (1, 0)
        ll = zeros(nh); uu = copy(nd.hU)
        _nz_sep_set_box!(P, ll, uu); set_optimizer_attribute(m, "Method", 1); _nz_solve_lp!(m)
        set_optimizer_attribute(m, "Method", meth)
        ts = Float64[]
        for d in 1:depth
            i = argmax(uu .- ll); uu[i] = 0.5 * (ll[i] + uu[i])
            _nz_sep_set_box!(P, ll, uu)
            push!(ts, @elapsed _nz_solve_lp!(m))
        end
        @printf("  warm 자식 (하강 %d 노드) Method %d: %s s\n", depth, meth, join([@sprintf("%.1f", t) for t in ts], ", "))
        flush(stdout)
    end
end
