"""
diag_node_profile.jl — α-B&B 노드 LP 비용이 S 에 따라 어떻게 커지는지 (큰 S 에서 노드 수가 적은 원인).

S 마다 x̄ 에서 노드 LP (_nz_build_sep) 를 만들고
  - 크기: 변수·제약 수 (기본 + 정적 RLT), 분리 후 활성 RLT 행 수
  - 루트: 분리 라운드 수, LP 풀이 횟수, 시간 (LP 풀이 / 위반 계산)
  - 자식 노드: 중점 분할을 따라 깊이 10 까지 내려가며 노드마다 시간·라운드·행 추가 수
환경변수: NP_S = "3,20,50,200"  NP_X = "1,2"  NP_EPS = 0.3  NP_DELTA = 0.1  NP_DEPTH = 10
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
ε = parse(Float64, envf("NP_EPS", "0.3")); δ = pd(envf("NP_DELTA", "0.1")); depth = parse(Int, envf("NP_DEPTH", "10"))
xi = parse.(Int, split(envf("NP_X", "1,2"), ","))
mk1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)

"분리 루프를 계측하며 실행 (_nz_sep_solve! 와 같은 규칙)"
function profiled_solve!(P; maxrounds=30, maxadd=5000, tol=1e-6, stall_tol=1e-6, stall_rounds=2)
    tlp = 0.0; tviol = 0.0; rounds = 0; added = 0; zprev = Inf; nstall = 0; z = NaN
    for _ in 1:maxrounds
        rounds += 1
        tlp += @elapsed (z = _nz_solve_lp!(P.O.model))
        (isnan(z) || isinf(z)) && break
        tviol += @elapsed (V = _nz_sep_violations(P; tol=tol))
        isempty(V) && break
        nstall = (zprev - z <= stall_tol * max(1.0, abs(z))) ? nstall + 1 : 0
        zprev = z
        nstall >= stall_rounds && break
        sort!(V; by=first)
        for (_, i, j, islo) in V[1:min(maxadd, length(V))]
            _nz_sep_add!(P, i, j, islo, islo ? P.l[i] : P.u[i]); added += 1
        end
    end
    iters = MOI.get(P.O.model, MOI.SimplexIterations())
    return z, rounds, added, tlp, tviol, iters
end

for S in parse.(Int, split(envf("NP_S", "3,20,50,200"), ","))
    nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=1,
                                               eps_hat=ε, eps_tilde=ε, beta=0.4), δ)
    nd = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
    x̄ = zeros(nd.nx); x̄[xi] .= 1
    hUp, _ = nz_presolve_hU(nd, x̄); nd = nz_with_hU(nd, hUp)
    tb = @elapsed P = _nz_build_sep(nd, x̄; optimizer=mk1)
    m = P.O.model
    nv = num_variables(m); nc = num_constraints(m; count_variable_in_set_constraints=false)
    @printf("\nS=%d: 변수 %d, 제약 %d (분리 전), RLT 후보 행 %d, 빌드 %.1fs\n", S, nv, nc, length(P.rows), tb)
    l = zeros(nz_nh(nd)); u = copy(nd.hU)
    _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
    z, r, a, tlp, tv, it = profiled_solve!(P)
    @printf("  루트: 상한 %.2f  라운드 %d  추가 RLT 행 %d  LP %.2fs  위반계산 %.2fs  simplex 반복 %d\n", z, r, a, tlp, tv, it)
    flush(stdout)
    ttot = 0.0
    for d in 1:depth
        α̂ = value.(P.O.v[:α])
        i = argmax(u .- l)                                   # 가장 넓은 좌표를 중점에서 (아래쪽 자식으로 내려감)
        u[i] = 0.5 * (l[i] + u[i])
        _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
        t = @elapsed (z, r, a, tlp, tv, it = profiled_solve!(P))
        ttot += t
        @printf("  깊이 %2d: 상한 %10.2f  라운드 %2d  추가 %5d  LP %6.2fs  위반 %5.2fs  반복 %6d  활성 RLT %d\n",
                d, z, r, a, tlp, tv, it, count(P.active))
        flush(stdout)
    end
    @printf("  자식 노드 평균 %.2fs / 노드\n", ttot / depth)
end
