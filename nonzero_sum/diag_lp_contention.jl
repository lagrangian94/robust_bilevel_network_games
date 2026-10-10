"""
diag_lp_contention.jl — B&B 안에서 barrier 노드 LP 가 따로 풀 때 (S=200 루트 3 s) 보다 훨씬 느린 (노드당 ~59 s) 원인 분리.

1. 분리 라운드: 새 노드 LP 를 barrier (Presolve 자동, 1 스레드) 로 루트 분리 수렴까지 풀며 라운드별 시간·추가 행.
2. 동시 실행 경쟁: 루트 최종 LP 의 복사본을 n = 1, 6, 12 개 동시에 (각 1 스레드 barrier) 풀어 각 풀이 시간.
실행: julia -t 14,1 diag_lp_contention.jl     환경변수: LC_S = 200
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
S = parse(Int, get(ENV, "LC_S", "200"))
envs = [Gurobi.Env() for _ in 1:13]
mkb(env) = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=1,
                                           eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
nd = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
x̄ = zeros(nd.nx); x̄[[1, 2]] .= 1
hUp, _ = nz_presolve_hU(nd, x̄); nd = nz_with_hU(nd, hUp)
nh = nz_nh(nd)

# ---- 1. 분리 라운드 (barrier) ----
P = _nz_build_sep(nd, x̄; optimizer=mkb(envs[1]))
m = P.O.model
set_optimizer_attribute(m, "Method", 2); set_optimizer_attribute(m, "Presolve", -1); m.ext[:lp_method] = 2
_nz_sep_set_box!(P, zeros(nh), copy(nd.hU))
@printf("S=%d 분리 라운드 (barrier 1 스레드, Presolve 자동):\n", S)
ttot = 0.0
for rnd in 1:30
    t = @elapsed z = _nz_solve_lp!(m)
    global ttot += t
    V = _nz_sep_violations(P)
    @printf("  라운드 %2d: LP %.1fs (Gurobi %.1fs)  상한 %.4f  위반 %d  행 %d\n", rnd, t, solve_time(m), z, length(V),
            num_constraints(m; count_variable_in_set_constraints=false))
    flush(stdout)
    isempty(V) && break
    sort!(V; by=first)
    for (_, i, j, islo) in V[1:min(5000, length(V))]; _nz_sep_add!(P, i, j, islo, islo ? P.l[i] : P.u[i]); end
end
@printf("  합계 %.1fs\n", ttot)

# ---- 2. 동시 실행 경쟁 ----
if get(ENV, "LC_CONT", "1") == "1"
copies = [first(copy_model(m)) for _ in 1:12]
for (k, g) in enumerate(copies)
    set_optimizer(g, mkb(envs[k + 1]))
    set_optimizer_attribute(g, "Method", 2); set_optimizer_attribute(g, "Presolve", -1)
end
for n in (1, 6, 12)
    ts = zeros(n)
    @sync for k in 1:n
        Threads.@spawn (ts[k] = @elapsed optimize!(copies[k]))
    end
    @printf("동시 %2d 개 barrier (각 1 스레드): 평균 %.1fs, 최대 %.1fs\n", n, sum(ts) / n, maximum(ts))
    flush(stdout)
end
end

# ---- 3. 자식 노드 (child_rounds=1 과 같이 노드당 LP 1 회) 하강: barrier 시간·반복·교차 ----
LC3 = get(ENV, "LC_DIVE", "1") == "1"
if LC3
    l = zeros(nh); u = copy(nd.hU)
    for (xo, name) in ((-1, "교차 자동"), (0, "교차 끔"))
        set_optimizer_attribute(m, "Crossover", xo)
        ll = zeros(nh); uu = copy(nd.hU)
        for d in 1:10
            i = argmax(uu .- ll); (d % 2 == 0) ? (ll[i] = 0.5 * (ll[i] + uu[i])) : (uu[i] = 0.5 * (ll[i] + uu[i]))
            _nz_sep_set_box!(P, ll, uu)
            t = @elapsed z = _nz_solve_lp!(m)
            @printf("  [%s] 깊이 %2d: %.1fs  상태 %s  barrier 반복 %d  simplex 반복 %d  상한 %.3f\n", name, d, t,
                    termination_status(m), MOI.get(m, MOI.BarrierIterations()), MOI.get(m, MOI.SimplexIterations()), z)
            flush(stdout)
        end
    end
end
