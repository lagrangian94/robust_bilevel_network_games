# p 스윕 메커니즘 진단 (2026-11-07). 결합의 최악 진짜 a 를 고정하고, 소매상 belief 가 q̂ (Data-DR) 일 때와 최악 d (결합) 일 때
# 예약 h (거점별) 와 진짜 a 아래 B 판매 기대값을 비교한다. h 는 belief 아래 follower 2SLP 의 최적해 중 a 아래 B 판매가 가장 적은 것 (pessimistic 근사:
# CVaR 가중 대신 a 기대값). 값 자체는 α-B&B 와 다를 수 있음 (진단용). 입력은 ps15/ps_s10.csv(.part) 의 결합 행 (a, d).
const SPD = joinpath(@__DIR__, "logs", "sgb15_fac") * "/"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Distributions
const GRB_ENV = Gurobi.Env()
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(@__DIR__, "nz_data.jl")); include(joinpath(@__DIR__, "fac_builder.jl"))
"belief d 아래 follower 최적 (최적값, 그 최적해 중 진짜 a 아래 B 판매 최소인 h, 시나리오별 B 판매·A 판매)"
function follower(nd, x̄, d, a)
    S, m, ny, nh = nd.S, nz_m(nd), nz_ny(nd), nz_nh(nd)
    M = Model(GRB1); @variable(M, h[1:nh] >= 0); @variable(M, y[1:ny, 1:S] >= 0)
    @constraint(M, nd.W * h .<= nd.wvec)
    for s in 1:S, k in 1:m
        rhs = nd.u[k, s] + (nd.xrow[k] > 0 ? nd.g[k, s] * x̄[nd.xrow[k]] : 0.0)
        @constraint(M, sum(nd.A[k, j] * y[j, s] for j in 1:ny if nd.A[k, j] != 0) <= rhs + (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0))
    end
    fobj = dot(nd.c0, h) + sum(d[s] * dot(nd.c, y[:, s]) for s in 1:S)
    @objective(M, Max, fobj); optimize!(M); termination_status(M) == MOI.OPTIMAL || error("follower LP: $(termination_status(M))")
    F = objective_value(M)
    # 같은 belief 최적값을 유지하면서 진짜 a 아래 B 판매 최소 (ℓ = −v·B flow 의 a 기대값 최대) — 각 시나리오 recourse 도 그 h 에서 follower 최적이어야 하므로
    # 시나리오별 follower 최적성은 d_s > 0 인 시나리오만 위 제약으로 보장됨 → 진단은 d_s = 0 시나리오의 y 를 따로 다시 풂
    @constraint(M, fobj >= F - 1e-7 * (1 + abs(F))); @objective(M, Max, sum(a[s] * dot(nd.ell, y[:, s]) for s in 1:S)); optimize!(M)
    termination_status(M) == MOI.OPTIMAL || error("pessimistic LP: $(termination_status(M))")
    hv = value.(h)
    Q, φ = nz_recourse_values(nd, x̄, hv)          # h 고정, 시나리오마다 follower 최적 + pessimistic
    return F, hv, φ
end
include(joinpath(@__DIR__, "nz_exact_eval.jl"))
fn = SPD * "ps15/ps_s10.csv"; isfile(fn) || (fn *= ".part")
want = split(get(ENV, "PM_X", "{4},{5},{6},{10},{12}"), ","); ps = parse.(Float64, split(get(ENV, "PM_P", "200,400"), ","))
for l in readlines(fn)[2:end]
    f = split(l, ","); p = parse(Float64, f[2]); (p in ps && f[6] == "C" && f[4] in want) || continue
    a = parse.(Float64, split(f[12], ";")); d = parse.(Float64, split(f[13], ";"))
    nd = make_sgb15_fac(; S=10, seed=10, dem=:region, p=p, f=0.25p, eps_hat=0.3, eps_tilde=0.3, beta=0.4)
    xs = parse.(Int, split(strip(f[4], ['{', '}']), ";")); x̄ = zeros(nd.nx); x̄[xs] .= 1.0
    cs = nd.meta[:cities]; nA = nd.meta[:nA]
    open_st = vcat(1:nA, [i + nA for i in xs])
    @printf("\n== p=%.0f %s (B: %s)\n", p, f[4], join(cs[[i + nA for i in xs]], ","))
    for (nm, bel) in (("belief q̂", nd.q_hat), ("belief d", d))
        F, hv, φ = follower(nd, x̄, bel, a)
        sales = -φ ./ nd.meta[:v]                        # B 판매량 (시나리오별)
        @printf("  %-9s 예약 %s | 진짜 a 아래 B 판매 기대 %.1f (시나리오: %s)\n", nm,
                join(["$(cs[i]) $(round(hv[i]; digits=1))" for i in open_st], ", "), dot(a, sales), join(round.(sales; digits=0), " "))
    end
end
