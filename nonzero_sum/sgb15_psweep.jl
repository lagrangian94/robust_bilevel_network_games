# 현장 프리미엄 p 를 키워 경합 고객 비중을 늘릴 때 정확한 증폭 (결합 − Data-DR) 이 따라 커지는가 (2026-11-07).
# 지역 요인 수요 SGB15 (S = 10) 한 시드. 결정 패널은 고정 (기본: s10 in-sample 세 결정 + 단일 출점 13 개).
# p 마다 예약비 f = PS_FRATIO · p (기본 1/4 = 기존 50/200) 로 둬서 follower 의 예약 문턱 (사용 확률 > f/p) 은 그대로, 경합 띠만 넓힌다.
# 경합 고객 = 가장 가까운 열린 B 점포와 가장 가까운 A 점포의 거리 차 < p 인 고객 (예약 위치에 따라 A·B 중 어디서 살지 바뀔 수 있음). 비중은 μ 가중.
# 평가는 sgb15_fac_exact2.jl 과 같음: nz_alpha_bnb_dr (θᵁ = 회로 정확값, λᵁ = 40θᵁ), Nominal·Data-DR·결합. 하한 (LB) 은 실제 반응의 값이라 λ 와 무관.
#   witness = incumbent α 에서 nz_exact_value (λ·θ 무관, ≤ V*) — UB − witness 가 작으면 λ 와 무관하게 인증.
# 인자: PS_SEED (10), PS_PS ("200,400,600,800"), PS_FRATIO (0.25), PS_XS ("|" 로 구분, 기본 패널), PS_TL (1200), PS_NW (12), PS_OUT, PS_TAG
root = dirname(@__DIR__)
const SPD = joinpath(@__DIR__, "logs", "sgb15_fac") * "/"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Distributions, Statistics
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
include(joinpath(root, "nonzero_sum", "nz_root_tighten.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(@__DIR__, "fac_builder.jl"))
seed = parse(Int, get(ENV, "PS_SEED", "10")); PS = parse.(Float64, split(get(ENV, "PS_PS", "200,400,600,800"), ","))
FR = parse(Float64, get(ENV, "PS_FRATIO", "0.25")); TL = parse(Float64, get(ENV, "PS_TL", "1200")); nw = parse(Int, get(ENV, "PS_NW", "12"))
outdir = get(ENV, "PS_OUT", SPD * "ps15"); TAG = get(ENV, "PS_TAG", "")
XS = [parse.(Int, split(strip(t, ['{', '}']), r"[,;]")) for t in split(get(ENV, "PS_XS",
        "{1,4,12}|{4,6,12}|{4,12,13}|" * join(["{$i}" for i in 1:13], "|")), "|")]
function mk(eh, et, δ, p)
    nd = nz_with_delta(make_sgb15_fac(; S=10, seed=seed, dem=:region, p=p, f=FR * p, eps_hat=eh, eps_tilde=et, beta=0.4), δ)
    θ = nz_theta_circuit_exact(nd)[1]; return nz_with_bounds(nd; thetaU=θ, lambdaU=40θ)
end
"경합 고객 비중 (μ 가중): |가장 가까운 열린 B − 가장 가까운 A| < p"
function contested(nd, xs, p)
    D = nd.meta[:dist]; nA = nd.meta[:nA]; ncu = nd.meta[:ncu]; μ = sgb15_mu(sgb15_kuhn().pop)
    dA = [minimum(D[1:nA, j]) for j in 1:ncu]
    dB = [isempty(xs) ? Inf : minimum(D[i+nA, j] for i in xs) for j in 1:ncu]
    return sum(μ[j] for j in 1:ncu if abs(dB[j] - dA[j]) < p; init=0.0) / sum(μ)
end
"nz_exact_value 의 LP 로 결합의 최악 (a, d) (sgb15_fac_exact2.jl 의 exact_ad 와 같음)"
function exact_ad(nd, x̄, α; tolF=1e-7)
    S, m, ny, nh = nd.S, nz_m(nd), nz_ny(nd), nz_nh(nd); nW = size(nd.W, 1)
    Q, φ = nz_recourse_values(nd, x̄, α); q, ε̂, ε̃, β = nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta
    M = Model(GRB1)
    @variable(M, 0 <= a[1:S] <= 1); @variable(M, b[1:S] >= 0); @variable(M, 0 <= r[1:S] <= 1)
    @constraint(M, sum(a) == 1); @constraint(M, sum(r) == 1)
    @constraint(M, [s = 1:S], a[s] - b[s] <= q[s]); @constraint(M, [s = 1:S], a[s] + b[s] >= q[s])
    @constraint(M, sum(b) <= 2ε̂); @constraint(M, [s = 1:S], (1 - β) * r[s] <= a[s])
    @variable(M, 0 <= d[1:S] <= 1); @variable(M, e[1:S] >= 0); @constraint(M, sum(d) == 1)
    @constraint(M, [s = 1:S], d[s] - e[s] <= q[s]); @constraint(M, [s = 1:S], d[s] + e[s] >= q[s]); @constraint(M, sum(e) <= 2ε̃)
    if nz_coupled(nd)
        @variable(M, cpl[1:S] >= 0)
        @constraint(M, [s = 1:S], a[s] - d[s] <= cpl[s]); @constraint(M, [s = 1:S], d[s] - a[s] <= cpl[s]); @constraint(M, sum(cpl) <= 2nz_delta(nd))
    end
    @variable(M, π̃[1:m, 1:S] >= 0); @variable(M, κ[1:nW] >= 0)
    @constraint(M, [j = 1:ny, s = 1:S], sum(nd.A[k, j] * π̃[k, s] for k in 1:m if nd.A[k, j] != 0) >= nd.c[j] * d[s])
    @constraint(M, [i = 1:nh], sum(nd.hcoef[k] * π̃[k, s] for k in 1:m, s in 1:S if nd.hrow[k] == i) + nd.c0[i] <= sum(nd.W[l, i] * κ[l] for l in 1:nW))
    B = [nz_rhs(nd, x̄, zeros(nh), s) for s in 1:S]; lhs = dot(nd.c0, α)
    @constraint(M, lhs + sum(Q[s] * d[s] for s in 1:S) >= sum(B[s][k] * π̃[k, s] for k in 1:m, s in 1:S if B[s][k] != 0) + dot(nd.wvec, κ)
                                                          - tolF * (1 + abs(lhs) + maximum(abs, Q)))
    @objective(M, Max, sum(φ[s] * r[s] for s in 1:S)); optimize!(M)
    st = termination_status(M)
    for (nf, meth) in ((3, 1), (0, 2))
        st == MOI.NUMERICAL_ERROR || break
        set_optimizer_attribute(M, "NumericFocus", nf); set_optimizer_attribute(M, "Method", meth); optimize!(M)
        st = termination_status(M)
    end
    st == MOI.OPTIMAL && return value.(a), value.(d), tolF
    (st == MOI.INFEASIBLE && tolF < 1e-5) && return exact_ad(nd, x̄, α; tolF=10tolF)
    error("exact LP: $st (tolF=$tolF)")
end
mkpath(outdir); fn = joinpath(outdir, "ps_s$(seed)$(TAG).csv"); io = open(fn * ".part", "w")
println(io, "seed,p,f,x,contested,model,LB,UB,exact,time,witness,a,d,tolF")
@printf("PS seed=%d p=%s f/p=%.3f 결정 %d 개, TL=%.0fs, workers=%d\n", seed, join(Int.(PS), ","), FR, length(XS), TL, nw); flush(stdout)
for p in PS
    models = (("N", mk(0.0, 0.0, Inf, p)), ("D", mk(0.3, 0.0, Inf, p)), ("C", mk(0.3, 0.3, 0.1, p)))
    @printf("p=%.0f f=%.1f θᵁ=%.3f\n", p, FR * p, models[3][2].thetaU); flush(stdout)
    for xs in XS
        x̄ = zeros(models[1][2].nx); x̄[xs] .= 1.0; xstr = "{" * join(xs, ";") * "}"
        cs = contested(models[1][2], xs, p); vals = Dict{String,Float64}()
        for (nm, nd) in models
            t0 = time()
            r = nz_alpha_bnb_dr(nd, x̄; nworkers=nw, time_limit=TL, rel_gap=1e-3, verbose=false, ipopt_time=120.0)
            qx = dot(nd.qx, x̄); vals[nm] = qx + r[:LB]
            wit = r[:α] === nothing ? -Inf : qx + _ev(nd, x̄, r[:α])     # nz_exact_value 는 (값, d) 또는 −Inf
            a, d, tF = nm == "C" && r[:α] !== nothing ? exact_ad(nd, x̄, r[:α]) : (Float64[], Float64[], NaN)
            @printf(io, "%d,%.0f,%.1f,%s,%.4f,%s,%.6f,%.6f,%s,%.1f,%.6f,%s,%s,%g\n", seed, p, FR * p, xstr, cs, nm, qx + r[:LB], qx + r[:UB],
                    string(r[:is_exact]), time() - t0, wit, join(round.(a; digits=4), ";"), join(round.(d; digits=4), ";"), tF); flush(io)
            @printf("  p=%.0f %-9s %s: [%.2f, %.2f] exact=%s witness=%.2f %.0fs\n", p, xstr, nm, qx + r[:LB], qx + r[:UB], string(r[:is_exact]), wit, time() - t0)
            flush(stdout)
        end
        @printf("  p=%.0f %-9s 경합 %.0f%%  진짜 헤지 (D−N) %+.2f, 증폭 (C−D) %+.2f\n", p, xstr, 100cs, vals["D"] - vals["N"], vals["C"] - vals["D"]); flush(stdout)
    end
end
close(io); mv(fn * ".part", fn; force=true)
println("PS_DONE seed=$seed")
