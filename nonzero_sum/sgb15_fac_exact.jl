# 지역 요인 수요 SGB15 (S = 10) 상위 결정의 정확한 평가 (2026-11-06). 결정은 fa15/fa_<dem>_s<seed>.csv 의 L0 순위 1~FE_TOP.
# 모형마다 고정 x 의 q_Bᵀx + V*(x) 를 nz_alpha_bnb_dr 로 (인스턴스·한계는 sgb15_insample.jl 과 같음: θᵁ = 회로 정확값, λᵁ = 40θᵁ):
#   Nominal (ε = 0), Data-DR (ε̂ = 0.3, ε̃ = 0), 결합 (ε̂ = ε̃ = 0.3, δ = 0.1). 결합의 최악 α 에서 (a, d) 는 nz_exact_value 와 같은 LP.
# 정확한 증폭 = V_C − V_D (결합이 Data-DR 보다 더 보는 피해 = belief 몫). 저장소 코드는 안 고침.
# 인자: FE_SEED, FE_DEM (region | iid), FE_TOP (3), FE_TL (1200), FE_NW (12), FE_OUT, FE_RANKS ("2,3" 처럼 일부 순위만, 기본 전부), FE_TAG (출력 파일 꼬리)
# (2026-11-06 판, 처음 판 sgb15_fac_exact.jl 을 대체): seed 2 {4;6;12} 결합에서 (a, d) 진단 LP 가 tolF = 1e-7 로 INFEASIBLE (α 가 최적성 경계에 걸림) → nz_exact_value 처럼
#   NUMERICAL_ERROR 면 NumericFocus 3·dual simplex, barrier 로 다시, INFEASIBLE 이면 tolF 를 1e-6, 1e-5 로 넓혀 다시 (쓴 tolF 를 CSV 에 기록).
#   셋 다 안 되면 오류. 값 (LB, UB) 은 α-B&B 그대로이고 이 LP 는 최악 (a, d) 를 보여 주기만 함.
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
seed = parse(Int, get(ENV, "FE_SEED", "1")); dem = Symbol(get(ENV, "FE_DEM", "region")); TOP = parse(Int, get(ENV, "FE_TOP", "3"))
TL = parse(Float64, get(ENV, "FE_TL", "1200")); nw = parse(Int, get(ENV, "FE_NW", "12")); outdir = get(ENV, "FE_OUT", SPD * "fe15")
RANKS = haskey(ENV, "FE_RANKS") ? parse.(Int, split(ENV["FE_RANKS"], ",")) : nothing; TAG = get(ENV, "FE_TAG", "")
function mk(eh, et, δ)
    nd = nz_with_delta(make_sgb15_fac(; S=10, seed=seed, dem=dem, eps_hat=eh, eps_tilde=et, beta=0.4), δ)
    θ = nz_theta_circuit_exact(nd)[1]; return nz_with_bounds(nd; thetaU=θ, lambdaU=40θ)
end
"nz_exact_value 의 LP (a·d 를 돌려줌)"
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
# 상위 결정 (fa15 CSV 의 rank)
if haskey(ENV, "FE_XS")     # 결정을 직접 줌 ("|" 로 구분, 원소는 "," 나 ";"), 순위 칸은 0·표본 L0 는 NaN
    top = [(0, "{" * join(split(strip(xs0, ['{', '}']), r"[,;]"), ";") * "}", NaN) for xs0 in split(ENV["FE_XS"], "|")]
else
rows = [split(l, ",") for l in readlines(SPD * "fa15/fa_$(dem)_s$(seed).csv")[2:end]]
top = unique([(parse(Int, r[3]), r[4], parse(Float64, r[5])) for r in rows if parse(Int, r[3]) <= TOP && (RANKS === nothing || parse(Int, r[3]) in RANKS)])
end
sort!(top)
models = (("N", mk(0.0, 0.0, Inf)), ("D", mk(0.3, 0.0, Inf)), ("C", mk(0.3, 0.3, 0.1)))
mkpath(outdir); io = open(joinpath(outdir, "fe_$(dem)_s$(seed)$(TAG).csv.part"), "w")
println(io, "seed,dem,rank,x,model,LB,UB,exact,time,a,d,tolF,alpha")
@printf("FE seed=%d dem=%s 상위 %d 개, TL=%.0fs, workers=%d\n", seed, dem, TOP, TL, nw); flush(stdout)
for (rk, xs, L0s) in top
    x̄ = zeros(models[1][2].nx); for t in split(strip(xs, ['{', '}']), ";"); x̄[parse(Int, t)] = 1.0; end
    vals = Dict{String,Float64}()
    for (nm, nd) in models
        t0 = time()
        r = nz_alpha_bnb_dr(nd, x̄; nworkers=nw, time_limit=TL, rel_gap=1e-3, verbose=false, ipopt_time=120.0)
        a, d, tF = nm == "C" && r[:α] !== nothing ? exact_ad(nd, x̄, r[:α]) : (Float64[], Float64[], NaN)
        qx = dot(nd.qx, x̄); vals[nm] = qx + r[:UB]
        @printf(io, "%d,%s,%d,%s,%s,%.6f,%.6f,%s,%.1f,%s,%s,%g,%s\n", seed, dem, rk, xs, nm, qx + r[:LB], qx + r[:UB], string(r[:is_exact]), time() - t0,
                join(round.(a; digits=4), ";"), join(round.(d; digits=4), ";"), tF, r[:α] === nothing ? "" : join(round.(r[:α]; digits=6), ";"))
        @printf("  [%d] %-10s %s: [%.2f, %.2f] exact=%s %.0fs%s\n", rk, xs, nm, qx + r[:LB], qx + r[:UB], string(r[:is_exact]), time() - t0,
                nm == "C" ? "  a의 큰 성분 " * join(["s$s:$(round(a[s]; digits=2))" for s in sortperm(a; rev=true)[1:3]], " ") : ""); flush(stdout)
    end
    b = abs(vals["N"]) / 100
    @printf("  [%d] %-10s 진짜 헤지 (D−N) %+.1f%%, belief 몫 (C−D) %+.1f%%  (표본 L0 %.1f)\n", rk, xs, (vals["D"] - vals["N"]) / b, (vals["C"] - vals["D"]) / b, L0s); flush(stdout)
end
close(io); mv(joinpath(outdir, "fe_$(dem)_s$(seed)$(TAG).csv.part"), joinpath(outdir, "fe_$(dem)_s$(seed)$(TAG).csv"); force=true)
println("FE_DONE seed=$seed dem=$dem")
