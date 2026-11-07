# SGB15 지역 요인 수요 시험 (2026-11-06, 사용자: "시나리오 생성을 factor 모형처럼"). 저장소 코드는 안 고침.
# 지역 (도시 경도의 큰 간격 두 곳): 서부 7 (San Diego·San Jose·San Francisco·Vancouver·Seattle·Tucson·Sacramento),
#   중부 5 (San Antonio·Winnipeg·Tulsa·Wichita·Saint Louis), 동부 3 (Toronto·Washington·Toledo).
# 수요 (FA_DEM = region): ξ_js = μ_j (0.5 + Φ(ρ Z_{R(j),s} + √(1−ρ²) E_js)), Z, E ~ N(0,1) 독립, ρ = FA_RHO (0.9).
#   주변 분포는 Yanıkoğlu–Kuhn 과 같은 U[μ/2, 3μ/2], 같은 지역 시장의 잠재 상관 ρ². FA_DEM = iid 면 저장소 그대로 (대조군).
#   수요는 recourse 우변 u 에만 들어가고 상계 (θ, λ, ϖ) 에는 안 쓰여 (nz_data.jl 400~500 줄 확인) 인스턴스를 만든 뒤 u·meta[:xi] 만 바꿈.
# 분해 (FA_MODE = screen): 결정 378 개의 L0 = L(x; q̂, q̂) → 상위 FA_TOP 개. 진짜 표본 = TV 공 꼭짓점 S(S−1) 개
#   + 지역 방향 기울임 (지역 R 의 평균 수요 표준화 z, ± 방향, TV = FA_R 의 각 r). 점마다 Δ_T (follower 는 q̂), Δ_D (belief = 진짜), Δ_B (진짜 q̂, belief = 점).
#   pessimistic 후보는 sgb15_s10.jl 과 같음 (꼭짓점 해, w = p, 꼭짓점 φ 순서의 CVaR 가중치).
# 인자: FA_SEED, FA_S (10), FA_DEM (region | iid), FA_RHO, FA_TOP (12), FA_R ("0.15,0.3"), FA_OUT
const ROOT = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Statistics, Random, Distributions
const GRB_ENV = Gurobi.Env()
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(ROOT, "nonzero_sum", "nz_data.jl")); include(joinpath(ROOT, "nonzero_sum", "nz_omega.jl"))
include(joinpath(ROOT, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(@__DIR__, "fac_builder.jl"))   # REGION, RNAME, make_sgb15_fac

seed = parse(Int, get(ENV, "FA_SEED", "1")); S = parse(Int, get(ENV, "FA_S", "10")); dem = Symbol(get(ENV, "FA_DEM", "region"))
ρ = parse(Float64, get(ENV, "FA_RHO", "0.9")); TOP = parse(Int, get(ENV, "FA_TOP", "12"))
Rs = parse.(Float64, split(get(ENV, "FA_R", "0.15,0.3"), ",")); outdir = get(ENV, "FA_OUT", joinpath(@__DIR__, "logs", "sgb15_fac", "fa15")); β = 0.4; ε = 0.3
nd = make_sgb15_fac(; S=S, seed=seed, dem=dem, ρ=ρ, eps_hat=0.0, eps_tilde=0.0, beta=β)
nh, ny, m = nz_nh(nd), nz_ny(nd), nz_m(nd); q̂ = nd.q_hat
cand = findall(nd.x_allowed); X = Vector{Vector{Float64}}()
for k in 0:nd.gamma, c in Iterators.product(ntuple(_ -> cand, k)...)
    all(c[t] < c[t + 1] for t in 1:(k - 1)) || continue
    x = zeros(nd.nx); for i in c; x[i] = 1.0; end; push!(X, x)
end
xstr(x) = "{" * join(findall(x .> 0.5), ";") * "}"
function cvar_upper(Z, p, β)
    ord = sortperm(Z; rev=true); cap = p ./ (1 - β); rest = 1.0; val = 0.0
    for s in ord
        mm = min(cap[s], rest); val += mm * Z[s]; rest -= mm
        rest <= 1e-12 && break
    end
    return val
end
function cvar_w(Z, p, β)
    ord = sortperm(Z; rev=true); cap = p ./ (1 - β); rest = 1.0; w = zeros(length(p))
    for s in ord; w[s] = min(cap[s], rest); rest -= w[s]; end
    return w
end
function follower(x, pt)
    fm = Model(GRB1); set_optimizer_attribute(fm, "Method", 0)
    @variable(fm, 0 <= h[i = 1:nh] <= nd.hU[i]); @variable(fm, y[1:ny, 1:S] >= 0)
    @constraint(fm, nd.W * h .<= nd.wvec)
    for s in 1:S
        r0 = nz_rhs(nd, x, zeros(nh), s)
        for k in 1:m
            @constraint(fm, sum(nd.A[k, j] * y[j, s] for j in 1:ny if nd.A[k, j] != 0) <= r0[k] + (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0))
        end
    end
    fobj = @expression(fm, dot(nd.c0, h) + sum(pt[s] * nd.c[j] * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0))
    @objective(fm, Max, fobj); optimize!(fm)
    termination_status(fm) == MOI.OPTIMAL || error("follower LP: $(termination_status(fm))")
    Fs = objective_value(fm); hv = clamp.(value.(h), 0.0, nd.hU)
    @constraint(fm, fobj >= Fs - 1e-7 * max(1.0, abs(Fs)))
    return fm, h, y, hv
end
const CACHE = Dict{Tuple{Vector{Float64},Vector{Float64}},Tuple{Vector{Float64},Vector{Float64}}}()
rec(x, h) = get!(() -> nz_recourse_values(nd, x, h), CACHE, (x, round.(h; digits=6)))
function Lfm(x, fm, h, y, hv, p)
    cands = [hv]
    for w in (p, cvar_w(rec(x, hv)[2], p, β))
        @objective(fm, Max, sum(w[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0)); optimize!(fm)
        termination_status(fm) == MOI.OPTIMAL || error("pess LP: $(termination_status(fm))")
        push!(cands, clamp.(value.(h), 0.0, nd.hU))
    end
    return maximum(dot(nd.qx, x) + cvar_upper(rec(x, hc)[2], p, β) for hc in cands)
end
Lval(x, p, pt) = (r = follower(x, pt); Lfm(x, r..., p))
tvd(a, b) = 0.5 * sum(abs.(a .- b))

# 진짜 표본: 꼭짓점 + 지역 방향 기울임
ξ = nd.meta[:xi]; reg = get(nd.meta, :region, [REGION[c] for c in nd.meta[:cities][nd.meta[:nst]+1:end]])
E(i) = (e = zeros(S); e[i] = 1.0; e)
P = [(q̂ .+ ε .* (E(i) .- E(j)), "v$(i)_$(j)") for i in 1:S for j in 1:S if i != j]
for R in 1:3, sg in (1, -1)
    F = vec(mean(ξ[reg .== R, :]; dims=1)); z = sg .* (F .- mean(F)) ./ std(F)
    tl(θ) = (e = q̂ .* exp.(θ .* z); e ./ sum(e))
    for r in Rs
        lo, hi = 0.0, 1.0
        while tvd(tl(hi), q̂) < r; hi *= 2; hi > 1e4 && error("기울임 θ (R=$R r=$r)"); end
        for _ in 1:100; mid = (lo + hi) / 2; tvd(tl(mid), q̂) < r ? (lo = mid) : (hi = mid); end
        push!(P, (tl((lo + hi) / 2), "R$(R)$(sg > 0 ? "+" : "-")_$(r)"))
    end
end
t0 = time()
L0 = [Lval(x, q̂, q̂) for x in X]; ord = sortperm(L0)
@printf("FA seed=%d S=%d dem=%s ρ=%.2f: 시나리오별 지역 평균 수요 (서·중·동) =\n", seed, S, dem, ρ)
for R in 1:3; println("   ", RNAME[R], " ", join(round.(vec(mean(ξ[reg .== R, :]; dims=1)); digits=0), " ")); end
@printf("  L0 상위 %d 개 (%.0fs): %s\n", TOP, time() - t0, join(["$(xstr(X[i])) $(round(L0[i]; digits=1))" for i in ord[1:TOP]], ", ")); flush(stdout)
io = open(joinpath(outdir, "fa_$(dem)_s$(seed).csv.part"), "w")
println(io, "seed,dem,rank,x,L0,kind,tag,tv,delta")
for (rk, i) in enumerate(ord[1:TOP])
    x = X[i]; fm0, h0, y0, hv0 = follower(x, q̂)
    for (p, tag) in P
        for (kd, val) in (("T", Lfm(x, fm0, h0, y0, hv0, p)), ("D", Lval(x, p, p)), ("B", Lval(x, q̂, p)))
            @printf(io, "%d,%s,%d,%s,%.6f,%s,%s,%.4f,%.6f\n", seed, dem, rk, xstr(x), L0[i], kd, tag, tvd(p, q̂), val - L0[i])
        end
    end
    @printf("  [%2d] %-10s 끝 (%.0fs)\n", rk, xstr(x), time() - t0); flush(stdout)
end
close(io); mv(joinpath(outdir, "fa_$(dem)_s$(seed).csv.part"), joinpath(outdir, "fa_$(dem)_s$(seed).csv"); force=true)
println("FA_DONE seed=$seed dem=$dem ($(round(Int, time() - t0))s)")
