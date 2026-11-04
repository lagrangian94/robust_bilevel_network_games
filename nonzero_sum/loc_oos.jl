#=
loc_oos.jl — 원고 6.1 의 Table loc-oos (기본 인스턴스의 out-of-sample, 2026-11-03).

결정: loc_insample.jl (LI_RES = LO_RES) 의 CSV 에서 모형마다 (quota = LO_QUOTA) 최적 x 를 읽는다 (KKT 전수 열거의 argmin).
  Nominal (0,0) / Data-DR (ε,0) / Coupled (ε,ε,δ) 각 δ / Coupled (product) (ε,ε,Inf)
평가 (모든 모형에 같은 난수): 진짜 분포 p ~ Dir(LO_CP · S · q̂), belief p̃ = (1 − τ) p + τ u, u ~ Dir(1), τ ~ U[0, 1].
  d_TV(p, p̃) 로 세 수준에 나눔 (수준마다 LO_N 쌍이 찰 때까지 추출):
    Close        d_TV ≤ δ₀ (LO_DELTA0 = 0.1, 결합 반경)
    Moderate     δ₀ < d_TV ≤ ε̃ (LO_EPS = 0.3)
    Misspecified d_TV > ε̃ (belief 가 D̃ 밖)
  retailer (follower) 는 p̃ 아래 최적 예약 h (2 단계 LP 의 꼭짓점 해, demo_value_nz.jl 의 follower_response 와 같음),
  recourse 는 그 h 에서 pessimistic (nz_recourse_values 의 φ: follower recourse 최적해 중 B 에 가장 불리한 값).
  B 의 이익 = −(q_Bᵀx + Σ_s p_s φ_s) (진짜 분포 아래 기대 이익). 함께 기록: 리더 목적 q_Bᵀx + CVaR_β^p[φ] (낮을수록 좋음).
요약 (수준 × 모형): 이익의 평균, CVaR (이익의 아래쪽 LO_TAIL 꼬리 평균), q05.

환경변수: LO_RES = pooled_store | pair (2026-11-04, 원고 6.1 모형 가치 표 = pair)   LO_CSV (기본 logs/loc_insample_sgb128_<LO_RES>_S3_seed1_eps0.30_beta0.40.csv)   LO_QUOTA = 150
          LO_EPS = 0.3   LO_DELTA0 = 0.1   LO_BETA = 0.4   LO_CP = 5   LO_N = 1000   LO_SEED = 2026   LO_TAIL = 0.05
          NZ_S = 3   NZ_SEED = 1   (loc_insample.jl 과 같은 인스턴스)
출력: logs/loc_oos/<CSV 이름>/draws.csv, summary.csv, 표준출력에 표
실행: julia nonzero_sum/loc_oos.jl
=#
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics, Distributions
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))

envf(k, d) = get(ENV, k, d); pf(k, d) = parse(Float64, envf(k, d)); pint(k, d) = parse(Int, envf(k, d))
S = pint("NZ_S", "3"); seed_inst = pint("NZ_SEED", "1")
ε = pf("LO_EPS", "0.3"); δ0 = pf("LO_DELTA0", "0.1"); β = pf("LO_BETA", "0.4"); quota = pf("LO_QUOTA", "150")
cp = pf("LO_CP", "5"); Npl = pint("LO_N", "1000"); seed = pint("LO_SEED", "2026"); tail = pf("LO_TAIL", "0.05")
RES = envf("LO_RES", "pooled_store"); RES in ("pair", "pooled_store") || error("LO_RES = pair | pooled_store")
csvin = envf("LO_CSV", joinpath(@__DIR__, "logs", @sprintf("loc_insample_sgb128_%s_S%d_seed%d_eps%.2f_beta%.2f.csv", RES, S, seed_inst, ε, β)))
isfile(csvin) || error("loc_insample CSV 없음: $csvin (LI_RES=$RES 로 loc_insample.jl 먼저)")
outdir = joinpath(@__DIR__, "logs", "loc_oos", splitext(basename(csvin))[1]); mkpath(outdir)

"CVaR_β^p[Z] (손실의 위쪽 꼬리) = max{νᵀZ : 0 ≤ ν ≤ p/(1−β), 1ᵀν = 1} (demo_value_nz.jl 과 같음)"
function cvar_upper(Z::AbstractVector, p::AbstractVector, β::Float64)
    ord = sortperm(Z; rev=true); cap = p ./ (1 - β); rest = 1.0; val = 0.0
    for s in ord
        m = min(cap[s], rest); val += m * Z[s]; rest -= m
        rest <= 1e-12 && break
    end
    return val
end

"retailer 최적 예약: belief p̃ 아래 max c0ᵀh + Σ_s p̃_s cᵀy_s (꼭짓점 해, demo_value_nz.jl 의 follower_response 와 같음)"
function follower_response(nd::NZData, x̄, pt)
    nh, ny, S, m = nz_nh(nd), nz_ny(nd), nd.S, nz_m(nd)
    fm = Model(GRB1); set_optimizer_attribute(fm, "Method", 0)
    @variable(fm, 0 <= hf[i = 1:nh] <= nd.hU[i]); @variable(fm, yf[1:ny, 1:S] >= 0)
    @constraint(fm, nd.W * hf .<= nd.wvec)
    for s in 1:S
        r0 = nz_rhs(nd, x̄, zeros(nh), s)
        for k in 1:m
            @constraint(fm, sum(nd.A[k, j] * yf[j, s] for j in 1:ny if nd.A[k, j] != 0) <=
                            r0[k] + (nd.hrow[k] > 0 ? nd.hcoef[k] * hf[nd.hrow[k]] : 0.0))
        end
    end
    @objective(fm, Max, dot(nd.c0, hf) + sum(pt[s] * nd.c[j] * yf[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0))
    optimize!(fm)
    termination_status(fm) == MOI.OPTIMAL || error("follower LP: $(termination_status(fm))")
    return clamp.(value.(hf), 0.0, nd.hU)
end

# 평가는 θ·λ·ε 와 무관 (정확한 LP). 데이터 (수요, 비용, 용량, 예약 한도) 만 씀 — loc_insample.jl 의 mk 와 같은 생성
nd = RES == "pair" ?
    make_location_instance(; coords=:sgb128, reservation=:pair, quota=quota, wres=300.0,
                           S=S, seed=seed_inst, eps_hat=0.0, eps_tilde=0.0, beta=β) :
    make_location_instance(; coords=:sgb128, reservation=:pooled, pool_limit=:store, quota=quota,
                           S=S, seed=seed_inst, eps_hat=0.0, eps_tilde=0.0, beta=β)
xstr(x) = "{" * join(findall(x .> 0.5), ",") * "}"
parsex(s) = (x = zeros(nd.nx); for t in split(strip(s, ['{', '}', '"']), ","; keepempty=false); x[parse(Int, t)] = 1.0; end; x)

# ---- 모형별 최적 x (CSV 의 argmin) ----
rows = Dict{Tuple{Float64,Float64,String},Vector{Tuple{String,Float64}}}()
for (i, line) in enumerate(eachline(csvin))
    i == 1 && continue
    m = match(r"^([^,]+),([^,]+),([^,]+),([^,]+),\"([^\"]*)\",([^,]+),([^,]+),", line)
    m === nothing && error("CSV 형식: $line")
    parse(Float64, m[1]) == quota || continue
    st = m[7]; st == "OPT" || error("CSV 의 KKT 가 OPTIMAL 이 아님 ($st): $line")
    push!(get!(rows, (parse(Float64, m[2]), parse(Float64, m[3]), m[4]), Tuple{String,Float64}[]), (m[5], parse(Float64, m[6])))
end
models = Tuple{String,Vector{Float64}}[]
for (key, name) in sort([(k, k[1] == 0 && k[2] == 0 ? "Nominal" : k[2] == 0 ? "Data-DR" :
                          (k[3] == "Inf" ? "Coupled (product)" : "Coupled δ=$(k[3])") * (k[2] == ε ? "" : " ε̃=$(k[2])"))
                         for k in keys(rows)]; by=last)
    r = rows[key]; length(r) == length(unique(first.(r))) || error("CSV 에 같은 x 가 두 번: $key")
    i = argmin(last.(r)); push!(models, (name, parsex(r[i][1])))
    @printf("  %-20s (ε̂=%.2f ε̃=%.2f δ=%s): x* = %s, 값 %.2f (x %d 개 중)\n", name, key..., r[i][1], r[i][2], length(r))
end
isempty(models) && error("quota=$quota 인 행이 CSV 에 없음")

# ---- 추출 (모든 모형에 같은 쌍) ----
levels = ["Close", "Moderate", "Misspecified"]
lev(tv) = tv <= δ0 ? 1 : tv <= ε ? 2 : 3
rng = MersenneTwister(seed)
pairs = [Tuple{Vector{Float64},Vector{Float64},Float64}[] for _ in 1:3]
ndraw = 0
while any(length.(pairs) .< Npl)
    global ndraw += 1
    ndraw > 1000 * Npl && error("수준을 채우지 못함: $(length.(pairs)) (LO_CP·ε·δ₀ 확인)")
    p = rand(rng, Dirichlet(cp .* S .* nd.q_hat)); u = rand(rng, Dirichlet(ones(S))); τ = rand(rng)
    pt = (1 - τ) .* p .+ τ .* u; tv = 0.5 * sum(abs.(p .- pt)); l = lev(tv)
    length(pairs[l]) < Npl && push!(pairs[l], (p, pt, tv))
end
@printf("loc_oos: SGB128 %s quota=%.0f S=%d seed=%d, ε̃=%.2f δ₀=%.2f, Dir(%.0f·S·q̂), 수준마다 %d 쌍 (추출 %d 번), β=%.2f\n",
        RES, quota, S, seed_inst, ε, δ0, cp, Npl, ndraw, β)
flush(stdout)

ux = unique(xstr(x) for (_, x) in models); xu = Dict(xstr(x) => x for (_, x) in models)
φcache = Dict{Tuple{String,Vector{Float64}},Vector{Float64}}()       # (x, 예약 h) → φ (꼭짓점 h 는 자주 반복)
io = open(joinpath(outdir, "draws.csv"), "w")
println(io, "level,draw,tv,model,x,profit,objective")
prof = Dict{Tuple{Int,String},Vector{Float64}}()
for l in 1:3, (n, (p, pt, tv)) in enumerate(pairs[l])
    res = Dict{String,Tuple{Float64,Float64}}()
    for k in ux
        x = xu[k]; h = round.(follower_response(nd, x, pt); digits=6)
        φ = get!(() -> nz_recourse_values(nd, x, h)[2], φcache, (k, h))
        res[k] = (-(dot(nd.qx, x) + dot(p, φ)), dot(nd.qx, x) + cvar_upper(φ, p, β))
    end
    for (name, x) in models
        pr, ob = res[xstr(x)]
        @printf(io, "%s,%d,%.5f,%s,%s,%.6f,%.6f\n", levels[l], n, tv, name, replace(xstr(x), "," => ";"), pr, ob)
        push!(get!(prof, (l, name), Float64[]), pr)
    end
end
close(io)

lowtail(v, a) = (k = max(1, ceil(Int, a * length(v))); mean(sort(v)[1:k]))
open(joinpath(outdir, "summary.csv"), "w") do f
    println(f, "level,model,x,mean,cvar_low$(tail),q05")
    @printf("\n%-20s %-10s", "model", "x*"); for L in levels; @printf(" | %-12s %9s %9s %9s", L, "mean", "CVaR", "q05"); end; println()
    for (name, x) in models
        @printf("%-20s %-10s", name, xstr(x))
        for l in 1:3
            v = prof[(l, name)]; mn, cv, q = mean(v), lowtail(v, tail), quantile(v, 0.05)
            @printf(" | %-12s %9.2f %9.2f %9.2f", "", mn, cv, q)
            @printf(f, "%s,%s,%s,%.6f,%.6f,%.6f\n", levels[l], name, replace(xstr(x), "," => ";"), mn, cv, q)
        end
        println()
    end
end
println("LOC_OOS_DONE $(outdir) (φ 캐시 $(length(φcache)))")
