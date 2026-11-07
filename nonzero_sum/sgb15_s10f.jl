# 사본 (2026-11-07): 지역 요인 수요 (fac_builder.jl 의 make_sgb15_fac) 와 logs/sgb15_fac/fi15/ 의 in-sample 결정 (N·D·C, sgb15_fac_insample.jl) 으로 바꿈.
#   나머지 (평가식, κ 설계, 난수) 는 sgb15_s10.jl 그대로. P 는 안 풀어서 뺌.
# SGB15 S = 10 확인 (2026-11-06): in-sample 로 정확히 푼 네 모형 결정 (logs/sgb15/S10_res10, 시드 1~10) 을 표본으로 평가.
# L(x; p, p̃) = q_Bᵀx + max_{h ∈ Ψ(x,p̃)} CVaR_β^p[φ] (원고 (1)). S = 10 이라 CVaR 가중치 꼭짓점을 다 셀 수 없어 pessimistic 후보는
#   LP 꼭짓점 해, w = p, w = 꼭짓점 해의 φ 순서로 탐욕적으로 채운 CVaR 가중치 의 세 개 (동률이 드물면 정확).
# 모드 SX_MODE:
#   decomp — 진짜 표본 = TV 공 꼭짓점 q̂ + ε(e_i − e_j) 90 개 + 공 안 무작위 SX_NT 개. 결정마다 Δ_T = L(p, q̂) − L0, Δ_D = L(p, p) − L0,
#            Δ_B = L(q̂, p) − L0 (belief 점 = 같은 표본). 공 안 최대의 하한 (표본).
#   kappa  — 진짜 = 공 안 무작위 SX_NT 개 (반경 r ~ U(0, ε), 방향 무작위), belief = 진짜 둘레 TV κ r 의 무작위 SX_M 방향 + p̃ = q̂.
# 인자: SX_SEED, SX_MODE, SX_X (결정, 기본: 네 모형의 x 를 로그에서), SX_K (κ), SX_NT, SX_M, SX_EPS (0.3), SX_RNG, SX_OUT
const ROOT = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Statistics, Random, Distributions
const GRB_ENV = Gurobi.Env()
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(ROOT, "nonzero_sum", "nz_data.jl")); include(joinpath(ROOT, "nonzero_sum", "nz_omega.jl"))
include(joinpath(ROOT, "nonzero_sum", "nz_exact_eval.jl"))
const SPD = joinpath(@__DIR__, "logs", "sgb15_fac") * "/"
include(joinpath(@__DIR__, "fac_builder.jl"))
β = 0.4; S = 10
seed = parse(Int, get(ENV, "SX_SEED", "8")); mode = get(ENV, "SX_MODE", "decomp"); κ = parse(Float64, get(ENV, "SX_K", "1"))
NT = parse(Int, get(ENV, "SX_NT", "60")); M = parse(Int, get(ENV, "SX_M", "6")); ε = parse(Float64, get(ENV, "SX_EPS", "0.3"))
rng = MersenneTwister(parse(Int, get(ENV, "SX_RNG", "7")) + 1000seed); outdir = get(ENV, "SX_OUT", joinpath(@__DIR__, "logs", "sgb15_fac", "sx10f"))
nd = make_sgb15_fac(; S=S, seed=seed, dem=:region, eps_hat=0.0, eps_tilde=0.0, beta=β)
nh, ny, m = nz_nh(nd), nz_ny(nd), nz_m(nd); q̂ = nd.q_hat
parsex(s) = (x = zeros(nd.nx); for t in split(strip(s, ['{', '}']), r"[;,]"; keepempty=false); x[parse(Int, t)] = 1.0; end; x)
LOG = SPD * "fi15"
function logx(model)
    f = joinpath(LOG, model == "nominal" ? "n_s$(seed).log" : "dc_s$(seed).log")
    l = filter(z -> startswith(z, "RESULT") && occursin(" model=$(model) ", z), readlines(f))
    length(l) == 1 || error("RESULT 줄 $(length(l)) 개: $model s$seed")
    return match(r" x=(\{[^}]*\})", l[1])[1], parse(Float64, match(r" UB=([-\d.eE+]+)", l[1])[1])
end
models = Dict(nm => logx(t) for (nm, t) in (("N", "nominal"), ("D", "datadr"), ("C", "coupled")))
xs = haskey(ENV, "SX_X") ? split(ENV["SX_X"], "|") : unique(models[k][1] for k in ("N", "D", "C"))

function cvar_upper(Z, p, β)
    ord = sortperm(Z; rev=true); cap = p ./ (1 - β); rest = 1.0; val = 0.0
    for s in ord
        mm = min(cap[s], rest); val += mm * Z[s]; rest -= mm
        rest <= 1e-12 && break
    end
    return val
end
function cvar_w(Z, p, β)            # 가중치 (cvar_upper 와 같은 탐욕)
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
tv(a, b) = 0.5 * sum(abs.(a .- b))
"TV 거리 1 인 무작위 방향 (합 0)"
randdir() = (d = randn(rng, S); d .-= mean(d); d ./ (0.5 * sum(abs.(d))))
"c 둘레 TV 거리 정확히 ρ 인 무작위 점: u ~ Dir(0.5·1) 와 섞은 (1−λ)c + λu, λ = ρ / d_TV(u, c) (단체 안 보장, 2026-11-06 κ=2 실패 뒤 바꿈)"
function randpt(c, ρ)
    ρ <= 1e-12 && return copy(c)
    for _ in 1:10_000
        u = [randgamma(0.5) for _ in 1:S]; u ./= sum(u)
        t = tv(u, c); t >= ρ && return (1 - ρ / t) .* c .+ (ρ / t) .* u
    end
    error("단체 안 점을 못 찾음 (ρ=$ρ)")
end
"Gamma(a, 1) (a < 1: Marsaglia–Tsang 의 a+1 표본 × U^(1/a))"
function randgamma(a)
    a < 1 && return randgamma(a + 1) * rand(rng)^(1 / a)
    d = a - 1 / 3; cc = 1 / sqrt(9d)
    while true
        z = randn(rng); v = (1 + cc * z)^3
        v > 0 && log(rand(rng)) < z^2 / 2 + d - d * v + d * log(v) && return d * v
    end
end

t0 = time()
@printf("SX seed=%d mode=%s κ=%.2f, 결정 %s, in-sample: %s\n", seed, mode, κ, join(xs, " "), join(["$k $(models[k][1]) UB $(round(models[k][2]; digits=1))" for k in ("N", "D", "C")], ", ")); flush(stdout)
tag = mode == "decomp" ? "dec_s$(seed)" : "kap_s$(seed)_k$(κ)"
io = open(joinpath(outdir, "sx_$(tag).csv.part"), "w")
println(io, "seed,mode,kappa,x,kind,i,r,tvb,L,L0")
for xstr in xs
    x = parsex(xstr); fm0, h0, y0, hv0 = follower(x, q̂); L0 = Lfm(x, fm0, h0, y0, hv0, q̂)
    if mode == "decomp"
        E(i) = (e = zeros(S); e[i] = 1.0; e)
        P = [q̂ .+ ε .* (E(i) .- E(j)) for i in 1:S for j in 1:S if i != j]
        append!(P, [randpt(q̂, ε * rand(rng)) for _ in 1:NT])
        for (k, p) in enumerate(P)
            r = tv(p, q̂)
            for (kd, val) in (("T", Lfm(x, fm0, h0, y0, hv0, p)), ("D", Lval(x, p, p)), ("B", Lval(x, q̂, p)))
                @printf(io, "%d,%s,%.2f,%s,%s,%d,%.5f,%.5f,%.6f,%.6f\n", seed, mode, κ, replace(xstr, "," => ";"), kd, k, r, kd == "T" ? r : 0.0, val, L0)
            end
        end
    else
        Random.seed!(rng, parse(Int, get(ENV, "SX_RNG", "7")) + 1000seed)    # 모든 결정에 같은 (p, p̃)
        for k in 1:NT
            p = randpt(q̂, ε * rand(rng)); r = tv(p, q̂)
            @printf(io, "%d,%s,%.2f,%s,qhat,%d,%.5f,%.5f,%.6f,%.6f\n", seed, mode, κ, replace(xstr, "," => ";"), k, r, r, Lfm(x, fm0, h0, y0, hv0, p), L0)
            for j in 1:M
                pt = κ * r <= 1e-12 ? p : randpt(p, κ * r)
                @printf(io, "%d,%s,%.2f,%s,%d,%d,%.5f,%.5f,%.6f,%.6f\n", seed, mode, κ, replace(xstr, "," => ";"), j, k, r, tv(pt, p), Lval(x, p, pt), L0)
                κ * r <= 1e-12 && break
            end
        end
    end
    @printf("  %s L0 %.3f (%.0fs)\n", xstr, L0, time() - t0); flush(stdout)
end
close(io); mv(joinpath(outdir, "sx_$(tag).csv.part"), joinpath(outdir, "sx_$(tag).csv"); force=true)
println("SX_DONE $tag ($(round(Int, time() - t0))s)")
