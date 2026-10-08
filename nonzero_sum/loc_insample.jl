"""
loc_insample.jl — 원고 6.1 의 Table loc-insample 과 Figure loc-delta 데이터 (SGB128 기본 인스턴스, KKT 전수 열거).

모든 x 의 qᵀx + V*(x) 를 독립 KKT 평가기 (big-M 없음) 로 구한다. 모형:
  Nominal (ε̂ = ε̃ = 0), Data-DR (ε̂ = ε, ε̃ = 0), Coupled (ε̂ = ε̃ = ε, δ), product (δ = Inf).
Figure: ε̂ = LI_EPS 고정, ε̃ ∈ LI_FIG_EPST, δ ∈ LI_FIG_DELTAS 에서 최적값·최적 결정 (|𝒫| 는 아직 계산 안 함).
δ ≥ ε̂ + ε̃ 는 product 와 같으므로 Inf 로 바꿔 한 번만 푼다.

환경변수
  LI_QUOTAS = "0,150"   (0 = 예약 없음, 원고 표의 w = 0 열)   LI_EPS = 0.3   LI_BETA = 0.4
  LI_TAB_DELTAS = "0.1,0"                         (표의 Coupled 행)
  LI_FIG_EPST = "0.1,0.2,0.3"   LI_FIG_DELTAS = "0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6"   (그림, quota = LI_FIG_QUOTA = 150)
  LI_TL = 300 (KKT 한 번의 시간 제한. 제한에 걸리면 incumbent 값, 표에 * 표시)
  NZ_SEED (1) NZ_S (3)
출력: 로그 (표준출력) + CSV logs/loc_insample_<tag>.csv (모든 x 의 값)
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
envf(k, d) = get(ENV, k, d)
pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
pl(k, d) = pd.(split(envf(k, d), ","))
TL = parse(Float64, envf("LI_TL", "300"))
ε = parse(Float64, envf("LI_EPS", "0.3"))
β = parse(Float64, envf("LI_BETA", "0.4"))
S = parse(Int, envf("NZ_S", "3")); seed = parse(Int, envf("NZ_SEED", "1"))
all_x(nd) = [Float64.(collect(bits)) for bits in Iterators.product(fill(0:1, nd.nx)...)
             if sum(bits) <= nd.gamma && all(nd.x_allowed[i] || bits[i] == 0 for i in 1:nd.nx)]
mk(quota, eh, et) = make_location_instance(; coords=:sgb128, reservation=:pair, quota=quota, wres=300.0,
                                           S=S, seed=seed, eps_hat=eh, eps_tilde=et, beta=β)
xstr(x) = "{" * join(findall(x .> 0.5), ",") * "}"

tag = @sprintf("sgb128_pair_S%d_seed%d_eps%.2f_beta%.2f", S, seed, ε, β)
csv = open(joinpath(@__DIR__, "logs", "loc_insample_$(tag).csv"), "w")
println(csv, "quota,eps_hat,eps_tilde,delta,x,value,status,time")

# (quota, ε̂, ε̃, δ) → 모든 x 의 (값, 상태). 같은 문제는 한 번만.
cache = Dict{NTuple{4,Float64},Vector{Tuple{Float64,Symbol}}}()
function evaluate(quota, eh, et, δ)
    δ = δ >= eh + et ? Inf : δ
    key = (quota, eh, et, δ)
    haskey(cache, key) && return cache[key]
    nd0 = mk(quota, eh, et); nd = nz_with_delta(nd0, δ)
    res = map(all_x(nd0)) do x
        t0 = time()
        r = try nz_kkt_value(nd, x; optimizer=GRB, time_limit=TL) catch e; (println("  KKT 오류 ", xstr(x), ": ", e); nothing) end
        v = r === nothing ? NaN : dot(nd0.qx, x) + r[:value]
        st = r === nothing ? :ERROR : (r[:status] == MOI.OPTIMAL ? :OPT : :TL)
        @printf(csv, "%g,%g,%g,%s,\"%s\",%.6f,%s,%.1f\n", quota, eh, et, string(δ), xstr(x), v, st, time() - t0)
        (v, st)
    end
    flush(csv)
    return cache[key] = res
end
xs = all_x(mk(150.0, ε, ε))
function best(res)
    vals = [isnan(v) ? Inf : v for (v, _) in res]
    i = argmin(vals)
    return xs[i], res[i][1], res[i][2]
end
fmtv(v, st) = @sprintf("%.2f%s", v, st == :OPT ? "" : "*")

# ---- Table loc-insample ----
quotas = pl("LI_QUOTAS", "0,150")
models = [("Nominal", 0.0, 0.0, Inf), ("Data-DR", ε, 0.0, Inf)]
for δ in pl("LI_TAB_DELTAS", "0.1,0"); push!(models, ("Coupled", ε, ε, δ)); end
push!(models, ("Coupled (product)", ε, ε, Inf))
println("="^100)
@printf("Table loc-insample: SGB128 pair, S=%d, seed=%d, β=%.2f, ε=%.2f, KKT TL=%.0fs (* = 시간 제한)\n", S, seed, β, ε, TL)
println("="^100)
@printf("%-18s %5s %5s %6s", "model", "ε̂", "ε̃", "δ"); for w in quotas; @printf(" | w=%-4g %-10s %10s", w, "x*", "value"); end; println()
for (name, eh, et, δ) in models
    @printf("%-18s %5.2f %5.2f %6s", name, eh, et, isfinite(δ) ? @sprintf("%.2f", δ) : "Inf")
    for w in quotas
        x, v, st = best(evaluate(w, eh, et, δ))
        @printf(" | %6s %-10s %10s", "", xstr(x), fmtv(v, st))
    end
    println(); flush(stdout)
end

# ---- Figure loc-delta 데이터 ----
fq = parse(Float64, envf("LI_FIG_QUOTA", "150"))
fdel = pl("LI_FIG_DELTAS", "0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6")
println("\n", "="^100)
@printf("Figure loc-delta: quota=%g, ε̂=%.2f, 행 = ε̃, 열 = δ, 칸 = x* 와 qᵀx* + V*\n", fq, ε)
println("="^100)
for et in pl("LI_FIG_EPST", "0.1,0.2,0.3")
    @printf("ε̃=%.2f\n", et)
    for δ in vcat(fdel, Inf)
        x, v, st = best(evaluate(fq, ε, et, δ))
        nTL = count(r -> r[2] != :OPT, evaluate(fq, ε, et, δ))
        @printf("  δ=%-5s x*=%-8s %10s%s\n", string(δ), xstr(x), fmtv(v, st), nTL > 0 ? "  (시간 제한 x $nTL 개)" : "")
        flush(stdout)
    end
end
close(csv)
println("CSV: logs/loc_insample_$(tag).csv")
