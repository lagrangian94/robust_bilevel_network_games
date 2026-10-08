"""
screen_coupling_decision.jl — location SGB128 pair 에서 순수 결합 효과가 리더 최적값·결정을 바꾸는 구성 탐색 (KKT 전수 열거).
ε̂ = ε̃ = ε 로 두면 유효 반경이 δ 와 무관 (Lemma geom) → rect 와 V*_δ 의 차이는 전부 순수 결합 효과.

각 (quota, β, ε) 에서 모든 x 의 qᵀx + V*_δ 를 δ ∈ SD_DELTAS 로 계산, δ 별 x* 와 rect 대비 감소를 출력.
환경변수: SD_QUOTAS = "100,150,200,300"  SD_BETAS = "0.2,0.4,0.6"  SD_EPS = "0.1,0.2,0.3"  SD_DELTAS = "Inf,0.1,0"  SD_TL = 120
          NZ_SEED (1) NZ_S (3)
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
TL = parse(Float64, envf("SD_TL", "120"))
all_x(nd) = [Float64.(collect(bits)) for bits in Iterators.product(fill(0:1, nd.nx)...)
             if sum(bits) <= nd.gamma && all(nd.x_allowed[i] || bits[i] == 0 for i in 1:nd.nx)]
deltas = pl("SD_DELTAS", "Inf,0.1,0")
nanargmin(v) = argmin(map(t -> isnan(t) ? Inf : t, v))   # 시간 제한 (NaN) 은 최적 후보에서 제외. 해당 구성은 TL 열로 표시됨
for quota in pl("SD_QUOTAS", "100,150,200,300"), β in pl("SD_BETAS", "0.2,0.4,0.6"), ε in pl("SD_EPS", "0.1,0.2,0.3")
    t0 = time()
    nd0 = make_location_instance(; coords=:sgb128, reservation=:pair, quota=quota, wres=300.0,
        S=parse(Int, envf("NZ_S", "3")), seed=parse(Int, envf("NZ_SEED", "1")), eps_hat=ε, eps_tilde=ε, beta=β)
    xs = all_x(nd0)
    val = Dict{Float64,Vector{Float64}}()
    nfail = 0
    for δ in deltas
        nd = nz_with_delta(nd0, δ)
        val[δ] = map(xs) do x
            r = try nz_kkt_value(nd, x; optimizer=GRB, time_limit=TL) catch; nothing end
            (r === nothing || r[:status] != MOI.OPTIMAL) && (nfail += 1)
            r === nothing ? NaN : dot(nd0.qx, x) + r[:value]
        end
    end
    vr = val[Inf]; ir = nanargmin(vr)
    @printf("quota=%3.0f β=%.1f ε=%.1f | rect x*=%-8s %10.4f", quota, β, ε, string(findall(xs[ir] .> 0.5)), vr[ir])
    for δ in deltas
        isfinite(δ) || continue
        v = val[δ]; i = nanargmin(v)
        @printf(" | δ=%-4s x*=%-8s %10.4f (x*_rect 에서 %7.2f)%s", string(δ), string(findall(xs[i] .> 0.5)), v[i],
                vr[ir] - v[ir], i != ir ? " ← 결정 변경" : "")
    end
    nx_bind = count(i -> vr[i] - val[minimum(deltas)][i] > 1e-4 * max(1, abs(vr[i])), eachindex(xs))
    @printf(" | binding x %d/%d | TL %d | %.0fs\n", nx_bind, length(xs), nfail, time() - t0)
    flush(stdout)
end
