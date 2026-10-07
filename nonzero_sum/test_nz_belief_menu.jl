"""
test_nz_belief_menu.jl — 비제로섬에서 확장 belief LP cut 검증 (html "확장 belief" 절의 "검증" 항목).

  (A) exactness:  각 x 에서 전역 Ω 해의 (belief, 인증서 σ) 를 고정한 LP 값이 Ω(x) 와 같은가.
                  인증서 :recompute (html (b), follower LP simplex dual) / :solution (html (a), ϖ̂/r̂),
                  고정 범위 :hrows (bilinear 행만) / :all (ϖ 전체, html 그대로).
  (B) validity:   x₁ 에서 얻은 원소를 다른 x₂ 에서 풀면 Ω(x₂) 이하인가 (restriction → 항상 유효해야 함).
  (C) 충분성:     모든 x 에서 모은 menu 의 max 가 각 x 에서 Ω(x) 를 복원하는가.
  (D) 대조:       인증서 없이 belief 만 고정하면 α·ϖ 가 남아 LP 가 아님 → 그 문제를 Gurobi 로 풀어
                  값만 비교 (belief 고정으로 충분하지만 LP 가 아니라는 점 확인).
  (E) Benders:    표준 Benders vs belief-menu Benders (확장 belief) — 같은 x*, 값, oracle 횟수, menu 크기.
실행: julia nonzero_sum/test_nz_belief_menu.jl   (환경변수 NZ_S, NZ_SEED, NZ_LAMBDA)
"""

root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Serialization
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)

include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl"))

all_x(nd) = [Float64.(collect(bits)) for bits in Iterators.product(fill(0:1, nd.nx)...)
             if sum(bits) <= nd.gamma && all(nd.x_allowed[i] || bits[i] == 0 for i in 1:nd.nx)]
xs_str(x) = string(findall(x .> 0.5))

include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))

function main()
nd = make_location_instance(; S=parse(Int, get(ENV, "NZ_S", "3")), seed=parse(Int, get(ENV, "NZ_SEED", "4")),
                            wres=parse(Float64, get(ENV, "NZ_WRES", "100")),
                            reservation=Symbol(get(ENV, "NZ_RES", "pooled")),          # pooled | pair (h_ij)
                            quota=parse(Float64, get(ENV, "NZ_QUOTA", get(ENV, "NZ_WRES", "100"))),
                            lambdaU=(haskey(ENV, "NZ_LAMBDA") ? parse(Float64, ENV["NZ_LAMBDA"]) : nothing))
@printf("%s: θᵁ=%.1f  max π̂ᵁ=%.1f  λᵁ=%.1f\n", nd.name, nd.thetaU, maximum(nd.piLU), nd.lambdaU)
X = all_x(nd)

O = build_nz_omega(nd; optimizer=GRB)
Bh = build_nz_omega(nd; optimizer=GRB, belief_lp=true, cert_rows=:hrows)
Ba = build_nz_omega(nd; optimizer=GRB, belief_lp=true, cert_rows=:all)

# ---------------------------------------------------------------- (A)
println("\n(A) exactness: LP(belief(x), σ) at x  vs  Ω(x)")
@printf("  %-10s %12s %12s | %12s %12s %12s %12s\n", "x", "KKT V*(x)", "Ω(x)", "recomp/hrow", "recomp/all", "sol/hrow", "sol/all")
Ωval = Dict{Vector{Float64},Float64}()
elems = Tuple{Vector{Float64},Dict{Symbol,Any},Matrix{Float64}}[]
lpval(B, bel, σ, x) = (nz_set_belief!(B, nd, bel, σ); nz_solve!(B, nd, x)[:Fval])
# (A) 는 x 마다 전역 Ω 라 느림 → Ω 값과 원소를 캐시 (logs/*.jls, git 제외)
cache = joinpath(root, "nonzero_sum", "logs", "cacheA_$(nd.name).jls")
if isfile(cache) && get(ENV, "NZ_USE_CACHE", "1") == "1"
    Ωval, elems = deserialize(cache)
    println("  (캐시 사용: $cache)")
end
for x in (isempty(Ωval) ? X : Vector{Float64}[])
    res = nz_solve!(O, nd, x; time_limit=300)
    Ωval[x] = res[:Fval]
    raw = nz_read_belief(O, nd)
    bel = nz_clean_belief(raw, nd)
    σb = nz_follower_duals(nd, x, raw[:α]; lp_optimizer=GRB)[1]
    σa = nz_repair_cert(nd, raw[:ϖ] ./ reshape(max.(raw[:r], 1e-12), 1, nd.S); lp_optimizer=GRB)
    push!(elems, (x, bel, σb))
    v = [lpval(Bh, bel, σb, x), lpval(Ba, bel, σb, x), lpval(Bh, bel, σa, x), lpval(Ba, bel, σa, x)]
    kk = nz_kkt_value(nd, x; optimizer=GRB)[:value]
    @printf("  %-10s %12.4f %12.4f | %12.4f %12.4f %12.4f %12.4f   max gap %.1e\n", xs_str(x), kk, Ωval[x], v...,
            maximum(Ωval[x] .- v))
    flush(stdout)
end
isfile(cache) || serialize(cache, (Ωval, elems))

# ---------------------------------------------------------------- (B), (C)
println("\n(B) validity / (C) 충분성: 원소 (x₁ 에서 얻은 belief, σ:recompute) 를 x₂ 에서")
worst_viol = -Inf; worst_pair = nothing
menu_max = Dict(x => -Inf for x in X)
for (x1, bel, σ) in elems, x2 in X
    v = lpval(Bh, bel, σ, x2)
    viol = v - Ωval[x2]
    viol > worst_viol && (worst_viol = viol; worst_pair = (xs_str(x1), xs_str(x2)))
    menu_max[x2] = max(menu_max[x2], v)
end
@printf("  max(LP − Ω) over all pairs = %.2e  (pair x₁=%s → x₂=%s; > 0 이면 cut 무효)\n", worst_viol, worst_pair...)
println("  (C) max_menu LP(x) vs Ω(x):")
for x in X
    @printf("    x=%-10s Ω=%12.4f  menu max=%12.4f  gap=%.1e\n", xs_str(x), Ωval[x], menu_max[x], Ωval[x] - menu_max[x])
end
best = argmin(x -> dot(nd.qx, x) + Ωval[x], X)
@printf("  전수 열거 최적: x*=%s  q·x + V* = %.4f\n", xs_str(best), dot(nd.qx, best) + Ωval[best])

# ---------------------------------------------------------------- (D)
println("\n(D) 대조: 인증서 없이 belief 만 고정 (α·ϖ bilinear 남음, Gurobi NonConvex 로 풀이)")
Ob = build_nz_omega(nd; optimizer=GRB)       # 전역 Ω 에 belief 만 고정
for (x1, bel, _) in elems[1:min(3, end)]
    for f in (:a, :b, :r, :d, :e), s in 1:nd.S
        fix(Ob.v[f][s], bel[f][s]; force=true)
    end
    r = nz_solve!(Ob, nd, x1; time_limit=300)
    @printf("  x=%-10s belief-only (QCP) = %12.4f [%s]  Ω = %12.4f\n", xs_str(x1), r[:Fval], r[:status], Ωval[x1])
    flush(stdout)
end

# ---------------------------------------------------------------- (E)
println("\n(E) Benders 비교"); flush(stdout)
println("-- 표준 Benders (매 반복 전역 Ω)")
rs = nz_standard_benders(nd; optimizer=GRB, tol=1e-5)
println("-- belief-menu Benders (확장 belief, σ:recompute, hrows)")
rm = nz_belief_menu_benders(nd; optimizer=GRB, tol=1e-5)
@printf("\n  %-22s %-10s %12s %12s %6s %8s %6s %8s\n", "", "x*", "LB", "UB", "iter", "oracle", "menu", "wall")
for (nm, r) in (("standard", rs), ("belief-menu", rm))
    @printf("  %-22s %-10s %12.4f %12.4f %6d %8d %6s %7.1fs\n", nm, xs_str(r[:x]), r[:LB], r[:UB], r[:iters],
            r[:oracle_calls], string(get(r, :menu_size, "-")), r[:wall])
end
println("  belief-menu exactness 기록 (새 원소마다 LP@x̄ vs Ω(x̄)):")
for e in rm[:exact_log]
    @printf("    it %3d x=%-10s Ω=%12.4f LP=%12.4f gap=%.1e\n", e.iter, string(e.x), e.omega, e.lp, e.gap)
end
end  # main

main()
