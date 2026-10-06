"""
analyze_nz_menu_cache.jl — test_nz_belief_menu.jl 의 (A) 캐시 (Ω(x), 원소 (x, belief, σ)) 로
  (B) validity: 모든 (원소, x₂) 쌍에서 LP ≤ Ω(x₂)
  (C) 충분성: x₂ 마다 max_원소 LP = Ω(x₂), 그리고 그 max 를 달성하는 원소 수 (= x₂ 를 인증하는 데 필요한 원소)
  인증서 패턴: 시나리오별 고유 σˢ 개수, 고유 belief 개수 (html "크기: 실험 ①")
실행: julia nonzero_sum/analyze_nz_menu_cache.jl  (NZ_SEED, NZ_S, NZ_WRES)
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Serialization
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))

function main()
    nd = make_location_instance(; S=parse(Int, get(ENV, "NZ_S", "3")), seed=parse(Int, get(ENV, "NZ_SEED", "4")),
                                wres=parse(Float64, get(ENV, "NZ_WRES", "100")))
    Ωval, elems = deserialize(joinpath(root, "nonzero_sum", "logs", "cacheA_$(nd.name).jls"))
    X = sort(collect(keys(Ωval)); by=x -> (sum(x), reverse(x)))
    xs(x) = string(findall(x .> 0.5))
    B = build_nz_omega(nd; optimizer=GRB, belief_lp=true, cert_rows=:hrows)
    n = length(elems)
    V = zeros(n, length(X))
    for (i, (x1, bel, σ)) in enumerate(elems), (j, x2) in enumerate(X)
        nz_set_belief!(B, nd, bel, σ)
        V[i, j] = nz_solve!(B, nd, x2)[:Fval]
    end
    viol = [V[i, j] - Ωval[X[j]] for i in 1:n, j in 1:length(X)]
    imax = argmax(viol)
    @printf("(B) max(LP − Ω) = %.2e  (원소 x₁=%s 를 x₂=%s 에서)  → %s\n", viol[imax], xs(elems[imax[1]][1]), xs(X[imax[2]]),
            viol[imax] <= 1e-4 * max(1, maximum(abs, values(Ωval))) ? "유효" : "위반!")
    println("(C) x₂ 별: Ω, menu max, gap, max 달성 원소 수 (상대 1e-6 이내)")
    for (j, x2) in enumerate(X)
        mx = maximum(V[:, j]); tol = 1e-6 * max(1, abs(mx))
        @printf("    x=%-14s Ω=%11.4f  max=%11.4f  gap=%.1e  달성 %2d/%d  (자기 원소 LP %11.4f)\n", xs(x2), Ωval[x2], mx,
                Ωval[x2] - mx, count(V[:, j] .>= mx - tol), n, V[findfirst(e -> e[1] == x2, elems), j])
    end
    # 원소별: 어떤 x₂ 에서 exact 인가
    println("  원소별 exact 인 x₂ 수 (LP ≥ Ω − 1e-6 상대):")
    for (i, e) in enumerate(elems)
        ex = [xs(X[j]) for j in 1:length(X) if V[i, j] >= Ωval[X[j]] - 1e-6 * max(1, abs(Ωval[X[j]]))]
        @printf("    원소 x₁=%-14s exact at %2d x: %s\n", xs(e[1]), length(ex), join(ex, " "))
    end
    # 최소 menu (greedy set cover: 모든 x 를 exact 로 덮는 원소 수)
    cover = Set{Int}(); uncovered = Set(1:length(X))
    exact(i, j) = V[i, j] >= Ωval[X[j]] - 1e-6 * max(1, abs(Ωval[X[j]]))
    while !isempty(uncovered)
        i = argmax(i -> count(j -> exact(i, j), uncovered), 1:n)
        push!(cover, i); setdiff!(uncovered, [j for j in uncovered if exact(i, j)])
    end
    @printf("  greedy 최소 menu (모든 x 를 exact 로 덮음): %d 개 (원소 x₁ = %s)\n", length(cover),
            join([xs(elems[i][1]) for i in sort(collect(cover))], ", "))
    # 인증서 / belief 패턴
    S = nd.S
    for s in 1:S
        pats = unique([round.(e[3][:, s]; digits=4) for e in elems])
        @printf("  시나리오 %d: 고유 인증서 σˢ %d 개 / 원소 %d\n", s, length(pats), n)
    end
    bels = unique([Tuple(round.(vcat(e[2][:r], e[2][:d]); digits=4)) for e in elems])
    @printf("  고유 belief (r, d) %d 개 / 원소 %d\n", length(bels), n)
    for b in bels
        @printf("    r=%s d=%s\n", string(collect(b[1:S])), string(collect(b[S+1:end])))
    end
end
main()
