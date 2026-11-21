# belief 고정 비관적 리더 비용 V_b(c) 의 단조성·부분가법성 대규모 검증 (2026-11-21).
#   V_b(c) = max Σ_s r_s ℓᵀy_s  s.t. (h, y) 가 belief d 에서 follower 최적 (2 단계 LP), 용량 벡터 c.
#   검사: (M1) 거점 추가 V(T ∪ j) ≤ V(T),  (M2) 크기 증가 V(c + Δe_j) ≤ V(c),
#         (S1) V(a ∪ b) ≥ V(a) + V(b),  (S2) V(abc) ≥ V(ab) + V(c) (세 분할),  (S3) V(abc) ≥ V(a) + V(b) + V(c),
#         (S4) 용량 증분: V(c + Δ e_j) ≥ V(c) + V_j(Δ) (V_j(Δ) = 거점 j 만 용량 Δ)
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics
const GRB_ENV = Gurobi.Env()
include(joinpath(root, "nonzero_sum", "loc_tkp.jl"))
envf(k, d) = get(ENV, k, d)
GEOS = parse.(Int, split(envf("GEOS", "1"), ",")); NBEL = parse(Int, envf("NBEL", "3")); NTRI = parse(Int, envf("NTRI", "200")); TOLR = 1e-6
out = open(joinpath(@__DIR__, "msa_$(envf("TAG", "x")).tsv"), "w")
println(out, "geo\tbelief\ttest\tn\tviol\tworst_rel")
for g in GEOS
    nd = make_loc_tkp(; g=g, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
    MM = nd.meta; nA, nB, K, S = MM[:nA], MM[:nB], MM[:K], nd.S; tiers = MM[:tiers]; qh = nd.q_hat; bx = MM[:box]
    m, ny, nh = nz_m(nd), nz_ny(nd), nz_nh(nd); A = nd.A
    rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m]
    F = Model(() -> (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o))
    @variable(F, 0 <= h[i=1:nh] <= nd.hU[i]); @constraint(F, nd.W * h .<= nd.wvec); @variable(F, y[1:ny, 1:S] >= 0)
    @constraint(F, con[k=1:m, s=1:S], sum(a * y[j, s] for (j, a) in rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0) <= nd.u[k, s])
    capk = [MM[:rcap](nA + st) for st in 1:nB]
    optc = @constraint(F, 0.0 * h[1] <= 1e30)
    function V(cap::Vector{Float64}, d, r)
        for st in 1:nB, s in 1:S; set_normalized_rhs(con[capk[st], s], nd.u[capk[st], s] + cap[st]); end
        CF = -(dot(nd.c0, h) + sum(d[s] * nd.c[j] * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0))
        CL = sum(r[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0)
        delete(F, optc)
        @objective(F, Min, CF); optimize!(F); vf = objective_value(F)
        global optc_ = @constraint(F, CF <= vf + 1e-7 * abs(vf))
        @objective(F, Max, CL); optimize!(F); v = objective_value(F)
        optc = optc_
        return v
    end
    # belief: 명목 + DRO 극단 (TV ∩ 상자 안의 d, CVaR 가중 r: p ≈ d 를 δ 안에서 흔들고 p/(1−β) 를 무작위 순서로 채움)
    rng = MersenneTwister(1000 + g)
    inset(dd) = all(dd .>= bx.dl .- 1e-12) && all(dd .<= bx.du .+ 1e-12) && sum(abs.(dd .- qh)) <= 2nd.eps_tilde + 1e-12
    bels = Any[(copy(qh), copy(qh))]
    while length(bels) < NBEL + 1
        d = copy(qh)
        for t in 1:4
            i, j = randperm(rng, S)[1:2]; mv = rand(rng) * min(d[i] - bx.dl[i], bx.du[j] - d[j]); dd = copy(d); dd[i] -= mv; dd[j] += mv; inset(dd) && (d = dd)
        end
        p = copy(d); i, j = randperm(rng, S)[1:2]; mv = min(0.5 * nz_delta(nd), p[i]); p[i] -= mv; p[j] += mv
        r = zeros(S); rem = 1.0
        for s in randperm(rng, S); w = min(p[s] / (1 - nd.beta), rem); r[s] = w; rem -= w; rem <= 1e-12 && break; end
        push!(bels, (d, r))
    end
    for (bi, (d, r)) in enumerate(bels)
        cap(sel) = (c = zeros(nB); for (a, k) in sel; c[a] = tiers[k]; end; c)
        V1 = Dict((a, k) => V(cap([(a, k)]), d, r) for a in 1:nB, k in 1:K)
        V0 = V(zeros(nB), d, r)
        V2 = Dict{Any,Float64}()
        for a in 1:nB, b in a+1:nB, ka in 1:K, kb in 1:K; V2[(a, ka, b, kb)] = V(cap([(a, ka), (b, kb)]), d, r); end
        # 단독 용량 Δ = 100 (크기 증분용)
        V100 = Dict(a => V((c = zeros(nB); c[a] = 100.0; c), d, r) for a in 1:nB)
        stats = Dict{String,Vector{Float64}}()
        rec(t, lhs, rhs) = push!(get!(stats, t, Float64[]), (rhs - lhs) / max(1.0, abs(rhs)))   # 양수면 위반 (lhs < rhs 이어야 하는 꼴로 넣음)
        for a in 1:nB, k in 1:K
            rec("M1 V({a}) ≤ V(∅)", -V1[(a, k)], -V0)                                  # V1 ≤ V0  →  rhs − lhs = V1 − V0 ... (부호 맞춤)
            k < K && rec("M2 크기↑ ≤", -V1[(a, k + 1)], -V1[(a, k)])
        end
        for a in 1:nB, b in a+1:nB, ka in 1:K, kb in 1:K
            v = V2[(a, ka, b, kb)]
            rec("M1 V(ab) ≤ V(a)", -v, -V1[(a, ka)]); rec("M1 V(ab) ≤ V(a)", -v, -V1[(b, kb)])
            rec("S1 V(ab) ≥ V(a)+V(b)", v, V1[(a, ka)] + V1[(b, kb)])
            if kb < K                                                                    # S4: b 의 크기를 한 단계 올림 = V(ab⁺) ≥ V(ab) + V_b(Δ)
                Δ = tiers[kb + 1] - tiers[kb]; vj = Δ == 100 ? V100[b] : V1[(b, findfirst(==(Δ), tiers))]
                rec("S4 V(c+Δ) ≥ V(c)+V_j(Δ)", V2[(a, ka, b, kb + 1)], v + vj)
            end
        end
        for t in 1:NTRI
            st3 = sort(randperm(rng, nB)[1:3]); ks = [rand(rng, 1:K) for _ in 1:3]; sel = collect(zip(st3, ks))
            v = V(cap(sel), d, r)
            pv(i, j) = V2[(sel[i][1], sel[i][2], sel[j][1], sel[j][2])]
            rec("S2 V(abc) ≥ V(ab)+V(c)", v, pv(1, 2) + V1[sel[3]]); rec("S2 V(abc) ≥ V(ab)+V(c)", v, pv(1, 3) + V1[sel[2]]); rec("S2 V(abc) ≥ V(ab)+V(c)", v, pv(2, 3) + V1[sel[1]])
            rec("S3 V(abc) ≥ ΣV", v, V1[sel[1]] + V1[sel[2]] + V1[sel[3]])
            rec("M1 V(abc) ≤ V(ab)", -v, -pv(1, 2))
        end
        for (tname, v) in sort(collect(stats))
            nv = count(>(TOLR), v)
            println(out, g, "\t", bi, "\t", tname, "\t", length(v), "\t", nv, "\t", round(maximum(v); sigdigits=3)); flush(out)
        end
        @printf("g=%d belief %d 끝\n", g, bi); flush(stdout)
    end
end
close(out)
