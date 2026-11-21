# belief 일반화 측정: x̂ 에서 얻은 belief (b, t) 의 고정 LP 값 v(x'; b, t) 가 다른 정수점 x' 에서 V*(x') 에 얼마나 가까운가 (선형화와 무관한 상한)
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
g = parse(Int, get(ENV, "GEO", "1")); NT = Threads.nthreads()
mk(lam) = make_loc_tkp(; g=g, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4, lambdaU=lam)
nd = mk(100.0); K = nd.meta[:K]; nB = nd.meta[:nB]
rng = MersenneTwister(11)
xs = Vector{Vector{Tuple{Int,Int}}}()          # (site, level)
push!(xs, Tuple{Int,Int}[])
for b in 1:4
    sites = sort(randperm(rng, nB)[1:3]); base = [(s, rand(rng, 1:K)) for s in sites]; push!(xs, base)
    for t in 1:2                                # 크기만 바꾼 이웃
        y = copy(base); j = rand(rng, 1:3); y[j] = (y[j][1], mod1(y[j][2] + rand(rng, 1:K-1), K)); push!(xs, y)
    end
    for t in 1:2                                # 거점 하나 교체
        y = copy(base); j = rand(rng, 1:3); ns = rand(rng, setdiff(1:nB, first.(base))); y[j] = (ns, y[j][2]); push!(xs, sort(y))
    end
    push!(xs, base[1:2])                        # 거점 하나 닫기
end
for t in 1:3; push!(xs, [(rand(rng, 1:nB), rand(rng, 1:K))]); end
xs = unique(xs); n = length(xs)
xvec(sel) = (x = zeros(nd.nx); for (s, k) in sel; x[(s - 1) * K + k] = 1; end; x)
rel(a, b) = a == b ? :same : (first.(a) == first.(b) ? :level : (length(symdiff(first.(a), first.(b))) <= 2 && abs(length(a) - length(b)) <= 1 ? :site1 : :far))
# 1) local 해 → belief
Oloc = build_nz_omega(nd; optimizer=GRB); set_optimizer_attribute(Oloc.model, "OptimalityTarget", 1)
bels = Vector{Any}(undef, n); vloc = fill(NaN, n)
for (i, sel) in enumerate(xs)
    x̄ = xvec(sel)
    t = @elapsed (rl = _nz_local_solve!(Oloc, nd, x̄; time_limit=120.0))
    if rl === nothing; bels[i] = nothing; println("local 해 없음 ", sel); continue; end
    bl = nz_read_belief(Oloc, nd)
    α̂ = copy(bl[:α]); nA = nd.meta[:nA]; tiers = nd.meta[:tiers]
    for st in 1:nB                                  # α 정리: 닫힌 거점 0, 용량 이하, 잡음 0
        cap = sum(tiers[k] * x̄[(st - 1) * K + k] for k in 1:K); ih = nd.meta[:hidx](nA + st, 1)
        α̂[ih] = min(α̂[ih], cap)
    end
    α̂ = [a < 1e-6 * max(1.0, nd.hU[j]) ? 0.0 : a for (j, a) in enumerate(α̂)]
    σ̂ = try nz_follower_duals(nd, x̄, α̂; lp_optimizer=GRB)[1] catch e; println("follower 쌍대 실패 ", sel, " α 원값 ", round.(bl[:α]; digits=6)); nothing end
    bels[i] = σ̂ === nothing ? nothing : (nz_clean_belief(bl, nd), σ̂); vloc[i] = rl[:Fval]
    @printf("local %2d x=%-22s Ω_loc=%10.2f (%.0fs)\n", i, string(sel), vloc[i], t); flush(stdout)
end
# 2) 행렬 v[b, x'] (λᵁ = 100, 1)
for lam in (100.0, 1.0)
    ndl = mk(lam)
    pool = Channel{Any}(NT)
    for env in _nz_envs(NT)
        mko = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
        put!(pool, build_nz_omega(ndl; optimizer=mko, belief_lp=true))
    end
    V = fill(NaN, n, n)
    jobs = [(b, j) for b in 1:n for j in 1:n if bels[b] !== nothing]
    tt = @elapsed Threads.@threads for (b, j) in jobs
        B = take!(pool)
        try
            nz_set_belief!(B, ndl, bels[b][1], bels[b][2])
            V[b, j] = try nz_solve!(B, ndl, xvec(xs[j]))[:Fval] catch; -Inf end
        finally
            put!(pool, B)
        end
    end
    vref = [max(isnan(vloc[j]) ? -Inf : vloc[j], maximum(filter(!isnan, V[:, j]); init=-Inf)) for j in 1:n]
    @printf("\n=== λᵁ = %.0f (LP %d 개, %.0fs). 비율 = v_b(x') / V_ref(x') (1 = 정확, 클수록 느슨), V_ref = max(local 값, 최고 belief 값)\n", lam, length(jobs), tt)
    for cls in (:same, :level, :site1, :far)
        r = [V[b, j] / vref[j] for (b, j) in jobs if rel(xs[b], xs[j]) == cls && !isempty(xs[b]) && !isempty(xs[j]) && isfinite(V[b, j])]
        ninf = count(((b, j),) -> rel(xs[b], xs[j]) == cls && !isempty(xs[b]) && !isempty(xs[j]) && !isfinite(V[b, j]), jobs)
        isempty(r) && continue
        @printf("  %-6s n=%4d  중앙값 %.3f  25%% %.3f  75%% %.3f  최대 %.1f  | 1.005 이내 %d%%, 1.05 이내 %d%%, 1.5 이내 %d%%  (LP 실패 %d)\n", cls, length(r),
                median(r), quantile(r, 0.25), quantile(r, 0.75), maximum(r), round(Int, 100 * mean(r .<= 1.005)), round(Int, 100 * mean(r .<= 1.05)), round(Int, 100 * mean(r .<= 1.5)), ninf)
    end
    r0 = [V[1, j] / vref[j] for j in 2:n if isfinite(V[1, j])]
    @printf("  ∅ 의 belief → 다른 x: 중앙값 %.1f, 최소 %.1f, 최대 %.1f\n", median(r0), minimum(r0), maximum(r0))
    # 각 x' 에서 자기 것 말고 가장 좋은 belief
    rb = [maximum(V[b, j] for b in 1:n if b != j && bels[b] !== nothing) / vref[j] for j in 2:n]
    @printf("  자기 것 뺀 최고 belief (x' 마다): 중앙값 %.3f, 1.005 이내 %d%%, 1.05 이내 %d%%, 최대 %.2f\n", median(rb), round(Int, 100 * mean(rb .<= 1.005)), round(Int, 100 * mean(rb .<= 1.05)), maximum(rb))
    println("  V_ref = ", round.(vref; digits=0))
end
