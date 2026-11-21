# belief 생성 시제품 (2026-11-20): min_x q·x + V*(x),  V*(x) ≥ g(x, d) (belief d 의 follower 최적 반응을 정확 평가).
#   하한 L(x) = max_{d ∈ 𝒟} g(x, d) 를 모든 x 에 유지. 후보 x̄ = argmin q·x + L(x). x̄ 에서 싼 belief 탐색 → 안 오르면 전역 oracle (α-B&B).
#   가지치기는 평가된 값으로만 (증명 안 된 구조 부등식은 쓰지 않음).
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl")); include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl")); include(joinpath(@__DIR__, "bgen_core.jl"))
envf(k, d) = get(ENV, k, d)
g = parse(Int, envf("GEO", "1")); MODEL = envf("MODEL", "coupled"); r = parse(Float64, envf("R", "0.2")); TL = parse(Float64, envf("TL", "3600")); TOL = 5e-3
ε̂, ε̃, δ = MODEL == "coupled" ? (r, r, r / 3) : MODEL == "belief" ? (0.0, r, Inf) : MODEL == "data" ? (r, 0.0, Inf) : MODEL == "nominal" ? (0.0, 0.0, Inf) : (r, r, Inf)
tiers = parse.(Float64, split(envf("TIERS", "200,300,400"), ","))
nd = make_loc_tkp(; g=g, S=parse(Int, envf("SS", "10")), seed=parse(Int, envf("SEED", "1")), eps_hat=ε̂, eps_tilde=ε̃, delta=δ, beta=0.4, tiers=tiers)
_BXROWS[] = unique(e.k for e in nz_xentries(nd)); K, nB, S = nd.meta[:K], nd.meta[:nB], nd.S; q̂ = nd.q_hat; bx = nd.meta[:box]
NT = Threads.nthreads(); pool = Channel{BEv}(NT); for env in _nz_envs(NT); put!(pool, BEv(nd, env)); end
# X
X = Vector{Vector{Float64}}(); push!(X, zeros(nd.nx))
sites = collect(1:nB)
if haskey(ENV, "SITES"); sites = parse.(Int, split(ENV["SITES"], ",")); end
for n in 1:nd.gamma, comb in Iterators.filter(c -> issorted(c) && allunique(c), Iterators.product(ntuple(_ -> sites, n)...)), lv in Iterators.product(ntuple(_ -> 1:K, n)...)
    x = zeros(nd.nx); for (s, k) in zip(comb, lv); x[(s - 1) * K + k] = 1; end; push!(X, x)
end
N = length(X); qx = [dot(nd.qx, x) for x in X]
@printf("--- bgen g=%d model=%s r=%.2f S=%d |X|=%d 스레드 %d TL=%.0f\n", g, MODEL, r, S, N, NT, TL); flush(stdout)
inset(d) = all(d .>= bx.dl .- 1e-12) && all(d .<= bx.du .+ 1e-12) && sum(abs.(d .- q̂)) <= 2nd.eps_tilde + 1e-12
function randbel(rng)
    d = copy(q̂)
    for t in 1:rand(rng, 1:4)
        i, j = randperm(rng, S)[1:2]; mv = rand(rng) * min(d[i] - bx.dl[i], bx.du[j] - d[j]); dd = copy(d); dd[i] -= mv; dd[j] += mv; inset(dd) && (d = dd)
    end
    return d
end
L = fill(-Inf, N); L[1] = 0.0                     # x = ∅: B 거점이 없으면 수익 0
n_eval = Threads.Atomic{Int}(0)
"belief d 를 idxs 의 x 에 평가해 L 을 올림"
function sweep!(idxs, d)
    newv = fill(-Inf, length(idxs))
    Threads.@threads for t in eachindex(idxs)
        ev = take!(pool)
        try
            bev_setx!(ev, nd, X[idxs[t]]); v, _, _ = bev_g!(ev, nd, d); newv[t] = v; Threads.atomic_add!(n_eval, 1)
        finally
            put!(pool, ev)
        end
    end
    up = 0
    for (t, i) in enumerate(idxs); newv[t] > L[i] + 1e-9 * max(1.0, abs(newv[t])) && (L[i] = newv[t]; up += 1); end
    return up
end
"한 쌍 (i → j) 으로 갈 수 있는 만큼 옮긴 belief 들 (상자 ∩ TV 볼의 모서리 쪽)"
function pairbels()
    out = Vector{Vector{Float64}}()
    for i in 1:S, j in 1:S
        i == j && continue
        mv = min(q̂[i] - bx.dl[i], bx.du[j] - q̂[j], nd.eps_tilde); mv <= 1e-9 && continue
        d = copy(q̂); d[i] -= mv; d[j] += mv; inset(d) && push!(out, d)
    end
    return out
end
const PAIRBELS = pairbels()
"x̄ 에서 싼 belief 탐색 (병렬): 후보 = 집합 + 쌍 이동 + 무작위. 가장 좋은 값과 그 belief. 정확 LP 가 돌려준 belief 로 몇 번 되풀이"
function search(i, 𝒟, rng; nrand=40)
    cand = vcat(𝒟, PAIRBELS, [randbel(rng) for _ in 1:nrand]); vals = fill(-Inf, length(cand)); dds = Vector{Any}(nothing, length(cand))
    Threads.@threads for t in eachindex(cand)
        local ev = take!(pool)
        try
            bev_setx!(ev, nd, X[i]); local v1, d1, _ = bev_g!(ev, nd, cand[t]); vals[t] = v1; dds[t] = d1; Threads.atomic_add!(n_eval, 1)
        finally
            put!(pool, ev)
        end
    end
    t = argmax(vals); best = vals[t]; bd = cand[t]; dd = dds[t]
    ev = take!(pool)
    try
        bev_setx!(ev, nd, X[i])
        for it in 1:3
            (dd === nothing || !inset(dd)) && break
            v2, dd2, _ = bev_g!(ev, nd, dd); Threads.atomic_add!(n_eval, 1)
            v2 > best + 1e-9 ? (best = v2; bd = dd; dd = dd2) : break
        end
    finally
        put!(pool, ev)
    end
    return best, bd
end
rel(a, b) = abs(a - b) / max(1.0, abs(b))
rng = MersenneTwister(2026); 𝒟 = [copy(q̂)]; UB = Inf; ibest = 0; UBx = Dict{Int,Float64}(); ncall = Dict{Int,Int}(); t_or = 0.0; n_or = 0
wall = time(); status = :TimeLimit
ts = @elapsed sweep!(collect(2:N), q̂)
@printf("  [sweep] 명목 belief 로 전체 %d 점 (%.0fs)\n", N - 1, ts); flush(stdout)
addbel!(d) = (any(sum(abs.(d .- e)) < 1e-7 for e in 𝒟) ? false : (push!(𝒟, copy(d)); true))
iter = 0
while time() - wall < TL
    global iter += 1
    f = qx .+ L; i1 = argmin(f); LBm = f[i1]
    if isfinite(UB) && UB - LBm <= TOL * max(1.0, abs(UB)); global status = :Optimal; break; end
    active = [i for i in 2:N if !isfinite(UB) || f[i] < UB - TOL * max(1.0, abs(UB))]
    xs = string(findall(X[i1] .> 0.5))
    # 1) 싼 belief 탐색
    tsr = @elapsed ((v, bd) = search(i1, 𝒟, rng))
    if v > L[i1] + 1e-4 * max(1.0, abs(v))
        old = L[i1]; L[i1] = v; isnew = addbel!(bd)
        tsw = 0.0; up = 0
        isnew && (tsw = @elapsed (up = sweep!(active, bd)))
        @printf("  [srch] it %3d LB=%10.2f UB=%10.2f x=%-12s L %.2f → %.2f (%.1fs)%s  |𝒟|=%d active=%d\n", iter, LBm, UB, xs, old, v, tsr,
                isnew ? @sprintf(", 새 belief 로 %d 점 갱신 (%.0fs)", up, tsw) : "", length(𝒟), length(active)); flush(stdout)
        continue
    end
    # 2) 전역 oracle
    if get(ncall, i1, 0) >= 3; global status = :Stalled; @printf("  [!] x=%s 에서 oracle 을 두 번 불러도 gap 이 남음\n", xs); break; end
    f2 = minimum(f[i] for i in 1:N if i != i1)
    # 목표값: x̄ 가 둘째 후보보다 허용오차 넘게 나빠지면 멈춤 (근소하게 밀리는 정도면 상한을 끝까지 증명). 이미 한 번 부른 x̄ 는 목표값 없이
    τ = get(ncall, i1, 0) == 0 ? f2 + TOL * max(1.0, abs(f2)) - qx[i1] : nothing
    gap_o = min(1e-3, 0.5 * TOL * max(1.0, abs(LBm)) / max(1.0, abs(L[i1])))
    lim = max(10.0, TL - (time() - wall))
    to = @elapsed (res = nz_alpha_bnb_dr(nd, X[i1]; nworkers=NT - 1, time_limit=lim, rel_gap=gap_o, target=τ, verbose=false))
    global t_or += to; global n_or += 1; ncall[i1] = get(ncall, i1, 0) + 1
    oldL = L[i1]; L[i1] = max(L[i1], res[:LB]); UBx[i1] = min(get(UBx, i1, Inf), res[:UB])
    if qx[i1] + UBx[i1] < UB; global UB = qx[i1] + UBx[i1]; global ibest = i1; end
    up = 0; isnew = false
    if res[:α] !== nothing && res[:LB] > oldL + 1e-6 * max(1.0, abs(oldL))
        z = nz_exact_value(nd, X[i1], res[:α])
        if z isa Tuple && inset(z[2]); isnew = addbel!(z[2]); isnew && (up = sweep!([i for i in 2:N if qx[i] + L[i] < UB - TOL * max(1.0, abs(UB))], z[2])); end
    end
    @printf("  [orcl] it %3d LB=%10.2f UB=%10.2f x=%-12s V ∈ [%.2f, %.2f] (이전 L %.2f, %.0fs)%s  |𝒟|=%d\n", iter, LBm, UB, xs, res[:LB], res[:UB], oldL, to,
            isnew ? @sprintf(", 새 belief 로 %d 점 갱신", up) : "", length(𝒟)); flush(stdout)
end
f = qx .+ L; LBm = minimum(f)
xstr = ibest == 0 ? "-" : loc_tkp_xstr(nd, X[ibest])
@printf("RESULT bgen g=%d model=%s r=%.2f S=%d |X|=%d status=%s x=%s LB=%.4f UB=%.4f gap=%.2e iters=%d beliefs=%d oracle=%d (%.0fs) evals=%d time=%.1f\n",
        g, MODEL, r, S, N, status, xstr, LBm, UB, rel(LBm, UB), iter, length(𝒟), n_or, t_or, n_eval[], time() - wall)
