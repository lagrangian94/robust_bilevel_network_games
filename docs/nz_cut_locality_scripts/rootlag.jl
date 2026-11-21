# 루트에서 Lagrangian cut 을 미리 채우는 Benders (2026-11-21): 후보지 7 곳, belief 블록을 cut 생성기로.
#   마스터 LP (x ∈ conv(X)) 의 분수 해 x̄ 에서 belief b 마다
#       max_π w_b(π),  w_b(π) = min_{z∈X} [L_b(z) − πᵀ(z − x̄)]          (L_b = belief 블록 값, z 마다 follower 반응 재최적화)
#   를 Kelley 절단면 (π-LP + 블록 하나짜리 MILP) 으로 풀고 cut  t0 ≥ w_b(π) + πᵀ(x − x̄)  을 넣는다 (w 는 MILP 하한 → 유효).
#   라운드마다 루트 LP 하한을 기록하고, 끝에 정수 마스터를 한 번 풀어 이 cut 들만의 하한을 본다.
# 저장소 코드는 읽기만 (nz_belief_ccg.jl 의 블록·평가기 함수 재사용).
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl")); include(joinpath(root, "nonzero_sum", "nz_belief_ccg.jl"))
const ENVG = Gurobi.Env()
envf(k, d) = get(ENV, k, d)
NT = parse(Int, envf("NT", "10")); MU = 10.0; ETA = 0.1
ROUNDS = parse(Int, envf("ROUNDS", "8")); KMAX = parse(Int, envf("KMAX", "12")); BUDGET = parse(Float64, envf("BUDGET", "5400"))
PB = parse(Float64, envf("PB", "3000"))        # π 상자 (값 크기 cut 을 찾으므로 수천)
GRB1() = (o = Gurobi.Optimizer(ENVG); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
nd = make_loc_tkp(; g=1, S=10, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2/3, beta=0.4)
K = nd.meta[:K]; nB = nd.meta[:nB]; S = nd.S; nx = nd.nx; m, ny = nz_m(nd), nz_ny(nd); A = nd.A; XE = nz_xentries(nd)
keep = [1, 4, 6, 8, 10, 11, 13]
for st in 1:nB, k in 1:K; st in keep || (nd.x_allowed[(st - 1) * K + k] = false); end
allowed = findall(nd.x_allowed)
xi(site, k) = (site - 1) * K + k
st2(store) = store - nd.meta[:nA]              # 점포 번호 → 거점 번호
tier(v) = findfirst(==(Float64(v)), nd.meta[:tiers])
mkx(pairs) = (x = zeros(nx); for (store, sz) in pairs; x[xi(st2(store), tier(sz))] = 1; end; x)
seedx = [mkx([(8, 400), (10, 200), (15, 300)]), mkx([(10, 200), (12, 300), (15, 300)]), mkx([(10, 200), (13, 300), (15, 300)]),
         mkx([(3, 300), (10, 200), (15, 300)]), mkx([(6, 200), (10, 300), (15, 300)]), mkx([(8, 400), (10, 200), (15, 200)]),
         mkx([(6, 400), (10, 200), (15, 200)])]
@printf("후보지 %s, |X| 허용 변수 %d, 스레드 %d, 라운드 %d, π 상자 %.0f\n", string(keep), length(allowed), NT, ROUNDS, PB); flush(stdout)
wall = time()

# ---- belief: 명목 + seed x 마다 싼 탐색 최고 ----
qh = nd.q_hat; bx = nd.meta[:box]; rng = MersenneTwister(2026)
ev = BCCGEval(nd, ENVG)
bels = Any[(copy(qh), copy(qh))]
for xs in seedx
    _bccg_setx!(ev, nd, xs); best = (v = -Inf, d = nothing, r = nothing)
    for i in 1:S, j in 1:S
        i == j && continue
        mv = min(qh[i] - bx.dl[i], bx.du[j] - qh[j], nd.eps_tilde); mv <= 1e-9 && continue
        d = copy(qh); d[i] -= mv; d[j] += mv
        v, dd, r, α = _bccg_g!(ev, nd, d)
        (dd !== nothing && v > best.v) && (best = (v = v, d = dd, r = r))
    end
    best.d === nothing && continue
    any(b -> maximum(abs.(b[1] .- best.d)) < 1e-6 && maximum(abs.(b[2] .- best.r)) < 1e-6, bels) || push!(bels, (best.d, best.r))
end
@printf("belief %d 개 (명목 포함), %.0f s\n", length(bels), time() - wall); flush(stdout)

# ---- belief 마다 블록 하나짜리 MILP 모형 ----
function mkblock(d, r)
    M = Model(() -> (o = Gurobi.Optimizer(ENVG); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), NT); o))
    set_optimizer_attribute(M, "MIPGap", 1e-4); set_optimizer_attribute(M, "IntFeasTol", 1e-9)
    @variable(M, z[1:nx], Bin); @variable(M, t >= -1e7)
    @constraint(M, sum(z) <= nd.gamma)
    for (idx, rhs) in get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[]); @constraint(M, sum(z[i] for i in idx) <= rhs); end
    for i in 1:nx; nd.x_allowed[i] || fix(z[i], 0.0; force=true); end
    C = (nd = nd, M = M, x = z, t0 = t, eta = ETA, vub = false, XE = XE, lp_opt = GRB1,
         rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m], colA = [[(k, A[k, j]) for k in 1:m if A[k, j] != 0] for j in 1:ny],
         ents_of_row = [[e for e in XE if e.k == k] for k in 1:m])
    _bccg_add_block!(C, d, r, MU)
    return (M = M, z = z, t = t, pts = Tuple{Vector{Float64},Float64}[])   # pts: 지금까지 얻은 (z_k, L_b(z_k))
end
blocks = [mkblock(d, r) for (d, r) in bels]

# ---- 마스터 LP ----
P = Model(GRB1); @variable(P, 0 <= x[1:nx] <= 1); @variable(P, t0 >= -1e7)
@constraint(P, sum(x) <= nd.gamma)
for (idx, rhs) in get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[]); @constraint(P, sum(x[i] for i in idx) <= rhs); end
for i in 1:nx; nd.x_allowed[i] || fix(x[i], 0.0; force=true); end
@objective(P, Min, dot(nd.qx, x) + t0)
cuts = Tuple{Float64,Vector{Float64},Vector{Float64}}[]   # (w, π, x̄)

"belief 블록 B 의 Lagrangian cut at x̄: Kelley 절단면. 반환 (w, π, MILP 수)"
function lagcut(B, x̄)
    L = Model(GRB1)
    @variable(L, -PB <= π[i=1:nx] <= PB); @variable(L, πa[1:nx] >= 0); @variable(L, η)
    for i in 1:nx; nd.x_allowed[i] || fix(π[i], 0.0; force=true); end
    @constraint(L, πa .>= π); @constraint(L, πa .>= -π)
    for (zk, Lk) in B.pts; @constraint(L, η <= Lk - dot(zk .- x̄, π)); end
    @constraint(L, η <= 1e7)
    @objective(L, Max, η - 1e-3 * sum(πa))              # 같은 값이면 평평한 π
    best = (w = -Inf, π = zeros(nx)); nm = 0
    for k in 1:KMAX
        optimize!(L); πv = value.(π); ub = value(η)
        @objective(B.M, Min, B.t - sum(πv[i] * (B.z[i] - x̄[i]) for i in allowed))
        optimize!(B.M); nm += 1
        w = objective_bound(B.M); zk = Float64.(value.(B.z) .> 0.5); Lk = value(B.t)
        push!(B.pts, (zk, Lk)); @constraint(L, η <= Lk - dot(zk .- x̄, π))
        w > best.w && (best = (w = w, π = πv))
        ub - best.w <= max(5.0, 1e-3 * abs(best.w)) && break
    end
    return best.w, best.π, nm
end

# seed 점의 블록 값 (Kelley 의 처음 절단면)
for B in blocks, xs in seedx
    for i in 1:nx; fix(B.z[i], xs[i]; force=true); end
    @objective(B.M, Min, B.t); optimize!(B.M); push!(B.pts, (copy(xs), value(B.t)))
    for i in 1:nx; unfix(B.z[i]); set_binary(B.z[i]); nd.x_allowed[i] || fix(B.z[i], 0.0; force=true); end
end
@printf("seed 블록 값 계산 끝, %.0f s\n", time() - wall); flush(stdout)

for rd in 1:ROUNDS
    optimize!(P); x̄ = value.(x); t̄ = value(t0); lb = objective_value(P)
    nfrac = count(v -> 1e-6 < v < 1 - 1e-6, x̄)
    @printf("[라운드 %d] 루트 LP 하한 %.2f  (분수 성분 %d, t0 %.2f)  %.0f s\n", rd, lb, nfrac, t̄, time() - wall); flush(stdout)
    added = 0; nmt = 0
    for (bi, B) in enumerate(blocks)
        time() - wall > BUDGET && break
        w, πv, nm = lagcut(B, x̄); nmt += nm
        if w > t̄ + 1e-3 * max(1.0, abs(t̄))
            @constraint(P, t0 >= w + sum(πv[i] * (x[i] - x̄[i]) for i in allowed)); push!(cuts, (w, πv, copy(x̄))); added += 1
        end
        @printf("   belief %d: cut 값 %.2f (t0 %.2f), |π|∞ %.0f, MILP %d\n", bi, w, t̄, maximum(abs, πv), nm); flush(stdout)
    end
    (added == 0 || time() - wall > BUDGET) && (optimize!(P); @printf("[끝] 루트 LP 하한 %.2f, cut %d, %.0f s\n", objective_value(P), length(cuts), time() - wall); break)
end
# 이 cut 들만으로 정수 마스터
for i in allowed; set_binary(x[i]); end
set_optimizer(P, () -> (o = Gurobi.Optimizer(ENVG); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), NT); o))
set_optimizer_attribute(P, "MIPGap", 1e-4)
tm = @elapsed optimize!(P)
@printf("정수 마스터 (루트 cut %d 개만): 하한 %.2f, x = %s, 노드 %d, %.1f s  (참고: 최적 ≈ −1,630.9 ~ −1,623.6)\n",
        length(cuts), objective_value(P), loc_tkp_xstr(nd, Float64.(value.(x) .> 0.5)), Int(MOI.get(P, MOI.NodeCount())), tm)
