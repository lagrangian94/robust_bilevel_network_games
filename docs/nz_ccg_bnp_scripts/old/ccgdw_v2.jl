# ccgdw.jl — belief 블록 CCG, 마스터를 Dantzig–Wolfe 열 생성 루트 (Farley 하한) 로 푸는 시험판 (2026-11-23, scratchpad)
#  저장소 부품 (블록 생성기, belief 탐색 평가기, α-B&B oracle, 확보 규칙) 은 nz_belief_ccg_tree.jl 과 같게 씀.
#  마스터 (블록 집합 C) : min_{x∈X} qᵀx + t, t ≥ L_b(x) (b ∈ C), t ≥ (안전장치 아핀식)
#  DW: 블록마다 λ^b (열 = X 의 점), x 공유. 안전장치 (no-good, restricted cut) 는 아핀 "블록" (가격이 선형이라 탐욕).
#  LB = Farley 하한 (마스터의 유효 하한), 마스터 UB 점 x̄ = 평가된 지지 열 중 qᵀz + max_b L_b(z) 최소.
#  CCG: x̄ (와 상위 몇 점) 에서 싼 belief 탐색 → 없으면 x̄ 에서 oracle (α-B&B) → belief 추가·확보 → 열 재사용하며 다시 열 생성.
#  환경 변수: GEO 1, R 0.2, TL 3600, TOL 5e-3, MU 10, ETA 0.1, MU_MAX 1e4, NS (이웃 표본, 0 = 전부), SEED_ALL (새 블록에 캐시된 점 전부를 열로, 1) ,
#            NTOP (탐색할 상위 정수점 수, 5), CGTOL (열 생성 멈춤: 마스터 UB − Farley ≤ CGTOL·|UB|, 1e-3), NRAND 40
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Serialization
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_belief_ccg.jl")); include(joinpath(root, "nonzero_sum", "nz_belief_ccg_tree.jl"))
envf(k, d) = get(ENV, k, d)
const GEO = parse(Int, envf("GEO", "1")); const RR = parse(Float64, envf("R", "0.2"))
const TL = parse(Float64, envf("TL", "3600")); const TOL = parse(Float64, envf("TOL", "5e-3"))
const MU = parse(Float64, envf("MU", "10")); const ETA = parse(Float64, envf("ETA", "0.1")); const MU_MAX = parse(Float64, envf("MU_MAX", "1e4"))
const NS = parse(Int, envf("NS", "0")); const SEED_ALL = envf("SEED_ALL", "1") == "1"; const NTOP = parse(Int, envf("NTOP", "5"))
const CGTOL = parse(Float64, envf("CGTOL", "1e-3")); const NRAND = parse(Int, envf("NRAND", "40"))
nd = make_loc_tkp(; g=GEO, S=10, seed=1, eps_hat=RR, eps_tilde=RR, delta=RR / 3, beta=0.4)
K = nd.meta[:K]; nx = nd.nx; m, ny = nz_m(nd), nz_ny(nd); A = nd.A; XE = nz_xentries(nd); S = nd.S
NT = Threads.nthreads()
@printf("--- ccgdw | g=%d coupled r=%.2f S=%d, 스레드 %d, tol %.1e, μ %.3g, η %.3g, NS %d, SEED_ALL %s, NTOP %d, CGTOL %.1e\n", GEO, RR, S, NT, TOL, MU, ETA, NS, SEED_ALL, NTOP, CGTOL); flush(stdout)
wall = time()
site(i) = (i - 1) ÷ K + 1
al = findall(nd.x_allowed); sites = unique(site.(al))
pts = [zeros(nx)]
for n in 1:nd.gamma, comb in Iterators.filter(c -> issorted(c) && allunique(c), Iterators.product(ntuple(_ -> sites, n)...)), lv in Iterators.product(ntuple(_ -> 1:K, n)...)
    z = zeros(nx); for (st, k) in zip(comb, lv); z[(st - 1) * K + k] = 1.0; end; push!(pts, z)
end
NP = length(pts); key(z) = Tuple(findall(>(0.5), z)); idx = Dict(key(pts[t]) => t for t in 1:NP)
qv = [dot(nd.qx, z) for z in pts]
nbrs_cache = Dict{Int,Vector{Int}}()
function nbrs(t)
    get!(nbrs_cache, t) do
        z = pts[t]; on = [(site(i), (i - 1) % K + 1) for i in findall(>(0.5), z)]; out = Int[]
        put(conf) = (w = zeros(nx); for (s, k) in conf; w[(s - 1) * K + k] = 1; end; get(idx, key(w), 0))
        for a in eachindex(on)
            rest = [on[b] for b in eachindex(on) if b != a]; push!(out, put(rest))
            for s in sites, k in 1:K; any(o -> o[1] == s, rest) && continue; push!(out, put(vcat(rest, [(s, k)]))); end
        end
        if length(on) < nd.gamma
            for s in sites, k in 1:K; any(o -> o[1] == s, on) && continue; push!(out, put(vcat(on, [(s, k)]))); end
        end
        filter(>(0), unique(out))
    end
end
"X 위 선형 최소 min c0 + cᵀz (거점당 단계 하나, 거점 ≤ γ): 탐욕 (정확)"
function linmin(c)
    best = [(minimum(c[(s - 1) * K + k] for k in 1:K), argmin([c[(s - 1) * K + k] for k in 1:K]), s) for s in sites]
    sort!(best; by=first); z = zeros(nx); v = 0.0
    for (val, k, s) in best[1:min(nd.gamma, length(best))]; val < 0 || break; z[(s - 1) * K + k] = 1; v += val; end
    return v, idx[key(z)]
end
envs = _nz_envs(NT + 3)                    # 1..NT 생성기·평가기, NT+1 가격 MILP, NT+2 RMP, NT+3 검사 LP
lp_opt = () -> (o = Gurobi.Optimizer(envs[NT + 3]); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
C = (nd = nd, eta = ETA, vub = false, wd = false, XE = XE, lp_opt = lp_opt,
     rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m], colA = [[(k, A[k, j]) for k in 1:m if A[k, j] != 0] for j in 1:ny],
     ents_of_row = [[e for e in XE if e.k == k] for k in 1:m])
pool_ev = Channel{BCCGEval}(NT); for k in 1:NT; put!(pool_ev, BCCGEval(nd, envs[k])); end

# ---------------- 블록 ----------------
mutable struct Blk
    kind::Symbol                    # :lp (belief 블록) | :aff (안전장치 아핀식 c0 + cᵀz)
    d::Vector{Float64}; r::Vector{Float64}; μ::Float64
    gens::Vector{Any}               # 스레드마다 생성기
    c0::Float64; c::Vector{Float64}
end
blks = Blk[]; cache = Dict{Tuple{Int,Int},Float64}(); visited = Int[]; visset = Set{Int}(); lastsupp = Int[]
t_build = Ref(0.0); t_lp = Ref(0.0); n_lp = Ref(0); t_mip = Ref(0.0); n_mip = Ref(0); nodes_mip = Ref(0); t_rmp = Ref(0.0); n_rmp = Ref(0)
function mkgens(d, r, μ)
    out = Vector{Any}(undef, NT)
    t_build[] += @elapsed (@sync for th in 1:NT; Threads.@spawn (out[th] = _bccg_gen(C, d, r, μ, envs[th], th)); end)
    out
end
add_lpblk!(d, r, μ) = (push!(blks, Blk(:lp, copy(d), copy(r), μ, mkgens(d, r, μ), 0.0, Float64[])); length(blks))
add_affblk!(c0, c) = (push!(blks, Blk(:aff, Float64[], Float64[], 0.0, Any[], c0, copy(c))); length(blks))
getL(c, b) = blks[b].kind == :aff ? blks[b].c0 + dot(blks[b].c, pts[c]) : cache[(c, b)]
"(점, 블록) 들을 블록 LP 로 (캐시 밖만, 스레드 병렬)"
function evalpairs!(prs)
    todo = unique([p for p in prs if blks[p[2]].kind == :lp && !haskey(cache, p)]); isempty(todo) && return
    out = Vector{Float64}(undef, length(todo))
    t_lp[] += @elapsed begin
        ch = Channel{Int}(length(todo)); for i in eachindex(todo); put!(ch, i); end; close(ch)
        @sync for th in 1:NT
            Threads.@spawn for i in ch
                c, b = todo[i]; out[i] = first(_bccg_gen_eval!(blks[b].gens[th], pts[c]))
            end
        end
    end
    for i in eachindex(todo); cache[todo[i]] = out[i]; c = todo[i][1]; c in visset || (push!(visset, c); push!(visited, c)); end; n_lp[] += length(todo)
end
evalall!(cs) = evalpairs!([(c, b) for c in cs for b in eachindex(blks)])
mval(c) = qv[c] + maximum(getL(c, b) for b in eachindex(blks))      # 마스터 목적 (모든 블록이 평가되어 있어야)

# ---------------- 가격 MILP ----------------
pmods = Dict{Int,Any}()
function pricemip(b, wb, πb, σb, stopv; tlim=600.0)
    if !haskey(pmods, b)
        Mp = Model(() -> (o = Gurobi.Optimizer(envs[NT + 1]); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), NT); o))
        @variable(Mp, xb[1:nx], Bin); @variable(Mp, tb >= -1e7); @constraint(Mp, sum(xb) <= nd.gamma)
        for (ix, rhs) in get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[]); @constraint(Mp, sum(xb[i] for i in ix) <= rhs); end
        for i in 1:nx; nd.x_allowed[i] || fix(xb[i], 0.0; force=true); end
        _bccg_add_block!(merge(C, (M = Mp, x = xb, t0 = tb)), blks[b].d, blks[b].r, blks[b].μ)
        set_optimizer_attribute(Mp, "MIPGap", 1e-6); set_optimizer_attribute(Mp, "IntFeasTol", 1e-9); set_optimizer_attribute(Mp, "IntegralityFocus", 1)
        set_optimizer_attribute(Mp, "PoolSolutions", 10)
        pmods[b] = (Mp, xb, tb)
    end
    Mp, xb, tb = pmods[b]
    @objective(Mp, Min, wb * tb - sum(πb[i] * xb[i] for i in al))
    set_optimizer_attribute(Mp, "BestBdStop", stopv); set_optimizer_attribute(Mp, "TimeLimit", tlim)
    t_mip[] += @elapsed optimize!(Mp); n_mip[] += 1; nodes_mip[] += Int(MOI.get(Mp, MOI.NodeCount()))
    zs = [idx[key(round.(value.(xb; result=k)))] for k in 1:result_count(Mp)]
    return objective_bound(Mp), zs, (has_values(Mp) ? objective_value(Mp) : Inf)
end

# ---------------- DW 열 생성 ----------------
cols = Vector{Set{Int}}()
function rmp()
    M = Model(() -> Gurobi.Optimizer(envs[NT + 2])); set_silent(M)
    set_optimizer_attribute(M, "Method", 2); set_optimizer_attribute(M, "Crossover", 0)
    nb = length(blks)
    @variable(M, x[al]); @variable(M, t)
    lam = Dict((b, c) => @variable(M, lower_bound = 0) for b in 1:nb for c in cols[b])
    ct = [@constraint(M, t - sum(getL(c, b) * lam[(b, c)] for c in cols[b]) >= 0) for b in 1:nb]
    cl = [Dict(i => @constraint(M, sum(pts[c][i] * lam[(b, c)] for c in cols[b]) - x[i] == 0) for i in al) for b in 1:nb]
    cc = [@constraint(M, sum(lam[(b, c)] for c in cols[b]) == 1) for b in 1:nb]
    @objective(M, Min, sum(nd.qx[i] * x[i] for i in al) + t)
    t_rmp[] += @elapsed optimize!(M); n_rmp[] += 1
    termination_status(M) == MOI.OPTIMAL || error("ccgdw: RMP $(termination_status(M))")
    w = dual.(ct); π = [[i in al ? dual(cl[b][i]) : 0.0 for i in 1:nx] for b in 1:nb]; σ = dual.(cc)
    supp = [Set(c for c in cols[b] if value(lam[(b, c)]) > 1e-9) for b in 1:nb]
    objective_value(M), w, π, σ, supp
end
rcv(b, c, w, π, σ) = w[b] * getL(c, b) - dot(π[b], pts[c]) - σ[b]
rng_ns = MersenneTwister(7)
"열 생성: (Farley LB, 마스터 UB, x̄ 색인, 상위 정수점 목록, 정보)"
function dwsolve!(; ubglob=Inf, exact=true)
    lb = -Inf; ubm = Inf; cbest = 0; it = 0; nex = 0; topl = Int[]
    while true
        it += 1
        v, w, π, σ, supp = rmp()
        tolabs = 1e-4 * max(1, abs(v))
        need = Tuple{Int,Int}[]; candb = Dict{Int,Vector{Int}}()
        for b in eachindex(blks)
            (w[b] > 1e-9 && blks[b].kind == :lp) || continue
            cs = Set{Int}()
            for c in supp[b]; push!(cs, c); nb_ = nbrs(c); union!(cs, (NS == 0 || length(nb_) <= NS) ? nb_ : shuffle(rng_ns, nb_)[1:NS]); end
            for c in visited; haskey(cache, (c, b)) && push!(cs, c); end     # 열 풀: 이미 값을 아는 점 (LP 없음)
            candb[b] = collect(cs); append!(need, [(c, b) for c in cs])
        end
        evalpairs!(need)
        added = 0
        for b in eachindex(blks)
            if haskey(candb, b)
                c = argmin(c -> rcv(b, c, w, π, σ), candb[b]); r = rcv(b, c, w, π, σ)
            else       # w_b = 0 이거나 아핀 블록: 환원비용이 선형 (아핀이면 w_b c)
                cl = blks[b].kind == :aff ? w[b] .* blks[b].c .- π[b] : -π[b]
                r, c = linmin(cl); r += (blks[b].kind == :aff ? w[b] * blks[b].c0 : 0.0) - σ[b]; evalpairs!([(c, b)])
            end
            if r < -tolabs && !(c in cols[b]); push!(cols[b], c); added += 1; end
        end
        added > 0 && continue
        # 마스터 UB 점 (지지 열에서 모든 블록 평가)
        sp = unique([c for b in eachindex(blks) for c in supp[b]]); evalall!(sp); global lastsupp = sp
        for c in sp; mv = mval(c); mv < ubm && (ubm = mv; cbest = c); end
        topl = sort(sp; by=mval)[1:min(NTOP, length(sp))]
        exact || return -Inf, ubm, cbest, topl, (it = it, nex = 0, rmp = v)
        # 정확 가격 (w_b > 0 인 belief 블록만 MILP)
        lbit = v; added2 = 0
        tgt = min(ubm, ubglob)
        for b in eachindex(blks)
            if blks[b].kind == :lp && w[b] > 1e-9
                bnd, zs, inc = pricemip(b, w[b], π[b], σ[b], σ[b] - tolabs)
                lbit += min(0.0, bnd - σ[b])
                if inc - σ[b] < -tolabs
                    evalpairs!([(zc, b) for zc in zs])          # 해 풀의 열을 모두 검사
                    for zc in zs; if rcv(b, zc, w, π, σ) < -tolabs && !(zc in cols[b]); push!(cols[b], zc); added2 += 1; end; end
                end
            else
                cl = blks[b].kind == :aff ? w[b] .* blks[b].c .- π[b] : -π[b]
                r, c = linmin(cl); r += (blks[b].kind == :aff ? w[b] * blks[b].c0 : 0.0) - σ[b]; lbit += min(0.0, r)
                if r < -tolabs && !(c in cols[b]); evalpairs!([(c, b)]); push!(cols[b], c); added2 += 1; end
            end
        end
        nex += 1; lb = max(lb, lbit)
        (added2 == 0 || tgt - lb <= CGTOL * max(1, abs(tgt))) && return lb, ubm, cbest, topl, (it = it, nex = nex, rmp = v)
    end
end
"새 블록의 열 씨앗: 열 = 지금 지지 열 (+ c). SEED_ALL 이면 방문한 점 전부를 이 블록 LP 로 평가해 열 풀에 둠 (RMP 에는 안 넣음)"
function seed!(b, c=0)
    base = unique(vcat(lastsupp, c > 0 ? [c] : Int[], [1]))
    if blks[b].kind == :lp
        evalpairs!([(cc, b) for cc in base])
        SEED_ALL && evalpairs!([(cc, b) for cc in copy(visited)])
    end
    push!(cols, Set(base))
end

# ---------------- belief 탐색·확보 (nz_belief_ccg_tree 와 같은 규칙) ----------------
qh = nd.q_hat; bx = nd.meta[:box]
inset(d) = all(d .>= bx.dl .- 1e-12) && all(d .<= bx.du .+ 1e-12) && sum(abs.(d .- qh)) <= 2nd.eps_tilde + 1e-12
pairbels = Vector{Vector{Float64}}()
for i in 1:S, j in 1:S
    i == j && continue
    mv = min(qh[i] - bx.dl[i], bx.du[j] - qh[j], nd.eps_tilde); mv <= 1e-9 && continue
    dd = copy(qh); dd[i] -= mv; dd[j] += mv; inset(dd) && push!(pairbels, dd)
end
rng = MersenneTwister(2026)
function randbel()
    d = copy(qh)
    for t in 1:rand(rng, 1:4)
        i, j = randperm(rng, S)[1:2]; mv = rand(rng) * min(d[i] - bx.dl[i], bx.du[j] - d[j]); dd = copy(d); dd[i] -= mv; dd[j] += mv; inset(dd) && (d = dd)
    end
    return d
end
t_srch = Ref(0.0); n_srch = Ref(0)
function search(xb)
    n_srch[] += 1
    t_srch[] += @elapsed begin
        cand = vcat([B.d for B in blks if B.kind == :lp], pairbels, [randbel() for _ in 1:NRAND]); vals = fill(-Inf, length(cand))
        Threads.@threads for t in eachindex(cand)
            local ev = take!(pool_ev)
            try; _bccg_setx!(ev, nd, xb); local v1, _, _, _ = _bccg_g!(ev, nd, cand[t]); vals[t] = v1; finally; put!(pool_ev, ev); end
        end
        tb = argmax(vals); best = (v = -Inf, d = nothing, r = nothing, α = nothing)
        ev2 = take!(pool_ev)
        try
            _bccg_setx!(ev2, nd, xb); dcur = cand[tb]
            for it in 1:4
                v2, dd, r2, α2 = _bccg_g!(ev2, nd, dcur)
                dd === nothing && break
                v2 > best.v + 1e-9 || break
                best = (v = v2, d = dd, r = r2, α = α2)
                inset(dd) || break
                dcur = dd
            end
        finally; put!(pool_ev, ev2); end
    end
    return best
end
function exact_value(xb, α)
    ev3 = take!(pool_ev)
    try
        _bccg_setx!(ev3, nd, xb)
        a1 = [(v = clamp(α[i], 0.0, nd.hU[i]); v < 1e-6 * max(1.0, nd.hU[i]) ? 0.0 : v) for i in eachindex(α)]
        v3, dd = _bccg_value!(ev3, nd, a1)
        if dd === nothing
            a2 = nz_feasible_alpha(nd, xb, a1; lp_optimizer=lp_opt, tol=1e-3); a1 = a2; v3, dd = _bccg_value!(ev3, nd, a2)
        end
        return v3, dd, (dd === nothing ? nothing : copy(ev3.lastr)), a1
    finally; put!(pool_ev, ev3); end
end
tlow = -1e7; n_ng = Ref(0); n_mu = Ref(0); n_rc = Ref(0)
function nogood!(c, v)
    xb = pts[c]; cv = zeros(nx); c0 = v
    for i in al; if xb[i] > 0.5; c0 -= (v - tlow); cv[i] = (v - tlow); else; cv[i] = -(v - tlow); end; end
    b = add_affblk!(c0, cv); seed!(b, c); n_ng[] += 1
end
function restricted!(c, α)
    F = nz_build_fixed_alpha(nd, pts[c]; optimizer=lp_opt); z = nz_eval_alpha!(F, nd, α); isfinite(z) || return false
    cut = nz_cut_from_solution(F, nd, pts[c]); b = add_affblk!(cut[:intercept], cut[:slope]); seed!(b, c); n_rc[] += 1; true
end
"belief 를 블록으로 넣고 c 에서 확보 (μ 올리기 → no-good)"
function add_and_secure!(c, v, d, r)
    xb = pts[c]; thr = 1e-4 * max(1.0, abs(v))
    b = add_lpblk!(d, r, MU); evalpairs!([(c, b)])
    getL(c, b) >= v - thr && (seed!(b, c); return :block)
    _, Linf, μstar = _bccg_block_at(C, xb, d, r, MU)
    if Linf >= v - thr && μstar <= MU_MAX
        μn = min(MU_MAX, max(2MU, 1.1μstar)); blks[b] = Blk(:lp, copy(d), copy(r), μn, mkgens(d, r, μn), 0.0, Float64[]); n_mu[] += 1
        for k in collect(keys(cache)); k[2] == b && delete!(cache, k); end
        evalpairs!([(c, b)])
        getL(c, b) >= v - thr && (seed!(b, c); return :mu)
    end
    seed!(b, c); nogood!(c, v); return :safeguard
end

# ---------------- CCG ----------------
b1 = add_lpblk!(qh, qh, MU); evalpairs!([(1, b1)]); push!(cols, Set([1]))
# t 하한 (no-good 용): min_z L_b0(z) 를 가격 MILP 로 (π = 0)
bnd = 0.0; tl0 = @elapsed (global bnd = first(pricemip(b1, 1.0, zeros(nx), 0.0, Inf)))
global tlow = bnd - 1e-6 * max(1.0, abs(bnd))
@printf("  [시작] 블록 생성기 %.1fs, t 하한 %.2f (가격 MILP %.1fs)\n", t_build[], tlow, tl0); flush(stdout)
"상위 정수점에서 싼 탐색, belief 하나 넣으면 true"
function trysearch(topl)
    for c in topl
        bb = search(pts[c]); tc = mval(c) - qv[c]
        if bb.d !== nothing && bb.v > tc + 1e-3 * max(1.0, abs(bb.v))
            how = add_and_secure!(c, bb.v, bb.d, bb.r)
            @printf("    [탐색] x=%-24s t %.2f → %.2f (%s), 블록 %d
", loc_tkp_xstr(nd, pts[c]), tc, bb.v, how, length(blks)); flush(stdout)
            return true
        end
    end
    false
end
UB = Inf; LB = -Inf; xbest = 0; status = :TimeLimit; iter = 0; t_dw = 0.0; t_or = 0.0; n_or = 0
while time() - wall < TL
    global iter += 1
    tdw = @elapsed (dwr = dwsolve!(; ubglob=UB, exact=false)); global t_dw += tdw; lbm, ubm, cb, topl, info = dwr
    @printf("  [dw-h] it %2d: 마스터 UB %.2f (x=%s), 반복 %d, 블록 %d, LP 누적 %d, %.0fs (경과 %.0fs)
",
            iter, ubm, loc_tkp_xstr(nd, pts[cb]), info.it, length(blks), n_lp[], tdw, time() - wall); flush(stdout)
    trysearch(topl) && continue
    tdw = @elapsed (dwr = dwsolve!(; ubglob=UB, exact=true)); global t_dw += tdw; lbm, ubm, cb, topl, info = dwr
    global LB = max(LB, lbm)
    @printf("  [dw-x] it %2d: Farley LB %.2f, 마스터 UB %.2f (x=%s), 간격 %.3f%%, 반복 %d (정확 %d), 블록 %d, LP 누적 %d, MILP 누적 %d, %.0fs (경과 %.0fs)
",
            iter, lbm, ubm, loc_tkp_xstr(nd, pts[cb]), 100 * (ubm - lbm) / abs(ubm), info.it, info.nex, length(blks), n_lp[], n_mip[], tdw, time() - wall); flush(stdout)
    if isfinite(UB) && UB - LB <= TOL * max(1.0, abs(UB)); global status = :Optimal; break; end
    trysearch(topl) && continue
    tb = ubm - qv[cb]
    # oracle (x̄)
    xb = pts[cb]; t0v = tb
    tau = t0v + 0.9 * TOL * max(1.0, abs(LB)); gap_o = min(1e-3, 0.4 * TOL * max(1.0, abs(LB)) / max(1.0, abs(t0v)))
    lim = max(10.0, TL - (time() - wall))
    to = @elapsed (res = nz_alpha_bnb_dr(nd, xb; nworkers=max(1, NT - 1), time_limit=lim, rel_gap=gap_o, target=tau, verbose=false))
    global t_or += to; global n_or += 1
    if qv[cb] + res[:UB] < UB; global UB = qv[cb] + res[:UB]; global xbest = cb; end
    how = :none
    if res[:α] !== nothing && res[:LB] > t0v + 1e-6 * max(1.0, abs(t0v))
        vv, dd, r2, a1 = exact_value(xb, res[:α])
        if dd !== nothing
            how = add_and_secure!(cb, vv, dd, r2)
            vv <= t0v + 1e-6 * max(1.0, abs(t0v)) && restricted!(cb, a1)
        end
    end
    if how == :none && res[:LB] > t0v + 1e-6 * max(1.0, abs(t0v)); nogood!(cb, res[:LB]); how = :nogood_lb; end
    @printf("  [orcl] it %2d: LB %.2f UB %.2f x=%-24s V in [%.2f, %.2f], t0 %.2f (%.0fs) %s, 블록 %d\n", iter, LB, UB, loc_tkp_xstr(nd, xb), res[:LB], res[:UB], t0v, to, how, length(blks)); flush(stdout)
    if isfinite(UB) && UB - LB <= TOL * max(1.0, abs(UB)); global status = :Optimal; break; end
    if how == :none
        global status = ubm - LB > TOL * max(1.0, abs(ubm)) ? :DWGap : :Stalled
        println("  [!] oracle 이 진전을 못 줌 ($(status))"); break
    end
end
tot = time() - wall
nbel = count(B -> B.kind == :lp, blks)
@printf("RESULT ccgdw g=%d status=%s x=%s LB=%.4f UB=%.4f gap=%.3f%% iters=%d beliefs=%d 안전장치 %d (no-good %d, restricted %d, μ 올림 %d) time=%.1f\n",
        GEO, status, xbest == 0 ? "-" : loc_tkp_xstr(nd, pts[xbest]), LB, UB, 100 * (UB - LB) / abs(UB), iter, nbel, n_ng[] + n_rc[], n_ng[], n_rc[], n_mu[], tot)
@printf("  시간: DW %.0fs [블록 LP %d 번 %.0fs, 가격 MILP %d 번 %.0fs (노드 %d), RMP %d 번 %.0fs, 생성기 구성 %.0fs], oracle %d 번 %.0fs, 탐색 %d 번 %.0fs; 서로 다른 점 %d / %d\n",
        t_dw, n_lp[], t_lp[], n_mip[], t_mip[], nodes_mip[], n_rmp[], t_rmp[], t_build[], n_or, t_or, n_srch[], t_srch[], length(unique([c for (c, _) in keys(cache)])), NP)
