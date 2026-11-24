# ccgbp_v6.jl — 이전 기본 판 v6 (2026-11-23). 2026-11-24 부터 기본 판은 v8 (ccgbp.jl). 절차는 docs/nz_ccg_bnp_2026-11-23.md 10 절.
#  저장소 부품 (블록 생성기 _bccg_gen, belief 탐색 평가기, α-B&B oracle nz_alpha_bnb_dr, 확보 규칙) 을 그대로 부름. X 를 미리 나열하지 않음 (만난 점에만 번호).
#  기본값 = 인증용 설정: η = 1/λᵁ, μ = μ 상한 = λᵁ = 1e4 (블록 = 원고 Prop certificate (ii) 의 벌점값을 belief 로 제한한 것, μ 올림 없음 → 모자라면 no-good).
#  환경 변수 (기본값): GEO 1, R 0.2, NB 3 (거점 수 상한), TL 3600, TOL 5e-3, ETA 1e-4, MU 1e4, MU_MAX 1e4, LAMU 1e4 (oracle λᵁ),
#    S 10 (시나리오 수), NS 10 (지지 열당 이웃 표본, 0 = 전부), SEED_ALL 0 (1 이면 새 블록에 평가한 점 전부를 LP 로), NTOP 5 (탐색할 상위 정수점), CGTOL 1e-3,
#    NRAND 40 (무작위 belief 후보), SB_K 3 (strong branching 후보), PH1 1 (1 단계), ORCL_TL 900 (oracle 호출당 상한), PSTOP 1 (가지치기 기준 가격 멈춤)
#  실행: GRB_LICENSE_FILE=... julia -t 12,1 --project=. docs/nz_ccg_bnp_scripts/ccgbp.jl   (oracle 때문에 interactive 스레드 1 개 필요)
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Serialization
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_belief_ccg.jl")); include(joinpath(root, "nonzero_sum", "nz_belief_ccg_tree.jl"))
envf(k, d) = get(ENV, k, d)
const GEO = parse(Int, envf("GEO", "1")); const RR = parse(Float64, envf("R", "0.2"))
const TL = parse(Float64, envf("TL", "3600")); const TOL = parse(Float64, envf("TOL", "5e-3"))
const MU = parse(Float64, envf("MU", "1e4")); const ETA = parse(Float64, envf("ETA", "1e-4")); const MU_MAX = parse(Float64, envf("MU_MAX", "1e4"))
const NS = parse(Int, envf("NS", "10")); const SEED_ALL = envf("SEED_ALL", "0") == "1"; const NTOP = parse(Int, envf("NTOP", "5"))
const CGTOL = parse(Float64, envf("CGTOL", "1e-3")); const NRAND = parse(Int, envf("NRAND", "40"))
const SB_K = parse(Int, envf("SB_K", "3")); const PH1 = envf("PH1", "1") == "1"; const ORCL_TL = parse(Float64, envf("ORCL_TL", "900")); const PSTOP = envf("PSTOP", "1") == "1"
const LAMU = parse(Float64, envf("LAMU", "1e4"))   # oracle 의 λᵁ (loc_tkp 자체 기본은 100)
nd = make_loc_tkp(; g=GEO, S=parse(Int, envf("S", "10")), seed=1, eps_hat=RR, eps_tilde=RR, delta=RR / 3, beta=0.4, lambdaU=LAMU, NB=parse(Int, envf("NB", "3")))
@printf("  [설정] η %.1e, μ %.1e (상한 %.1e), oracle λᵁ %.1e, μη = %.3g\n", ETA, MU, MU_MAX, LAMU, MU * ETA)
K = nd.meta[:K]; nx = nd.nx; m, ny = nz_m(nd), nz_ny(nd); A = nd.A; XE = nz_xentries(nd); S = nd.S
NT = Threads.nthreads()
@printf("--- ccgbp v6 | SB_K %d PH1 %s ORCL_TL %.0f PSTOP %s\n", SB_K, PH1, ORCL_TL, PSTOP); @printf(" g=%d coupled r=%.2f S=%d, 스레드 %d, tol %.1e, μ %.3g, η %.3g, NS %d, SEED_ALL %s, NTOP %d, CGTOL %.1e\n", GEO, RR, S, NT, TOL, MU, ETA, NS, SEED_ALL, NTOP, CGTOL); flush(stdout)
wall = time()
site(i) = (i - 1) ÷ K + 1
al = findall(nd.x_allowed); sites = unique(site.(al))
# 점 번호는 만난 점에만 붙인다 (v6, 2026-11-23): X 를 미리 나열하지 않음. |X| 는 보고용으로 공식으로만 계산
pts = Vector{Vector{Float64}}(); qv = Float64[]; idx = Dict{Tuple,Int}()
key(z) = Tuple(findall(>(0.5), z))
"점 z 의 번호 (처음 만나면 등록). z 는 X 의 점이어야 함 (호출하는 곳이 실행가능하게 만듦)"
function pid(z)
    k = key(z); c = get(idx, k, 0); c > 0 && return c
    push!(pts, Float64.(z .> 0.5)); push!(qv, dot(nd.qx, pts[end])); idx[k] = length(pts)
    return length(pts)
end
pid(zeros(nx))                                                   # 빈 x = 번호 1
NP = sum(binomial(length(sites), n) * K^n for n in 0:nd.gamma)   # |X| (보고용)
nbrs_cache = Dict{Int,Vector{Int}}()
function nbrs(t)
    get!(nbrs_cache, t) do
        z = pts[t]; on = [(site(i), (i - 1) % K + 1) for i in findall(>(0.5), z)]; out = Int[]
        put(conf) = (w = zeros(nx); for (s, k) in conf; w[(s - 1) * K + k] = 1; end; pid(w))
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


# ---------------- 노드 (분기 고정) ----------------
struct Node
    f0::Set{Int}; f1::Set{Int}; opn::Set{Int}     # x_jk = 0, x_jk = 1, 거점 j 개설 강제
    lb::Float64; est::Float64; depth::Int; tag::String
end
sitevars(j) = [(j - 1) * K + k for k in 1:K if nd.x_allowed[(j - 1) * K + k]]
consistent(c, N) = (z = pts[c]; all(z[i] < 0.5 for i in N.f0) && all(z[i] > 0.5 for i in N.f1) && all(any(z[i] > 0.5 for i in sitevars(j)) for j in N.opn))
"노드와 맞는 정수점 하나 (없으면 0)"
function seedpt(N)
    z = zeros(nx); for i in N.f1; z[i] = 1.0; end
    for j in N.opn
        any(z[i] > 0.5 for i in sitevars(j)) && continue
        ks = [i for i in sitevars(j) if !(i in N.f0)]; isempty(ks) && return 0
        z[ks[1]] = 1.0
    end
    sum(z) <= nd.gamma + 0.5 || return 0
    for j in sites; count(i -> z[i] > 0.5, sitevars(j)) <= 1 || return 0; end
    any(z[i] > 0.5 for i in N.f0) && return 0
    pid(z)
end
"노드 위 X 에서 선형 최소 min cᵀz (탐욕, 정확)"
function linmin(c, N)
    forced = 0.0; nf = 0; opt = Tuple{Float64,Int}[]; z = zeros(nx)
    for j in sites
        sv = sitevars(j); f1j = [i for i in sv if i in N.f1]
        if !isempty(f1j); z[f1j[1]] = 1.0; forced += c[f1j[1]]; nf += 1; continue; end
        av = [i for i in sv if !(i in N.f0)]
        if isempty(av); j in N.opn && return Inf, 0; continue; end
        ib = av[argmin([c[i] for i in av])]
        if j in N.opn; z[ib] = 1.0; forced += c[ib]; nf += 1; else; push!(opt, (c[ib], ib)); end
    end
    nf > nd.gamma && return Inf, 0
    sort!(opt; by=first); v = forced
    for (val, ib) in opt[1:min(nd.gamma - nf, length(opt))]; val < 0 || break; z[ib] = 1.0; v += val; end
    return v, pid(z)
end

# ---------------- 가격 MILP (노드 고정 반영) ----------------
pmods = Dict{Int,Any}()
function pricemip(b, wb, πb, σb, stopv, N; tlim=600.0)
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
    fx = Int[]
    for i in N.f0; is_fixed(xb[i]) || (fix(xb[i], 0.0; force=true); push!(fx, i)); end
    for i in N.f1; is_fixed(xb[i]) || (fix(xb[i], 1.0; force=true); push!(fx, i)); end
    oc = [@constraint(Mp, sum(xb[i] for i in sitevars(j)) >= 1) for j in N.opn]
    @objective(Mp, Min, wb * tb - sum(πb[i] * xb[i] for i in al))
    set_optimizer_attribute(Mp, "BestBdStop", stopv); set_optimizer_attribute(Mp, "TimeLimit", tlim)
    t_mip[] += @elapsed optimize!(Mp); n_mip[] += 1; nodes_mip[] += Int(MOI.get(Mp, MOI.NodeCount()))
    st = termination_status(Mp)
    bnd = st == MOI.INFEASIBLE ? Inf : objective_bound(Mp)
    zs = [pid(round.(value.(xb; result=k))) for k in 1:result_count(Mp)]
    inc = has_values(Mp) ? objective_value(Mp) : Inf
    for i in fx; unfix(xb[i]); end
    for c in oc; delete(Mp, c); end
    return bnd, zs, inc
end

# ---------------- DW 열 생성 (노드) ----------------
cols = Vector{Set{Int}}(); n_rmpfix = Ref(0)
function rmp(N)
    M = Model(() -> Gurobi.Optimizer(envs[NT + 2])); set_silent(M)
    set_optimizer_attribute(M, "Method", 2); set_optimizer_attribute(M, "Crossover", 0)
    nb = length(blks); cN = [[c for c in cols[b] if consistent(c, N)] for b in 1:nb]
    @variable(M, x[al]); @variable(M, t)
    lam = Dict((b, c) => @variable(M, lower_bound = 0) for b in 1:nb for c in cN[b])
    ct = [@constraint(M, t - sum(getL(c, b) * lam[(b, c)] for c in cN[b]) >= 0) for b in 1:nb]
    cl = [Dict(i => @constraint(M, sum(pts[c][i] * lam[(b, c)] for c in cN[b]) - x[i] == 0) for i in al) for b in 1:nb]
    cc = [@constraint(M, sum(lam[(b, c)] for c in cN[b]) == 1) for b in 1:nb]
    @objective(M, Min, sum(nd.qx[i] * x[i] for i in al) + t)
    t_rmp[] += @elapsed optimize!(M); n_rmp[] += 1
    if termination_status(M) != MOI.OPTIMAL       # barrier (crossover 끔) 가 수치 문제로 멈춤 → crossover 켜고 다시 (쌍대 실행가능성이 Farley 하한에 필요)
        n_rmpfix[] += 1
        set_optimizer_attribute(M, "Crossover", -1); t_rmp[] += @elapsed optimize!(M)
        if termination_status(M) != MOI.OPTIMAL
            set_optimizer_attribute(M, "Method", 1); t_rmp[] += @elapsed optimize!(M)
        end
    end
    termination_status(M) == MOI.OPTIMAL || error("ccgbp: RMP $(termination_status(M))")
    w = dual.(ct); π = [[i in al ? dual(cl[b][i]) : 0.0 for i in 1:nx] for b in 1:nb]; σ = dual.(cc)
    supp = [Set(c for c in cN[b] if value(lam[(b, c)]) > 1e-9) for b in 1:nb]
    xv = zeros(nx); for i in al; xv[i] = value(x[i]); end
    objective_value(M), w, π, σ, supp, xv
end
rcv(b, c, w, π, σ) = w[b] * getL(c, b) - dot(π[b], pts[c]) - σ[b]
rng_ns = MersenneTwister(7)
"노드 N 에서 열 생성. 반환 (lb (exact 일 때만, 아니면 −Inf; 실행불가면 Inf), RMP 값, x̃, 마스터 UB, 그 점, 상위 정수점, 정보)"
function dwsolve!(N; exact=true, ubglob=Inf)
    lb = -Inf; ubm = Inf; cbest = 0; it = 0; nex = 0; topl = Int[]; v = NaN; xv = zeros(nx)
    s0 = seedpt(N); s0 == 0 && return Inf, Inf, xv, Inf, 0, Int[], (it = 0, nex = 0)
    evalall!([s0]); for b in eachindex(blks); push!(cols[b], s0); end
    while true
        it += 1
        v, w, π, σ, supp, xv = rmp(N)
        tolabs = 1e-4 * max(1, abs(v))
        need = Tuple{Int,Int}[]; candb = Dict{Int,Vector{Int}}()
        for b in eachindex(blks)
            (w[b] > 1e-9 && blks[b].kind == :lp) || continue
            cs = Set{Int}()
            for c in supp[b]
                push!(cs, c); nb_ = filter(c2 -> consistent(c2, N), nbrs(c))
                union!(cs, (NS == 0 || length(nb_) <= NS) ? nb_ : shuffle(rng_ns, nb_)[1:NS])
            end
            for c in visited; haskey(cache, (c, b)) && consistent(c, N) && push!(cs, c); end
            candb[b] = collect(cs); append!(need, [(c, b) for c in cs])
        end
        evalpairs!(need)
        added = 0
        for b in eachindex(blks)
            if haskey(candb, b)
                c = argmin(c -> rcv(b, c, w, π, σ), candb[b]); r = rcv(b, c, w, π, σ)
            else
                cl = blks[b].kind == :aff ? w[b] .* blks[b].c .- π[b] : -π[b]
                r, c = linmin(cl, N); r += (blks[b].kind == :aff ? w[b] * blks[b].c0 : 0.0) - σ[b]; evalpairs!([(c, b)])
            end
            if r < -tolabs && !(c in cols[b]); push!(cols[b], c); added += 1; end
        end
        added > 0 && continue
        sp = unique([c for b in eachindex(blks) for c in supp[b]]); evalall!(sp); global lastsupp = sp
        for c in sp; mv = mval(c); mv < ubm && (ubm = mv; cbest = c); end
        topl = sort(sp; by=mval)[1:min(NTOP, length(sp))]
        exact || return -Inf, v, xv, ubm, cbest, topl, (it = it, nex = 0)
        lbit = v; added2 = 0
        # 가격 MILP 멈춤 기준: 가지치기에 필요한 여유 (RMP − (UB − tol)) 를 블록에 나눔. 여유가 없으면 엄격 (tolabs)
        nw = count(b -> blks[b].kind == :lp && w[b] > 1e-9, eachindex(blks))
        need = isfinite(ubglob) ? v - (ubglob - TOL * max(1.0, abs(ubglob))) : 0.0
        allow = (PSTOP && need > 0 && nw > 0) ? max(tolabs, 0.9 * need / nw) : tolabs
        for b in eachindex(blks)
            if blks[b].kind == :lp && w[b] > 1e-9
                bnd, zs, inc = pricemip(b, w[b], π[b], σ[b], σ[b] - allow, N)
                lbit += min(0.0, bnd - σ[b])
                if inc - σ[b] < -tolabs
                    evalpairs!([(zc, b) for zc in zs])
                    for zc in zs
                        if consistent(zc, N) && rcv(b, zc, w, π, σ) < -tolabs && !(zc in cols[b]); push!(cols[b], zc); added2 += 1; end
                    end
                end
            else
                cl = blks[b].kind == :aff ? w[b] .* blks[b].c .- π[b] : -π[b]
                r, c = linmin(cl, N); r += (blks[b].kind == :aff ? w[b] * blks[b].c0 : 0.0) - σ[b]; lbit += min(0.0, r)
                if r < -tolabs && !(c in cols[b]); evalpairs!([(c, b)]); push!(cols[b], c); added2 += 1; end
            end
        end
        nex += 1; lb = max(lb, lbit)
        # 노드를 자를 수 있으면 더 조일 필요 없음 (가지치기 기준 = UB − tol)
        (isfinite(ubglob) && lb >= ubglob - TOL * max(1.0, abs(ubglob))) && return lb, v, xv, ubm, cbest, topl, (it = it, nex = nex)
        tgt = min(ubm, ubglob)
        (added2 == 0 || tgt - lb <= CGTOL * max(1, abs(tgt))) && return lb, v, xv, ubm, cbest, topl, (it = it, nex = nex)
    end
end
cinc = 0          # 마스터 incumbent (모든 블록으로 평가된 점 중 최소 mval)
"새 블록의 열 씨앗"
function seed!(b, c=0)
    base = unique(vcat(lastsupp, c > 0 ? [c] : Int[], [1], cinc > 0 ? [cinc] : Int[]))
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
    if MU < MU_MAX && Linf >= v - thr && μstar <= MU_MAX      # μ 가 이미 상한이면 올리지 않고 바로 no-good
        μn = min(MU_MAX, max(2MU, 1.1μstar)); blks[b] = Blk(:lp, copy(d), copy(r), μn, mkgens(d, r, μn), 0.0, Float64[]); n_mu[] += 1
        for k in collect(keys(cache)); k[2] == b && delete!(cache, k); end
        evalpairs!([(c, b)])
        getL(c, b) >= v - thr && (seed!(b, c); return :mu)
    end
    seed!(b, c); nogood!(c, v); return :safeguard
end


# ---------------- CCG + branch-and-price ----------------
ROOT = Node(Set{Int}(), Set{Int}(), Set{Int}(), -Inf, -Inf, 0, "root")
"상위 정수점에서 싼 탐색, belief 하나 넣으면 true"
function trysearch(topl)
    for c in topl
        bb = search(pts[c]); tc = mval(c) - qv[c]
        if bb.d !== nothing && bb.v > tc + 1e-3 * max(1.0, abs(bb.v))
            how = add_and_secure!(c, bb.v, bb.d, bb.r)
            @printf("    [탐색] x=%-24s t %.2f → %.2f (%s), 블록 %d\n", loc_tkp_xstr(nd, pts[c]), tc, bb.v, how, length(blks)); flush(stdout)
            return true
        end
    end
    false
end
UB = Inf; LB = -Inf; xbest = 0; status = :TimeLimit; t_or = 0.0; n_or = 0; certified = Dict{Int,Float64}()   # 점 → oracle 로 확인했을 때의 마스터 값 (값이 오를 때만 다시)
needs_oracle(c) = (evalall!([c]); mval(c) > get(certified, c, -Inf) + 1e-6 * max(1.0, abs(mval(c))))
function upd_inc!(cs)
    for c in cs
        all(b -> blks[b].kind == :aff || haskey(cache, (c, b)), eachindex(blks)) || continue
        (cinc == 0 || mval(c) < mval(cinc) - 1e-9) && (global cinc = c)
    end
end
"oracle (α-B&B) 을 c 에서. 목표는 지금 LB 기준 (끝에 UB − LB ≤ tol 이 되게)"
orLB = Dict{Int,Float64}(); orcap = Dict{Int,Float64}()
orcl_log = NamedTuple[]
function oracle_at!(c; cap=ORCL_TL)
    evalall!([c]); t0v = mval(c) - qv[c]; xb = pts[c]
    tau = isfinite(LB) ? max(t0v + 1e-6 * max(1.0, abs(t0v)), LB + 0.95 * TOL * max(1.0, abs(LB)) - qv[c]) : t0v + 0.9 * TOL * max(1.0, abs(qv[c] + t0v))
    gap_o = min(1e-3, 0.4 * TOL * max(1.0, abs(qv[c] + t0v)) / max(1.0, abs(t0v)))
    lim = max(10.0, min(cap, TL - (time() - wall))); orcap[c] = cap
    to = @elapsed (res = nz_alpha_bnb_dr(nd, xb; nworkers=max(1, NT - 1), time_limit=lim, rel_gap=gap_o, target=tau, verbose=false))
    global t_or += to; global n_or += 1; orLB[c] = max(get(orLB, c, -Inf), res[:LB])
    if qv[c] + res[:UB] < UB; global UB = qv[c] + res[:UB]; global xbest = c; end
    how = :none
    if res[:α] !== nothing && res[:LB] > t0v + 1e-6 * max(1.0, abs(t0v))
        vv, dd, r2, a1 = exact_value(xb, res[:α])
        if dd !== nothing
            how = add_and_secure!(c, vv, dd, r2)
            vv <= t0v + 1e-6 * max(1.0, abs(t0v)) && restricted!(c, a1)
        end
    end
    if how == :none && res[:LB] > t0v + 1e-6 * max(1.0, abs(t0v)); nogood!(c, res[:LB]); how = :nogood_lb; end
    certified[c] = mval(c)
    @printf("  [orcl] x=%-24s V in [%.2f, %.2f], t0 %.2f, 목표 %.2f (%.0fs, 상한 %.0fs) %s, 블록 %d, LB %.2f UB %.2f (경과 %.0fs)\n", loc_tkp_xstr(nd, xb), res[:LB], res[:UB], t0v, tau, to, lim, how, length(blks), LB, UB, time() - wall)
    @printf("         α-B&B: 노드 %d, 이완 %.0fs, 평가 %.0fs, 범위 %.0fs (MIP %d, 실패 %d), 재확인 %.0fs, ipopt %d, 고정 비율 %.2f, 수치 오류 %d\n",
            res[:nodes], res[:t_relax], res[:t_eval], res[:t_range], res[:range_mips], res[:range_fail], res[:t_recheck], res[:ipopt_calls], res[:fixfrac], res[:numerr]); flush(stdout)
    # 재현용 기록 (α-B&B 조정): 같은 인자로 nz_alpha_bnb_dr 를 따로 부를 수 있게
    push!(orcl_log, (g=GEO, S=S, x=copy(xb), tau=tau, gap=gap_o, lim=lim, LB=res[:LB], UB=res[:UB], time=to, nodes=res[:nodes]))
    serialize(@sprintf("orcl_calls_g%d_S%d.jls", GEO, S), orcl_log)
    return how
end

# ---- 시작 ----
b1 = add_lpblk!(qh, qh, MU); evalpairs!([(1, b1)]); push!(cols, Set([1]))
bnd = 0.0; tl0 = @elapsed (global bnd = first(pricemip(b1, 1.0, zeros(nx), 0.0, Inf, ROOT)))
global tlow = bnd - 1e-6 * max(1.0, abs(bnd))
@printf("  [시작] t 하한 %.2f (가격 MILP %.1fs)\n", tlow, tl0); flush(stdout)

# ---- 1 단계 (루트, 휴리스틱 열 생성 + 탐색 + oracle): belief 와 UB 모으기 ----
if PH1
    while time() - wall < TL
        r = dwsolve!(ROOT; exact=false); upd_inc!(r[6])
        @printf("  [ph1] 루트 RMP %.2f, 마스터 UB %.2f (x=%s), 블록 %d, LP 누적 %d (경과 %.0fs)\n", r[2], r[4], loc_tkp_xstr(nd, pts[r[5]]), length(blks), n_lp[], time() - wall); flush(stdout)
        trysearch(r[6]) && continue
        oracle_at!(r[5]) == :none && break
    end
end
t_ph1 = time() - wall
@printf("  [1 단계 끝] %.0fs, belief %d, UB %.2f\n", t_ph1, count(B -> B.kind == :lp, blks), UB); flush(stdout)

# ---- 2 단계: branch-and-price ----
"분기: 거점 개설 o_j (분수) 를 strong branching (휴리스틱 열 생성 값의 곱 점수) 으로, 없으면 크기 x_jk"
function children(N, xv, vpar)
    o = Dict(j => sum(xv[i] for i in sitevars(j)) for j in sites)
    frac = [j for j in sites if !(j in N.opn) && 1e-5 < o[j] < 1 - 1e-5]
    mk_open(j, e) = Node(copy(N.f0), copy(N.f1), union(N.opn, [j]), N.lb, e, N.depth + 1, N.tag * @sprintf(" o%d=1", j))
    mk_close(j, e) = Node(union(N.f0, sitevars(j)), copy(N.f1), copy(N.opn), N.lb, e, N.depth + 1, N.tag * @sprintf(" o%d=0", j))
    if !isempty(frac)
        sort!(frac; by=j -> -min(o[j], 1 - o[j])); cands = frac[1:min(max(SB_K, 1), length(frac))]
        SB_K <= 1 && return (mk_open(cands[1], vpar), mk_close(cands[1], vpar))
        best = nothing; bsc = -Inf
        for j in cands
            r1 = dwsolve!(mk_open(j, vpar); exact=false); r0 = dwsolve!(mk_close(j, vpar); exact=false)
            e1 = r1[2]; e0 = r0[2]; upd_inc!(vcat(r1[6], r0[6]))
            sc = max(e1 - vpar, 1e-6) * max(e0 - vpar, 1e-6)
            @printf("      [sb] o%d (%.3f): 열림 %.2f, 닫힘 %.2f, 점수 %.3g\n", j, o[j], e1, e0, sc); flush(stdout)
            sc > bsc && (bsc = sc; best = (mk_open(j, e1), mk_close(j, e0)))
        end
        return best
    end
    fx = [i for i in al if 1e-5 < xv[i] < 1 - 1e-5]
    isempty(fx) && return nothing
    i = fx[argmax([min(xv[i], 1 - xv[i]) for i in fx])]
    return (Node(copy(N.f0), union(N.f1, [i]), copy(N.opn), N.lb, vpar, N.depth + 1, N.tag * @sprintf(" x%d=1", i)),
            Node(union(N.f0, [i]), copy(N.f1), copy(N.opn), N.lb, vpar, N.depth + 1, N.tag * @sprintf(" x%d=0", i)))
end
closedq = Tuple{Node,Int}[]       # 정수로 닫은 노드와 그 점 (하한 = max(마스터 값, q + oracle 하한))
closed_lb(e) = (evalall!([e[2]]); max(e[1].lb, mval(e[2]), qv[e[2]] + get(orLB, e[2], -Inf)))   # 닫은 뒤 생긴 블록도 먼저 평가 (2026-11-24 지리 14 KeyError)
global_lb(extra=Inf) = min(extra, minimum((o.lb for o in openq); init=Inf), minimum((closed_lb(e) for e in closedq); init=Inf))
function finish_or_deepen!()
    while time() - wall < TL
        if isempty(openq)
            lbg = isfinite(UB) ? min(UB - TOL * max(1.0, abs(UB)), global_lb()) : global_lb()
            global LB = max(LB, lbg)
            if isfinite(UB) && (isempty(closedq) || UB - LB <= TOL * max(1.0, abs(UB)) * (1 + 1e-9)); global status = :Optimal; return; end   # 모든 노드가 UB − tol 로 잘림
            k = argmin([closed_lb(e) for e in closedq]); N, c = closedq[k]
            cap2 = 2 * get(orcap, c, ORCL_TL)
            @printf("  [deepen] 닫힌 노드 %s (점 %s, 하한 %.2f) 이 막음 → oracle 상한 %.0fs 로 다시\n", N.tag, loc_tkp_xstr(nd, pts[c]), closed_lb(closedq[k]), cap2); flush(stdout)
            h = oracle_at!(c; cap=cap2)
            if h != :none
                deleteat!(closedq, k); push!(openq, N)          # 블록이 늘었으니 이 노드를 다시 엶
                return :reopen
            end
            continue
        end
        return :open
    end
end
openq = [ROOT]; nproc = 0; nprune = 0; nexact = 0; nint = 0
gapok(lb) = isfinite(UB) && lb >= UB - TOL * max(1.0, abs(UB))
while time() - wall < TL
while !isempty(openq) && time() - wall < TL
    sort!(openq; by=N -> (N.lb, N.est)); N = popfirst!(openq)
    global LB = max(LB, global_lb(N.lb))
    if isfinite(UB) && UB - LB <= TOL * max(1.0, abs(UB)); global status = :Optimal; break; end
    gapok(N.lb) && (global nprune += 1; continue)
    global nproc += 1
    local r
    while true
        r = dwsolve!(N; exact=false); r[1] == Inf && break
        upd_inc!(r[6]); trysearch(filter(c -> !gapok(mval(c)), r[6])) || break    # 이미 UB − tol 위인 점의 belief 는 가지치기를 못 바꿈
    end
    if r[1] == Inf; global nprune += 1; continue; end            # 실행불가 노드
    v, xv = r[2], r[3]; lbN = N.lb
    integral = all(i -> abs(xv[i] - round(xv[i])) < 1e-6, al)
    if gapok(v) || integral
        r = dwsolve!(N; exact=true, ubglob=UB); global nexact += 1; upd_inc!(r[6])
        lbN = max(lbN, r[1]); v, xv = r[2], r[3]; integral = all(i -> abs(xv[i] - round(xv[i])) < 1e-6, al)
    end
    @printf("  [node %3d] %-34s 깊이 %d: RMP %.2f, LB %.2f, %s, 열린 %d, LB* %.2f UB %.2f, 마스터 inc %.2f, 블록 %d, MILP %d (경과 %.0fs)\n",
            nproc, N.tag, N.depth, v, lbN, integral ? "정수" : "분수", length(openq), LB, UB, cinc > 0 ? mval(cinc) : NaN, length(blks), n_mip[], time() - wall); flush(stdout)
    # oracle: 마스터 incumbent 가 UB 를 고칠 수 있고 지금 블록으로 확인 안 된 경우
    if cinc > 0 && mval(cinc) < UB - 0.1 * TOL * max(1.0, abs(UB)) && needs_oracle(cinc)
        h = oracle_at!(cinc)
        if h != :none; push!(openq, Node(N.f0, N.f1, N.opn, lbN, v, N.depth, N.tag)); continue; end   # 블록이 늘었으니 이 노드를 다시
    end
    gapok(lbN) && (global nprune += 1; continue)
    if integral
        c = pid(round.(xv)); global nint += 1
        if needs_oracle(c)
            h = oracle_at!(c)
            h != :none && (push!(openq, Node(N.f0, N.f1, N.opn, lbN, v, N.depth, N.tag)); continue)
        end
        push!(closedq, (Node(N.f0, N.f1, N.opn, lbN, v, N.depth, N.tag), c))   # 정수 노드: 하한을 남기고 닫음
        continue
    end
    ch = children(N, xv, v)
    ch === nothing && (println("  [!] 분기 변수 없음"); continue)
    push!(openq, ch[1]); push!(openq, ch[2])
end   # while (B&P 노드)
status == :Optimal && break
finish_or_deepen!() == :reopen || break
end   # 바깥
tot = time() - wall
nbel = count(B -> B.kind == :lp, blks)
@printf("RESULT ccgbp g=%d status=%s x=%s LB=%.4f UB=%.4f gap=%.3f%% beliefs=%d 노드 %d (가지침 %d, 정확 %d, 정수 %d, 남은 %d, 닫힌 정수 %d) 안전장치 %d (no-good %d, restricted %d, μ 올림 %d) time=%.1f (1 단계 %.0fs)\n",
        GEO, status, xbest == 0 ? "-" : loc_tkp_xstr(nd, pts[xbest]), LB, UB, 100 * (UB - LB) / abs(UB), nbel, nproc, nprune, nexact, nint, length(openq), length(closedq), n_ng[] + n_rc[], n_ng[], n_rc[], n_mu[], tot, t_ph1)
@printf("  시간: oracle %d 번 %.0fs, 탐색 %d 번 %.0fs, 블록 LP %d 번 %.0fs, 가격 MILP %d 번 %.0fs (노드 %d), RMP %d 번 %.0fs (crossover 재시도 %d), 생성기 구성 %.0fs; 서로 다른 점 %d / %d\n",
        n_or, t_or, n_srch[], t_srch[], n_lp[], t_lp[], n_mip[], t_mip[], nodes_mip[], n_rmp[], t_rmp[], n_rmpfix[], t_build[], length(unique([c for (c, _) in keys(cache)])), NP)
@printf("  열: RMP 에 들어간 열 (블록 합) %d, 서로 다른 열 %d, 번호를 붙인 점 %d / |X| %d\n", sum(length.(cols)), length(union(cols...)), length(pts), NP)
