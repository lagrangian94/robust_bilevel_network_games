# 공동 DW 루트 + Farley 하한, 실제 블록 LP 와 실제 가격 MILP 로 (지리 1 전체 X, 최종 belief 11 개 고정) — 2026-11-23
#  RMP: min qᵀx + t,  t ≥ Σ_z λ^b_z L_b(z) (b),  Σ_z λ^b_z z = x (b),  Σ_z λ^b_z = 1 (b)
#  휴리스틱 가격: w_b > 0 블록 = 그 블록 지지 열의 이웃을 블록 LP 로 평가, w_b = 0 블록 = X 위 선형 가격 (탐욕, LP 없음)
#  정확 가격: w_b > 0 블록마다 가격 MILP min w_b L_b(z) − π_bᵀz (블록 + 이진 z), BestBdStop = σ_b − tol 로 "음의 환원비용 없음" 만 증명
#  Farley 하한 = RMP + Σ_b min(0, 가격 하한 − σ_b).  표 (tab_g1.jls) 는 끝난 뒤 검증에만 씀.
root = "C:/Users/user/robust_bilevel_network_games"
D = "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/74d8cabb-c37c-465f-ab96-877d8b61d634/scratchpad"
using JuMP, Gurobi, Printf, LinearAlgebra, Serialization
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_belief_ccg_tree.jl"))
B = deserialize(joinpath(D, "bel_g1.jls"))
nd = make_loc_tkp(; g=B.g, S=B.S, seed=B.seed, eps_hat=B.r, eps_tilde=B.r, delta=B.r / 3, beta=0.4)
K = nd.meta[:K]; nx = nd.nx; m, ny = nz_m(nd), nz_ny(nd); A = nd.A; XE = nz_xentries(nd)
nb = length(B.beliefs); NT = Threads.nthreads()
site(i) = (i - 1) ÷ K + 1
al = findall(nd.x_allowed); sites = unique(site.(al))
cards = get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[])
# 정수점 색인 (조합 색인일 뿐, 평가는 아님)
pts = [zeros(nx)]
for n in 1:nd.gamma, comb in Iterators.filter(c -> issorted(c) && allunique(c), Iterators.product(ntuple(_ -> sites, n)...)), lv in Iterators.product(ntuple(_ -> 1:K, n)...)
    z = zeros(nx); for (st, k) in zip(comb, lv); z[(st - 1) * K + k] = 1.0; end; push!(pts, z)
end
NP = length(pts); key(z) = Tuple(findall(>(0.5), z)); idx = Dict(key(pts[t]) => t for t in 1:NP)
function nbrs(t)
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
"X 위 선형 최소 min cᵀz (거점당 단계 하나, 거점 ≤ γ): 탐욕"
function linmin(c)
    best = [(minimum(c[(s - 1) * K + k] for k in 1:K), argmin([c[(s - 1) * K + k] for k in 1:K]), s) for s in sites]
    sort!(best; by=first); z = zeros(nx); v = 0.0
    for (val, k, s) in best[1:min(nd.gamma, length(best))]; val < 0 || break; z[(s - 1) * K + k] = 1; v += val; end
    return v, idx[key(z)]
end
# 블록 생성기: 스레드마다 블록 11 개 사본
envs = [Gurobi.Env() for _ in 1:NT + 2]
lp_opt = () -> (o = Gurobi.Optimizer(envs[end]); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
C = (nd = nd, eta = 0.1, vub = false, wd = false, XE = XE, lp_opt = lp_opt,
     rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m], colA = [[(k, A[k, j]) for k in 1:m if A[k, j] != 0] for j in 1:ny],
     ents_of_row = [[e for e in XE if e.k == k] for k in 1:m])
tbuild = @elapsed gens = [[_bccg_gen(C, bl.d, bl.r, bl.μ, envs[th], th) for bl in B.beliefs] for th in 1:NT]
cache = Dict{Tuple{Int,Int},Float64}(); n_lp = Ref(0); t_lp = Ref(0.0)
"(점, 블록) 쌍들을 블록 LP 로 평가 (캐시 밖만, 스레드 병렬)"
function evalpairs!(prs)
    todo = unique([p for p in prs if !haskey(cache, p)]); isempty(todo) && return
    out = Vector{Float64}(undef, length(todo))
    t_lp[] += @elapsed begin
        ch = Channel{Int}(length(todo)); for i in eachindex(todo); put!(ch, i); end; close(ch)
        @sync for th in 1:NT
            Threads.@spawn for i in ch
                c, b = todo[i]; out[i] = first(_bccg_gen_eval!(gens[th][b], pts[c]))
            end
        end
    end
    for i in eachindex(todo); cache[todo[i]] = out[i]; end; n_lp[] += length(todo)
end
# 가격 MILP (블록마다 하나, 목적만 바꿔 다시 씀)
pmods = Dict{Int,Any}(); t_mip = Ref(0.0); n_mip = Ref(0); nodes_mip = Ref(0)
function pricemip(b, wb, πb, σb, tolabs)
    if !haskey(pmods, b)
        Mp = Model(() -> (o = Gurobi.Optimizer(envs[NT + 1]); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), NT); o))
        @variable(Mp, xb[1:nx], Bin); @variable(Mp, tb >= -1e7); @constraint(Mp, sum(xb) <= nd.gamma)
        for (ix, rhs) in cards; @constraint(Mp, sum(xb[i] for i in ix) <= rhs); end
        for i in 1:nx; nd.x_allowed[i] || fix(xb[i], 0.0; force=true); end
        bl = B.beliefs[b]; _bccg_add_block!(merge(C, (M = Mp, x = xb, t0 = tb)), bl.d, bl.r, bl.μ)
        set_optimizer_attribute(Mp, "MIPGap", 1e-6); set_optimizer_attribute(Mp, "TimeLimit", 300.0)
        set_optimizer_attribute(Mp, "IntFeasTol", 1e-9); set_optimizer_attribute(Mp, "IntegralityFocus", 1)   # 솔버 (nz_belief_ccg_tree) 와 같게
        pmods[b] = (Mp, xb, tb)
    end
    Mp, xb, tb = pmods[b]
    @objective(Mp, Min, wb * tb - sum(πb[i] * xb[i] for i in al))
    set_optimizer_attribute(Mp, "BestBdStop", σb - tolabs)
    t_mip[] += @elapsed optimize!(Mp); n_mip[] += 1; nodes_mip[] += Int(MOI.get(Mp, MOI.NodeCount()))
    bnd = objective_bound(Mp)
    zc = has_values(Mp) ? idx[key(round.(value.(xb)))] : 0
    return bnd - σb, zc, (has_values(Mp) ? objective_value(Mp) - σb : Inf), termination_status(Mp)
end
# RMP
cols = [Set([1]) for _ in 1:nb]; evalpairs!([(1, b) for b in 1:nb])
if get(ENV, "RESUME", "0") == "1" && isfile("dw_ckpt.jls")
    ck = deserialize("dw_ckpt.jls"); merge!(cache, ck.cache); cols = ck.cols; println("체크포인트에서 시작: 캐시 $(length(cache)), 열 $(sum(length.(cols)))")
end
t_rmp = Ref(0.0)
function rmp()
    M = Model(() -> Gurobi.Optimizer(envs[NT + 2])); set_silent(M)
    set_optimizer_attribute(M, "Method", 2); set_optimizer_attribute(M, "Crossover", 0)    # 중심 쌍대 (퇴화 진동 줄임)
    @variable(M, x[al]); @variable(M, t)
    lam = Dict((b, c) => @variable(M, lower_bound = 0) for b in 1:nb for c in cols[b])
    ct = [@constraint(M, t - sum(cache[(c, b)] * lam[(b, c)] for c in cols[b]) >= 0) for b in 1:nb]
    cl = [Dict(i => @constraint(M, sum(pts[c][i] * lam[(b, c)] for c in cols[b]) - x[i] == 0) for i in al) for b in 1:nb]
    cc = [@constraint(M, sum(lam[(b, c)] for c in cols[b]) == 1) for b in 1:nb]
    @objective(M, Min, sum(nd.qx[i] * x[i] for i in al) + t)
    t_rmp[] += @elapsed optimize!(M)
    w = dual.(ct); π = [[i in al ? dual(cl[b][i]) : 0.0 for i in 1:nx] for b in 1:nb]; σ = dual.(cc)
    supp = [Set(c for c in cols[b] if value(lam[(b, c)]) > 1e-9) for b in 1:nb]
    objective_value(M), w, π, σ, supp
end
rcv(b, c, w, π, σ) = w[b] * cache[(c, b)] - dot(π[b], pts[c]) - σ[b]
@printf("블록 생성기 %d 스레드 × %d 개 생성 %.1f s\n", NT, nb, tbuild); flush(stdout)
wall = time(); it = 0; lb = -Inf; v = NaN; tolrel = 1e-4
while true
    global it += 1
    global v, w, π, σ, supp = rmp()
    tolabs = tolrel * max(1, abs(v))
    # 휴리스틱 가격
    need = Tuple{Int,Int}[]; candb = Dict{Int,Vector{Int}}()
    for b in 1:nb
        w[b] > 1e-9 || continue
        cs = Set{Int}(); for c in supp[b]; push!(cs, c); union!(cs, nbrs(c)); end
        candb[b] = collect(cs); append!(need, [(c, b) for c in cs])
    end
    evalpairs!(need)
    added = 0
    for b in 1:nb
        if w[b] > 1e-9
            c = argmin(c -> rcv(b, c, w, π, σ), candb[b])
            r = rcv(b, c, w, π, σ)
        else
            r, c = linmin(-π[b]); r -= σ[b]; evalpairs!([(c, b)])
        end
        if r < -tolabs && !(c in cols[b]); push!(cols[b], c); added += 1; end
    end
    it % 10 == 0 && (@printf("it %3d: RMP %.2f, LP %d (%.0f s), %.0f s\n", it, v, n_lp[], t_lp[], time() - wall); flush(stdout))
    added > 0 && continue
    # 정확 가격 (w_b > 0 블록만 MILP)
    isfile("dw_ckpt.jls") || serialize("dw_ckpt.jls", (cache = cache, cols = cols))
    if @isdefined(lastadd)
        for (b, c) in lastadd; @printf("  [진단] 지난 라운드에 넣은 열 (블록 %d, 점 %d): 새 쌍대에서 rc %.3f, RMP 에 있음 %s\n", b, c, rcv(b, c, w, π, σ), c in cols[b]); end
        @printf("  [진단] 쌍대 변화: |Δw| %.3e, |Δπ| %.3e, |Δσ| %.3e\n", norm(w - lastdual[1]), norm(reduce(vcat, π) - reduce(vcat, lastdual[2])), norm(σ - lastdual[3]))
    end
    global lastadd = Tuple{Int,Int}[]; global lastdual = (copy(w), deepcopy(π), copy(σ))
    lbit = v; added2 = 0; rep = String[]
    for b in 1:nb
        if w[b] > 1e-9
            rb, zc, rinc, st = pricemip(b, w[b], π[b], σ[b], tolabs)
            lbit += min(0.0, rb); push!(rep, @sprintf("b%d:%s %.2f", b, st == MOI.OBJECTIVE_LIMIT ? "증명" : string(st), rb))
            if rinc < -tolabs && zc > 0
                evalpairs!([(zc, b)]); rtrue = rcv(b, zc, w, π, σ)
                push!(rep, @sprintf("(열 rc MILP %.2f / LP %.2f%s)", rinc, rtrue, zc in cols[b] ? ", 이미 있음" : ""))
                if rtrue < -tolabs && !(zc in cols[b]); push!(cols[b], zc); added2 += 1; push!(lastadd, (b, zc)); end
            end
        else
            r, c = linmin(-π[b]); lbit += min(0.0, r - σ[b])
        end
    end
    global lb = max(lb, lbit)
    # 정수 상한: 지지 열에서 모든 블록을 블록 LP 로
    sp = unique([c for b in 1:nb for c in supp[b]]); evalpairs!([(c, b) for c in sp for b in 1:nb])
    ubi = minimum(dot(nd.qx, pts[c]) + maximum(cache[(c, b)] for b in 1:nb) for c in sp)
    global ubbest = min(@isdefined(ubbest) ? ubbest : Inf, ubi)
    @printf("it %3d: RMP %.2f  Farley LB %.2f (최선 %.2f), 정수 UB %.2f, 간격 %.3f%%  [%s]  새 열 %d, MILP 누적 %d (%.0f s, 노드 %d), LP %d (%.0f s), 경과 %.0f s\n",
            it, v, lbit, lb, ubbest, 100 * (ubbest - lb) / abs(ubbest), join(rep, ", "), added2, n_mip[], t_mip[], nodes_mip[], n_lp[], t_lp[], time() - wall); flush(stdout)
    if !@isdefined(t_half) && ubbest - lb <= 5e-3 * abs(ubbest)
        global t_half = time() - wall; @printf("  ★ CCG 허용오차 0.5%% 도달: 경과 %.0f s, MILP %d 번\n", t_half, n_mip[]); flush(stdout)
    end
    added2 == 0 && break
end
twall = time() - wall
# 정수 상한: RMP 지지 열과 평가된 점 중 상위 몇 개에서 모든 블록 평가
evald = unique([c for (c, _) in keys(cache)])
sc(c) = maximum(get(cache, (c, b), -Inf) for b in 1:nb)
top = sort(evald; by=c -> dot(nd.qx, pts[c]) + sc(c))[1:min(30, length(evald))]
evalpairs!([(c, b) for c in top for b in 1:nb])
ub, cub = findmin(c -> dot(nd.qx, pts[c]) + maximum(cache[(c, b)] for b in 1:nb), top)
# 검증 (표)
Ltab = deserialize(joinpath(D, "tab_g1.jls"))
dev = maximum(abs(cache[(c, b)] - Ltab[c, b]) for (c, b) in keys(cache))
zst = minimum([dot(nd.qx, pts[t]) for t in 1:NP] .+ vec(maximum(Ltab; dims=2)))
@printf("\n끝 (%.0f s): Farley LB %.2f, 정수 UB %.2f (x = %s), 간격 %.3f%%\n", twall, lb, ub, string(key(pts[top[cub]])), 100 * (ub - lb) / abs(ub))
@printf("  표 검증: 정수 최적 %.2f, 천장 −1,633.94, 캐시 값과 표의 최대 차 %.2e\n", zst, dev)
@printf("  시간: 블록 LP %d 번 %.0f s (스레드 %d), 가격 MILP %d 번 %.0f s (노드 %d), RMP %d 번 %.0f s\n", n_lp[], t_lp[], NT, n_mip[], t_mip[], nodes_mip[], it, t_rmp[])
@printf("  서로 다른 점 %d (|X| = %d), 블록 생성 %.0f s 는 제외\n", length(evald), NP, tbuild)
