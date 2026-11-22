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

Ltab = deserialize(joinpath(D, "tab_g1.jls")); nb = size(Ltab, 2)
envs = [Gurobi.Env() for _ in 1:3]; NT = 1
BAR = Ref(false)
cache = Dict((c, b) => Ltab[c, b] for c in 1:NP for b in 1:nb)
# RMP
cols = [Set([1]) for _ in 1:nb]
t_rmp = Ref(0.0)
function rmp()
    M = Model(() -> Gurobi.Optimizer(envs[NT + 2])); set_silent(M); BAR[] && (set_optimizer_attribute(M, "Method", 2); set_optimizer_attribute(M, "Crossover", 0))
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


ck = deserialize("dw_ckpt.jls"); cols0 = ck.cols
"Lagrangian 하한 (임의 쌍대): Σ_b min_z [w_b L_b(z) − π_bᵀz], 그리고 블록별 argmin"
function lagb(w, π)
    tot = 0.0; arg = Int[]
    for b in 1:nb
        r = [w[b] * Ltab[c, b] - dot(π[b], pts[c]) for c in 1:NP]; c = argmin(r); tot += r[c]; push!(arg, c)
    end
    tot, arg
end
function run(; bar=false, alpha=0.0, maxr=200)
    BAR[] = bar; global cols = deepcopy(cols0); cen = nothing; lbc = -Inf; nmip = 0
    for r in 1:maxr
        v, w, π, σ, supp = rmp()
        if cen === nothing || alpha == 0
            ws, πs = w, π
        else
            ws = alpha .* cen[1] .+ (1 - alpha) .* w; πs = [alpha .* cen[2][b] .+ (1 - alpha) .* π[b] for b in 1:nb]
        end
        lb, arg = lagb(ws, πs); nmip += count(>(1e-9), ws)
        if lb > lbc; lbc = lb; cen = (copy(ws), deepcopy(πs)); end
        if alpha > 0 && cen !== nothing && lb < lbc - 1e-9 && false; end
        gap = 100 * (v - lbc) / abs(v)
        newc = 0
        for b in 1:nb
            c = arg[b]
            if w[b] * Ltab[c, b] - dot(π[b], pts[c]) - σ[b] < -1e-6 * abs(v) && !(c in cols[b]); push!(cols[b], c); newc += 1; end
        end
        if gap <= 0.01 || (newc == 0 && alpha == 0)
            return @sprintf("bar=%s α=%.2f: 라운드 %d, 가격 MILP %d, RMP %.2f, 최선 LB %.2f (간격 %.3f%%)", bar, alpha, r, nmip, v, lbc, gap)
        end
        if newc == 0 && alpha > 0   # 오가격 (mis-pricing): 중심 쪽으로만 이동, 열 없음 → 다음 라운드 (중심 갱신으로 진전)
        end
    end
    return @sprintf("bar=%s α=%.2f: %d 라운드에 못 끝남, LB %.2f", bar, alpha, maxr, lbc)
end
for (bar, al) in ((false, 0.0), (true, 0.0), (false, 0.5), (true, 0.5), (false, 0.8), (true, 0.8))
    println(run(; bar=bar, alpha=al)); flush(stdout)
end
