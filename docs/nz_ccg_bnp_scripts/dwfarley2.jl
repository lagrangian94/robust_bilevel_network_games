# 공동 DW 루트 (블록 11 개가 x 를 공유) 를 휴리스틱 가격 + 마지막에만 정확 가격 (Farley/Lagrangian 하한) 으로 풀 때
# 필요한 정확 가격 (= 가격 MILP 한 번 ≈ 20 s) 횟수와 블록 LP 평가 수를 표 (tab_g1.jls) 로 시뮬레이션.
# 표 조회 = 블록 LP 하나 (정수 z). "정확 가격" = 표 전체 argmin (실제로는 가격 MILP).
root = "C:/Users/user/robust_bilevel_network_games"
D = "C:/Users/user/AppData/Local/Temp/claude/a31bffbb1639accd/74d8cabb-c37c-465f-ab96-877d8b61d634/scratchpad"
using JuMP, Gurobi, Printf, LinearAlgebra, Serialization
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_belief_ccg_tree.jl"))
B = deserialize(joinpath(D, "bel_g1.jls")); Ltab = deserialize(joinpath(D, "tab_g1.jls"))
nd = make_loc_tkp(; g=B.g, S=B.S, seed=B.seed, eps_hat=B.r, eps_tilde=B.r, delta=B.r / 3, beta=0.4)
K = nd.meta[:K]; nx = nd.nx
site(i) = (i - 1) ÷ K + 1
al = findall(nd.x_allowed); sites = unique(site.(al))
pts = [zeros(nx)]
for n in 1:nd.gamma, comb in Iterators.filter(c -> issorted(c) && allunique(c), Iterators.product(ntuple(_ -> sites, n)...)), lv in Iterators.product(ntuple(_ -> 1:K, n)...)
    z = zeros(nx); for (st, k) in zip(comb, lv); z[(st - 1) * K + k] = 1.0; end; push!(pts, z)
end
NP, nb = size(Ltab); @assert NP == length(pts)
Pm = reduce(hcat, pts); qv = [dot(nd.qx, z) for z in pts]
key(z) = Tuple(findall(>(0.5), z)); idx = Dict(key(pts[t]) => t for t in 1:NP)
zst = minimum(qv .+ vec(maximum(Ltab; dims=2)))
# 이웃: 거점 하나 빼기·더하기·크기 바꾸기·옮기기
function nbrs(t)
    z = pts[t]; on = [(site(i), (i - 1) % K + 1) for i in findall(>(0.5), z)]; out = Int[]
    put(conf) = (w = zeros(nx); for (s, k) in conf; w[(s - 1) * K + k] = 1; end; get(idx, key(w), 0))
    for a in eachindex(on)
        rest = [on[b] for b in eachindex(on) if b != a]; push!(out, put(rest))
        for s in sites, k in 1:K
            any(o -> o[1] == s, rest) && continue; push!(out, put(vcat(rest, [(s, k)])))
        end
    end
    if length(on) < nd.gamma
        for s in sites, k in 1:K; any(o -> o[1] == s, on) && continue; push!(out, put(vcat(on, [(s, k)]))); end
    end
    filter(>(0), unique(out))
end
genv = Gurobi.Env()
cols = [Set([1]) for _ in 1:nb]                    # 블록마다 열 (∅ 로 시작)
evald = Set{Int}([1]); evbl = Set{Tuple{Int,Int}}()                              # 블록 LP 로 평가한 정수점 (점 하나 = 블록 nb 개 LP)
function rmp()
    M = Model(() -> Gurobi.Optimizer(genv)); set_silent(M)
    @variable(M, x[al]); @variable(M, t)
    lam = Dict((b, c) => @variable(M, lower_bound = 0) for b in 1:nb for c in cols[b])
    ct = [@constraint(M, t - sum(Ltab[c, b] * lam[(b, c)] for c in cols[b]) >= 0) for b in 1:nb]
    cl = [Dict(i => @constraint(M, sum(Pm[i, c] * lam[(b, c)] for c in cols[b]) - x[i] == 0) for i in al) for b in 1:nb]
    cc = [@constraint(M, sum(lam[(b, c)] for c in cols[b]) == 1) for b in 1:nb]
    @objective(M, Min, sum(nd.qx[i] * x[i] for i in al) + t); optimize!(M)
    w = dual.(ct); π = [Dict(i => dual(cl[b][i]) for i in al) for b in 1:nb]; σ = dual.(cc)
    supp = Set(c for b in 1:nb for c in cols[b] if value(lam[(b, c)]) > 1e-9)
    objective_value(M), w, π, σ, supp
end
rc(b, c, w, π, σ) = w[b] * Ltab[c, b] - sum(π[b][i] * Pm[i, c] for i in al) - σ[b]
nexact = 0; it = 0; lb_best = -Inf
while true
    global it += 1
    v, w, π, σ, supp = rmp()
    # 휴리스틱 가격 (v2): w_b > 0 블록만 지지 열 (그 블록의) 이웃을 블록 LP 로 평가, w_b = 0 블록은 X 위 선형 가격 (L_b 불필요)
    added = 0
    for b in 1:nb
        if w[b] > 1e-9
            sb = [c for c in cols[b] if c in supp]
            cand = Set{Int}(); for c in sb; push!(cand, c); union!(cand, nbrs(c)); end
            for c in cand; push!(evbl, (c, b)); end
            best = argmin(c -> rc(b, c, w, π, σ), collect(cand))
        else
            best = argmin([rc(b, c, w, π, σ) for c in 1:NP])          # w_b = 0: 선형, 실제로는 탐욕/작은 LP
            push!(evbl, (best, b))
        end
        if rc(b, best, w, π, σ) < -1e-7 * max(1, abs(v)); push!(cols[b], best); added += 1; end
    end
    union!(evald, [c for (c, _) in evbl])
    if added == 0
        # 정확 가격: w_b > 0 인 블록만 MILP 필요 (w_b = 0 이면 X 위 선형 최소, 쉬움)
        hard = [b for b in 1:nb if w[b] > 1e-9]; global nexact += length(hard)
        lb = v; added2 = 0
        for b in 1:nb
            r = [rc(b, c, w, π, σ) for c in 1:NP]; c = argmin(r); lb += min(0.0, r[c])
            if r[c] < -1e-7 * max(1, abs(v)); push!(cols[b], c); added2 += 1; end
        end
        global lb_best = max(lb_best, lb)
        @printf("it %3d: RMP %.2f  Farley LB %.2f (z* 대비 %.3f%%)  정확 가격 블록 %d (누적 %d), 개선 열 %d, 평가 점 %d\n",
                it, v, lb, 100 * (zst - lb) / abs(zst), length(hard), nexact, added2, length(evald)); flush(stdout)
        added2 == 0 && break
    elseif it % 10 == 0
        @printf("it %3d: RMP %.2f, 평가 점 %d\n", it, v, length(evald)); flush(stdout)
    end
end
@printf("끝: z* = %.2f, 천장 −1,633.94, 반복 %d, 정확 가격 %d 번, 서로 다른 점 %d, 블록 수 %d, (점,블록) LP 평가 %d (|X| = %d)\n",
        zst, it, nexact, length(evald), nb, length(evbl), NP)
