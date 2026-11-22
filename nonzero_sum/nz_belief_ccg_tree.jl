"""
nz_belief_ccg_tree.jl — belief 블록 CCG 의 한 나무 (single-tree) 판 (2026-11-21).

nz_belief_ccg (마스터 안에 블록 사본을 그대로 넣는 판) 과 같은 블록 L_b(x) 를 쓰되, 블록을 마스터에서 빼서 cut 생성기로 둔다.
  블록 생성기: belief b 마다 LP 하나 (x 는 고정 등식 x = x̄ 로 둔 변수). 값 v_b(x̄) 와 고정 등식의 쌍대 g_b 로
               cut  t0 ≥ v_b(x̄) + g_bᵀ(x − x̄).  블록 LP 의 값은 x (우변) 에 볼록이라 모든 x ∈ [0,1]ⁿ 에서 유효, 정수 x̄ 에서는 정확.
  마스터: min q_xᵀx + t0, x 이진, t0 ≥ (cut 들), 안전장치 (no-good, restricted ISP cut).
    루트: LP 완화에서 Kelley 반복 (root_rounds 번까지) — 블록마다 분리가 수렴하면 명시 블록 판의 루트 LP 값과 같음.
    정수 후보 (lazy callback): 모든 생성기를 x̄ 에서 풀어 위반 cut 을 넣음. 위반이 없으면 싼 belief 탐색 (nz_belief_ccg 의 search),
      새 belief 가 x̄ 에서 값을 올리면 나무 안에서 바로 생성기로 추가하고 cut (x̄ 확보: μ 올리기 → 안 되면 no-good).
    분수 노드 (user cut callback): 처음 ucut_calls 번만 분리 (노드마다 분리하면 노드 비용이 커짐).
  나무 밖: 나무 안에서 belief 가 하나도 안 늘었으면 incumbent 에서 α-B&B oracle (nz_alpha_bnb_dr) → UB, 새 belief.
           belief 가 늘었으면 (incumbent 의 값이 옛 블록 기준이라) 지금까지의 cut 을 모두 실제 제약으로 넣고 다시 푼다.

유효성: 나무 중간에 블록이 늘어도 노드 하한은 그 시점 블록의 하한이라 최종 블록 집합에서도 유효 → objective_bound 는 유효한 LB.
        incumbent 의 값은 옛 블록 기준일 수 있어 (그래서 다시 풂) UB 는 oracle 로만 갱신.
한계 (명시 블록 판 대비): 노드 하한이 cut 근사라 명시 블록 판보다 약하거나 같음 (분리를 노드마다 수렴시키면 같음) → 나무는 같거나 큼.
        이득은 블록이 마스터 노드 LP 에 안 들어가 노드 LP 가 작다는 것과, belief 가 나무 안에서 늘어 나무를 다시 푸는 횟수가 줄 수 있다는 것.
        cut 계수는 μ·McCormick 상한 크기라 클 수 있음. MW (Pareto-optimal cut) 는 이 판에 없음.

인자 (nz_belief_ccg 와 같은 것은 같은 뜻): time_limit, tol, mu, eta, mu_max, safeguard, master_gap, wd, nrand, threads, seed, verbose
  | 인자          | 기본 | 뜻 |
  |---------------|------|----|
  | root_rounds   | 30   | 루트 LP Kelley 반복 상한 (처음과 나무를 다시 풀 때마다) |
  | ucut_calls    | 0    | 분수 노드 분리를 하는 user cut callback 호출 수 상한 |
  | cut_tol       | 1e-6 | cut 위반 판정 (상대) |
  | parallel      | true | 생성기 평가를 묶음별 task 로 병렬 (콜백 안 포함). false 면 차례로 |
  | lazy_attr     | 3    | 콜백 cut 을 실제 제약으로 옮길 때 Gurobi 의 Lazy 속성 (0 = 보통 제약, 1·2·3 = Gurobi Lazy, 3 은 분수 LP 해도 자름).
  |               |      | 2026-11-21 전체 X: 다시 푸는 나무 한 번 476~501 s (0) → 125~130 s (3), 노드 수는 비슷 → 기본 3 |
  | chain         | false| 사슬 submodular cut (Nemhauser–Wolsey 의 부분집합 사슬 꼴, 문서 15 절): t0 ≥ L(S) − Σ_{j∈S} ρ̄_j(S)(1 − x_j) + Σ_{j∉S} ρ_j(∅) x_j.
  |               |      | ρ̄_j(S) = max_{U⊆S, j∈U} [L(U) − L(U−j)], ρ_j(∅) = L({j}) − L(∅). 유효 조건: 모든 V ∈ X 에서 ρ_j(V) ≥ ρ_j(∅) (첫 원소의 이득이 가장 큼).
  |               |      | 이 조건은 증명이 없고 측정으로만 확인 → 정수 후보마다 그 점의 부분집합에서 검사하고 깨지면 오류. belief 마다 S = ∅ 와 원소 쌍의 cut 을 넣고,
  |               |      | 위반 블록이 있는 정수 후보에서는 LP cut 과 함께 그 점의 사슬 cut 을 넣음 |
  | mw_eps        | 0.0  | > 0 이면 정수 후보에서 위반 cut 마다 섭동점 x̂ = x̄ + mw_eps (x⁰ − x̄) 의 쌍대로 cut 을 하나 더 (x⁰ = conv X 안쪽 점).
  |               |      | 모든 x 에서 유효, x̄ 에서는 조금 낮을 수 있음 (그래서 x̄ 에서 정확한 cut 과 함께 넣음). 문서 12.5 |
결과 Dict 의 :impl = "nz_belief_ccg_tree".
"""

isdefined(Main, :nz_belief_ccg) || include(joinpath(@__DIR__, "nz_belief_ccg.jl"))

mutable struct BCCGGen
    P::Model; xv::Vector{VariableRef}; fixc; d::Vector{Float64}; r::Vector{Float64}; μ::Float64; grp::Int
    cache::Dict{Vector{Float64},Float64}       # 정수점의 블록 값 (사슬 cut 용)
end

"belief (d, r), 벌점 μ 의 블록 생성기 LP 를 env 위에 만든다"
function _bccg_gen(C, d, r, μ, env, grp)
    nx = C.nd.nx
    P = Model(() -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o))
    xv = @variable(P, [1:nx]); tP = @variable(P)
    fixc = @constraint(P, [i=1:nx], xv[i] == 0.0)
    _bccg_add_block!(merge(C, (M = P, x = xv, t0 = tP)), d, r, μ)
    @objective(P, Min, tP)
    return BCCGGen(P, xv, fixc, copy(d), copy(r), μ, grp, Dict{Vector{Float64},Float64}())
end

"생성기를 x̄ 에서 풀어 (값, 기울기)"
function _bccg_gen_eval!(G::BCCGGen, x̄)
    for i in eachindex(x̄); set_normalized_rhs(G.fixc[i], x̄[i]); end
    optimize!(G.P)
    termination_status(G.P) == MOI.OPTIMAL || error("nz_belief_ccg_tree: 블록 생성기 LP $(termination_status(G.P))")
    return objective_value(G.P), [dual(G.fixc[i]) for i in eachindex(x̄)]
end

function nz_belief_ccg_tree(nd::NZData; time_limit=3600.0, tol=5e-3, mu=10.0, eta=0.1, mu_max=1e4, safeguard::Symbol=:nogood,
                            master_gap=1e-3, wd::Bool=false, nrand::Int=40, root_rounds::Int=30, ucut_calls::Int=0, cut_tol=1e-6, parallel::Bool=true, lazy_attr::Int=3, mw_eps=0.0, chain::Bool=false,
                            threads::Int=Threads.nthreads(), seed::Int=2026, verbose::Bool=true)
    safeguard in (:nogood, :restricted, :both) || error("nz_belief_ccg_tree: safeguard = :nogood | :restricted | :both")
    wall = time()
    m, ny, nh, nx, S = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.nx, nd.S; A = nd.A; XE = nz_xentries(nd)
    NG = threads                                       # 생성기 묶음 수 (묶음 하나 = env 하나 = task 하나)
    envs = _nz_envs(threads + 2)                       # 1..threads: 생성기·탐색 평가기, threads+1: 마스터, threads+2: 검사 LP
    lp_opt = () -> (o = Gurobi.Optimizer(envs[end]); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
    pool_ev = Channel{BCCGEval}(threads); for k in 1:threads; put!(pool_ev, BCCGEval(nd, envs[k])); end
    M = Model(() -> (o = Gurobi.Optimizer(envs[threads + 1]); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), threads); o))
    set_optimizer_attribute(M, "IntFeasTol", 1e-9); set_optimizer_attribute(M, "IntegralityFocus", 1)
    @variable(M, x[1:nx], Bin); @variable(M, t0 >= -1e7)
    @constraint(M, sum(x) <= nd.gamma)
    for (idx, rhs) in get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[]); @constraint(M, sum(x[i] for i in idx) <= rhs); end
    for i in 1:nx; nd.x_allowed[i] || fix(x[i], 0.0; force=true); end
    @objective(M, Min, dot(nd.qx, x) + t0)
    C = (nd = nd, eta = Float64(eta), vub = false, wd = wd, XE = XE, lp_opt = lp_opt,
         rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m], colA = [[(k, A[k, j]) for k in 1:m if A[k, j] != 0] for j in 1:ny],
         ents_of_row = [[e for e in XE if e.k == k] for k in 1:m])
    verbose && @printf("  [설정] nz_belief_ccg_tree: μ %.3g, η %.3g, μ 상한 %.3g, 안전장치 %s, 마스터 gap %.0e, 블록 약쌍대 %s, 루트 Kelley %d, user cut 호출 %d, 무작위 후보 %d, Lazy 속성 %d, MW 섭동 %s, 사슬 cut %s, 스레드 %d\n",
                       mu, eta, mu_max, safeguard, master_gap, wd ? "켬" : "끔", root_rounds, ucut_calls, nrand, lazy_attr, mw_eps > 0 ? @sprintf("%.0e", mw_eps) : "끔", chain ? "켬 (유효 조건은 측정으로만 확인, 후보마다 검사)" : "끔", threads); flush(stdout)

    # MW 섭동의 안쪽 점 x⁰ (conv X 안쪽)
    nsites_al = count(st -> any(nd.x_allowed[(st - 1) * nd.meta[:K] + k] for k in 1:nd.meta[:K]), 1:nd.meta[:nB])
    x0 = [nd.x_allowed[i] ? 0.5 * min(1 / nd.meta[:K], nd.gamma / max(1, nsites_al) / nd.meta[:K]) : 0.0 for i in 1:nx]
    n_mwcut = Ref(0)
    # belief 후보 (nz_belief_ccg 와 같음)
    qh = nd.q_hat; bx = nd.meta[:box]
    inset(d) = all(d .>= bx.dl .- 1e-12) && all(d .<= bx.du .+ 1e-12) && sum(abs.(d .- qh)) <= 2nd.eps_tilde + 1e-12
    pairbels = Vector{Vector{Float64}}()
    for i in 1:S, j in 1:S
        i == j && continue
        mv = min(qh[i] - bx.dl[i], bx.du[j] - qh[j], nd.eps_tilde); mv <= 1e-9 && continue
        dd = copy(qh); dd[i] -= mv; dd[j] += mv; inset(dd) && push!(pairbels, dd)
    end
    rng = MersenneTwister(seed)
    function randbel()
        d = copy(qh)
        for t in 1:rand(rng, 1:4)
            i, j = randperm(rng, S)[1:2]; mv = rand(rng) * min(d[i] - bx.dl[i], bx.du[j] - d[j]); dd = copy(d); dd[i] -= mv; dd[j] += mv; inset(dd) && (d = dd)
        end
        return d
    end

    # 생성기
    gens = BCCGGen[]; Dd = Vector{Vector{Float64}}()
    t_gen = Ref(0.0); n_geneval = Ref(0)
    function new_gen!(d, r, μ)
        grp = mod1(length(gens) + 1, NG)
        G = _bccg_gen(C, d, r, μ, envs[grp], grp); push!(gens, G); push!(Dd, copy(d)); return G
    end
    "모든 생성기를 x̄ 에서 (묶음별 병렬). 반환 = [(값, 기울기)]"
    function eval_all(x̄)
        out = Vector{Tuple{Float64,Vector{Float64}}}(undef, length(gens))
        t_gen[] += @elapsed if parallel
            @sync for grp in 1:NG
                ids = [b for b in eachindex(gens) if gens[b].grp == grp]
                isempty(ids) && continue
                Threads.@spawn for b in ids
                    out[b] = _bccg_gen_eval!(gens[b], x̄)
                end
            end
        else
            for b in eachindex(gens); out[b] = _bccg_gen_eval!(gens[b], x̄); end
        end
        n_geneval[] += length(gens)
        return out
    end
    cutrec = Vector{Tuple{Float64,Vector{Float64}}}()      # 콜백에서 넣은 cut (다시 풀 때 실제 제약으로)
    ngrec = Vector{Tuple{Vector{Float64},Float64}}()       # 콜백에서 넣은 no-good
    n_cuts_real = Ref(0)
    cutexpr(a, g) = a + sum(g[i] * x[i] for i in 1:nx if g[i] != 0; init=AffExpr(0.0))
    "x̄ 에서의 값 v, 기울기 g → cut (a, g'). |g_i| < 1e-9 인 성분은 지우고 그만큼 절편을 낮춰 [0,1]ⁿ 에서 유효하게 둠"
    function makecut(v, g, x̄)
        g2 = copy(g); a = v
        for i in eachindex(g2)
            if abs(g2[i]) < 1e-9
                a -= abs(g2[i]); g2[i] = 0.0
            else
                a -= g2[i] * x̄[i]
            end
        end
        return a, g2
    end
    function add_real_cut!(v, g, x̄)
        a, g2 = makecut(v, g, x̄); @constraint(M, t0 >= cutexpr(a, g2)); n_cuts_real[] += 1
    end

    # 사슬 submodular cut (chain)
    al = findall(nd.x_allowed); siteof(i) = (i - 1) ÷ nd.meta[:K] + 1
    unitv(j) = (z = zeros(nx); z[j] = 1.0; z)
    n_chain = Ref(0); n_chain_chk = Ref(0)
    function Lval(G::BCCGGen, z)
        v = get(G.cache, z, nothing)
        v === nothing || return v
        v = first(_bccg_gen_eval!(G, z)); G.cache[copy(z)] = v; return v
    end
    "S (정수점) 에서 belief 생성기 G 의 사슬 cut (a, g). 조건 ρ_j(U−j) ≥ ρ_j(∅) 을 S 의 부분집합에서 검사"
    function chaincut(G::BCCGGen, Sx)
        L0 = Lval(G, zeros(nx)); ρ0 = Dict(j => Lval(G, unitv(j)) - L0 for j in al)
        els = [i for i in al if Sx[i] > 0.5]
        g = zeros(nx); a = Lval(G, Sx)
        for j in els
            best = -Inf
            for mask in 0:(2^length(els) - 1)
                U = zeros(nx); for (b, e) in enumerate(els); (mask >> (b - 1)) & 1 == 1 && (U[e] = 1.0); end
                U[j] > 0.5 || continue
                W = copy(U); W[j] = 0.0
                mg = Lval(G, U) - Lval(G, W)
                n_chain_chk[] += 1
                mg >= ρ0[j] - 1e-6 * max(1.0, abs(ρ0[j])) ||
                    error("nz_belief_ccg_tree (chain): 유효 조건 위반 — 원소 $(j) 의 한계값 $(mg) < 단일 원소 한계값 $(ρ0[j]) (U = $(findall(U .> 0.5))). 사슬 cut 을 쓸 수 없음")
                best = max(best, mg)
            end
            a -= best; g[j] = best
        end
        for j in al; Sx[j] > 0.5 || (g[j] = ρ0[j]); end
        n_chain[] += 1
        return a, g
    end
    "새 생성기의 사슬 cut 묶음 (S = ∅ 와 다른 거점의 원소 쌍). 반환 = [(a, g)]"
    function chainfamily(G::BCCGGen)
        out = [chaincut(G, zeros(nx))]
        for i in al, j in al
            (i < j && siteof(i) != siteof(j)) || continue
            z = zeros(nx); z[i] = 1.0; z[j] = 1.0; push!(out, chaincut(G, z))
        end
        return out
    end
    function add_family_real!(G)
        for (a, g) in chainfamily(G)
            cr = @constraint(M, t0 >= cutexpr(a, g)); n_cuts_real[] += 1
            lazy_attr > 0 && MOI.set(M, Gurobi.ConstraintAttribute("Lazy"), cr, lazy_attr)
        end
    end

    # 탐색 (nz_belief_ccg 의 search 와 같음)
    function search(xb)
        cand = vcat(Dd, pairbels, [randbel() for _ in 1:nrand]); vals = fill(-Inf, length(cand))
        Threads.@threads for t in eachindex(cand)
            local ev = take!(pool_ev)
            try
                _bccg_setx!(ev, nd, xb); local v1, _, _, _ = _bccg_g!(ev, nd, cand[t]); vals[t] = v1
            finally
                put!(pool_ev, ev)
            end
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
        finally
            put!(pool_ev, ev2)
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
                a2 = nz_feasible_alpha(nd, xb, a1; lp_optimizer=lp_opt, tol=1e-3)
                a1 = a2; v3, dd = _bccg_value!(ev3, nd, a2)
            end
            return v3, dd, (dd === nothing ? nothing : copy(ev3.lastr)), a1
        finally
            put!(pool_ev, ev3)
        end
    end

    # 안전장치
    tlow = -1e7
    n_ng = Ref(0); n_rc = Ref(0); n_mu = Ref(0)
    ngexpr(xb) = sum((1 - x[i]) for i in 1:nx if xb[i] > 0.5; init=AffExpr(0.0)) + sum(x[i] for i in 1:nx if xb[i] < 0.5 && nd.x_allowed[i]; init=AffExpr(0.0))
    nogood_real!(xb, v) = (@constraint(M, t0 >= v - (v - tlow) * ngexpr(xb)); n_ng[] += 1)
    function restricted!(xb, α)
        F = nz_build_fixed_alpha(nd, xb; optimizer=lp_opt)
        z = nz_eval_alpha!(F, nd, α)
        isfinite(z) || return false
        cut = nz_cut_from_solution(F, nd, xb)
        @constraint(M, t0 >= cut[:intercept] + dot(cut[:slope], x)); n_rc[] += 1
        return true
    end
    """새 belief 를 생성기로 넣고 x̄ 확보 (원고 Algorithm 2 의 10–11 줄). 반환 (방법, x̄ 에서의 cut (값, 기울기) 또는 nothing).
       :safeguard 이면 no-good 이 필요 (호출자가 실제 제약 또는 lazy 로 넣음)"""
    function add_and_secure!(xb, v, d, r)
        G = new_gen!(d, r, mu)
        thr = 1e-4 * max(1.0, abs(v))
        vb, gb = _bccg_gen_eval!(G, xb)
        vb >= v - thr && return :block, (vb, gb)
        _, Linf, μstar = _bccg_block_at(C, xb, d, r, mu)
        if Linf >= v - thr && μstar <= mu_max
            μn = min(mu_max, max(2mu, 1.1μstar))
            gens[end] = _bccg_gen(C, d, r, μn, envs[G.grp], G.grp); n_mu[] += 1
            vb2, gb2 = _bccg_gen_eval!(gens[end], xb)
            vb2 >= v - thr && return :mu, (vb2, gb2)
            return :safeguard, (vb2, gb2)
        end
        return :safeguard, (vb, gb)
    end

    # 루트 Kelley (LP 완화 위의 cut 분리). obj = 목적식
    function kelley!(obj, rounds)
        undo = relax_integrality(M); @objective(M, Min, obj)
        set_optimizer_attribute(M, "MIPGap", master_gap)
        last = -Inf; nadd = 0; k = 0
        for k_ in 1:rounds
            k = k_
            optimize!(M); termination_status(M) == MOI.OPTIMAL || error("nz_belief_ccg_tree: 루트 LP $(termination_status(M))")
            z = objective_value(M); x̃ = value.(x); t̃ = value(t0)
            vals = eval_all(x̃); nv = 0
            for (v, g) in vals
                v > t̃ + cut_tol * max(1.0, abs(v)) || continue
                add_real_cut!(v, g, x̃); nv += 1
            end
            nadd += nv
            (nv == 0 || (k_ > 5 && z - last <= 1e-6 * max(1.0, abs(z)))) && (last = z; break)
            last = z
        end
        undo(); @objective(M, Min, dot(nd.qx, x) + t0)
        return last, k, nadd
    end

    # 콜백
    seen = Dict{Vector{Float64},Int}()
    n_lazy = Ref(0); n_lazycut = Ref(0); n_ucb = Ref(0); n_ucut = Ref(0); n_tree_bel = Ref(0); n_srch = Ref(0)
    t_srch = Ref(0.0); t_cb = Ref(0.0); newbel_solve = Ref(0); n_keep = Ref(0)
    seen_x = Set{Vector{Float64}}()                      # 정수 후보로 불린 서로 다른 x (진단)
    function lazy_cb(cb)
        callback_node_status(cb, M) == MOI.CALLBACK_NODE_STATUS_INTEGER || return
        tc0 = time()
        try
            n_lazy[] += 1
            xb = [callback_value(cb, x[i]) > 0.5 ? 1.0 : 0.0 for i in 1:nx]; tv = callback_value(cb, t0)
            push!(seen_x, xb)
            nv = 0; viol = Int[]
            for (b, (v, g)) in enumerate(eval_all(xb))
                v > tv + cut_tol * max(1.0, abs(v)) || continue
                a, g2 = makecut(v, g, xb)
                MOI.submit(M, MOI.LazyConstraint(cb), @build_constraint(t0 >= cutexpr(a, g2)))
                push!(cutrec, (a, g2)); nv += 1; push!(viol, b)
            end
            n_lazycut[] += nv
            if chain && nv > 0
                for b in viol
                    a, g2 = chaincut(gens[b], xb)
                    MOI.submit(M, MOI.LazyConstraint(cb), @build_constraint(t0 >= cutexpr(a, g2))); push!(cutrec, (a, g2))
                end
            end
            if mw_eps > 0 && nv > 0
                x̂ = xb .+ mw_eps .* (x0 .- xb)
                for (b, (v, g)) in enumerate(eval_all(x̂))
                    b in viol || continue
                    a, g2 = makecut(v, g, x̂)
                    MOI.submit(M, MOI.LazyConstraint(cb), @build_constraint(t0 >= cutexpr(a, g2))); push!(cutrec, (a, g2)); n_mwcut[] += 1
                end
            end
            nv > 0 && return
            get(seen, xb, 0) >= 3 && return
            seen[xb] = get(seen, xb, 0) + 1
            n_srch[] += 1
            t_srch[] += @elapsed (b = search(xb))
            (b.d !== nothing && b.v > tv + 1e-3 * max(1.0, abs(b.v))) || return
            how, (vb, gb) = add_and_secure!(xb, b.v, b.d, b.r)
            n_tree_bel[] += 1; newbel_solve[] += 1
            if vb > tv + cut_tol * max(1.0, abs(vb))
                a, g2 = makecut(vb, gb, xb)
                MOI.submit(M, MOI.LazyConstraint(cb), @build_constraint(t0 >= cutexpr(a, g2))); push!(cutrec, (a, g2))
                if mw_eps > 0
                    x̂ = xb .+ mw_eps .* (x0 .- xb); v3, g3 = _bccg_gen_eval!(gens[end], x̂); a3, g4 = makecut(v3, g3, x̂)
                    MOI.submit(M, MOI.LazyConstraint(cb), @build_constraint(t0 >= cutexpr(a3, g4))); push!(cutrec, (a3, g4)); n_mwcut[] += 1
                end
            end
            if chain
                for (a, g2) in chainfamily(gens[end])
                    MOI.submit(M, MOI.LazyConstraint(cb), @build_constraint(t0 >= cutexpr(a, g2))); push!(cutrec, (a, g2))
                end
            end
            if how == :safeguard
                safeguard in (:restricted, :both) && error("nz_belief_ccg_tree: 나무 안의 restricted ISP cut 은 아직 없음 (safeguard = :nogood 만)")
                MOI.submit(M, MOI.LazyConstraint(cb), @build_constraint(t0 >= b.v - (b.v - tlow) * ngexpr(xb))); push!(ngrec, (xb, b.v)); n_ng[] += 1
            end
            verbose && (@printf("    [tree] belief +1 (%s) x=%-24s t0 %.2f → %.2f, beliefs=%d (%.0fs)\n", how, loc_tkp_xstr(nd, xb), tv, b.v, length(gens), time() - wall); flush(stdout))
        finally
            t_cb[] += time() - tc0
        end
    end
    function ucut_cb(cb)
        n_ucb[] >= ucut_calls && return
        callback_node_status(cb, M) == MOI.CALLBACK_NODE_STATUS_FRACTIONAL || return
        tc0 = time()
        try
            n_ucb[] += 1
            xf = [callback_value(cb, x[i]) for i in 1:nx]; tv = callback_value(cb, t0)
            for (v, g) in eval_all(xf)
                v > tv + 1e-4 * max(1.0, abs(v)) || continue
                a, g2 = makecut(v, g, xf)
                MOI.submit(M, MOI.UserCut(cb), @build_constraint(t0 >= cutexpr(a, g2))); push!(cutrec, (a, g2)); n_ucut[] += 1
            end
        finally
            t_cb[] += time() - tc0
        end
    end
    set_attribute(M, MOI.LazyConstraintCallback(), lazy_cb)
    ucut_calls > 0 && set_attribute(M, MOI.UserCutCallback(), ucut_cb)

    # 시작: 명목 블록, no-good 의 t 하한 (Kelley 근사의 min t0 ≤ min_x L_b0(x)), 루트 Kelley
    new_gen!(qh, qh, mu)
    chain && add_family_real!(gens[1])
    tk = @elapsed begin
        zt, _, _ = kelley!(1.0 * t0, root_rounds)
        tlow = zt - 1e-6 * max(1.0, abs(zt))
        zr, kr, nr = kelley!(dot(nd.qx, x) + t0, root_rounds)
    end
    verbose && (@printf("  [root] t 하한 %.2f, 루트 LP %.2f (Kelley %d 회, cut %d, %.1fs)\n", tlow, zr, kr, nr, tk); flush(stdout))

    UB = Inf; LB = -Inf; xbest = nothing; status = :TimeLimit; iter = 0; t_ms = 0.0; t_or = 0.0; n_or = 0; n_flush = 0
    set_optimizer_attribute(M, "MIPGap", master_gap)
    while time() - wall < time_limit
        iter += 1
        newbel_solve[] = 0; ncb0 = n_lazy[]; tcb0 = t_cb[]
        set_time_limit_sec(M, max(10.0, time_limit - (time() - wall)))
        tm = @elapsed optimize!(M); t_ms += tm
        termination_status(M) in (MOI.OPTIMAL, MOI.TIME_LIMIT) || error("nz_belief_ccg_tree: 마스터 $(termination_status(M))")
        LB = max(LB, objective_bound(M))
        nodes = try Int(MOI.get(M, MOI.NodeCount())) catch; -1 end
        tstat = termination_status(M); hv = has_values(M)
        xb = hv ? Float64.(value.(x) .> 0.5) : zeros(nx); t0v = hv ? objective_value(M) - dot(nd.qx, xb) : NaN
        # 콜백에서 넣은 cut·no-good 을 실제 제약으로 (다음 풀이에서도 유지). 모형이 바뀌어 해가 지워지므로 해를 먼저 읽음
        for (a, g) in cutrec
            cr = @constraint(M, t0 >= cutexpr(a, g)); n_flush += 1
            lazy_attr > 0 && MOI.set(M, Gurobi.ConstraintAttribute("Lazy"), cr, lazy_attr)
        end
        empty!(cutrec)
        for (xn, v) in ngrec; @constraint(M, t0 >= v - (v - tlow) * ngexpr(xn)); end; empty!(ngrec)
        hv || break
        if isfinite(UB) && UB - LB <= tol * max(1.0, abs(UB)); status = :Optimal; break; end
        tstat == MOI.TIME_LIMIT && break
        cbinfo = @sprintf("마스터 %.1fs (콜백 %.1fs, 정수 후보 %d) 노드 %d", tm, t_cb[] - tcb0, n_lazy[] - ncb0, nodes)
        if newbel_solve[] > 0
            # 나무 안에서 belief 가 늘었음. incumbent 값이 지금 생성기 전부로도 맞으면 incumbent 는 지금 블록 집합에서 최적
            # (가지친 노드의 하한 ≥ 그때 incumbent 값 ≥ 마지막 incumbent 값, 블록이 늘면 값함수는 커지기만 함) → 다시 풀지 않고 oracle 로
            tchk = maximum(first, eval_all(xb))
            if tchk <= t0v + 1e-6 * max(1.0, abs(t0v))
                newbel_solve[] = 0; n_keep[] += 1
                verbose && (@printf("  [tree] it %3d incumbent 값이 블록 %d 개 전부로 맞음 (%.2f ≤ %.2f) → 다시 풀지 않음
", iter, length(gens), tchk, t0v); flush(stdout))
            end
        end
        if newbel_solve[] > 0
            # incumbent 의 t0 가 옛 블록 기준 → 루트 Kelley 후 다시 풂
            zr, kr, nr = kelley!(dot(nd.qx, x) + t0, root_rounds)
            verbose && (@printf("  [tree] it %3d LB=%10.2f UB=%10.2f x=%-24s belief +%d, %s, beliefs=%d, 루트 %.2f (cut %d)\n",
                                iter, LB, UB, loc_tkp_xstr(nd, xb), newbel_solve[], cbinfo, length(gens), zr, nr); flush(stdout))
            continue
        end
        # 전역 oracle (나무가 지금 블록 집합에서 정확할 때만)
        tau = t0v + 0.9 * tol * max(1.0, abs(LB))
        gap_o = min(1e-3, 0.4 * tol * max(1.0, abs(LB)) / max(1.0, abs(t0v)))
        lim = max(10.0, time_limit - (time() - wall))
        to = @elapsed (res = nz_alpha_bnb_dr(nd, xb; nworkers=max(1, threads - 1), time_limit=lim, rel_gap=gap_o, target=tau, verbose=false))
        t_or += to; n_or += 1
        if dot(nd.qx, xb) + res[:UB] < UB; UB = dot(nd.qx, xb) + res[:UB]; xbest = copy(xb); end
        how = :none
        if res[:α] !== nothing && res[:LB] > t0v + 1e-6 * max(1.0, abs(t0v))
            vv, dd, r2, a1 = exact_value(xb, res[:α])
            if dd !== nothing
                how, (vb, gb) = add_and_secure!(xb, vv, dd, r2)
                add_real_cut!(vb, gb, xb)
                chain && add_family_real!(gens[end])
                if how == :safeguard
                    safeguard in (:nogood, :both) && nogood_real!(xb, vv)
                    safeguard in (:restricted, :both) && (restricted!(xb, a1) || safeguard == :both || nogood_real!(xb, vv))
                end
                if vv <= t0v + 1e-6 * max(1.0, abs(t0v)) && safeguard == :nogood
                    restricted!(xb, a1)          # 정확 평가가 진전을 못 주면 (λᵁ ≥ λ̄ 아래의) restricted ISP cut 로 진전 보장 (nz_belief_ccg 와 같음)
                end
            end
        end
        # oracle 이 t0 보다 큰 하한을 증명했는데 belief 를 못 만든 경우 (α 의 정확 평가 실패, 2026-11-22 지리 8·13 에서 Stalled):
        # 증명된 하한 v = res[:LB] ≤ V*(x̄) 로 no-good cut (유효, x̄ 에서 진전 보장)
        if how == :none && res[:LB] > t0v + 1e-6 * max(1.0, abs(t0v))
            nogood_real!(xb, res[:LB]); how = :nogood_lb
        end
        verbose && @printf("  [orcl] it %3d LB=%10.2f UB=%10.2f x=%-24s V in [%.2f, %.2f], t0 %.2f (%.0fs, %s) %s, beliefs=%d\n",
                           iter, LB, UB, loc_tkp_xstr(nd, xb), res[:LB], res[:UB], t0v, to, cbinfo, how, length(gens)); flush(stdout)
        if isfinite(UB) && UB - LB <= tol * max(1.0, abs(UB)); status = :Optimal; break; end
        if how == :none
            time() - wall >= time_limit - 1.0 && break
            status = :Stalled; verbose && println("  [!] oracle 이 t0 보다 큰 하한도 새 belief 도 못 줌"); break
        end
        how != :none && kelley!(dot(nd.qx, x) + t0, root_rounds)
    end
    return Dict(:impl => "nz_belief_ccg_tree", :status => status, :x => xbest, :UB => UB, :LB => LB, :gap => abs(UB - LB) / max(1.0, abs(UB)),
                :iters => iter, :beliefs => length(gens), :tree_beliefs => n_tree_bel[], :kept_incumbent => n_keep[], :master_time => t_ms, :callback_time => t_cb[],
                :gen_time => t_gen[], :gen_evals => n_geneval[], :search_time => t_srch[], :searches => n_srch[],
                :belief_list => [(d = copy(G.d), r = copy(G.r), μ = G.μ) for G in gens], :chain_cuts => n_chain[], :chain_checks => n_chain_chk[], :lazy_calls => n_lazy[], :lazy_distinct => length(seen_x), :lazy_cuts => n_lazycut[], :mw_cuts => n_mwcut[], :ucut_calls => n_ucb[], :ucuts => n_ucut[], :real_cuts => n_cuts_real[] + n_flush,
                :oracle_calls => n_or, :oracle_time => t_or, :nogood => n_ng[], :restricted => n_rc[], :mu_raises => n_mu[], :time => time() - wall,
                :args => (mu = mu, eta = eta, mu_max = mu_max, safeguard = safeguard, master_gap = master_gap, wd = wd, nrand = nrand,
                          root_rounds = root_rounds, ucut_calls = ucut_calls, cut_tol = cut_tol, parallel = parallel, lazy_attr = lazy_attr, mw_eps = mw_eps, chain = chain))
end
