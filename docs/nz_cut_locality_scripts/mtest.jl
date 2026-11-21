# belief 를 생성하는 column-and-constraint generation 시제품 (2026-11-20).
#   마스터 (MILP):  min q·x + t0,  belief (d, r) 마다 follower 2 단계 LP 의 최적성 (원·쌍대 실행가능 + 강쌍대, x·π 는 McCormick) 과 t0 ≥ Σ_s r_s ℓᵀy_s.
#     → 모든 정수 x 에서 t0 ≥ [d 에서 follower 가 최적 반응할 때 리더 비용 (리더에게 유리한 tie)] ≤ V*(x). x 의 탐색은 솔버의 분지한정.
#   x̄ 에서: 싼 belief 탐색 (LP) → 마스터 값보다 높으면 그 (d, r) 추가. 아니면 전역 oracle (α-B&B) 로 상한, 더 나쁜 belief 가 나오면 추가.
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl")); include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl")); include(joinpath(@__DIR__, "bgen_core.jl"))
envf(k, d) = get(ENV, k, d)
g = parse(Int, envf("GEO", "1")); MODEL = envf("MODEL", "coupled"); rr = parse(Float64, envf("R", "0.2")); TL = parse(Float64, envf("TL", "3600")); TOL = 5e-3
eh, et, dl = MODEL == "coupled" ? (rr, rr, rr / 3) : MODEL == "belief" ? (0.0, rr, Inf) : MODEL == "data" ? (rr, 0.0, Inf) : MODEL == "nominal" ? (0.0, 0.0, Inf) : (rr, rr, Inf)
nd = make_loc_tkp(; g=g, S=parse(Int, envf("SS", "10")), seed=parse(Int, envf("SEED", "1")), eps_hat=eh, eps_tilde=et, delta=dl, beta=0.4)
_BXROWS[] = unique(e.k for e in nz_xentries(nd)); K, nB, S = nd.meta[:K], nd.meta[:nB], nd.S; qh = nd.q_hat; bx = nd.meta[:box]
m, ny, nh, nx = nz_m(nd), nz_ny(nd), nz_nh(nd), nd.nx; nW = size(nd.W, 1); A = nd.A; XE = nz_xentries(nd)
NT = Threads.nthreads(); pool = Channel{BEv}(NT); for env in _nz_envs(NT); put!(pool, BEv(nd, env)); end
if haskey(ENV, "SITES")
    keep = parse.(Int, split(ENV["SITES"], ",")); for st in 1:nB, k in 1:K; st in keep || (nd.x_allowed[(st - 1) * K + k] = false); end
end
nX = let ns = count(st -> nd.x_allowed[(st - 1) * K + 1], 1:nB); sum(binomial(ns, n) * K^n for n in 0:nd.gamma) end
@printf("--- dromu μ·120=%s ε=%s VUB=%s | g=%d model=%s r=%.2f S=%d |X|=%d 스레드 %d TL=%.0f\n", envf("MUK","30"), envf("EPSP","1e-3"), envf("VUB","1"), g, MODEL, rr, S, nX, NT, TL); flush(stdout)

# ---------- 마스터 ----------
M = Model(() -> (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), NT); o))
set_optimizer_attribute(M, "MIPGap", parse(Float64, envf("MGAP", "1e-3"))); set_optimizer_attribute(M, "IntFeasTol", 1e-9)
@variable(M, x[1:nx], Bin); @variable(M, t0 >= -1e5)
@constraint(M, sum(x) <= nd.gamma)
for (idx, rhs) in get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[]); @constraint(M, sum(x[i] for i in idx) <= rhs); end
for i in 1:nx; nd.x_allowed[i] || fix(x[i], 0.0; force=true); end
@objective(M, Min, dot(nd.qx, x) + t0)
colA = [[(k, A[k, j]) for k in 1:m if A[k, j] != 0] for j in 1:ny]; rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m]
ents_of_row = [[e for e in XE if e.k == k] for k in 1:m]
nbel = Ref(0)

const MUK = parse(Float64, envf("MUK", "30")); const MU = MUK / 120.0; const EPSP = parse(Float64, envf("EPSP", "1e-3"))
const VUB = envf("VUB", "1") == "1"
"""
belief (d, r) 의 μ 하한 블록 (모든 μ ≥ 0 에서 유효한 V* 의 하한):
  LB(x) = min_{(h,y)} [ μ·(C_F(h,y) − ε·C_L(y)) + C_L(y) ] − μ·V_F^ε(x),   V_F^ε(x) = min (C_F − ε·C_L)
  C_F = follower 비용 (belief d), C_L = 리더 비용 (가중 r). ε 섭동은 follower 동률을 리더에게 불리한 쪽으로 깸 (비관적).
  첫 항은 원문제 사본 (x 는 우변), V_F^ε 는 쌍대 (x·π 는 McCormick, 마스터가 최소화하므로 약쌍대로 유효).
"""
function add_belief!(d, r)
    nbel[] += 1
    h = @variable(M, [i=1:nh], lower_bound = 0.0, upper_bound = nd.hU[i]); @constraint(M, nd.W * h .<= nd.wvec)
    y = @variable(M, [1:ny, 1:S], lower_bound = 0.0)
    @constraint(M, [k=1:m, s=1:S], sum(a * y[j, s] for (j, a) in rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0)
                                   - sum(e.g[s] * x[e.i] for e in ents_of_row[k]; init=0.0) <= nd.u[k, s])
    if VUB
        MMv = nd.meta
        for st in 1:nB, j in 1:MMv[:ncu], s in 1:S
            @constraint(M, y[MMv[:iyR](MMv[:nA] + st, j), s] + y[MMv[:iyS](MMv[:nA] + st, j), s] <= -nd.u[MMv[:rdem](j), s] * sum(x[(st - 1) * K + k] for k in 1:K))
        end
    end
    CF = -(dot(nd.c0, h) + sum(d[s] * nd.c[j] * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0))
    CL = sum(r[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0)
    # V_F^ε 의 쌍대: follower 최대화 −(C_F − ε C_L) = c0ᵀh + Σ d cᵀy + ε Σ r ℓᵀy 의 쌍대 (최소화) = −V_F^ε
    cε(j, s) = d[s] * nd.c[j] + EPSP * r[s] * nd.ell[j]
    pd = @variable(M, [1:m, 1:S], lower_bound = 0.0); kp = @variable(M, [1:nW], lower_bound = 0.0); om = @variable(M, [1:nh], lower_bound = 0.0)
    @constraint(M, [j=1:ny, s=1:S], sum(a * pd[k, s] for (k, a) in colA[j]) >= cε(j, s))
    @constraint(M, [i=1:nh], -sum(nd.hcoef[k] * pd[k, s] for k in 1:m, s in 1:S if nd.hrow[k] == i) + sum(nd.W[l, i] * kp[l] for l in 1:nW) + om[i] >= nd.c0[i])
    dobj = AffExpr(0.0)
    add_to_expression!(dobj, sum(nd.u[k, s] * pd[k, s] for k in 1:m, s in 1:S if nd.u[k, s] != 0) + dot(nd.wvec, kp) + dot(nd.hU, om))
    psv = Dict{Tuple{Int,Int},VariableRef}()
    for (ei, e) in enumerate(XE), s in 1:S
        U = d[s] * nd.piFU[e.k] + EPSP * r[s] * maximum(abs, nd.ell); set_upper_bound(pd[e.k, s], U)
        ps = @variable(M, lower_bound = 0.0); @constraint(M, ps >= pd[e.k, s] - U * (1 - x[e.i])); add_to_expression!(dobj, e.g[s], ps)
        psv[(ei, s)] = ps
        envf("V1", "0") == "1" && @constraint(M, ps <= pd[e.k, s])
    end
    if envf("V1", "0") == "1"                       # 거점 단위: Σ_k ψ_ik = z_i π  →  Σ_k ψ_ik ≥ π − U (1 − Σ_k x_ik)
        for k in unique(e.k for e in XE), s in 1:S
            ents = [ei for (ei, e) in enumerate(XE) if e.k == k]
            U = d[s] * nd.piFU[k] + EPSP * r[s] * maximum(abs, nd.ell)
            @constraint(M, sum(psv[(ei, s)] for ei in ents) >= pd[k, s] - U * (1 - sum(x[XE[ei].i] for ei in ents)))
        end
    end
    # t0 ≥ μ(C_F − ε C_L) + C_L − μ V_F^ε  ,  −V_F^ε (최대화 값) ≤ dobj
    @constraint(M, t0 >= MU * (CF - EPSP * CL) + CL + MU * dobj)
    return nothing
end
"belief (d, r) 의 follower 최적성 블록 (강쌍대, 리더에게 유리한 tie) 을 마스터에 추가"
function add_belief_opt!(d, r)
    nbel[] += 1
    h = @variable(M, [i=1:nh], lower_bound = 0.0, upper_bound = nd.hU[i]); @constraint(M, nd.W * h .<= nd.wvec)
    y = @variable(M, [1:ny, 1:S], lower_bound = 0.0); pd = @variable(M, [1:m, 1:S], lower_bound = 0.0); kp = @variable(M, [1:nW], lower_bound = 0.0)
    # 원문제 실행가능 (x 는 우변)
    @constraint(M, [k=1:m, s=1:S], sum(a * y[j, s] for (j, a) in rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0)
                                   - sum(e.g[s] * x[e.i] for e in ents_of_row[k]; init=0.0) <= nd.u[k, s])
    # 쌍대 실행가능
    @constraint(M, [j=1:ny, s=1:S], sum(a * pd[k, s] for (k, a) in colA[j]) >= nd.c[j] * d[s])
    @constraint(M, [i=1:nh], sum(nd.hcoef[k] * pd[k, s] for k in 1:m, s in 1:S if nd.hrow[k] == i) + nd.c0[i] <= sum(nd.W[l, i] * kp[l] for l in 1:nW))
    # x·π 의 McCormick (강쌍대 우변을 키우는 방향만 필요): psi ≥ π − U (1 − x), psi ≥ 0,  U = d_s π_Fᵁ
    psi = @variable(M, [1:length(XE), 1:S], lower_bound = 0.0)
    for (ei, e) in enumerate(XE), s in 1:S
        U = d[s] * nd.piFU[e.k]
        @constraint(M, psi[ei, s] >= pd[e.k, s] - U * (1 - x[e.i]))
    end
    for k in unique(e.k for e in XE), s in 1:S
        set_upper_bound(pd[k, s], d[s] * nd.piFU[k])              # 상계 자체 (McCormick 의 전제)
    end
    # 강쌍대: 원 목적 ≥ 쌍대 목적
    @constraint(M, dot(nd.c0, h) + sum(d[s] * nd.c[j] * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0)
                   >= sum(nd.u[k, s] * pd[k, s] for k in 1:m, s in 1:S if nd.u[k, s] != 0)
                      + sum(e.g[s] * psi[ei, s] for (ei, e) in enumerate(XE), s in 1:S) + dot(nd.wvec, kp))
    # 리더 비용
    @constraint(M, t0 >= sum(r[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0))
    return nothing
end

inset(d) = all(d .>= bx.dl .- 1e-12) && all(d .<= bx.du .+ 1e-12) && sum(abs.(d .- qh)) <= 2nd.eps_tilde + 1e-12
function randbel(rng)
    d = copy(qh)
    for t in 1:rand(rng, 1:4)
        i, j = randperm(rng, S)[1:2]; mv = rand(rng) * min(d[i] - bx.dl[i], bx.du[j] - d[j]); dd = copy(d); dd[i] -= mv; dd[j] += mv; inset(dd) && (d = dd)
    end
    return d
end
const PAIRBELS = let out = Vector{Vector{Float64}}()
    for i in 1:S, j in 1:S
        i == j && continue
        mv = min(qh[i] - bx.dl[i], bx.du[j] - qh[j], nd.eps_tilde); mv <= 1e-9 && continue
        d = copy(qh); d[i] -= mv; d[j] += mv; inset(d) && push!(out, d)
    end
    out
end
n_eval = Threads.Atomic{Int}(0)

"x̄ 에서 싼 belief 탐색 (병렬). 반환 (값, 정확 LP 의 belief, 재가중 r) — 뒤의 둘은 같은 LP 해라 서로 정합"
function search(xb, Dd, rng; nrand=40)
    cand = vcat(Dd, PAIRBELS, [randbel(rng) for _ in 1:nrand]); vals = fill(-Inf, length(cand))
    Threads.@threads for t in eachindex(cand)
        local ev = take!(pool)
        try
            bev_setx!(ev, nd, xb); local v1, _, _ = bev_g!(ev, nd, cand[t]); vals[t] = v1; Threads.atomic_add!(n_eval, 1)
        finally
            put!(pool, ev)
        end
    end
    tb = argmax(vals); ev2 = take!(pool); best = -Inf; bd = nothing; br = nothing
    try
        bev_setx!(ev2, nd, xb); dcur = cand[tb]
        for it in 1:4
            v2, dd, _ = bev_g!(ev2, nd, dcur); Threads.atomic_add!(n_eval, 1)
            dd === nothing && break
            if v2 > best + 1e-9
                best = v2; bd = dd; br = copy(ev2.lastr)
            else
                break
            end
            inset(dd) || break
            dcur = dd
        end
    finally
        put!(pool, ev2)
    end
    return best, bd, br
end

"α 의 정확 평가에서 (값, belief, r)"
function value_of(xb, al)
    ev3 = take!(pool)
    try
        bev_setx!(ev3, nd, xb)
        a1 = [(v = clamp(al[i], 0.0, nd.hU[i]); v < 1e-6 * max(1.0, nd.hU[i]) ? 0.0 : v) for i in eachindex(al)]
        v3, dd = bev_value!(ev3, nd, a1)
        if dd === nothing                                   # α-B&B 의 α 가 잡음만큼 실행가능 영역 밖이면 가장 가까운 실행가능 반응으로 사영
            a2 = try nz_feasible_alpha(nd, xb, a1; lp_optimizer=GRB1, tol=1e-3) catch e; println("    [α 사영 실패] ", first(string(e), 120)); nothing end
            a2 === nothing || ((v3, dd) = bev_value!(ev3, nd, a2))
            println("    [α 정리] 원 α 의 정확 평가 실패 → 사영 후 ", dd === nothing ? "여전히 실패" : @sprintf("%.3f", v3))
        end
        return v3, dd, (dd === nothing ? nothing : copy(ev3.lastr))
    finally
        put!(pool, ev3)
    end
end

rng = MersenneTwister(2026); Dd = [copy(qh)]; add_belief!(qh, qh)

if envf("V2", "0") == "1"
    @variable(M, z[1:nB], Bin)
    for st in 1:nB; @constraint(M, z[st] == sum(x[(st - 1) * K + k] for k in 1:K)); MOI.set(M, Gurobi.VariableAttribute("BranchPriority"), z[st], 10); end
end
set_optimizer_attribute(M, "OutputFlag", 1); set_optimizer_attribute(M, "LogToConsole", 0)
lf = joinpath(@__DIR__, "mtest_$(envf("V1","0"))$(envf("V2","0")).log"); isfile(lf) && rm(lf); set_optimizer_attribute(M, "LogFile", lf)
tm = @elapsed optimize!(M)
root = filter(l -> occursin("Root relaxation", l), readlines(lf))
@printf("MTEST V1=%s V2=%s: 목적 %.2f 하한 %.2f 노드 %d 시간 %.1fs | %s\n", envf("V1","0"), envf("V2","0"), objective_value(M), objective_bound(M),
        Int(MOI.get(M, MOI.NodeCount())), tm, isempty(root) ? "" : root[1])
