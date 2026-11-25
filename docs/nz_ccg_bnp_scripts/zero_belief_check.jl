# 진단 (저장소 안 고침): 블록이 belief 확률 0 인 시나리오에서 낮게 나오는지 (safeguard 원인 가설), 그 시나리오도 섭동하면 정확해지는지
#  GEO, S, XS ("{..}|{..}" 여러 점)
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl")); include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_belief_ccg.jl"))
envf(k, d) = get(ENV, k, d)
g = parse(Int, envf("GEO", "5")); S = parse(Int, envf("S", "10")); RR = 0.2; ETA = 1e-4; MU = 1e4
nd = make_loc_tkp(; g=g, S=S, seed=1, eps_hat=RR, eps_tilde=RR, delta=RR / 3, beta=0.4, lambdaU=1e4, NB=3)
qh = nd.q_hat; bx = nd.meta[:box]; K = nd.meta[:K]; tiers = nd.meta[:tiers]; nA = nd.meta[:nA]
@printf("g=%d S=%d  q̂ = %s\n  dl = %s  (dl=0 인 시나리오 %d 개)\n", g, S, string(round.(qh; digits=3)), string(round.(bx.dl; digits=3)), count(<=(1e-12), bx.dl))
m, ny = nz_m(nd), nz_ny(nd); A = nd.A; XE = nz_xentries(nd)
env = Gurobi.Env()
lp_opt = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
C = (nd = nd, eta = ETA, vub = false, wd = false, XE = XE, lp_opt = lp_opt,
     rowA = [[(j, A[k, j]) for j in 1:ny if A[k, j] != 0] for k in 1:m], colA = [[(k, A[k, j]) for k in 1:m if A[k, j] != 0] for j in 1:ny],
     ents_of_row = [[e for e in XE if e.k == k] for k in 1:m])
"블록 값 at x̄ — allzero=true 면 d_s = 0 시나리오도 섭동 (나머지는 _bccg_block_at 과 같음)"
function blockval(x̄, d, r, μ; allzero=false)
    nh = nz_nh(nd)
    ce(j, s) = d[s] * nd.c[j] + ((allzero || d[s] > 0) ? ETA * r[s] * nd.ell[j] : 0.0)
    P = Model(lp_opt)
    @variable(P, 0 <= h[i=1:nh] <= nd.hU[i]); @constraint(P, nd.W * h .<= nd.wvec); @variable(P, y[1:ny, 1:S] >= 0)
    @constraint(P, [k=1:m, s=1:S], sum(a * y[j, s] for (j, a) in C.rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0) <= nd.u[k, s] + nz_gx(nd, k, s, x̄))
    fη = @expression(P, dot(nd.c0, h) + sum(ce(j, s) * y[j, s] for j in 1:ny, s in 1:S))
    CL = @expression(P, sum(r[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0; init=AffExpr(0.0)))
    @objective(P, Max, fη); optimize!(P); zη = objective_value(P)
    @objective(P, Min, CL + μ * (zη - fη)); optimize!(P)
    return objective_value(P)
end
"정확한 pessimistic 값 V_d(x̄): follower 최적 plan 중 C_L 최대 (LP 둘)"
function pessval(x̄, d, r)
    nh = nz_nh(nd)
    P = Model(lp_opt)
    @variable(P, 0 <= h[i=1:nh] <= nd.hU[i]); @constraint(P, nd.W * h .<= nd.wvec); @variable(P, y[1:ny, 1:S] >= 0)
    @constraint(P, [k=1:m, s=1:S], sum(a * y[j, s] for (j, a) in C.rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0) <= nd.u[k, s] + nz_gx(nd, k, s, x̄))
    f0 = @expression(P, dot(nd.c0, h) + sum(d[s] * nd.c[j] * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0))
    CL = @expression(P, sum(r[s] * nd.ell[j] * y[j, s] for j in 1:ny, s in 1:S if nd.ell[j] != 0; init=AffExpr(0.0)))
    @objective(P, Max, f0); optimize!(P); z0 = objective_value(P)
    @constraint(P, f0 >= z0 - 1e-9 * max(1.0, abs(z0)))
    @objective(P, Max, CL); optimize!(P)
    return objective_value(P)
end
"belief d 에서 α (h 고정) 의 follower 비최적: (z0 − f0(α), 상대값). 최대화형"
function fsub(x̄, d, α)
    nh = nz_nh(nd)
    P = Model(lp_opt)
    @variable(P, 0 <= h[i=1:nh] <= nd.hU[i]); @constraint(P, nd.W * h .<= nd.wvec); @variable(P, y[1:ny, 1:S] >= 0)
    @constraint(P, [k=1:m, s=1:S], sum(a * y[j, s] for (j, a) in C.rowA[k]) - (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0) <= nd.u[k, s] + nz_gx(nd, k, s, x̄))
    f0 = @expression(P, dot(nd.c0, h) + sum(d[s] * nd.c[j] * y[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0))
    @objective(P, Max, f0); optimize!(P); z0 = objective_value(P)
    for i in 1:nh; fix(h[i], clamp(α[i], 0.0, nd.hU[i]); force=true); end
    optimize!(P); termination_status(P) == MOI.OPTIMAL || return (Inf, Inf)
    fa = objective_value(P)
    return (z0 - fa, (z0 - fa) / max(1.0, abs(z0)))
end
function parsex(xs)
    x̄ = zeros(nd.nx)
    for p in split(strip(xs, ['{', '}']), ";"; keepempty=false)
        i, t = parse.(Int, split(p, ":")); k = findfirst(==(Float64(t)), tiers); x̄[(i - nA - 1) * K + k] = 1.0
    end
    loc_tkp_xstr(nd, x̄) == xs || error("x 복원 실패"); x̄
end
inset(d) = all(d .>= bx.dl .- 1e-12) && all(d .<= bx.du .+ 1e-12) && sum(abs.(d .- qh)) <= 2nd.eps_tilde + 1e-12
cands = [copy(qh)]
for i in 1:S, j in 1:S
    i == j && continue
    mv = min(qh[i] - bx.dl[i], bx.du[j] - qh[j], nd.eps_tilde); mv <= 1e-9 && continue
    dd = copy(qh); dd[i] -= mv; dd[j] += mv; inset(dd) && push!(cands, dd)
end
ev = BCCGEval(nd, env)
for xs in split(envf("XS", "{10:400}|{3:200;10:200;12:300}|{4:200;7:200;12:200}"), "|")
    x̄ = parsex(String(xs)); _bccg_setx!(ev, nd, x̄)
    rows = NamedTuple[]
    n = 0; nz = 0; nlow = 0; nlow_z = 0; nfix = 0; nbad2 = 0; worst = 0.0; worst2 = 0.0; nex1 = 0; nex2 = 0; nsg = 0; nsg_z = 0; nsg_fix = 0
    for d0 in cands
        v, dd, r, α = _bccg_g!(ev, nd, d0); dd === nothing && continue
        n += 1; z = count(<=(1e-9), dd); z > 0 && (nz += 1)
        L1 = blockval(x̄, dd, r, MU); L2 = blockval(x̄, dd, r, MU; allzero=true); Vx = pessval(x̄, dd, r)
        thr = 1e-4 * max(1.0, abs(Vx))
        L1 >= Vx - thr && (nex1 += 1); abs(L2 - Vx) <= thr && (nex2 += 1)
        if L1 < Vx - thr; nlow += 1; z > 0 && (nlow_z += 1); worst = max(worst, Vx - L1); L2 >= Vx - thr && (nfix += 1); end
        L2 > Vx + thr && (nbad2 += 1; worst2 = max(worst2, L2 - Vx))
        if L1 < v - 1e-4 * max(1.0, abs(v)); nsg += 1; z > 0 && (nsg_z += 1); L2 >= v - 1e-4 * max(1.0, abs(v)) && (nsg_fix += 1); end   # ccgbp 의 safeguard 판정 (블록 < 평가값 v)
        push!(rows, (v=v, Vx=Vx, L1=L1, sub=fsub(x̄, dd, α)))
    end
    sort!(rows; by=t -> -(t.v - t.Vx))
    for t in rows[1:min(5, length(rows))]
        @printf("    v %.3f, V_d %.3f (v − V_d %.3f), 블록 %.3f | α 의 follower 비최적 (dd 에서) %.3e (상대 %.2e)\n", t.v, t.Vx, t.v - t.Vx, t.L1, t.sub[1], t.sub[2])
    end
    @printf("    v − V_d > 1e-4|v| 인 것 %d / %d, 그 비최적 범위 [%.2e, %.2e] (절대)\n", count(t -> t.v - t.Vx > 1e-4 * max(1, abs(t.v)), rows), length(rows),
            minimum((t.sub[1] for t in rows if t.v - t.Vx > 1e-4 * max(1, abs(t.v))); init=NaN), maximum((t.sub[1] for t in rows if t.v - t.Vx > 1e-4 * max(1, abs(t.v))); init=NaN))
    @printf("x=%-26s belief %3d (0 확률 %3d) | 지금 블록: 정확 %3d, V_d 보다 낮음 %3d (그중 0 확률 %3d, 최대 부족 %.2f) | 0 시나리오도 섭동: 정확 %3d, 낮던 것 해결 %3d, V_d 보다 큼 %d (최대 %.2f) | safeguard 판정 %d (0 확률 %d, 섭동판으로 해결 %d)\n",
            xs, n, nz, nex1, nlow, nlow_z, worst, nex2, nfix, nbad2, worst2, nsg, nsg_z, nsg_fix); flush(stdout)
end
