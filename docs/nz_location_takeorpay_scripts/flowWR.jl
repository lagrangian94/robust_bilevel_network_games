# dwLeanP.jl (2026-11-16 night) = dwLean.jl + SC_VSM: leader margin on B at-once units = VSM * v (price-based revenue f*h + p*spot, VSM = p/f). VSM = 1 is dwLean.jl.
# dwTier (2026-11-15): 자료 세계 (리더가 q̂ 를 추정) + 약정 물량 방식 + 거점별 용량 단계. dwS.jl (결정) 과 otS.jl (진짜 F 평가) 를 tkT.jl 의 결정 공간에 맞춰 합친 것.
#   결정 = (입지 ≤ 3 곳, 거점마다 용량 단계), x̄_i = b_i / bBmax 가 용량 행에만 들어감, s_i 는 용량 제약 안. 값은 개설비 없이 저장하고 점포 비용 q0 + c·b 는 사후.
#   리더: 표본 시나리오 10 개 (q̂ 균등). N = argmin V_N, D_ε = argmin V_D (TV(q̂, ε) ∩ 상자, 전체 열거), B_δ = argmin_{후보} V_N + adv_δ, C = argmin_{후보} V_Ch (ε̂ = 0.3, δ = 0.1).
#   후보 = 비용 시나리오별 V_N 상위·V_D 상위 + 총 용량 구간별 상위 입지. 평가 = 진짜 F 시험 표본 500 개의 CVaR, 소매상 belief = F 에서 새로 뽑은 N_F 개 × K 번.
#   출력 geo/dw/vn_<TAG>.csv (전체 결정), geo/dw/dw_<TAG>.csv (후보), geo/dw/sel_<TAG>.csv (비용 시나리오·모형별 결정), geo/oos/true/ot_<TAG>.csv (nf,k,x,J0). 저장소 코드는 include 만 한다.
root = "C:/Users/user/robust_bilevel_network_games"
const SPD = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Distributions, Statistics
const GRB_ENV = Gurobi.Env()
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(SPD * "geo/nz_data_geo2.jl")
include(SPD * "box/nz_exact_eval_box.jl"); include(joinpath(root, "nonzero_sum", "nz_resp_heuristic.jl"))
include(SPD * "geo/fac_builder_geo2.jl")
E(k, d) = get(ENV, k, d)
SEED = parse(Int, ENV["C0H_SEED"]); TAG = ENV["C0H_TAG"]; const β = parse(Float64, E("SC_BETA", "0.4")); const P = 800.0; const FR = parse(Float64, E("SC_FR", "0.25"))
const BB = parse(Float64, E("SC_BB", "300")); const WQ = 300.0; const NR = 20; const DS = parse.(Float64, split(E("DT_DS", "0.05,0.1,0.2,0.3"), ",")); const NOC = E("DT_NOC", "0") == "1"; const NOOT = E("DT_NOOT", "0") == "1"; const EDS = [0.1, 0.2, 0.3]
const TIERS = parse.(Float64, split(E("SC_TIERS", "200,250,300"), ",")); const SCEN = [(0.0, 2.5), (150.0, 2.0)]
const NCN = parse.(Int, split(E("DT_NCN", "12,6"), ",")); const NCD = parse.(Int, split(E("DT_NCD", "4,2"), ",")); const STRK = parse(Int, E("DT_STRK", "3")); const STRMIN = 450.0
const NFS = parse.(Int, split(E("DT_NF", "5,20"), ",")); const K = parse(Int, E("DT_K", "40")); const NT = parse(Int, E("DT_NT", "500")); const TSEED = 7770001
const MAXC = parse(Int, E("DT_MAXC", "0"))   # 시험용: 후보 수 상한 (0 = 제한 없음)
nd0 = make_sgb15_fac(; S=2, seed=SEED, dem=:region, p=P, f=FR * P, eps_hat=0.0, eps_tilde=0.0, beta=β, bB=BB)
const nA = nd0.meta[:nA]; const nst = nd0.meta[:nst]; const ncu = nd0.meta[:ncu]
"약정 물량 방식 (tkT.jl 과 같은 구성, s_i 를 용량 제약에 포함)"
function with_takeorpay(nd)
    m, ny = nz_m(nd), nz_ny(nd); S = nd.S; rres = nd.meta[:rres]; iyR = nd.meta[:iyR]; hidx = nd.meta[:hidx]; nB = nst - nA
    A = zeros(m + nB, ny + nB); A[1:m, 1:ny] = nd.A
    hrow = vcat(nd.hrow, zeros(Int, nB)); hcoef = vcat(nd.hcoef, zeros(nB))
    for (k, i) in enumerate(nA+1:nst)
        A[rres(i, 1), ny+k] = 1.0
        A[nd.meta[:rcap](i), ny+k] = 1.0
        for j in 1:ncu; A[m+k, iyR(i, j)] = -1.0; end
        A[m+k, ny+k] = -1.0; hrow[m+k] = hidx(i, 1); hcoef[m+k] = -1.0
    end
    return NZData(nd.name * "_top", A, vcat(nd.c, zeros(nB)), vcat(nd.ell, fill(-nd.meta[:v], nB)), vcat(nd.u, zeros(nB, S)), hrow, hcoef, vcat(nd.xrow, zeros(Int, nB)), vcat(nd.g, zeros(nB, S)), nd.W, nd.wvec, nd.c0, nd.hU,
        nd.nx, nd.x_allowed, nd.gamma, nd.qx, nd.S, nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta,
        nd.thetaU, vcat(nd.piLU, zeros(nB)), vcat(nd.piFU, zeros(nB)), nd.lambdaU, vcat(nd.varpiU, fill(Inf, nB)), nd.meta)
end
const VSM = parse(Float64, E("SC_VSM", "1.0")); setvs!(ndx) = (iyS = ndx.meta[:iyS]; for i in nA+1:nst, j in 1:ncu; ndx.ell[iyS(i, j)] = -VSM * ndx.meta[:v]; end; ndx); mk0(S, seed) = setvs!(with_takeorpay(make_sgb15_fac(; S=S, seed=seed, dem=:region, p=P, f=FR * P, eps_hat=0.0, eps_tilde=0.0, beta=β, quota=vcat(fill(WQ, nA), fill(WQ, nst - nA)), bB=BB)))
# dwLeanR.jl (2026-11-17 night): the retailer may leave demand unserved.  Column u_j >= 0 in the demand row (sum_i (yR + yS) + u_j >= xi_j), retailer cost r u_j (r = value of a unit sold to the customer, env SC_R; 0 = off), leader gets nothing from u_j.
const RSELL = parse(Float64, E("SC_R", "0"))
function with_shortage(ndx, r)
    r > 0 || return ndx
    m, ny = nz_m(ndx), nz_ny(ndx); rd = ndx.meta[:rdem]; A = hcat(ndx.A, zeros(m, ncu)); for j in 1:ncu; A[rd(j), ny+j] = -1.0; end
    return NZData(ndx.name * "_short", A, vcat(ndx.c, fill(-r, ncu)), vcat(ndx.ell, zeros(ncu)), ndx.u, ndx.hrow, ndx.hcoef, ndx.xrow, ndx.g, ndx.W, ndx.wvec, ndx.c0, ndx.hU,
        ndx.nx, ndx.x_allowed, ndx.gamma, ndx.qx, ndx.S, ndx.q_hat, ndx.eps_hat, ndx.eps_tilde, ndx.beta, ndx.thetaU, ndx.piLU, ndx.piFU, ndx.lambdaU, ndx.varpiU, ndx.meta)
end
mk(S, seed) = with_shortage(mk0(S, seed), RSELL)
nd = mk(10, SEED); const BBMAX = nd.meta[:bB]; @assert maximum(TIERS) <= BBMAX + 1e-9
const S = nd.S; const q = nd.q_hat; nh = nz_nh(nd); nh == nst || error("h 가 거점별이 아님: nh=$nh nst=$nst"); const apess = vcat(nd.hU[1:nA], zeros(nh - nA))
dstr(sites, lv) = "{" * join(["$(i):$(round(Int, b))" for (i, b) in zip(sites, lv)], ";") * "}"
function xfrac(sites, lv); v = zeros(nd.nx); for (i, b) in zip(sites, lv); v[i] = b / BBMAX; end; v; end
function cvar(Z, p)
    ord = sortperm(Z; rev=true); cap = p ./ (1 - β); rest = 1.0; val = 0.0
    for s in ord; m = min(cap[s], rest); val += m * Z[s]; rest -= m; rest <= 1e-12 && break; end
    return val
end
"max_a CVaR^a[Z]: a ∈ TV(q̂, ε) ∩ 상자 (Γ = 2ε/√S), (δ 유한이면) TV(a, d) ≤ δ"
function worst(Z, d, δ, eps)
    Γ = 2eps / sqrt(S); m = Model(GRB1); lo = max.(0.0, q .- Γ); hi = min.(1.0, q .+ Γ)
    @variable(m, lo[s] <= a[s=1:S] <= hi[s]); @variable(m, t[1:S] >= 0); @variable(m, w[1:S] >= 0)
    @constraint(m, sum(a) == 1); @constraint(m, [s=1:S], t[s] >= a[s] - q[s]); @constraint(m, [s=1:S], t[s] >= q[s] - a[s]); @constraint(m, sum(t) <= 2eps)
    if isfinite(δ)
        @variable(m, u[1:S] >= 0); @constraint(m, [s=1:S], u[s] >= a[s] - d[s]); @constraint(m, [s=1:S], u[s] >= d[s] - a[s]); @constraint(m, sum(u) <= 2δ)
    end
    @constraint(m, [s=1:S], w[s] <= a[s] / (1 - β)); @constraint(m, sum(w) == 1); @objective(m, Max, dot(w, Z)); optimize!(m)
    return termination_status(m) == MOI.OPTIMAL ? (objective_value(m), value.(a)) : (-Inf, nothing)
end
function ball_pts(t, rng)
    pts = Vector{Vector{Float64}}()
    for i in 1:S, j in 1:S
        i == j && continue
        d = copy(q); m = min(t, d[i]); d[i] -= m; d[j] += m
        rest = t - m; k = 1; o = sortperm(q; rev=true)
        while rest > 1e-12 && k <= S
            s = o[k]; (s == j || d[s] <= 0) && (k += 1; continue); mm = min(rest, d[s]); d[s] -= mm; d[j] += mm; rest -= mm; k += 1
        end
        push!(pts, d)
    end
    for _ in 1:NR
        Q = rand(rng, Dirichlet(0.5 .* ones(S))); dd = 0.5 * sum(abs.(Q .- q)); dd < 1e-9 && continue
        d = q .+ (t / dd) .* (Q .- q); minimum(d) >= 0 && push!(pts, d ./ sum(d))
    end
    return pts
end
const EPS = 0.3; const DEL = EPS / 3; const ΓC = 2EPS / sqrt(S); const NS = 60; const NRD = 30; const RND = 3
function belief_pts(rng)
    pts = [copy(q)]
    for t in (0.1, 0.2, 0.3)
        kf = ceil(Int, t / 0.1 - 1e-9); kt = ceil(Int, t / ΓC - 1e-9)
        for _ in 1:NS
            p = randperm(rng, S); d = copy(q); d[p[1:kf]] .-= t / kf; d[p[kf+1:kf+kt]] .+= t / kt; push!(pts, d)
        end
        for _ in 1:NRD
            Q = rand(rng, Dirichlet(0.5 .* ones(S))); dd = 0.5 * sum(abs.(Q .- q)); dd < 1e-9 && continue
            d = q .+ (t / dd) .* (Q .- q); (minimum(d) >= 0 && maximum(abs.(d .- q)) <= ΓC + 1e-9) && push!(pts, d ./ sum(d))
        end
    end
    return unique(x -> round.(x; digits=8), pts)
end
"결정 x̄ 의 belief d ↦ 시나리오별 φ (개설비 없음)"
function evalx(x̄)
    R = _nz_build_resp(nd, x̄; optimizer=GRB1); hc = Dict{Vector{Float64},Vector{Float64}}(); dc = Dict{Vector{Float64},Vector{Float64}}()
    return d -> get!(dc, round.(d; digits=8)) do
        h = _nz_resp!(R, nd, d, apess); h === nothing && error("응답 실패")
        get!(() -> nz_recourse_values(nd, x̄, h)[2], hc, round.(h; digits=6))
    end
end
# zsW.jl (2026-11-17) = dwLeanP.jl lines 1-99 + this body. Evaluation on the zero-sum grid (manuscript 6.1 OOS): support = the 10 training scenarios, only probabilities change.
#   draw: alpha = max(gamma (1 + zeta z), 0.01), z ~ N(0, I);  p ~ Dir(alpha);  independently alpha2 (same rule), p2 ~ Dir(alpha2).
#   world K  (leader knows the demand distribution): truth = q̂, retailer belief = p.            cost = store cost + CVaR_q̂[phi(x, h(x, p))]
#   world U  (leader does not):                       truth = p,  retailer belief = rho p + (1 - rho) p2, rho in ZS_RHO.   cost = store cost + CVaR_p[phi(x, h(x, belief))]
#   Decisions are the stored ones of config ZS_CFG (N, B0.1, D0.1, D0.3, C if present); nothing is re-solved.
const G = ENV["FB_G"]; const SS = parse(Int, ENV["ON_SEED"]); const CFGZ = E("ZS_CFG", "dwpl75"); const OUT = E("ZS_OUT", "zs"); const MODELS = Tuple(String.(split(E("ZS_MODELS", "N,B0.1,D0.1,D0.3,C"), ",")))
const GAMS = parse.(Float64, split(E("ZS_GAMMA", "0.1,0.3,0.5,1"), ",")); const ZETS = parse.(Float64, split(E("ZS_ZETA", "0,0.5,1,2,5"), ",")); const MZ = parse(Int, E("ZS_M", "50")); const RHOS = parse.(Float64, split(E("ZS_RHO", "1,0.5,0"), ","))
const SEEDB = parse(Int, E("ZS_SEEDBASE", "9191"))
t0 = time()
SITES = [collect(c) for k in 1:3 for c in Iterators.product(ntuple(_ -> 1:nd.nx, k)...) if issorted(collect(c)) && allunique(collect(c))]
ALLD = [(s, collect(lv)) for s in SITES for lv in Iterators.product(ntuple(_ -> TIERS, length(s))...)]; LAB = [dstr(s, lv) for (s, lv) in ALLD]; nALL = length(ALLD); IDX = Dict(LAB[i] => i for i in 1:nALL)
CAP = [sum(lv) for (s, lv) in ALLD]; const NK = length(SCEN)
DEC = Dict{Tuple{Int,String},Int}()
for l in readlines(SPD * "geo/dw/sel_$(CFGZ)_g$(G)_s$(SS).csv")[2:end]
    f = split(l, ","); f[4] in MODELS || continue; DEC[(parse(Int, f[1]), String(f[4]))] = IDX[f[5]]
end
for k in 1:NK; haskey(DEC, (k, "N")) || error("no stored N for scen $k"); end
EV = sort(unique(collect(values(DEC))))
function resp_setup(ndx)
    m, ny, S = nz_m(ndx), nz_ny(ndx), ndx.S; xr = [k for k in 1:m if ndx.xrow[k] > 0]; jc = [j for j in 1:ny if ndx.c[j] != 0]
    function base()
        mdl = Model(GRB1)
        @variable(mdl, 0 <= α[i=1:nh] <= ndx.hU[i]); @constraint(mdl, [l=1:size(ndx.W, 1)], sum(ndx.W[l, i] * α[i] for i in 1:nh) <= ndx.wvec[l]); @variable(mdl, y[1:ny, 1:S] >= 0)
        con = [@constraint(mdl, sum(ndx.A[k, j] * y[j, s] for j in 1:ny if ndx.A[k, j] != 0) - (ndx.hrow[k] > 0 ? ndx.hcoef[k] * α[ndx.hrow[k]] : 0.0) <= ndx.u[k, s]) for k in 1:m, s in 1:S]
        return mdl, α, y, con
    end
    mA, αA, yA, cA = base(); mB, αB, yB, cB = base()
    @variable(mB, t[1:nh] >= 0); @constraint(mB, [i=1:nh], t[i] - αB[i] >= -apess[i]); @constraint(mB, [i=1:nh], t[i] + αB[i] >= apess[i]); @objective(mB, Min, sum(t))
    copt = Ref{Any}(nothing)
    function val(α, y, p)
        e = AffExpr(0.0); for i in 1:nh; add_to_expression!(e, ndx.c0[i], α[i]); end
        for s in 1:S; p[s] == 0 && continue; for j in jc; add_to_expression!(e, p[s] * ndx.c[j], y[j, s]); end; end
        return e
    end
    function setp!(p)
        p = p ./ sum(p); set_objective(mA, MOI.MAX_SENSE, val(αA, yA, p)); copt[] === nothing || delete(mB, copt[]); copt[] = @constraint(mB, val(αB, yB, p) >= -1e30); return nothing
    end
    function solve(x̄)
        for k in xr, s in 1:S; r = ndx.u[k, s] + ndx.g[k, s] * x̄[ndx.xrow[k]]; set_normalized_rhs(cA[k, s], r); set_normalized_rhs(cB[k, s], r); end
        optimize!(mA); termination_status(mA) == MOI.OPTIMAL || error("response LP: $(termination_status(mA))")
        v = objective_value(mA); set_normalized_rhs(copt[], v - 1e-7 * max(1.0, abs(v))); optimize!(mB)
        return clamp.(termination_status(mB) == MOI.OPTIMAL ? value.(αB) : value.(αA), 0.0, ndx.hU)
    end
    return setp!, solve
end
function phi_setup(ndx)
    m, ny, S = nz_m(ndx), nz_ny(ndx), ndx.S; mdl = Model(GRB1); @variable(mdl, y[1:ny] >= 0)
    cons = [@constraint(mdl, sum(ndx.A[k, j] * y[j] for j in 1:ny if ndx.A[k, j] != 0) <= 0.0) for k in 1:m]
    cq = @constraint(mdl, dot(ndx.c, y) >= -1e30); oQ = @expression(mdl, dot(ndx.c, y)); oL = @expression(mdl, dot(ndx.ell, y))
    return (x̄, h) -> begin
        φ = zeros(S)
        for s in 1:S
            rr = nz_rhs(ndx, x̄, h, s); for k in 1:m; set_normalized_rhs(cons[k], rr[k]); end
            set_normalized_rhs(cq, -1e30); @objective(mdl, Max, oQ); optimize!(mdl); termination_status(mdl) == MOI.OPTIMAL || error("recourse LP s=$s: $(termination_status(mdl))")
            Q = objective_value(mdl); set_normalized_rhs(cq, Q - 1e-9 * max(1.0, abs(Q))); @objective(mdl, Max, oL); optimize!(mdl); termination_status(mdl) == MOI.OPTIMAL || error("pessimistic LP s=$s: $(termination_status(mdl))")
            φ[s] = objective_value(mdl)
        end
        φ
    end
end
"one draw of the zero-sum scheme: Dir(max(gamma (1 + zeta z), 0.01)); redrawn if the sample is not a probability vector (underflow)"
function draw_dir(rng, γ, ζ)
    for _ in 1:1000
        α = max.(γ .* (1.0 .+ ζ .* randn(rng, S)), 0.01); p = rand(rng, Dirichlet(α))
        (all(isfinite, p) && abs(sum(p) - 1) < 1e-9 && minimum(p) >= 0) && return p ./ sum(p)
    end
    error("draw_dir failed (γ = $γ, ζ = $ζ)")
end
tv(a, b) = 0.5 * sum(abs.(a .- b))
nds = mk(10, SS); qs = nds.q_hat; maximum(abs.(qs .- q)) < 1e-12 || error("q̂ is not uniform"); setps!, solves = resp_setup(nds); phis = phi_setup(nds)
ZC = Dict{Tuple{Int,Vector{Float64}},Vector{Float64}}()
Zof(i, h) = (hr = round.(h; digits=4); get!(() -> phis(xfrac(ALLD[i]...), hr), ZC, (i, hr)))
gi = parse(Int, G); mkpath(SPD * "geo/$(OUT)")
@printf("ZS g%s seed %d: VSM %.4f, decisions of %s (%d distinct), γ = %s, ζ = %s, M = %d, ρ = %s, models built %.0f s\n", G, SS, VSM, CFGZ, length(EV), string(GAMS), string(ZETS), MZ, string(RHOS), time() - t0); flush(stdout)
# ---- q̂ pass: what the leader sees
setps!(qs); Vq = Dict(i => cvar(Zof(i, solves(xfrac(ALLD[i]...))), qs) for i in EV)
open(SPD * "geo/$(OUT)/dec_g$(G)_s$(SS).csv", "w") do io
    println(io, "seed,scen,q0,c,who,x,cap,Vq")
    for k in 1:NK, w in MODELS; haskey(DEC, (k, w)) || continue; i = DEC[(k, w)]; println(io, join([string(SS), string(k), string(SCEN[k][1]), string(SCEN[k][2]), w, LAB[i], @sprintf("%.0f", CAP[i]), @sprintf("%.4f", Vq[i])], ",")); end
end
# flowW.jl (2026-11-17 night) = dwLeanP.jl lines 1-99 + zs_body.jl lines 1-80 + this body.  Actual shipments of the retailer for the Nominal decision (cost table scen FL_SCEN), belief q̂.
#   Output geo/<ZS_OUT>/flow_g_s.csv: seed,x,i,kind,sreg,open_cap,j,creg,dist,mu,yR,yS  (yR, yS = probability-weighted mean shipments over the 10 scenarios)
function flow_setup(ndx)
    m, ny, S = nz_m(ndx), nz_ny(ndx), ndx.S; mdl = Model(GRB1); @variable(mdl, y[1:ny] >= 0)
    cons = [@constraint(mdl, sum(ndx.A[k, j] * y[j] for j in 1:ny if ndx.A[k, j] != 0) <= 0.0) for k in 1:m]
    cq = @constraint(mdl, dot(ndx.c, y) >= -1e30); oQ = @expression(mdl, dot(ndx.c, y)); oL = @expression(mdl, dot(ndx.ell, y))
    return (x̄, h) -> begin
        Y = zeros(ny, S)
        for s in 1:S
            rr = nz_rhs(ndx, x̄, h, s); for k in 1:m; set_normalized_rhs(cons[k], rr[k]); end
            set_normalized_rhs(cq, -1e30); @objective(mdl, Max, oQ); optimize!(mdl); termination_status(mdl) == MOI.OPTIMAL || error("recourse LP s=$s: $(termination_status(mdl))")
            Q = objective_value(mdl); set_normalized_rhs(cq, Q - 1e-9 * max(1.0, abs(Q))); @objective(mdl, Max, oL); optimize!(mdl); termination_status(mdl) == MOI.OPTIMAL || error("pessimistic LP s=$s: $(termination_status(mdl))")
            Y[:, s] = value.(y)
        end
        Y
    end
end
const FSC = parse(Int, E("FL_SCEN", "2")); iN = DEC[(FSC, "N")]; sites, lv = ALLD[iN]; x̄ = xfrac(sites, lv)
setps!(qs); h = solves(x̄); Y = flow_setup(nds)(x̄, round.(h; digits=4))
iyR = nds.meta[:iyR]; iyS = nds.meta[:iyS]; cities = nds.meta[:cities]; dist = nds.meta[:dist]; creg = nds.meta[:region]; μ = sgb15_mu(sgb15_kuhn().pop); nstl = nds.meta[:nst]; ncul = nds.meta[:ncu]
open(SPD * "geo/$(OUT)/flow_g$(G)_s$(SS).csv", "w") do io
    println(io, "seed;x;i;kind;sreg;cap;h;j;creg;dist;mu;yR;yS")
    for i in 1:nstl
        capi = i <= nA ? nds.meta[:bA] : (k = findfirst(==(i - nA), sites); k === nothing ? 0.0 : lv[k]); (i > nA && capi == 0) && continue
        for j in 1:ncul
            println(io, join([SS, LAB[iN], i, i <= nA ? "A" : "B", REGION[cities[i]], capi, round(h[i]; digits=2), j, creg[j], dist[i, j], round(μ[j]; digits=1),
                round(dot(qs, Y[iyR(i, j), :]); digits=3), round(dot(qs, Y[iyS(i, j), :]); digits=3)], ";"))
        end
    end
end
if RSELL > 0
    open(SPD * "geo/$(OUT)/flow_g$(G)_s$(SS).csv", "a") do io
        nyl = size(Y, 1)
        for j in 1:ncul; println(io, join([SS, LAB[iN], 0, "U", 0, 0, 0, j, creg[j], 0, round(μ[j]; digits=1), 0, round(dot(qs, Y[nyl-ncul+j, :]); digits=3)], ";")); end
    end
end
@printf("FLOW_DONE g%s seed %d: %.0f s\n", G, SS, time() - t0)
