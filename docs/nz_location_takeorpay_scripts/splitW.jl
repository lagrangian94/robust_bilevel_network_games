# dwLean.jl (2026-11-16) = dwTier.jl + switches DT_DS (belief radii), DT_NOC (skip Coupled), DT_NOOT (stop after the decisions; no true-cost evaluation).
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
mk(S, seed) = with_takeorpay(make_sgb15_fac(; S=S, seed=seed, dem=:region, p=P, f=FR * P, eps_hat=0.0, eps_tilde=0.0, beta=β, quota=vcat(fill(WQ, nA), fill(WQ, nst - nA)), bB=BB))
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
const G = ENV["FB_G"]; const NSEED = parse(Int, E("TW_SEEDS", "10")); const MAXD = parse(Int, E("TW_MAXD", "0")); const KS = parse(Int, E("TW_KS", "1000")); const KD = parse(Int, E("SP_KD", "0"))
t0 = time(); cfgof(s) = s <= 3 ? "dw75" : "dwl75"
SITES = [collect(c) for k in 1:3 for c in Iterators.product(ntuple(_ -> 1:nd.nx, k)...) if issorted(collect(c)) && allunique(collect(c))]
ALLD = [(s, collect(lv)) for s in SITES for lv in Iterators.product(ntuple(_ -> TIERS, length(s))...)]; LAB = [dstr(s, lv) for (s, lv) in ALLD]; nALL = length(ALLD); IDX = Dict(LAB[i] => i for i in 1:nALL)
costv(q0, c) = [q0 * length(s) + c * sum(lv) for (s, lv) in ALLD]; CST = [costv(q0, c) for (q0, c) in SCEN]; CAP = [sum(lv) for (s, lv) in ALLD]; vv = nd.meta[:v]; const NK = length(SCEN)
# ---- decisions made from the training samples (seeds 1..NSEED): all model decisions (for the screen) and N per (cost scenario, seed)
DECI = Int[]; NDEC = Dict{Tuple{Int,Int},Int}()
for s in 1:NSEED, l in readlines(SPD * "geo/dw/sel_$(cfgof(s))_g$(G)_s$(s).csv")[2:end]
    f = split(l, ","); f[4] in ("N", "D0.1", "D0.3", "B0.1", "C") && push!(DECI, IDX[f[5]]); f[4] == "N" && (NDEC[(parse(Int, f[1]), s)] = IDX[f[5]])
end
# ---- the true F (test sample) and its 100-scenario representation F_B (same as truthW.jl)
ndT = mk(NT, TSEED); const pT = fill(1 / NT, NT)
subset(ndx, idx) = NZData(ndx.name * "_sub", ndx.A, ndx.c, ndx.ell, ndx.u[:, idx], ndx.hrow, ndx.hcoef, ndx.xrow, ndx.g[:, idx], ndx.W, ndx.wvec, ndx.c0, ndx.hU, ndx.nx, ndx.x_allowed, ndx.gamma, ndx.qx,
    length(idx), fill(1 / length(idx), length(idx)), ndx.eps_hat, ndx.eps_tilde, ndx.beta, ndx.thetaU, ndx.piLU, ndx.piFU, ndx.lambdaU, ndx.varpiU, ndx.meta)
totT = vec(sum(ndT.meta[:xi]; dims=1)); idxB = sortperm(totT)[3:5:NT]; ndB = subset(ndT, idxB); const NB = length(idxB); const pB = fill(1 / NB, NB)
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
phiT = phi_setup(ndT); phiB = phi_setup(ndB); setpB!, solveB = resp_setup(ndB); setpB!(pB)
JC = Dict{Tuple{Int,Vector{Float64}},Float64}()
Jof(i, h) = (hr = round.(h; digits=4); get!(() -> cvar(phiT(xfrac(ALLD[i]...), hr), pT), JC, (i, hr)))
mkpath(SPD * "geo/tw"); @printf("SP g%s: %d decisions, models built %.0f s\n", G, nALL, time() - t0); flush(stdout)
# ---- 1. screened decisions under F_B (same screen as truthW.jl): commitment of a retailer who knows the truth h_B(x), value V_B(x), N*
VS = zeros(nALL, NSEED)
for s in 1:NSEED
    L = readlines(SPD * "geo/dw/vn_$(cfgof(s))_g$(G)_s$(s).csv")[2:end]; length(L) == nALL || error("vn rows $(length(L)) != $nALL (seed $s)")
    for (i, l) in enumerate(L); f = split(l, ","); f[1] == LAB[i] || error("vn label mismatch seed $s row $i"); VS[i, s] = parse(Float64, f[4]); end
end
vbar = vec(mean(VS; dims=2)); SCRORD = [sortperm(collect(zip(vbar .+ CST[k], LAB))) for k in 1:NK]
ENUM = MAXD > 0 ? sort(unique(vcat(collect(1:MAXD), DECI))) : KS > 0 ? sort(unique(vcat([o[1:min(KS, nALL)] for o in SCRORD]..., DECI))) : collect(1:nALL)
@printf("  enumerating %d decisions (screen KS = %d)\n", length(ENUM), KS); flush(stdout)
H = zeros(nh, nALL); VB = fill(Inf, nALL)
for (c, i) in enumerate(ENUM)
    x̄ = xfrac(ALLD[i]...); H[:, i] = solveB(x̄); VB[i] = cvar(phiB(x̄, H[:, i]), pB)
    c % 500 == 0 && (@printf("  enum %d / %d  %.0f s\n", c, length(ENUM), time() - t0); flush(stdout))
end
NST = [ENUM[argmin([(VB[i] + CST[k][i], LAB[i]) for i in ENUM])] for k in 1:NK]
open(SPD * "geo/tw/hB_g$(G).csv", "w") do io
    println(io, "x,VB," * join(["h$(j)" for j in 1:nh], ",")); for i in ENUM; println(io, LAB[i], ",", @sprintf("%.4f", VB[i]), ",", join([@sprintf("%.3f", H[j, i]) for j in 1:nh], ",")); end
end
@printf("  truth enumeration done %.0f s; N* = %s\n", time() - t0, join([LAB[NST[k]] for k in 1:NK], " / ")); flush(stdout)
# ---- 2. per training sample: three leaders
#   N  (standard)        : own demand sample, commitment predicted from the same sample     V_ss(x) = CVaR_q[phi_s(x, h_s(x))]
#   Nr (order known)     : own demand sample, commitment of the truth-knowing retailer       V_rs(x) = CVaR_q[phi_s(x, h_B(x))]
#   Nd (demand known)    : true demand F_B, commitment predicted from the own sample         V_ds(x) = CVaR_pB[phi_B(x, h_s(x))]
ROWS = String[]; NEW = Int[]; bord = [sortperm([(VB[i] + CST[k][i], LAB[i]) for i in ENUM]) for k in 1:NK]
for s in 1:NSEED
    nds = mk(10, s); qs = nds.q_hat; setps!, solves = resp_setup(nds); setps!(qs); phis = phi_setup(nds)
    Vss = fill(Inf, nALL); Vrs = fill(Inf, nALL); Vds = fill(Inf, nALL); HS = zeros(nh, nALL)
    for i in ENUM
        x̄ = xfrac(ALLD[i]...); HS[:, i] = solves(x̄); Vss[i] = cvar(phis(x̄, HS[:, i]), qs); Vrs[i] = cvar(phis(x̄, H[:, i]), qs)
    end
    dmax = maximum(abs.(Vss[ENUM] .- VS[ENUM, s])); t1 = time() - t0
    vds!(i) = (isfinite(Vds[i]) || (Vds[i] = cvar(phiB(xfrac(ALLD[i]...), HS[:, i]), pB)); Vds[i])
    for k in 1:NK
        ED = KD > 0 ? unique(vcat(ENUM[bord[k][1:min(KD, end)]], ENUM[sortperm([(Vss[i] + CST[k][i], LAB[i]) for i in ENUM])[1:min(KD, end)]], ENUM[sortperm([(Vrs[i] + CST[k][i], LAB[i]) for i in ENUM])[1:min(KD, end)]])) : ENUM
        for i in ED; vds!(i); end
        iN = ENUM[argmin([(Vss[i] + CST[k][i], LAB[i]) for i in ENUM])]; iR = ENUM[argmin([(Vrs[i] + CST[k][i], LAB[i]) for i in ENUM])]; iD = ED[argmin([(Vds[i] + CST[k][i], LAB[i]) for i in ED])]
        i0 = NDEC[(k, s)]; okN = iN == i0
        okN || @printf("  NOTE seed %d scen %d: recomputed N %s (%.4f) differs from stored N %s (%.4f)\n", s, k, LAB[iN], Vss[iN] + CST[k][iN], LAB[i0], Vss[i0] + CST[k][i0])
        for (who, i) in (("N", i0), ("Nr", iR), ("Nd", iD), ("N*", NST[k]))
            push!(ROWS, join([string(k), string(SCEN[k][1]), string(SCEN[k][2]), string(s), who, LAB[i], @sprintf("%.0f", CAP[i]), @sprintf("%.4f,%.4f,%.4f,%.4f", Vss[i], Vrs[i], vds!(i), VB[i]),
                @sprintf("%.2f,%.2f,%.2f", sum(HS[nA+1:end, i]), sum(H[nA+1:end, i]), sum(abs.(HS[nA+1:end, i] .- H[nA+1:end, i]))), string(okN), @sprintf("%.5f", dmax)], ","))
        end
        push!(NEW, iR, iD)
    end
    @printf("  seed %d done %.0f s (own-sample part %.0f s), max |V_ss - stored| = %.5f\n", s, time() - t0, t1, dmax); flush(stdout)
end
open(SPD * "geo/tw/split_g$(G).csv", "w") do io
    println(io, "scen,q0,c,seed,who,x,cap,Vss,Vrs,Vds,VB,hB_pred,hB_true,hB_l1,okN,dmax"); for r in ROWS; println(io, r); end
end
# ---- 3. true-cost evaluation of the new decisions (same worlds and draws as truthW.jl); decisions already in eval_g<G>.csv are skipped
HAVE = Set{String}(String(split(l, ",")[1]) for l in readlines(SPD * "geo/tw/eval_g$(G).csv")[2:end])
TODO = [i for i in sort(unique(NEW)) if !(LAB[i] in HAVE)]; @printf("  new decisions %d, to evaluate %d\n", length(unique(NEW)), length(TODO)); flush(stdout)
open(SPD * "geo/tw/eval2_g$(G).csv", "w") do ioe
    println(ioe, "x,world,k,J")
    for i in TODO; @printf(ioe, "%s,T,0,%.6f\n", LAB[i], Jof(i, H[:, i])); end
    for nf in NFS, kd in 1:K
        ndF = mk(nf, 6_000_000 + 1000 * parse(Int, G) + kd + (nf == 5 ? 0 : 10_000_000 * nf)); pF = fill(1 / nf, nf)
        for i in TODO
            x̄ = xfrac(ALLD[i]...); h = _nz_resp!(_nz_build_resp(ndF, x̄; optimizer=GRB1), ndF, pF, apess); h === nothing && error("retailer LP failed (nf=$nf k=$kd $(LAB[i]))")
            @printf(ioe, "%s,%d,%d,%.6f\n", LAB[i], nf, kd, Jof(i, h))
        end
    end
end
@printf("SP_DONE g%s: %d new decisions evaluated, %.0f s\n", G, length(TODO), time() - t0)
