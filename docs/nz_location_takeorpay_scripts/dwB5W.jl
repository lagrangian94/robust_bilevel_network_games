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
const VSM = parse(Float64, E("SC_VSM", "1.0")); setvs!(ndx) = (iyS = ndx.meta[:iyS]; for i in nA+1:nst, j in 1:ncu; ndx.ell[iyS(i, j)] = -VSM * ndx.meta[:v]; end; ndx); mk(S, seed) = setvs!(with_takeorpay(make_sgb15_fac(; S=S, seed=seed, dem=:region, p=P, f=FR * P, eps_hat=0.0, eps_tilde=0.0, beta=β, quota=vcat(fill(WQ, nA), fill(WQ, nst - nA)), bB=BB)))
nd = mk(10, SEED); const BBMAX = nd.meta[:bB]; @assert maximum(TIERS) <= BBMAX + 1e-9
const S = nd.S; const q = nd.q_hat; nh = nz_nh(nd); nh == nst || error("h 가 거점별이 아님: nh=$nh nst=$nst"); const apess = vcat(nd.hU[1:nA], zeros(nh - nA))
dstr(sites, lv) = "{" * join(["$(i):$(round(Int, b))" for (i, b) in zip(sites, lv)], ";") * "}"
function xfrac(sites, lv); v = zeros(nd.nx); for (i, b) in zip(sites, lv); v[i] = b / BBMAX; end; v; end
const BOXT = parse(Float64, E("SC_BOXT", "0")); const BOXTB = parse(Float64, E("SC_BOXTB", "0")); boxw(r, t) = t > 0 ? 2t * r .* q : fill(2r / sqrt(S), S)   # box half-width of a ball of radius r
function cvar(Z, p)
    ord = sortperm(Z; rev=true); cap = p ./ (1 - β); rest = 1.0; val = 0.0
    for s in ord; m = min(cap[s], rest); val += m * Z[s]; rest -= m; rest <= 1e-12 && break; end
    return val
end
"max_a CVaR^a[Z]: a ∈ TV(q̂, ε) ∩ 상자 (Γ = 2ε/√S), (δ 유한이면) TV(a, d) ≤ δ"
function worst(Z, d, δ, eps)
    Γ = boxw(eps, BOXT); m = Model(GRB1); lo = max.(0.0, q .- Γ); hi = min.(1.0, q .+ Γ)
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
"belief ball of radius t WITH the proportional box (SC_BOXTB): NVB random extreme points (mass t taken from some scenarios, at most the box width each, and given to others) + NR random directions inside the box"
const NVB = parse(Int, E("SC_NVB", "150"))
function ball_pts_box(t, rng)
    Γ = boxw(t, BOXTB); pts = Vector{Vector{Float64}}()
    for _ in 1:NVB
        p = randperm(rng, S); d = copy(q); rest = t; k = 1
        while rest > 1e-12 && k <= S; s = p[k]; mm = min(rest, Γ[s], d[s]); d[s] -= mm; rest -= mm; k += 1; end
        kd = k - 1; rest = t; k = S
        while rest > 1e-12 && k > kd; s = p[k]; mm = min(rest, Γ[s]); d[s] += mm; rest -= mm; k -= 1; end
        rest <= 1e-9 && push!(pts, d)
    end
    for _ in 1:NR, _ in 1:200
        Q = rand(rng, Dirichlet(0.5 .* ones(S))); dd = 0.5 * sum(abs.(Q .- q)); dd < 1e-9 && continue
        d = q .+ (t / dd) .* (Q .- q); (minimum(d) >= 0 && all(abs.(d .- q) .<= Γ .+ 1e-9)) && (push!(pts, d ./ sum(d)); break)
    end
    return unique(x -> round.(x; digits=8), pts)
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
# dwB5W.jl (2026-11-18 evening) = dwLeanB.jl lines 1-117 + this body.  Belief-only at radius DT_BNEW (default 0.5) on the extended candidates candx_<cfg>_<eps>_g_s.csv of dwD5W.jl.  Ledger 4.33.
#   Same rule as dwLeanB.jl and dwB3W.jl: adversarial damage of a candidate = max over the belief points of the ball of radius t (with the box SC_BOXTB, seed hash((9292, SEED, round(1000 t))))
#   of CVaR_q̂[phi(d)] - CVaR_q̂[phi(q̂)], made monotone in t.  Stored candidates take the damage of radius 0.3 from dwx_<cfg>_B0.3_g_s.csv,
#   new candidates (column new = 1) get the damage of the radii 0.1, 0.2, 0.3 computed here with the same belief points.
#   Output geo/dw/dwx_<cfg>_B<t>_g_s.csv (x, new, V0, adv_prev, adv_raw, adv) and selx_<cfg>_B<t>_g_s.csv.  Test: B5_MAXC limits the candidates, D5_SUF is appended to the input candx and the output names.
const TNEW = parse(Float64, E("DT_BNEW", "0.5")); const BCFG = E("B3_CFG", "dwps3"); const GG = ENV["FB_G"]; const SUF = E("D5_SUF", ""); const B5MAX = parse(Int, E("B5_MAXC", "0")); const TLOW = [0.1, 0.2, 0.3]
t0 = time()
SITES = [collect(c) for k in 1:3 for c in Iterators.product(ntuple(_ -> 1:nd.nx, k)...) if issorted(collect(c)) && allunique(collect(c))]
ALLD = [(s, collect(lv)) for s in SITES for lv in Iterators.product(ntuple(_ -> TIERS, length(s))...)]; LAB = [dstr(s, lv) for (s, lv) in ALLD]; IDX = Dict(LAB[i] => i for i in eachindex(LAB))
tag0 = "$(BCFG)_g$(GG)_s$(SEED)"
C = [split(l, ",") for l in readlines(SPD * "geo/dw/candx_$(BCFG)_$(TNEW)_g$(GG)_s$(SEED)$(SUF).csv")[2:end]]; B5MAX > 0 && (C = vcat(C[1:min(B5MAX, length(C))], [c for c in C[min(B5MAX, length(C))+1:end] if c[2] == "1"]))
cand = [IDX[String(c[1])] for c in C]; isnew = [c[2] == "1" for c in C]; v0f = [parse(Float64, c[3]) for c in C]; nc = length(cand)
L3 = readlines(SPD * "geo/dw/dwx_$(BCFG)_B0.3_g$(GG)_s$(SEED).csv"); split(L3[1], ",")[5] == "adv0.3" || error("column 5 of dwx_$(BCFG)_B0.3 is not adv0.3")
A3 = Dict(String(split(l, ",")[1]) => parse(Float64, split(l, ",")[5]) for l in L3[2:end])
PT(t) = (BOXTB > 0 ? ball_pts_box : ball_pts)(t, MersenneTwister(hash((9292, SEED, round(Int, 1000t)))))
ptsN = PT(TNEW); ptsL = [PT(t) for t in TLOW]; V0 = zeros(nc); RAW = zeros(nc); ADVP = zeros(nc)
for (ci, i) in enumerate(cand)
    phi = evalx(xfrac(ALLD[i]...)); V0[ci] = cvar(phi(q), q); abs(V0[ci] - v0f[ci]) < 1e-3 || error("V0 of $(LAB[i]) differs from the stored value: $(V0[ci]) vs $(v0f[ci])")
    RAW[ci] = max(0.0, maximum(cvar(phi(d), q) for d in ptsN) - V0[ci])
    ADVP[ci] = isnew[ci] ? maximum(max(0.0, maximum(cvar(phi(d), q) for d in pts) - V0[ci]) for pts in ptsL) : A3[LAB[i]]
end
ADV = max.(ADVP, RAW)
open(SPD * "geo/dw/dwx_$(BCFG)_B$(TNEW)_g$(GG)_s$(SEED)$(SUF).csv", "w") do io
    println(io, "x,new,V0,adv0.3,adv_raw,adv$(TNEW)"); for (ci, i) in enumerate(cand); println(io, LAB[i], ",", isnew[ci] ? 1 : 0, ",", @sprintf("%.4f,%.4f,%.4f,%.4f", V0[ci], ADVP[ci], RAW[ci], ADV[ci])); end
end
open(SPD * "geo/dw/selx_$(BCFG)_B$(TNEW)_g$(GG)_s$(SEED)$(SUF).csv", "w") do io
    println(io, "scen,q0,c,model,x")
    for (k, (q0, c)) in enumerate(SCEN)
        cs = [q0 * length(ALLD[i][1]) + c * sum(ALLD[i][2]) for i in cand]; ci = sortperm(collect(zip(V0 .+ cs .+ ADV, LAB[cand])))[1]
        println(io, k, ",", q0, ",", c, ",B$(TNEW),", LAB[cand[ci]]); @printf("SELB scen %d: B%s=%s (adv %.2f, raw %.2f, new %s)\n", k, string(TNEW), LAB[cand[ci]], ADV[ci], RAW[ci], string(isnew[ci]))
    end
end
@printf("BOX data t = %.2f, belief t = %.2f\n", BOXT, BOXTB)
@printf("B5_DONE g%s seed %d: %d candidates (%d new), %d belief points, %.0f s\n", GG, SEED, nc, count(isnew), length(ptsN), time() - t0)
