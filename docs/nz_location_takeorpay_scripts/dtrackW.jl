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
# dtrackW.jl (2026-11-17) = dwLeanB.jl lines 1-117 + this body.  "Data-DR with the belief following the truth" against Data-DR, on the same set U(r) = TV(q̂, r) ∩ box (SC_BOXT).
#   V_D(x) = max_{a in U(r)} CVaR_a[phi(x, h(x, q̂))]   (Data-DR: the retailer keeps believing q̂; exact LP)
#   V_T(x) = max_{a in U(r)} CVaR_a[phi(x, h(x, a))]    (D-track: the retailer believes the distribution the adversary picks; nonconvex, point search = lower bound)
#   candidates: top TR_KC decisions by V0 + cost and by VD(0.1, 0.2, 0.3) + cost for each cost table, from the stored enumeration vn_<TR_VN>_g_s.csv
const RT = parse.(Float64, split(E("TR_R", "0.1,0.2"), ",")); const KC = parse(Int, E("TR_KC", "40")); const NPT = parse(Int, E("TR_NPT", "40")); const NLS = parse(Int, E("TR_NLS", "20"))
const OUTT = E("TR_OUT", "dtr"); const VNCFG = E("TR_VN", "dwpl75"); const GG = ENV["FB_G"]
t0 = time()
SITES = [collect(c) for k in 1:3 for c in Iterators.product(ntuple(_ -> 1:nd.nx, k)...) if issorted(collect(c)) && allunique(collect(c))]
ALLD = [(s, collect(lv)) for s in SITES for lv in Iterators.product(ntuple(_ -> TIERS, length(s))...)]; LAB = [dstr(s, lv) for (s, lv) in ALLD]; nALL = length(ALLD)
costv(q0, c) = [q0 * length(s) + c * sum(lv) for (s, lv) in ALLD]; CST = [costv(q0, c) for (q0, c) in SCEN]; CAP = [sum(lv) for (s, lv) in ALLD]
L = readlines(SPD * "geo/dw/vn_$(VNCFG)_g$(GG)_s$(SEED).csv")[2:end]; length(L) == nALL || error("vn rows $(length(L)) != $nALL")
VS = zeros(nALL, 4); for (i, l) in enumerate(L); f = split(l, ","); f[1] == LAB[i] || error("vn label mismatch row $i"); VS[i, :] = parse.(Float64, f[4:7]); end
ordby(v) = sortperm(collect(zip(v, LAB)))
cand = sort(unique(vcat([ordby(VS[:, c] .+ CST[k])[1:KC] for k in 1:length(SCEN), c in 1:4]...))); nc = length(cand)
"random extreme points of U(r): mass r taken from some scenarios (at most the box width each) and given to others"
function set_pts(r, rng, n)
    Γ = boxw(r, BOXT); pts = Vector{Vector{Float64}}()
    for _ in 1:n
        p = randperm(rng, S); d = copy(q); rest = r; k = 1
        while rest > 1e-12 && k <= S; s = p[k]; mm = min(rest, Γ[s], d[s]); d[s] -= mm; rest -= mm; k += 1; end
        kd = k - 1; rest = r; k = S
        while rest > 1e-12 && k > kd; s = p[k]; mm = min(rest, Γ[s]); d[s] += mm; rest -= mm; k -= 1; end
        rest <= 1e-9 && push!(pts, d)
    end
    return unique(x -> round.(x; digits=8), pts)
end
inset(d, r) = (Γ = boxw(r, BOXT); minimum(d) >= -1e-12 && all(abs.(d .- q) .<= Γ .+ 1e-9) && 0.5 * sum(abs.(d .- q)) <= r + 1e-9)
PTS = [set_pts(r, MersenneTwister(hash((9797, SEED, round(Int, 1000r)))), NPT) for r in RT]
@printf("DTRACK g%s seed %d: VSM %.4f, box t = %.4f, radii %s, %d candidates of %d, points %s\n", GG, SEED, VSM, BOXT, string(RT), nc, nALL, string(length.(PTS))); flush(stdout)
V0 = zeros(nc); VD = zeros(nc, length(RT)); VT = zeros(nc, length(RT)); AT = Matrix{Vector{Float64}}(undef, nc, length(RT)); PH = Vector{Any}(undef, nc)
for (ci, i) in enumerate(cand)
    phi = evalx(xfrac(ALLD[i]...)); PH[ci] = phi; Zq = phi(q); V0[ci] = cvar(Zq, q)
    for (ir, r) in enumerate(RT)
        vd, aD = worst(Zq, q, Inf, r); VD[ci, ir] = vd
        best = cvar(phi(aD), aD); ba = aD; a = aD
        for _ in 1:3   # the adversary answers the commitment of a retailer who believes a
            _, a2 = worst(phi(a), q, Inf, r); v2 = cvar(phi(a2), a2); v2 > best && (best = v2; ba = a2); maximum(abs.(a2 .- a)) < 1e-9 && break; a = a2
        end
        for d in PTS[ir]; v = cvar(phi(d), d); v > best && (best = v; ba = d); end
        VT[ci, ir] = best; AT[ci, ir] = ba
    end
end
@printf("  point search done %.0f s\n", time() - t0); flush(stdout)
# ---- local moves (one pair of scenarios at a time) for the decisions that can become the D-track optimum
function improve!(ci, ir)
    r = RT[ir]; Γ = boxw(r, BOXT); phi = PH[ci]; imp = true; rounds = 0
    while imp && rounds < 2
        imp = false; rounds += 1; base = AT[ci, ir]
        for i1 in 1:S, j1 in 1:S
            i1 == j1 && continue
            for mm in (Γ[i1], Γ[i1] / 2)
                d = copy(base); m = min(mm, d[i1]); m <= 1e-9 && continue; d[i1] -= m; d[j1] += m
                inset(d, r) || continue
                v = cvar(phi(d), d); v > VT[ci, ir] + 1e-7 && (VT[ci, ir] = v; AT[ci, ir] = d; imp = true)
            end
        end
    end
end
for (k, _) in enumerate(SCEN), ir in 1:length(RT)
    for ci in sortperm(VT[:, ir] .+ CST[k][cand])[1:min(NLS, nc)]; improve!(ci, ir); end
end
@printf("  local moves done %.0f s\n", time() - t0); flush(stdout)
mkpath(SPD * "geo/$(OUTT)")
open(SPD * "geo/$(OUTT)/dtr_$(TAG).csv", "w") do io
    println(io, "x,n,cap,V0," * join(["VD$(r)" for r in RT], ",") * "," * join(["VT$(r)" for r in RT], ","))
    for (ci, i) in enumerate(cand); println(io, LAB[i], ",", length(ALLD[i][1]), ",", @sprintf("%.0f", CAP[i]), ",", @sprintf("%.4f", V0[ci]), ",", join([@sprintf("%.4f", v) for v in VD[ci, :]], ","), ",", join([@sprintf("%.4f", v) for v in VT[ci, :]], ",")); end
end
open(SPD * "geo/$(OUTT)/sel_$(TAG).csv", "w") do io
    println(io, "scen,q0,c,model,x,cap,V0,VD,VT")
    for (k, (q0, c)) in enumerate(SCEN)
        cs = CST[k][cand]; pick(v) = sortperm(collect(zip(v .+ cs, LAB[cand])))[1]
        iN = pick(V0); println(io, join([string(k), string(q0), string(c), "N", LAB[cand[iN]], @sprintf("%.0f", CAP[cand[iN]]), @sprintf("%.4f", V0[iN]), "", ""], ","))
        for (ir, r) in enumerate(RT)
            iD = pick(VD[:, ir]); iT = pick(VT[:, ir])
            println(io, join([string(k), string(q0), string(c), "D$(r)", LAB[cand[iD]], @sprintf("%.0f", CAP[cand[iD]]), @sprintf("%.4f", V0[iD]), @sprintf("%.4f", VD[iD, ir]), @sprintf("%.4f", VT[iD, ir])], ","))
            println(io, join([string(k), string(q0), string(c), "T$(r)", LAB[cand[iT]], @sprintf("%.0f", CAP[cand[iT]]), @sprintf("%.4f", V0[iT]), @sprintf("%.4f", VD[iT, ir]), @sprintf("%.4f", VT[iT, ir])], ","))
        end
    end
end
@printf("DTR_DONE g%s seed %d: %.0f s\n", GG, SEED, time() - t0)
