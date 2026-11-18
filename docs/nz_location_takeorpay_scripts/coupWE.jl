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
# coupW.jl (2026-11-17) = dwLeanB.jl lines 1-117 + this body.  Coupled for the price-based model with the sets of ledger 4.22-4.24:
#   V_C(x) = max { CVaR_a[phi(x, h(x, d))] : a in U(CP_EH), d in U(CP_ET), TV(a, d) <= CP_DL },   U(r) = TV(q̂, r) ∩ box (SC_BOXT, width r/sqrt(S)).
#   For a fixed belief d the max over a is an LP (worst()).  The max over d is nonconvex: point search (lower bound):
#     q̂, the Data-DR worst a and its midpoint, "the adversary's a becomes the belief" iterations, CP_NPT random extreme points of U(ET) and of U(ET/2), pair moves for the best decisions.
#   Candidates = the candidate list of the decisions run CP_CFG (geo/dw/dw_<CP_CFG>_g_s.csv).  Output: geo/dw/cp_<CP_CFG>_g_s.csv and sel_<CP_CFG>c_g_s.csv (= sel_<CP_CFG> + rows of model C).
const EH = parse(Float64, E("CP_EH", "0.1")); const ET = parse(Float64, E("CP_ET", "0.1")); const DL = parse(Float64, E("CP_DL", string(ET / 3)))   # coupling radius: a third of the belief radius unless CP_DL is given
const CLAB = ENV["CP_LAB"]   # coupWE.jl (2026-11-18): model label and output file names from CP_LAB, the files of the 0.1 run are not touched
const NPT = parse(Int, E("CP_NPT", "30")); const NLS = parse(Int, E("CP_NLS", "8")); const CCFG = E("CP_CFG", "dwps3"); const GG = ENV["FB_G"]
t0 = time()
SITES = [collect(c) for k in 1:3 for c in Iterators.product(ntuple(_ -> 1:nd.nx, k)...) if issorted(collect(c)) && allunique(collect(c))]
ALLD = [(s, collect(lv)) for s in SITES for lv in Iterators.product(ntuple(_ -> TIERS, length(s))...)]; LAB = [dstr(s, lv) for (s, lv) in ALLD]; IDX = Dict(LAB[i] => i for i in eachindex(LAB))
costv(q0, c) = [q0 * length(s) + c * sum(lv) for (s, lv) in ALLD]; CST = [costv(q0, c) for (q0, c) in SCEN]; CAP = [sum(lv) for (s, lv) in ALLD]
tag0 = "$(CCFG)_g$(GG)_s$(SEED)"
cand = [IDX[String(split(l, ",")[1])] for l in readlines(SPD * "geo/dw/dw_$(tag0).csv")[2:end]]; nc = length(cand)
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
PTS = vcat(set_pts(ET, MersenneTwister(hash((9898, SEED, 1))), NPT), set_pts(ET / 2, MersenneTwister(hash((9898, SEED, 2))), NPT))
@printf("COUPLED g%s seed %d: VSM %.4f, box t = %.4f, eps_hat %.2f, eps_tilde %.2f, delta %.2f, %d candidates, %d belief points\n", GG, SEED, VSM, BOXT, EH, ET, DL, nc, length(PTS)); flush(stdout)
V0 = zeros(nc); VD = zeros(nc); VC = zeros(nc); DB = Vector{Vector{Float64}}(undef, nc); PH = Vector{Any}(undef, nc)
val(phi, d) = worst(phi(d), d, DL, EH)   # (value, a): the worst demand distribution given the belief d
for (ci, i) in enumerate(cand)
    phi = evalx(xfrac(ALLD[i]...)); PH[ci] = phi; Zq = phi(q); V0[ci] = cvar(Zq, q)
    vD, aD = worst(Zq, q, Inf, EH); VD[ci] = vD
    best, _ = val(phi, q); bd = copy(q)
    for d in (aD, 0.5 .* (aD .+ q)); inset(d, ET) || continue; v, _ = val(phi, d); v > best && (best = v; bd = d); end
    d = bd
    for _ in 1:3   # the adversary's distribution becomes the retailer's belief
        _, a = val(phi, d); (a === nothing || !inset(a, ET)) && break
        v, _ = val(phi, a); v > best && (best = v; bd = a); maximum(abs.(a .- d)) < 1e-9 && break; d = a
    end
    for d in PTS; v, _ = val(phi, d); v > best && (best = v; bd = d); end
    VC[ci] = best; DB[ci] = bd
end
@printf("  point search done %.0f s\n", time() - t0); flush(stdout)
function improve!(ci)
    Γ = boxw(ET, BOXT); phi = PH[ci]; imp = true; rounds = 0
    while imp && rounds < 2
        imp = false; rounds += 1; base = DB[ci]
        for i1 in 1:S, j1 in 1:S
            i1 == j1 && continue
            for mm in (Γ[i1], Γ[i1] / 2)
                d = copy(base); m = min(mm, d[i1]); m <= 1e-9 && continue; d[i1] -= m; d[j1] += m
                inset(d, ET) || continue
                v, _ = val(phi, d); v > VC[ci] + 1e-7 && (VC[ci] = v; DB[ci] = d; imp = true)
            end
        end
    end
end
for (k, _) in enumerate(SCEN); for ci in sortperm(VC .+ CST[k][cand])[1:min(NLS, nc)]; improve!(ci); end; end
@printf("  pair moves done %.0f s\n", time() - t0); flush(stdout)
open(SPD * "geo/dw/cpx_$(CCFG)_$(CLAB)_g$(GG)_s$(SEED).csv", "w") do io
    println(io, "x,n,cap,V0,VD,VC,tv_d_q")
    for (ci, i) in enumerate(cand); println(io, LAB[i], ",", length(ALLD[i][1]), ",", @sprintf("%.0f", CAP[i]), ",", @sprintf("%.4f,%.4f,%.4f,%.4f", V0[ci], VD[ci], VC[ci], 0.5 * sum(abs.(DB[ci] .- q)))); end
end
open(SPD * "geo/dw/selx_$(CCFG)_$(CLAB)_g$(GG)_s$(SEED).csv", "w") do io
    println(io, "scen,q0,c,model,x")
    for (k, (q0, c)) in enumerate(SCEN)
        ci = sortperm(collect(zip(VC .+ CST[k][cand], LAB[cand])))[1]; println(io, k, ",", q0, ",", c, ",", CLAB, ",", LAB[cand[ci]])
        @printf("SELC scen %d: %s=%s cap %.0f (V0 %.2f, VC %.2f)
", k, CLAB, LAB[cand[ci]], CAP[cand[ci]], V0[ci], VC[ci])
    end
end
@printf("CP_DONE g%s seed %d: %.0f s\n", GG, SEED, time() - t0)
