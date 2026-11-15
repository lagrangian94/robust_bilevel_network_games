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
const BB = parse(Float64, E("SC_BB", "300")); const WQ = 300.0; const NR = 20; const DS = [0.05, 0.1, 0.2, 0.3]; const EDS = [0.1, 0.2, 0.3]
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
# ---- 1. 전체 열거: V_N, V_D (ε̂ = 0.1·0.2·0.3), 개설비 없이
t0 = time()
SITES = [collect(c) for k in 1:3 for c in Iterators.product(ntuple(_ -> 1:nd.nx, k)...) if issorted(collect(c)) && allunique(collect(c))]
ALLD = [(s, collect(lv)) for s in SITES for lv in Iterators.product(ntuple(_ -> TIERS, length(s))...)]; LAB = [dstr(s, lv) for (s, lv) in ALLD]
V0 = zeros(length(ALLD)); VD0 = zeros(length(ALLD), length(EDS))
for (i, (s, lv)) in enumerate(ALLD)
    Z = evalx(xfrac(s, lv))(q); V0[i] = cvar(Z, q)
    for (e, eps) in enumerate(EDS); VD0[i, e] = worst(Z, q, Inf, eps)[1]; end
end
mkpath(SPD * "geo/dw"); open(SPD * "geo/dw/vn_$(TAG).csv", "w") do io
    println(io, "x,n,cap,V0," * join(["VD$(e)" for e in EDS], ","))
    for (i, (s, lv)) in enumerate(ALLD); println(io, LAB[i], ",", length(s), ",", @sprintf("%.0f", sum(lv)), ",", @sprintf("%.4f", V0[i]), ",", join([@sprintf("%.4f", VD0[i, e]) for e in 1:length(EDS)], ",")); end
end
@printf("ENUM %d 결정 %.0f s, 단계 %s, bBmax %.0f\n", length(ALLD), time() - t0, string(TIERS), BBMAX); flush(stdout)
# ---- 2. 후보
costv(q0, c) = [q0 * length(s) + c * sum(lv) for (s, lv) in ALLD]
ordby(v) = sortperm(collect(zip(v, LAB)))
sel = Int[]
for (k, (q0, c)) in enumerate(SCEN)
    cst = costv(q0, c); append!(sel, ordby(V0 .+ cst)[1:NCN[k]])
    for e in 1:length(EDS); append!(sel, ordby(VD0[:, e] .+ cst)[1:NCD[k]]); end
end
for tot in sort(unique(sum(lv) for (s, lv) in ALLD))
    tot < STRMIN && continue
    idx = [i for i in eachindex(ALLD) if sum(ALLD[i][2]) == tot]; idx = idx[ordby(V0[idx])]; seen = Set{Vector{Int}}()
    for i in idx
        ALLD[i][1] in seen && continue
        push!(seen, ALLD[i][1]); push!(sel, i); length(seen) >= STRK && break
    end
end
sel = unique(sel); MAXC > 0 && (sel = sel[1:min(MAXC, length(sel))]); nc = length(sel)
# ---- 3. 후보마다 적대 피해 (Belief-only) 와 결합 값
BPTS = [ball_pts(t, MersenneTwister(hash((9292, SEED, round(Int, 1000t))))) for t in DS]; PTS = belief_pts(MersenneTwister(hash((9393, SEED))))
ADV = zeros(nc, length(DS)); VCH0 = zeros(nc)
for (ci, i) in enumerate(sel)
    phi = evalx(xfrac(ALLD[i]...)); vn = cvar(phi(q), q)
    ADV[ci, :] = accumulate(max, [max(0.0, maximum(cvar(phi(d), q) for d in pts) - vn) for pts in BPTS])
    Zq = phi(q); vdr, aD = worst(Zq, q, Inf, EPS); best = -Inf; bd = q; ba = aD
    for d in vcat(PTS, [q .+ λ .* (aD .- q) for λ in (1 / 3, 2 / 3, 1.0)])
        v, a = worst(phi(d), d, DEL, EPS); v > best && (best = v; bd = d; ba = a)
    end
    for _ in 1:RND
        imp = false; nb = Vector{Vector{Float64}}()
        for base in (ba, bd), mv in (DEL, DEL / 2), i1 in 1:S, j1 in 1:S
            i1 == j1 && continue
            mm = min(mv, base[i1] - max(0.0, q[i1] - ΓC), q[j1] + ΓC - base[j1]); mm <= 1e-9 && continue
            d = copy(base); d[i1] -= mm; d[j1] += mm
            0.5 * sum(abs.(d .- q)) <= EPS + 1e-9 && push!(nb, d)
        end
        for d in unique(x -> round.(x; digits=8), nb)
            v, a = worst(phi(d), d, DEL, EPS); v > best + 1e-6 && (best = v; bd = d; ba = a; imp = true)
        end
        imp || break
    end
    VCH0[ci] = best
end
open(SPD * "geo/dw/dw_$(TAG).csv", "w") do io
    println(io, "x,n,cap,V0," * join(["VD$(e)" for e in EDS], ",") * ",VCh," * join(["adv$(t)" for t in DS], ","))
    for (ci, i) in enumerate(sel)
        println(io, LAB[i], ",", length(ALLD[i][1]), ",", @sprintf("%.0f", sum(ALLD[i][2])), ",", @sprintf("%.4f", V0[i]), ",", join([@sprintf("%.4f", VD0[i, e]) for e in 1:length(EDS)], ","), ",", @sprintf("%.4f", VCH0[ci]), ",", join([@sprintf("%.4f", ADV[ci, k]) for k in 1:length(DS)], ","))
    end
end
@printf("CAND %d 후보 %.0f s\n", nc, time() - t0); flush(stdout)
# ---- 4. 비용 시나리오·모형별 결정
SELM = Tuple{Int,String,Int}[]   # (비용 시나리오, 모형, ALLD 색인)
for (k, (q0, c)) in enumerate(SCEN)
    cst = costv(q0, c); cs = cst[sel]
    push!(SELM, (k, "N", ordby(V0 .+ cst)[1]))
    for (e, eps) in enumerate(EDS); push!(SELM, (k, "D$(eps)", ordby(VD0[:, e] .+ cst)[1])); end
    for (j, t) in enumerate(DS); push!(SELM, (k, "B$(t)", sel[sortperm(collect(zip(V0[sel] .+ cs .+ ADV[:, j], LAB[sel])))[1]])); end
    push!(SELM, (k, "C", sel[sortperm(collect(zip(VCH0 .+ cs, LAB[sel])))[1]]))
end
MAXC == 0 && @assert all(i in sel for (_, _, i) in SELM) "선택된 결정이 후보 밖"
open(SPD * "geo/dw/sel_$(TAG).csv", "w") do io
    println(io, "scen,q0,c,model,x"); for (k, m, i) in SELM; println(io, k, ",", SCEN[k][1], ",", SCEN[k][2], ",", m, ",", LAB[i]); end
end
for k in 1:length(SCEN); println("SEL scen $k: ", join(["$m=$(LAB[i])" for (kk, m, i) in SELM if kk == k], " ")); end
println("DW_DONE"); flush(stdout)
# ---- 5. 진짜 F 평가 (otS.jl 과 같은 방식): 시험 표본 NT 개의 pessimistic φ 의 CVaR, 소매상 belief = F 에서 새로 뽑은 nf 개의 경험분포
ndT = mk(NT, TSEED)
function phi_test(x̄, h)
    m, ny = nz_m(ndT), nz_ny(ndT); mdl = Model(GRB1)
    @variable(mdl, y[1:ny] >= 0)
    cons = [@constraint(mdl, sum(ndT.A[k, j] * y[j] for j in 1:ny if ndT.A[k, j] != 0) <= 0.0) for k in 1:m]
    cq = @constraint(mdl, dot(ndT.c, y) >= -1e30); φ = zeros(NT)
    for s in 1:NT
        rr = nz_rhs(ndT, x̄, h, s); for k in 1:m; set_normalized_rhs(cons[k], rr[k]); end
        set_normalized_rhs(cq, -1e30); @objective(mdl, Max, dot(ndT.c, y)); optimize!(mdl)
        termination_status(mdl) == MOI.OPTIMAL || error("recourse LP s=$s: $(termination_status(mdl))")
        Q = objective_value(mdl); set_normalized_rhs(cq, Q - 1e-9 * max(1.0, abs(Q)))
        @objective(mdl, Max, dot(ndT.ell, y)); optimize!(mdl)
        termination_status(mdl) == MOI.OPTIMAL || error("pessimistic LP s=$s: $(termination_status(mdl))")
        φ[s] = objective_value(mdl)
    end
    return φ
end
const pT = fill(1 / NT, NT); Jc = Dict{Tuple{Int,Vector{Float64}},Float64}()
J0(i, h) = get!(() -> cvar(phi_test(xfrac(ALLD[i]...), h), pT), Jc, (i, h))
ux = unique([i for (_, _, i) in SELM])
mkpath(SPD * "geo/oos/true"); io = open(SPD * "geo/oos/true/ot_$(TAG).csv", "w"); println(io, "nf,k,x,J0")
# belief 를 아는 세계 (nf = 0): 소매상 belief = 리더의 q̂ (같은 표본), 비용은 진짜 F 시험 표본
for i in ux
    x̄ = xfrac(ALLD[i]...); h = _nz_resp!(_nz_build_resp(nd, x̄; optimizer=GRB1), nd, q, apess); h === nothing && error("follower LP 실패 (공통 belief $(LAB[i]))")
    @printf(io, "%d,%d,%s,%.6f
", 0, 0, LAB[i], J0(i, round.(h; digits=4)))
end
for nf in NFS, k in 1:K
    ndF = mk(nf, 5_000_000 + 1000 * SEED + k + (nf == 5 ? 0 : 10_000_000 * nf))
    for i in ux
        x̄ = xfrac(ALLD[i]...); h = _nz_resp!(_nz_build_resp(ndF, x̄; optimizer=GRB1), ndF, fill(1 / nf, nf), apess)
        h === nothing && error("follower LP 실패 (nf=$nf k=$k $(LAB[i]))")
        @printf(io, "%d,%d,%s,%.6f\n", nf, k, LAB[i], J0(i, round.(h; digits=4)))
    end
    flush(io)
end
close(io); @printf("OT_DONE seed=%d 결정 %d 개, 평가 %d, %.0f s\n", SEED, length(ux), length(Jc), time() - t0)
