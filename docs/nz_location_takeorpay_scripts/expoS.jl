# expoR.jl (2026-11-15) = expo.jl + the direct-path block at the end (true-F cost of N vs other decisions, belief = q-hat).
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
const BB = parse(Float64, E("SC_BB", "300")); const WQ = parse(Float64, E("SC_WQ", "300")); const NR = 20; const DS = [0.05, 0.1, 0.2, 0.3]; const EDS = [0.1, 0.2, 0.3]
const TIERS = parse.(Float64, split(E("SC_TIERS", "200,250,300"), ",")); const SCEN = E("SC_CV3", "0") == "1" ? [(0.0, 2.5), (150.0, 2.0), (450.0, 1.0)] : [(0.0, 2.5), (150.0, 2.0)]
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
# ---- expo (2026-11-15): 두 노출 지표 선별. 전체 열거로 N·D 를 구하고 (c/v 0.5, 0.4), N 결정에서 약정량·추가 주문 비중·자료 노출 (V_D − V_N)·belief 노출 (adv_δ) 을 잰다.
t0 = time()
SITES = [collect(c) for k in 1:3 for c in Iterators.product(ntuple(_ -> 1:nd.nx, k)...) if issorted(collect(c)) && allunique(collect(c))]
ALLD = [(s, collect(lv)) for s in SITES for lv in Iterators.product(ntuple(_ -> TIERS, length(s))...)]; LAB = [dstr(s, lv) for (s, lv) in ALLD]
V0 = zeros(length(ALLD)); VD = zeros(length(ALLD)); MS = zeros(length(ALLD)); VD1 = zeros(length(ALLD)); VD2 = zeros(length(ALLD))
for (i, (s, lv)) in enumerate(ALLD)
    Z = evalx(xfrac(s, lv))(q); V0[i] = cvar(Z, q); VD[i] = worst(Z, q, Inf, 0.3)[1]; VD1[i] = worst(Z, q, Inf, 0.1)[1]; VD2[i] = worst(Z, q, Inf, 0.2)[1]; MS[i] = dot(q, Z)
end
vv = nd.meta[:v]; BPTS = [ball_pts(t, MersenneTwister(hash((9292, SEED, round(Int, 1000t))))) for t in (0.1, 0.2)]
mkpath(SPD * "geo/expo"); io = open(SPD * "geo/expo/expo_$(TAG).csv", "w")
println(io, "cv,xN,capN,VN,hB,hint,sales_mean,sales_cvar,spot_mean,spot_cvar,expo_data,xD,capD,same,prem_D,adv0.1,adv0.2")
for (q0, c) in SCEN
    cst = [q0 * length(s) + c * sum(lv) for (s, lv) in ALLD]; n = sortperm(collect(zip(V0 .+ cst, LAB)))[1]; d = sortperm(collect(zip(VD .+ cst, LAB)))[1]
    sites, lv = ALLD[n]; x̄ = xfrac(sites, lv); h = _nz_resp!(_nz_build_resp(nd, x̄; optimizer=GRB1), nd, q, apess); hB = sum(h[nA+1:end])
    hint = count(i -> 0.5 < h[nA+i] < min(WQ, lv[findfirst(==(i), sites)]) - 0.5, sites)   # 약정량이 한도·용량에 안 묶인 거점 수
    phi = evalx(x̄); vn = cvar(phi(q), q); adv = [max(0.0, maximum(cvar(phi(dd), q) for dd in pts) - vn) for pts in BPTS]
    sc = abs(V0[n] + cst[n]) / 100
    @printf(io, "%.1f,%s,%.0f,%.2f,%.1f,%d,%.1f,%.1f,%.1f,%.1f,%.3f,%s,%.0f,%d,%.3f,%.3f,%.3f\n", c / vv, LAB[n], sum(lv), V0[n] + cst[n], hB, hint, -MS[n] / vv, -V0[n] / vv, -MS[n] / vv - hB, -V0[n] / vv - hB,
        (VD[n] - V0[n]) / sc, LAB[d], sum(ALLD[d][2]), LAB[d] == LAB[n], (V0[d] + cst[d] - V0[n] - cst[n]) / sc, adv[1] / sc, max(adv[1], adv[2]) / sc)
end
close(io)
# ---- expoR (2026-11-15): direct path. True-F cost (test sample NT) of the N decision vs other decisions, with the retailer's belief = the leader's q-hat.
const NT = parse(Int, E("DT_NT", "500")); const TSEED = 7770001; const KTOP = parse(Int, E("ER_KTOP", "30"))
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
const pT = fill(1 / NT, NT); JC = Dict{Int,Tuple{Float64,Float64,Float64}}()
function Jtrue(i)   # (CVaR cost, mean B sales, B reservation) under the true F, without store cost
    get!(JC, i) do
        x̄ = xfrac(ALLD[i]...); h = _nz_resp!(_nz_build_resp(nd, x̄; optimizer=GRB1), nd, q, apess); h === nothing && error("response failed")
        φ = phi_test(x̄, h); (cvar(φ, pT), -mean(φ) / vv, sum(h[nA+1:end]))
    end
end
io2 = open(SPD * "geo/expo/expoR_$(TAG).csv", "w")
println(io2, "cv,xN,capN,hB,band,salesT,spotT,optimism,regretN,xbest,capbest,gainD,ncand")
# ---- expoS (2026-11-15): absolute true-F costs of N, best, and Data-DR at eps 0.1, 0.2, 0.3 (for tails over training samples)
io3 = open(SPD * "geo/expo/expoS_$(TAG).csv", "w")
println(io3, "cv,VN,JN,Jbest,xN,xbest,xD1,JD1,xD2,JD2,xD3,JD3,capN,capD1,capD2,capD3")
for (q0, c) in SCEN
    cst = [q0 * length(s) + c * sum(lv) for (s, lv) in ALLD]; ord = sortperm(collect(zip(V0 .+ cst, LAB))); n = ord[1]; d = sortperm(collect(zip(VD .+ cst, LAB)))[1]
    cand = Set(ord[1:min(KTOP, length(ord))]); push!(cand, d); d1 = sortperm(collect(zip(VD1 .+ cst, LAB)))[1]; d2 = sortperm(collect(zip(VD2 .+ cst, LAB)))[1]; push!(cand, d1); push!(cand, d2)
    for tot in unique(sum(lv) for (s, lv) in ALLD); idx = [i for i in eachindex(ALLD) if sum(ALLD[i][2]) == tot]; push!(cand, idx[argmin(V0[idx] .+ cst[idx])]); end
    J = Dict(i => Jtrue(i)[1] + cst[i] for i in cand); b = argmin(J); sc = abs(V0[n] + cst[n]) / 100; _, sT, hB = Jtrue(n); capN = sum(ALLD[n][2])
    @printf(io2, "%.1f,%s,%.0f,%.1f,%.3f,%.1f,%.1f,%.3f,%.3f,%s,%.0f,%.3f,%d\n", c / vv, LAB[n], capN, hB, (capN - hB) / capN, sT, sT - hB, (J[n] - V0[n] - cst[n]) / sc, (J[n] - J[b]) / sc, LAB[b], sum(ALLD[b][2]), (J[d] - J[n]) / sc, length(cand))
    @printf(io3, "%.1f,%.4f,%.4f,%.4f,%s,%s,%s,%.4f,%s,%.4f,%s,%.4f,%.0f,%.0f,%.0f,%.0f
", c / vv, V0[n] + cst[n], J[n], J[b], LAB[n], LAB[b], LAB[d1], J[d1], LAB[d2], J[d2], LAB[d], J[d], capN, sum(ALLD[d1][2]), sum(ALLD[d2][2]), sum(ALLD[d][2]))
end
close(io2); close(io3)
@printf("EXPO_DONE %s %d 결정 %.0f s FR=%.2f WQ=%.0f 단계 %s β=%.2f\n", TAG, length(ALLD), time() - t0, FR, WQ, string(TIERS), β)
