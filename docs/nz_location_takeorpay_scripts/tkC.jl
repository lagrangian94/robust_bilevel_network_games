# tkS (2026-11-13): tkA.jl 의 평가 (V_N 상위 8, 적대 피해 4 반지름, 표본 belief 피해 N_F 별) + scr.jl 의 손잡이 (SC_BETA·SC_FR·SC_VS·SC_VOL·SC_WA·SC_WB).
#   출력은 tk.jl 과 같은 형식 geo/tk/tk_<TAG>.csv (tk_agg2.py·tk_decomp.py 로 집계). 저장소 코드는 include 만 한다.
root = "C:/Users/user/robust_bilevel_network_games"
const SPD = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Distributions, Statistics
const GRB_ENV = Gurobi.Env()
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(SPD * "geo/nz_data_geo2.jl")
include(SPD * "box/nz_exact_eval_box.jl"); include(joinpath(root, "nonzero_sum", "nz_resp_heuristic.jl"))
include(SPD * "geo/fac_builder_geo2.jl")
E(k, d) = get(ENV, k, d)
SEED = parse(Int, ENV["C0H_SEED"]); TAG = ENV["C0H_TAG"]; β = parse(Float64, E("SC_BETA", "0.4")); P = parse(Float64, E("SC_P", "800")); FR = parse(Float64, E("SC_FR", "0.25"))
WA = parse(Float64, E("SC_WA", "300")); WB = parse(Float64, E("SC_WB", "300")); NR = 20
NFS = parse.(Int, split(E("TK_NF", "3,5,10,30"), ",")); DS = parse.(Float64, split(E("TK_D", "0.05,0.1,0.2,0.3"), ",")); B = parse(Int, E("TK_B", "150"))
nd0 = make_sgb15_fac(; S=10, seed=SEED, dem=:region, p=P, f=FR * P, eps_hat=0.0, eps_tilde=0.0, beta=β)
nA = nd0.meta[:nA]; nst = nd0.meta[:nst]; ncu = nd0.meta[:ncu]
nd = make_sgb15_fac(; S=10, seed=SEED, dem=:region, p=P, f=FR * P, eps_hat=0.0, eps_tilde=0.0, beta=β, quota=vcat(fill(WA, nA), fill(WB, nst - nA)))
V = nd.meta[:v]; VS = parse(Float64, E("SC_VS", string(V)))
if VS != V   # B 의 spot 단위 마진만 바꾼다 (리더 손실 계수 ℓ)
    iyS = nd.meta[:iyS]; for i in nA+1:nst, j in 1:ncu; nd.ell[iyS(i, j)] = -VS; end
end
if haskey(ENV, "SC_VOL")
    a0 = parse.(Float64, split(ENV["SC_VOL"], ",")); a = a0[randperm(MersenneTwister(hash((5151, SEED))), 3)]
    custs = nd.meta[:cities][nst+1:nst+ncu]; reg = [REGION[c] for c in custs]; μ = sgb15_mu(sgb15_kuhn().pop)
    rng = MersenneTwister(hash((4242, SEED))); Z = randn(rng, 3, nd.S); EE = randn(rng, ncu, nd.S); ρ = 0.9
    ξ = [round(μ[j] * (1 + a[reg[j]] * (cdf(Normal(), ρ * Z[reg[j], s] + sqrt(1 - ρ^2) * EE[j, s]) - 0.5)); digits=1) for j in 1:ncu, s in 1:nd.S]
    rd = nd.meta[:rdem]; for s in 1:nd.S, j in 1:ncu; nd.u[rd(j), s] = -ξ[j, s]; end
    nd.meta[:xi] = ξ
    @assert nA * nd.meta[:bA] > maximum(sum(ξ; dims=1)) "A 총용량 > 최대 총수요 필요"
end
# 수정안 2 (2026-11-13): 비예약 물량의 공급 한도. 거점 i 에 Σ_j y^S_ij ≤ σ b_i (t_i + x_i). SC_SA·SC_SB = σ (음수면 한도 없음).
#   A 에도 한도를 두면 (SC_SA ≥ 0) 수요를 못 채울 수 있어서 고객별 외부 조달 열 (단가 cmax + 2p, 리더 손실 0) 을 넣는다.
const SA = parse(Float64, E("SC_SA", "-1")); const SB = parse(Float64, E("SC_SB", "-1"))
function with_spotcap(nd, σA, σB)
    m, ny = nz_m(nd), nz_ny(nd); S = nd.S; iyS = nd.meta[:iyS]; bA = nd.meta[:bA]; bB = nd.meta[:bB]; rd = nd.meta[:rdem]
    rows = [i for i in 1:nst if (i <= nA ? σA : σB) >= 0]; nr = length(rows); no = σA >= 0 ? ncu : 0; pout = nd.meta[:cmax] + 2 * nd.meta[:p]
    A = zeros(m + nr, ny + no); A[1:m, 1:ny] = nd.A
    u = vcat(nd.u, zeros(nr, S)); g = vcat(nd.g, zeros(nr, S)); hrow = vcat(nd.hrow, zeros(Int, nr)); hcoef = vcat(nd.hcoef, zeros(nr)); xrow = vcat(nd.xrow, zeros(Int, nr))
    for (k, i) in enumerate(rows)
        for j in 1:ncu; A[m+k, iyS(i, j)] = 1.0; end
        if i <= nA; u[m+k, :] .= σA * bA; else xrow[m+k] = i - nA; g[m+k, :] .= σB * bB; end
    end
    for j in 1:no; A[rd(j), ny+j] = -1.0; end
    return NZData(nd.name * "_spotcap", A, vcat(nd.c, fill(-pout, no)), vcat(nd.ell, zeros(no)), u, hrow, hcoef, xrow, g, nd.W, nd.wvec, nd.c0, nd.hU,
        nd.nx, nd.x_allowed, nd.gamma, nd.qx, nd.S, nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta,
        nd.thetaU, vcat(nd.piLU, zeros(nr)), vcat(nd.piFU, zeros(nr)), nd.lambdaU, vcat(nd.varpiU, fill(Inf, nr)), nd.meta)
end
(SA >= 0 || SB >= 0) && (global nd = with_spotcap(nd, SA, SB); @printf("SPOTCAP σA=%.2f σB=%.2f bA=%.0f bB=%.0f 행 %d 열 %d\n", SA, SB, nd.meta[:bA], nd.meta[:bB], nz_m(nd), nz_ny(nd)))
# 약정 물량 방식 (2026-11-14, SC_TOP=1): B 거점의 예약 제약 Σ_j y^R_ij ≤ h_i 의 여유 s_i 를 명시 열로 두고 리더 계수 −v 를 준다.
#   B 의 판매량이 (실제로 쓴 예약분) 이 아니라 (약정한 예약분 h_i) 로 잡힌다: φ = −v Σ_{i∈I_B} (h_i + Σ_j y^S_ij). 소매상 비용은 그대로 (여유 열 비용 0).
#   s_i 를 여유로 고정하려고 반대쪽 행 −Σ_j y^R_ij − s_i ≤ −h_i 를 더한다 (등식). 보상 항으로 고정하면 사전식 2 단계의 허용오차 안에서 s_i 가 줄 수 있어서 쓰지 않는다.
const TOP = parse(Int, E("SC_TOP", "0"))
function with_takeorpay(nd)
    m, ny = nz_m(nd), nz_ny(nd); S = nd.S; rres = nd.meta[:rres]; iyR = nd.meta[:iyR]; hidx = nd.meta[:hidx]; nB = nst - nA
    A = zeros(m + nB, ny + nB); A[1:m, 1:ny] = nd.A
    hrow = vcat(nd.hrow, zeros(Int, nB)); hcoef = vcat(nd.hcoef, zeros(nB))
    for (k, i) in enumerate(nA+1:nst)
        A[rres(i, 1), ny+k] = 1.0
        for j in 1:ncu; A[m+k, iyR(i, j)] = -1.0; end
        A[m+k, ny+k] = -1.0; hrow[m+k] = hidx(i, 1); hcoef[m+k] = -1.0
    end
    return NZData(nd.name * "_top", A, vcat(nd.c, zeros(nB)), vcat(nd.ell, fill(-nd.meta[:v], nB)), vcat(nd.u, zeros(nB, S)), hrow, hcoef, vcat(nd.xrow, zeros(Int, nB)), vcat(nd.g, zeros(nB, S)), nd.W, nd.wvec, nd.c0, nd.hU,
        nd.nx, nd.x_allowed, nd.gamma, nd.qx, nd.S, nd.q_hat, nd.eps_hat, nd.eps_tilde, nd.beta,
        nd.thetaU, vcat(nd.piLU, zeros(nB)), vcat(nd.piFU, zeros(nB)), nd.lambdaU, vcat(nd.varpiU, fill(Inf, nB)), nd.meta)
end
TOP == 1 && (global nd = with_takeorpay(nd); println("TAKEORPAY 열 ", nz_ny(nd)))
S = nd.S; q = nd.q_hat; nh = nz_nh(nd); apess = vcat(nd.hU[1:nA], zeros(nh - nA)); const RF = parse(Float64, E("SC_RF", "0"))
nh == nst || error("h 가 거점별이 아님: nh=$nh nst=$nst")
xb(x) = (v = zeros(nd.nx); v[x] .= 1.0; v); xstr(x) = "{" * join(x, ";") * "}"
function cvar(Z, p)
    ord = sortperm(Z; rev=true); cap = p ./ (1 - β); rest = 1.0; val = 0.0
    for s in ord; m = min(cap[s], rest); val += m * Z[s]; rest -= m; rest <= 1e-12 && break; end
    return val
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
function valfn(x)
    x̄ = xb(x); R = _nz_build_resp(nd, x̄; optimizer=GRB1); qx = dot(nd.qx, x̄); cache = Dict{Vector{Float64},Float64}(); dc = Dict{Vector{Float64},Float64}()
    return d -> get!(dc, round.(d; digits=8)) do
        h = _nz_resp!(R, nd, d, apess); h === nothing && error("응답 실패 $(xstr(x))")
        get!(() -> qx - RF * sum(h[nA+1:end]) + cvar(nz_recourse_values(nd, x̄, h)[2], q), cache, round.(h; digits=6))   # RF > 0: B 가 예약 단위마다 받는 예약료 마진 (시나리오와 무관)
    end
end
t0 = time()
ALLX = [collect(c) for k in 1:3 for c in Iterators.product(ntuple(_ -> 1:nd.nx, k)...) if issorted(collect(c)) && allunique(collect(c))]
VNALL = [valfn(x)(q) for x in ALLX]; XS = ALLX[sortperm(collect(zip(VNALL, xstr.(ALLX))))[1:8]]
mkpath(SPD * "geo/tk"); open(SPD * "geo/tk/vn_$(TAG).csv", "w") do io; println(io, "x,VN"); for (x, v) in zip(ALLX, VNALL); println(io, xstr(x), ",", @sprintf("%.4f", v)); end; end
@printf("ENUM %d 결정 %.0f s, β=%.2f FR=%.2f VS=%.1f\n", length(ALLX), time() - t0, β, FR, VS)
PTS = [ball_pts(t, MersenneTwister(hash((9292, SEED, round(Int, 1000t))))) for t in DS]
BOOTS = [(rb = MersenneTwister(hash((8484, SEED, nf))); [(c = zeros(S); for _ in 1:nf; c[rand(rb, 1:S)] += 1; end; c ./ nf) for _ in 1:B]) for nf in NFS]
top10(Δ) = (o = sort(Δ; rev=true); mean(o[1:max(1, round(Int, 0.1 * length(o)))]))
open(SPD * "geo/tk/tk_$(TAG).csv", "w") do io
    println(io, "x,VN," * join(["adv$(t)" for t in DS], ",") * "," * join(["bm$(n),bq$(n),bt$(n),bx$(n)" for n in NFS], ","))
    for x in XS
        v = valfn(x); vn = v(q); cols = Float64[vn]
        for pts in PTS; push!(cols, max(0.0, maximum(v(d) for d in pts) - vn)); end
        for bs in BOOTS; Δ = [v(d) - vn for d in bs]; append!(cols, [mean(Δ), quantile(Δ, 0.9), top10(Δ), maximum(Δ)]); end
        println(io, xstr(x) * "," * join([@sprintf("%.4f", c) for c in cols], ","))
    end
end
@printf("TK seed=%d cands=%d time=%.0f\n", SEED, length(XS), time() - t0)
println("TK_DONE")
