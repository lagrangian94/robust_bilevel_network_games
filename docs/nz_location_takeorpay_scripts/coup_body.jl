# coupW.jl (2026-11-17) = dwLeanB.jl lines 1-117 + this body.  Coupled for the price-based model with the sets of ledger 4.22-4.24:
#   V_C(x) = max { CVaR_a[phi(x, h(x, d))] : a in U(CP_EH), d in U(CP_ET), TV(a, d) <= CP_DL },   U(r) = TV(q̂, r) ∩ box (SC_BOXT, width r/sqrt(S)).
#   For a fixed belief d the max over a is an LP (worst()).  The max over d is nonconvex: point search (lower bound):
#     q̂, the Data-DR worst a and its midpoint, "the adversary's a becomes the belief" iterations, CP_NPT random extreme points of U(ET) and of U(ET/2), pair moves for the best decisions.
#   Candidates = the candidate list of the decisions run CP_CFG (geo/dw/dw_<CP_CFG>_g_s.csv).  Output: geo/dw/cp_<CP_CFG>_g_s.csv and sel_<CP_CFG>c_g_s.csv (= sel_<CP_CFG> + rows of model C).
const EH = parse(Float64, E("CP_EH", "0.1")); const ET = parse(Float64, E("CP_ET", "0.1")); const DL = parse(Float64, E("CP_DL", string(ET / 3)))   # coupling radius: a third of the belief radius unless CP_DL is given
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
open(SPD * "geo/dw/cp_$(tag0).csv", "w") do io
    println(io, "x,n,cap,V0,VD,VC,tv_d_q")
    for (ci, i) in enumerate(cand); println(io, LAB[i], ",", length(ALLD[i][1]), ",", @sprintf("%.0f", CAP[i]), ",", @sprintf("%.4f,%.4f,%.4f,%.4f", V0[ci], VD[ci], VC[ci], 0.5 * sum(abs.(DB[ci] .- q)))); end
end
sel = readlines(SPD * "geo/dw/sel_$(tag0).csv")
open(SPD * "geo/dw/sel_$(CCFG)c_g$(GG)_s$(SEED).csv", "w") do io
    for l in sel; println(io, l); end
    for (k, (q0, c)) in enumerate(SCEN)
        ci = sortperm(collect(zip(VC .+ CST[k][cand], LAB[cand])))[1]; println(io, k, ",", q0, ",", c, ",C,", LAB[cand[ci]])
        @printf("SELC scen %d: C=%s cap %.0f (V0 %.2f, VC %.2f)\n", k, LAB[cand[ci]], CAP[cand[ci]], V0[ci], VC[ci])
    end
end
@printf("CP_DONE g%s seed %d: %.0f s\n", GG, SEED, time() - t0)
