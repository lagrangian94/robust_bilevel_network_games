# dwB3W.jl (2026-11-18) = dwLeanB.jl lines 1-117 + this body.  Belief-only at one more radius (DT_BNEW, default 0.3) on the stored candidates of the decisions run B3_CFG (default dwps3).
#   Same rule as dwLeanB.jl (belief ball with the box SC_BOXTB): adversarial damage of a candidate = max over the belief points of the ball of radius t (seed hash((9292, SEED, round(1000 t)))) of CVaR_q̂[phi(d)] - CVaR_q̂[phi(q̂)],
#   made monotone in t by taking the max with the stored damage of the next smaller radius (column adv<DT_BPREV>, default 0.2).  Nothing is re-enumerated.
#   Output geo/dw/dwx_<cfg>_B<t>_g_s.csv (x, V0, adv_prev, adv_raw, adv) and geo/dw/selx_<cfg>_B<t>_g_s.csv (scen, q0, c, model, x).
const TNEW = parse(Float64, E("DT_BNEW", "0.3")); const TPREV = E("DT_BPREV", "0.2"); const BCFG = E("B3_CFG", "dwps3"); const GG = ENV["FB_G"]
t0 = time()
SITES = [collect(c) for k in 1:3 for c in Iterators.product(ntuple(_ -> 1:nd.nx, k)...) if issorted(collect(c)) && allunique(collect(c))]
ALLD = [(s, collect(lv)) for s in SITES for lv in Iterators.product(ntuple(_ -> TIERS, length(s))...)]; LAB = [dstr(s, lv) for (s, lv) in ALLD]; IDX = Dict(LAB[i] => i for i in eachindex(LAB))
tag0 = "$(BCFG)_g$(GG)_s$(SEED)"; L = readlines(SPD * "geo/dw/dw_$(tag0).csv"); hd = split(L[1], ","); kp = findfirst(==("adv$(TPREV)"), hd); kp === nothing && error("no column adv$(TPREV) in dw_$(tag0).csv")
cand = [IDX[String(split(l, ",")[1])] for l in L[2:end]]; advp = [parse(Float64, split(l, ",")[kp]) for l in L[2:end]]; v0f = [parse(Float64, split(l, ",")[4]) for l in L[2:end]]; nc = length(cand)
pts = (BOXTB > 0 ? ball_pts_box : ball_pts)(TNEW, MersenneTwister(hash((9292, SEED, round(Int, 1000TNEW))))); V0 = zeros(nc); RAW = zeros(nc)
for (ci, i) in enumerate(cand)
    phi = evalx(xfrac(ALLD[i]...)); V0[ci] = cvar(phi(q), q); abs(V0[ci] - v0f[ci]) < 1e-3 || error("V0 of $(LAB[i]) differs from the stored value: $(V0[ci]) vs $(v0f[ci])")
    RAW[ci] = max(0.0, maximum(cvar(phi(d), q) for d in pts) - V0[ci])
end
ADV = max.(advp, RAW)
open(SPD * "geo/dw/dwx_$(BCFG)_B$(TNEW)_g$(GG)_s$(SEED).csv", "w") do io
    println(io, "x,V0,adv$(TPREV),adv_raw,adv$(TNEW)"); for (ci, i) in enumerate(cand); println(io, LAB[i], ",", @sprintf("%.4f,%.4f,%.4f,%.4f", V0[ci], advp[ci], RAW[ci], ADV[ci])); end
end
open(SPD * "geo/dw/selx_$(BCFG)_B$(TNEW)_g$(GG)_s$(SEED).csv", "w") do io
    println(io, "scen,q0,c,model,x")
    for (k, (q0, c)) in enumerate(SCEN)
        cs = [q0 * length(ALLD[i][1]) + c * sum(ALLD[i][2]) for i in cand]; ci = sortperm(collect(zip(V0 .+ cs .+ ADV, LAB[cand])))[1]
        println(io, k, ",", q0, ",", c, ",B$(TNEW),", LAB[cand[ci]]); @printf("SELB scen %d: B%s=%s (adv %.2f, raw %.2f)\n", k, string(TNEW), LAB[cand[ci]], ADV[ci], RAW[ci])
    end
end
@printf("BOX data t = %.2f, belief t = %.2f
", BOXT, BOXTB)
@printf("B3_DONE g%s seed %d: %d candidates, %d belief points, %.0f s\n", GG, SEED, nc, length(pts), time() - t0)
