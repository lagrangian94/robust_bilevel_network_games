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
