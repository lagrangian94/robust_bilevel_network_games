# dwD5W.jl (2026-11-18 evening) = dwLeanB.jl lines 1-117 + this body.  Data-DR at one more radius (D5_EPS, default 0.5) by full enumeration, the same rule as dwLeanB.jl section 1
#   (V_D(x) = max over a in TV(q̂, eps) ∩ box of CVaR_a[phi(x, h(x, q̂))], decision = argmin V_D + store cost over ALL decisions).  Ledger 4.33.
#   V0 of every decision is checked against the stored enumeration vn_<cfg>_g_s.csv.
#   Output geo/dw/vnx_<cfg>_D<eps>_g_s.csv (x, V0, VD), selx_<cfg>_D<eps>_g_s.csv (scen, q0, c, model, x) and
#          candx_<cfg>_<eps>_g_s.csv = stored candidates of dw_<cfg>_g_s.csv + the top NCD[k] decisions of V_D(eps) + cost of each cost scenario (x, new, V0, VD).
#   Test: D5_MAXD = n > 0 enumerates only the stored candidates and the first n decisions, D5_SUF is appended to the output file names.
const DEPS = parse(Float64, E("D5_EPS", "0.5")); const DCFG = E("D5_CFG", "dwps3"); const GG = ENV["FB_G"]; const D5MAX = parse(Int, E("D5_MAXD", "0")); const SUF = E("D5_SUF", "")
t0 = time()
SITES = [collect(c) for k in 1:3 for c in Iterators.product(ntuple(_ -> 1:nd.nx, k)...) if issorted(collect(c)) && allunique(collect(c))]
ALLD = [(s, collect(lv)) for s in SITES for lv in Iterators.product(ntuple(_ -> TIERS, length(s))...)]; LAB = [dstr(s, lv) for (s, lv) in ALLD]; IDX = Dict(LAB[i] => i for i in eachindex(LAB))
costv(q0, c) = [q0 * length(s) + c * sum(lv) for (s, lv) in ALLD]
tag0 = "$(DCFG)_g$(GG)_s$(SEED)"
VN = Dict(String(split(l, ",")[1]) => parse(Float64, split(l, ",")[4]) for l in readlines(SPD * "geo/dw/vn_$(tag0).csv")[2:end]); length(VN) == length(ALLD) || error("vn_$(tag0).csv has $(length(VN)) decisions, expected $(length(ALLD))")
cand = [IDX[String(split(l, ",")[1])] for l in readlines(SPD * "geo/dw/dw_$(tag0).csv")[2:end]]
idxs = D5MAX > 0 ? unique(vcat(cand, collect(1:min(D5MAX, length(ALLD))))) : collect(1:length(ALLD))
V0 = fill(Inf, length(ALLD)); VD = fill(Inf, length(ALLD))
for i in idxs
    Z = evalx(xfrac(ALLD[i]...))(q); V0[i] = cvar(Z, q)
    abs(V0[i] - VN[LAB[i]]) < 1e-3 || error("V0 of $(LAB[i]) differs from the stored enumeration: $(V0[i]) vs $(VN[LAB[i]])")
    VD[i] = worst(Z, q, Inf, DEPS)[1]; isfinite(VD[i]) || error("worst-case LP failed for $(LAB[i])")
end
@printf("ENUM %d decisions %.0f s, eps %.2f, box t = %.2f\n", length(idxs), time() - t0, DEPS, BOXT); flush(stdout)
open(SPD * "geo/dw/vnx_$(DCFG)_D$(DEPS)_g$(GG)_s$(SEED)$(SUF).csv", "w") do io
    println(io, "x,n,cap,V0,VD$(DEPS)")
    for i in idxs; println(io, LAB[i], ",", length(ALLD[i][1]), ",", @sprintf("%.0f", sum(ALLD[i][2])), ",", @sprintf("%.4f,%.4f", V0[i], VD[i])); end
end
ordby(v) = idxs[sortperm(collect(zip(v[idxs], LAB[idxs])))]
ext = copy(cand)
open(SPD * "geo/dw/selx_$(DCFG)_D$(DEPS)_g$(GG)_s$(SEED)$(SUF).csv", "w") do io
    println(io, "scen,q0,c,model,x")
    for (k, (q0, c)) in enumerate(SCEN)
        o = ordby(VD .+ costv(q0, c)); println(io, k, ",", q0, ",", c, ",D$(DEPS),", LAB[o[1]])
        @printf("SELD scen %d: D%s=%s cap %.0f (V0 %.2f, VD %.2f), in stored candidates %s\n", k, string(DEPS), LAB[o[1]], sum(ALLD[o[1]][2]), V0[o[1]], VD[o[1]], string(o[1] in cand))
        append!(ext, o[1:min(NCD[k], length(o))])
    end
end
ext = unique(ext); nnew = length(ext) - length(cand)
open(SPD * "geo/dw/candx_$(DCFG)_$(DEPS)_g$(GG)_s$(SEED)$(SUF).csv", "w") do io
    println(io, "x,new,V0,VD$(DEPS)")
    for i in ext; println(io, LAB[i], ",", i in cand ? 0 : 1, ",", @sprintf("%.4f,%.4f", V0[i], VD[i])); end
end
@printf("D5_DONE g%s seed %d: %d decisions, %d stored candidates + %d new, %.0f s\n", GG, SEED, length(idxs), length(cand), nnew, time() - t0)
