# mechSW.jl (2026-11-18) = dwLeanP.jl lines 1-99 + zs_body.jl lines 1-80 + this body.  Which stores keep their commitment when the retailer's belief moves?
#   For every stored decision (ZS_MODELS of config ZS_CFG) write the commitment per store (existing operator's stores first, then the 13 leader candidates) and the leader loss per scenario
#   at belief q̂ and at the 90 pair-move beliefs.  Output geo/<ZS_OUT>/mechs_g_s.csv: seed;x;bel;i;j;h1..h<nst>;Z1..ZS   (delimiter ";" , the label x contains ";" so read from the right)
open(SPD * "geo/$(OUT)/mechs_g$(G)_s$(SS).csv", "w") do io
    println(io, "seed|x|bel|i|j|" * join(["h$k" for k in 1:nst], "|") * "|" * join(["Z$s" for s in 1:S], "|"))
    function row(bel, i1, j1)
        for i in EV
            h = solves(xfrac(ALLD[i]...)); Z = Zof(i, h)
            println(io, join([string(SS), LAB[i], bel, string(i1), string(j1), join([@sprintf("%.3f", x) for x in h], "|"), join([@sprintf("%.4f", z) for z in Z], "|")], "|"))
        end
    end
    setps!(qs); row("q", 0, 0)
    for i1 in 1:S, j1 in 1:S
        i1 == j1 && continue
        d = copy(qs); d[j1] += d[i1]; d[i1] = 0.0; setps!(d); row("pair", i1, j1)
    end
end
@printf("MECHS_DONE g%s seed %d: nA %d nst %d, %.0f s\n", G, SS, nA, nst, time() - t0)
