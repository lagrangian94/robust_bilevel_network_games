# mechW.jl (2026-11-17 night) = dwLeanP.jl lines 1-99 + zs_body.jl lines 1-80 + this body.  Why is Data-DR so much more conservative than Belief-only?
#   For every stored decision (ZS_MODELS of config ZS_CFG) write the commitment and the leader loss per scenario
#   (1) at belief q̂, (2) at the 90 beliefs "all the probability of scenario i moves to scenario j" (the extreme points of the TV ball of radius 0.1 around the uniform q̂).
#   Output geo/<ZS_OUT>/mech_g_s.csv: seed,v,vsm,x,cap,nsite,bel,i,j,hB,hA,Z1..ZS   (Z = leader loss per scenario without store cost; revenue = -Z = v (hB + vsm * spot)).
open(SPD * "geo/$(OUT)/mech_g$(G)_s$(SS).csv", "w") do io
    println(io, "seed,v,vsm,x,cap,nsite,bel,i,j,hB,hA," * join(["Z$s" for s in 1:S], ","))
    function row(bel, i1, j1)
        for i in EV
            h = solves(xfrac(ALLD[i]...)); Z = Zof(i, h)
            println(io, join([string(SS), string(nds.meta[:v]), string(VSM), LAB[i], @sprintf("%.0f", CAP[i]), string(length(ALLD[i][1])), bel, string(i1), string(j1),
                @sprintf("%.3f", sum(h[nA+1:end])), @sprintf("%.3f", sum(h[1:nA])), join([@sprintf("%.4f", z) for z in Z], ",")], ","))
        end
    end
    setps!(qs); row("q", 0, 0)
    for i1 in 1:S, j1 in 1:S
        i1 == j1 && continue
        d = copy(qs); d[j1] += d[i1]; d[i1] = 0.0; setps!(d); row("pair", i1, j1)
    end
end
@printf("MECH_DONE g%s seed %d: %.0f s\n", G, SS, time() - t0)
