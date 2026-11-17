# xidW.jl = dwLeanP.jl lines 1-99 + this body: dump the cities, coordinates, regions, mean demand and the 10 demand scenarios of geography FB_G, seed C0H_SEED
ξ = nd.meta[:xi]; reg = nd.meta[:region]; cities = nd.meta[:cities]; coord = nd.meta[:coord]; μ = sgb15_mu(sgb15_kuhn().pop)
open(SPD * "geo/mech/xid_g$(ENV["FB_G"])_s$(SEED).csv", "w") do io
    println(io, "kind;name;x;y;region;mu;" * join(["xi$k" for k in 1:S], ";"))
    for i in 1:nst; println(io, join([i <= nA ? "A" : "B", cities[i], coord[i, 1], coord[i, 2], REGION[cities[i]], "", fill("", S)...], ";")); end
    for j in 1:ncu; println(io, join(["C", cities[nst+j], coord[nst+j, 1], coord[nst+j, 2], reg[j], μ[j], ξ[j, :]...], ";")); end
end
println("XID_DONE")
