# xiW.jl = dwLeanP.jl lines 1-99 + this body: demand scenarios of geography FB_G, seeds 1..10: totals, regional totals, share of customers below their own median
open(SPD * "geo/mech/xi_g$(ENV["FB_G"]).csv", "w") do io
    for s in 1:10
        n = mk(10, s); ξ = n.meta[:xi]; reg = n.meta[:region]; nc = size(ξ, 1)
        println(io, join(vcat(["$s", "tot"], [@sprintf("%.1f", sum(ξ[:, k])) for k in 1:10]), ","))
        for r in 1:3; println(io, join(vcat(["$s", "reg$r:$(count(==(r), reg))"], [@sprintf("%.1f", sum(ξ[reg .== r, k])) for k in 1:10]), ",")); end
        println(io, join(vcat(["$s", "below"], [@sprintf("%.3f", count(j -> ξ[j, k] < median(ξ[j, :]), 1:nc) / nc) for k in 1:10]), ","))
    end
end
println("XI_DONE")
