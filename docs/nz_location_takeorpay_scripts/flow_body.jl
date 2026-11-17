# flowW.jl (2026-11-17 night) = dwLeanP.jl lines 1-99 + zs_body.jl lines 1-80 + this body.  Actual shipments of the retailer for the Nominal decision (cost table scen FL_SCEN), belief q̂.
#   Output geo/<ZS_OUT>/flow_g_s.csv: seed,x,i,kind,sreg,open_cap,j,creg,dist,mu,yR,yS  (yR, yS = probability-weighted mean shipments over the 10 scenarios)
function flow_setup(ndx)
    m, ny, S = nz_m(ndx), nz_ny(ndx), ndx.S; mdl = Model(GRB1); @variable(mdl, y[1:ny] >= 0)
    cons = [@constraint(mdl, sum(ndx.A[k, j] * y[j] for j in 1:ny if ndx.A[k, j] != 0) <= 0.0) for k in 1:m]
    cq = @constraint(mdl, dot(ndx.c, y) >= -1e30); oQ = @expression(mdl, dot(ndx.c, y)); oL = @expression(mdl, dot(ndx.ell, y))
    return (x̄, h) -> begin
        Y = zeros(ny, S)
        for s in 1:S
            rr = nz_rhs(ndx, x̄, h, s); for k in 1:m; set_normalized_rhs(cons[k], rr[k]); end
            set_normalized_rhs(cq, -1e30); @objective(mdl, Max, oQ); optimize!(mdl); termination_status(mdl) == MOI.OPTIMAL || error("recourse LP s=$s: $(termination_status(mdl))")
            Q = objective_value(mdl); set_normalized_rhs(cq, Q - 1e-9 * max(1.0, abs(Q))); @objective(mdl, Max, oL); optimize!(mdl); termination_status(mdl) == MOI.OPTIMAL || error("pessimistic LP s=$s: $(termination_status(mdl))")
            Y[:, s] = value.(y)
        end
        Y
    end
end
const FSC = parse(Int, E("FL_SCEN", "2")); iN = DEC[(FSC, "N")]; sites, lv = ALLD[iN]; x̄ = xfrac(sites, lv)
setps!(qs); h = solves(x̄); Y = flow_setup(nds)(x̄, round.(h; digits=4))
iyR = nds.meta[:iyR]; iyS = nds.meta[:iyS]; cities = nds.meta[:cities]; dist = nds.meta[:dist]; creg = nds.meta[:region]; μ = sgb15_mu(sgb15_kuhn().pop); nstl = nds.meta[:nst]; ncul = nds.meta[:ncu]
open(SPD * "geo/$(OUT)/flow_g$(G)_s$(SS).csv", "w") do io
    println(io, "seed;x;i;kind;sreg;cap;h;j;creg;dist;mu;yR;yS")
    for i in 1:nstl
        capi = i <= nA ? nds.meta[:bA] : (k = findfirst(==(i - nA), sites); k === nothing ? 0.0 : lv[k]); (i > nA && capi == 0) && continue
        for j in 1:ncul
            println(io, join([SS, LAB[iN], i, i <= nA ? "A" : "B", REGION[cities[i]], capi, round(h[i]; digits=2), j, creg[j], dist[i, j], round(μ[j]; digits=1),
                round(dot(qs, Y[iyR(i, j), :]); digits=3), round(dot(qs, Y[iyS(i, j), :]); digits=3)], ";"))
        end
    end
end
@printf("FLOW_DONE g%s seed %d: %.0f s\n", G, SS, time() - t0)
