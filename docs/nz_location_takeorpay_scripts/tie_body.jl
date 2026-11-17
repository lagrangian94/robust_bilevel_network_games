const G = ENV["FB_G"]; const SS = parse(Int, E("TIE_SEED", "2")); t0 = time(); cfgof(s) = s <= 3 ? "dw75" : "dwl75"
SITES = [collect(c) for k in 1:3 for c in Iterators.product(ntuple(_ -> 1:nd.nx, k)...) if issorted(collect(c)) && allunique(collect(c))]
ALLD = [(s, collect(lv)) for s in SITES for lv in Iterators.product(ntuple(_ -> TIERS, length(s))...)]; LAB = [dstr(s, lv) for (s, lv) in ALLD]; nALL = length(ALLD); IDX = Dict(LAB[i] => i for i in 1:nALL)
ENUM = [IDX[String(split(l, ",")[1])] for l in readlines(SPD * "geo/tw/hB_g$(G).csv")[2:end]]
VSt = Dict(String(split(l, ",")[1]) => parse(Float64, split(l, ",")[4]) for l in readlines(SPD * "geo/dw/vn_$(cfgof(SS))_g$(G)_s$(SS).csv")[2:end])
nds = mk(10, SS); qs = nds.q_hat
# range of each commitment over the retailer's optimal set, and over the optimal commitments closest to apess
function ranges(x̄)
    m, ny, S = nz_m(nds), nz_ny(nds), nds.S; mdl = Model(GRB1)
    @variable(mdl, 0 <= α[i=1:nh] <= nds.hU[i]); @constraint(mdl, [l=1:size(nds.W, 1)], sum(nds.W[l, i] * α[i] for i in 1:nh) <= nds.wvec[l]); @variable(mdl, y[1:ny, 1:S] >= 0)
    @constraint(mdl, [k=1:m, s=1:S], sum(nds.A[k, j] * y[j, s] for j in 1:ny if nds.A[k, j] != 0) - (nds.hrow[k] > 0 ? nds.hcoef[k] * α[nds.hrow[k]] : 0.0) <= nds.u[k, s] + (nds.xrow[k] > 0 ? nds.g[k, s] * x̄[nds.xrow[k]] : 0.0))
    obj = @expression(mdl, sum(nds.c0[i] * α[i] for i in 1:nh) + sum(qs[s] * nds.c[j] * y[j, s] for j in 1:ny, s in 1:S if nds.c[j] != 0))
    @objective(mdl, Max, obj); optimize!(mdl); v = objective_value(mdl); @constraint(mdl, obj >= v - 1e-7 * max(1.0, abs(v)))
    r1 = zeros(nh, 2); for i in 1:nh; @objective(mdl, Min, α[i]); optimize!(mdl); r1[i, 1] = objective_value(mdl); @objective(mdl, Max, α[i]); optimize!(mdl); r1[i, 2] = objective_value(mdl); end
    @variable(mdl, t[1:nh] >= 0); @constraint(mdl, [i=1:nh], t[i] - α[i] >= -apess[i]); @constraint(mdl, [i=1:nh], t[i] + α[i] >= apess[i]); @objective(mdl, Min, sum(t)); optimize!(mdl); d = objective_value(mdl); @constraint(mdl, sum(t) <= d + 1e-7 * max(1.0, abs(d)))
    r2 = zeros(nh, 2); for i in 1:nh; @objective(mdl, Min, α[i]); optimize!(mdl); r2[i, 1] = objective_value(mdl); @objective(mdl, Max, α[i]); optimize!(mdl); r2[i, 2] = objective_value(mdl); end
    return v, r1, r2
end
nbig = 0; ntie1 = 0; ntie2 = 0; dv = Float64[]; shown = 0
for i in ENUM
    x̄ = xfrac(ALLD[i]...); h = _nz_resp!(_nz_build_resp(nds, x̄; optimizer=GRB1), nds, qs, apess); V = cvar(nz_recourse_values(nds, x̄, h)[2], qs); push!(dv, V - VSt[LAB[i]])
    v, r1, r2 = ranges(x̄); w1 = maximum(r1[:, 2] .- r1[:, 1]); w2 = maximum(r2[:, 2] .- r2[:, 1]); global ntie1 += w1 > 0.5; global ntie2 += w2 > 0.5
    if abs(dv[end]) > 0.5 || w2 > 0.5
        global nbig += abs(dv[end]) > 0.5
        if shown < 8
            global shown += 1; @printf("  %s: V fresh %.3f, stored %.3f | optimal-set width max %.1f | closest-to-apess set width max %.1f | h fresh B %s\n", LAB[i], V, VSt[LAB[i]], w1, w2, string(round.(h[nA+1:end][h[nA+1:end] .> 0.01]; digits=1)))
            for j in 1:nh; (r2[j, 2] - r2[j, 1] > 0.5) && @printf("      site %d: [%.1f, %.1f]\n", j, r2[j, 1], r2[j, 2]); end
        end
    end
end
@printf("TIE g%s seed %d: %d decisions | |V fresh - stored| > 0.5 in %d (max %.3f) | retailer optimal commitment not unique in %d | not unique even after the closest-to-apess rule in %d | %.0f s\n", G, SS, length(ENUM), nbig, maximum(abs.(dv)), ntie1, ntie2, time() - t0)
