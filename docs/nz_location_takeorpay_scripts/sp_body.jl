const G = ENV["FB_G"]; const NSEED = parse(Int, E("TW_SEEDS", "10")); const MAXD = parse(Int, E("TW_MAXD", "0")); const KS = parse(Int, E("TW_KS", "1000")); const KD = parse(Int, E("SP_KD", "0"))
t0 = time(); cfgof(s) = s <= 3 ? "dw75" : "dwl75"
SITES = [collect(c) for k in 1:3 for c in Iterators.product(ntuple(_ -> 1:nd.nx, k)...) if issorted(collect(c)) && allunique(collect(c))]
ALLD = [(s, collect(lv)) for s in SITES for lv in Iterators.product(ntuple(_ -> TIERS, length(s))...)]; LAB = [dstr(s, lv) for (s, lv) in ALLD]; nALL = length(ALLD); IDX = Dict(LAB[i] => i for i in 1:nALL)
costv(q0, c) = [q0 * length(s) + c * sum(lv) for (s, lv) in ALLD]; CST = [costv(q0, c) for (q0, c) in SCEN]; CAP = [sum(lv) for (s, lv) in ALLD]; vv = nd.meta[:v]; const NK = length(SCEN)
# ---- decisions made from the training samples (seeds 1..NSEED): all model decisions (for the screen) and N per (cost scenario, seed)
DECI = Int[]; NDEC = Dict{Tuple{Int,Int},Int}()
for s in 1:NSEED, l in readlines(SPD * "geo/dw/sel_$(cfgof(s))_g$(G)_s$(s).csv")[2:end]
    f = split(l, ","); f[4] in ("N", "D0.1", "D0.3", "B0.1", "C") && push!(DECI, IDX[f[5]]); f[4] == "N" && (NDEC[(parse(Int, f[1]), s)] = IDX[f[5]])
end
# ---- the true F (test sample) and its 100-scenario representation F_B (same as truthW.jl)
ndT = mk(NT, TSEED); const pT = fill(1 / NT, NT)
subset(ndx, idx) = NZData(ndx.name * "_sub", ndx.A, ndx.c, ndx.ell, ndx.u[:, idx], ndx.hrow, ndx.hcoef, ndx.xrow, ndx.g[:, idx], ndx.W, ndx.wvec, ndx.c0, ndx.hU, ndx.nx, ndx.x_allowed, ndx.gamma, ndx.qx,
    length(idx), fill(1 / length(idx), length(idx)), ndx.eps_hat, ndx.eps_tilde, ndx.beta, ndx.thetaU, ndx.piLU, ndx.piFU, ndx.lambdaU, ndx.varpiU, ndx.meta)
totT = vec(sum(ndT.meta[:xi]; dims=1)); idxB = sortperm(totT)[3:5:NT]; ndB = subset(ndT, idxB); const NB = length(idxB); const pB = fill(1 / NB, NB)
function resp_setup(ndx)
    m, ny, S = nz_m(ndx), nz_ny(ndx), ndx.S; xr = [k for k in 1:m if ndx.xrow[k] > 0]; jc = [j for j in 1:ny if ndx.c[j] != 0]
    function base()
        mdl = Model(GRB1)
        @variable(mdl, 0 <= α[i=1:nh] <= ndx.hU[i]); @constraint(mdl, [l=1:size(ndx.W, 1)], sum(ndx.W[l, i] * α[i] for i in 1:nh) <= ndx.wvec[l]); @variable(mdl, y[1:ny, 1:S] >= 0)
        con = [@constraint(mdl, sum(ndx.A[k, j] * y[j, s] for j in 1:ny if ndx.A[k, j] != 0) - (ndx.hrow[k] > 0 ? ndx.hcoef[k] * α[ndx.hrow[k]] : 0.0) <= ndx.u[k, s]) for k in 1:m, s in 1:S]
        return mdl, α, y, con
    end
    mA, αA, yA, cA = base(); mB, αB, yB, cB = base()
    @variable(mB, t[1:nh] >= 0); @constraint(mB, [i=1:nh], t[i] - αB[i] >= -apess[i]); @constraint(mB, [i=1:nh], t[i] + αB[i] >= apess[i]); @objective(mB, Min, sum(t))
    copt = Ref{Any}(nothing)
    function val(α, y, p)
        e = AffExpr(0.0); for i in 1:nh; add_to_expression!(e, ndx.c0[i], α[i]); end
        for s in 1:S; p[s] == 0 && continue; for j in jc; add_to_expression!(e, p[s] * ndx.c[j], y[j, s]); end; end
        return e
    end
    function setp!(p)
        p = p ./ sum(p); set_objective(mA, MOI.MAX_SENSE, val(αA, yA, p)); copt[] === nothing || delete(mB, copt[]); copt[] = @constraint(mB, val(αB, yB, p) >= -1e30); return nothing
    end
    function solve(x̄)
        for k in xr, s in 1:S; r = ndx.u[k, s] + ndx.g[k, s] * x̄[ndx.xrow[k]]; set_normalized_rhs(cA[k, s], r); set_normalized_rhs(cB[k, s], r); end
        optimize!(mA); termination_status(mA) == MOI.OPTIMAL || error("response LP: $(termination_status(mA))")
        v = objective_value(mA); set_normalized_rhs(copt[], v - 1e-7 * max(1.0, abs(v))); optimize!(mB)
        return clamp.(termination_status(mB) == MOI.OPTIMAL ? value.(αB) : value.(αA), 0.0, ndx.hU)
    end
    return setp!, solve
end
function phi_setup(ndx)
    m, ny, S = nz_m(ndx), nz_ny(ndx), ndx.S; mdl = Model(GRB1); @variable(mdl, y[1:ny] >= 0)
    cons = [@constraint(mdl, sum(ndx.A[k, j] * y[j] for j in 1:ny if ndx.A[k, j] != 0) <= 0.0) for k in 1:m]
    cq = @constraint(mdl, dot(ndx.c, y) >= -1e30); oQ = @expression(mdl, dot(ndx.c, y)); oL = @expression(mdl, dot(ndx.ell, y))
    return (x̄, h) -> begin
        φ = zeros(S)
        for s in 1:S
            rr = nz_rhs(ndx, x̄, h, s); for k in 1:m; set_normalized_rhs(cons[k], rr[k]); end
            set_normalized_rhs(cq, -1e30); @objective(mdl, Max, oQ); optimize!(mdl); termination_status(mdl) == MOI.OPTIMAL || error("recourse LP s=$s: $(termination_status(mdl))")
            Q = objective_value(mdl); set_normalized_rhs(cq, Q - 1e-9 * max(1.0, abs(Q))); @objective(mdl, Max, oL); optimize!(mdl); termination_status(mdl) == MOI.OPTIMAL || error("pessimistic LP s=$s: $(termination_status(mdl))")
            φ[s] = objective_value(mdl)
        end
        φ
    end
end
phiT = phi_setup(ndT); phiB = phi_setup(ndB); setpB!, solveB = resp_setup(ndB); setpB!(pB)
JC = Dict{Tuple{Int,Vector{Float64}},Float64}()
Jof(i, h) = (hr = round.(h; digits=4); get!(() -> cvar(phiT(xfrac(ALLD[i]...), hr), pT), JC, (i, hr)))
mkpath(SPD * "geo/tw"); @printf("SP g%s: %d decisions, models built %.0f s\n", G, nALL, time() - t0); flush(stdout)
# ---- 1. screened decisions under F_B (same screen as truthW.jl): commitment of a retailer who knows the truth h_B(x), value V_B(x), N*
VS = zeros(nALL, NSEED)
for s in 1:NSEED
    L = readlines(SPD * "geo/dw/vn_$(cfgof(s))_g$(G)_s$(s).csv")[2:end]; length(L) == nALL || error("vn rows $(length(L)) != $nALL (seed $s)")
    for (i, l) in enumerate(L); f = split(l, ","); f[1] == LAB[i] || error("vn label mismatch seed $s row $i"); VS[i, s] = parse(Float64, f[4]); end
end
vbar = vec(mean(VS; dims=2)); SCRORD = [sortperm(collect(zip(vbar .+ CST[k], LAB))) for k in 1:NK]
ENUM = MAXD > 0 ? sort(unique(vcat(collect(1:MAXD), DECI))) : KS > 0 ? sort(unique(vcat([o[1:min(KS, nALL)] for o in SCRORD]..., DECI))) : collect(1:nALL)
@printf("  enumerating %d decisions (screen KS = %d)\n", length(ENUM), KS); flush(stdout)
H = zeros(nh, nALL); VB = fill(Inf, nALL)
for (c, i) in enumerate(ENUM)
    x̄ = xfrac(ALLD[i]...); H[:, i] = solveB(x̄); VB[i] = cvar(phiB(x̄, H[:, i]), pB)
    c % 500 == 0 && (@printf("  enum %d / %d  %.0f s\n", c, length(ENUM), time() - t0); flush(stdout))
end
NST = [ENUM[argmin([(VB[i] + CST[k][i], LAB[i]) for i in ENUM])] for k in 1:NK]
open(SPD * "geo/tw/hB_g$(G).csv", "w") do io
    println(io, "x,VB," * join(["h$(j)" for j in 1:nh], ",")); for i in ENUM; println(io, LAB[i], ",", @sprintf("%.4f", VB[i]), ",", join([@sprintf("%.3f", H[j, i]) for j in 1:nh], ",")); end
end
@printf("  truth enumeration done %.0f s; N* = %s\n", time() - t0, join([LAB[NST[k]] for k in 1:NK], " / ")); flush(stdout)
# ---- 2. per training sample: three leaders
#   N  (standard)        : own demand sample, commitment predicted from the same sample     V_ss(x) = CVaR_q[phi_s(x, h_s(x))]
#   Nr (order known)     : own demand sample, commitment of the truth-knowing retailer       V_rs(x) = CVaR_q[phi_s(x, h_B(x))]
#   Nd (demand known)    : true demand F_B, commitment predicted from the own sample         V_ds(x) = CVaR_pB[phi_B(x, h_s(x))]
ROWS = String[]; NEW = Int[]; bord = [sortperm([(VB[i] + CST[k][i], LAB[i]) for i in ENUM]) for k in 1:NK]
for s in 1:NSEED
    nds = mk(10, s); qs = nds.q_hat; setps!, solves = resp_setup(nds); setps!(qs); phis = phi_setup(nds)
    Vss = fill(Inf, nALL); Vrs = fill(Inf, nALL); Vds = fill(Inf, nALL); HS = zeros(nh, nALL)
    for i in ENUM
        x̄ = xfrac(ALLD[i]...); HS[:, i] = solves(x̄); Vss[i] = cvar(phis(x̄, HS[:, i]), qs); Vrs[i] = cvar(phis(x̄, H[:, i]), qs)
    end
    dmax = maximum(abs.(Vss[ENUM] .- VS[ENUM, s])); t1 = time() - t0
    vds!(i) = (isfinite(Vds[i]) || (Vds[i] = cvar(phiB(xfrac(ALLD[i]...), HS[:, i]), pB)); Vds[i])
    for k in 1:NK
        ED = KD > 0 ? unique(vcat(ENUM[bord[k][1:min(KD, end)]], ENUM[sortperm([(Vss[i] + CST[k][i], LAB[i]) for i in ENUM])[1:min(KD, end)]], ENUM[sortperm([(Vrs[i] + CST[k][i], LAB[i]) for i in ENUM])[1:min(KD, end)]])) : ENUM
        for i in ED; vds!(i); end
        iN = ENUM[argmin([(Vss[i] + CST[k][i], LAB[i]) for i in ENUM])]; iR = ENUM[argmin([(Vrs[i] + CST[k][i], LAB[i]) for i in ENUM])]; iD = ED[argmin([(Vds[i] + CST[k][i], LAB[i]) for i in ED])]
        i0 = NDEC[(k, s)]; okN = iN == i0
        okN || @printf("  NOTE seed %d scen %d: recomputed N %s (%.4f) differs from stored N %s (%.4f)\n", s, k, LAB[iN], Vss[iN] + CST[k][iN], LAB[i0], Vss[i0] + CST[k][i0])
        for (who, i) in (("N", i0), ("Nr", iR), ("Nd", iD), ("N*", NST[k]))
            push!(ROWS, join([string(k), string(SCEN[k][1]), string(SCEN[k][2]), string(s), who, LAB[i], @sprintf("%.0f", CAP[i]), @sprintf("%.4f,%.4f,%.4f,%.4f", Vss[i], Vrs[i], vds!(i), VB[i]),
                @sprintf("%.2f,%.2f,%.2f", sum(HS[nA+1:end, i]), sum(H[nA+1:end, i]), sum(abs.(HS[nA+1:end, i] .- H[nA+1:end, i]))), string(okN), @sprintf("%.5f", dmax)], ","))
        end
        push!(NEW, iR, iD)
    end
    @printf("  seed %d done %.0f s (own-sample part %.0f s), max |V_ss - stored| = %.5f\n", s, time() - t0, t1, dmax); flush(stdout)
end
open(SPD * "geo/tw/split_g$(G).csv", "w") do io
    println(io, "scen,q0,c,seed,who,x,cap,Vss,Vrs,Vds,VB,hB_pred,hB_true,hB_l1,okN,dmax"); for r in ROWS; println(io, r); end
end
# ---- 3. true-cost evaluation of the new decisions (same worlds and draws as truthW.jl); decisions already in eval_g<G>.csv are skipped
HAVE = Set{String}(String(split(l, ",")[1]) for l in readlines(SPD * "geo/tw/eval_g$(G).csv")[2:end])
TODO = [i for i in sort(unique(NEW)) if !(LAB[i] in HAVE)]; @printf("  new decisions %d, to evaluate %d\n", length(unique(NEW)), length(TODO)); flush(stdout)
open(SPD * "geo/tw/eval2_g$(G).csv", "w") do ioe
    println(ioe, "x,world,k,J")
    for i in TODO; @printf(ioe, "%s,T,0,%.6f\n", LAB[i], Jof(i, H[:, i])); end
    for nf in NFS, kd in 1:K
        ndF = mk(nf, 6_000_000 + 1000 * parse(Int, G) + kd + (nf == 5 ? 0 : 10_000_000 * nf)); pF = fill(1 / nf, nf)
        for i in TODO
            x̄ = xfrac(ALLD[i]...); h = _nz_resp!(_nz_build_resp(ndF, x̄; optimizer=GRB1), ndF, pF, apess); h === nothing && error("retailer LP failed (nf=$nf k=$kd $(LAB[i]))")
            @printf(ioe, "%s,%d,%d,%.6f\n", LAB[i], nf, kd, Jof(i, h))
        end
    end
end
@printf("SP_DONE g%s: %d new decisions evaluated, %.0f s\n", G, length(TODO), time() - t0)
