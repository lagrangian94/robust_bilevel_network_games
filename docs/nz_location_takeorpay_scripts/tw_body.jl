const G = ENV["FB_G"]; const NSEED = parse(Int, E("TW_SEEDS", "10")); const KB = parse(Int, E("TW_KB", "10")); const KO = parse(Int, E("TW_KO", "10")); const NBEL = parse(Int, E("TW_NBEL", "0")); const MAXD = parse(Int, E("TW_MAXD", "0"))
const TSEED = 7770001; t0 = time(); cfgof(s) = s <= 3 ? "dw75" : "dwl75"
SITES = [collect(c) for k in 1:3 for c in Iterators.product(ntuple(_ -> 1:nd.nx, k)...) if issorted(collect(c)) && allunique(collect(c))]
ALLD = [(s, collect(lv)) for s in SITES for lv in Iterators.product(ntuple(_ -> TIERS, length(s))...)]; LAB = [dstr(s, lv) for (s, lv) in ALLD]; nALL = length(ALLD); IDX = Dict(LAB[i] => i for i in 1:nALL)
costv(q0, c) = [q0 * length(s) + c * sum(lv) for (s, lv) in ALLD]; CST = [costv(q0, c) for (q0, c) in SCEN]; CAP = [sum(lv) for (s, lv) in ALLD]; vv = nd.meta[:v]
# ---- decisions made from the training samples (seeds 1..NSEED)
DEC = Tuple{Int,String,Int,Int}[]   # (cost scenario, model, seed, decision index)
for s in 1:NSEED, l in readlines(SPD * "geo/dw/sel_$(cfgof(s))_g$(G)_s$(s).csv")[2:end]
    f = split(l, ","); f[4] in ("N", "D0.1", "D0.3", "B0.1", "C") && push!(DEC, (parse(Int, f[1]), String(f[4]), s, IDX[f[5]]))
end
# ---- the true F (test sample, NT scenarios) and its 100-scenario representation F_B (every 5th scenario in the order of total demand)
ndT = mk(NT, TSEED); const pT = fill(1 / NT, NT)
subset(ndx, idx) = NZData(ndx.name * "_sub", ndx.A, ndx.c, ndx.ell, ndx.u[:, idx], ndx.hrow, ndx.hcoef, ndx.xrow, ndx.g[:, idx], ndx.W, ndx.wvec, ndx.c0, ndx.hU, ndx.nx, ndx.x_allowed, ndx.gamma, ndx.qx,
    length(idx), fill(1 / length(idx), length(idx)), ndx.eps_hat, ndx.eps_tilde, ndx.beta, ndx.thetaU, ndx.piLU, ndx.piFU, ndx.lambdaU, ndx.varpiU, ndx.meta)
totT = vec(sum(ndT.meta[:xi]; dims=1)); idxB = sortperm(totT)[3:5:NT]; ndB = subset(ndT, idxB); const NB = length(idxB); const pB = fill(1 / NB, NB)
"retailer response on the scenario set of ndx: the same two LPs as _nz_resp! (optimal value, then the optimal commitment closest to apess), built once; belief via setp!, decision via solve"
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
"pessimistic leader cost per scenario of ndx for a given commitment h (same two LPs per scenario as phi_test in dwTier.jl), model built once"
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
Jof(i, h) = (hr = round.(h; digits=4); get!(() -> cvar(phiT(xfrac(ALLD[i]...), hr), pT), JC, (i, hr)))   # true-F cost of decision i when the retailer commits h (no store cost)
mkpath(SPD * "geo/tw"); @printf("TW g%s: %d decisions, models built %.0f s\n", G, nALL, time() - t0); flush(stdout)
# ---- 1. all decisions under F_B (leader and retailer both know F_B): commitment h_B(x), value V_B(x); N* = argmin
#   screen: the KS best decisions per cost scenario by the seed-average in-sample value (vn files), plus every model decision. KS = 0 enumerates everything.
const KS = parse(Int, E("TW_KS", "1000")); VS = zeros(nALL, NSEED)
for s in 1:NSEED
    L = readlines(SPD * "geo/dw/vn_$(cfgof(s))_g$(G)_s$(s).csv")[2:end]; length(L) == nALL || error("vn rows $(length(L)) != $nALL (seed $s)")
    for (i, l) in enumerate(L); f = split(l, ","); f[1] == LAB[i] || error("vn label mismatch seed $s row $i"); VS[i, s] = parse(Float64, f[4]); end
end
vbar = vec(mean(VS; dims=2)); SCRORD = [sortperm(collect(zip(vbar .+ CST[k], LAB))) for k in 1:length(SCEN)]
ENUM = MAXD > 0 ? sort(unique(vcat(collect(1:MAXD), [d[4] for d in DEC]))) : KS > 0 ? sort(unique(vcat([o[1:min(KS, nALL)] for o in SCRORD]..., [d[4] for d in DEC]))) : collect(1:nALL)
@printf("  enumerating %d decisions (screen KS = %d)\n", length(ENUM), KS); flush(stdout)
H = zeros(nh, nALL); VB = fill(Inf, nALL)
for (c, i) in enumerate(ENUM)
    x̄ = xfrac(ALLD[i]...); H[:, i] = solveB(x̄); VB[i] = cvar(phiB(x̄, H[:, i]), pB)
    c % 1000 == 0 && (@printf("  enum %d / %d  %.0f s\n", c, length(ENUM), time() - t0); flush(stdout))
end
NST = [ENUM[argmin([(VB[i] + CST[k][i], LAB[i]) for i in ENUM])] for k in 1:length(SCEN)]
open(SPD * "geo/tw/star_g$(G).csv", "w") do io
    println(io, "x,n,cap,hB,VB"); for i in ENUM; println(io, LAB[i], ",", length(ALLD[i][1]), ",", @sprintf("%.0f,%.1f,%.4f", CAP[i], sum(H[nA+1:end, i]), VB[i])); end
end
for k in 1:length(SCEN); @printf("  N* (c/v %.1f) = %s, V_B + cost %.2f, screen rank %d\n", SCEN[k][2] / vv, LAB[NST[k]], VB[NST[k]] + CST[k][NST[k]], findfirst(==(NST[k]), SCRORD[k])); end
@printf("  enumeration done %.0f s\n", time() - t0); flush(stdout)
# ---- 2. evaluation under the true F: retailer knows F_B (world T), or holds its own nf-sample (K common draws per geography)
WHO = Tuple{Int,String,Int,Int}[(k, "N*", 0, NST[k]) for k in 1:length(SCEN)]; append!(WHO, DEC)
for k in 1:length(SCEN); o = sort(ENUM; by=i -> (VB[i] + CST[k][i], LAB[i])); for i in o[1:min(KO, length(o))]; push!(WHO, (k, "O", 0, i)); end; end
ioe = open(SPD * "geo/tw/eval_g$(G).csv", "w"); println(ioe, "x,world,k,J"); DONE = Set{Int}()
function evalset(S)
    S = [i for i in unique(S) if !(i in DONE)]; isempty(S) && return
    for i in S; @printf(ioe, "%s,T,0,%.6f\n", LAB[i], Jof(i, H[:, i])); end
    for nf in NFS, kd in 1:K
        ndF = mk(nf, 6_000_000 + 1000 * parse(Int, G) + kd + (nf == 5 ? 0 : 10_000_000 * nf)); pF = fill(1 / nf, nf)
        for i in S
            x̄ = xfrac(ALLD[i]...); h = _nz_resp!(_nz_build_resp(ndF, x̄; optimizer=GRB1), ndF, pF, apess); h === nothing && error("retailer LP failed (nf=$nf k=$kd $(LAB[i]))")
            @printf(ioe, "%s,%d,%d,%.6f\n", LAB[i], nf, kd, Jof(i, h))
        end
    end
    union!(DONE, S); flush(ioe)
end
evalset([w[4] for w in WHO]); @printf("  evaluation 1 done: %d decisions, %.0f s\n", length(DONE), time() - t0); flush(stdout)
# ---- 3. B*: Belief-only (delta 0.1) around F_B. Adversarial beliefs: move mass 0.1 between the ten total-demand deciles of F_B
CB = Int[]
for k in 1:length(SCEN)
    o = sort(ENUM; by=i -> (VB[i] + CST[k][i], LAB[i])); append!(CB, o[1:min(KB, length(o))])
    for tot in sort(unique(CAP[ENUM])); tot < 500 && continue; idx = [i for i in ENUM if CAP[i] == tot]; push!(CB, idx[argmin([(VB[i] + CST[k][i], LAB[i]) for i in idx])]); end
end
CB = sort(unique(CB)); cl = [ceil(Int, r / (NB / 10)) for r in 1:NB]   # idxB is already in increasing order of total demand
BEL = [(a, b) for a in 1:10 for b in 1:10 if a != b]; NBEL > 0 && (BEL = BEL[round.(Int, range(1, length(BEL); length=NBEL))]); ADVF = Dict(i => 0.0 for i in CB); PC = Dict{Tuple{Int,Vector{Float64}},Float64}()
for (a, b) in BEL
    p = copy(pB); p[cl .== a] .= 0.0; p[cl .== b] .*= 2.0; setpB!(p)
    for i in CB; x̄ = xfrac(ALLD[i]...); hr = round.(solveB(x̄); digits=4); ADVF[i] = max(ADVF[i], get!(() -> cvar(phiB(x̄, hr), pB), PC, (i, hr)) - VB[i]); end
end
setpB!(pB)
BST = [CB[argmin([(VB[i] + CST[k][i] + ADVF[i], LAB[i]) for i in CB])] for k in 1:length(SCEN)]
open(SPD * "geo/tw/adv_g$(G).csv", "w") do io; println(io, "x,cap,VB,advB"); for i in CB; println(io, LAB[i], ",", @sprintf("%.0f,%.4f,%.4f", CAP[i], VB[i], ADVF[i])); end; end
for k in 1:length(SCEN); push!(WHO, (k, "B*", 0, BST[k])); @printf("  B* (c/v %.1f) = %s (N* = %s)\n", SCEN[k][2] / vv, LAB[BST[k]], LAB[NST[k]]); end
@printf("  B* done: %d candidates x %d beliefs, %.0f s\n", length(CB), length(BEL), time() - t0); flush(stdout)
evalset(BST); close(ioe)
open(SPD * "geo/tw/dec_g$(G).csv", "w") do io; println(io, "scen,q0,c,who,seed,x,cap"); for (k, w, s, i) in WHO; println(io, k, ",", SCEN[k][1], ",", SCEN[k][2], ",", w, ",", s, ",", LAB[i], ",", @sprintf("%.0f", CAP[i])); end; end
@printf("TW_DONE g%s: evaluated %d decisions, %d true-cost evaluations, %.0f s\n", G, length(DONE), length(JC), time() - t0)
