# onW.jl (2026-11-16 night) = dwLean.jl lines 1-99 + this body. The truth-unknown world ON THE SUPPORT: the 10 training scenarios are the support,
#   the leader uses the uniform q̂, the true distribution p and the retailer belief are other probability vectors on the same 10 scenarios.
#   Decisions (N, B0.1, D0.1, D0.3, C) are the stored ones (geo/dw/sel_*); only the evaluation changes. No test sample.
#   truth p: TV(p, q̂) = d and |p - q̂|_inf <= 2d/sqrt(S) (inside the model set of radius d), random Dirichlet(1) direction, ON_M draws per d.
#   retailer: knows p (world T), or the frequencies of nf draws from p (worlds 5, 20; ON_K draws each).
#   cost of x = store cost + CVaR_p[phi(x, h(x, belief))].  N*(p) = argmin over the screened decisions with belief = p.
#   Nr = argmin CVaR_q̂[phi(x, h(x, p))] (own demand weights, true orders),  Nd = argmin CVaR_p[phi(x, h(x, q̂))] (true demand weights, orders predicted from q̂).
const G = ENV["FB_G"]; const SS = parse(Int, ENV["ON_SEED"]); const KS = parse(Int, E("TW_KS", "1000")); const MAXD = parse(Int, E("TW_MAXD", "0"))
const DTV = parse.(Float64, split(E("ON_D", "0.1,0.2,0.3"), ",")); const MP = parse(Int, E("ON_M", "5")); const KB = parse(Int, E("ON_K", "20")); const MODELS = ("N", "B0.1", "D0.1", "D0.3", "C")
t0 = time(); cfgof(s) = s <= 3 ? "dw75" : "dwl75"
SITES = [collect(c) for k in 1:3 for c in Iterators.product(ntuple(_ -> 1:nd.nx, k)...) if issorted(collect(c)) && allunique(collect(c))]
ALLD = [(s, collect(lv)) for s in SITES for lv in Iterators.product(ntuple(_ -> TIERS, length(s))...)]; LAB = [dstr(s, lv) for (s, lv) in ALLD]; nALL = length(ALLD); IDX = Dict(LAB[i] => i for i in 1:nALL)
costv(q0, c) = [q0 * length(s) + c * sum(lv) for (s, lv) in ALLD]; CST = [costv(q0, c) for (q0, c) in SCEN]; CAP = [sum(lv) for (s, lv) in ALLD]; const NK = length(SCEN)
# ---- stored decisions: all seeds (for the screen, same screen as splitW.jl) and this seed's model decisions
DECI = Int[]; DEC = Dict{Tuple{Int,String},Int}()
for s in 1:10, l in readlines(SPD * "geo/dw/sel_$(cfgof(s))_g$(G)_s$(s).csv")[2:end]
    f = split(l, ","); f[4] in MODELS || continue
    push!(DECI, IDX[f[5]]); s == SS && (DEC[(parse(Int, f[1]), String(f[4]))] = IDX[f[5]])
end
VS = zeros(nALL, 10)
for s in 1:10
    L = readlines(SPD * "geo/dw/vn_$(cfgof(s))_g$(G)_s$(s).csv")[2:end]; length(L) == nALL || error("vn rows $(length(L)) != $nALL (seed $s)")
    for (i, l) in enumerate(L); f = split(l, ","); f[1] == LAB[i] || error("vn label mismatch seed $s row $i"); VS[i, s] = parse(Float64, f[4]); end
end
vbar = vec(mean(VS; dims=2)); SCRORD = [sortperm(collect(zip(vbar .+ CST[k], LAB))) for k in 1:NK]
ENUM = MAXD > 0 ? sort(unique(vcat(collect(1:MAXD), DECI))) : KS > 0 ? sort(unique(vcat([o[1:min(KS, nALL)] for o in SCRORD]..., DECI))) : collect(1:nALL)
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
"truth at TV distance d from q̂, inside the box 2d/sqrt(S): random Dirichlet(1) direction, redrawn until inside"
function draw_p(rng, d)
    Γ = 2d / sqrt(S)
    for _ in 1:100000
        Q = rand(rng, Dirichlet(ones(S))); dd = 0.5 * sum(abs.(Q .- q)); dd < 1e-6 && continue
        p = q .+ (d / dd) .* (Q .- q); (minimum(p) >= 0 && maximum(abs.(p .- q)) <= Γ + 1e-9) && return p ./ sum(p)
    end
    error("draw_p failed (d = $d)")
end
function draw_b(rng, p, nf); c = zeros(S); cp = cumsum(p); for _ in 1:nf; c[min(S, searchsortedfirst(cp, rand(rng)))] += 1; end; c ./ nf; end
nds = mk(10, SS); qs = nds.q_hat; maximum(abs.(qs .- q)) < 1e-12 || error("q̂ is not uniform"); setps!, solves = resp_setup(nds); phis = phi_setup(nds)
ZC = Dict{Tuple{Int,Vector{Float64}},Vector{Float64}}()
Zof(i, h) = (hr = round.(h; digits=4); get!(() -> phis(xfrac(ALLD[i]...), hr), ZC, (i, hr)))
gi = parse(Int, G); mkpath(SPD * "geo/on"); @printf("ON g%s seed %d: %d decisions screened of %d, d = %s, %d truths per d, %d belief draws, models built %.0f s\n", G, SS, length(ENUM), nALL, string(DTV), MP, KB, time() - t0); flush(stdout)
# ---- pass under q̂: the commitment the leader predicts h_q(x) and the scenario values Z_q(x)
setps!(qs); HQ = Dict{Int,Vector{Float64}}(); ZQ = Dict{Int,Vector{Float64}}()
for i in ENUM; HQ[i] = solves(xfrac(ALLD[i]...)); ZQ[i] = Zof(i, HQ[i]); end
Vss = Dict(i => cvar(ZQ[i], qs) for i in ENUM); dmax = maximum(abs(Vss[i] - VS[i, SS]) for i in ENUM)
for k in 1:NK
    iN = ENUM[argmin([(Vss[i] + CST[k][i], LAB[i]) for i in ENUM])]; i0 = DEC[(k, "N")]
    iN == i0 || @printf("  NOTE scen %d: recomputed N %s (%.4f) differs from stored N %s (%.4f)\n", k, LAB[iN], Vss[iN] + CST[k][iN], LAB[i0], Vss[i0] + CST[k][i0])
end
@printf("  q̂ pass done %.0f s, max |V_ss - stored| = %.5f\n", time() - t0, dmax); flush(stdout)
iod = open(SPD * "geo/on/dec_g$(G)_s$(SS).csv", "w"); ioe = open(SPD * "geo/on/eval_g$(G)_s$(SS).csv", "w"); iop = open(SPD * "geo/on/p_g$(G)_s$(SS).csv", "w")
println(iod, "seed,d,m,scen,q0,c,who,x,cap,Vss,Vrs,Vds,Vpp,hB_pred,hB_true,hB_l1"); println(ioe, "seed,d,m,x,world,k,J"); println(iop, "seed,d,m," * join(["p$(s)" for s in 1:S], ","))
for d in DTV
    rp = MersenneTwister(hash((7171, gi, SS, round(Int, 100d))))
    for m in 1:MP
        p = draw_p(rp, d); println(iop, SS, ",", d, ",", m, ",", join([@sprintf("%.6f", x) for x in p], ","))
        # ---- pass under the truth p: commitment of a retailer who knows the truth h_p(x), scenario values Z_p(x)
        setps!(p); HP = Dict{Int,Vector{Float64}}(); ZP = Dict{Int,Vector{Float64}}()
        for i in ENUM; HP[i] = solves(xfrac(ALLD[i]...)); ZP[i] = Zof(i, HP[i]); end
        Vpp = Dict(i => cvar(ZP[i], p) for i in ENUM); Vrs = Dict(i => cvar(ZP[i], qs) for i in ENUM); Vds = Dict(i => cvar(ZQ[i], p) for i in ENUM)
        EV = Int[]
        for k in 1:NK
            iS = ENUM[argmin([(Vpp[i] + CST[k][i], LAB[i]) for i in ENUM])]; iR = ENUM[argmin([(Vrs[i] + CST[k][i], LAB[i]) for i in ENUM])]; iD = ENUM[argmin([(Vds[i] + CST[k][i], LAB[i]) for i in ENUM])]
            who = vcat([(w, DEC[(k, w)]) for w in MODELS if haskey(DEC, (k, w))], [("N*", iS), ("Nr", iR), ("Nd", iD)])
            for (w, i) in who
                println(iod, join([string(SS), string(d), string(m), string(k), string(SCEN[k][1]), string(SCEN[k][2]), w, LAB[i], @sprintf("%.0f", CAP[i]), @sprintf("%.4f,%.4f,%.4f,%.4f", Vss[i], Vrs[i], Vds[i], Vpp[i]),
                    @sprintf("%.2f,%.2f,%.2f", sum(HQ[i][nA+1:end]), sum(HP[i][nA+1:end]), sum(abs.(HQ[i][nA+1:end] .- HP[i][nA+1:end])))], ","))
                push!(EV, i)
            end
        end
        EV = sort(unique(EV))
        for i in EV; @printf(ioe, "%d,%s,%d,%s,T,0,%.6f\n", SS, string(d), m, LAB[i], Vpp[i]); end
        # ---- retailer with an own sample of nf draws from p
        for nf in NFS
            rb = MersenneTwister(hash((7272, gi, SS, round(Int, 100d), m, nf)))
            for kb in 1:KB
                b = draw_b(rb, p, nf); setps!(b)
                for i in EV; @printf(ioe, "%d,%s,%d,%s,%d,%d,%.6f\n", SS, string(d), m, LAB[i], nf, kb, cvar(Zof(i, solves(xfrac(ALLD[i]...))), p)); end
            end
        end
        flush(iod); flush(ioe); flush(iop)
        @printf("  d %.1f truth %d done %.0f s (decisions evaluated %d)\n", d, m, time() - t0, length(EV)); flush(stdout)
    end
end
close(iod); close(ioe); close(iop)
@printf("ON_DONE g%s seed %d: %.0f s\n", G, SS, time() - t0)
