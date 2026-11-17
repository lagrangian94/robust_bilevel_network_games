# zsW.jl (2026-11-17) = dwLeanP.jl lines 1-99 + this body. Evaluation on the zero-sum grid (manuscript 6.1 OOS): support = the 10 training scenarios, only probabilities change.
#   draw: alpha = max(gamma (1 + zeta z), 0.01), z ~ N(0, I);  p ~ Dir(alpha);  independently alpha2 (same rule), p2 ~ Dir(alpha2).
#   world K  (leader knows the demand distribution): truth = q̂, retailer belief = p.            cost = store cost + CVaR_q̂[phi(x, h(x, p))]
#   world U  (leader does not):                       truth = p,  retailer belief = rho p + (1 - rho) p2, rho in ZS_RHO.   cost = store cost + CVaR_p[phi(x, h(x, belief))]
#   Decisions are the stored ones of config ZS_CFG (N, B0.1, D0.1, D0.3, C if present); nothing is re-solved.
const G = ENV["FB_G"]; const SS = parse(Int, ENV["ON_SEED"]); const CFGZ = E("ZS_CFG", "dwpl75"); const OUT = E("ZS_OUT", "zs"); const MODELS = Tuple(String.(split(E("ZS_MODELS", "N,B0.1,D0.1,D0.3,C"), ",")))
const GAMS = parse.(Float64, split(E("ZS_GAMMA", "0.1,0.3,0.5,1"), ",")); const ZETS = parse.(Float64, split(E("ZS_ZETA", "0,0.5,1,2,5"), ",")); const MZ = parse(Int, E("ZS_M", "50")); const RHOS = parse.(Float64, split(E("ZS_RHO", "1,0.5,0"), ","))
const SEEDB = parse(Int, E("ZS_SEEDBASE", "9191"))
t0 = time()
SITES = [collect(c) for k in 1:3 for c in Iterators.product(ntuple(_ -> 1:nd.nx, k)...) if issorted(collect(c)) && allunique(collect(c))]
ALLD = [(s, collect(lv)) for s in SITES for lv in Iterators.product(ntuple(_ -> TIERS, length(s))...)]; LAB = [dstr(s, lv) for (s, lv) in ALLD]; nALL = length(ALLD); IDX = Dict(LAB[i] => i for i in 1:nALL)
CAP = [sum(lv) for (s, lv) in ALLD]; const NK = length(SCEN)
DEC = Dict{Tuple{Int,String},Int}()
for l in readlines(SPD * "geo/dw/sel_$(CFGZ)_g$(G)_s$(SS).csv")[2:end]
    f = split(l, ","); f[4] in MODELS || continue; DEC[(parse(Int, f[1]), String(f[4]))] = IDX[f[5]]
end
for k in 1:NK; haskey(DEC, (k, "N")) || error("no stored N for scen $k"); end
EV = sort(unique(collect(values(DEC))))
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
"one draw of the zero-sum scheme: Dir(max(gamma (1 + zeta z), 0.01)); redrawn if the sample is not a probability vector (underflow)"
function draw_dir(rng, γ, ζ)
    for _ in 1:1000
        α = max.(γ .* (1.0 .+ ζ .* randn(rng, S)), 0.01); p = rand(rng, Dirichlet(α))
        (all(isfinite, p) && abs(sum(p) - 1) < 1e-9 && minimum(p) >= 0) && return p ./ sum(p)
    end
    error("draw_dir failed (γ = $γ, ζ = $ζ)")
end
tv(a, b) = 0.5 * sum(abs.(a .- b))
nds = mk(10, SS); qs = nds.q_hat; maximum(abs.(qs .- q)) < 1e-12 || error("q̂ is not uniform"); setps!, solves = resp_setup(nds); phis = phi_setup(nds)
ZC = Dict{Tuple{Int,Vector{Float64}},Vector{Float64}}()
Zof(i, h) = (hr = round.(h; digits=4); get!(() -> phis(xfrac(ALLD[i]...), hr), ZC, (i, hr)))
gi = parse(Int, G); mkpath(SPD * "geo/$(OUT)")
@printf("ZS g%s seed %d: VSM %.4f, decisions of %s (%d distinct), γ = %s, ζ = %s, M = %d, ρ = %s, models built %.0f s\n", G, SS, VSM, CFGZ, length(EV), string(GAMS), string(ZETS), MZ, string(RHOS), time() - t0); flush(stdout)
# ---- q̂ pass: what the leader sees
setps!(qs); Vq = Dict(i => cvar(Zof(i, solves(xfrac(ALLD[i]...))), qs) for i in EV)
open(SPD * "geo/$(OUT)/dec_g$(G)_s$(SS).csv", "w") do io
    println(io, "seed,scen,q0,c,who,x,cap,Vq")
    for k in 1:NK, w in MODELS; haskey(DEC, (k, w)) || continue; i = DEC[(k, w)]; println(io, join([string(SS), string(k), string(SCEN[k][1]), string(SCEN[k][2]), w, LAB[i], @sprintf("%.0f", CAP[i]), @sprintf("%.4f", Vq[i])], ",")); end
end
ioe = open(SPD * "geo/$(OUT)/eval_g$(G)_s$(SS).csv", "w"); iod = open(SPD * "geo/$(OUT)/draw_g$(G)_s$(SS).csv", "w")
println(ioe, "seed,gamma,zeta,m,x,JK," * join(["JU$(ρ)" for ρ in RHOS], ",")); println(iod, "seed,gamma,zeta,m,tv_p_q,tv_p2_q,tv_p_p2")
for (ig, γ) in enumerate(GAMS), (iz, ζ) in enumerate(ZETS)
    rng = MersenneTwister(hash((SEEDB, gi, SS, ig, iz)))
    for m in 1:MZ
        p = draw_dir(rng, γ, ζ); p2 = draw_dir(rng, γ, ζ)
        @printf(iod, "%d,%s,%s,%d,%.4f,%.4f,%.4f\n", SS, string(γ), string(ζ), m, tv(p, qs), tv(p2, qs), tv(p, p2))
        JK = Dict{Int,Float64}(); JU = Dict{Int,Vector{Float64}}(i => zeros(length(RHOS)) for i in EV)
        setps!(p); for i in EV; Z = Zof(i, solves(xfrac(ALLD[i]...))); JK[i] = cvar(Z, qs); for (ir, ρ) in enumerate(RHOS); ρ == 1 && (JU[i][ir] = cvar(Z, p)); end; end
        for (ir, ρ) in enumerate(RHOS)
            ρ == 1 && continue
            setps!(ρ .* p .+ (1 - ρ) .* p2); for i in EV; JU[i][ir] = cvar(Zof(i, solves(xfrac(ALLD[i]...))), p); end
        end
        for i in EV; println(ioe, join([string(SS), string(γ), string(ζ), string(m), LAB[i], @sprintf("%.4f", JK[i]), join([@sprintf("%.4f", v) for v in JU[i]], ",")], ",")); end
    end
    flush(ioe); flush(iod); @printf("  γ %.1f ζ %.1f done %.0f s (cache %d)\n", γ, ζ, time() - t0, length(ZC)); flush(stdout)
end
close(ioe); close(iod)
@printf("ZS_DONE g%s seed %d: %.0f s\n", G, SS, time() - t0)
