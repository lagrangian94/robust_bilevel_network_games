# dtrackW.jl (2026-11-17) = dwLeanB.jl lines 1-117 + this body.  "Data-DR with the belief following the truth" against Data-DR, on the same set U(r) = TV(q̂, r) ∩ box (SC_BOXT).
#   V_D(x) = max_{a in U(r)} CVaR_a[phi(x, h(x, q̂))]   (Data-DR: the retailer keeps believing q̂; exact LP)
#   V_T(x) = max_{a in U(r)} CVaR_a[phi(x, h(x, a))]    (D-track: the retailer believes the distribution the adversary picks; nonconvex, point search = lower bound)
#   candidates: top TR_KC decisions by V0 + cost and by VD(0.1, 0.2, 0.3) + cost for each cost table, from the stored enumeration vn_<TR_VN>_g_s.csv
const RT = parse.(Float64, split(E("TR_R", "0.1,0.2"), ",")); const KC = parse(Int, E("TR_KC", "40")); const NPT = parse(Int, E("TR_NPT", "40")); const NLS = parse(Int, E("TR_NLS", "20"))
const OUTT = E("TR_OUT", "dtr"); const VNCFG = E("TR_VN", "dwpl75"); const GG = ENV["FB_G"]
t0 = time()
SITES = [collect(c) for k in 1:3 for c in Iterators.product(ntuple(_ -> 1:nd.nx, k)...) if issorted(collect(c)) && allunique(collect(c))]
ALLD = [(s, collect(lv)) for s in SITES for lv in Iterators.product(ntuple(_ -> TIERS, length(s))...)]; LAB = [dstr(s, lv) for (s, lv) in ALLD]; nALL = length(ALLD)
costv(q0, c) = [q0 * length(s) + c * sum(lv) for (s, lv) in ALLD]; CST = [costv(q0, c) for (q0, c) in SCEN]; CAP = [sum(lv) for (s, lv) in ALLD]
L = readlines(SPD * "geo/dw/vn_$(VNCFG)_g$(GG)_s$(SEED).csv")[2:end]; length(L) == nALL || error("vn rows $(length(L)) != $nALL")
VS = zeros(nALL, 4); for (i, l) in enumerate(L); f = split(l, ","); f[1] == LAB[i] || error("vn label mismatch row $i"); VS[i, :] = parse.(Float64, f[4:7]); end
ordby(v) = sortperm(collect(zip(v, LAB)))
cand = sort(unique(vcat([ordby(VS[:, c] .+ CST[k])[1:KC] for k in 1:length(SCEN), c in 1:4]...))); nc = length(cand)
"random extreme points of U(r): mass r taken from some scenarios (at most the box width each) and given to others"
function set_pts(r, rng, n)
    Γ = boxw(r, BOXT); pts = Vector{Vector{Float64}}()
    for _ in 1:n
        p = randperm(rng, S); d = copy(q); rest = r; k = 1
        while rest > 1e-12 && k <= S; s = p[k]; mm = min(rest, Γ[s], d[s]); d[s] -= mm; rest -= mm; k += 1; end
        kd = k - 1; rest = r; k = S
        while rest > 1e-12 && k > kd; s = p[k]; mm = min(rest, Γ[s]); d[s] += mm; rest -= mm; k -= 1; end
        rest <= 1e-9 && push!(pts, d)
    end
    return unique(x -> round.(x; digits=8), pts)
end
inset(d, r) = (Γ = boxw(r, BOXT); minimum(d) >= -1e-12 && all(abs.(d .- q) .<= Γ .+ 1e-9) && 0.5 * sum(abs.(d .- q)) <= r + 1e-9)
PTS = [set_pts(r, MersenneTwister(hash((9797, SEED, round(Int, 1000r)))), NPT) for r in RT]
@printf("DTRACK g%s seed %d: VSM %.4f, box t = %.4f, radii %s, %d candidates of %d, points %s\n", GG, SEED, VSM, BOXT, string(RT), nc, nALL, string(length.(PTS))); flush(stdout)
V0 = zeros(nc); VD = zeros(nc, length(RT)); VT = zeros(nc, length(RT)); AT = Matrix{Vector{Float64}}(undef, nc, length(RT)); PH = Vector{Any}(undef, nc)
for (ci, i) in enumerate(cand)
    phi = evalx(xfrac(ALLD[i]...)); PH[ci] = phi; Zq = phi(q); V0[ci] = cvar(Zq, q)
    for (ir, r) in enumerate(RT)
        vd, aD = worst(Zq, q, Inf, r); VD[ci, ir] = vd
        best = cvar(phi(aD), aD); ba = aD; a = aD
        for _ in 1:3   # the adversary answers the commitment of a retailer who believes a
            _, a2 = worst(phi(a), q, Inf, r); v2 = cvar(phi(a2), a2); v2 > best && (best = v2; ba = a2); maximum(abs.(a2 .- a)) < 1e-9 && break; a = a2
        end
        for d in PTS[ir]; v = cvar(phi(d), d); v > best && (best = v; ba = d); end
        VT[ci, ir] = best; AT[ci, ir] = ba
    end
end
@printf("  point search done %.0f s\n", time() - t0); flush(stdout)
# ---- local moves (one pair of scenarios at a time) for the decisions that can become the D-track optimum
function improve!(ci, ir)
    r = RT[ir]; Γ = boxw(r, BOXT); phi = PH[ci]; imp = true; rounds = 0
    while imp && rounds < 2
        imp = false; rounds += 1; base = AT[ci, ir]
        for i1 in 1:S, j1 in 1:S
            i1 == j1 && continue
            for mm in (Γ[i1], Γ[i1] / 2)
                d = copy(base); m = min(mm, d[i1]); m <= 1e-9 && continue; d[i1] -= m; d[j1] += m
                inset(d, r) || continue
                v = cvar(phi(d), d); v > VT[ci, ir] + 1e-7 && (VT[ci, ir] = v; AT[ci, ir] = d; imp = true)
            end
        end
    end
end
for (k, _) in enumerate(SCEN), ir in 1:length(RT)
    for ci in sortperm(VT[:, ir] .+ CST[k][cand])[1:min(NLS, nc)]; improve!(ci, ir); end
end
@printf("  local moves done %.0f s\n", time() - t0); flush(stdout)
mkpath(SPD * "geo/$(OUTT)")
open(SPD * "geo/$(OUTT)/dtr_$(TAG).csv", "w") do io
    println(io, "x,n,cap,V0," * join(["VD$(r)" for r in RT], ",") * "," * join(["VT$(r)" for r in RT], ","))
    for (ci, i) in enumerate(cand); println(io, LAB[i], ",", length(ALLD[i][1]), ",", @sprintf("%.0f", CAP[i]), ",", @sprintf("%.4f", V0[ci]), ",", join([@sprintf("%.4f", v) for v in VD[ci, :]], ","), ",", join([@sprintf("%.4f", v) for v in VT[ci, :]], ",")); end
end
open(SPD * "geo/$(OUTT)/sel_$(TAG).csv", "w") do io
    println(io, "scen,q0,c,model,x,cap,V0,VD,VT")
    for (k, (q0, c)) in enumerate(SCEN)
        cs = CST[k][cand]; pick(v) = sortperm(collect(zip(v .+ cs, LAB[cand])))[1]
        iN = pick(V0); println(io, join([string(k), string(q0), string(c), "N", LAB[cand[iN]], @sprintf("%.0f", CAP[cand[iN]]), @sprintf("%.4f", V0[iN]), "", ""], ","))
        for (ir, r) in enumerate(RT)
            iD = pick(VD[:, ir]); iT = pick(VT[:, ir])
            println(io, join([string(k), string(q0), string(c), "D$(r)", LAB[cand[iD]], @sprintf("%.0f", CAP[cand[iD]]), @sprintf("%.4f", V0[iD]), @sprintf("%.4f", VD[iD, ir]), @sprintf("%.4f", VT[iD, ir])], ","))
            println(io, join([string(k), string(q0), string(c), "T$(r)", LAB[cand[iT]], @sprintf("%.0f", CAP[cand[iT]]), @sprintf("%.4f", V0[iT]), @sprintf("%.4f", VD[iT, ir]), @sprintf("%.4f", VT[iT, ir])], ","))
        end
    end
end
@printf("DTR_DONE g%s seed %d: %.0f s\n", GG, SEED, time() - t0)
