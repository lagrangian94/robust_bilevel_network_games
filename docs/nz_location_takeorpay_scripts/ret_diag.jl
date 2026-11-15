# ret_diag (2026-11-15): retailer-alone diagnosis. Rebuilds the retailer's two-stage program from the raw data (distances, demand, capacities),
# independent of the NZData matrices. Checks: reservation profile, reserved vs spot volumes, VSS / EVPI, ties in the first stage, ties in the second stage.
# env: C0H_SEED, RD_X = "{7:300;8:300;13:200}" (B site index within B : capacity), SC_FR, RD_W (reservation cap), RD_TAG
root = "C:/Users/user/robust_bilevel_network_games"
const SPD = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Distributions, Statistics
const GRB_ENV = Gurobi.Env()
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(SPD * "geo/nz_data_geo2.jl"); include(SPD * "geo/fac_builder_geo2.jl")
E(k, d) = get(ENV, k, d)
SEED = parse(Int, ENV["C0H_SEED"]); P = 800.0; FR = parse(Float64, E("SC_FR", "0.25")); F = FR * P; W = parse(Float64, E("RD_W", "300")); V = 5.0; TAG = E("RD_TAG", "x")
nd = make_sgb15_fac(; S=10, seed=SEED, dem=:region, p=P, f=F, eps_hat=0.0, eps_tilde=0.0, beta=0.4)
nA = nd.meta[:nA]; nst = nd.meta[:nst]; ncu = nd.meta[:ncu]; dist = nd.meta[:dist]; xi = nd.meta[:xi]; bA = nd.meta[:bA]; S = 10; q = fill(0.1, S)
pr = [split(t, ":") for t in split(strip(ENV["RD_X"], ['{', '}']), ";")]; cap = vcat(fill(bA, nA), zeros(nst - nA)); for (i, b) in pr; cap[nA+parse(Int, i)] = parse(Float64, b); end
OPEN = [i for i in 1:nst if cap[i] > 0]; isB(i) = i > nA
"retailer two-stage program on scenarios XI (ncu x K) with probabilities pr; hfix fixes the reservation"
function build(XI, prb; hfix=nothing)
    K = size(XI, 2); m = Model(GRB1)
    @variable(m, 0 <= h[i in OPEN] <= min(W, cap[i])); @variable(m, yR[i in OPEN, 1:ncu, 1:K] >= 0); @variable(m, yS[i in OPEN, 1:ncu, 1:K] >= 0)
    @constraint(m, [j=1:ncu, k=1:K], sum(yR[i, j, k] + yS[i, j, k] for i in OPEN) >= XI[j, k])
    @constraint(m, [i in OPEN, k=1:K], sum(yR[i, j, k] for j in 1:ncu) <= h[i])
    @constraint(m, [i in OPEN, k=1:K], sum(yR[i, j, k] + yS[i, j, k] for j in 1:ncu) <= cap[i])
    hfix === nothing || for i in OPEN; fix(h[i], hfix[i]; force=true); end
    fee = F * sum(h); tr = sum(prb[k] * dist[i, j] * (yR[i, j, k] + yS[i, j, k]) for i in OPEN, j in 1:ncu, k in 1:K); prem = P * sum(prb[k] * yS[i, j, k] for i in OPEN, j in 1:ncu, k in 1:K)
    @objective(m, Min, fee + tr + prem)
    return m, h, yR, yS, fee, tr, prem
end
solve!(m) = (optimize!(m); termination_status(m) == MOI.OPTIMAL || error("retailer LP: $(termination_status(m))"); objective_value(m))
m, h, yR, yS, fee, tr, prem = build(xi, q); RP = solve!(m); hv = Dict(i => value(h[i]) for i in OPEN)
fv, tv, pv = value(fee), value(tr), value(prem)
# ties in the first stage: range of total B reservation over the retailer's optimal set
hB = sum(h[i] for i in OPEN if isB(i)); c1 = @constraint(m, fee + tr + prem <= RP + 1e-7 * abs(RP))
@objective(m, Min, hB); optimize!(m); hBlo = value(hB); hpess = Dict(i => value(h[i]) for i in OPEN)
@objective(m, Max, hB); optimize!(m); hBhi = value(hB)
# wait-and-see and expected-value solutions
WS = sum(q[s] * solve!(build(xi[:, s:s], [1.0])[1]) for s in 1:S)
mev = build(reshape(xi * q, ncu, 1), [1.0]); solve!(mev[1]); hev = Dict(i => value(mev[2][i]) for i in OPEN); EEV = solve!(build(xi, q; hfix=hev)[1])
# second stage at the pessimistic reservation: volumes and the range of B spot sales over optimal recourse plans
tot = vec(sum(xi; dims=1)); RA = zeros(S); RBv = zeros(S); SA = zeros(S); SBlo = zeros(S); SBhi = zeros(S)
for s in 1:S
    ms, _, yRs, ySs, _, _, _ = build(xi[:, s:s], [1.0]; hfix=hpess); Q = solve!(ms)
    sb = sum(ySs[i, j, 1] for i in OPEN, j in 1:ncu if isB(i)); @constraint(ms, objective_function(ms) <= Q + 1e-7 * abs(Q))
    @objective(ms, Min, sb); optimize!(ms); SBlo[s] = value(sb)
    RA[s] = sum(value(yRs[i, j, 1]) for i in OPEN, j in 1:ncu if !isB(i)); RBv[s] = sum(value(yRs[i, j, 1]) for i in OPEN, j in 1:ncu if isB(i)); SA[s] = sum(value(ySs[i, j, 1]) for i in OPEN, j in 1:ncu if !isB(i))
    @objective(ms, Max, sb); optimize!(ms); SBhi[s] = value(sb)
end
hA = sum(hpess[i] for i in OPEN if !isB(i)); hBp = sum(hpess[i] for i in OPEN if isB(i)); capB = sum(cap[i] for i in OPEN if isB(i))
st(i) = (x = hpess[i]; x < 0.5 ? "zero" : x > min(W, cap[i]) - 0.5 ? (cap[i] < W ? "at-capacity" : "at-cap-w") : "interior")
@printf("RD %s seed %d f/p %.2f w %.0f x %s\n", TAG, SEED, FR, W, ENV["RD_X"])
@printf("  demand total: min %.0f mean %.0f max %.0f | capacity A %.0f x %d, B %.0f\n", minimum(tot), dot(q, tot), maximum(tot), bA, nA, capB)
println("  reservation: ", join([@sprintf("%s%d %.0f/%.0f (%s)", isB(i) ? "B" : "A", isB(i) ? i - nA : i, hpess[i], min(W, cap[i]), st(i)) for i in OPEN], ", "))
@printf("  retailer cost %.0f = fee %.0f%% + transport %.0f%% + spot premium %.0f%% | VSS %.2f%% | EVPI %.2f%%\n", RP, 100fv / RP, 100tv / RP, 100pv / RP, 100 * (EEV - RP) / RP, 100 * (RP - WS) / RP)
@printf("  first-stage ties: total B reservation between %.1f and %.1f\n", hBlo, hBhi)
@printf("  mean volumes: A reserved-used %.0f (of %.0f), A spot %.0f | B reserved-used %.0f (of %.0f), B spot %.0f..%.0f | spot share of demand %.1f%%\n", dot(q, RA), hA, dot(q, SA), dot(q, RBv), hBp, dot(q, SBlo), dot(q, SBhi), 100 * (dot(q, SA) + dot(q, SBlo)) / dot(q, tot))
println("  per scenario (sorted by total demand): demand | B unused reservation | B spot (min..max over optimal plans) | leader sales")
for s in sortperm(tot); @printf("    %5.0f | %5.0f | %5.0f..%-5.0f | %5.0f\n", tot[s], hBp - RBv[s], SBlo[s], SBhi[s], hBp + SBlo[s]); end
sal = hBp .+ SBlo; @printf("  leader sales: min %.0f mean %.0f max %.0f | scenarios with B spot %d/10 | second-stage tie width mean %.1f\n", minimum(sal), dot(q, sal), maximum(sal), count(>(0.5), SBlo), dot(q, SBhi .- SBlo))
println("RD_DONE")
