# -*- coding: utf-8 -*-
# Is the extra order a natural insurance against a belief error?  (re-check of the 2026-11-15 hypothesis)  Input geo/mechc (config dwps3c, cost table scen 2, 150 runs).
# Truth stays q̂.  Belief perturbations = the 90 pair moves.  Keep those where the leader's commitment falls by more than 1 unit.
# Scenarios are split by what happens under belief q̂:  "floor" = only the commitment sells (Z = -v h),  "other" = some extra order.
# Reported per decision: commitment lost, change of the mean cost, of CVaR_0.4, of the cost in floor / other scenarios, in % of Nominal's in-sample cost;
# and the recovered share = 1 - (change of scenario cost) / (v * commitment lost): 0 = nothing recovered, 1 = fully recovered, > 1 = the leader gains.
import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
WHO = ("N", "B0.1", "D0.1", "C"); BETA = 0.4; S = 10; q = np.full(S, 0.1)
def cvar(Z, p):
    o = np.argsort(-Z); cap = p / (1 - BETA); rest = 1.0; v = 0.0
    for s in o:
        m = min(cap[s], rest); v += m * Z[s]; rest -= m
        if rest <= 1e-12: break
    return v
A = {w: [] for w in WHO}; gg = []
for g in range(1, 16):
    for s in range(1, 11):
        dec = {r["who"]: r for r in csv.DictReader(open(O + "mechc/dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
        q0, c = float(dec["N"]["q0"]), float(dec["N"]["c"]); M = {}; v = None
        for r in csv.DictReader(open(O + "mechc/mech_g%d_s%d.csv" % (g, s), encoding="utf-8")):
            v = float(r["v"]); M.setdefault(r["x"], {})[(r["bel"], int(r["i"]), int(r["j"]))] = (float(r["hB"]), np.array([float(r["Z%d" % k]) for k in range(1, S + 1)]), float(r["cap"]), int(r["nsite"]))
        hN, ZN, capN, nsN = M[dec["N"]["x"]][("q", 0, 0)]; L = abs(cvar(ZN, q) + q0 * nsN + c * capN); gg.append(g)
        for w in WHO:
            m = M[dec[w]["x"]]; h0, Z0, cap, ns = m[("q", 0, 0)]; fl = np.abs(Z0 + v * h0) < 1e-3 * v * h0; rows = []
            for (b, i, j), (h, Z, _, _) in m.items():
                if b != "pair" or h0 - h <= 1.0: continue
                dh = h0 - h; dZ = Z - Z0
                rows.append((dh, 100 * dZ.mean() / L, 100 * (cvar(Z, q) - cvar(Z0, q)) / L, 100 * dZ[fl].mean() / L if fl.any() else np.nan, 100 * dZ[~fl].mean() / L if (~fl).any() else np.nan,
                             1 - dZ[fl].mean() / (v * dh) if fl.any() else np.nan, 1 - dZ[~fl].mean() / (v * dh) if (~fl).any() else np.nan, 1 - dZ.mean() / (v * dh), 100 * v * dh / L))
            if rows: A[w].append(np.r_[np.nanmean(np.array(rows), axis=0), fl.sum(), len(rows), cap - h0])
print("belief moves so that the commitment falls (truth stays q̂); means over runs")
print("%-5s | runs | moves | floor scen | slack | commit lost | = %% of cost | d mean | d CVaR | d floor scen | d other scen | recovered: floor | other | mean" % "")
for w in WHO:
    a = np.array(A[w]); m = np.nanmean(a, axis=0)
    print("%-5s | %4d | %5.1f | %10.1f | %5.0f | %11.1f | %10.2f | %+6.2f | %+6.2f | %+12.2f | %+12.2f | %16.2f | %5.2f | %4.2f" % (w, len(a), m[10], m[9], m[11], m[0], m[8], m[1], m[2], m[3], m[4], m[5], m[6], m[7]))
