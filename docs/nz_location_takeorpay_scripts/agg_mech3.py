# -*- coding: utf-8 -*-
# Level of the cost in the worst 10 of the 90 pair-move perturbations (store cost + CVaR_a[Z(d)]), decision minus Nominal, % of Nominal's in-sample cost.  geo/mechc, scen 2, 150 runs.
import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
WHO = ("N", "B0.1", "B0.2", "D0.1", "D0.2", "C"); BETA = 0.4; S = 10; q = np.full(S, 0.1)
def cvar(Z, p):
    o = np.argsort(-Z); cap = p / (1 - BETA); rest = 1.0; v = 0.0
    for s in o:
        m = min(cap[s], rest); v += m * Z[s]; rest -= m
        if rest <= 1e-12: break
    return v
def se(v, g):
    v = np.asarray(v, float); g = np.asarray(g); m = v.mean(); G = np.unique(g); e = np.array([(v[g == k] - m).sum() for k in G]); return m, np.sqrt((e ** 2).sum() * len(G) / (len(G) - 1)) / len(v)
R = {}; gg = []
for g in range(1, 16):
    for s in range(1, 11):
        dec = {r["who"]: r for r in csv.DictReader(open(O + "mechc/dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
        q0, c = float(dec["N"]["q0"]), float(dec["N"]["c"]); M = {}
        for r in csv.DictReader(open(O + "mechc/mech_g%d_s%d.csv" % (g, s), encoding="utf-8")):
            M.setdefault(r["x"], {})[(r["bel"], int(r["i"]), int(r["j"]))] = (np.array([float(r["Z%d" % k]) for k in range(1, S + 1)]), float(r["cap"]), int(r["nsite"]))
        gg.append(g); lev = {}
        for w in WHO:
            m = M[dec[w]["x"]]; Zq, cap, ns = m[("q", 0, 0)]; cost = q0 * ns + c * cap; D, B, T = [], [], []
            for i in range(1, S + 1):
                for j in range(1, S + 1):
                    if i == j: continue
                    a = q.copy(); a[j - 1] += a[i - 1]; a[i - 1] = 0.0; Z = m[("pair", i, j)][0]
                    D.append(cvar(Zq, a) + cost); B.append(cvar(Z, q) + cost); T.append(cvar(Z, a) + cost)
            lev[w] = (cvar(Zq, q) + cost, np.sort(D)[-10:].mean(), np.sort(B)[-10:].mean(), np.sort(T)[-10:].mean(), np.mean(T))
        L = abs(lev["N"][0])
        for w in WHO[1:]: R.setdefault(w, []).append([100 * (lev[w][k] - lev["N"][k]) / L for k in range(5)])
print("decision minus Nominal, %% of Nominal's in-sample cost, %d runs" % len(gg))
print("%-6s | at q̂ (in-sample)  | worst 10: truth only | worst 10: belief only | worst 10: both     | mean of the 90 (both)" % "")
for w in WHO[1:]:
    A = np.array(R[w]); print("%-6s | " % w + " | ".join("%+5.2f ± %.2f      " % se(A[:, k], gg) for k in range(5)))
