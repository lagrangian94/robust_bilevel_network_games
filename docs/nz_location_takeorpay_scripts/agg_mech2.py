# -*- coding: utf-8 -*-
# Why does Coupled pay the same mean as Data-DR but cut the tail much more?  Decisions of config dwps3c (geo/mechc), cost table scen 2, 150 runs.
# Cost of a decision when the truth is a and the retailer believes d:  store cost + CVaR_a[Z(d)].  Perturbations: all the probability of scenario i moves to scenario j (90 of them).
# Direction: "low" = j has lower total demand than i (the distribution moves toward low demand), "high" = the opposite.  Values = decision minus Nominal, % of Nominal's in-sample cost.
import csv, os, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
WHO = ("N", "B0.1", "D0.1", "C"); BETA = 0.4; S = 10; q = np.full(S, 0.1)
def cvar(Z, p):
    o = np.argsort(-Z); cap = p / (1 - BETA); rest = 1.0; v = 0.0
    for s in o:
        m = min(cap[s], rest); v += m * Z[s]; rest -= m
        if rest <= 1e-12: break
    return v
def se(v, g):
    v = np.asarray(v, float); g = np.asarray(g); m = v.mean(); G = np.unique(g); e = np.array([(v[g == k] - m).sum() for k in G]); return m, np.sqrt((e ** 2).sum() * len(G) / (len(G) - 1)) / len(v)
RES = {}; gg = []; DET = {w: [] for w in WHO}
for g in range(1, 16):
    T = {}
    for r in csv.reader(open(O + "mech/xi_g%d.csv" % g)):
        if r[1] == "tot": T[int(r[0])] = np.array([float(x) for x in r[2:]])
    for s in range(1, 11):
        dec = {r["who"]: r for r in csv.DictReader(open(O + "mechc/dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
        q0, c = float(dec["N"]["q0"]), float(dec["N"]["c"]); M = {}
        for r in csv.DictReader(open(O + "mechc/mech_g%d_s%d.csv" % (g, s), encoding="utf-8")):
            M.setdefault(r["x"], {})[(r["bel"], int(r["i"]), int(r["j"]))] = (float(r["hB"]), np.array([float(r["Z%d" % k]) for k in range(1, S + 1)]), float(r["cap"]), int(r["nsite"]))
        t = T[s]; gg.append(g); cost = {}; Zq = {}
        for w in WHO:
            m = M[dec[w]["x"]]; hB, Z, cap, ns = m[("q", 0, 0)]; cost[w] = q0 * ns + c * cap; Zq[w] = Z
        L = abs(cvar(Zq["N"], q) + cost["N"]); row = {}
        for w in WHO:
            m = M[dec[w]["x"]]; acc = {}
            for i in range(1, S + 1):
                for j in range(1, S + 1):
                    if i == j: continue
                    a = q.copy(); a[j - 1] += a[i - 1]; a[i - 1] = 0.0; h, Z = m[("pair", i, j)][:2]; dr = "low" if t[j - 1] < t[i - 1] else "high"
                    for mode, v in (("data", cvar(Zq[w], a)), ("belief", cvar(Z, q)), ("both", cvar(Z, a))): acc.setdefault((mode, dr), []).append(v + cost[w])
                    acc.setdefault(("commit", dr), []).append(h - m[("q", 0, 0)][0])
            row[w] = {k: np.mean(v) for k, v in acc.items()}; row[w][("base", "-")] = cvar(Zq[w], q) + cost[w]
        for w in WHO[1:]:
            for k in row[w]:
                if k[0] == "commit": RES.setdefault((w, k), []).append(row[w][k])
                else: RES.setdefault((w, k), []).append(100 * (row[w][k] - row["N"][k]) / L)
        for k in row["N"]:
            if k[0] == "commit": RES.setdefault(("N", k), []).append(row["N"][k])
print("decision minus Nominal (%% of Nominal's in-sample cost), mean over %d runs ± SE (clustered by geography)" % len(gg))
print("%-6s | at q̂        | truth moves low   | truth moves high  | belief moves low  | belief moves high | both low          | both high" % "")
for w in WHO[1:]:
    f = lambda k: "%+5.2f ± %.2f" % se(RES[(w, k)], gg)
    print("%-6s | %s | %s     | %s     | %s     | %s     | %s     | %s" % (w, f(("base", "-")), f(("data", "low")), f(("data", "high")), f(("belief", "low")), f(("belief", "high")), f(("both", "low")), f(("both", "high"))))
print("change of the commitment when the retailer's belief moves (units): low / high")
for w in WHO: print("   %-5s %+6.1f / %+6.1f" % (w, np.mean(RES[(w, ("commit", "low"))]), np.mean(RES[(w, ("commit", "high"))])))
