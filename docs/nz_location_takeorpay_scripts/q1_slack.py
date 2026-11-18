# -*- coding: utf-8 -*-
# H76 a, b.  Why does Coupled cut the slack capacity less than Data-DR per changed decision?   config dwps3c, cost table scen 2 (q0 150, c 2.0), 150 runs.
#   (a) runs split by what Data-DR(0.1) and Coupled do; slack change of each (commitment from geo/mechc at belief q̂).
#   (b) candidate values geo/dw/cp_dwps3_g_s.csv (x, n, cap, V0, VD, VC):  objective of a model = V + store cost.
#       in runs where C != D0.1: Coupled's objective of C's decision minus that of D's decision, split into nominal cost, data damage (VD - V0), belief add-on (VC - VD).
#       candidate pairs with the same sites and different sizes: change of each part per 100 units of capacity.
import csv, os, collections, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
def par(x): return {int(a): float(b) for a, b in (p.split(":") for p in x.strip("{}").split(";") if p)}
def se(v, g):
    v = np.asarray(v, float); g = np.asarray(g); m = v.mean(); G = np.unique(g)
    if len(G) < 2: return m, float("nan")
    e = np.array([(v[g == k] - m).sum() for k in G]); return m, np.sqrt((e ** 2).sum() * len(G) / (len(G) - 1)) / len(v)
CL = collections.defaultdict(list); DEC = collections.defaultdict(list); PAIR = []; miss = 0
for g in range(1, 16):
    for s in range(1, 11):
        dec = {r["who"]: r for r in csv.DictReader(open(O + "mechc/dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
        q0, c = float(dec["N"]["q0"]), float(dec["N"]["c"]); H = {}
        for r in csv.DictReader(open(O + "mechc/mech_g%d_s%d.csv" % (g, s), encoding="utf-8")):
            if r["bel"] == "q": H[r["x"]] = float(r["hB"])
        X = {w: dec[w]["x"] for w in ("N", "B0.1", "D0.1", "C")}; cap = lambda x: sum(par(x).values()); sl = lambda x: cap(x) - H[x]
        cls = ("D=N" if X["D0.1"] == X["N"] else "D!=N") + ", " + ("C=N" if X["C"] == X["N"] else "C=D" if X["C"] == X["D0.1"] else "C=B" if X["C"] == X["B0.1"] else "C other")
        CL[cls].append((g, sl(X["D0.1"]) - sl(X["N"]), sl(X["C"]) - sl(X["N"]), H[X["D0.1"]] - H[X["N"]], H[X["C"]] - H[X["N"]], cap(X["D0.1"]) - cap(X["N"]), cap(X["C"]) - cap(X["N"])))
        V = {r["x"]: (int(r["n"]), float(r["cap"]), float(r["V0"]), float(r["VD"]), float(r["VC"])) for r in csv.DictReader(open(O + "dw/cp_dwps3_g%d_s%d.csv" % (g, s), encoding="utf-8"))}
        cost = lambda x: q0 * V[x][0] + c * V[x][1]; L = abs(V[X["N"]][2] + cost(X["N"])) if X["N"] in V else None
        if L is None or X["D0.1"] not in V or X["C"] not in V: miss += 1; continue
        part = lambda x: np.array([V[x][2] + cost(x), V[x][3] - V[x][2], V[x][4] - V[x][3]]) * 100 / L     # nominal objective, data damage, belief add-on
        if X["C"] != X["D0.1"]: DEC["C - D0.1 (runs C != D)"].append((g, *(part(X["C"]) - part(X["D0.1"]))))
        if X["D0.1"] != X["N"]: DEC["D0.1 - N (runs D != N)"].append((g, *(part(X["D0.1"]) - part(X["N"]))))
        if X["C"] != X["N"]: DEC["C - N (runs C != N)"].append((g, *(part(X["C"]) - part(X["N"]))))
        if X["B0.1"] in V and X["C"] != X["B0.1"]: DEC["C - B0.1 (runs C != B)"].append((g, *(part(X["C"]) - part(X["B0.1"]))))
        K = list(V)
        for a in range(len(K)):
            for b in range(a + 1, len(K)):
                xa, xb = par(K[a]), par(K[b])
                if set(xa) != set(xb) or V[K[a]][1] == V[K[b]][1]: continue
                hi, lo = (K[a], K[b]) if V[K[a]][1] > V[K[b]][1] else (K[b], K[a]); dc = V[hi][1] - V[lo][1]
                PAIR.append((g, dc, *((part(lo) - part(hi)) * 100 / dc)))     # effect of cutting 100 units of capacity at the same sites
print("candidate files where N, D0.1 or C is missing: %d" % miss)
print("\n(a) runs by what Data-DR(0.1) and Coupled do.  changes vs Nominal (units)")
print("%-16s | runs | slack: D0.1    C   | commitment: D0.1   C  | capacity: D0.1    C" % "class")
tot = np.zeros(6); n = 0
for k in sorted(CL):
    a = np.array([r[1:] for r in CL[k]]); m = a.mean(0); tot += a.sum(0); n += len(a)
    print("%-16s | %4d | %+6.0f %+6.0f      | %+6.0f %+6.0f          | %+6.0f %+6.0f" % (k, len(a), m[0], m[1], m[2], m[3], m[4], m[5]))
print("%-16s | %4d | %+6.0f %+6.0f      | %+6.0f %+6.0f          | %+6.0f %+6.0f" % ("all runs", n, *(tot / n)[[0, 1, 2, 3, 4, 5]]))
a = np.array([r[1:] for k in CL if k.startswith("D!=N") for r in CL[k]]); print("runs where D0.1 != N (%d): slack change D0.1 %+.0f, Coupled %+.0f (%.0f%% of Data-DR's)" % (len(a), a[:, 0].mean(), a[:, 1].mean(), 100 * a[:, 1].mean() / a[:, 0].mean()))
a = np.array([r[1:] for k in CL if k.startswith("D=N") for r in CL[k]]); print("runs where D0.1 == N (%d): slack change Coupled %+.0f, commitment %+.0f" % (len(a), a[:, 1].mean(), a[:, 3].mean()))
print("\n(b1) difference of two decisions, in %% of Nominal's cost: nominal objective | data damage (VD - V0) | belief add-on in the coupled set (VC - VD) | sum = Coupled's objective")
for k, v in DEC.items():
    a = np.array([r[1:] for r in v]); g = [r[0] for r in v]
    print("   %-24s %3d runs | %+5.2f ± %.2f | %+5.2f ± %.2f | %+5.2f ± %.2f | %+5.2f" % (k, len(a), *se(a[:, 0], g), *se(a[:, 1], g), *se(a[:, 2], g), a.sum(1).mean()))
a = np.array([r[1:] for r in PAIR]); g = [r[0] for r in PAIR]
print("\n(b2) candidate pairs with the same sites, per 100 units of capacity cut (%d pairs, mean cut %.0f): nominal objective %+5.2f ± %.2f | data damage %+5.2f ± %.2f | belief add-on %+5.2f ± %.2f" % (len(a), a[:, 0].mean(), *se(a[:, 1], g), *se(a[:, 2], g), *se(a[:, 3], g)))
print("     ratio belief add-on change / data damage change: %.2f;  share of pairs where the belief add-on rises when capacity is cut: %.0f%%" % (a[:, 3].mean() / a[:, 2].mean(), 100 * np.mean(a[:, 3] > 1e-6)))
