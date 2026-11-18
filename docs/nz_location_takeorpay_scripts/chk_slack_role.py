# -*- coding: utf-8 -*-
# Does slack capacity protect against a belief error?  Check of the sentence "slack is also the way to win back a lost commitment".
# Input geo/mechc (config dwps3c, cost table scen 2, 150 runs).  Damage = worst 10 of the 90 pair moves, % of Nominal's in-sample cost.
# (1) paired difference decision - Nominal of the belief damage and of the data damage, for all runs where the decision differs, and for the sub-case
#     "same sites, smaller capacity" (a pure capacity cut).   (2) correlation over runs of the change of slack with the change of each damage.
import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
WHO = ("B0.1", "B0.2", "D0.1", "D0.2", "C"); BETA = 0.4; S = 10; q = np.full(S, 0.1)
def cvar(Z, p):
    o = np.argsort(-Z); cap = p / (1 - BETA); rest = 1.0; v = 0.0
    for s in o:
        m = min(cap[s], rest); v += m * Z[s]; rest -= m
        if rest <= 1e-12: break
    return v
def par(x): return {int(a): float(b) for a, b in (p.split(":") for p in x.strip("{}").split(";") if p)}
def se(v, g):
    v = np.asarray(v, float); g = np.asarray(g); m = v.mean(); G = np.unique(g)
    if len(G) < 2: return m, float("nan")
    e = np.array([(v[g == k] - m).sum() for k in G]); return m, np.sqrt((e ** 2).sum() * len(G) / (len(G) - 1)) / len(v)
R = {w: [] for w in WHO}
for g in range(1, 16):
    for s in range(1, 11):
        dec = {r["who"]: r for r in csv.DictReader(open(O + "mechc/dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
        q0, c = float(dec["N"]["q0"]), float(dec["N"]["c"]); M = {}
        for r in csv.DictReader(open(O + "mechc/mech_g%d_s%d.csv" % (g, s), encoding="utf-8")):
            M.setdefault(r["x"], {})[(r["bel"], int(r["i"]), int(r["j"]))] = (float(r["hB"]), np.array([float(r["Z%d" % k]) for k in range(1, S + 1)]), float(r["cap"]), int(r["nsite"]))
        def stat(x):
            m = M[x]; h0, Z0, cap, ns = m[("q", 0, 0)]; base = cvar(Z0, q); dd = []; db = []; dh = []
            for i in range(1, S + 1):
                for j in range(1, S + 1):
                    if i == j: continue
                    a = q.copy(); a[j - 1] += a[i - 1]; a[i - 1] = 0.0; h, Z = m[("pair", i, j)][:2]; dd.append(cvar(Z0, a) - base); db.append(cvar(Z, q) - base); dh.append(h0 - h)
            ob = np.argsort(db)[-10:]; return np.sort(dd)[-10:].mean(), np.mean([db[k] for k in ob]), np.mean([dh[k] for k in ob]), cap - h0, h0, base + q0 * ns + c * cap
        n = stat(dec["N"]["x"]); L = abs(n[5]); xn = par(dec["N"]["x"])
        for w in WHO:
            if w not in dec or dec[w]["x"] == dec["N"]["x"]: continue
            t = stat(dec[w]["x"]); xw = par(dec[w]["x"]); pure = set(xw) == set(xn) and sum(xw.values()) < sum(xn.values())
            R[w].append((g, 100 * (t[0] - n[0]) / L, 100 * (t[1] - n[1]) / L, t[2] - n[2], t[3] - n[3], t[4] - n[4], pure))
print("decision minus Nominal, runs where the decision differs.  damage = worst 10 of 90 pair moves, % of Nominal's cost")
print("%-5s | runs | d slack | d commit | d data damage  | d belief damage | d commitment lost (worst beliefs) | corr(d slack, d data dmg) | corr(d slack, d belief dmg)" % "")
for w in WHO:
    a = np.array([r[1:6] for r in R[w]]); g = [r[0] for r in R[w]]
    print("%-5s | %4d | %+7.0f | %+8.0f | %+5.2f ± %.2f  | %+5.2f ± %.2f   | %+6.1f                            | %+.2f                     | %+.2f" % (
        w, len(a), a[:, 3].mean(), a[:, 4].mean(), *se(a[:, 0], g), *se(a[:, 1], g), a[:, 2].mean(), np.corrcoef(a[:, 3], a[:, 0])[0, 1], np.corrcoef(a[:, 3], a[:, 1])[0, 1]))
print("\npure capacity cut (same sites as Nominal, smaller capacity):")
for w in WHO:
    rr = [r for r in R[w] if r[6]]
    if len(rr) < 3: print("%-5s | %d runs" % (w, len(rr))); continue
    a = np.array([r[1:6] for r in rr]); g = [r[0] for r in rr]
    print("%-5s | %4d | d slack %+5.0f | d commit %+5.0f | d data damage %+5.2f ± %.2f | d belief damage %+5.2f ± %.2f | d commitment lost %+5.1f" % (w, len(a), a[:, 3].mean(), a[:, 4].mean(), *se(a[:, 0], g), *se(a[:, 1], g), a[:, 2].mean()))
