# -*- coding: utf-8 -*-
# Data-DR(0.1) minus Nominal in runs where they differ, split by kind of change: same sites & smaller capacity (pure cut) vs other.
#  (i) pair moves, at each decision's 10 joint-worst moves: data, belief, cross (geo/mechc);  (ii) coupled-set values: VD - V0, VC - VD (geo/dw/cp_dwps3)
import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
BETA = 0.4; S = 10; q = np.full(S, 0.1)
def wts(Z, p):
    o = np.argsort(-Z, kind="stable"); cap = p / (1 - BETA); rest = 1.0; w = np.zeros(S)
    for s in o:
        m = min(cap[s], rest); w[s] = m; rest -= m
        if rest <= 1e-12: break
    return w
cvar = lambda Z, p: float(wts(Z, p) @ Z)
def par(x): return {int(a): float(b) for a, b in (p.split(":") for p in x.strip("{}").split(";") if p)}
def se(v, g):
    v = np.asarray(v, float); g = np.asarray(g); m = v.mean(); G = np.unique(g)
    if len(G) < 2: return m, float("nan")
    e = np.array([(v[g == k] - m).sum() for k in G]); return m, np.sqrt((e ** 2).sum() * len(G) / (len(G) - 1)) / len(v)
for W in ("D0.1", "C", "B0.1"):
    R = {"pure cut": [], "other": []}
    for g in range(1, 16):
        for s in range(1, 11):
            dec = {r["who"]: r for r in csv.DictReader(open(O + "mechc/dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
            if dec[W]["x"] == dec["N"]["x"]: continue
            q0, c = float(dec["N"]["q0"]), float(dec["N"]["c"]); M = {}
            for r in csv.DictReader(open(O + "mechc/mech_g%d_s%d.csv" % (g, s), encoding="utf-8")):
                M.setdefault(r["x"], {})[(r["bel"], int(r["i"]), int(r["j"]))] = (float(r["hB"]), np.array([float(r["Z%d" % k]) for k in range(1, S + 1)]), float(r["cap"]), int(r["nsite"]))
            V = {r["x"]: (int(r["n"]), float(r["cap"]), float(r["V0"]), float(r["VD"]), float(r["VC"])) for r in csv.DictReader(open(O + "dw/cp_dwps3_g%d_s%d.csv" % (g, s), encoding="utf-8"))}
            hN, ZN, capN, nsN = M[dec["N"]["x"]][("q", 0, 0)]; L = abs(cvar(ZN, q) + q0 * nsN + c * capN) / 100
            def st(x):
                m = M[x]; h0, Z0, cap, ns = m[("q", 0, 0)]; base = cvar(Z0, q); rows = []
                for i in range(1, S + 1):
                    for j in range(1, S + 1):
                        if i == j: continue
                        a = q.copy(); a[j - 1] += a[i - 1]; a[i - 1] = 0.0; h, Z = m[("pair", i, j)][:2]; dat = cvar(Z0, a) - base; bel = cvar(Z, q) - base; both = cvar(Z, a) - base
                        rows.append((both, dat, bel, both - dat - bel, h0 - h))
                a = np.array(rows); o = np.argsort(a[:, 0])[-10:]; return np.r_[a[o].mean(0)[:4] / L, a[o].mean(0)[4], cap - h0, h0]
            d = st(dec[W]["x"]) - st(dec["N"]["x"]); xa, xn = par(dec[W]["x"]), par(dec["N"]["x"]); pure = set(xa) == set(xn) and sum(xa.values()) < sum(xn.values())
            x1, x0 = dec[W]["x"], dec["N"]["x"]; vv = [((V[x1][3] - V[x1][2]) - (V[x0][3] - V[x0][2])) / L, ((V[x1][4] - V[x1][3]) - (V[x0][4] - V[x0][3])) / L]
            R["pure cut" if pure else "other"].append((g, *d, *vv))
    print("== %s minus Nominal, runs where they differ" % W)
    for k, v in list(R.items()) + [("all", R["pure cut"] + R["other"])]:
        if len(v) < 3: print("   %-8s %d runs" % (k, len(v))); continue
        a = np.array([r[1:] for r in v]); g = [r[0] for r in v]; f = lambda c: "%+5.2f ± %.2f" % se(a[:, c], g)
        print("   %-8s %3d runs | slack %+5.0f commit %+5.0f | joint-worst moves: both %s = data %s + belief %s + cross %s | commitment lost %+5.1f | coupled set: data damage %s, belief add-on %s" % (
            k, len(a), a[:, 5].mean(), a[:, 6].mean(), f(0), f(1), f(2), f(3), a[:, 4].mean(), f(7), f(8)))
