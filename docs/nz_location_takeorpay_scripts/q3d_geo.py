# -*- coding: utf-8 -*-
# How do Belief-only and Coupled push the belief-sensitive commitment onto the existing operator (A)?  Geometry of the decision vs Nominal in runs where they differ (geo/mechs + xid).
import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
WHO = ("N", "B0.1", "D0.1", "C"); S = 10
def par(x): return {int(a): float(b) for a, b in (p.split(":") for p in x.strip("{}").split(";") if p)}
def se(v, g):
    v = np.asarray(v, float); g = np.asarray(g); m = v.mean(); G = np.unique(g); e = np.array([(v[g == k] - m).sum() for k in G]); return m, np.sqrt((e ** 2).sum() * len(G) / (len(G) - 1)) / len(v)
R = {w: [] for w in WHO}
for g in range(1, 16):
    for s in range(1, 11):
        dec = {r["who"]: r["x"] for r in csv.DictReader(open(O + "mechs/dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
        X = list(csv.DictReader(open(O + "mech/xid_g%d_s%d.csv" % (g, s), encoding="utf-8"), delimiter=";")); st = [r for r in X if r["kind"] in "AB"]; cu = [r for r in X if r["kind"] == "C"]; nA = sum(r["kind"] == "A" for r in st); nst = len(st)
        sxy = np.array([[float(r["x"]), float(r["y"])] for r in st]); cxy = np.array([[float(r["x"]), float(r["y"])] for r in cu]); xi = np.array([[float(r["xi%d" % k]) for k in range(1, S + 1)] for r in cu])
        dist = np.sqrt(((sxy[:, None, :] - cxy[None, :, :]) ** 2).sum(2)); dss = np.sqrt(((sxy[:, None, :] - sxy[None, :, :]) ** 2).sum(2)); mu = xi.mean(1); M = {}
        for l in open(O + "mechs/mechs_g%d_s%d.csv" % (g, s), encoding="utf-8").read().splitlines()[1:]:
            p = l.split("|")
            if p[2] == "q": M[p[1]] = np.array([float(v) for v in p[5:5 + nst]])
        for w in WHO:
            x = par(dec[w]); h0 = M[dec[w]]; idx = [nA + k - 1 for k in x]; opn = list(range(nA)) + idx; near = np.array(opn)[np.argmin(dist[opn, :], axis=0)]; isA = near < nA
            dB = dist[idx, :].min(0); dAc = dist[:nA, :].min(0); gap = np.abs(dB - dAc)          # how contested a customer is: small gap = about as far from the leader as from A
            R[w].append((g, dec[w] != dec["N"], sum(x.values()), len(x), h0[nA:].sum(), h0[:nA].sum(), isA.sum(), mu[isA].sum(), dss[np.ix_(idx, range(nA))].min(1).mean(), (mu * (gap < 300)).sum(), dB.mean(),
                         (mu * dB).sum() / mu.sum(), mu[~isA].sum()))
print("runs where the decision differs from Nominal: decision | Nominal | difference ± SE")
lab = ("capacity", "stores", "leader commitment at q̂", "A commitment at q̂", "customers nearest to an A store", "their mean demand", "distance leader store - nearest A store", "demand of customers about as far from the leader as from A (gap < 300)",
       "mean distance customer - nearest leader store", "same, demand weighted", "demand of customers nearest to a leader store")
for w in WHO[1:]:
    idx = [k for k, r in enumerate(R[w]) if r[1]]; g = [R[w][k][0] for k in idx]; d = np.array([R[w][k][2:] for k in idx]); n = np.array([R["N"][k][2:] for k in idx]); print("== %s (%d runs)" % (w, len(idx)))
    for c, l in enumerate(lab): print("   %-70s %7.1f | %7.1f | %+6.1f ± %.1f" % (l, d[:, c].mean(), n[:, c].mean(), *se(d[:, c] - n[:, c], g)))
