# -*- coding: utf-8 -*-
# Where does the lost commitment go?  Over the 45 low pair moves: change of the leader's commitment and of the existing operator's (A) commitment, per decision (geo/mechs, xid for total demand).
# Also: leader stores split by whether the store is full at q̂ (commitment at min(size, 300)); and by whether one of its catchment customers is closer to an A store than to any other leader store ("contested with A").
import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
WHO = ("N", "B0.1", "D0.1", "C"); S = 10; RC = 300.0
def par(x): return {int(a): float(b) for a, b in (p.split(":") for p in x.strip("{}").split(";") if p)}
def se(v, g):
    v = np.asarray(v, float); g = np.asarray(g); m = v.mean(); G = np.unique(g); e = np.array([(v[g == k] - m).sum() for k in G]); return m, np.sqrt((e ** 2).sum() * len(G) / (len(G) - 1)) / len(v)
R = {w: [] for w in WHO}
for g in range(1, 16):
    for s in range(1, 11):
        dec = {r["who"]: r["x"] for r in csv.DictReader(open(O + "mechs/dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
        X = list(csv.DictReader(open(O + "mech/xid_g%d_s%d.csv" % (g, s), encoding="utf-8"), delimiter=";")); st = [r for r in X if r["kind"] in "AB"]; cu = [r for r in X if r["kind"] == "C"]; nA = sum(r["kind"] == "A" for r in st); nst = len(st)
        xi = np.array([[float(r["xi%d" % k]) for k in range(1, S + 1)] for r in cu]); tot = xi.sum(0); M = {}
        for l in open(O + "mechs/mechs_g%d_s%d.csv" % (g, s), encoding="utf-8").read().splitlines()[1:]:
            p = l.split("|"); M.setdefault(p[1], {})[(p[2], int(p[3]), int(p[4]))] = np.array([float(v) for v in p[5:5 + nst]])
        low = [(i, j) for i in range(1, S + 1) for j in range(1, S + 1) if i != j and tot[j - 1] < tot[i - 1]]
        for w in WHO:
            x = par(dec[w]); m = M[dec[w]]; h0 = m[("q", 0, 0)]; d = np.array([m[("pair", a, c)] - h0 for a, c in low]); dB = d[:, nA:].sum(1); dA = d[:, :nA].sum(1)
            full = np.array([h0[nA + k - 1] >= min(b, RC) - 0.5 for k, b in x.items()]); idx = np.array([nA + k - 1 for k in x])
            up = np.clip(d[:, idx], 0, None).sum(1).mean(); dn = np.clip(d[:, idx], None, 0).sum(1).mean()
            R[w].append((g, dec[w] != dec["N"], dB.mean(), dA.mean(), d[:, idx[full]].sum(1).mean() if full.any() else 0.0, d[:, idx[~full]].sum(1).mean() if (~full).any() else 0.0, up, dn, h0[:nA].sum(), np.mean((dB < -1) & (dA > 1)), np.mean(dB < -1)))
print("change of the commitment over the 45 low moves (units, mean over 150 runs)")
print("%-5s | leader total | existing operator | leader: full stores | not full | leader stores going up | going down | A's commitment at q̂ | moves where leader falls: %% | of which A rises" % "")
for w in WHO:
    a = np.array([r[2:] for r in R[w]]); m = a.mean(0)
    print("%-5s | %+6.2f       | %+6.2f            | %+6.2f              | %+6.2f   | %+6.2f                 | %+6.2f     | %5.0f               | %3.0f%%                        | %3.0f%%" % (w, m[0], m[1], m[2], m[3], m[4], m[5], m[6], 100 * m[8], 100 * m[7] / max(m[8], 1e-9)))
print("runs where the decision differs from Nominal: decision minus Nominal")
for w in WHO[1:]:
    idx = [k for k, r in enumerate(R[w]) if r[1]]; g = [R[w][k][0] for k in idx]; d = np.array([R[w][k][2:] for k in idx]) - np.array([R["N"][k][2:] for k in idx])
    print("   %-5s %3d runs | leader %+5.2f ± %.2f | existing operator %+5.2f ± %.2f | full stores %+5.2f | not full %+5.2f ± %.2f | going up %+5.2f | going down %+5.2f" % (w, len(idx), *se(d[:, 0], g), *se(d[:, 1], g), d[:, 2].mean(), *se(d[:, 3], g), d[:, 4].mean(), d[:, 5].mean()))
