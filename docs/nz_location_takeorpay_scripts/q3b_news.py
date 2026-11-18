# -*- coding: utf-8 -*-
# H76 d, second look.  Does a per-store newsvendor on the store's catchment explain how the commitment moves with the retailer's belief?   geo/mechs + geo/mech/xid.
#   catchment of a store = customers whose nearest open store it is.  D = catchment demand per scenario.  critical ratio 1 - f/p = 0.25.
#   guess of the commitment under belief d:  min(cap, smallest D with P_d(D' <= D) >= 0.25),  cap = min(size, 300).
#   stability measure = fall of the commitment averaged over the 45 "low" pair moves (mass goes to a scenario with lower total demand); same moves for every decision.
import csv, os, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
WHO = ("N", "B0.1", "D0.1", "C"); S = 10; q = np.full(S, 0.1); RC = 300.0; CR = 0.25
def par(x): return {int(a): float(b) for a, b in (p.split(":") for p in x.strip("{}").split(";") if p)}
def quant(D, p):
    o = np.argsort(D, kind="stable"); c = np.cumsum(p[o]); return D[o][np.searchsorted(c, CR - 1e-9)]
def se(v, g):
    v = np.asarray(v, float); g = np.asarray(g); m = v.mean(); G = np.unique(g)
    if len(G) < 2: return m, float("nan")
    e = np.array([(v[g == k] - m).sum() for k in G]); return m, np.sqrt((e ** 2).sum() * len(G) / (len(G) - 1)) / len(v)
ST = []; DE = {w: [] for w in WHO}; PA = []; PP = []
for g in range(1, 16):
    for s in range(1, 11):
        dec = {r["who"]: r["x"] for r in csv.DictReader(open(O + "mechs/dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
        X = list(csv.DictReader(open(O + "mech/xid_g%d_s%d.csv" % (g, s), encoding="utf-8"), delimiter=";")); st = [r for r in X if r["kind"] in "AB"]; cu = [r for r in X if r["kind"] == "C"]; nA = sum(r["kind"] == "A" for r in st); nst = len(st)
        sxy = np.array([[float(r["x"]), float(r["y"])] for r in st]); cxy = np.array([[float(r["x"]), float(r["y"])] for r in cu]); xi = np.array([[float(r["xi%d" % k]) for k in range(1, S + 1)] for r in cu]); creg = [r["region"] for r in cu]
        dist = np.sqrt(((sxy[:, None, :] - cxy[None, :, :]) ** 2).sum(2)); tot = xi.sum(0); M = {}
        for l in open(O + "mechs/mechs_g%d_s%d.csv" % (g, s), encoding="utf-8").read().splitlines()[1:]:
            p = l.split("|"); M.setdefault(p[1], {})[(p[2], int(p[3]), int(p[4]))] = np.array([float(v) for v in p[5:5 + nst]])
        low = [(i, j) for i in range(1, S + 1) for j in range(1, S + 1) if i != j and tot[j - 1] < tot[i - 1]]
        for w in WHO:
            x = par(dec[w]); m = M[dec[w]]; h0 = m[("q", 0, 0)]; opn = list(range(nA)) + [nA + k - 1 for k in x]; near = np.array(opn)[np.argmin(dist[opn, :], axis=0)]
            dA = dist[:nA, :].min(0); a_tot = 0.0; p_tot = 0.0; feats = []
            for k, b in x.items():
                i = nA + k - 1; lim = min(b, RC); mine = near == i; D = xi[mine].sum(0) if mine.any() else np.zeros(S); Ds = np.sort(D)
                act = np.array([h0[i] - m[("pair", a, c)][i] for a, c in low]); pre = []
                for a, c in low:
                    d = q.copy(); d[c - 1] += d[a - 1]; d[a - 1] = 0.0; pre.append(min(lim, quant(D, q)) - min(lim, quant(D, d)))
                pre = np.array(pre); PA += list(act); PP += list(pre); a_tot += act.mean(); p_tot += pre.mean()
                own = dist[i, mine]; ST.append((g, w, b, h0[i], min(lim, quant(D, q)), act.mean(), pre.mean(), min(lim, Ds[2]) - min(lim, Ds[1]), Ds[2] - lim, int(mine.sum()), len({creg[j] for j in np.where(mine)[0]}),
                                               float(np.mean(dA[mine] - own)) if mine.any() else np.nan, D.std() / D.mean() if D.mean() > 0 else np.nan, b - h0[i]))
            DE[w].append((g, dec[w] != dec["N"], a_tot, p_tot, sum(x.values()) - h0[nA:].sum()))
PA = np.array(PA); PP = np.array(PP); A = np.array([r[2:] for r in ST], float); W = np.array([r[1] for r in ST])
print("stores: %d (four decisions, 150 runs);  store x low move pairs: %d" % (len(A), len(PA)))
print("(1) per-store newsvendor on the catchment vs the retailer's actual commitment")
print("    level at q̂: corr %.2f, mean actual %.0f, guess %.0f | change over low moves: corr %.2f, mean actual fall %.2f, guess %.2f | per store (mean over its low moves): corr %.2f" % (
    np.corrcoef(A[:, 1], A[:, 2])[0, 1], A[:, 1].mean(), A[:, 2].mean(), np.corrcoef(PA, PP)[0, 1], PA.mean(), PP.mean(), np.corrcoef(A[:, 3], A[:, 4])[0, 1]))
print("    moves where the guess says no change (%.0f%%): actual fall %.2f | guess says fall (%.0f%%): actual %.2f, guess %.2f" % (100 * np.mean(PP == 0), PA[PP == 0].mean(), 100 * np.mean(PP > 0), PA[PP > 0].mean(), PP[PP > 0].mean()))
print("\n(2) stores by the headroom of the catchment: 3rd lowest catchment demand minus cap  (positive = the store is small for its customers)")
print("    headroom        | stores | commitment / cap | fall of the commitment (low moves) | as %% of commitment | slack at the store")
for lo, hi in ((-1e9, -60), (-60, -20), (-20, 0), (0, 30), (30, 80), (80, 1e9)):
    m = A[(A[:, 6] >= lo) & (A[:, 6] < hi)]
    if len(m): print("    [%5.0f, %5.0f)  | %5d  | %3.0f%%             | %5.2f                              | %4.1f%%              | %5.0f" % (max(lo, -999), min(hi, 999), len(m), 100 * m[:, 1].sum() / np.minimum(m[:, 0], RC).sum(), m[:, 3].mean(), 100 * m[:, 3].sum() / m[:, 1].sum(), m[:, 11].mean()))
print("\n(3) the four decisions, all runs: fall of the leader's commitment over the same low moves: actual | newsvendor guess | share of stores with headroom >= 30 | mean headroom | slack")
for w in WHO:
    d = np.array([r[2:] for r in DE[w]]); a = A[W == w]
    print("    %-5s actual %5.2f | guess %5.2f | headroom >= 30: %3.0f%% of stores, %3.0f%% of commitment | mean headroom %+6.1f | slack %4.0f" % (w, d[:, 0].mean(), d[:, 1].mean(), 100 * np.mean(a[:, 6] >= 30), 100 * a[a[:, 6] >= 30][:, 1].sum() / a[:, 1].sum(), a[:, 6].mean(), d[:, 2].mean()))
print("    runs where the decision differs from Nominal (decision vs Nominal, same runs)")
for w in WHO[1:]:
    idx = [k for k, r in enumerate(DE[w]) if r[1]]; g = [DE[w][k][0] for k in idx]; d = np.array([DE[w][k][2:] for k in idx]); n = np.array([DE["N"][k][2:] for k in idx])
    print("    %-5s %3d runs | actual fall %5.2f vs %5.2f (diff %+5.2f ± %.2f) | guess %5.2f vs %5.2f (diff %+5.2f ± %.2f) | corr over runs of the two diffs %.2f" % (
        w, len(idx), d[:, 0].mean(), n[:, 0].mean(), *se(d[:, 0] - n[:, 0], g), d[:, 1].mean(), n[:, 1].mean(), *se(d[:, 1] - n[:, 1], g), np.corrcoef(d[:, 0] - n[:, 0], d[:, 1] - n[:, 1])[0, 1]))
print("\n(4) what else goes with a stable commitment?  corr over stores of the fall (%% of commitment) with: headroom %.2f | customers in catchment %.2f | regions in catchment %.2f | how much closer the store is than the existing operator %.2f | CV of catchment demand %.2f | size %.2f" % tuple(
    np.corrcoef(np.nan_to_num(A[:, 3] / np.maximum(A[:, 1], 1)), np.nan_to_num(A[:, c]))[0, 1] for c in (6, 7, 8, 9, 10, 0)))
