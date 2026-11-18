# -*- coding: utf-8 -*-
# H76 d.  Which stores keep their commitment when the retailer's belief moves?   geo/mechs (per-store commitment, decisions N, B0.1, D0.1, C of dwps3c), cost table scen 2.
#   store state at belief q̂:  "pinned" = commitment at min(size, 300) (the pre-purchase cap is 300),  "interior" = below.
#   commitment lost = commitment at q̂ minus commitment at the decision's 10 worst beliefs (largest loss when the truth stays q̂), per store.
#   catchment of a store = customers whose nearest open store (existing operator's 2 + the decision's stores) it is; D(k) = k-th lowest catchment demand over the 10 scenarios.
#   newsvendor guess: overage f, underage p - f, critical ratio 1 - f/p = 0.25, so commitment ~ min(cap, D(3)); a belief that adds 0.1 to a low scenario moves it toward D(2).
import csv, os, sys, collections, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
WHO = ("N", "B0.1", "D0.1", "C"); BETA = 0.4; S = 10; q = np.full(S, 0.1); RC = 300.0
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
ROW = []; DECS = {w: [] for w in WHO}; nrun = 0
for g in range(1, 16):
    for s in range(1, 11):
        fm = O + "mechs/mechs_g%d_s%d.csv" % (g, s); fx = O + "mech/xid_g%d_s%d.csv" % (g, s)
        if not (os.path.exists(fm) and os.path.exists(fx)): continue
        dec = {r["who"]: r["x"] for r in csv.DictReader(open(O + "mechs/dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
        X = list(csv.DictReader(open(fx, encoding="utf-8"), delimiter=";")); st = [r for r in X if r["kind"] in "AB"]; cu = [r for r in X if r["kind"] == "C"]; nA = sum(r["kind"] == "A" for r in st); nst = len(st)
        sxy = np.array([[float(r["x"]), float(r["y"])] for r in st]); cxy = np.array([[float(r["x"]), float(r["y"])] for r in cu]); xi = np.array([[float(r["xi%d" % k]) for k in range(1, S + 1)] for r in cu])
        dist = np.sqrt(((sxy[:, None, :] - cxy[None, :, :]) ** 2).sum(2)); M = {}
        L = open(fm, encoding="utf-8").read().splitlines()
        for l in L[1:]:
            p = l.split("|"); h = np.array([float(v) for v in p[5:5 + nst]]); Z = np.array([float(v) for v in p[5 + nst:5 + nst + S]]); M.setdefault(p[1], {})[(p[2], int(p[3]), int(p[4]))] = (h, Z)
        nrun += 1
        for w in WHO:
            x = par(dec[w]); m = M[dec[w]]; h0, Z0 = m[("q", 0, 0)]; base = cvar(Z0, q)
            mv = [(cvar(m[("pair", i, j)][1], q) - base, i, j) for i in range(1, S + 1) for j in range(1, S + 1) if i != j]; mv.sort(); worst = mv[-10:]
            hw = np.mean([m[("pair", i, j)][0] for _, i, j in worst], axis=0); hall = np.array([m[("pair", i, j)][0] for _, i, j in mv])
            opn = list(range(nA)) + [nA + k - 1 for k in x]; near = np.array(opn)[np.argmin(dist[opn, :], axis=0)]
            tot = {"cap": 0.0, "pin": 0.0, "rob": 0.0, "h": 0.0, "lost": 0.0, "lost_pin": 0.0, "lost_int": 0.0, "h_pin": 0.0, "h_int": 0.0, "n": 0, "npin": 0}
            for k, b in x.items():
                i = nA + k - 1; lim = min(b, RC); pinned = h0[i] >= lim - 0.5; robust = pinned and hall[:, i].min() >= lim - 0.5; lost = h0[i] - hw[i]
                Dk = np.sort(xi[near == i].sum(0)) if (near == i).any() else np.zeros(S)
                dA = dist[:nA, :][:, near == i].min() if (near == i).any() else np.nan     # how close the existing operator is to this store's customers
                ROW.append((g, s, w, k, b, h0[i], lost, pinned, robust, Dk[2], Dk[1], Dk[0], Dk.mean(), int((near == i).sum()), lim))
                tot["cap"] += b; tot["h"] += h0[i]; tot["lost"] += lost; tot["n"] += 1; tot["npin"] += pinned; tot["pin"] += h0[i] * pinned; tot["rob"] += h0[i] * robust
                tot["lost_pin" if pinned else "lost_int"] += lost; tot["h_pin" if pinned else "h_int"] += h0[i]
            DECS[w].append((g, tot, dec[w] != dec["N"]))
print("runs with per-store data: %d" % nrun)
print("\n(1) decisions: share of stores pinned at min(size, 300) | share of the commitment that is pinned | never moves in the 90 beliefs | commitment lost (worst 10 beliefs): total, from pinned stores, from interior stores")
for w in WHO:
    T = [t for _, t, _ in DECS[w]]; f = lambda k: np.mean([t[k] for t in T])
    print("   %-5s stores pinned %3.0f%% | commitment pinned %3.0f%% | never moves %3.0f%% | lost %5.1f = pinned %5.1f + interior %5.1f | commitment %4.0f (pinned %4.0f, interior %4.0f)" % (
        w, 100 * sum(t["npin"] for t in T) / sum(t["n"] for t in T), 100 * sum(t["pin"] for t in T) / sum(t["h"] for t in T), 100 * sum(t["rob"] for t in T) / sum(t["h"] for t in T), f("lost"), f("lost_pin"), f("lost_int"), f("h"), f("h_pin"), f("h_int")))
print("   only runs where the decision differs from Nominal (decision vs Nominal in the same runs): commitment pinned share, never-moves share, lost")
for w in WHO[1:]:
    idx = [k for k, (_, _, d) in enumerate(DECS[w]) if d]; A = [DECS[w][k][1] for k in idx]; B = [DECS["N"][k][1] for k in idx]; sh = lambda T, k: 100 * sum(t[k] for t in T) / sum(t["h"] for t in T)
    print("   %-5s %3d runs | pinned %3.0f%% vs N %3.0f%% | never moves %3.0f%% vs %3.0f%% | lost %5.1f vs %5.1f | interior commitment %4.0f vs %4.0f" % (w, len(idx), sh(A, "pin"), sh(B, "pin"), sh(A, "rob"), sh(B, "rob"),
          np.mean([t["lost"] for t in A]), np.mean([t["lost"] for t in B]), np.mean([t["h_int"] for t in A]), np.mean([t["h_int"] for t in B])))
R = [r for r in ROW if r[2] == "N"]; a = np.array([[r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14]] for r in R], float)
print("\n(2) Nominal's stores (%d): by size" % len(a))
for b in (200, 300, 400):
    m = a[a[:, 0] == b]
    if len(m): print("   size %d: %4d stores | pinned %3.0f%% | never moves %3.0f%% | commitment %5.1f | lost %5.1f | catchment: customers %.1f, mean demand %4.0f, 3rd lowest %4.0f, 2nd lowest %4.0f, lowest %4.0f" % (
        b, len(m), 100 * m[:, 3].mean(), 100 * m[:, 4].mean(), m[:, 1].mean(), m[:, 2].mean(), m[:, 9].mean(), m[:, 8].mean(), m[:, 5].mean(), m[:, 6].mean(), m[:, 7].mean()))
pin = a[a[:, 3] == 1]; it = a[a[:, 3] == 0]
print("   pinned stores: lost %.1f (%.1f%% of their commitment) | interior stores: lost %.1f (%.1f%%)" % (pin[:, 2].mean(), 100 * pin[:, 2].sum() / pin[:, 1].sum(), it[:, 2].mean(), 100 * it[:, 2].sum() / it[:, 1].sum()))
print("\n(3) newsvendor guess on all stores of the four decisions: commitment vs min(cap, D(3)), lost vs the fall of min(cap, D(k)) from k = 3")
A = np.array([[r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14]] for r in ROW], float); lim = A[:, 10]
g3 = np.minimum(lim, A[:, 5]); g2 = np.minimum(lim, A[:, 6]); g1 = np.minimum(lim, A[:, 7])
print("   corr(commitment, min(cap, D3)) %.2f | mean commitment %.0f, guess %.0f | corr(lost, guess D3 - D2) %.2f | corr(lost, guess D3 - D1) %.2f | mean lost %.1f, guess D3-D2 %.1f, D3-D1 %.1f" % (
    np.corrcoef(A[:, 1], g3)[0, 1], A[:, 1].mean(), g3.mean(), np.corrcoef(A[:, 2], g3 - g2)[0, 1], np.corrcoef(A[:, 2], g3 - g1)[0, 1], A[:, 2].mean(), (g3 - g2).mean(), (g3 - g1).mean()))
mar = A[:, 6] - lim   # D(2) of the catchment minus the cap: positive = the store stays full even after a one-step fall of the quantile
for lo, hi in ((-1e9, -100), (-100, -30), (-30, 0), (0, 50), (50, 1e9)):
    m = A[(mar >= lo) & (mar < hi)]
    if len(m): print("   margin D(2) - cap in [%5.0f, %5.0f): %4d stores | pinned %3.0f%% | never moves %3.0f%% | lost %5.1f" % (max(lo, -999), min(hi, 999), len(m), 100 * m[:, 3].mean(), 100 * m[:, 4].mean(), m[:, 2].mean()))
print("\n(4) the four decisions: mean of margin D(2) - cap over stores (weighted by commitment), share of stores with margin >= 0")
for w in WHO:
    m = np.array([[r[5], r[10] - r[14]] for r in ROW if r[2] == w], float); print("   %-5s margin %+6.1f | stores with margin >= 0: %3.0f%% | stores %d" % (w, (m[:, 0] * m[:, 1]).sum() / m[:, 0].sum(), 100 * np.mean(m[:, 1] >= 0), len(m)))
