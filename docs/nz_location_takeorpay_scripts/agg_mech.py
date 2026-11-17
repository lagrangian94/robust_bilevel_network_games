# -*- coding: utf-8 -*-
# Why is Data-DR so much more conservative than Belief-only?  (ledger 4.27, prediction H72)  Input geo/mech/{dec,mech}_g_s.csv.
# All values in % of |V_q(N) + store cost(N)| unless a unit is given.  SE clustered by geography.
import csv, os, sys, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/" + os.environ.get("AM_DIR", "mech") + "/"
WHO = os.environ.get("AM_WHO", "N,B0.1,D0.1,D0.3").split(","); SCENS = os.environ.get("AM_SCEN", "2").split(","); BETA = 0.4; S = 10; q = np.full(S, 0.1)
def cvar(Z, p):
    o = np.argsort(-Z); cap = p / (1 - BETA); rest = 1.0; v = 0.0
    for s in o:
        m = min(cap[s], rest); v += m * Z[s]; rest -= m
        if rest <= 1e-12: break
    return v
def se(v, g):
    v = np.asarray(v, float); g = np.asarray(g); m = v.mean(); G = np.unique(g)
    if len(G) < 2: return m, float("nan")
    e = np.array([(v[g == k] - m).sum() for k in G]); return m, np.sqrt((e ** 2).sum() * len(G) / (len(G) - 1)) / len(v)
def fm(t): return "%+6.2f ± %.2f" % t
for scen in SCENS:
    R = []
    for g in range(1, 16):
        for s in range(1, 11):
            fd, fmch = O + "dec_g%d_s%d.csv" % (g, s), O + "mech_g%d_s%d.csv" % (g, s)
            if not (os.path.exists(fd) and os.path.exists(fmch)): continue
            dec = {r["who"]: r for r in csv.DictReader(open(fd, encoding="utf-8")) if r["scen"] == scen}
            if not all(w in dec for w in WHO): continue
            q0, c = float(dec["N"]["q0"]), float(dec["N"]["c"]); M = {}
            for r in csv.DictReader(open(fmch, encoding="utf-8")):
                M.setdefault(r["x"], {})[(r["bel"], int(r["i"]), int(r["j"]))] = (float(r["hB"]), np.array([float(r["Z%d" % k]) for k in range(1, S + 1)]), float(r["cap"]), int(r["nsite"]), float(r["v"]), float(r["vsm"]))
            R.append((g, s, q0, c, dec, M))
    n = len(R); gg = np.array([r[0] for r in R])
    print("==== cost table scen %s (q0 %.0f, c %.1f): %d runs" % (scen, R[0][2], R[0][3], n))
    # ---------- part 1: the decisions at belief q̂
    T = {w: [] for w in WHO}
    for (g, s, q0, c, dec, M) in R:
        xN = dec["N"]["x"]; hB, Z, cap, ns, v, vsm = M[xN][("q", 0, 0)]; L = abs(cvar(Z, q) + q0 * ns + c * cap)
        for w in WHO:
            hB, Z, cap, ns, v, vsm = M[dec[w]["x"]][("q", 0, 0)]; Rv = np.sort(-Z); fl = v * hB; full = v * (hB + vsm * (cap - hB)); tol = 1e-4 * abs(fl) + 1e-6
            cost = q0 * ns + c * cap
            w5 = np.array([0.163, 0.137, 0.1, 0.1, 0.1, 0.0]) / 0.6          # Data-DR 0.1, old box: the 6th worst scenario leaves the tail
            w3 = np.array([0.29, 0.21, 0.1, 0.0, 0.0, 0.0]) / 0.6            # Data-DR 0.3, old box: only the worst three stay
            T[w].append((cap, hB, cap - hB, np.sum(Rv <= fl + tol), np.sum(Rv >= full - tol), np.sum(Rv[:6] > fl + tol),
                         100 * cost / L, 100 * Rv[:3].mean() / L, 100 * Rv[3:6].mean() / L, 100 * Rv[6:].mean() / L, 100 * Rv.mean() / L,
                         100 * (cost - Rv[:6].mean()) / L, 100 * (cost - Rv.mean()) / L,
                         100 * (Rv[:6].mean() - fl) / L, 100 * (Rv[:6] @ w5 - fl) / L, 100 * (Rv[:6] @ w3 - fl) / L, 100 * (Rv.mean() - fl) / L, 100 * fl / L, ns))
    A = {w: np.array(T[w]) for w in WHO}
    print("-- part 1a: levels at belief q̂ (means over runs)")
    print("   %-6s | cap  commit slack | #floor #soldout #spot-in-worst6 | sites" % "who")
    for w in WHO: a = A[w].mean(0); print("   %-6s | %4.0f %5.0f  %4.0f  | %4.1f   %4.1f     %4.1f           | %.2f" % (w, a[0], a[1], a[2], a[3], a[4], a[5], a[18]))
    print("-- part 1b: revenue from capacity above the commitment (spot), in %: weighted as Nominal sees it (worst 6 equal), as Data-DR 0.1 sees it (6th leaves the tail), as Data-DR 0.3 sees it (worst 3 only), plain mean")
    for w in WHO: a = A[w].mean(0); print("   %-6s | floor %6.2f | spot: tail(N) %5.2f  tail(D0.1) %5.2f  tail(D0.3) %5.2f  mean %5.2f" % (w, a[17], a[13], a[14], a[15], a[16]))
    print("-- part 1c: decision minus Nominal at belief q̂ (all runs)   store cost | revenue worst3 | mid3 | top4 | mean | objective CVaR | objective mean")
    for w in WHO[1:]:
        d = A[w] - A["N"]; print("   %-6s | " % w + " | ".join(fm(se(d[:, k], gg)) for k in (6, 7, 8, 9, 10, 11, 12)))
    print("-- part 1c': same, only runs where the decision differs from Nominal")
    for w in WHO[1:]:
        m = np.array([r[4][w]["x"] != r[4]["N"]["x"] for r in R]); d = (A[w] - A["N"])[m]
        print("   %-6s (%3d runs, %2.0f%%) | " % (w, m.sum(), 100 * m.mean()) + " | ".join(fm(se(d[:, k], gg[m])) for k in (6, 7, 8, 9, 10, 11, 12)) + " | cap %+5.0f commit %+5.0f slack %+5.0f" % (d[:, 0].mean(), d[:, 1].mean(), d[:, 2].mean()))
    # ---------- part 2: the same 90 perturbations (all the probability of scenario i moves to scenario j) applied to the truth, to the belief, to both
    print("-- part 2: damage of the 90 pair-move perturbations (TV 0.1), per decision, in %.  data = truth moves, retailer believes q̂;  belief = truth is q̂, retailer believes the moved one;  both = truth moves and the retailer follows")
    print("   %-6s | data: mean  worst10  worst1 | belief: mean  worst10  worst1  share>0 | both: mean  worst10  worst1 | belief worst10: commit  rev worst3  mid3  top4  mean" % "who")
    P2 = {}
    for w in WHO:
        rows = []
        for (g, s, q0, c, dec, M) in R:
            xN = dec["N"]["x"]; hB0, Z0, cap, ns, v, vsm = M[xN][("q", 0, 0)]; L = abs(cvar(Z0, q) + q0 * ns + c * cap)
            m = M[dec[w]["x"]]; hq, Zq = m[("q", 0, 0)][:2]; base = cvar(Zq, q); Rq = np.sort(-Zq)
            dD, dB, dT, det = [], [], [], []
            for i in range(1, S + 1):
                for j in range(1, S + 1):
                    if i == j: continue
                    a = q.copy(); a[j - 1] += a[i - 1]; a[i - 1] = 0.0; h, Z = m[("pair", i, j)][:2]; Rb = np.sort(-Z)
                    dD.append(cvar(Zq, a) - base); dB.append(cvar(Z, q) - base); dT.append(cvar(Z, a) - base)
                    det.append((h - hq, Rb[:3].mean() - Rq[:3].mean(), Rb[3:6].mean() - Rq[3:6].mean(), Rb[6:].mean() - Rq[6:].mean(), Rb.mean() - Rq.mean()))
            dD, dB, dT, det = np.array(dD), np.array(dB), np.array(dT), np.array(det); ob = np.argsort(-dB)[:10]
            k = 100 / L
            rows.append((k * dD.mean(), k * np.sort(dD)[-10:].mean(), k * dD.max(), k * dB.mean(), k * dB[ob].mean(), k * dB.max(), 100 * np.mean(dB > 1e-6 * L), k * dT.mean(), k * np.sort(dT)[-10:].mean(), k * dT.max(),
                         det[ob, 0].mean(), k * det[ob, 1].mean(), k * det[ob, 2].mean(), k * det[ob, 3].mean(), k * det[ob, 4].mean()))
        P2[w] = np.array(rows); a = P2[w].mean(0)
        print("   %-6s | %5.2f  %5.2f  %5.2f       | %5.2f  %5.2f  %5.2f   %3.0f%%      | %5.2f  %5.2f  %5.2f      | %+6.1f  %+5.2f  %+5.2f  %+5.2f  %+5.2f" % ((w,) + tuple(a)))
    print("-- part 2': decision minus Nominal (all runs): data worst10 | belief worst10 | both worst10")
    for w in WHO[1:]:
        d = P2[w] - P2["N"]; print("   %-6s | " % w + " | ".join(fm(se(d[:, k], gg)) for k in (1, 4, 8)))
