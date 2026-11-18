# -*- coding: utf-8 -*-
# H76 c.  Why is the damage of Data-DR's decision larger than the sum of the two damages when both errors come together?   geo/mechc, cost table scen 2, 150 runs.
#   pair move i -> j: truth a (mass of i moved to j), belief d = a.  Zq = loss per scenario at belief q̂, Zd = at belief d.
#   data = cvar_a(Zq) - cvar_q(Zq),  belief = cvar_q(Zd) - cvar_q(Zq),  both = cvar_a(Zd) - cvar_q(Zq),  cross = both - data - belief.
#   approximation of cross: sum_s (w_a(s) - w_q(s)) (Zd(s) - Zq(s)),  w = CVaR tail weights computed on Zq.  Its main term is the scenario j that received the mass.
import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
WHO = ("N", "B0.1", "D0.1", "C"); BETA = 0.4; S = 10; q = np.full(S, 0.1)
def wts(Z, p):
    o = np.argsort(-Z, kind="stable"); cap = p / (1 - BETA); rest = 1.0; w = np.zeros(S)
    for s in o:
        m = min(cap[s], rest); w[s] = m; rest -= m
        if rest <= 1e-12: break
    return w
cvar = lambda Z, p: float(wts(Z, p) @ Z)
def se(v, g):
    v = np.asarray(v, float); g = np.asarray(g); m = v.mean(); G = np.unique(g); e = np.array([(v[g == k] - m).sum() for k in G]); return m, np.sqrt((e ** 2).sum() * len(G) / (len(G) - 1)) / len(v)
R = {w: [] for w in WHO}; gg = []; ALLX = []; ALLY = []
for g in range(1, 16):
    for s in range(1, 11):
        dec = {r["who"]: r for r in csv.DictReader(open(O + "mechc/dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
        q0, c = float(dec["N"]["q0"]), float(dec["N"]["c"]); M = {}; v = None
        for r in csv.DictReader(open(O + "mechc/mech_g%d_s%d.csv" % (g, s), encoding="utf-8")):
            v = float(r["v"]); M.setdefault(r["x"], {})[(r["bel"], int(r["i"]), int(r["j"]))] = (float(r["hB"]), np.array([float(r["Z%d" % k]) for k in range(1, S + 1)]), float(r["cap"]), int(r["nsite"]))
        hN, ZN, capN, nsN = M[dec["N"]["x"]][("q", 0, 0)]; L = abs(cvar(ZN, q) + q0 * nsN + c * capN) / 100; gg.append(g)
        for w in WHO:
            m = M[dec[w]["x"]]; h0, Z0, cap, ns = m[("q", 0, 0)]; base = cvar(Z0, q); fl = np.abs(Z0 + v * h0) < 1e-3 * v * h0; rows = []
            for i in range(1, S + 1):
                for j in range(1, S + 1):
                    if i == j: continue
                    a = q.copy(); a[j - 1] += a[i - 1]; a[i - 1] = 0.0; h, Z = m[("pair", i, j)][:2]
                    dat = cvar(Z0, a) - base; bel = cvar(Z, q) - base; both = cvar(Z, a) - base; cross = both - dat - bel
                    dw = wts(Z0, a) - wts(Z0, q); apx = float(dw @ (Z - Z0)); main = dw[j - 1] * (Z - Z0)[j - 1]
                    rows.append((both / L, dat / L, bel / L, cross / L, apx / L, main / L, h0 - h, (Z - Z0)[j - 1] / L, float(fl[j - 1]), dw[j - 1]))
            a = np.array(rows); o = np.argsort(a[:, 0])[-10:]; R[w].append(np.r_[a[o].mean(0), a.mean(0)[3:5], fl.sum()])
            if w == "N": ALLX += list(a[:, 4]); ALLY += list(a[:, 3])
print("the 10 pair moves that hurt each decision most when both errors come together (means over 150 runs, % of Nominal's cost)")
print("%-5s | both  = data + belief + cross  | approx of cross | of which scenario j | commitment lost | loss change in scenario j | j is a floor scenario | added tail weight on j | floor scen" % "")
for w in WHO:
    a = np.array(R[w]); m = a.mean(0)
    print("%-5s | %5.2f = %5.2f + %5.2f + %5.2f | %5.2f           | %5.2f               | %6.1f          | %+6.2f                    | %3.0f%%                  | %.3f                  | %.1f" % (w, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], 100 * m[8], m[9], m[12]))
print("\ncross term, decision minus Nominal (worst 10): " + " | ".join("%s %+5.2f ± %.2f" % (w, *se(np.array(R[w])[:, 3] - np.array(R["N"])[:, 3], gg)) for w in WHO[1:]))
print("approximation, decision minus Nominal:        " + " | ".join("%s %+5.2f ± %.2f" % (w, *se(np.array(R[w])[:, 4] - np.array(R["N"])[:, 4], gg)) for w in WHO[1:]))
print("corr over runs of (cross, approx) of the worst-10 means: " + " | ".join("%s %.2f" % (w, np.corrcoef(np.array(R[w])[:, 3], np.array(R[w])[:, 4])[0, 1]) for w in WHO))
print("corr over all 90 moves x 150 runs (Nominal's decision): %.2f" % np.corrcoef(ALLX, ALLY)[0, 1])
d = np.array(R["D0.1"]) - np.array(R["N"])
print("Data-DR minus Nominal over runs: corr(cross, commitment lost) %.2f | corr(cross, loss change in j) %.2f" % (np.corrcoef(d[:, 3], d[:, 6])[0, 1], np.corrcoef(d[:, 3], d[:, 7])[0, 1]))
