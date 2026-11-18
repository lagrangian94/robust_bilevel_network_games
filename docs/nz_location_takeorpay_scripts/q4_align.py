# -*- coding: utf-8 -*-
# H78.  Why is the belief add-on (VC - VD) of Data-DR's decision larger than that of Nominal's?  runs where W != N (W = D0.1 by default; also C, B0.1).
#  pair moves (geo/mechc), per decision:  own-worst 10 data damage, own-worst 10 belief damage, joint-worst 10: data, belief, cross.
#  misalignment loss = (own-worst data + own-worst belief) - (data + belief at the joint-worst moves);  overlap of the two top-10 sets;  corr of data and belief damage over the 90 moves.
#  coupled-set values (geo/dw/cp_dwps3): VD - V0, VC - VD;  belief-only damage adv0.1 (geo/dw/dw_dwps3).
import csv, sys, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
BETA = 0.4; S = 10; q = np.full(S, 0.1)
def wts(Z, p):
    o = np.argsort(-Z, kind="stable"); cap = p / (1 - BETA); rest = 1.0; w = np.zeros(S)
    for s in o:
        m = min(cap[s], rest); w[s] = m; rest -= m
        if rest <= 1e-12: break
    return w
cvar = lambda Z, p: float(wts(Z, p) @ Z)
def se(v, g):
    v = np.asarray(v, float); g = np.asarray(g); m = v.mean(); G = np.unique(g)
    e = np.array([(v[g == k] - m).sum() for k in G]); return m, np.sqrt((e ** 2).sum() * len(G) / (len(G) - 1)) / len(v)
NAMES = ["own-worst data", "own-worst belief", "joint: both", "joint: data", "joint: belief", "joint: cross", "misalignment loss", "top-10 overlap", "corr(data, belief)", "coupled: VD-V0", "coupled: VC-VD", "belief-only adv0.1", "VC-VD minus adv0.1", "coupled: VC-V0"]
for W in sys.argv[1:] or ["D0.1"]:
    RN, RW, G = [], [], []
    for g in range(1, 16):
        for s in range(1, 11):
            dec = {r["who"]: r for r in csv.DictReader(open(O + "mechc/dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
            if dec[W]["x"] == dec["N"]["x"]: continue
            q0, c = float(dec["N"]["q0"]), float(dec["N"]["c"]); M = {}
            for r in csv.DictReader(open(O + "mechc/mech_g%d_s%d.csv" % (g, s), encoding="utf-8")):
                M.setdefault(r["x"], {})[(r["bel"], int(r["i"]), int(r["j"]))] = (float(r["hB"]), np.array([float(r["Z%d" % k]) for k in range(1, S + 1)]), float(r["cap"]), int(r["nsite"]))
            V = {r["x"]: r for r in csv.DictReader(open(O + "dw/cp_dwps3_g%d_s%d.csv" % (g, s), encoding="utf-8"))}
            A = {r["x"]: r for r in csv.DictReader(open(O + "dw/dw_dwps3_g%d_s%d.csv" % (g, s), encoding="utf-8"))}
            hN, ZN, capN, nsN = M[dec["N"]["x"]][("q", 0, 0)]; L = abs(cvar(ZN, q) + q0 * nsN + c * capN) / 100
            def st(x):
                m = M[x]; h0, Z0, cap, ns = m[("q", 0, 0)]; base = cvar(Z0, q); rows = []
                for i in range(1, S + 1):
                    for j in range(1, S + 1):
                        if i == j: continue
                        a = q.copy(); a[j - 1] += a[i - 1]; a[i - 1] = 0.0; h, Z = m[("pair", i, j)][:2]; dat = cvar(Z0, a) - base; bel = cvar(Z, q) - base; both = cvar(Z, a) - base
                        rows.append((both, dat, bel, both - dat - bel))
                a = np.array(rows) / L; oj = np.argsort(a[:, 0])[-10:]; od = np.argsort(a[:, 1])[-10:]; ob = np.argsort(a[:, 2])[-10:]
                wd, wb = a[od, 1].mean(), a[ob, 2].mean(); J = a[oj].mean(0); cr = np.corrcoef(a[:, 1], a[:, 2])[0, 1] if a[:, 1].std() > 1e-9 and a[:, 2].std() > 1e-9 else np.nan
                v = V[x]; vd = (float(v["VD"]) - float(v["V0"])) / L; ad = (float(v["VC"]) - float(v["VD"])) / L; bo = float(A[x]["adv0.1"]) / L
                return [wd, wb, J[0], J[1], J[2], J[3], wd + wb - J[1] - J[2], len(set(od) & set(ob)), cr, vd, ad, bo, ad - bo, vd + ad]
            RN.append(st(dec["N"]["x"])); RW.append(st(dec[W]["x"])); G.append(g)
    RN, RW = np.array(RN), np.array(RW)
    print("== %s vs Nominal, %d runs where they differ (%% of Nominal's cost)" % (W, len(G)))
    print("   %-22s | Nominal's decision | %s's decision | difference" % ("", W))
    for k, nm in enumerate(NAMES):
        ok = ~np.isnan(RN[:, k]) & ~np.isnan(RW[:, k]); g = np.array(G)[ok]
        print("   %-22s | %6.2f             | %6.2f        | %+5.2f ± %.2f  (%d)" % (nm, RN[ok, k].mean(), RW[ok, k].mean(), *se(RW[ok, k] - RN[ok, k], g), ok.sum()))
