# -*- coding: utf-8 -*-
# H78 follow-up: how much of each one-sided hedge survives in the coupled set.  own-side reduction vs reduction of the coupled damage VC - V0.
import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
def se(v, g):
    v = np.asarray(v, float); g = np.asarray(g); m = v.mean(); G = np.unique(g)
    e = np.array([(v[g == k] - m).sum() for k in G]); return m, np.sqrt((e ** 2).sum() * len(G) / (len(G) - 1)) / len(v)
for W, own in (("D0.1", "data"), ("B0.1", "belief"), ("C", "data+belief")):
    R = []; G = []
    for g in range(1, 16):
        for s in range(1, 11):
            dec = {r["who"]: r for r in csv.DictReader(open(O + "mechc/dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
            if dec[W]["x"] == dec["N"]["x"]: continue
            q0, c = float(dec["N"]["q0"]), float(dec["N"]["c"])
            V = {r["x"]: r for r in csv.DictReader(open(O + "dw/cp_dwps3_g%d_s%d.csv" % (g, s), encoding="utf-8"))}
            A = {r["x"]: r for r in csv.DictReader(open(O + "dw/dw_dwps3_g%d_s%d.csv" % (g, s), encoding="utf-8"))}
            xn = dec["N"]["x"]; L = abs(float(V[xn]["V0"]) + q0 * int(V[xn]["n"]) + c * float(V[xn]["cap"])) / 100
            f = lambda x: np.array([float(V[x]["VD"]) - float(V[x]["V0"]), float(A[x]["adv0.1"]), float(V[x]["VC"]) - float(V[x]["V0"])]) / L
            d = f(dec[W]["x"]) - f(xn); o = d[0] if own == "data" else d[1] if own == "belief" else d[0] + d[1]
            R.append((d[0], d[1], d[2], o, d[2] - o)); G.append(g)
    R = np.array(R)
    print("%-5s %3d runs | data-only damage %+5.2f ± %.2f | belief-only damage %+5.2f ± %.2f | coupled damage %+5.2f ± %.2f | own side (%s) %+5.2f | coupled minus own %+5.2f ± %.2f | share kept %.0f%%" % (
        W, len(G), *se(R[:, 0], G), *se(R[:, 1], G), *se(R[:, 2], G), own, R[:, 3].mean(), *se(R[:, 4], G), 100 * R[:, 2].mean() / R[:, 3].mean()))
