# -*- coding: utf-8 -*-
# look at raw: runs where D0.1 != N. For N's and D's decision: data damage (VD-V0), belief add-on (VC-VD), belief-only damage (adv0.1), scenario costs Z at belief q (sorted), commitment, capacity.
import csv, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
def par(x): return {int(a): float(b) for a, b in (p.split(":") for p in x.strip("{}").split(";") if p)}
for g in range(1, 16):
    for s in range(1, 11):
        dec = {r["who"]: r for r in csv.DictReader(open(O + "mechc/dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
        if dec["D0.1"]["x"] == dec["N"]["x"]: continue
        q0, c = float(dec["N"]["q0"]), float(dec["N"]["c"])
        V = {r["x"]: r for r in csv.DictReader(open(O + "dw/cp_dwps3_g%d_s%d.csv" % (g, s), encoding="utf-8"))}
        A = {r["x"]: r for r in csv.DictReader(open(O + "dw/dw_dwps3_g%d_s%d.csv" % (g, s), encoding="utf-8"))}
        M = {}
        for r in csv.DictReader(open(O + "mechc/mech_g%d_s%d.csv" % (g, s), encoding="utf-8")):
            if r["bel"] == "q": M[r["x"]] = (float(r["hB"]), np.array([float(r["Z%d" % k]) for k in range(1, 11)]))
        xn, xd = dec["N"]["x"], dec["D0.1"]["x"]; L = abs(float(V[xn]["V0"]) + q0 * 3 + c * float(V[xn]["cap"])) / 100
        pure = set(par(xn)) == set(par(xd))
        print("g%d s%d %s" % (g, s, "PURE" if pure else "site"))
        for w, x in (("N", xn), ("D", xd), ("C", dec["C"]["x"])):
            v = V[x]; h, Z = M[x]; Zs = np.sort(Z + 0.0)[::-1] / L
            print("   %s %-28s cap %4.0f h %5.0f | dat %5.2f add %5.2f belonly %5.2f | Z-spread(top6 worst - mean all) %s" % (w, x, float(v["cap"]), h, (float(v["VD"]) - float(v["V0"])) / L, (float(v["VC"]) - float(v["VD"])) / L, float(A[x]["adv0.1"]) / L, " ".join("%.1f" % (z - Zs.min()) for z in Zs)))
