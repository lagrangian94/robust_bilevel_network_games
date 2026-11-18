# -*- coding: utf-8 -*-
# Radius sweep (ledger 4.31, prediction H77): Belief-only, Data-DR, Coupled at radius 0.1, 0.2, 0.3.  geo/zss3m (N, B0.1, B0.2, D0.1, D0.2, C) + geo/zss3x (N, B0.3, D0.3, C0.2, C0.3).
#   cost table scen 2 (c/f 0.4), 150 runs, 30 cells (gamma x zeta), M = 100.  worlds: K = truth q̂, belief p;  U rho = truth p, belief rho p + (1 - rho) p2 (rho 1, 0.5, 0).
#   per run: cell-average of the mean and of the q95 of total cost, model minus Nominal, % of |Nominal's training cost|.  SE clustered by geography.
#   also the quiet cell (gamma 10, zeta 0).  Output: printed tables and agg_sweep_out.json (for the figure).
import csv, collections, json, numpy as np
B = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
SRC = {"zss3m": ("N", "B0.1", "B0.2", "D0.1", "D0.2", "C"), "zss3x": ("N", "B0.3", "D0.3", "C0.2", "C0.3")}
NAME = {"C": "C0.1"}; WN = ("K", "rho 1", "rho 0.5", "rho 0")
def par(x): pr = [p.split(":") for p in x.strip("{}").split(";") if p]; return [int(a) for a, _ in pr], [float(b) for _, b in pr]
def se(v, g):
    v = np.asarray(v, float); g = np.asarray(g); m = v.mean(); G = np.unique(g)
    e = np.array([(v[g == k] - m).sum() for k in G]); return m, np.sqrt((e ** 2).sum() * len(G) / (len(G) - 1)) / len(v)
T = {}; TQ = {}; G = []; LV = []; XD = collections.defaultdict(dict); NCHK = []
for g in range(1, 16):
    for s in range(1, 11):
        G.append(g); nref = None
        for d, who in SRC.items():
            O = B + d + "/"; dec = {r["who"]: r for r in csv.DictReader(open(O + "dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
            rd = csv.reader(open(O + "eval_g%d_s%d.csv" % (g, s), encoding="utf-8")); hd = next(rd); assert hd[5:] == ["JK", "JU1.0", "JU0.5", "JU0.0"], hd
            J = collections.defaultdict(lambda: collections.defaultdict(list))
            for r in rd: J[(float(r[1]), float(r[2]))][r[4]].append([float(v) for v in r[5:]])
            assert len(J) == 30, (d, g, s, len(J))
            q0, c = float(dec["N"]["q0"]), float(dec["N"]["c"]); cost = lambda x: q0 * len(par(x)[0]) + c * sum(par(x)[1]); L = abs(float(dec["N"]["Vq"]) + cost(dec["N"]["x"])) / 100
            def st(w, cells):
                x = dec[w]["x"]; a = np.array([[(np.array(J[cl][x])[:, iw] + cost(x)).mean() for iw in range(4)] for cl in cells]).mean(0)
                b = np.array([[np.quantile(np.array(J[cl][x])[:, iw] + cost(x), .95) for iw in range(4)] for cl in cells]).mean(0); return np.r_[a, b] / L
            n = st("N", list(J)); nq = st("N", [(10.0, 0.0)])
            if nref is None: nref = n; LV.append(n)
            else: NCHK.append(np.abs(n - nref).max())
            for w in who:
                if w == "N": continue
                k = NAME.get(w, w); T.setdefault(k, []).append(st(w, list(J)) - n); TQ.setdefault(k, []).append(st(w, [(10.0, 0.0)]) - nq); XD[k][(g, s)] = (dec[w]["x"], dec["N"]["x"], float(dec[w]["cap"]), float(dec["N"]["cap"]))
G = np.array(G); LV = np.array(LV); T = {w: np.array(v) for w, v in T.items()}; TQ = {w: np.array(v) for w, v in TQ.items()}; f = lambda v: "%+5.2f ± %.2f" % se(v, G)
print("runs %d, Nominal agrees between zss3m and zss3x: max abs difference %.2e (%% units)" % (len(G), max(NCHK)))
print("Nominal level, q95: " + " | ".join("%s %.1f" % (WN[k], LV[:, 4 + k].mean()) for k in range(4)) + " ; mean: " + " | ".join("%.1f" % LV[:, k].mean() for k in range(4)))
MOD = (("Belief-only", "B"), ("Data-DR", "D"), ("Coupled", "C")); RAD = ("0.1", "0.2", "0.3"); OUT = {}
print("\n== model minus Nominal, %% of Nominal's training cost.  columns: K | rho 1 | rho 0.5 | rho 0")
for nm, p in MOD:
    for r in RAD:
        w = p + r; OUT[w] = {"q95": [se(T[w][:, 4 + k], G) for k in range(4)], "mean": [se(T[w][:, k], G) for k in range(4)]}
        print("%-12s %s  q95  " % (nm, r) + " | ".join(f(T[w][:, 4 + k]) for k in range(4)))
        print("%-12s %s  mean " % (nm, r) + " | ".join(f(T[w][:, k]) for k in range(4)))
print("\n== paired, q95 (negative = first is better).  columns: K | rho 1 | rho 0.5 | rho 0")
PR = [("C" + r, "D" + r) for r in RAD] + [("C" + r, "B" + r) for r in RAD] + [("C0.2", "D0.3"), ("C0.1", "D0.2"), ("C0.1", "D0.3"), ("C0.2", "C0.1"), ("C0.3", "C0.2"), ("D0.3", "D0.2"), ("B0.3", "B0.2"), ("B0.2", "B0.1"), ("D0.2", "D0.1")]
for a, b in PR:
    print("   %-5s - %-5s q95  " % (a, b) + " | ".join(f(T[a][:, 4 + k] - T[b][:, 4 + k]) for k in range(4)))
    print("   %-5s - %-5s mean " % (a, b) + " | ".join(f(T[a][:, k] - T[b][:, k]) for k in range(4)))
print("\n== quiet cell (gamma 10, zeta 0), model minus Nominal: q95 at rho 1 | mean at rho 1 | q95 at K")
for nm, p in MOD:
    for r in RAD: w = p + r; print("   %-12s %s  %s | %s | %s" % (nm, r, f(TQ[w][:, 5]), f(TQ[w][:, 1]), f(TQ[w][:, 4])))
print("\n== decisions (cost table 2): share different from Nominal, mean total capacity (Nominal %.0f), share equal to the next smaller radius" % np.mean([v[3] for v in XD["B0.1"].values()]))
for nm, p in MOD:
    for i, r in enumerate(RAD):
        w = p + r; d = XD[w]; K = sorted(d); line = "   %-12s %s  differs from N %3.0f%%, capacity %.0f" % (nm, r, 100 * np.mean([d[k][0] != d[k][1] for k in K]), np.mean([d[k][2] for k in K]))
        if i > 0: line += ", same as radius %s %3.0f%%" % (RAD[i - 1], 100 * np.mean([d[k][0] == XD[p + RAD[i - 1]][k][0] for k in K]))
        print(line)
print("   same decision across models at the same radius: " + ", ".join("C=D %s %.0f%%, C=B %s %.0f%%" % (r, 100 * np.mean([XD["C" + r][k][0] == XD["D" + r][k][0] for k in XD["C" + r]]), r, 100 * np.mean([XD["C" + r][k][0] == XD["B" + r][k][0] for k in XD["C" + r]])) for r in RAD))
json.dump({w: {k: [[float(a), float(b)] for a, b in v] for k, v in d.items()} for w, d in OUT.items()}, open("agg_sweep_out.json", "w"))
