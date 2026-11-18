# -*- coding: utf-8 -*-
# Tail advantage by how the retailer's belief is tied to the true distribution.  geo/zss3m (N, B0.1, B0.2, D0.1, D0.2, C), cost table scen 2 (c/f 0.4), 150 runs, 30 cells.
#   worlds: K = truth q̂, belief p;  U rho = truth p, belief rho p + (1 - rho) p2  (rho 1, 0.5, 0).
#   per run: cell-average of mean and q95 of total cost, model minus Nominal, % of Nominal's training cost.  SE clustered by geography.  Paired differences between worlds and between models.
import csv, collections, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/zss3x/"
def par(x): pr = [p.split(":") for p in x.strip("{}").split(";") if p]; return [int(a) for a, _ in pr], [float(b) for _, b in pr]
def se(v, g):
    v = np.asarray(v, float); g = np.asarray(g); m = v.mean(); G = np.unique(g)
    e = np.array([(v[g == k] - m).sum() for k in G]); return m, np.sqrt((e ** 2).sum() * len(G) / (len(G) - 1)) / len(v)
WHO = ("B0.3", "D0.3", "C0.2", "C0.3"); WN = ("K", "U1.0", "U0.5", "U0.0"); T = {}; G = []; LV = []
for g in range(1, 16):
    for s in range(1, 11):
        if "ZS_DONE" not in open(O + "zs_g%d_s%d.log" % (g, s), encoding="utf-8", errors="ignore").read(): continue
        dec = {r["who"]: r for r in csv.DictReader(open(O + "dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
        rd = csv.reader(open(O + "eval_g%d_s%d.csv" % (g, s), encoding="utf-8")); hd = next(rd); assert hd[5:] == ["JK", "JU1.0", "JU0.5", "JU0.0"], hd
        J = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rd: J[(r[1], r[2])][r[4]].append([float(v) for v in r[5:]])
        q0, c = float(dec["N"]["q0"]), float(dec["N"]["c"]); cost = lambda x: q0 * len(par(x)[0]) + c * sum(par(x)[1]); L = abs(float(dec["N"]["Vq"]) + cost(dec["N"]["x"])) / 100
        def st(w):
            x = dec[w]["x"]; a = np.array([[(np.array(J[cl][x])[:, iw] + cost(x)).mean() for iw in range(4)] for cl in J]).mean(0); b = np.array([[np.quantile(np.array(J[cl][x])[:, iw] + cost(x), .95) for iw in range(4)] for cl in J]).mean(0); return np.r_[a, b] / L
        n = st("N"); LV.append(n); G.append(g)
        for w in WHO: T.setdefault(w, []).append(st(w) - n)
G = np.array(G); LV = np.array(LV); T = {w: np.array(v) for w, v in T.items()}; f = lambda v: "%+5.2f ± %.2f" % se(v, G)
print("runs %d.  columns: K | rho 1 | rho 0.5 | rho 0" % len(G))
print("Nominal level: mean " + " | ".join("%.1f" % LV[:, k].mean() for k in range(4)) + " ; q95 " + " | ".join("%.1f" % LV[:, 4 + k].mean() for k in range(4)))
print("Nominal q95, rho 1 minus rho 0: " + f(LV[:, 5] - LV[:, 7]) + " ; mean: " + f(LV[:, 1] - LV[:, 3]))
for w in WHO:
    print("%-5s q95  " % w + " | ".join(f(T[w][:, 4 + k]) for k in range(4)))
    print("%-5s mean " % w + " | ".join(f(T[w][:, k]) for k in range(4)))
print("\npaired: q95 advantage at rho 1 minus at rho 0 (negative = larger advantage when belief follows truth);  rho 1 minus K")
for w in WHO: print("   %-5s rho1 - rho0: %s | rho0.5 - rho0: %s | rho1 - K: %s" % (w, f(T[w][:, 5] - T[w][:, 7]), f(T[w][:, 6] - T[w][:, 7]), f(T[w][:, 5] - T[w][:, 4])))
print("\npaired between models, q95 (negative = first is better):  K | rho 1 | rho 0.5 | rho 0")
for a, b in (("C0.3", "C0.2"), ("C0.3", "D0.3"), ("C0.2", "D0.3"), ("C0.3", "B0.3")): print("   %-5s - %-5s " % (a, b) + " | ".join(f(T[a][:, 4 + k] - T[b][:, 4 + k]) for k in range(4)))
