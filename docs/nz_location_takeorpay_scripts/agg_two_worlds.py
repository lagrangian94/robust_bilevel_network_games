# -*- coding: utf-8 -*-
# Same decisions (N, B0.1 chosen from the 10-scenario sample), evaluated in two worlds:
#   A (tkv75): the truth IS the 10-scenario sample; retailer belief = nf draws from it.   B (dw75): the truth is F (test sample); retailer belief = nf draws from F.
import csv, collections, os, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
q90 = lambda v: np.quantile(v, 0.9); rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
def par(x): pr = [p.split(":") for p in x.strip("{}").split(";")]; return [int(a) for a, _ in pr], [float(b) for _, b in pr]
def cl(d):
    gg = collections.defaultdict(list)
    for (g, s), v in d.items(): gg[g].append(v)
    m = np.array([np.mean(v) for v in gg.values()]); return "%+6.2f +- %.2f" % (np.mean(list(d.values())), m.std(ddof=1) / np.sqrt(len(m)))
L = {}; miss = 0
for g in range(1, 16):
    for s in range(1, 4):
        sel = collections.defaultdict(dict); sc = {}
        for r in rd(O + "dw/sel_dw75_g%d_s%d.csv" % (g, s)): sel[int(r["scen"])][r["model"]] = r["x"]; sc[int(r["scen"])] = (float(r["q0"]), float(r["c"]))
        cand = {r["x"]: float(r["V0"]) for r in rd(O + "dw/dw_dw75_g%d_s%d.csv" % (g, s))}
        JB = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rd(O + "oos/true/ot_dw75_g%d_s%d.csv" % (g, s)): JB[int(r["nf"])][r["x"]].append(float(r["J0"]))
        JA = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rd(O + "tkd/tkd_tkv75_g%d_s%d.csv" % (g, s)): JA[int(r["nf"])][r["x"]].append(float(r["J"]))
        L[(g, s)] = (sel, sc, cand, JA, JB)
for k in (1, 2):
    q0, c = next(iter(L.values()))[1][k]; cost = lambda x: q0 * len(par(x)[0]) + c * sum(par(x)[1]); R = {}
    for key, (sel, sc, cand, JA, JB) in L.items():
        xn, xb = sel[k]["N"], sel[k]["B0.1"]
        if any(x not in JA[5] for x in (xn, xb)): miss += 1; continue
        d = {}
        for m, x in (("N", xn), ("B", xb)):
            d[m] = dict(V=cand[x] + cost(x), J0=JB[0][x][0] + cost(x), A5=np.array(JA[5][x]) + cost(x), A20=np.array(JA[20][x]) + cost(x), B5=np.array(JB[5][x]) + cost(x), B20=np.array(JB[20][x]) + cost(x))
        d["sc"] = abs(d["N"]["V"]) / 100; R[key] = d
    print("\n===== c/v %.1f : %d runs (skipped %d: decision not among the world-A candidates)" % (c / 5, len(R), miss))
    f = lambda fn: cl({k_: fn(d) / d["sc"] for k_, d in R.items()})
    print("Nominal (%% of its in-sample value; + = worse)                     world A (truth = sample)     world B (truth = F)")
    print("  retailer shares the leader's distribution: true cost - estimate   %s             %s" % ("  0 (by construction)", f(lambda d: d["N"]["J0"] - d["N"]["V"])))
    for nf in ("5", "20"):
        print("  retailer own sample %2s: mean true cost - estimate                 %s            %s" % (nf, f(lambda d: d["N"]["A" + nf].mean() - d["N"]["V"]), f(lambda d: d["N"]["B" + nf].mean() - d["N"]["V"])))
        print("  retailer own sample %2s: q90 true cost - estimate                  %s            %s" % (nf, f(lambda d: q90(d["N"]["A" + nf]) - d["N"]["V"]), f(lambda d: q90(d["N"]["B" + nf]) - d["N"]["V"])))
    print("Belief-only (delta 0.1) - Nominal")
    print("  differs from N in %.0f%% of runs" % (100 * np.mean([not np.array_equal(d["B"]["A5"], d["N"]["A5"]) for d in R.values()])))
    print("  retailer shares the leader's distribution                         %s            %s" % (f(lambda d: d["B"]["V"] - d["N"]["V"]), f(lambda d: d["B"]["J0"] - d["N"]["J0"])))
    for nf in ("5", "20"):
        print("  retailer own sample %2s: mean                                      %s            %s" % (nf, f(lambda d: d["B"]["A" + nf].mean() - d["N"]["A" + nf].mean()), f(lambda d: d["B"]["B" + nf].mean() - d["N"]["B" + nf].mean())))
        print("  retailer own sample %2s: q90                                       %s            %s" % (nf, f(lambda d: q90(d["B"]["A" + nf]) - q90(d["N"]["A" + nf])), f(lambda d: q90(d["B"]["B" + nf]) - q90(d["N"]["B" + nf]))))
