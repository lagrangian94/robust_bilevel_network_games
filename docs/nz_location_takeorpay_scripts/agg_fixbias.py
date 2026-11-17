# -*- coding: utf-8 -*-
# python agg_fixbias.py : decomposition of (in-sample estimate of N) - (true cost of N), configuration expo15
import csv, os
import numpy as np
OLD = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/expo/"
def se(v): return np.std(v, ddof=1) / np.sqrt(len(v))
for cv in ("0.5", "0.4", "0.2"):
    rows = {}
    for g in range(1, 16):
        f = OLD + "fixbias_g%d.csv" % g
        if not os.path.exists(f): continue
        R = [r for r in csv.DictReader(open(f, encoding="utf-8")) if r["cv"] == cv]
        if len(R) < 10: continue
        JN = [float(r["JN"]) for s in range(1, 11) for r in csv.DictReader(open(OLD + "expoS_expo15_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["cv"] == cv]
        a = {k: np.array([float(r[k]) for r in R]) for k in ("VN", "Va", "Jb", "Vc", "Jd", "hB_s", "hB_T")}; a["JN"] = np.array(JN); rows[g] = a
    if not rows: continue
    print("== c/v %s : %d geos" % (cv, len(rows)))
    def line(lab, fn):
        v = np.array([100 * fn(a) / abs(a["Jd"].mean()) for a in rows.values()])
        print("  %-58s %+6.2f +- %.2f   (geos <0: %d, >0: %d)" % (lab, v.mean(), se(v), (v < -0.01).sum(), (v > 0.01).sum()))
    line("net: VN - JN (estimate of N minus its true cost)", lambda a: (a["VN"] - a["JN"]).mean())
    line(" 1 selection: VN - Va (min over x vs fixed x0)", lambda a: (a["VN"] - a["Va"]).mean())
    line(" 2 fixed x0: Va - Jb (estimate minus true, same h)", lambda a: (a["Va"] - a["Jb"]).mean())
    line(" 3 Jb - JN (true cost of fixed x0 minus N decision)", lambda a: (a["Jb"] - a["JN"]).mean())
    line(" 2a retailer knows truth: Vc - Jd (pure sample bias)", lambda a: (a["Vc"] - a["Jd"]).mean())
    line(" 2b in-sample effect of sample-based h: Va - Vc", lambda a: (a["Va"] - a["Vc"]).mean())
    line(" 2c true-cost effect of sample-based h: Jb - Jd", lambda a: (a["Jb"] - a["Jd"]).mean())
    h = np.array([[a["hB_s"].mean(), a["hB_T"].mean(), a["hB_s"].std(ddof=1)] for a in rows.values()])
    print("  B reservation: sample-based mean %.1f (sd over seeds %.1f)  truth-based %.1f" % (h[:, 0].mean(), h[:, 2].mean(), h[:, 1].mean()))
