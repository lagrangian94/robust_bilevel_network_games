# -*- coding: utf-8 -*-
# Data-DR (belief fixed at q̂) vs D-track (belief follows the adversary's distribution): python agg_dtrack.py   (AD_DIR other folder)
#   reads geo/dtr/sel_g<g>_s<s>.csv.  values in % of |V0(N) + store cost(N)|; SE over geographies.
import csv, os, collections, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/" + os.environ.get("AD_DIR", "dtr") + "/"
def par(x): pr = [p.split(":") for p in x.strip("{}").split(";") if p]; return [int(a) for a, _ in pr], [float(b) for _, b in pr]
def cl(vals):
    gg = collections.defaultdict(list)
    for (g, s), v in vals.items(): gg[g].append(v)
    m = np.array([np.mean(v) for v in gg.values()]); return "%+.2f +- %.2f" % (np.mean(list(vals.values())), m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else float("nan"))
R = {}
for g in range(1, 16):
    for s in range(1, 11):
        f = O + "sel_g%d_s%d.csv" % (g, s)
        if os.path.exists(f): R[(g, s)] = list(csv.DictReader(open(f, encoding="utf-8")))
print("runs: %d (geographies %d)" % (len(R), len({g for g, _ in R})))
if not R: raise SystemExit
RADII = sorted({r["model"][1:] for rows in R.values() for r in rows if r["model"].startswith("T")}, key=float)
for scen, lab in (("2", "0.4"), ("1", "0.5")):
    T = {k: {r["model"]: r for r in rows if r["scen"] == scen} for k, rows in R.items()}
    q0, c = float(next(iter(T.values()))["N"]["q0"]), float(next(iter(T.values()))["N"]["c"]); cost = lambda x: q0 * len(par(x)[0]) + c * sum(par(x)[1])
    sc = {k: abs(float(t["N"]["V0"]) + cost(t["N"]["x"])) / 100 for k, t in T.items()}
    capm = lambda w: np.mean([float(t[w]["cap"]) for t in T.values()])
    print("== c/f %s | Nominal capacity %.0f" % (lab, capm("N")))
    for r in RADII:
        D, X = "D" + r, "T" + r
        same = np.mean([t[D]["x"] == t[X]["x"] for t in T.values()]); dn = np.mean([t[D]["x"] != t["N"]["x"] for t in T.values()]); tn = np.mean([t[X]["x"] != t["N"]["x"] for t in T.values()])
        more = np.mean([float(t[X]["cap"]) > float(t[D]["cap"]) for t in T.values()]); less = np.mean([float(t[X]["cap"]) < float(t[D]["cap"]) for t in T.values()])
        gapD = {k: (float(t[D]["VT"]) - float(t[D]["VD"])) / sc[k] for k, t in T.items()}     # at Data-DR's decision: tracked worst case minus Data-DR's worst case
        gapN = {k: (float(t[D]["VD"]) - float(t[D]["V0"])) / sc[k] for k, t in T.items()}     # Data-DR's own pessimism at its decision
        regT = {k: (float(t[D]["VT"]) + cost(t[D]["x"]) - float(t[X]["VT"]) - cost(t[X]["x"])) / sc[k] for k, t in T.items()}   # under the tracked criterion: Data-DR's decision minus D-track's
        regD = {k: (float(t[X]["VD"]) + cost(t[X]["x"]) - float(t[D]["VD"]) - cost(t[D]["x"])) / sc[k] for k, t in T.items()}   # under Data-DR's criterion: D-track's decision minus Data-DR's
        prem = {k: (float(t[X]["V0"]) + cost(t[X]["x"]) - float(t[D]["V0"]) - cost(t[D]["x"])) / sc[k] for k, t in T.items()}   # nominal cost: D-track minus Data-DR
        print("   radius %s: capacity Data-DR %.0f, D-track %.0f (D-track larger in %2.0f%%, smaller in %2.0f%%) | same decision %2.0f%% | differs from N: Data-DR %2.0f%%, D-track %2.0f%%" % (r, capm(D), capm(X), 100 * more, 100 * less, 100 * same, 100 * dn, 100 * tn))
        print("      at Data-DR's decision: V_D - V0 = %s | V_T - V_D = %s (positive: following the truth is more pessimistic; share > 0: %2.0f%%)" % (cl(gapN), cl(gapD), 100 * np.mean([v > 1e-6 for v in gapD.values()])))
        print("      loss of Data-DR's decision under the tracked criterion %s | loss of D-track's decision under Data-DR's criterion %s | nominal cost, D-track - Data-DR %s" % (cl(regT), cl(regD), cl(prem)))
