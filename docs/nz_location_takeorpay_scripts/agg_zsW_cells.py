# -*- coding: utf-8 -*-
# Per-cell report of the gamma-zeta grid evaluation (zsW.jl): python agg_zsW_cells.py   (AZ_DIR, AZ_ALL=1, AZ_WHO=B0.1,D0.3  AZ_WORLD=JK,JU1.0,...)
#   for each world, model and statistic: a gamma (rows) x zeta (columns) table of (model - Nominal) in % of |V_q(N) + store cost(N)|, mean over runs +- SE over the 15 geographies.
#   statistic of a cell = mean / q05 / q95 / width of the total cost over the draws of that cell, computed per run.
import csv, os, collections, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/" + os.environ.get("AZ_DIR", "zs") + "/"
ALL = os.environ.get("AZ_ALL", "0") == "1"; WHOSEL = os.environ.get("AZ_WHO", "B0.1").split(","); WSEL = [w for w in os.environ.get("AZ_WORLD", "").split(",") if w]
def par(x): pr = [p.split(":") for p in x.strip("{}").split(";") if p]; return [int(a) for a, _ in pr], [float(b) for _, b in pr]
R = {}
for g in range(1, 16):
    for s in range(1, 11):
        lg = O + "zs_g%d_s%d.log" % (g, s); fd = O + "dec_g%d_s%d.csv" % (g, s); fe = O + "eval_g%d_s%d.csv" % (g, s)
        done = os.path.exists(lg) and "ZS_DONE" in open(lg, encoding="utf-8", errors="replace").read()
        if not (done or (ALL and os.path.exists(fe))): continue
        rd = csv.reader(open(fe, encoding="utf-8")); hd = next(rd); W = hd[5:]
        J = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rd: J[(r[1], r[2])][r[4]].append([float(v) for v in r[5:]])
        R[(g, s)] = (list(csv.DictReader(open(fd, encoding="utf-8"))), {c: {x: np.array(v) for x, v in d.items()} for c, d in J.items()}, W)
print("runs done: %d (geographies %d)" % (len(R), len({g for g, _ in R})))
if not R: raise SystemExit
W = next(iter(R.values()))[2]; CELLS = {c for _, J, _ in R.values() for c in J}; GAMS = sorted({c[0] for c in CELLS}, key=float); ZETS = sorted({c[1] for c in CELLS}, key=float)
WL = lambda wn: "K (truth q̂, belief drawn)" if wn == "JK" else "U rho = %s" % wn[2:]
def cl(vals):
    gg = collections.defaultdict(list)
    for (g, s), v in vals.items(): gg[g].append(v)
    m = np.array([np.mean(v) for v in gg.values()]); return np.mean(list(vals.values())), (m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else float("nan"))
STAT = (("mean", lambda v: v.mean()), ("q95", lambda v: np.quantile(v, .95)), ("q05", lambda v: np.quantile(v, .05)), ("width", lambda v: np.quantile(v, .95) - np.quantile(v, .05)))
for scen in ("1", "2"):
    D = {k: {r["who"]: r for r in dec if r["scen"] == scen} for k, (dec, _, _) in R.items()}
    r0 = next(iter(D.values()))["N"]; q0, c = float(r0["q0"]), float(r0["c"]); cost = lambda x: q0 * len(par(x)[0]) + c * sum(par(x)[1])
    scale = {k: abs(float(D[k]["N"]["Vq"]) + cost(D[k]["N"]["x"])) / 100 for k in D}
    print("\n==================== c/f %.1f | runs %d" % (c / 5, len(D)))
    for iw, wn in enumerate(W):
        if WSEL and wn not in WSEL: continue
        for w in WHOSEL:
            if not all(w in D[k] for k in D): continue
            for sn, f in STAT:
                print("-- world %s | %s - N | %s   (rows gamma, columns zeta = %s)" % (WL(wn), w, sn, ", ".join(ZETS)))
                for gm in GAMS:
                    cells = []
                    for z in ZETS:
                        v = {k: (f(R[k][1][(gm, z)][D[k][w]["x"]][:, iw] + cost(D[k][w]["x"])) - f(R[k][1][(gm, z)][D[k]["N"]["x"]][:, iw] + cost(D[k]["N"]["x"]))) / scale[k] for k in D}
                        cells.append("%+5.2f±%.2f" % cl(v))
                    print("   gamma %-4s " % gm + "  ".join(cells))
            if w != WHOSEL[0]: continue
            # level of N for reference (once per world)
            print("-- world %s | level of N, q95 - mean (%% of scale)" % WL(wn))
            for gm in GAMS:
                print("   gamma %-4s " % gm + "  ".join("%5.2f" % np.mean([(np.quantile(R[k][1][(gm, z)][D[k]["N"]["x"]][:, iw], .95) - R[k][1][(gm, z)][D[k]["N"]["x"]][:, iw].mean()) / scale[k] for k in D]) for z in ZETS))
