# -*- coding: utf-8 -*-
# Evaluation on the zero-sum gamma-zeta grid (zsW.jl output geo/zs/{dec,eval}_g<g>_s<s>.csv): python agg_zsW.py   (AZ_DIR other folder, AZ_ALL=1 also unfinished runs)
#   worlds: K = truth q̂, belief p ~ Dir(gamma(1 + zeta z));  U<rho> = truth p, belief rho p + (1 - rho) p2.
#   per run and cell (gamma, zeta): mean, q05, q95, width of total cost (store cost + CVaR) over the draws; then averaged over cells; SE over the 15 geographies.
#   differences are model - Nominal in % of |V_q(N) + store cost(N)|; negative = better than Nominal.
import csv, os, collections, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/" + os.environ.get("AZ_DIR", "zs") + "/"
ALL = os.environ.get("AZ_ALL", "0") == "1"
def par(x): pr = [p.split(":") for p in x.strip("{}").split(";") if p]; return [int(a) for a, _ in pr], [float(b) for _, b in pr]
R = {}
for g in range(1, 16):
    for s in range(1, 11):
        lg = O + "zs_g%d_s%d.log" % (g, s); fd = O + "dec_g%d_s%d.csv" % (g, s); fe = O + "eval_g%d_s%d.csv" % (g, s)
        done = os.path.exists(lg) and "ZS_DONE" in open(lg, encoding="utf-8", errors="replace").read()
        if not (done or (ALL and os.path.exists(fe))): continue
        rd = csv.reader(open(fe, encoding="utf-8")); hd = next(rd); W = hd[5:]
        J = collections.defaultdict(lambda: collections.defaultdict(list))   # J[(gamma, zeta)][x] -> rows of world values
        for r in rd: J[(r[1], r[2])][r[4]].append([float(v) for v in r[5:]])
        R[(g, s)] = (list(csv.DictReader(open(fd, encoding="utf-8"))), {c: {x: np.array(v) for x, v in d.items()} for c, d in J.items()}, W)
print("runs done: %d (geographies %d)" % (len(R), len({g for g, _ in R})))
if not R: raise SystemExit
W = next(iter(R.values()))[2]; WL = {"JK": "K  (truth q̂, belief drawn)"}; CELLS = sorted({c for _, J, _ in R.values() for c in J}, key=lambda c: (float(c[0]), float(c[1])))
GAMS = sorted({c[0] for c in CELLS}, key=float); ZETS = sorted({c[1] for c in CELLS}, key=float)
def cl(vals):   # mean and SE clustered by geography; vals: {(g, s): v}
    gg = collections.defaultdict(list)
    for (g, s), v in vals.items(): gg[g].append(v)
    m = np.array([np.mean(v) for v in gg.values()]); return np.mean(list(vals.values())), (m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else float("nan"))
fm = lambda t: "%+6.2f +- %.2f" % t
STAT = (("mean", lambda v: v.mean()), ("q05", lambda v: np.quantile(v, .05)), ("q95", lambda v: np.quantile(v, .95)), ("width", lambda v: np.quantile(v, .95) - np.quantile(v, .05)))
for scen in ("1", "2"):
    D = {}
    for key, (dec, J, _) in R.items(): D[key] = {r["who"]: r for r in dec if r["scen"] == scen}
    r0 = next(iter(D.values()))["N"]; q0, c = float(r0["q0"]), float(r0["c"]); cost = lambda x: q0 * len(par(x)[0]) + c * sum(par(x)[1])
    scale = {k: abs(float(D[k]["N"]["Vq"]) + cost(D[k]["N"]["x"])) / 100 for k in D}
    WHO = [w for w in ("N", "B0.1", "B0.2", "B0.3", "D0.1", "D0.2", "D0.3", "C") if all(w in D[k] for k in D)]
    print("\n==================== c/f %.1f | runs %d | models %s" % (c / 5, len(D), " ".join(WHO)))
    print("   differs from N: " + " | ".join("%s %.0f%%" % (w, 100 * np.mean([D[k][w]["x"] != D[k]["N"]["x"] for k in D])) for w in WHO[1:]) + " | capacity: " + " ".join("%s %.0f" % (w, np.mean([float(D[k][w]["cap"]) for k in D])) for w in WHO))
    def stat(key, w, iw, cells, f):   # average over cells of f(costs of decision w in world iw)
        x = D[key][w]["x"]; J = R[key][1]; return np.mean([f(J[cl_][x][:, iw] + cost(x)) for cl_ in cells]) / scale[key]
    for iw, wn in enumerate(W):
        print("---- world %s" % WL.get(wn, "U rho = %s (truth p, belief rho p + (1 - rho) p2)" % wn[2:]))
        for sel, cells in [("all cells", CELLS)] + [("zeta = %s" % z, [c_ for c_ in CELLS if c_[1] == z]) for z in ZETS] + [("gamma <= 0.3", [c_ for c_ in CELLS if float(c_[0]) <= 0.3]), ("gamma >= 0.5", [c_ for c_ in CELLS if float(c_[0]) >= 0.5])]:
            if sel == "all cells":
                print("   level (%% of scale, N): " + " | ".join("%s %s" % (sn, fm(cl({k: stat(k, "N", iw, cells, f) for k in D}))) for sn, f in STAT))
            for w in WHO[1:]:
                print("   %-13s %-5s - N: " % (sel, w) + " | ".join("%s %s" % (sn, fm(cl({k: stat(k, w, iw, cells, f) - stat(k, "N", iw, cells, f) for k in D}))) for sn, f in STAT))
        # share of (run, cell) where each model has the smallest q95 / narrowest width (ties count for all)
        for sn, f in (("smallest q95", STAT[2][1]), ("narrowest", STAT[3][1])):
            cnt = collections.Counter(); n = 0
            for k in D:
                for cl_ in CELLS:
                    v = {w: f(R[k][1][cl_][D[k][w]["x"]][:, iw] + cost(D[k][w]["x"])) for w in WHO}; b = min(v.values()); n += 1
                    for w in WHO:
                        if v[w] <= b + 1e-6 * max(1.0, abs(b)): cnt[w] += 1
            print("   %s in a cell: " % sn + " | ".join("%s %.0f%%" % (w, 100 * cnt[w] / n) for w in WHO))
