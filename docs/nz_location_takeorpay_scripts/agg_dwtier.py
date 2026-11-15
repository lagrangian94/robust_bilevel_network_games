# data world + per-site capacity tiers (dwTier.jl): model minus Nominal under the true F, by cost scenario and by world (nf = 0: retailer belief = leader's q-hat; nf > 0: retailer's own sample)
import csv, collections, os, sys, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
CFG = os.environ.get("AD_CFG", "dwtier"); SHI = int(os.environ.get("AQ_SHI", "5")); q90 = lambda v: np.quantile(v, 0.9)
MODS = ["D0.1", "D0.2", "D0.3", "B0.05", "B0.1", "B0.2", "B0.3", "C"]; PAIRS = [("C", "B0.1"), ("C", "D0.3"), ("B0.1", "D0.3")]
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
def cl(d):
    gg = collections.defaultdict(list)
    for (g, s), v in d.items(): gg[g].append(v)
    m = np.array([np.mean(v) for v in gg.values()]); return np.mean(list(d.values())), (m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else float("nan")), int((m < -0.01).sum()), int((m > 0.01).sum())
def par(x): pr = [p.split(":") for p in x.strip("{}").split(";")]; return tuple(int(a) for a, _ in pr), tuple(float(b) for _, b in pr)
def load(g, s):
    lg = O + "dw/dw_%s_g%d_s%d.log" % (CFG, g, s)
    if not (os.path.exists(lg) and "OT_DONE" in open(lg, encoding="utf-8", errors="replace").read()): return None
    sel = collections.defaultdict(dict); sc = {}
    for r in rd(O + "dw/sel_%s_g%d_s%d.csv" % (CFG, g, s)): sel[int(r["scen"])][r["model"]] = r["x"]; sc[int(r["scen"])] = (float(r["q0"]), float(r["c"]))
    J = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rd(O + "oos/true/ot_%s_g%d_s%d.csv" % (CFG, g, s)): J[int(r["nf"])][r["x"]].append(float(r["J0"]))
    return sel, sc, J
L = {(g, s): load(g, s) for g in range(1, 16) for s in range(1, SHI + 1)}; L = {k: v for k, v in L.items() if v is not None}
print("%s: %d runs" % (CFG, len(L)))
if not L: sys.exit()
scens = sorted(next(iter(L.values()))[1]); nfs = sorted(next(iter(L.values()))[2])
for k in scens:
    q0, c = next(iter(L.values()))[1][k]; cost = lambda x: q0 * len(par(x)[0]) + c * sum(par(x)[1]); cap = lambda x: sum(par(x)[1])
    print("\n========== cost scenario %d: q0 %.0f, c %.1f (c/v %.1f)" % (k, q0, c, c / 5))
    print("   capacity: " + ", ".join("%s %.0f" % (m, np.mean([cap(v[0][k][m]) for v in L.values()])) for m in ["N"] + MODS))
    print("   same decision as N: " + ", ".join("%s %.0f%%" % (m, 100 * np.mean([v[0][k][m] == v[0][k]["N"] for v in L.values()])) for m in MODS))
    for nf in nfs:
        print("-- " + ("retailer belief = leader's q-hat (belief known)" if nf == 0 else "retailer belief from %d own samples (belief unknown)" % nf))
        def val(v, m): return np.array(v[2][nf][v[0][k][m]]) + cost(v[0][k][m])
        scl = {key: abs(val(v, "N").mean()) / 100 for key, v in L.items()}
        if nf > 0: print("   N: q90 - mean over belief draws %.1f%%" % np.mean([(q90(val(v, "N")) - val(v, "N").mean()) / scl[key] for key, v in L.items()]))
        for a, b in [(m, "N") for m in MODS] + PAIRS:
            m_, sm, w1, l1 = cl({key: (val(v, a).mean() - val(v, b).mean()) / scl[key] for key, v in L.items()})
            line = "   %-5s - %-5s mean %+.2f +- %.2f [geo %d/%d]" % (a, b, m_, sm, w1, l1)
            if nf > 0:
                q_, sq, w2, l2 = cl({key: (q90(val(v, a)) - q90(val(v, b))) / scl[key] for key, v in L.items()}); line += " | q90 %+.2f +- %.2f [geo %d/%d]" % (q_, sq, w2, l2)
            print(line)
