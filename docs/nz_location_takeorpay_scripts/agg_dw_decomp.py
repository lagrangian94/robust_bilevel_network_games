# -*- coding: utf-8 -*-
# Decomposition of the full world (leader estimates q-hat, retailer has its own sample): python agg_dw_decomp.py  (AD_CFG, default dw75)
# Per run and cost scenario: estimate V_M (the model's own in-sample value), J0 (true cost, retailer belief = leader's q-hat), JF (true cost, retailer's own sample, K draws).
import csv, collections, os, numpy as np
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/"
CFG = os.environ.get("AD_CFG", "dw75"); q90 = lambda v: np.quantile(v, 0.9); rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
def par(x): pr = [p.split(":") for p in x.strip("{}").split(";")]; return [int(a) for a, _ in pr], [float(b) for _, b in pr]
def cl(d):
    gg = collections.defaultdict(list)
    for (g, s), v in d.items(): gg[g].append(v)
    m = np.array([np.mean(v) for v in gg.values()]); return "%+6.2f +- %.2f [%2d/%2d]" % (np.mean(list(d.values())), m.std(ddof=1) / np.sqrt(len(m)), (m < -0.01).sum(), (m > 0.01).sum())
L = {}
for g in range(1, 16):
    for s in range(1, 6):
        lg = O + "dw/dw_%s_g%d_s%d.log" % (CFG, g, s)
        if not (os.path.exists(lg) and "OT_DONE" in open(lg, encoding="utf-8", errors="replace").read()): continue
        sel = collections.defaultdict(dict); sc = {}
        for r in rd(O + "dw/sel_%s_g%d_s%d.csv" % (CFG, g, s)): sel[int(r["scen"])][r["model"]] = r["x"]; sc[int(r["scen"])] = (float(r["q0"]), float(r["c"]))
        cand = {r["x"]: r for r in rd(O + "dw/dw_%s_g%d_s%d.csv" % (CFG, g, s))}
        J = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rd(O + "oos/true/ot_%s_g%d_s%d.csv" % (CFG, g, s)): J[int(r["nf"])][r["x"]].append(float(r["J0"]))
        L[(g, s)] = (sel, sc, cand, J)
print("%s: %d runs" % (CFG, len(L)))
def own(cand, m, x):   # the model's own in-sample value of its decision (without store cost)
    r = cand[x]; v0 = float(r["V0"])
    if m == "N": return v0
    if m[0] == "D": return float(r["VD" + m[1:]])
    if m[0] == "B": return v0 + float(r["adv" + m[1:]])
    return float(r["VCh"])
MODS = ["N", "D0.1", "D0.3", "B0.1", "B0.2", "C"]
for k in sorted(next(iter(L.values()))[1]):
    q0, c = next(iter(L.values()))[1][k]; cost = lambda x: q0 * len(par(x)[0]) + c * sum(par(x)[1])
    print("\n========== c/v %.1f" % (c / 5))
    R = {}
    for key, (sel, sc, cand, J) in L.items():
        d = {}
        for m in MODS:
            x = sel[k][m]; cs = cost(x)
            d[m] = dict(x=x, cap=sum(par(x)[1]), V=own(cand, m, x) + cs, V0=float(cand[x]["V0"]) + cs, J0=J[0][x][0] + cs, F5=np.array(J[5][x]) + cs, F20=np.array(J[20][x]) + cs)
        d["sc"] = abs(d["N"]["J0"]) / 100; R[key] = d
    print("A. Nominal: estimate -> truth (%% of true cost, + = estimate worse than truth)")
    print("   estimate - true cost, retailer shares q-hat          " + cl({k_: (d["N"]["V"] - d["N"]["J0"]) / d["sc"] for k_, d in R.items()}))
    for nf in ("F5", "F20"):
        print("   %-3s belief path: mean true cost - shared-belief cost  " % nf + cl({k_: (d["N"][nf].mean() - d["N"]["J0"]) / d["sc"] for k_, d in R.items()}))
        print("   %-3s belief path: q90 true cost - shared-belief cost   " % nf + cl({k_: (q90(d["N"][nf]) - d["N"]["J0"]) / d["sc"] for k_, d in R.items()}))
        print("   %-3s net: estimate - mean true cost                    " % nf + cl({k_: (d["N"]["V"] - d["N"][nf].mean()) / d["sc"] for k_, d in R.items()}))
    print("B. model - Nominal")
    for m in MODS[1:] + ["C-B0.1"]:
        a, b = (("C", "B0.1") if m == "C-B0.1" else (m, "N"))
        print("  %-7s differs %3.0f%%  capacity %+5.0f" % (m, 100 * np.mean([d[a]["x"] != d[b]["x"] for d in R.values()]), np.mean([d[a]["cap"] - d[b]["cap"] for d in R.values()])))
        print("      nominal in-sample premium (V0)         " + cl({k_: (d[a]["V0"] - d[b]["V0"]) / d["sc"] for k_, d in R.items()}))
        print("      true cost, retailer shares q-hat       " + cl({k_: (d[a]["J0"] - d[b]["J0"]) / d["sc"] for k_, d in R.items()}))
        for nf in ("F5", "F20"):
            print("      %-3s mean                               " % nf + cl({k_: (d[a][nf].mean() - d[b][nf].mean()) / d["sc"] for k_, d in R.items()}))
            print("      %-3s q90                                " % nf + cl({k_: (q90(d[a][nf]) - q90(d[b][nf])) / d["sc"] for k_, d in R.items()}))
            print("      %-3s belief-noise part of mean (mean - shared) " % nf + cl({k_: ((d[a][nf].mean() - d[a]["J0"]) - (d[b][nf].mean() - d[b]["J0"])) / d["sc"] for k_, d in R.items()}))
            print("      %-3s belief-noise part of q90                  " % nf + cl({k_: ((q90(d[a][nf]) - d[a]["J0"]) - (q90(d[b][nf]) - d[b]["J0"])) / d["sc"] for k_, d in R.items()}))
    print("C. each model's own value as a certificate (F5 / F20): value - mean true cost, share of belief draws with true cost <= value")
    for m in MODS:
        print("  %-5s" % m + "  F5: " + cl({k_: (d[m]["V"] - d[m]["F5"].mean()) / d["sc"] for k_, d in R.items()}) + "  covered %3.0f%%" % (100 * np.mean([(d[m]["F5"] <= d[m]["V"]).mean() for d in R.values()]))
              + "   F20: " + cl({k_: (d[m]["V"] - d[m]["F20"].mean()) / d["sc"] for k_, d in R.items()}) + "  covered %3.0f%%" % (100 * np.mean([(d[m]["F20"] <= d[m]["V"]).mean() for d in R.values()])))
