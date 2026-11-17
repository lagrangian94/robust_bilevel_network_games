import csv, numpy as np, itertools
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/mech/"
cv = []; rng = []; cr = []; lo = []; hi = []; cvr = []; sizes = []; share = []; mechc = []
for g in range(1, 16):
    R = {}
    for r in csv.reader(open(O + "xi_g%d.csv" % g)):
        R.setdefault(int(r[0]), {})[r[1]] = np.array([float(x) for x in r[2:]])
    for s, d in R.items():
        t = d["tot"]; cv.append(t.std() / t.mean()); rng.append(t.min() / t.max())
        regs = [(k, v) for k, v in d.items() if k.startswith("reg") and int(k.split(":")[1]) > 0]
        for (a, x), (b, y) in itertools.combinations(regs, 2):
            if x.std() > 0 and y.std() > 0: cr.append(np.corrcoef(x, y)[0, 1])
        for k, v in regs: cvr.append(v.std() / v.mean())
        if g == 1 and s == 1: sizes = [k for k, _ in regs]
        o = np.argsort(t); lo.append(d["below"][o[:3]].mean()); hi.append(d["below"][o[-3:]].mean())
        # share of the variance of the total explained by the largest region
        v = np.array([x for _, x in regs]); share.append(max(np.var(x) for x in v) / np.var(t))
        # leader-revenue ranking vs total demand: Nominal decision's loss at belief q (scen 2)
        try:
            dec = {r["who"]: r for r in csv.DictReader(open(O + "dec_g%d_s%d.csv" % (g, s), encoding="utf-8")) if r["scen"] == "2"}
            for r in csv.DictReader(open(O + "mech_g%d_s%d.csv" % (g, s), encoding="utf-8")):
                if r["x"] == dec["N"]["x"] and r["bel"] == "q":
                    Z = np.array([float(r["Z%d" % k]) for k in range(1, 11)])
                    if Z.std() > 0: mechc.append(np.corrcoef(-Z, t)[0, 1])
        except Exception as e: print("skip", g, s, e)
n = len(cv)
print("runs %d | regions (g1): %s" % (n, sizes))
print("total demand over the 10 scenarios: CV %.3f, min/max %.2f" % (np.mean(cv), np.mean(rng)))
print("regional total: CV %.3f | correlation between two regions over the 10 scenarios: mean %.2f, mean |.| %.2f, share > 0.5: %.0f%%, share < -0.5: %.0f%%" % (np.mean(cvr), np.mean(cr), np.mean(np.abs(cr)), 100 * np.mean(np.array(cr) > 0.5), 100 * np.mean(np.array(cr) < -0.5)))
print("share of customers below their own median: in the 3 lowest-total scenarios %.2f, in the 3 highest-total scenarios %.2f" % (np.mean(lo), np.mean(hi)))
print("variance of the largest region / variance of the total: %.2f" % np.mean(share))
print("correlation between Nominal's revenue and total demand over scenarios: mean %.2f, share > 0.7: %.0f%% (%d runs)" % (np.mean(mechc), 100 * np.mean(np.array(mechc) > 0.7), len(mechc)))
