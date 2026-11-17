# -*- coding: utf-8 -*-
# What does Belief-only (0.1) change relative to Nominal?  price-based decisions (geo/onp dec files, d = 0 rows): stores, capacity, predicted commitment, slack, in-sample premium.
import csv, numpy as np, collections
O = "C:/Users/user/AppData/Local/Temp/claude/C--Users-user-robust-bilevel-network-games/52fc5b93-ef5c-4aee-815c-8f3e7523e9c7/scratchpad/geo/onp/"
def par(x): pr = [p.split(":") for p in x.strip("{}").split(";") if p]; return [int(a) for a, _ in pr], [float(b) for _, b in pr]
for scen, lab in (("2", "0.4"), ("1", "0.5")):
    rows = []
    for g in range(1, 16):
        for s in range(1, 11):
            t = {}
            for r in csv.DictReader(open(O + "dec_g%d_s%d.csv" % (g, s), encoding="utf-8")):
                if r["scen"] == scen and float(r["d"]) == 0 and r["who"] in ("N", "B0.1", "D0.1"): t[r["who"]] = r
            q0, c = float(t["N"]["q0"]), float(t["N"]["c"]); cost = lambda x: q0 * len(par(x)[0]) + c * sum(par(x)[1])
            sc = abs(float(t["N"]["Vss"]) + cost(t["N"]["x"])) / 100
            rec = {}
            for w in ("N", "B0.1", "D0.1"):
                x = t[w]["x"]; si, lv = par(x)
                rec[w] = dict(n=len(si), cap=sum(lv), h=float(t[w]["hB_pred"]), big=400.0 in lv, small=200.0 in lv, J=(float(t[w]["Vss"]) + cost(x)) / sc, sites=tuple(si), x=x, store=cost(x) / sc, rev=-float(t[w]["Vss"]) / sc)
            rows.append(rec)
    print("== c/f %s, %d runs" % (lab, len(rows)))
    for w in ("N", "B0.1", "D0.1"):
        f = lambda k: np.mean([r[w][k] for r in rows])
        print("   %-5s stores %.2f | capacity %.0f | predicted commitment %.0f | slack (capacity - commitment) %.0f | uses 400: %2.0f%% | uses 200: %2.0f%% | store cost %.1f%% of scale | in-sample revenue (-CVaR) %.1f%%" % (w, f("n"), f("cap"), f("h"), f("cap") - f("h"), 100 * f("big"), 100 * f("small"), f("store"), f("rev")))
    for w in ("B0.1", "D0.1"):
        d = [r for r in rows if r[w]["x"] != r["N"]["x"]]
        print("   %s differs from N in %d runs (%.0f%%): same sites %2.0f%% | fewer stores %2.0f%%, more stores %2.0f%% | less capacity %2.0f%%, more %2.0f%%, equal %2.0f%% | commitment lower %2.0f%%, higher %2.0f%% | slack smaller %2.0f%%, larger %2.0f%%" % (
            w, len(d), 100 * len(d) / len(rows), 100 * np.mean([r[w]["sites"] == r["N"]["sites"] for r in d]), 100 * np.mean([r[w]["n"] < r["N"]["n"] for r in d]), 100 * np.mean([r[w]["n"] > r["N"]["n"] for r in d]),
            100 * np.mean([r[w]["cap"] < r["N"]["cap"] for r in d]), 100 * np.mean([r[w]["cap"] > r["N"]["cap"] for r in d]), 100 * np.mean([r[w]["cap"] == r["N"]["cap"] for r in d]),
            100 * np.mean([r[w]["h"] < r["N"]["h"] - 1 for r in d]), 100 * np.mean([r[w]["h"] > r["N"]["h"] + 1 for r in d]),
            100 * np.mean([(r[w]["cap"] - r[w]["h"]) < (r["N"]["cap"] - r["N"]["h"]) - 1 for r in d]), 100 * np.mean([(r[w]["cap"] - r[w]["h"]) > (r["N"]["cap"] - r["N"]["h"]) + 1 for r in d])))
        prem = np.array([r[w]["J"] - r["N"]["J"] for r in rows]); premd = np.array([r[w]["J"] - r["N"]["J"] for r in d])
        print("      in-sample premium (cost under q̂, belief q̂): all runs %+.2f, runs that differ %+.2f | of which store cost %+.2f, revenue lost %+.2f" % (prem.mean(), premd.mean(), np.mean([r[w]["store"] - r["N"]["store"] for r in d]), np.mean([r["N"]["rev"] - r[w]["rev"] for r in d])))
